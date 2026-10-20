import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from config import LOG_LEVEL, SAMPLE_RATE
from audio import RollingAudioBuffer, listen_for_wake_word, record_utterance
from transcriber import transcribe_audio
from ha_client import fetch_ha_devices, execute_ha_command
from intent_parser import parse_intent

# Robust logging configuration supporting both strings and integers for LOG_LEVEL
if isinstance(LOG_LEVEL, int):
    resolved_log_level = LOG_LEVEL
else:
    resolved_log_level = getattr(logging, str(LOG_LEVEL).upper(), logging.INFO)

logging.basicConfig(
    level=resolved_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global shutdown flag for graceful termination
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handles SIGINT (Ctrl+C) and SIGTERM gracefully."""
    logger.info("🛑 Termination signal received. Shutting down gracefully...")
    shutdown_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def handle_voice_command(noise_floor):
    """
    Orchestrates a single voice command interaction:
    1. Fetches HA devices in parallel with recording utterance.
    2. Builds a lean vocabulary prompt from active device names.
    3. Transcribes audio via Whisper using the priming prompt.
    4. Parses intent and executes the Home Assistant service call.
    """
    if shutdown_event.is_set():
        return

    logger.info("🎤 Command mode active. Initializing parallel tasks...")
    
    # Run Home Assistant device fetching in parallel with recording utterance
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_devices = executor.submit(fetch_ha_devices)
        audio_data = record_utterance(noise_floor)
        controllable_devices = future_devices.result()

    if shutdown_event.is_set() or len(audio_data) == 0:
        logger.warning("⚠️ No audio recorded or shutdown requested. Returning to standby.")
        return

    # Build a lean context prompt: strict action verbs and your actual device names
    device_names = [d.get("name", "") for d in controllable_devices]
    prompt_context = (
        f"Turn on. Turn off. "
        f"Devices: {', '.join(device_names)}."
    )

    logger.info("🔄 Transcribing command via Whisper (with lean context prompt)...")
    transcript = transcribe_audio(audio_data, prompt=prompt_context)
    logger.info("➜ Transcript: '%s'", transcript)

    if shutdown_event.is_set() or not transcript or "[blank_audio]" in transcript.lower():
        logger.info("⚠️ Empty transcript or shutdown requested. Returning to standby.")
        return

    logger.info("🧠 Parsing intent with Qwen SLM...")
    intent = parse_intent(transcript, controllable_devices)
    
    if shutdown_event.is_set():
        return

    if intent and "domain" in intent and "service" in intent and "entity_id" in intent:
        logger.info(
            "⚡ Executing on Home Assistant: %s -> %s", 
            intent['service'], 
            intent['entity_id']
        )
        execute_ha_command(intent['domain'], intent['service'], intent['entity_id'])
    else:
        logger.warning("⚠️ Could not resolve intent or missing entity mapping.")

def main_loop():
    """Main background ingestion and wake-word listening loop."""
    logger.info("🚀 Starting background audio stream & ring buffer...")
    rolling_buffer = RollingAudioBuffer(duration_seconds=3.0)
    
    import sounddevice as sd
    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.debug("Audio stream status: %s", status)
        if not shutdown_event.is_set():
            rolling_buffer.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        while not shutdown_event.is_set():
            wake_triggered, noise_floor = listen_for_wake_word(rolling_buffer)
            
            if wake_triggered and not shutdown_event.is_set():
                handle_voice_command(noise_floor)
                logger.info("💤 Returning to standby...\n")

if __name__ == "__main__":
    try:
        main_loop()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Assistant stopped cleanly.")