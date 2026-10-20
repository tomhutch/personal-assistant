import logging
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

def handle_voice_command(noise_floor):
    """
    Orchestrates a single voice command interaction:
    1. Fetches HA devices in parallel with recording.
    2. Records the utterance.
    3. Transcribes and parses intent.
    4. Executes the Home Assistant service call.
    """
    logger.info("🎤 Command mode active. Initializing parallel tasks...")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_devices = executor.submit(fetch_ha_devices)
        audio_data = record_utterance(noise_floor)
        controllable_devices = future_devices.result()

    if len(audio_data) == 0:
        logger.warning("⚠️ No audio recorded. Returning to standby.")
        return

    logger.info("🔄 Transcribing command via Whisper...")
    transcript = transcribe_audio(audio_data)
    logger.info("➜ Transcript: '%s'", transcript)

    if not transcript or "[blank_audio]" in transcript.lower():
        logger.info("⚠️ Empty or silent transcript. Returning to standby.")
        return

    logger.info("🧠 Parsing intent with Qwen SLM...")
    intent = parse_intent(transcript, controllable_devices)
    
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
        rolling_buffer.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        while True:
            wake_triggered, noise_floor = listen_for_wake_word(rolling_buffer)
            
            if wake_triggered:
                handle_voice_command(noise_floor)
                logger.info("💤 Returning to standby...\n")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Assistant stopped by user.")