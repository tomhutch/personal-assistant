import logging
import signal
import sys
import threading
import sounddevice as sd
from config import LOG_LEVEL, SAMPLE_RATE
from audio import RollingAudioBuffer, WakeWordManager, find_input_device
from command_handler import handle_voice_command
from led import LED, AssistantState

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

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main_loop():
    """Main background ingestion and wake-word listening loop."""
    logger.info("🚀 Starting background audio stream & ring buffer...")
    rolling_buffer = RollingAudioBuffer(duration_seconds=3.0)

    wake_word_manager = WakeWordManager()

    input_device = find_input_device()
    if input_device is None:
        logger.error("❌ No usable audio input device found. Connect a microphone and restart.")
        sys.exit(1)
    logger.info("🎤 Using input device %d: %s", input_device, sd.query_devices(input_device)["name"])

    led = LED()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.debug("Audio stream status: %s", status)
        if not shutdown_event.is_set():
            chunk = indata[:, 0].copy()
            rolling_buffer.append(chunk)
            wake_word_manager.push_audio_chunk(chunk)

    try:
        with sd.InputStream(device=input_device, samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=1280, callback=audio_callback):
            while not shutdown_event.is_set():
                wake_triggered, noise_floor = wake_word_manager.listen_for_wake_word(rolling_buffer, shutdown_event)

                if wake_triggered and not shutdown_event.is_set():
                    handle_voice_command(
                        input_device,
                        noise_floor,
                        shutdown_event=shutdown_event,
                        led=led,
                    )
                    logger.debug("💤 Returning to standby...\n")

        led.set_state(AssistantState.PASSIVE)
    finally:
        led.close()

    logger.info("🛑 Assistant stopped cleanly.")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Assistant stopped cleanly.")