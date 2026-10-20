import logging
import signal
import sys
import threading
from config import LOG_LEVEL, SAMPLE_RATE
from audio import RollingAudioBuffer, listen_for_wake_word
from command_handler import handle_voice_command

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