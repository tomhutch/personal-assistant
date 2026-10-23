import logging
import signal
import sys
import threading
from config import LOG_LEVEL, SAMPLE_RATE
from audio import RollingAudioBuffer, listen_for_wake_word, find_input_device
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

    input_device = find_input_device()
    if input_device is None:
        logger.error("❌ No usable audio input device found. Connect a microphone and restart.")
        for idx, dev in enumerate(sd.query_devices()):
            logger.error("  device %d: %s (inputs=%d, outputs=%d)", idx, dev["name"], dev["max_input_channels"], dev["max_output_channels"])
        sys.exit(1)
    logger.info("🎧 Using input device %d: %s", input_device, sd.query_devices(input_device)["name"])

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.debug("Audio stream status: %s", status)
        if not shutdown_event.is_set():
            rolling_buffer.append(indata[:, 0].copy())

    with sd.InputStream(device=input_device, samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        while not shutdown_event.is_set():
            wake_triggered, noise_floor = listen_for_wake_word(rolling_buffer)
            
            if wake_triggered and not shutdown_event.is_set():
                handle_voice_command(noise_floor)
                logger.info("💤 Returning to standby...\n")

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("🛑 Assistant stopped cleanly.")