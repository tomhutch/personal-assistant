import os
import logging

# API Endpoints
WHISPER_URL = "http://localhost:8081/inference"
SLM_URL = "http://localhost:8080/v1/chat/completions"  # Qwen 2.5 3B endpoint
HA_URL = "http://localhost:8123"
HA_TOKEN = os.getenv("HA_TOKEN")

# Audio Configuration
SAMPLE_RATE = 16000          # 16kHz is ideal for Whisper
BLOCK_DURATION = 0.1         # 100ms per audio block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WAKE_WORD = "susan"
WAKE_WORD_MULTIPLIER = 1.8   # Wake word sensitivity multiplier (Lower = more sensitive, Higher = less sensitive)

# Adaptive VAD Configuration (Relative Threshold)
VAD_MULTIPLIER = 3.0         # Speech energy must be 3x the current room noise floor
VAD_SMOOTHING = 0.95         # How fast the baseline adapts to background noise changes (0.9 - 0.99)
MIN_NOISE_FLOOR = 0.0001     # Absolute floor safety limit to prevent division by zero

# Timing & Windows (in seconds)
WAKE_WORD_WINDOW = 2.5       # Size of rolling window sent to Whisper for wake word
START_TIMEOUT = 5.0          # Max time to wait for you to start talking after wake word
SILENCE_TIMEOUT = 1.2        # Pause duration to consider speech finished during command
MAX_COMMAND_TIME = 10.0      # Absolute max recording length to prevent hanging on noise
QUEUE_TIMEOUT = 0.1          # Timeout for checking the audio queue

# Read log level from environment variable, default to INFO if not set
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)