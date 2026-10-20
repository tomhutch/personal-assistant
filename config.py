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

# Command recording configuration
COMMAND_MAX_DURATION = 10.0         # Hard safety limit for command recording in seconds
COMMAND_SPEECH_MULTIPLIER = 1.8     # Multiplier above noise floor to detect speech
COMMAND_POST_SPEECH_WINDOW = 2      # Fixed recording duration after speech starts (music-proof)

# Read log level from environment variable, default to INFO if not set
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)