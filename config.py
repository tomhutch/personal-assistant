import os
import logging

# API Endpoints (overridable via environment variables, e.g. from .env under systemd)
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8081/inference")
SLM_URL = os.getenv("SLM_URL", "http://localhost:8080/v1/chat/completions")  # llama endpoint
HA_URL = os.getenv("HA_URL", "http://localhost:8123")
HA_TOKEN = os.getenv("HA_TOKEN")

# Audio Config
SAMPLE_RATE = 16000  # openWakeWord natively expects 16kHz audio

# Wake Word Config
WAKE_WORD_MODEL = "hey_jarvis"  # Options: "alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy" (https://github.com/dscripka/openWakeWord/tree/main/docs/models)
WAKE_WORD_THRESHOLD = 0.5   # Confidence threshold between 0.0 and 1.0

# Command recording configuration
COMMAND_MAX_DURATION = 10.0         # Hard safety limit for command recording in seconds
COMMAND_SPEECH_MULTIPLIER = 1.8     # Multiplier above noise floor to detect speech
COMMAND_POST_SPEECH_WINDOW = 2      # Fixed recording duration after speech starts (music-proof)

# Intent Parsing
SLM_TIMEOUT_SECONDS=10.0

# Read log level from environment variable, default to INFO if not set
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# LED Config
LED_ENABLED = True
LED_BRIGHTNESS = 0.1

# Text-to-speech voice model downloaded by piper.download_voices
PIPER_VOICE_MODEL = os.getenv(
	"PIPER_VOICE_MODEL", "en_US-norman-medium.onnx"
)
PIPER_OUTPUT_DEVICE = os.getenv("PIPER_OUTPUT_DEVICE", "")
