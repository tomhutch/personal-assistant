import logging
import requests
import tempfile
import os
from scipy.io.wavfile import write

from config import SAMPLE_RATE, WHISPER_URL

logger = logging.getLogger(__name__)

def transcribe_audio(audio_np):
    """Sends audio buffer to Whisper server and returns transcript string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        temp_wav_path = tmp_audio.name

    try:
        write(temp_wav_path, SAMPLE_RATE, audio_np)
        logger.info("🔄 Transcribing audio via Whisper...")
        
        with open(temp_wav_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": f},
                data={"response_format": "json"}
            )
            
        resp.raise_for_status()
        transcript = resp.json().get("text", "").strip()
        logger.debug("Whisper response text: '%s'", transcript)
        return transcript
        
    except Exception as e:
        logger.error("❌ Transcription failed: %s", e)
        return ""
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)