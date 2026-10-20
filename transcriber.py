import logging
import requests
import numpy as np
import io
import wave
from config import SAMPLE_RATE

logger = logging.getLogger(__name__)

WHISPER_URL = "http://localhost:8081/inference"

def transcribe_audio(audio_np: np.ndarray) -> str:
    """Sends audio numpy array as an in-memory WAV file to the local Whisper endpoint."""
    logger.info("🔄 Transcribing audio via Whisper...")
    
    if len(audio_np) == 0:
        return ""

    try:
        # Convert float32 numpy array [-1.0, 1.0] to 16-bit PCM bytes
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Write to an in-memory WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        # Send as multipart/form-data file upload (required by whisper.cpp)
        files = {
            'file': ('audio.wav', wav_buffer.read(), 'audio/wav')
        }
        
        response = requests.post(
            WHISPER_URL,
            files=files,
            timeout=5.0
        )
        
        if response.status_code != 200:
            logger.error("❌ Whisper API returned error status %d: %s", response.status_code, response.text)
            return ""
            
        data = response.json()
        text = data.get("text", "") or data.get("transcription", "")
        if not isinstance(text, str):
            return ""
            
        return text.strip()
        
    except requests.RequestException as e:
        logger.error("❌ Failed to connect to Whisper service: %s", e)
        return ""
    except Exception as e:
        logger.error("❌ Error preparing audio for Whisper: %s", e)
        return ""