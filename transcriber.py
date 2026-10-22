import logging
import requests
import numpy as np
import io
import wave
from config import SAMPLE_RATE, WHISPER_URL

logger = logging.getLogger(__name__)

def transcribe_audio(audio_np: np.ndarray, prompt: str = "") -> str:
    """Sends audio numpy array as an in-memory WAV file to Whisper with an optional vocabulary prompt."""
    logger.info("🔄 Transcribing audio via Whisper...")
    
    if len(audio_np) == 0:
        return ""

    try:
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        files = {
            'file': ('audio.wav', wav_buffer.read(), 'audio/wav')
        }
        
        # Whisper.cpp and API endpoints accept an optional 'prompt' form field
        data = {}
        if prompt:
            data['prompt'] = prompt

        response = requests.post(
            WHISPER_URL,
            files=files,
            data=data,
            timeout=5.0
        )
        
        if response.status_code != 200:
            logger.error("❌ Whisper API returned error status %d: %s", response.status_code, response.text)
            return ""
            
        res_json = response.json()
        text = res_json.get("text", "") or res_json.get("transcription", "")
        if not isinstance(text, str):
            return ""
            
        return text.strip()
        
    except requests.RequestException as e:
        logger.error("❌ Failed to connect to Whisper service: %s", e)
        return ""
    except Exception as e:
        logger.error("❌ Error preparing audio for Whisper: %s", e)
        return ""