import requests
import tempfile
import os
from scipy.io.wavfile import write

from config import SAMPLE_RATE, WHISPER_URL

def transcribe_audio(audio_np):
    """Sends audio buffer to Whisper server and returns transcript string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        temp_wav_path = tmp_audio.name

    try:
        write(temp_wav_path, SAMPLE_RATE, audio_np)
        print("🔄 Transcribing command via Whisper...")
        with open(temp_wav_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": f},
                data={"response_format": "json"}
            )
        return resp.json().get("text", "").strip()
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
