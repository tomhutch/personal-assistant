import logging
import requests
import numpy as np

logger = logging.getLogger(__name__)

WHISPER_URL = "http://localhost:8081/inference"

def transcribe_audio(audio_np: np.ndarray) -> str:
    """Sends audio numpy array to local Whisper endpoint and returns transcript text."""
    logger.info("🔄 Transcribing audio via Whisper...")
    
    # Ensure audio is float32 and convert to bytes if needed by the endpoint
    audio_bytes = audio_np.astype(np.float32).tobytes()
    
    try:
        response = requests.post(
            WHISPER_URL,
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=5.0
        )
        
        # Guard against server error status codes (e.g., 500, 404)
        if response.status_code != 200:
            logger.error("❌ Whisper API returned error status %d", response.status_code)
            return ""
            
        data = response.json()
        # Handle different whisper response formats safely
        text = data.get("text", "") or data.get("transcription", "")
        if not isinstance(text, str):
            return ""
            
        return text.strip()
        
    except requests.RequestException as e:
        logger.error("❌ Failed to connect to Whisper service: %s", e)
        return ""