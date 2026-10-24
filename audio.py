import logging
import sounddevice as sd
import numpy as np
import threading
import queue
from collections import deque
from openwakeword.model import Model
from config import (
    SAMPLE_RATE,
    COMMAND_MAX_DURATION, 
    COMMAND_SPEECH_MULTIPLIER,
    COMMAND_POST_SPEECH_WINDOW,
    WAKE_WORD_MODEL,
    WAKE_WORD_THRESHOLD
)

logger = logging.getLogger(__name__)

# Global model instance cache
_oww_model = None

# Thread-safe queue to hold exact 1280-sample audio chunks
audio_queue = queue.Queue()

class RollingAudioBuffer:
    def __init__(self, duration_seconds=3.0):
        self.max_samples = int(SAMPLE_RATE * duration_seconds)
        self.buffer = deque()
        self.current_samples = 0
        self.lock = threading.Lock()
        self.total_blocks_received = 0

    def append(self, block):
        with self.lock:
            self.buffer.append(block)
            self.current_samples += len(block)
            self.total_blocks_received += 1
            
            while self.current_samples > self.max_samples and len(self.buffer) > 1:
                removed = self.buffer.popleft()
                self.current_samples -= len(removed)

    def get_audio(self):
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self.buffer), axis=0)

    def clear(self):
        with self.lock:
            self.buffer.clear()
            self.current_samples = 0

def get_audio_rms(block):
    return np.sqrt(np.mean(block**2))

def find_input_device():
    """
    Finds a usable audio input device by index, avoiding raw ALSA hardware 
    paths (hw:) which can cause sample rate or resource lock errors.
    """
    try:
        devices = sd.query_devices()
        logger.debug(f"available sounddevice devices: {devices}")
    except Exception:
        logger.error("Failed to query PortAudio devices", exc_info=True)
        return None

    # Priority 1: USB input devices managed by the sound server (excluding raw 'hw:')
    for dev in devices:
        if (dev["max_input_channels"] > 0 
            and "usb" in dev["name"].lower() 
            and "hw:" not in dev["name"]):
            logger.debug(f"selected usb sounddevice: {dev["index"]} {dev}")
            return dev["index"]

    # Priority 2: Any input device that is not a loopback and not a raw 'hw:' device
    for dev in devices:
        if (dev["max_input_channels"] > 0 
            and "loopback" not in dev["name"].lower() 
            and "hw:" not in dev["name"]):
            logger.debug(f"selected non usb sounddevice: {dev["index"]} {dev}")
            return dev["index"]

    return None

def push_audio_chunk(chunk):
    """ Receives exact 80ms blocks from the main stream callback. """
    audio_queue.put(chunk)

def get_wakeword_model():
    global _oww_model
    if _oww_model is None:
        logger.info("📦 [openWakeWord] Initializing and loading model '%s' (this may download weights on first run)...", WAKE_WORD_MODEL)
        try:
            _oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
            logger.info("✅ [openWakeWord] Model loaded successfully!")
        except Exception as e:
            logger.error("❌ [openWakeWord] Failed to load model: %s", e, exc_info=True)
            raise e
    return _oww_model

def listen_for_wake_word(rolling_buffer, shutdown_event):
    """
    Continuously inspects background audio stream using openWakeWord 
    with support for graceful shutdown.
    """
    model = get_wakeword_model()
    logger.info("💤 Listening for wake word ('%s') [openWakeWord Active]...", WAKE_WORD_MODEL)
    
    while not audio_queue.empty():
        audio_queue.get_nowait()
    
    while not shutdown_event.is_set():
        try:
            # Use short timeout so it periodically checks shutdown_event
            audio_chunk = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
            
        if shutdown_event.is_set():
            break
            
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        model.predict(audio_int16)
        score = model.prediction_buffer[WAKE_WORD_MODEL][-1]
        
        logger.debug("Wake word score ('%s'): %.4f", WAKE_WORD_MODEL, score)
        
        if score >= WAKE_WORD_THRESHOLD:
            logger.info("✨ Wake word '%s' detected! Confidence: %.2f", WAKE_WORD_MODEL, score)
            rolling_buffer.clear()
            model.reset()
            
            while not audio_queue.empty():
                audio_queue.get_nowait()
                
            return True, 0.001
            
    return False, 0.001

def record_utterance(noise_floor):
    """
    Captures command utterance using a lightning-fast, music-proof 
    fixed post-speech window driven by configuration.
    """
    logger.info("🎤 Command mode active. Speak your command...")
    
    speech_threshold = max(noise_floor * COMMAND_SPEECH_MULTIPLIER, 0.003)
    
    chunks = []
    chunk_duration = 0.1  # 100ms blocks
    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    
    speech_detected = False
    post_speech_duration = 0.0
    total_duration = 0.0

    device = find_input_device()
    if device is None:
        logger.error("❌ No usable audio input device found; cannot record utterance.")
        return np.array([], dtype=np.float32)

    with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while total_duration < COMMAND_MAX_DURATION:
            audio_block, _ = stream.read(chunk_samples)
            block = audio_block[:, 0]
            chunks.append(block)
            total_duration += chunk_duration
            
            rms = np.sqrt(np.mean(block**2))
            
            if rms > speech_threshold:
                if not speech_detected:
                    logger.debug("🗣️ Speech confirmed active (RMS: %.5f > Threshold: %.5f)", rms, speech_threshold)
                speech_detected = True
            
            # Uses the configurable post-speech window from config.py
            if speech_detected:
                post_speech_duration += chunk_duration
                if post_speech_duration >= COMMAND_POST_SPEECH_WINDOW:
                    logger.debug("🛑 Capture window complete (%.2fs post-speech).", post_speech_duration)
                    break
                
            if not speech_detected and total_duration >= 3.0:
                logger.debug("⚠️ No speech detected during command window, timing out.")
                break

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)