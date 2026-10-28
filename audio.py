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

class WakeWordManager:
    """Encapsulates the audio queue and wake-word model state."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self._oww_model = None

    def push_audio_chunk(self, chunk):
        self.audio_queue.put(chunk)

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def get_wakeword_model(self):
        if self._oww_model is None:
            logger.info("📦 [openWakeWord] Initializing and loading model '%s'...", WAKE_WORD_MODEL)
            try:
                self._oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                logger.info("✅ [openWakeWord] Model loaded successfully!")
            except Exception as e:
                logger.error("❌ [openWakeWord] Failed to load model: %s", e, exc_info=True)
                raise e
        return self._oww_model

    def listen_for_wake_word(self, rolling_buffer, shutdown_event=None):
        if shutdown_event is None:
            class DummyEvent:
                def is_set(self): return False
            shutdown_event = DummyEvent()

        model = self.get_wakeword_model()
        logger.debug("💤 Listening for wake word ('%s') [openWakeWord Active]...", WAKE_WORD_MODEL)
        
        while not shutdown_event.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                return False, 0.001
                
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
                self.clear_queue()
                return True, 0.001
                
        return False, 0.001

def find_input_device():
    try:
        devices = sd.query_devices()
        logger.debug(f"available sounddevice devices: {devices}")
    except Exception:
        logger.error("Failed to query PortAudio devices", exc_info=True)
        return None

    input_devices = [
        (global_idx, dev) for global_idx, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
    ]

    for idx, dev in input_devices:
        name = dev["name"].lower()
        if "usb" in name and "hw:" not in name and "loopback" not in name and ".monitor" not in name:
            return idx

    for idx, dev in input_devices:
        name = dev["name"].lower()
        if "hw:" not in name and "loopback" not in name and ".monitor" not in name:
            return idx

    return None

def record_utterance(input_device, noise_floor, cancel_event=None):
    logger.debug("🎤 Command mode active. Speak your command...")
    speech_threshold = max(noise_floor * COMMAND_SPEECH_MULTIPLIER, 0.003)
    chunks = []
    chunk_duration = 0.1
    chunk_samples = int(SAMPLE_RATE * chunk_duration)
    total_duration = 0.0
    speech_detected = False
    post_speech_duration = 0.0

    if input_device is None:
        logger.error("❌ No usable audio input device found; cannot record utterance.")
        return np.array([], dtype=np.float32)
    if cancel_event and cancel_event.is_set():
        return np.array([], dtype=np.float32)

    with sd.InputStream(device=input_device, samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while (
            total_duration < COMMAND_MAX_DURATION
            and not (cancel_event and cancel_event.is_set())
        ):
            audio_block, _ = stream.read(chunk_samples)
            block = audio_block[:, 0]
            chunks.append(block)
            total_duration += chunk_duration
            
            rms = np.sqrt(np.mean(block**2))
            if rms > speech_threshold:
                speech_detected = True
            
            if speech_detected:
                post_speech_duration += chunk_duration
                if post_speech_duration >= COMMAND_POST_SPEECH_WINDOW:
                    break
                
            if not speech_detected and total_duration >= 3.0:
                break

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)

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