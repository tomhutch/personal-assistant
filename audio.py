import logging
import sounddevice as sd
import numpy as np
import threading
import time
from collections import deque
from config import SAMPLE_RATE, WAKE_WORD, WAKE_WORD_MULTIPLIER
from transcriber import transcribe_audio

logger = logging.getLogger(__name__)

class RollingAudioBuffer:
    """A thread-safe ring buffer that dynamically tracks exact sample counts for N seconds of audio."""
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

def listen_for_wake_word(rolling_buffer):
    """Continuously inspects the background ring buffer using an adaptive relative noise gate."""
    logger.info("💤 Listening for wake word ('Hey Susan')... [Adaptive Noise Gate Active]")
    
    # Initialize background noise floor baseline
    noise_floor = 0.001
    last_debug_time = time.time()
    
    while True:
        now = time.time()
        if now - last_debug_time > 3.0:
            audio_np = rolling_buffer.get_audio()
            logger.debug(
                "Buffer status: %d blocks, length: %.2fs | Estimated Noise Floor: %.5f", 
                rolling_buffer.total_blocks_received, 
                len(audio_np) / SAMPLE_RATE,
                noise_floor
            )
            last_debug_time = now

        audio_np = rolling_buffer.get_audio()
        
        # Wait until we have at least 2 seconds of pre-rolled audio in memory
        if len(audio_np) < SAMPLE_RATE * 2.0:
            time.sleep(0.2)
            continue

        # Check RMS of the latest half-second
        recent_chunk = audio_np[-int(SAMPLE_RATE * 0.5):]
        rms = get_audio_rms(recent_chunk)
        
        # Dynamically update the background noise floor if the sound is low (ambient room noise)
        # We cap it so active speech doesn't permanently inflate the baseline noise floor.
        if rms < 0.015:
            # Exponential Moving Average (EMA) to smoothly track ambient noise changes
            noise_floor = (0.95 * noise_floor) + (0.05 * rms)
            # Ensure noise floor doesn't drop to absolute zero
            noise_floor = max(noise_floor, 0.0005)

        # Relative check: Is the current sound significantly louder than the room's background noise?
        # Multiplier of 3.0 means the sound must be 3x louder than the current room baseline.
        relative_threshold = noise_floor * WAKE_WORD_MULTIPLIER
        
        if rms < relative_threshold:
            time.sleep(0.2)
            continue

        logger.debug("🔊 Relative speech detected! RMS: %.4f (Noise Floor: %.5f, Threshold: %.5f). Submitting to Whisper...", rms, noise_floor, relative_threshold)
        transcript = transcribe_audio(audio_np).lower()
        
        logger.debug("Whisper Output: '%s'", transcript)
        
        if WAKE_WORD in transcript:
            logger.info("✨ Wake word detected! Transcript: '%s'", transcript)
            rolling_buffer.clear()
            return True
        
        time.sleep(0.5)

def record_utterance():
    """Synchronously captures the command utterance after the wake word triggers."""
    logger.info("🎤 Command mode active. Speak your command...")
    audio = sd.rec(int(6.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]