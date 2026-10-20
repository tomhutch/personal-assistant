import logging
import sounddevice as sd
import numpy as np
import threading
import time
from collections import deque
from config import SAMPLE_RATE, WAKE_WORD
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
            
            # Pop old blocks from the left until we fit within max_samples
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
    """Continuously inspects the background ring buffer for 'Hey Susan'."""
    logger.info("💤 Listening for wake word ('Hey Susan')...")
    
    last_debug_time = time.time()
    
    while True:
        # Diagnostic check every 3 seconds (only evaluates string formatting if DEBUG is enabled)
        now = time.time()
        if now - last_debug_time > 3.0:
            audio_np = rolling_buffer.get_audio()
            logger.debug(
                "Buffer status: %d blocks received, current audio length: %.2fs", 
                rolling_buffer.total_blocks_received, 
                len(audio_np) / SAMPLE_RATE
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
        
        # If too quiet, skip Whisper
        if rms < 0.003:
            time.sleep(0.2)
            continue

        logger.debug("🔊 Speech activity detected! RMS: %.4f. Submitting to Whisper...", rms)
        transcript = transcribe_audio(audio_np).lower()
        
        logger.debug("Whisper Output: '%s'", transcript)
        
        if WAKE_WORD in transcript:
            logger.info("✨ Wake word detected! Transcript: '%s'", transcript)
            rolling_buffer.clear()
            return True
        
        # Cooldown after a failed check to avoid hammering Whisper
        time.sleep(0.5)

def record_utterance():
    """Synchronously captures the command utterance after the wake word triggers."""
    logger.info("🎤 Command mode active. Speak your command...")
    audio = sd.rec(int(6.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]