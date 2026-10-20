import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE, WAKE_WORD, MAX_COMMAND_TIME
from transcriber import transcribe_audio

def get_audio_rms(block):
    return np.sqrt(np.mean(block**2))

def listen_for_wake_word():
    """Continuously records 2.5-second chunks synchronously and checks Whisper."""
    print("💤 Listening for wake word ('Hey Susan')... [Sync Mode]")
    
    while True:
        # Record 2.5 seconds synchronously (blocking, 100% reliable)
        audio = sd.rec(int(2.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio_flat = audio[:, 0]
        
        # Optional: Skip if it's pure silence to save Whisper CPU cycles
        if get_audio_rms(audio_flat) < 0.001:
            continue

        transcript = transcribe_audio(audio_flat).lower()
        
        if transcript.strip():
            print(f"   [Whisper Heard]: '{transcript}'")
        
        if WAKE_WORD in transcript:
            print(f"✨ Wake word detected! Transcript: '{transcript}'")
            return True

def record_utterance():
    """Records the command utterance after the wake word is triggered."""
    print("🎤 Command mode active. Speak your command...")
    
    # Record up to 6 seconds of command audio synchronously
    # (or you can say your command and it captures the window)
    audio = sd.rec(int(6.0 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]