import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from audio import RollingAudioBuffer, listen_for_wake_word, record_utterance
from config import SAMPLE_RATE

class TestRollingAudioBuffer:
    def test_buffer_respects_max_duration(self):
        """Scenario: Pushing audio exceeding the ring buffer duration drops older blocks."""
        buffer = RollingAudioBuffer(duration_seconds=1.0)
        max_samples = buffer.max_samples
        
        block_size = int(SAMPLE_RATE * 0.5)
        block = np.ones(block_size, dtype=np.float32)
        
        buffer.append(block)
        buffer.append(block)
        buffer.append(block)  # 1.5s total pushed
        
        audio_np = buffer.get_audio()
        assert len(audio_np) <= max_samples

    def test_buffer_clear(self):
        """Scenario: Clearing the buffer resets its sample count and internal storage."""
        buffer = RollingAudioBuffer(duration_seconds=2.0)
        buffer.append(np.ones(int(SAMPLE_RATE * 1.0), dtype=np.float32))
        
        buffer.clear()
        audio_np = buffer.get_audio()
        assert len(audio_np) == 0

class TestListenForWakeWord:
    @patch("audio.transcribe_audio")
    def test_scenario_music_hallucinations_ignored(self, mock_transcribe):
        """Scenario: Background music triggers volume gate, but Whisper outputs music tags which are filtered out."""
        buffer = RollingAudioBuffer(duration_seconds=3.0)
        buffer.append(np.random.normal(0, 0.05, int(SAMPLE_RATE * 2.5)).astype(np.float32))
        
        # Whisper returns music tags first, then valid wake word
        mock_transcribe.side_effect = [
            "[music playing]",
            "[blank_audio]",
            "(drumming)",
            "hey susan, turn off plug one"
        ]
        
        with patch("time.sleep", return_value=None):
            triggered, noise_floor = listen_for_wake_word(buffer)
            
        assert triggered is True
        assert isinstance(noise_floor, float)
        assert mock_transcribe.call_count == 4

    @patch("audio.transcribe_audio")
    def test_scenario_clean_wake_word(self, mock_transcribe):
        """Scenario: Clean user voice saying the wake word triggers immediately on first try."""
        buffer = RollingAudioBuffer(duration_seconds=3.0)
        buffer.append(np.random.normal(0, 0.05, int(SAMPLE_RATE * 2.5)).astype(np.float32))
        
        mock_transcribe.return_value = "hey susan"
        
        with patch("time.sleep", return_value=None):
            triggered, noise_floor = listen_for_wake_word(buffer)
            
        assert triggered is True
        assert mock_transcribe.call_count == 1

class TestRecordUtterance:
    @patch("sounddevice.InputStream")
    def test_scenario_continuous_music_noise_resilience(self, mock_input_stream_cls):
        """Scenario: Continuous background music keeps audio elevated, but fixed post-speech window terminates cleanly."""
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        
        # Simulate persistent loud music chunks (RMS > speech threshold)
        music_chunk = np.full((chunk_samples, 1), 0.015, dtype=np.float32)
        mock_stream.read.side_effect = [(music_chunk, False) for _ in range(60)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(noise_floor)
        
        assert isinstance(audio_result, np.ndarray)
        assert len(audio_result) > 0
        # Ensures it didn't hang to max duration (10s), but cut off early via post-speech window
        assert len(audio_result) <= int(SAMPLE_RATE * 6.0)

    @patch("sounddevice.InputStream")
    def test_scenario_no_speech_timeout(self, mock_input_stream_cls):
        """Scenario: User triggers wake word but stays completely silent; recording times out safely."""
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)
        
        # Simulate complete silence (RMS below threshold)
        silent_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)
        mock_stream.read.side_effect = [(silent_chunk, False) for _ in range(40)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(noise_floor)
        
        assert isinstance(audio_result, np.ndarray)