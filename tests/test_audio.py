import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from audio import RollingAudioBuffer, listen_for_wake_word, record_utterance, find_input_device
from config import SAMPLE_RATE

def make_devices(*specs):
    """Builds a fake PortAudio device list from (name, input_ch, output_ch) specs without explicit indices."""
    return [
        {"name": name, "max_input_channels": inputs, "max_output_channels": outputs}
        for name, inputs, outputs in specs
    ]

class TestFindInputDevice:
    @patch("sounddevice.query_devices")
    def test_prefers_usb_input_device(self, mock_query):
        """Scenario: A USB input device is chosen even when other inputs exist first."""
        mock_query.return_value = make_devices(
            ("PulseAudio", 2, 2),                 # Index 0
            ("USB Conference Speakerphone", 1, 1), # Index 1 (USB wins)
            ("Pulse Output", 0, 2),                 # Index 2
        )
        assert find_input_device() == 1

    @patch("sounddevice.query_devices")
    def test_skips_monitor_devices(self, mock_query):
        """Scenario: A .monitor device with input channels is skipped in favor of a true USB input."""
        mock_query.return_value = make_devices(
            ("alsa_output.usb-ANKER.monitor", 2, 0),        # Index 0 (Skipped: .monitor)
            ("alsa_input.usb-ANKER.analog-stereo", 2, 0), # Index 1 (Wins)
        )
        assert find_input_device() == 1

    @patch("sounddevice.query_devices")
    def test_usb_without_inputs_is_skipped(self, mock_query):
        """Scenario: A USB device with no input channels is not selected; fallback picks the real input."""
        mock_query.return_value = make_devices(
            ("USB Sound Card (out only)", 0, 2), # Index 0 (Skipped: 0 inputs)
            ("PulseAudio", 2, 2),                 # Index 1 (Wins)
        )
        assert find_input_device() == 1

    @patch("sounddevice.query_devices")
    def test_fallback_skips_loopback_devices(self, mock_query):
        """Scenario: With no USB input, the first non-loopback input device wins."""
        mock_query.return_value = make_devices(
            ("PulseAudio", 2, 2), # Index 0 (Wins)
            ("Loopback 1", 2, 0), # Index 1 (Skipped: loopback)
        )
        assert find_input_device() == 0

    @patch("sounddevice.query_devices")
    def test_only_loopback_returns_none(self, mock_query):
        """Scenario: When only a virtual loopback is available, no device is returned."""
        mock_query.return_value = make_devices(
            ("Loopback 1", 2, 0),
            ("Pulse Output", 0, 2),
        )
        assert find_input_device() is None

    @patch("sounddevice.query_devices")
    def test_no_input_devices_returns_none(self, mock_query):
        """Scenario: An output-only or empty device list yields None."""
        mock_query.return_value = make_devices(("Pulse Output", 0, 2))
        assert find_input_device() is None

        mock_query.return_value = []
        assert find_input_device() is None

    @patch("sounddevice.query_devices")
    def test_query_failure_returns_none(self, mock_query):
        """Scenario: If PortAudio cannot be queried at all, None is returned instead of raising."""
        mock_query.side_effect = RuntimeError("PortAudio initialization failed")
        assert find_input_device() is None

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
    @patch("sounddevice.query_devices")
    @patch("sounddevice.InputStream")
    def test_scenario_continuous_music_noise_resilience(self, mock_input_stream_cls, mock_query):
        """Scenario: Continuous background music keeps audio elevated, but fixed post-speech window terminates cleanly."""
        mock_query.return_value = make_devices(("PulseAudio", 2, 2))
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        
        music_chunk = np.full((chunk_samples, 1), 0.015, dtype=np.float32)
        mock_stream.read.side_effect = [(music_chunk, False) for _ in range(60)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(noise_floor)
        
        assert isinstance(audio_result, np.ndarray)
        assert len(audio_result) > 0
        assert len(audio_result) <= int(SAMPLE_RATE * 6.0)
        mock_input_stream_cls.assert_called_once_with(
            device=0, samplerate=SAMPLE_RATE, channels=1, dtype='float32'
        )

    @patch("sounddevice.query_devices")
    @patch("sounddevice.InputStream")
    def test_scenario_no_speech_timeout(self, mock_input_stream_cls, mock_query):
        """Scenario: User triggers wake word but stays completely silent; recording times out safely."""
        mock_query.return_value = make_devices(
            ("PulseAudio", 2, 2),                 # Index 0
            ("USB Conference Speakerphone", 1, 1), # Index 1
        )
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)
        
        silent_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)
        mock_stream.read.side_effect = [(silent_chunk, False) for _ in range(40)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(noise_floor)
        
        assert isinstance(audio_result, np.ndarray)
        mock_input_stream_cls.assert_called_once_with(
            device=1, samplerate=SAMPLE_RATE, channels=1, dtype='float32'
        )

    @patch("sounddevice.query_devices")
    @patch("sounddevice.InputStream")
    def test_scenario_no_input_device_aborts(self, mock_input_stream_cls, mock_query):
        """Scenario: No usable input device exists; recording aborts with empty audio and no stream."""
        mock_query.return_value = make_devices(("Pulse Output", 0, 2))
        
        audio_result = record_utterance(0.001)
        
        assert isinstance(audio_result, np.ndarray)
        assert len(audio_result) == 0
        mock_input_stream_cls.assert_not_called()