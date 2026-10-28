import numpy as np
import pytest
import threading
from unittest.mock import patch, MagicMock
from audio import WakeWordManager, RollingAudioBuffer, record_utterance, find_input_device
from config import SAMPLE_RATE, WAKE_WORD_MODEL

class DummyEvent:
    def is_set(self):
        return False

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
    @patch("audio.WAKE_WORD_THRESHOLD", 0.5)
    @patch("audio.Model")
    def test_scenario_clean_wake_word(self, mock_model_cls):
        """Scenario: openWakeWord detects score above threshold using an isolated pipeline instance."""
        mock_model = MagicMock()
        mock_model.prediction_buffer = {WAKE_WORD_MODEL: [0.95]}
        mock_model_cls.return_value = mock_model

        pipeline = WakeWordManager()
        buffer = RollingAudioBuffer(duration_seconds=3.0)
        
        pipeline.push_audio_chunk(np.zeros(1280, dtype=np.float32))

        triggered, noise_floor = pipeline.listen_for_wake_word(buffer, DummyEvent())
        
        assert triggered is True
        assert isinstance(noise_floor, float)
        mock_model.predict.assert_called_once()

    @patch("audio.Model")
    def test_scenario_queue_timeout_returns_false(self, mock_model_cls):
        """Scenario: Queue empties without wake word trigger; function returns False cleanly."""
        mock_model = MagicMock()
        mock_model.prediction_buffer = {WAKE_WORD_MODEL: [0.01]}
        mock_model_cls.return_value = mock_model

        pipeline = WakeWordManager()
        buffer = RollingAudioBuffer(duration_seconds=3.0)
        
        triggered, noise_floor = pipeline.listen_for_wake_word(buffer, DummyEvent())
        
        assert triggered is False

class TestRecordUtterance:
    @patch("sounddevice.InputStream")
    def test_cancel_event_stops_recording_between_reads(self, mock_input_stream_cls):
        mock_stream = MagicMock()
        cancel_event = threading.Event()
        chunk_samples = int(SAMPLE_RATE * 0.1)
        silent_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)

        def read_and_cancel(samples):
            cancel_event.set()
            return silent_chunk, False

        mock_stream.read.side_effect = read_and_cancel
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream

        audio_result = record_utterance(0, 0.001, cancel_event)

        assert len(audio_result) == chunk_samples
        mock_stream.read.assert_called_once_with(chunk_samples)

    @patch("sounddevice.InputStream")
    def test_scenario_continuous_music_noise_resilience(self, mock_input_stream_cls):
        """Scenario: Continuous background music keeps audio elevated, but fixed post-speech window terminates cleanly."""
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)  # 100ms chunks
        
        music_chunk = np.full((chunk_samples, 1), 0.015, dtype=np.float32)
        mock_stream.read.side_effect = [(music_chunk, False) for _ in range(60)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(0, noise_floor)
        
        assert isinstance(audio_result, np.ndarray)
        assert len(audio_result) > 0
        assert len(audio_result) <= int(SAMPLE_RATE * 6.0)
        mock_input_stream_cls.assert_called_once_with(
            device=0, samplerate=SAMPLE_RATE, channels=1, dtype='float32'
        )

    @patch("sounddevice.InputStream")
    def test_scenario_no_speech_timeout(self, mock_input_stream_cls):
        """Scenario: User triggers wake word but stays completely silent; recording times out safely."""
        mock_stream = MagicMock()
        chunk_samples = int(SAMPLE_RATE * 0.1)
        
        silent_chunk = np.zeros((chunk_samples, 1), dtype=np.float32)
        mock_stream.read.side_effect = [(silent_chunk, False) for _ in range(40)]
        mock_input_stream_cls.return_value.__enter__.return_value = mock_stream
        
        noise_floor = 0.001
        audio_result = record_utterance(0, noise_floor)
        
        assert isinstance(audio_result, np.ndarray)
        mock_input_stream_cls.assert_called_once_with(
            device=0, samplerate=SAMPLE_RATE, channels=1, dtype='float32'
        )

    @patch("sounddevice.InputStream")
    def test_scenario_no_input_device_aborts(self, mock_input_stream_cls):
        """Scenario: No usable input device exists; recording aborts with empty audio and no stream."""
        
        audio_result = record_utterance(None, 0.001)
        
        assert isinstance(audio_result, np.ndarray)
        assert len(audio_result) == 0
        mock_input_stream_cls.assert_not_called()