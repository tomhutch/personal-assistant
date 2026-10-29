from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from speech import speak


@patch("speech._load_voice")
@patch("speech.sd.OutputStream")
@patch(
    "speech.sd.query_devices",
    return_value={"name": "Test Output", "index": 1, "default_samplerate": 48000},
)
def test_speak_resamples_and_streams_at_output_sample_rate(
    mock_query_devices, mock_output_stream, mock_load_voice
):
    audio = np.ones(1000, dtype=np.float32)
    voice = MagicMock()
    voice.config.sample_rate = 22050
    voice.synthesize.return_value = [SimpleNamespace(audio_float_array=audio)]
    mock_load_voice.return_value = voice
    stream = mock_output_stream.return_value.__enter__.return_value

    speak("I couldn't access Home Assistant.")

    mock_output_stream.assert_called_once_with(
        device=1,
        samplerate=48000,
        channels=1,
        dtype="float32",
    )
    voice.synthesize.assert_called_once_with("I couldn't access Home Assistant.")
    written_audio = stream.write.call_args_list[0].args[0]
    assert written_audio[0, 0] != 0
    assert np.max(np.abs(written_audio)) <= 1
    stream.write.assert_called_once()
    mock_query_devices.assert_called_once_with(kind="output")