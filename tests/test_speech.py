from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from speech import speak


@patch("speech._load_voice")
@patch("speech.sd.OutputStream")
def test_speak_streams_voice_audio_at_model_sample_rate(mock_output_stream, mock_load_voice):
    audio = np.array([0.1, -0.1], dtype=np.float32)
    voice = MagicMock()
    voice.config.sample_rate = 22050
    voice.synthesize.return_value = [SimpleNamespace(audio_float_array=audio)]
    mock_load_voice.return_value = voice
    stream = mock_output_stream.return_value.__enter__.return_value

    speak("I couldn't access Home Assistant.")

    mock_output_stream.assert_called_once_with(
        samplerate=22050,
        channels=1,
        dtype="float32",
    )
    voice.synthesize.assert_called_once_with("I couldn't access Home Assistant.")
    np.testing.assert_array_equal(stream.write.call_args.args[0], audio.reshape(-1, 1))