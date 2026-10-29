from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from speech import _select_output_device, speak


@patch("speech._load_voice")
@patch("speech.sd.OutputStream")
def test_speak_resamples_and_streams_at_output_sample_rate(
    mock_output_stream, mock_load_voice, monkeypatch
):
    monkeypatch.setattr(
        "speech._select_output_device",
        lambda: (1, {"name": "Test Output", "default_samplerate": 48000}),
    )
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


@patch("speech.sd.query_devices")
@patch("speech.sd.query_hostapis")
def test_select_output_device_prefers_matching_pulseaudio_sink(
    mock_query_hostapis, mock_query_devices, monkeypatch
):
    monkeypatch.setattr("speech.PIPER_OUTPUT_DEVICE", "Anker PowerConf S330")
    mock_query_devices.return_value = [
        {
            "name": "Anker PowerConf S330: USB Audio (hw:1,0)",
            "max_output_channels": 2,
            "hostapi": 0,
            "default_samplerate": 48000,
        },
        {
            "name": "alsa_output.usb-ANKER_Anker_PowerConf_S330.analog-stereo",
            "max_output_channels": 2,
            "hostapi": 1,
            "default_samplerate": 44100,
        },
    ]
    mock_query_hostapis.return_value = [{"name": "ALSA"}, {"name": "PulseAudio"}]

    index, device = _select_output_device()

    assert index == 1
    assert "analog-stereo" in device["name"]


@patch("speech.sd.query_devices")
def test_select_output_device_uses_system_default_when_unset(
    mock_query_devices, monkeypatch, caplog
):
    default_device = {
        "name": "default",
        "index": 25,
        "default_samplerate": 44100,
    }
    monkeypatch.setattr("speech.PIPER_OUTPUT_DEVICE", "")
    mock_query_devices.side_effect = [[], default_device]

    index, device = _select_output_device()

    assert index == 25
    assert device is default_device
    mock_query_devices.assert_any_call()
    assert mock_query_devices.call_count == 2
    mock_query_devices.assert_called_with(kind="output")
    assert "Configured Piper output device" not in caplog.text


@patch("speech.sd.query_devices")
@patch("speech.sd.query_hostapis")
def test_select_output_device_auto_prefers_usb_pulseaudio_alias(
    mock_query_hostapis, mock_query_devices, monkeypatch
):
    monkeypatch.setattr("speech.PIPER_OUTPUT_DEVICE", "")
    mock_query_devices.side_effect = [[
        {
            "name": "Anker PowerConf S330: USB Audio (hw:1,0)",
            "max_output_channels": 2,
            "hostapi": 0,
            "default_samplerate": 48000,
        },
        {
            "name": "alsa_output.usb-ANKER_Anker_PowerConf_S330.analog-stereo",
            "max_output_channels": 2,
            "hostapi": 1,
            "default_samplerate": 44100,
        },
        {
            "name": "alsa_output.platform-snd_aloop.0.monitor",
            "max_output_channels": 2,
            "hostapi": 1,
            "default_samplerate": 44100,
        },
    ]]
    mock_query_hostapis.return_value = [{"name": "ALSA"}, {"name": "PulseAudio"}]

    index, device = _select_output_device()

    assert index == 1
    assert "analog-stereo" in device["name"]