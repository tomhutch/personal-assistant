import logging
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from config import PIPER_OUTPUT_DEVICE, PIPER_VOICE_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_voice(model_path: str):
    from piper import PiperVoice

    return PiperVoice.load(model_path)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    divisor = math.gcd(source_rate, target_rate)
    resampled_audio = resample_poly(
        audio,
        up=target_rate // divisor,
        down=source_rate // divisor,
    ).astype(np.float32)
    return np.clip(resampled_audio, -1.0, 1.0)


def _select_output_device():
    devices = sd.query_devices()
    output_devices = [
        (index, device)
        for index, device in enumerate(devices)
        if device["max_output_channels"] > 0
    ]

    def prefer_pulseaudio(matches):
        if not matches:
            return None
        hostapis = sd.query_hostapis()
        return next(
            (
                (index, device)
                for index, device in matches
                if "pulse" in hostapis[device["hostapi"]]["name"].casefold()
            ),
            matches[0],
        )

    configured_name = re.sub(r"[^a-z0-9]", "", PIPER_OUTPUT_DEVICE.casefold())
    if configured_name:
        matches = [
            (index, device) for index, device in output_devices
            if configured_name in re.sub(r"[^a-z0-9]", "", device["name"].casefold())
        ]
        if matches:
            return prefer_pulseaudio(matches)

        logger.warning(
            "Configured Piper output device '%s' was not found; selecting automatically.",
            PIPER_OUTPUT_DEVICE,
        )

    def is_non_hardware_output(device):
        name = device["name"].casefold()
        return "hw:" not in name and "loopback" not in name and ".monitor" not in name

    usb_outputs = [
        (index, device)
        for index, device in output_devices
        if "usb" in device["name"].casefold() and is_non_hardware_output(device)
    ]
    if usb_outputs:
        return prefer_pulseaudio(usb_outputs)

    other_outputs = [
        (index, device)
        for index, device in output_devices
        if is_non_hardware_output(device)
    ]
    pulseaudio_output = prefer_pulseaudio(other_outputs)
    if pulseaudio_output:
        return pulseaudio_output

    device = sd.query_devices(kind="output")
    return int(device["index"]), device


def speak(text: str):
    model_path = Path(PIPER_VOICE_MODEL).expanduser().resolve()
    voice = _load_voice(str(model_path))
    output_index, output_device = _select_output_device()
    output_rate = int(output_device["default_samplerate"])
    logger.debug(
        "Piper TTS output device: %s (index %s, %d Hz)",
        output_device["name"],
        output_index,
        output_rate,
    )
    audio_chunks = [chunk.audio_float_array for chunk in voice.synthesize(text)]
    if not audio_chunks:
        return

    audio = np.concatenate(audio_chunks)
    audio = _resample_audio(audio, voice.config.sample_rate, output_rate)

    with sd.OutputStream(
        device=output_index,
        samplerate=output_rate,
        channels=1,
        dtype="float32",
    ) as stream:
        stream.write(audio.reshape(-1, 1))