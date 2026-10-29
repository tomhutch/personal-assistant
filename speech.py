import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from config import PIPER_VOICE_MODEL

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


def speak(text: str):
    model_path = Path(PIPER_VOICE_MODEL).expanduser().resolve()
    voice = _load_voice(str(model_path))
    output_device = sd.query_devices(kind="output")
    output_rate = int(output_device["default_samplerate"])
    logger.debug(
        "Piper TTS output device: %s (index %s, %d Hz)",
        output_device["name"],
        output_device["index"],
        output_rate,
    )
    audio_chunks = [chunk.audio_float_array for chunk in voice.synthesize(text)]
    if not audio_chunks:
        return

    audio = np.concatenate(audio_chunks)
    audio = _resample_audio(audio, voice.config.sample_rate, output_rate)

    with sd.OutputStream(
        device=output_device["index"],
        samplerate=output_rate,
        channels=1,
        dtype="float32",
    ) as stream:
        stream.write(audio.reshape(-1, 1))