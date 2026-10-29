import logging
from functools import lru_cache
from pathlib import Path

import sounddevice as sd

from config import PIPER_VOICE_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_voice(model_path: str):
    from piper import PiperVoice

    return PiperVoice.load(model_path)


def speak(text: str):
    model_path = Path(PIPER_VOICE_MODEL).expanduser().resolve()
    voice = _load_voice(str(model_path))

    with sd.OutputStream(
        samplerate=voice.config.sample_rate,
        channels=1,
        dtype="float32",
    ) as stream:
        for audio_chunk in voice.synthesize(text):
            stream.write(audio_chunk.audio_float_array.reshape(-1, 1))