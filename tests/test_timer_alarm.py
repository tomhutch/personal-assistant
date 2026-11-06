import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

import command_handler
import timers


@pytest.fixture
def alarm_state(monkeypatch):
    monkeypatch.setattr(timers, "_timers", [])
    monkeypatch.setattr(timers, "_alarm_stop", threading.Event())
    monkeypatch.setattr(timers, "_alarm_active", threading.Event())


@pytest.mark.parametrize(
    "transcript",
    ["stop", "Stop!", "stop it", "stop the timer", "stop timer", "dismiss", "silence"],
)
def test_stop_phrases_only_match_while_alarm_is_active(alarm_state, transcript):
    assert not timers.is_stop_command(transcript)
    timers._alarm_active.set()
    assert timers.is_stop_command(transcript)


def test_voice_stop_silences_a_ringing_timer(alarm_state, monkeypatch):
    started = threading.Event()
    stopped = threading.Event()

    def play_chime(stop_event, max_seconds):
        assert max_seconds == timers.ALARM_SECONDS
        started.set()
        if stop_event.wait(timeout=2):
            stopped.set()

    monkeypatch.setattr(timers, "play_chime_until", play_chime)
    monkeypatch.setattr(command_handler, "fetch_ha_devices", lambda: [])
    monkeypatch.setattr(
        command_handler, "record_utterance",
        lambda *args: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(command_handler, "transcribe_audio", lambda *args, **kwargs: "Stop.")
    parse_intent = MagicMock()
    speak = MagicMock()
    monkeypatch.setattr(command_handler, "parse_intent", parse_intent)
    monkeypatch.setattr(command_handler, "speak", speak)

    timer = {"name": None}
    timers._timers.append(timer)
    alarm_thread = threading.Thread(target=timers._fire, args=(timer,))
    alarm_thread.start()
    try:
        assert started.wait(timeout=1)
        command_handler.handle_voice_command(0, noise_floor=0.001)
        assert stopped.wait(timeout=1)
    finally:
        timers._alarm_stop.set()
        alarm_thread.join(timeout=2)

    assert not alarm_thread.is_alive()
    assert not timers._alarm_active.is_set()
    assert timer not in timers._timers
    parse_intent.assert_not_called()
    speak.assert_not_called()
