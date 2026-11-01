import logging
import re
import threading
import time

from speech import play_chime_until, speak

logger = logging.getLogger(__name__)

ALARM_SECONDS = 60.0
_lock = threading.Lock()
_timers: list[dict] = []
_alarm_stop = threading.Event()
_alarm_active = threading.Event()
_STOP_WORDS = {"stop", "stop it", "stop the timer", "stop timer", "dismiss", "silence"}


def _remaining(timer: dict) -> float:
    if timer["paused_remaining"] is not None:
        return timer["paused_remaining"]
    return max(0.0, timer["end"] - time.monotonic())


def _schedule(timer: dict, seconds: float) -> None:
    if timer["handle"]:
        timer["handle"].cancel()
    timer["end"] = time.monotonic() + seconds
    timer["paused_remaining"] = None
    handle = threading.Timer(seconds, _fire, args=(timer,))
    handle.daemon = True
    timer["handle"] = handle
    handle.start()


def _fire(timer: dict) -> None:
    with _lock:
        if timer not in _timers:
            return
        _timers.remove(timer)
    label = f"{timer['name']} timer" if timer["name"] else "timer"
    logger.info("⏰ Timer finished: %s", label)
    _alarm_stop.clear()
    _alarm_active.set()
    try:
        play_chime_until(_alarm_stop, ALARM_SECONDS)
    except Exception as exc:
        logger.error("Failed to play timer chime: %s", exc)
    finally:
        _alarm_active.clear()


def stop_alarm() -> bool:
    """Silence a ringing timer; returns True if one was ringing."""
    ringing = _alarm_active.is_set()
    _alarm_stop.set()
    return ringing


def is_stop_command(transcript: str) -> bool:
    cleaned = re.sub(r"[^a-z ]", "", transcript.lower()).strip()
    return cleaned in _STOP_WORDS and _alarm_active.is_set()


def _describe(seconds: float) -> str:
    total = int(round(seconds))
    parts = []
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        value, total = divmod(total, size)
        if value:
            parts.append(f"{value} {unit}{'s' if value != 1 else ''}")
    if not parts:
        return "0 seconds"
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]


def _select(intent: dict) -> list[dict]:
    matches = list(_timers)
    if number := intent.get("timer_number"):
        matches = [t for t in matches if t["number"] == number]
    if start_seconds := intent.get("timer_start_seconds"):
        matches = [t for t in matches if round(t["initial"]) == round(start_seconds)]
    if name := intent.get("timer_name"):
        matches = [t for t in matches if (t["name"] or "").lower() == name.lower()]
    return matches


def _label(timer: dict) -> str:
    return f"{timer['name']} timer" if timer["name"] else f"timer {timer['number']}"


def handle_timer_intent(intent: dict) -> str:
    """Run a timer intent locally and return the text to speak."""
    action = intent["timer_action"]
    name = intent.get("timer_name")
    seconds = float(intent.get("timer_seconds", 0))

    if action in {"HassCancelTimer", "HassCancelAllTimers"} and stop_alarm():
        return "Okay, stopped."

    with _lock:
        if action == "HassStartTimer":
            if seconds <= 0:
                return "Sorry, I didn't catch how long the timer should be."
            used = {t["number"] for t in _timers}
            number = next(n for n in range(1, len(used) + 2) if n not in used)
            timer = {
                "name": name, "number": number, "initial": seconds,
                "handle": None, "end": 0.0, "paused_remaining": None,
            }
            _schedule(timer, seconds)
            _timers.append(timer)
            _timers.sort(key=lambda t: t["number"])
            if len(_timers) == 1 and not name:
                return f"Okay, timer set for {_describe(seconds)}."
            return f"Okay, {_label(timer)} set for {_describe(seconds)}."

        if action == "HassCancelAllTimers":
            count = len(_timers)
            for timer in _timers:
                timer["handle"].cancel()
            _timers.clear()
            return "Cancelled all timers." if count else "There are no timers running."

        matches = _select(intent)
        if not matches:
            return "Sorry, I couldn't find that timer."
        if len(matches) > 1 and action != "HassTimerStatus":
            listing = "; ".join(
                f"{_label(t)}, {_describe(_remaining(t))} left" for t in matches
            )
            return (
                f"You have {len(matches)} timers: {listing}. "
                "Say, for example, cancel the first timer."
            )

        if action == "HassTimerStatus":
            return " ".join(
                f"{_label(t).capitalize()} has "
                f"{_describe(_remaining(t))} left"
                f"{', paused' if t['paused_remaining'] is not None else ''}."
                for t in matches
            )

        timer = matches[0]
        if action == "HassCancelTimer":
            timer["handle"].cancel()
            _timers.remove(timer)
            return f"{_label(timer).capitalize()} cancelled."
        if action == "HassPauseTimer":
            if timer["paused_remaining"] is None:
                timer["paused_remaining"] = _remaining(timer)
                timer["handle"].cancel()
            return f"{_label(timer).capitalize()} paused."
        if action == "HassUnpauseTimer":
            if timer["paused_remaining"] is not None:
                _schedule(timer, timer["paused_remaining"])
            return f"{_label(timer).capitalize()} resumed."
        if action in {"HassIncreaseTimer", "HassDecreaseTimer"}:
            delta = seconds if action == "HassIncreaseTimer" else -seconds
            new_remaining = max(1.0, _remaining(timer) + delta)
            if timer["paused_remaining"] is not None:
                timer["paused_remaining"] = new_remaining
            else:
                _schedule(timer, new_remaining)
            return f"{_label(timer).capitalize()} now has {_describe(new_remaining)} left."

    return "Sorry, I can't do that with timers."
