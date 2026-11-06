import logging
import config as app_config
from intents.ai_intents import parse_intent_ai
from intents.deterministic_intents import parse_intent_deterministic

logger = logging.getLogger(__name__)


def parse_intent(transcript: str, controllable_devices: list) -> dict | None:
    """Entry point for the configured smart-home intent strategy."""
    if app_config.USE_DETERMINISTIC_INTENT:
        logger.debug("⚡ Parsing intent via deterministic strategy...")
        return parse_intent_deterministic(transcript, controllable_devices)

    deterministic_intent = parse_intent_deterministic(
        transcript, controllable_devices
    )
    if deterministic_intent and (
        "media_search_query" in deterministic_intent
        or "media_control_entity_id" in deterministic_intent
        or deterministic_intent.get("intent_name") in {
            "HassMediaPause", "HassMediaUnpause", "HassMediaNext", "HassMediaPrevious",
            "HassMediaPlayerMute", "HassMediaPlayerUnmute", "HassSetVolume", "HassSetVolumeRelative",
        }
    ):
        return deterministic_intent

    logger.debug("⚡ Parsing intent via AI strategy...")
    return parse_intent_ai(transcript, controllable_devices)
