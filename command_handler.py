import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from audio import record_utterance
from transcriber import transcribe_audio
from ha_client import fetch_ha_devices, execute_ha_command
from intent_parser import parse_intent
from led import AssistantState

logger = logging.getLogger(__name__)

def _run_named_task(name, function, *args):
    try:
        return function(*args)
    except Exception as exc:
        raise RuntimeError(f"{name} failed") from exc

def _fetch_devices_while_recording(input_device, noise_floor):
    cancel_recording = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        try:
            future_devices = executor.submit(
                _run_named_task, "fetch_ha_devices", fetch_ha_devices
            )
            future_audio = executor.submit(
                _run_named_task,
                "record_utterance",
                record_utterance,
                input_device,
                noise_floor,
                cancel_recording,
            )
            wait((future_devices, future_audio), return_when=FIRST_COMPLETED)

            if future_devices.done():
                controllable_devices = future_devices.result()
                if controllable_devices is None:
                    cancel_recording.set()
                audio_data = future_audio.result()
            else:
                audio_data = future_audio.result()
                controllable_devices = future_devices.result()
        except Exception:
            cancel_recording.set()
            raise
    return audio_data, controllable_devices

def handle_voice_command(input_device, noise_floor, shutdown_event=None, led=None):
    """
    Orchestrates a single voice command interaction:
    1. Fetches HA devices in parallel with recording utterance.
    2. Builds a lean vocabulary prompt from active device names.
    3. Transcribes audio via Whisper using the priming prompt.
    4. Parses intent and executes the Home Assistant service call.
    """
    if shutdown_event and shutdown_event.is_set():
        return

    if led:
        led.set_state(AssistantState.LISTENING)

    logger.debug("🎤 Command mode active. Initializing parallel tasks...")

    try:
        try:
            audio_data, controllable_devices = _fetch_devices_while_recording(
                input_device, noise_floor
            )
        except Exception as exc:
            logger.exception("❌ Parallel command setup failed: %s", exc)
            if led:
                led.set_state(AssistantState.FAILURE)
            return

        if shutdown_event and shutdown_event.is_set():
            return
        if controllable_devices is None:
            if led:
                led.set_state(AssistantState.FAILURE)
            return
        if len(audio_data) == 0:
            logger.warning("⚠️ No audio recorded. Returning to standby.")
            if led:
                led.set_state(AssistantState.FAILURE)
            return

        if led:
            led.set_state(AssistantState.PROCESSING)

        # Build a lean context prompt: strict action verbs and your actual device names
        device_names = [d.get("name", "") for d in controllable_devices]
        prompt_context = (
            f"Turn on. Turn off. "
            f"Devices: {', '.join(device_names)}."
        )

        logger.debug("🔄 Transcribing command via Whisper (with lean context prompt)...")
        transcript = transcribe_audio(audio_data, prompt=prompt_context)
        logger.debug("➜ Transcript: '%s'", transcript)

        if shutdown_event and shutdown_event.is_set():
            return
        if not transcript or "[blank_audio]" in transcript.lower():
            logger.info("⚠️ Empty transcript. Returning to standby.")
            if led:
                led.set_state(AssistantState.FAILURE)
            return

        logger.debug("🧠 Parsing intent with Qwen SLM...")
        intent = parse_intent(transcript, controllable_devices)

        if shutdown_event and shutdown_event.is_set():
            return

        if intent and "domain" in intent and "service" in intent and "entity_id" in intent:
            logger.info(
                "⚡ Executing on Home Assistant: %s -> %s",
                intent['service'],
                intent['entity_id']
            )
            command_succeeded = execute_ha_command(
                intent['domain'], intent['service'], intent['entity_id']
            )
            if led:
                led.set_state(
                    AssistantState.SUCCESS if command_succeeded else AssistantState.FAILURE
                )
            if not command_succeeded:
                logger.error("❌ Home Assistant command failed for %s.", intent['entity_id'])
        else:
            logger.warning("⚠️ Could not resolve intent or missing entity mapping.")
            if led:
                led.set_state(AssistantState.FAILURE)
    except Exception:
        if led:
            led.set_state(AssistantState.FAILURE)
        raise