import logging
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from audio import record_utterance
from transcriber import transcribe_audio
from ha_client import (
    execute_ha_command,
    fetch_ha_devices,
    get_ha_todo_items,
    get_ha_weather_forecast,
    process_ha_conversation,
)
from intent_parser import parse_intent
from led import AssistantState
from speech import speak
from timers import handle_timer_intent, is_stop_command, stop_alarm

logger = logging.getLogger(__name__)

def _announce_problem(message):
    try:
        speak(message)
    except Exception:
        logger.exception("Could not play spoken failure message.")

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
            if "fetch_ha_devices failed" in str(exc):
                _announce_problem("I couldn't access Home Assistant. Check its connection and access token.")
            elif "record_utterance failed" in str(exc):
                _announce_problem("I couldn't record your command. Please check the microphone.")
            else:
                _announce_problem("I couldn't start processing your command.")
            return

        if shutdown_event and shutdown_event.is_set():
            return
        if controllable_devices is None:
            if led:
                led.set_state(AssistantState.FAILURE)
            _announce_problem("I couldn't access Home Assistant. Check its connection and access token.")
            return
        if len(audio_data) == 0:
            logger.warning("⚠️ No audio recorded. Returning to standby.")
            if led:
                led.set_state(AssistantState.FAILURE)
            _announce_problem("I didn't hear a command.")
            return

        if led:
            led.set_state(AssistantState.PROCESSING)

        # Build a lean context prompt: strict action verbs and your actual device names
        device_names = [
            f"{device.get('name', '')} in {device['area']}"
            if device.get("area") else device.get("name", "")
            for device in controllable_devices
        ]
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
            _announce_problem("Sorry, I couldn't understand that.")
            return

        if is_stop_command(transcript):
            stop_alarm()
            if led:
                led.set_state(AssistantState.SUCCESS)
            return

        logger.debug("🧠 Parsing intent...")
        intent = parse_intent(transcript, controllable_devices)

        if shutdown_event and shutdown_event.is_set():
            return

        if intent is None:
            logger.error("❌ No intent could be parsed from the command.")
            if led:
                led.set_state(AssistantState.FAILURE)
            _announce_problem("Sorry, I couldn't parse the command.")
            return

        if "timer_action" in intent:
            speak(handle_timer_intent(intent))
            if led:
                led.set_state(AssistantState.SUCCESS)
            return

        if intent.get("conversation_text"):
            conversation_result = process_ha_conversation(intent["conversation_text"])
            speech_text = (
                conversation_result.get("response", {})
                .get("speech", {})
                .get("plain", {})
                .get("speech")
                if conversation_result else None
            )
            if not speech_text:
                logger.error("Home Assistant returned no speech for the recognized intent.")
                if led:
                    led.set_state(AssistantState.FAILURE)
                _announce_problem("Sorry, I couldn't complete that request.")
                return

            speak(speech_text)
            if led:
                led.set_state(AssistantState.SUCCESS)
            return

        if "weather_entity_id" in intent:
            forecast = get_ha_weather_forecast(intent["weather_entity_id"])
            if not forecast:
                if led:
                    led.set_state(AssistantState.FAILURE)
                _announce_problem("Sorry, I couldn't get the weather forecast.")
                return

            condition = str(forecast.get("condition", "unknown"))
            condition = {
                "clear-night": "clear at night",
                "partlycloudy": "partly cloudy",
                "lightning-rainy": "thunderstorms and rain",
                "snowy-rainy": "snow and rain",
                "windy-variant": "windy and cloudy",
            }.get(condition, condition.replace("-", " ").replace("_", " "))
            response = (
                f"Today's forecast for {intent.get('weather_name', 'your area')} "
                f"is {condition}"
            )
            unit = intent.get("temperature_unit") or forecast.get("temperature_unit")
            high = forecast.get("temperature")
            low = forecast.get("templow")
            if high is not None and low is not None and unit:
                response += f", high {high} {unit} and low {low} {unit}"
            elif high is not None and unit:
                response += f", around {high} {unit}"
            response += "."

            speak(response)
            if led:
                led.set_state(AssistantState.SUCCESS)
            return

        if "todo_items_entity_id" in intent:
            entity_id = intent["todo_items_entity_id"]
            items = get_ha_todo_items(entity_id)
            if items is None:
                if led:
                    led.set_state(AssistantState.FAILURE)
                _announce_problem("Sorry, I couldn't read that to-do list.")
                return

            summaries = [
                str(item["summary"])
                for item in items
                if item.get("status", "needs_action") == "needs_action"
                and item.get("summary")
            ]
            list_name = intent.get("todo_list_name", "Your to-do list")
            if not summaries:
                response = f"{list_name} is empty."
            else:
                item_label = "item" if len(summaries) == 1 else "items"
                response = (
                    f"{list_name} has {len(summaries)} {item_label}: "
                    f"{', '.join(summaries)}."
                )

            speak(response)
            if led:
                led.set_state(AssistantState.SUCCESS)
            return

        if intent and "domain" in intent and "service" in intent and "entity_id" in intent:
            brightness_pct = intent.get("brightness_pct")
            brightness_description = (
                f" at {brightness_pct}% brightness" if brightness_pct is not None else ""
            )
            logger.info(
                "⚡ Executing on Home Assistant: %s -> %s%s",
                intent['service'],
                intent['entity_id'],
                brightness_description,
            )
            command_succeeded = execute_ha_command(
                intent['domain'],
                intent['service'],
                intent['entity_id'],
                brightness_pct=brightness_pct,
            )
            if led:
                led.set_state(
                    AssistantState.SUCCESS if command_succeeded else AssistantState.FAILURE
                )
            if not command_succeeded:
                logger.error("❌ Home Assistant command failed for %s.", intent['entity_id'])
                _announce_problem("Hmm, I couldn't control that device. Please check Home Assistant.")
        else:
            logger.warning("⚠️ Could not resolve intent or missing entity mapping.")
            if led:
                led.set_state(AssistantState.FAILURE)
            _announce_problem("Sorry, I couldn't determine which device you meant.")
    except Exception:
        if led:
            led.set_state(AssistantState.FAILURE)
        _announce_problem("Sorry, something went wrong while handling that command.")
        raise

