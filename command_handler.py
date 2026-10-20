import logging
from concurrent.futures import ThreadPoolExecutor
from audio import record_utterance
from transcriber import transcribe_audio
from ha_client import fetch_ha_devices, execute_ha_command
from intent_parser import parse_intent

logger = logging.getLogger(__name__)

def handle_voice_command(noise_floor, shutdown_event=None):
    """
    Orchestrates a single voice command interaction:
    1. Fetches HA devices in parallel with recording utterance.
    2. Builds a lean vocabulary prompt from active device names.
    3. Transcribes audio via Whisper using the priming prompt.
    4. Parses intent and executes the Home Assistant service call.
    """
    if shutdown_event and shutdown_event.is_set():
        return

    logger.info("🎤 Command mode active. Initializing parallel tasks...")
    
    # Run Home Assistant device fetching in parallel with recording utterance
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_devices = executor.submit(fetch_ha_devices)
        audio_data = record_utterance(noise_floor)
        controllable_devices = future_devices.result()

    if (shutdown_event and shutdown_event.is_set()) or len(audio_data) == 0:
        logger.warning("⚠️ No audio recorded or shutdown requested. Returning to standby.")
        return

    # Build a lean context prompt: strict action verbs and your actual device names
    device_names = [d.get("name", "") for d in controllable_devices]
    prompt_context = (
        f"Turn on. Turn off. "
        f"Devices: {', '.join(device_names)}."
    )

    logger.info("🔄 Transcribing command via Whisper (with lean context prompt)...")
    transcript = transcribe_audio(audio_data, prompt=prompt_context)
    logger.info("➜ Transcript: '%s'", transcript)

    if (shutdown_event and shutdown_event.is_set()) or not transcript or "[blank_audio]" in transcript.lower():
        logger.info("⚠️ Empty transcript or shutdown requested. Returning to standby.")
        return

    logger.info("🧠 Parsing intent with Qwen SLM...")
    intent = parse_intent(transcript, controllable_devices)
    
    if shutdown_event and shutdown_event.is_set():
        return

    if intent and "domain" in intent and "service" in intent and "entity_id" in intent:
        logger.info(
            "⚡ Executing on Home Assistant: %s -> %s", 
            intent['service'], 
            intent['entity_id']
        )
        execute_ha_command(intent['domain'], intent['service'], intent['entity_id'])
    else:
        logger.warning("⚠️ Could not resolve intent or missing entity mapping.")