import json
import logging
import sounddevice as sd
from config import HA_TOKEN, LOG_LEVEL, SAMPLE_RATE
from ha_client import fetch_ha_devices, trigger_ha_service
from intent_parser import parse_intent_with_slm
from transcriber import transcribe_audio
from audio import RollingAudioBuffer, listen_for_wake_word, record_utterance

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def main_loop():
    if not HA_TOKEN:
        logger.error("❌ Error: HA_TOKEN environment variable is not set!")
        return

    logger.info("🔍 Fetching live devices from Home Assistant...")
    devices = fetch_ha_devices()
    if not devices:
        logger.warning("⚠️ No devices found or failed to connect to Home Assistant.")
        return
    
    device_list_str = json.dumps(devices, indent=2)
    logger.info("➜ Discovered %d controllable device(s).", len(devices))
    for d in devices:
        logger.info("  - %s (%s)", d['name'], d['entity_id'])

    dynamic_system_prompt = f"""You are a low-latency smart home tool router for Home Assistant.
Analyze the user's transcript and match their intent to one of the available devices below.

Available Devices:
{device_list_str}

Target JSON schema:
{{
  "domain": "<domain from matched device>",
  "service": "turn_on", 
  "entity_id": "<entity_id from matched device>"
}}

Rules:
1. "service": use "turn_on", "turn_off", or "toggle".
2. Match the user's spoken name or alias as closely as possible to the "name" or "entity_id" in the available devices list.
3. OUTPUT STRICT RAW JSON ONLY. No markdown syntax, no explanations.
"""

    # Initialize the 3-second ring buffer
    rolling_buffer = RollingAudioBuffer(duration_seconds=3.0)

    def audio_callback(indata, frames, time_info, status):
        block = indata[:, 0].copy()
        rolling_buffer.append(block)

    logger.info("🚀 Starting background audio stream & ring buffer...")
    
    try:
        # Start the continuous background audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            while True:
                # Step 1: Monitor ring buffer for 'Hey Susan'
                wake_triggered = listen_for_wake_word(rolling_buffer)
                
                if wake_triggered:
                    # Step 2: Record command
                    command_audio = record_utterance()
                    
                    # Step 3: Transcribe command via Whisper
                    logger.info("🔄 Transcribing command via Whisper...")
                    transcript = transcribe_audio(command_audio)
                    logger.info("➜ Transcript: '%s'", transcript)

                    if not transcript.strip():
                        logger.warning("⚠️ Empty transcript, returning to standby.\n")
                        rolling_buffer.clear()
                        continue

                    # Step 4: Parse intent via Qwen SLM
                    logger.info("🧠 Parsing intent with Qwen SLM...")
                    tool_payload = parse_intent_with_slm(transcript, dynamic_system_prompt)
                    if not tool_payload:
                        logger.warning("⚠️ Failed to parse intent, returning to standby.\n")
                        rolling_buffer.clear()
                        continue

                    domain = tool_payload.get("domain", "switch")
                    service = tool_payload.get("service", "turn_on")
                    entity_id = tool_payload.get("entity_id")

                    if not entity_id:
                        logger.error("❌ SLM response missing target entity_id.\n")
                        rolling_buffer.clear()
                        continue

                    # Step 5: Fire Home Assistant Service Request
                    logger.info("⚡ Executing on Home Assistant: %s -> %s", service, entity_id)
                    trigger_ha_service(domain, service, entity_id)
                    
                    rolling_buffer.clear()
                    logger.info("💤 Returning to standby...\n")
                
    except Exception as e:
        logger.error("❌ Error in main loop: %s", e)

if __name__ == "__main__":
    main_loop()