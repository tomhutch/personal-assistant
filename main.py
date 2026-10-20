import json
import os
from config import HA_TOKEN, SAMPLE_RATE
from ha_client import fetch_ha_devices, trigger_ha_service
from intent_parser import parse_intent_with_slm
from transcriber import transcribe_audio
from audio import listen_for_wake_word, record_utterance

def main_loop():
    if not HA_TOKEN:
        print("❌ Error: HA_TOKEN environment variable is not set!")
        return

    print("🔍 Fetching live devices from Home Assistant...")
    devices = fetch_ha_devices()
    if not devices:
        print("⚠️ No devices found or failed to connect to Home Assistant.")
        return
    
    device_list_str = json.dumps(devices, indent=2)
    print(f"   ➜ Discovered {len(devices)} controllable device(s).")
    for d in devices:
        print(f"     - {d['name']} ({d['entity_id']})")

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

    print("🚀 Starting synchronous assistant loop...")

    while True:
        try:
            # Step 1: Block and listen for 'Hey Susan'
            wake_triggered = listen_for_wake_word()
            
            if wake_triggered:
                # Step 2: Record command
                command_audio = record_utterance()
                
                # Step 3: Transcribe command via Whisper
                print("🔄 Transcribing command via Whisper...")
                transcript = transcribe_audio(command_audio)
                print(f"   ➜ Transcript: '{transcript}'")

                if not transcript.strip():
                    print("⚠️ Empty transcript, returning to standby.\n")
                    continue

                # Step 4: Parse intent via Qwen SLM
                print("🧠 Parsing intent with Qwen SLM...")
                tool_payload = parse_intent_with_slm(transcript, dynamic_system_prompt)
                if not tool_payload:
                    print("⚠️ Failed to parse intent, returning to standby.\n")
                    continue

                domain = tool_payload.get("domain", "switch")
                service = tool_payload.get("service", "turn_on")
                entity_id = tool_payload.get("entity_id")

                if not entity_id:
                    print("❌ SLM response missing target entity_id.\n")
                    continue

                # Step 5: Fire Home Assistant Service Request
                print(f"⚡ Executing on Home Assistant: {service} -> {entity_id}")
                trigger_ha_service(domain, service, entity_id)
                print("💤 Returning to standby...\n")
                
        except Exception as e:
            print(f"❌ Error in main loop: {e}")

if __name__ == "__main__":
    main_loop()