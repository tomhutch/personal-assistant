#!/usr/bin/env python3
import sounddevice as sd
from scipy.io.wavfile import write
import requests
import json
import re
import tempfile
import os
import numpy as np

WHISPER_URL = "http://localhost:8081/inference"
SLM_URL = "http://localhost:8080/v1/chat/completions"  # Qwen 2.5 3B endpoint

# Home Assistant Configuration
HA_URL = "http://localhost:8123"
HA_TOKEN = os.getenv("HA_TOKEN")

SAMPLE_RATE = 16000  # 16kHz is ideal for Whisper

def fetch_ha_devices():
    """Dynamically queries Home Assistant to get a live list of controllable entities."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(f"{HA_URL}/api/states", headers=headers)
        if resp.status_code != 200:
            print(f"⚠️ Failed to fetch devices from HA: {resp.status_code}")
            return []
        
        states = resp.json()
        devices = []
        # Filter for controllable domains
        supported_domains = ("switch.", "light.", "fan.")
        
        for state in states:
            entity_id = state.get("entity_id", "")
            # Skip auxiliary toggle entities like indicator LEDs
            if "_network_led" in entity_id or "_indicator" in entity_id:
                continue

            if entity_id.startswith(supported_domains):
                friendly_name = state.get("attributes", {}).get("friendly_name", entity_id)
                domain = entity_id.split(".")[0]
                devices.append({
                    "entity_id": entity_id,
                    "name": friendly_name,
                    "domain": domain
                })
        return devices
    except Exception as e:
        print(f"⚠️ Error connecting to HA for device discovery: {e}")
        return []

def record_and_process():
    if not HA_TOKEN:
        print("❌ Error: HA_TOKEN environment variable is not set!")
        return

    # 1. Fetch live devices from Home Assistant for dynamic prompting
    print("🔍 Fetching live devices from Home Assistant...")
    devices = fetch_ha_devices()
    if not devices:
        print("⚠️ No devices found or failed to connect to Home Assistant.")
        return
    
    device_list_str = json.dumps(devices, indent=2)
    print(f"   ➜ Discovered {len(devices)} controllable device(s).")
    for d in devices:
        print(f"      • {d['name']} ({d['entity_id']})")

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
1. "service": use "turn_on" for turn on/activate, "turn_off" for turn off/deactivate, or "toggle".
2. Match the user's spoken name or alias as closely as possible to the "name" or "entity_id" in the available devices list.
3. OUTPUT STRICT RAW JSON ONLY. No markdown syntax, no explanations.
"""

    print("🎙️ Listening... Speak into your MacBook mic. Press **Enter** when done.")
    
    audio_data = []
    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    # Stream audio from default microphone
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        input()  # Blocks until you press Enter

    print("🛑 Recording stopped. Processing audio...")

    audio_np = np.concatenate(audio_data, axis=0)

    # Create a secure temporary file that deletes itself upon closing/exit
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        temp_wav_path = tmp_audio.name

    try:
        # Write audio data to the temp file
        write(temp_wav_path, SAMPLE_RATE, audio_np)

        # 2. Transcribe via Whisper
        print("🔄 Transcribing via Whisper (Port 8081)...")
        with open(temp_wav_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": f},
                data={"response_format": "json"}
            )
        
        print(f"   ➜ Raw Server Response Status: {resp.status_code}")

        try:
            res_json = resp.json()
            transcript = res_json.get("text", "").strip()
        except Exception as e:
            print(f"❌ JSON Decode Failed: {e}")
            return

        print(f"   ➜ Transcript: '{transcript}'")

        if not transcript:
            print("⚠️ Empty transcript, skipping SLM processing.")
            return

        # 3. Parse via Qwen 3B SLM using dynamic device list
        print("⚡ Parsing intent via Qwen 3B (Port 8080)...")
        payload = {
            "model": "qwen2.5-3b",
            "messages": [
                {"role": "system", "content": dynamic_system_prompt},
                {"role": "user", "content": transcript}
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }

        slm_resp = requests.post(SLM_URL, json=payload)
        print(f"   ➜ SLM Status Code: {slm_resp.status_code}")

        try:
            slm_data = slm_resp.json()
            raw_content = slm_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"❌ Failed to parse SLM response as JSON: {e}")
            return

        cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        
        try:
            tool_payload = json.loads(cleaned_json)
        except json.JSONDecodeError:
            print(f"❌ Failed to decode cleaned JSON string: {cleaned_json}")
            return

        # 4. Fire API request to Home Assistant Service
        domain = tool_payload.get("domain", "switch")
        service = tool_payload.get("service", "turn_on")
        entity_id = tool_payload.get("entity_id")

        if not entity_id:
            print("❌ SLM response missing target entity_id.")
            return

        ha_endpoint = f"{HA_URL}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }
        ha_payload = {"entity_id": entity_id}

        print(f"🚀 Firing POST to Home Assistant ({ha_endpoint}) for {entity_id}...")
        api_resp = requests.post(ha_endpoint, headers=headers, json=ha_payload)
        print(f"   ➜ HA API Status Code: {api_resp.status_code}")
        print(f"   ➜ HA Response: {api_resp.text}\n")
    
    finally:
        # Cleanup: ensure the temporary wav file is always deleted from disk
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

if __name__ == "__main__":
    while True:
        record_and_process()
        cont = input("Press Enter to talk again, or 'q' then Enter to quit: ")
        if cont.lower() == 'q':
            break