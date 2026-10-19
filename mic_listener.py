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
AUDIO_FILENAME = "mic_input.wav"

SYSTEM_PROMPT = """You are a low-latency smart home tool router for Home Assistant.
Analyze the user's transcript and produce a JSON payload to control a smart plug.

Available Devices & Aliases:
- "Plug 1", "living room plug", "socket" -> Entity ID: "switch.sonoff_s60zbtpg" (Domain: switch)

Target JSON schema:
{
  "domain": "switch",
  "service": "turn_on", 
  "entity_id": "switch.sonoff_s60zbtpg"
}

Rules:
1. "service": use "turn_on" for turn on/activate, "turn_off" for turn off/deactivate, or "toggle".
2. "domain": match the domain of the selected entity.
3. OUTPUT STRICT RAW JSON ONLY. No markdown syntax, no explanations.
"""

def record_and_process():
    if not HA_TOKEN:
        print("❌ Error: HA_TOKEN environment variable is not set!")
        return

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

        # 1. Transcribe via Whisper
        print("🔄 Transcribing via Whisper (Port 8081)...")
        with open(temp_wav_path, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": f},
                data={"response_format": "json"}
            )
        
        print(f"   ➜ Raw Server Response Status: {resp.status_code}")

        res_json = resp.json()
        transcript = res_json.get("text", "").strip()
        print(f"   ➜ Transcript: '{transcript}'")

        if not transcript:
            print("⚠️ Empty transcript, skipping SLM processing.")
            return

        # 2. Parse via Qwen 3B SLM
        print("⚡ Parsing intent via Qwen 3B (Port 8080)...")
        payload = {
            "model": "qwen2.5-3b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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

        # 3. Fire API request to Home Assistant Service
        domain = tool_payload.get("domain", "switch")
        service = tool_payload.get("service", "turn_on")
        entity_id = tool_payload.get("entity_id", "switch.living_room_sonoff_s60zbtpg")

        ha_endpoint = f"{HA_URL}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }
        ha_payload = {"entity_id": entity_id}

        print(f"🚀 Firing POST to Home Assistant ({ha_endpoint})...")
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