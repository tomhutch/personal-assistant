import requests
from config import HA_URL, HA_TOKEN

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
        supported_domains = ("switch.", "light.", "fan.")
        
        for state in states:
            entity_id = state.get("entity_id", "")
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

def trigger_ha_service(domain, service, entity_id):
    """Fires a POST request to Home Assistant service endpoint."""
    ha_endpoint = f"{HA_URL}/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    ha_payload = {"entity_id": entity_id}

    print(f"🚀 Firing POST to Home Assistant for {entity_id}...")
    api_resp = requests.post(ha_endpoint, headers=headers, json=ha_payload)
    print(f"   ➜ HA API Status Code: {api_resp.status_code}")
    print(f"   ➜ HA Response: {api_resp.text}\n")
    return api_resp
