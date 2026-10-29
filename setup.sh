#!/usr/bin/env bash
#
# setup.sh - Master installation script for the Personal Assistant.
#
# Deploys the full local AI stack dynamically on this machine:
#   0. Stopping existing services (if running)
#   1. System dependencies (build tools, PortAudio/ALSA runtime libs)
#   2. Qwen 2.5 1.5B GGUF model download
#   3. llama.cpp build -> /usr/local/bin/llama-server (parallel 1 for safety)
#   4. whisper.cpp build + base.en model download (parallel 1 for safety)
#   5. Standalone app binary check & auto-permissions fix
#   6. Environment file setup (.env)
#   7. Dynamic systemd service registration (system-wide for servers, user-space for assistant)

set -euo pipefail

# --- Configuration ------------------------------------------------------------
DEPLOY_USER="${DEPLOY_USER:-$(whoami)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
APP_BIN="personal-assistant"
LLAMA_SERVER_BIN="/usr/local/bin/llama-server"

QWEN_MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/${QWEN_MODEL}"
WHISPER_MODEL="base.en"
PIPER_VOICE="en_GB-southern_english_female-low"
PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/southern_english_female/low/${PIPER_VOICE}"

# --- Sanity checks -------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: do not run as root. Run as '$DEPLOY_USER' (sudo is invoked internally where needed)." >&2
    exit 1
fi

if [ ! -f main.py ] || [ ! -d systemd ]; then
    echo "ERROR: $SCRIPT_DIR does not look like the personal-assistant repo (missing main.py or systemd/)." >&2
    exit 1
fi

cd "$BASE_DIR"

# --- Stop existing services to prevent file locks and stale processes ---
echo "=== 0. Stopping existing services (if running) ==="
sudo systemctl stop llama-server whisper-server 2>/dev/null || true
systemctl --user stop personal-assistant 2>/dev/null || true

echo "=== 1. Installing system dependencies ==="
sudo apt update
sudo apt install -y cmake build-essential git curl
sudo apt install -y libportaudio2
sudo apt install -y libasound2 2>/dev/null || sudo apt install -y libasound2t64
sudo usermod -aG audio "$DEPLOY_USER"
sudo apt install -y libusb-1.0-0 libusb-1.0-0-dev

echo "=== Installing Piper voice ==="
for extension in onnx onnx.json; do
    voice_file="${PIPER_VOICE}.${extension}"
    if [ ! -s "$voice_file" ]; then
        curl -L --fail -o "${voice_file}.part" "${PIPER_VOICE_URL}.${extension}?download=true"
        mv "${voice_file}.part" "$voice_file"
    else
        echo "$voice_file already present, skipping."
    fi
done

echo "Configuring BlinkStick USB access for $DEPLOY_USER"
sudo groupadd --force plugdev
sudo usermod -aG plugdev "$DEPLOY_USER"
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="20a0", ATTR{idProduct}=="41e5", MODE="0666", GROUP="plugdev"' \
    | sudo tee /etc/udev/rules.d/85-blinkstick.rules >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger --action=add --subsystem-match=usb
echo "BlinkStick udev rule installed and applied to USB devices."

# Enable user lingering so user services run persistently even when logged out
echo "Enabling systemd linger for user: $DEPLOY_USER"
sudo loginctl enable-linger "$DEPLOY_USER"

echo "=== 2. Downloading Qwen 2.5 1.5B model ==="
mkdir -p models
if [ ! -f "models/${QWEN_MODEL}" ]; then
    echo "Downloading ${QWEN_MODEL}..."
    curl -L --fail -o "models/${QWEN_MODEL}.part" "$QWEN_URL"
    mv "models/${QWEN_MODEL}.part" "models/${QWEN_MODEL}"
else
    echo "${QWEN_MODEL} already present, skipping."
fi

echo "=== 3. Building llama.cpp (llama-server) ==="
if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    [ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp.git
    cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build llama.cpp/build --config Release --target llama-server --parallel 1
    sudo cp llama.cpp/build/bin/llama-server "$LLAMA_SERVER_BIN"
else
    echo "llama-server already installed at $LLAMA_SERVER_BIN, skipping."
fi

echo "=== 4. Building whisper.cpp ==="
[ -d whisper.cpp ] || git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper.cpp/build --config Release --target whisper-server --parallel 1
( cd whisper.cpp && bash ./models/download-ggml-model.sh "$WHISPER_MODEL" )

echo "=== 5. Checking the standalone app binary ==="
if [ -f "$APP_BIN" ] && [ ! -x "$APP_BIN" ]; then
    echo "Found $APP_BIN, but it lacks execution permissions. Fixing..."
    chmod +x "$APP_BIN"
fi

if [ ! -x "$APP_BIN" ]; then
    echo "ERROR: the standalone app binary is missing: $BASE_DIR/$APP_BIN" >&2
    echo "Build it on your dev machine and copy it to the server:" >&2
    echo "  make binary" >&2
    echo "  scp dist/${APP_BIN} ${DEPLOY_USER}@<server>:${BASE_DIR}/" >&2
    exit 1
fi
echo "Found and verified ./${APP_BIN}"

echo "=== 6. Preparing .env ==="
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env from .env.example."
    echo ">>> EDIT .env AND SET TOKENS/CONFIG BEFORE CONTROLLING SERVICES."
else
    echo ".env already exists, leaving it untouched."
fi

echo "=== 7. Registering systemd services ==="
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Prepare user service directory
mkdir -p ~/.config/systemd/user/

# Dynamically substitute BASE_DIR and QWEN_MODEL into templates
for service_file in systemd/*.service; do
    filename=$(basename "$service_file")
    sed -e "s|%%BASE_DIR%%|${BASE_DIR}|g" \
        -e "s|%%DEPLOY_USER%%|${DEPLOY_USER}|g" \
        -e "s|%%QWEN_MODEL%%|${QWEN_MODEL}|g" \
        "$service_file" > "$TEMP_DIR/$filename"
done

# Install backend servers globally (system-level)
sudo cp "$TEMP_DIR/llama-server.service" "$TEMP_DIR/whisper-server.service" /etc/systemd/system/ 2>/dev/null || true
sudo systemctl daemon-reload
sudo systemctl enable llama-server whisper-server
sudo systemctl restart llama-server whisper-server

# Install personal assistant as a user service
cp "$TEMP_DIR/personal-assistant.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable personal-assistant.service
systemctl --user restart personal-assistant.service

echo "=== Deployment complete! ==="
echo "Check server status:       sudo systemctl status llama-server whisper-server"
echo "Check assistant status:    systemctl --user status personal-assistant"
echo "Check assistant logs:      journalctl --user -u personal-assistant -f"