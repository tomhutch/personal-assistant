#!/usr/bin/env bash
#
# setup.sh - Master installation script for the Personal Assistant.
#
# Deploys the full local AI stack dynamically on this machine:
#   1. System dependencies (build tools, PortAudio/ALSA runtime libs)
#   2. Qwen 2.5 1.5B GGUF model download
#   3. llama.cpp build -> /usr/local/bin/llama-server (parallel 1 for safety)
#   4. whisper.cpp build + base.en model download (parallel 1 for safety)
#   5. Standalone app binary check & auto-permissions fix
#   6. Environment file setup (.env)
#   7. Dynamic systemd service registration (llama-server, whisper-server, personal-assistant)

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

echo "=== 1. Installing system dependencies ==="
sudo apt update
sudo apt install -y cmake build-essential git curl
sudo apt install -y libportaudio2
sudo apt install -y libasound2 2>/dev/null || sudo apt install -y libasound2t64
sudo usermod -aG audio "$DEPLOY_USER"

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

# Dynamically substitute BASE_DIR and DEPLOY_USER into templates
for service_file in systemd/*.service; do
    filename=$(basename "$service_file")
    sed -e "s|%%BASE_DIR%%|${BASE_DIR}|g" \
        -e "s|%%DEPLOY_USER%%|${DEPLOY_USER}|g" \
        "$service_file" > "$TEMP_DIR/$filename"
done

sudo cp "$TEMP_DIR"/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl enable --now whisper-server
sudo systemctl enable --now personal-assistant

echo "=== Deployment complete! ==="
echo "Check status:  systemctl status personal-assistant llama-server whisper-server"
echo "Check logs:    sudo journalctl -u personal-assistant -f"