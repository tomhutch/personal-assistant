#!/usr/bin/env bash
#
# setup.sh - Master installation script for the Personal Assistant.
#
# Deploys the full local AI stack on this Ubuntu machine:
#   1. System dependencies (build tools, PortAudio/ALSA runtime libs)
#   2. Qwen 2.5 3B GGUF model download
#   3. llama.cpp build  -> /usr/local/bin/llama-server
#   4. whisper.cpp build + base.en model download
#   5. Standalone app binary check
#   6. systemd service registration (llama-server, whisper-server, python-app)
#
# The Python app runs as a standalone PyInstaller binary, so the server needs
# NO Python, pip, or venv. Build the binary on your dev machine first:
#
#   make binary    # or: docker build --platform linux/amd64 -f Dockerfile.builder -o ./dist .
#   scp dist/personal-assistant ubuntu@<server>:/home/ubuntu/personal-assistant/
#
# Then on the server:
#   git clone <your-repo-url> /home/ubuntu/personal-assistant
#   cd /home/ubuntu/personal-assistant
#   sudo bash setup.sh
#
# Run as the 'ubuntu' user (NOT root); sudo is invoked internally where needed.
# Override the defaults if your target differs:
#   BASE_DIR=/opt/assistant DEPLOY_USER=myuser sudo bash setup.sh

set -euo pipefail

# --- Configuration ------------------------------------------------------------
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
BASE_DIR="${BASE_DIR:-/home/ubuntu/personal-assistant}"
APP_BIN="personal-assistant"
LLAMA_SERVER_BIN="/usr/local/bin/llama-server"

QWEN_MODEL="qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/${QWEN_MODEL}"
WHISPER_MODEL="base.en"

# --- Sanity checks -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(id -u)" -eq 0 ]; then
    echo "ERROR: do not run as root. Run as '$DEPLOY_USER' (sudo is invoked internally where needed)." >&2
    exit 1
fi

if [ ! -d "$BASE_DIR" ] || [ "$SCRIPT_DIR" != "$BASE_DIR" ]; then
    echo "ERROR: run setup.sh from the repo checked out at $BASE_DIR (found: $SCRIPT_DIR)." >&2
    echo "  git clone <your-repo-url> $BASE_DIR && cd $BASE_DIR && sudo bash setup.sh" >&2
    exit 1
fi

if [ ! -f main.py ] || [ ! -d systemd ]; then
    echo "ERROR: $SCRIPT_DIR does not look like the personal-assistant repo (missing main.py or systemd/)." >&2
    exit 1
fi

cd "$BASE_DIR"

echo "=== 1. Installing system dependencies ==="
sudo apt update
# Build tools for llama.cpp / whisper.cpp. No Python packages are needed -
# the app ships as a standalone binary (see Dockerfile.builder).
sudo apt install -y cmake build-essential git curl
# Runtime audio libs for the app's microphone capture. The Linux 'sounddevice'
# wheel is pure Python and loads the system PortAudio library at runtime,
# which itself needs ALSA.
sudo apt install -y libportaudio2
# libasound2 was renamed to libasound2t64 on Ubuntu 24.04+.
sudo apt install -y libasound2 2>/dev/null || sudo apt install -y libasound2t64
# The app captures microphone input; give the service user access to audio devices.
sudo usermod -aG audio "$DEPLOY_USER"

echo "=== 2. Downloading Qwen 2.5 3B model ==="
mkdir -p models
if [ ! -f "models/${QWEN_MODEL}" ]; then
    echo "Downloading ${QWEN_MODEL}..."
    # Download to .part and move atomically so an interrupted run never
    # leaves a truncated file that the existence check would treat as done.
    curl -L --fail -o "models/${QWEN_MODEL}.part" "$QWEN_URL"
    mv "models/${QWEN_MODEL}.part" "models/${QWEN_MODEL}"
else
    echo "${QWEN_MODEL} already present, skipping."
fi

echo "=== 3. Building llama.cpp (llama-server) ==="
if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    [ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp.git
    cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build llama.cpp/build --config Release --target llama-server --parallel
    sudo cp llama.cpp/build/bin/llama-server "$LLAMA_SERVER_BIN"
else
    echo "llama-server already installed at $LLAMA_SERVER_BIN, skipping."
fi

echo "=== 4. Building whisper.cpp ==="
[ -d whisper.cpp ] || git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper.cpp/build --config Release --target whisper-server --parallel
( cd whisper.cpp && bash ./models/download-ggml-model.sh "$WHISPER_MODEL" )

echo "=== 5. Checking the standalone app binary ==="
if [ ! -x "$APP_BIN" ]; then
    echo "ERROR: the standalone app binary is missing: $BASE_DIR/$APP_BIN" >&2
    echo "Build it on your dev machine and copy it to the server:" >&2
    echo "  make binary" >&2
    echo "  scp dist/${APP_BIN} ${DEPLOY_USER}@<server>:${BASE_DIR}/" >&2
    exit 1
fi
echo "Found ./${APP_BIN}"

echo "=== 6. Preparing .env ==="
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env from .env.example."
    echo ">>> EDIT .env AND SET HA_TOKEN BEFORE THE APP CAN CONTROL HOME ASSISTANT."
else
    echo ".env already exists, leaving it untouched."
fi

echo "=== 7. Registering systemd services ==="
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
sudo systemctl enable --now whisper-server
sudo systemctl enable --now python-app

echo "=== Deployment complete! ==="
echo "Note: python-app needs a microphone attached to this host (audio group)."
echo "Check status:  systemctl status python-app llama-server whisper-server"
echo "Check logs:    sudo journalctl -u python-app -f   (or: llama-server / whisper-server)"
echo "Restart all:   sudo systemctl restart llama-server whisper-server python-app"
echo "Stop services: sudo systemctl stop python-app"
