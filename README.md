# Personal Assistant

A lightweight local voice controlled personal assistant that captures microphone input, transcribes via local `whisper-server`, parses intent using an SLM (Qwen 2.5 3B), and controls Home Assistant devices.

---

## Quickstart Guide

### Prerequisites

```bash
# macOS
brew install cmake
# Ubuntu
sudo apt update && sudo apt install -y cmake build-essential git curl
```

### Install Python 3.12 (required for openwakeword)

macOS (via Homebrew):
```
brew install python@3.12
```

Ubuntu / Debian:
```
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

### Set Up Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
```

### Install openwakeword

On OSX:
```
pip install openwakeword onnxruntime
```

On Ubuntu this requires downloading from source and downloading pretrained model weights:
```
.venv/bin/pip install git+https://github.com/dscripka/openWakeWord.git --no-deps
.venv/bin/python -c "import openwakeword.utils; openwakeword.utils.download_models()"
```

### Start the Llama Server (Qwen 2.5 3B SLM, port 8080)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build llama.cpp/build --config Release --target llama-server --parallel

mkdir -p models
curl -L --fail -o models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf

./llama.cpp/build/bin/llama-server \
  -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --port 8080 --ctx-size 2048 \
  -fa on --cache-prompt --cache-reuse 256 \
  --host 127.0.0.1
```

### Start the Whisper Server (port 8081)

In a second terminal:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper.cpp/build --config Release --target whisper-server --parallel
(cd whisper.cpp && bash ./models/download-ggml-model.sh base.en)

./whisper.cpp/build/bin/whisper-server \
  -m whisper.cpp/models/ggml-base.en.bin \
  --port 8081 --host 127.0.0.1
```

### Test

```bash
make test
```

### Run

With both servers running:

```
make run
```

The app finds the servers via `WHISPER_URL` (default `http://localhost:8081/inference`) and
`SLM_URL` (default `http://localhost:8080/v1/chat/completions`), so no extra configuration
is needed.

---

## Deploying on Ubuntu (bare metal)

The production deployment runs the whole stack under **systemd**: two model servers
(`llama-server` on port 8080, `whisper-server` on port 8081) and the Python app itself,
packaged as a **standalone binary** via PyInstaller — the server needs no Python, pip,
or virtualenv. `setup.sh` automates server-side installation; the `systemd/` directory
holds the three unit files.

### 1. Build the standalone binary (dev machine)

With Docker Desktop running, from the repo root:

```bash
make binary
# or, explicitly:
docker build --platform linux/amd64 -f Dockerfile.builder -o ./dist .
```

This produces `dist/personal-assistant`, a self-contained Linux executable (the
interpreter and all dependencies are embedded). The binary is platform-specific —
build it for the server's architecture. For an ARM64 server (e.g. AWS Graviton):

```bash
make binary PLATFORM=linux/arm64
```

### 2. Deploy to the server

```bash
git clone <your-repo-url> /home/ubuntu/personal-assistant
# copy the built binary into the repo root on the server:
scp dist/personal-assistant ubuntu@<server>:/home/ubuntu/personal-assistant/

# on the server:
cd /home/ubuntu/personal-assistant
sudo bash setup.sh
```

`setup.sh` installs system dependencies (build tools, PortAudio/ALSA runtime libs),
downloads the Qwen 2.5 3B GGUF, builds `llama-server` into `/usr/local/bin` and
`whisper.cpp`, verifies the app binary is present, generates `.env` from
`.env.example`, and registers + starts the three systemd services. It is safe to
re-run — completed steps are skipped.

> After the first run, edit `/home/ubuntu/personal-assistant/.env` and set your
> `HA_TOKEN`, then `sudo systemctl restart python-app`.

### Configuration (`.env`)

The app reads all endpoints from environment variables (see `config.py`), injected by
systemd via `EnvironmentFile=/home/ubuntu/personal-assistant/.env`:

| Variable      | Default                          | Description                              |
| ------------- | -------------------------------- | ---------------------------------------- |
| `WHISPER_URL` | `http://localhost:8081/inference`| Full Whisper inference endpoint          |
| `SLM_URL`     | `http://localhost:8080/v1/chat/completions` | Full Qwen SLM endpoint (the instructions' `LLAMA_URL`) |
| `HA_URL`      | `http://localhost:8123`          | Home Assistant base URL                  |
| `HA_TOKEN`    | *(empty)*                        | Home Assistant long-lived access token   |
| `LOG_LEVEL`   | `INFO`                           | Logging level (the instructions' `DEBUG` flag) |

### Managing the stack

```bash
systemctl status python-app llama-server whisper-server   # status of all three
sudo journalctl -u python-app -f                          # follow logs (swap the unit name)
sudo systemctl restart llama-server whisper-server python-app  # restart everything
sudo systemctl stop python-app                            # stop the app (servers keep running)
```

All three units use `Restart=always` with a 5 s backoff, and `WantedBy=multi-user.target`
so the whole stack comes back automatically on boot. Note that `python-app` needs a
microphone attached to the host, and the `ubuntu` user is added to the `audio` group by
`setup.sh` for device access.
