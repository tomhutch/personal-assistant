# Personal Assistant

A lightweight local voice controlled personal assistant that captures microphone input, transcribes via local `whisper-server`, parses intent using an SLM (Qwen 2.5 3B), and controls Home Assistant devices.

---

## Quickstart Guide

### Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Test

```bash
make test
```

### Run

```
make run
```

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
