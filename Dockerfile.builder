# Multi-stage builder for the Personal Assistant standalone binary.
#
# The target Ubuntu server needs no Python, pip, or venv - PyInstaller bundles
# the interpreter and all app dependencies into a single executable.
#
# Build from the repo root (Docker Desktop running):
#   make binary
#   # or, explicitly:
#   docker build --platform linux/amd64 -f Dockerfile.builder -o ./dist .
#
# For an ARM64 server (e.g. AWS Graviton), use --platform linux/arm64 instead.
# Output: ./dist/personal-assistant

# --- Stage 1: build the executable ---
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the app's pinned dependencies and PyInstaller
COPY requirements.txt .
RUN pip install --no-cache-dir pyinstaller -r requirements.txt

# Copy the application source (build artifacts and secrets excluded by .dockerignore)
COPY . .

# Bundle into a single self-contained executable.
# Note: on Linux, 'sounddevice' resolves to the pure-Python wheel, which loads
# the system libportaudio at runtime - the server must have libportaudio2 and
# ALSA installed (setup.sh handles this).
RUN pyinstaller --onefile --name personal-assistant main.py

# --- Stage 2: export artifact ---
FROM scratch AS export
COPY --from=builder /app/dist/personal-assistant /personal-assistant
