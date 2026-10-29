# --- Stage 1: build the executable ---
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the app's pinned dependencies and PyInstaller
COPY requirements.txt .
RUN pip install --no-cache-dir pyinstaller -r requirements.txt

# Install openWakeWord from GitHub with --no-deps to bypass the legacy tflite-runtime check
RUN pip install --no-cache-dir --no-deps "git+https://github.com/dscripka/openWakeWord.git@v0.6.0"

# Pre-download openWakeWord model weights inside the container
RUN python -c "import openwakeword.utils; openwakeword.utils.download_models()"

# Copy the application source
COPY . .

# Bundle into a single executable, collecting all assets
RUN pyinstaller --onefile --name personal-assistant \
    --collect-all openwakeword --collect-all piper main.py

# --- Stage 2: export artifact ---
FROM scratch AS export
COPY --from=builder /app/dist/personal-assistant /personal-assistant