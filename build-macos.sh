#!/bin/bash
set -e

echo "=== Building Personal Assistant for macOS ==="

# Ensure Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found. Please install Python 3.13."
    exit 1
fi

# Create/activate a temporary virtual environment (recommended for clean builds)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and PyInstaller
echo "Installing dependencies..."
pip install --upgrade pip
pip install --no-cache-dir pyinstaller -r requirements.txt

# Run PyInstaller
echo "Running PyInstaller..."
pyinstaller --onefile --name personal-assistant --paths src \
    --add-data "src/intents/custom_intents.yaml:intents" \
    --add-data "vendor/ohf-voice-intents/sentences/en:vendor/ohf-voice-intents/sentences/en" \
    --add-data "vendor/ohf-voice-intents/rules/en:vendor/ohf-voice-intents/rules/en" \
    --add-data "vendor/ohf-voice-intents/lists/en:vendor/ohf-voice-intents/lists/en" \
    src/main.py

echo "=== Build complete! Binary is located at: ./dist/personal-assistant ==="
