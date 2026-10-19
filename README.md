# Susan

A lightweight local voice controlled personal assistant that captures microphone input, transcribes via local `whisper-server`, parses intent using an SLM (Qwen 2.5 3B), and controls Home Assistant devices.

---

## Quickstart Guide

### 1. Clone & Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
