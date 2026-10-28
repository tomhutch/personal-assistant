# Load .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

.PHONY: test run debug binary help

help:
	@echo "Available commands:"
	@echo "  make run     - Run the Home Assistant voice assistant (INFO level)"
	@echo "  make debug   - Run the voice assistant with DEBUG logging enabled"
	@echo "  make test    - Run the per-function test suite via pytest"
	@echo "  make binary  - Build the standalone Linux binary via Docker (dist/)"

run:
	PYTHONPATH=. python main.py

debug:
	LOG_LEVEL=DEBUG PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -u main.py

test:
	PYTHONPATH=. python -m pytest -v

# Ensure the local bin directory exists
bin:
	mkdir -p bin

# Requires Docker. Output: dist/personal-assistant - copy it to the server's
# repo root before running setup.sh there (see README, "Deploying on Ubuntu").
# The binary is platform-specific; for an ARM64 server use:
#   make binary PLATFORM=linux/arm64
PLATFORM ?= linux/amd64

binary:
	docker build --platform $(PLATFORM) -f Dockerfile.builder -o ./bin .

# Build, copy to service path, and restart systemd
deploy: binary
	chmod +x ./dist/personal-assistant
	bash setup.sh
	systemctl --user restart personal-assistant
