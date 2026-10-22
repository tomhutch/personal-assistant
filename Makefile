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
	LOG_LEVEL=DEBUG PYTHONPATH=. python main.py

test:
	PYTHONPATH=. python -m pytest -v

# Requires Docker. Output: dist/personal-assistant - copy it to the server's
# repo root before running setup.sh there (see README, "Deploying on Ubuntu").
binary:
	docker build --platform linux/amd64 -f Dockerfile.builder -o ./dist .
