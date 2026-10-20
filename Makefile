.PHONY: test run debug help

help:
	@echo "Available commands:"
	@echo "  make run    - Run the Home Assistant voice assistant (INFO level)"
	@echo "  make debug  - Run the voice assistant with DEBUG logging enabled"
	@echo "  make test   - Run the per-function test suite via pytest"

run:
	PYTHONPATH=. python main.py

debug:
	LOG_LEVEL=DEBUG PYTHONPATH=. python main.py

test:
	PYTHONPATH=. pytest -v