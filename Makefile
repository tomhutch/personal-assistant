.PHONY: test run help

help:
	@echo "Available commands:"
	@echo "  make run    - Run the Home Assistant voice assistant"
	@echo "  make test   - Run the per-function test suite via pytest"

run:
	PYTHONPATH=. python main.py

test:
	PYTHONPATH=. pytest -v