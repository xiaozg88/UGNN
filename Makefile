.PHONY: install lint format run ablation comparison clean check

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

check: lint format-check

run:
	python -m ugnn.main

ablation:
	python ablation/run_ablation.py

comparison:
	python baselines/run_comparison.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/ 2>/dev/null || true
