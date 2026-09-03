CONDA_ENV ?= py312
CONDA_RUN := conda run -n $(CONDA_ENV)

.PHONY: setup conda-setup run pipeline demo headless test lint format clean help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup          - Alias for conda-setup (recommended)"
	@echo "  conda-setup    - Create and configure conda environment"
	@echo "  run            - Launch Webots + the pipeline (the whole demo, one command)"
	@echo "  pipeline       - Run the pipeline against a Webots you opened yourself"
	@echo "  demo           - Run camera-only demo (no Webots)"
	@echo "  headless       - Run headless for 100 frames (no camera window)"
	@echo "  test           - Run pytest"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with black and ruff"
	@echo "  clean          - Remove caches (leaves logs/ alone)"
	@echo ""
	@echo "Conda environment: $(CONDA_ENV)"
	@echo "Activate with: conda activate $(CONDA_ENV)"

setup: conda-setup

conda-setup:
	bash scripts/setup_conda.sh

# One command brings up both halves (PRD FR-9 / acceptance criterion 1).
# Attaches to an already-running Webots rather than opening a second instance.
run:
	$(CONDA_RUN) python run.py --launch-webots

demo:
	$(CONDA_RUN) python run.py --no-webots

headless:
	$(CONDA_RUN) python run.py --no-webots --no-display --max-frames 100

# Pipeline only, against a Webots you opened yourself (docs/RUN_INSTRUCTIONS.md
# Option B).
pipeline:
	$(CONDA_RUN) python run.py --no-launch-webots

test:
	$(CONDA_RUN) pytest

# main/ is included on purpose: it holds every line of robot maths and the
# controller itself. Linting only src/ and tests/ meant the code that actually
# drives the NAO was never checked.
LINT_PATHS := src tests main run.py

lint:
	$(CONDA_RUN) ruff check $(LINT_PATHS)

format:
	$(CONDA_RUN) black $(LINT_PATHS) && $(CONDA_RUN) ruff check --fix $(LINT_PATHS)

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
