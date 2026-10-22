PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip

.DEFAULT_GOAL := help

.PHONY: help setup data evaluate sweep test lint clean

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Create the virtual environment and install the dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:  ## Regenerate the fictional corpus and the labelled question set
	$(PYTHON) scripts/generate_data.py

evaluate:  ## Run the baseline and the trust layer, write the artefacts
	$(PYTHON) scripts/run_evaluation.py

sweep:  ## Sweep the trust thresholds and report the trade-off
	$(PYTHON) scripts/threshold_sweep.py

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Check formatting and typing of the package
	$(PYTHON) -m compileall -q src scripts tests

clean:  ## Remove caches and generated artefacts
	rm -rf artifacts .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
