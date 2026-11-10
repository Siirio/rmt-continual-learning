.PHONY: install test lint validate examples figures

install:
	python -m pip install -e ".[experiment,reporting,dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts

validate:
	python scripts/validate_dataset.py

examples:
	python scripts/generate_examples.py

figures:
	python scripts/generate_figures.py --input examples/results --output examples/figures

