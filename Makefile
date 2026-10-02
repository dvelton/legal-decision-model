.PHONY: setup generate validate train evaluate serve check scan

setup:
	uv sync

generate:
	uv run legal-decision-model generate

validate:
	uv run legal-decision-model validate

train:
	uv run legal-decision-model train --device mps --epochs 24

evaluate:
	uv run legal-decision-model evaluate

serve:
	uv run legal-decision-model serve

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest

scan:
	uv run legal-decision-model scan-public
