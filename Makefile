.PHONY: help validate validate-release smoke bench test report all dry-run-causal dry-run-bi dry-run-reranker clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  validate          structural validation of the repo (stdlib)"
	@echo "  validate-release  2026 release gate: provenance/overclaim/weights checks (stdlib)"
	@echo "  smoke             deterministic CPU smoke tests (stdlib)"
	@echo "  bench             local toy German retrieval benchmark (stdlib)"
	@echo "  test              unittest suite (stdlib)"
	@echo "  report            write validation/smoke/bench reports to outputs/"
	@echo "  all               validate + smoke + bench + test + report"
	@echo "  dry-run-causal    parse causal config + wire inputs (no weights)"
	@echo "  dry-run-bi        parse bidirectional config + wire inputs (no weights)"
	@echo "  dry-run-reranker  parse reranker config + wire inputs (no weights)"

validate:
	$(PY) scripts/validate_repo.py --format markdown

validate-release:
	$(PY) scripts/validate_release_2026.py --format markdown

smoke:
	$(PY) scripts/run_smoke_tests.py --format markdown

bench:
	$(PY) scripts/run_local_benchmark.py --format markdown

test:
	$(PY) -m unittest discover -s tests

report:
	$(PY) scripts/write_reports.py

all: validate smoke bench test report

dry-run-causal:
	$(PY) scripts/train_causal.py --config configs/training_causal.json --dry-run

dry-run-bi:
	$(PY) scripts/train_bidirectional.py --config configs/training_bidirectional.json --dry-run

dry-run-reranker:
	$(PY) scripts/train_reranker.py --config configs/training_reranker.json --dry-run

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
