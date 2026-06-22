.PHONY: help validate validate-release smoke bench test report all dry-run-causal dry-run-bi dry-run-reranker autoresearch-smoke clean

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
	@echo "  autoresearch-smoke  dense AutoResearch dry-run trial + score + log (stdlib)"
	@echo "  autoresearch-loop   one end-to-end AutoResearch iteration via the CLI orchestrator"

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

# Non-invasive: deliberately NOT part of `make all`.
autoresearch-smoke:
	$(PY) scripts/ar_run_trial.py --config configs/autoresearch/experiments/current.json --budget-minutes 20 --out outputs/autoresearch/runs/make-smoke --dry-run
	$(PY) scripts/ar_score.py --run outputs/autoresearch/runs/make-smoke/metrics.json --baseline outputs/autoresearch/runs/make-smoke/metrics.json --out outputs/autoresearch/runs/make-smoke/score.json
	$(PY) scripts/ar_log_result.py --run outputs/autoresearch/runs/make-smoke --status discard --notes "make smoke"

# One end-to-end AutoResearch iteration via the CLI orchestrator (dry-run plumbing).
autoresearch-loop:
	$(PY) scripts/ar_loop.py --dry-run --status discard --notes "make autoresearch-loop"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
