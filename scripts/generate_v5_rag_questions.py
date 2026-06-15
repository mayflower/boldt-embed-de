#!/usr/bin/env python3
"""Generate teacher-validated German RAG questions for non-FAQ corpora (no API calls).

Four modes (see docs/v5-question-generation.md):
  dry_run_templates          deterministic weak templates (tests/wiring; no ML)
  teacher_prompt_export      write German JSON-output prompts for an external LLM (no calls)
  local_llm_jsonl            consume pre-generated local-LLM JSONL, join trusted passage provenance
  optional_local_transformers  ONLY with --allow-local-llm (lazy transformers import)

Every emitted question is PROVISIONAL: synthetic_query=true, must_teacher_validate=true. It is not
training data until a Qwen3-Reranker teacher score passes threshold. License/provenance always come
from the trusted --passages records, never from the LLM.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import v5_question_generation as G  # noqa: E402


def _read_jsonl(path: pathlib.Path) -> list:
    # split("\n"), not splitlines(): web/WebFAQ text carries U+2028/U+2029.
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _write_jsonl(path: pathlib.Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=list(G.GENERATION_METHODS))
    ap.add_argument("--passages", required=True, help="trusted passage records JSONL")
    ap.add_argument("--llm-output", help="local-LLM outputs JSONL (mode local_llm_jsonl)")
    ap.add_argument("--styles", default="", help="comma list; default = all 10 styles")
    ap.add_argument("--output", default="data/processed/v5/generated_questions.jsonl")
    ap.add_argument("--prompts-output", default="outputs/v5-small-rag/question_prompts.jsonl")
    ap.add_argument("--report", default="outputs/v5-small-rag/question_generation_report.json")
    ap.add_argument("--teacher-threshold", type=float, default=4.0)
    ap.add_argument("--allow-local-llm", action="store_true",
                    help="required for mode optional_local_transformers (lazy ML import)")
    ap.add_argument("--model", default="", help="local model id (transformers mode)")
    args = ap.parse_args()

    styles = tuple(s.strip() for s in args.styles.split(",") if s.strip()) or G.QUERY_STYLES
    bad_styles = [s for s in styles if s not in G.QUERY_STYLES]
    if bad_styles:
        print(f"ERROR: unknown --styles {bad_styles}", file=sys.stderr)
        return 2

    ppath = pathlib.Path(args.passages)
    if not ppath.exists():
        print(f"ERROR: --passages not found: {ppath}", file=sys.stderr)
        return 2
    passages = _read_jsonl(ppath)
    perr: list = []
    for i, p in enumerate(passages):
        perr += G.validate_passage(p, i)
    if perr:
        print(f"ERROR: {len(perr)} passage problem(s); first: {perr[0]}", file=sys.stderr)
        return 1

    rows: list = []
    rejected = 0
    errors: list = []

    if args.mode == "teacher_prompt_export":
        prompts = G.export_prompts(passages, styles)
        _write_jsonl(pathlib.Path(args.prompts_output), prompts)
        report = {"mode": args.mode, "prompts_written": len(prompts),
                  "prompts_output": str(args.prompts_output),
                  "query_styles": list(styles), "note": "no API calls; feed to external/local LLM"}
        pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        assert "torch" not in sys.modules, "prompt export must not import torch"
        print(f"[v5-qgen] mode={args.mode} prompts={len(prompts)} -> {args.prompts_output}")
        return 0

    if args.mode == "dry_run_templates":
        rows = G.generate_from_templates(passages, styles)
        assert "torch" not in sys.modules, "dry_run_templates must not import torch"

    elif args.mode == "local_llm_jsonl":
        if not args.llm_output or not pathlib.Path(args.llm_output).exists():
            print("ERROR: --llm-output JSONL required and must exist for local_llm_jsonl",
                  file=sys.stderr)
            return 2
        llm_rows = _read_jsonl(pathlib.Path(args.llm_output))
        by_id = {str(p["source_passage_id"]): p for p in passages}
        rows, rej, errors = G.rows_from_local_llm(llm_rows, by_id)
        rejected = len(rej)
        assert "torch" not in sys.modules, "local_llm_jsonl must not import torch"

    elif args.mode == "optional_local_transformers":
        if not args.allow_local_llm:
            print("ERROR: optional_local_transformers requires --allow-local-llm "
                  "(this mode runs a local model; no external APIs are ever called)",
                  file=sys.stderr)
            return 2
        if not args.model:
            print("ERROR: --model required for optional_local_transformers", file=sys.stderr)
            return 2
        # Lazy ML import — only reached behind the explicit flag; still no external API.
        import re as _re

        from transformers import pipeline  # noqa: E402
        gen = pipeline("text-generation", model=args.model)
        for p in passages:
            for style in styles:
                out = gen(G.build_prompt(p, style), max_new_tokens=128, do_sample=False)
                text = out[0]["generated_text"]
                m = _re.search(r"\{.*\}", text, _re.DOTALL)
                if not m:
                    errors.append(f"{p['source_passage_id']}/{style}: model returned no JSON")
                    continue
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError as exc:
                    errors.append(f"{p['source_passage_id']}/{style}: bad JSON ({exc})")
                    continue
                q = obj.get("query")
                if not isinstance(q, str) or not q.strip():
                    continue
                awp = obj.get("answerable_without_passage")
                row = G.make_row(p, q.strip(), style, "optional_local_transformers",
                                 answerable_without_passage=awp)
                if awp is True:
                    rejected += 1
                else:
                    rows.append(row)

    # Validate every emitted row (schema + provisional contract + license).
    row_errors: list = []
    for i, r in enumerate(rows):
        row_errors += G.validate_generated_row(r, i)
    report = G.summarize(rows, mode=args.mode, rejected=rejected, errors=errors + row_errors)
    report["teacher_threshold"] = args.teacher_threshold
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"[v5-qgen] mode={args.mode} generated={len(rows)} "
          f"rejected_no_passage_needed={rejected} styles={len(report['query_styles_present'])}/10 "
          f"training_ready={report['training_ready_rows']} (all provisional) -> {args.report}")
    if row_errors:
        print(f"FAIL — {len(row_errors)} invalid generated row(s); first: {row_errors[0]}",
              file=sys.stderr)
        return 1

    _write_jsonl(pathlib.Path(args.output), rows)
    print(f"[v5-qgen] wrote {len(rows)} PROVISIONAL questions -> {args.output} "
          f"(not training data until Qwen3-Reranker teacher score >= {args.teacher_threshold})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
