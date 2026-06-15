# v5 teacher-validated German RAG question generation

Replaces weak template-only synthetic queries with **teacher-validated, LLM-generated** German RAG
questions for non-FAQ corpora — but a generated question is **provisional** and never training data
until a Qwen3-Reranker teacher score passes threshold.

- Module: `src/boldt_embed/v5_question_generation.py` (pure stdlib core, no API calls)
- CLI: `scripts/generate_v5_rag_questions.py`
- Output: `data/processed/v5/generated_questions.jsonl`

## Why

v2 showed template-generated admin/FAQ/legal queries over Wikipedia mostly **failed** teacher
validation. v3 showed real FAQ works. v5 needs non-FAQ questions, so it generates them with an
LLM — but only keeps them if the teacher reranker confirms the question is actually answered by
the passage. Synthetic questions are **provisional by construction**: every row carries
`synthetic_query=true` and `must_teacher_validate=true`, and `is_training_ready()` returns false
until a `teacher_score >= threshold` is attached downstream.

## Generation modes

| mode | ML? | what it does |
|---|---|---|
| `dry_run_templates` | none | deterministic weak templates — tests/wiring only (still must be teacher-validated) |
| `teacher_prompt_export` | none | writes German JSON-output prompts for an external/local LLM; **no calls** |
| `local_llm_jsonl` | none | consumes pre-generated local-LLM JSONL; joins trusted passage provenance |
| `optional_local_transformers` | lazy | runs a **local** model; only behind `--allow-local-llm`; still no external API |

**No mode ever calls an external API.** Only `optional_local_transformers` imports `transformers`,
and only when `--allow-local-llm` is passed.

## Question styles (10)

`germanquad_fact`, `definition`, `how_to`, `comparison`, `reason_why`, `evidence_support`,
`long_doc_locating`, `ambiguous_needs_context`, `short_web_search`, `german_stress`
(negation / date / compound / entity).

## Prompt contract

Every prompt (German) instructs the model to produce GENAU EINE question that can be answered
**ausschließlich** from the passage (not from general knowledge), in the requested style, and to
return a single JSON object:

```
{"query": "<deine Frage auf Deutsch>", "query_style": "definition", "answerable_only_from_passage": true, "answerable_without_passage": false}
```

`build_prompt(passage, style)` is deterministic, so `teacher_prompt_export` is reproducible.

## Trusted provenance, not LLM-claimed

License and provenance (`license`, `domain`, `source_id`, `document`, `title`) always come from the
**trusted passage record**, never from the LLM. The LLM only supplies the `query`, `query_style`,
and the `answerable_without_passage` flag. Passages are leakage-checked before generation: a
passage flagged `public_benchmark`/`eval_only`, or whose id/url/domain references a public benchmark
(germanquad, dt_test, …), is **rejected** — v5 never generates questions from eval text.

## Reject-if-answerable-without-passage

If the local LLM reports `answerable_without_passage: true`, the row is **rejected** (a question
answerable without the passage is not a real RAG question). Rejections are counted in the report.

## Output row

```
{
  "source_passage_id": "web-0",
  "query": "Welche Frist nennt der Abschnitt?",
  "document": "Abschnitt 0: ...",
  "title": "Thema 0",
  "query_style": "germanquad_fact",
  "generation_method": "local-llama-3.1-8b",
  "synthetic_query": true,
  "license": "CC-BY-4.0",
  "source_id": "web-0",
  "domain": "web_nonfaq",
  "must_teacher_validate": true,
  "answerable_without_passage": false
}
```

## CLI examples

```
# 1) export prompts for an external LLM (no calls, no ML)
python scripts/generate_v5_rag_questions.py --mode teacher_prompt_export \
  --passages data/raw/v5/passages.jsonl --prompts-output outputs/v5-small-rag/question_prompts.jsonl \
  --report outputs/v5-small-rag/question_generation_report.json

# 2) ingest local-LLM outputs (join trusted provenance, reject no-passage-needed)
python scripts/generate_v5_rag_questions.py --mode local_llm_jsonl \
  --passages data/raw/v5/passages.jsonl --llm-output data/raw/v5/llm_out.jsonl \
  --output data/processed/v5/generated_questions.jsonl \
  --report outputs/v5-small-rag/question_generation_report.json

# 3) local transformers (opt-in; lazy ML import; still no external API)
python scripts/generate_v5_rag_questions.py --mode optional_local_transformers \
  --allow-local-llm --model <local-model-id> \
  --passages data/raw/v5/passages.jsonl --output data/processed/v5/generated_questions.jsonl \
  --report outputs/v5-small-rag/question_generation_report.json
```

## Report

`rows_by_query_style`, `rows_by_generation_method`, `rows_by_domain`, `rows_by_license`,
`query_styles_present` / `query_styles_missing`, `rejected_answerable_without_passage`,
`training_ready_rows` (always `0` here — provisional), `all_must_teacher_validate`, and
`examples_per_style`.

## Acceptance

- Synthetic questions are treated as **provisional** (`must_teacher_validate=true`,
  `is_training_ready()` false until teacher-scored).
- Teacher validation is **mandatory** before training — generation never emits training-ready rows.
