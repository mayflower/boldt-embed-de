# Synthetic German query generation

Generate diverse German query→passage training pairs from German passages, then let the
teacher cache score and filter them. Template-based and **deterministic** today (no external
APIs, no network); a local-LLM path is stubbed for later.

## Query styles

`src/boldt_embed/synthetic_queries.py` emits up to nine query styles per passage, each
tagged with `metadata.query_style` / `template_id` / `generation_method`:

| style | example (German) |
|---|---|
| `factual` | „Was versteht man unter **Nettokaltmiete**?“ |
| `keyword` | „**Nettokaltmiete Mietkaution**“ |
| `admin` (formal) | „Welche Voraussetzungen gelten für **Mietkaution**?“ |
| `colloquial` | „Wie funktioniert das mit der **Mietkaution** eigentlich?“ |
| `legal` | „Was regelt **§ 551**?“ |
| `negation` | „Wann gilt **Mietkaution** nicht?“ |
| `date_number` | „Was geschah im Jahr **2025** im Zusammenhang mit der **Grundsteuer**?“ |
| `entity` | „Bezieht sich **München** hier auf die Stadt oder das Bundesland?“ |
| `faq` | „Häufige Frage: Wie kann ich **Mietkaution** beantragen?“ |

Slots (topic, keyword, number, year, legal reference, entity) are extracted deterministically
from the passage; a template that lacks its required slot is simply skipped.

## Traceability & license

Every generated row is a standard candidate (`positive=True`) and carries:

- `metadata.source_passage_id` — the originating passage id,
- `metadata.source_domain` — the passage's domain,
- `metadata.generation_method` / `metadata.template_id` / `metadata.query_style`,
- `license` — **inherited verbatim** from the source passage (no relicensing).

## Run it

```bash
python scripts/generate_synthetic_queries.py \
  --passages data/processed/passages.jsonl \
  --output data/processed/synthetic_candidates.jsonl \
  --queries-per-passage 4 --domains factual admin legal date_number

# inspect without writing
python scripts/generate_synthetic_queries.py --passages tests/fixtures/passages.jsonl --dry-run
```

## End-to-end: generate → score → filter → train

Synthetic queries are *candidates*, not gold. The teacher decides which survive:

```bash
# 1. generate synthetic query candidates from passages
python scripts/generate_synthetic_queries.py \
  --passages data/processed/passages.jsonl \
  --output data/processed/synthetic_candidates.jsonl --queries-per-passage 4

# 2. score them with the Qwen3 teachers (GPU)
python scripts/build_teacher_cache.py \
  --input data/processed/synthetic_candidates.jsonl \
  --output outputs/teacher-cache/synthetic_scores.jsonl --mode both

# 3. drop low-teacher-score pairs (a generated query the teacher cannot match to its
#    passage is a bad query) — handled by the negative-miner / trainer filters

# 4. train the student on the surviving high-quality pairs
```

A generated query whose passage the embedding/reranker teacher scores *low* is a poor query
(ambiguous, off-topic, or ungrammatical) and is filtered out before training. This keeps
template noise from polluting the student.

## Local-LLM upgrade path (not yet implemented)

`src/boldt_embed/local_llm_generation.py` defines the `LocalLLMGenerator` interface for
generating queries with a **local** German instruction model (e.g. via vLLM/transformers — no
remote API). It currently raises `NotImplementedError`; the template generator above is the
supported path. When implemented, downstream scoring/filtering is unchanged.
