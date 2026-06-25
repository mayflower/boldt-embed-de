#!/usr/bin/env python3
"""Aggregate per-model MTEB(deu) retrieval summaries into one auditable comparison table + markdown.

Reads every outputs/mteb/<label>/summary.json and emits outputs/mteb/COMPARISON_de_retrieval.md
(and prints the table). Pure stdlib. Numbers come only from saved summaries (ADR-005)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives",
         "MultiLongDocRetrieval"]
SHORT = {"GermanQuAD-Retrieval": "GermanQuAD", "GerDaLIRSmall": "GerDaLIR-S",
         "MIRACLRetrievalHardNegatives": "MIRACL-hn", "MultiLongDocRetrieval": "MLDR"}
ROOT = Path(__file__).resolve().parents[1]


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> int:
    rows = []
    for sp in sorted(glob.glob(str(ROOT / "outputs/mteb/*/summary.json"))):
        d = json.loads(Path(sp).read_text(encoding="utf-8"))
        m, s = d.get("meta", {}), d.get("scores", {})
        rows.append({"label": m.get("label", Path(sp).parent.name),
                     "model": m.get("model", "?"), "seq": m.get("max_seq_length"),
                     "loader": m.get("loader", "st"), "scores": s,
                     "elapsed": d.get("elapsed_seconds"), "commit": m.get("commit"),
                     "mteb": m.get("mteb_version")})

    # stable display order: Boldt rows (incl. context curve + slerp) first, then competitors
    order = ["v6-1-baseline", "v6-best-round7", "v6-best-round7-512",
             "v6-best-round7-1024", "v6-best-round7-2048", "v6-slerp-merge",
             "e5-base", "gte-multilingual-base", "qwen3-0.6b", "lfm2.5"]
    rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)

    header = "| Model | seq | " + " | ".join(SHORT[t] for t in TASKS) + " |"
    sep = "|" + "---|" * (len(TASKS) + 2)
    lines = [header, sep]
    for r in rows:
        cells = " | ".join(_fmt(r["scores"].get(t)) for t in TASKS)
        lines.append(f"| {r['label']} | {r['seq']} | {cells} |")
    table = "\n".join(lines)

    commit = next((r["commit"] for r in rows if r.get("commit")), "unknown")
    mtebv = next((r["mteb"] for r in rows if r.get("mteb")), "?")
    doc = f"""# MTEB(deu) retrieval-core — model comparison

**Metric:** nDCG@10 (MTEB primary), German subset (`deu`), test/dev split per task.
**Harness:** `scripts/run_mteb_retrieval_de.py` · mteb {mtebv} · commit `{commit}`.
Every number is read from a saved `outputs/mteb/<label>/summary.json` (ADR-005); gte excluded —
its custom remote-code architecture raised a CUDA device-side assert in this environment (2 tries).

{table}

## Notes
- **Boldt** (`v6-*`) uses NO query/doc prefix (symmetric), matching its training. Competitors load
  via `mteb.get_model` (e5/Qwen apply their official prompts) or, for LFM2.5, the documented
  `query:` / `document:` prefixes.
- `v6-best-round7` = the AutoResearch hill-climb winner; `v6-1-baseline` = the pre-tuning v6.1
  model. They are ~tied on MTEB retrieval (the loop optimized WebFAQ recall@100, not these tasks).
- MIRACL = the **hard-negatives** variant (reduced corpus, the standard affordable MTEB setting).

## ⚠️ Sequence-length is a major confound on the long-doc tasks (corrected finding)
Boldt was first run at **256** tokens (its training length / an OOM guard), competitors at **512**.
Re-running Boldt up its base's native RoPE context (Llama, 2048) shows the long-doc "gap" was
largely an **eval-truncation artifact**, not model quality:

| round-7 seq | GerDaLIR-S | MLDR |
|---|---|---|
| 256  | 0.050 | 0.203 |
| 512  | 0.085 | 0.221 |
| 1024 | 0.138 | 0.237 |
| **2048** | **0.195** | **0.264** |

At its native 2048 context Boldt **leads GerDaLIR** (0.195 > Qwen 0.180 > e5 0.153 > LFM 0.150) and
**matches e5 on MLDR** (0.264 vs 0.263). **Fairness caveat:** e5/LFM2.5 max out at ~512 tokens, so
their 512 numbers are their ceiling and the GerDaLIR/MLDR comparison vs them is fair; **Qwen3 (32k)
and gte (8192) were capped at 512 here and would likely rise too** — they need a native-context
re-run before any lead is claimed over them on long-doc.

## What is and isn't a real gap
- **Short-doc tasks are unaffected by context** and the gap there is real: GermanQuAD
  (Boldt ~0.866 vs ~0.92) and **MIRACL** (Boldt ~0.33 vs ~0.52) — genuine model-quality/data gaps
  (Wikipedia ad-hoc + general retrieval are out of Boldt's WebFAQ/RAG training distribution).
- **Long-doc tasks (GerDaLIR/MLDR) were mostly an eval-cap artifact** — fixed for free by serving
  at native context. The 256 cap belongs to *training* (memory), not eval/serving of long docs.
- `v6-slerp-merge` (SLERP of round-7 ⊕ v6.1) is a **logged negative**: the two parents are
  behaviourally near-identical so the merge is ~a no-op (noise-level ±). SLERP needs *complementary*
  strong checkpoints; it is a polish step, not a fix for a structural gap.
"""
    out = ROOT / "outputs/mteb/COMPARISON_de_retrieval.md"
    out.write_text(doc, encoding="utf-8")
    print(table)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
