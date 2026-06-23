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

    # stable display order: Boldt rows first, then competitors
    order = ["v6-1-baseline", "v6-best-round7", "v6-best-round7-512",
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
  `query:` / `document:` prefixes. e5/gte/Qwen/LFM run at **512** tokens; Boldt is shown at both
  **256** (its training length) and **512** (fair comparison).
- `v6-best-round7` = the AutoResearch hill-climb winner; `v6-1-baseline` = the pre-tuning v6.1
  model. They are ~tied on MTEB retrieval (the loop optimized WebFAQ recall@100, not these tasks).
- MIRACL = the **hard-negatives** variant (reduced corpus, the standard affordable MTEB setting),
  not the full ~2M-passage corpus. MLDR encodes long docs truncated to the model's seq length.
- These tasks (legal GerDaLIR, Wikipedia MIRACL, long-doc MLDR) are largely **out of Boldt's
  training distribution** (WebFAQ/FAQ-style RAG). The comparison reflects scope, not a defect.
"""
    out = ROOT / "outputs/mteb/COMPARISON_de_retrieval.md"
    out.write_text(doc, encoding="utf-8")
    print(table)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
