#!/usr/bin/env python3
"""Unembedding-lens diagnostic (v7): does Boldt exhibit the paper's "frequent-token lens", and does
EmbedFilter reduce it?

For a small German diagnostic set it: mean-pools the model's last hidden states, decodes them
through the unembedding matrix (top tokens), and compares **before** vs **after** the EmbedFilter
bulk projection — reporting the punctuation/stopword/subword share of the top tokens and an
anisotropy proxy (mean pairwise cosine). DIAGNOSTIC ONLY — never a quality claim.

``torch``/``transformers`` are imported only in the real path; ``--dry-run`` imports no ML.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import unicodedata
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import embed_filter as EF  # noqa: E402

# Small stdlib German stopword / function-word list (sufficient for a lens diagnostic).
GERMAN_STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem", "einen",
    "und", "oder", "aber", "doch", "denn", "sondern", "wie", "als", "wenn", "weil", "dass", "ob",
    "in", "im", "an", "am", "auf", "aus", "bei", "mit", "nach", "seit", "von", "vom", "vor", "zu",
    "zum", "zur", "über", "unter", "durch", "für", "gegen", "ohne", "um", "ist", "sind", "war",
    "waren", "sein", "hat", "haben", "wird", "werden", "ich", "du", "er", "sie", "es", "wir", "ihr",
    "man", "sich", "auch", "noch", "schon", "nur", "so", "dann", "nicht", "kein", "keine", "mehr",
    "sehr", "etwas", "viel", "viele", "am", "fast",
}


def token_category(tok: str, stopwords=GERMAN_STOPWORDS) -> str:
    """Classify a SentencePiece token: punctuation | stopword | subword | content. Stdlib/pure."""
    raw = tok.replace("▁", " ").strip()   # "▁" marks a word start in SentencePiece
    if raw == "":
        return "whitespace"
    if all(unicodedata.category(c).startswith("P") or not c.isalnum() for c in raw):
        return "punctuation"
    if raw.lower() in stopwords:
        return "stopword"
    if not tok.startswith("▁"):           # no word-start marker → continuation fragment
        return "subword"
    return "content"


def noncontent_ratio(tokens: List[str]) -> float:
    """Share of tokens that are punctuation/stopword/subword/whitespace (i.e. not content)."""
    if not tokens:
        return 0.0
    non = sum(1 for t in tokens if token_category(t) != "content")
    return round(non / len(tokens), 4)


def plan(model: str, texts: str, top_k: int, embed_filter: Optional[str],
         out: str) -> Dict[str, Any]:
    return {
        "status": "dry_run", "model": model, "texts": texts, "top_k": top_k,
        "embed_filter": embed_filter, "out": out,
        "diagnostics": ["top_tokens_before_after", "noncontent_ratio_before_after",
                        "anisotropy_before_after"],
        "note": "diagnostic only — not a quality claim",
    }


def _read_texts(path: str) -> List[Dict[str, Any]]:
    rows = []
    for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Boldt/Boldt-DC-350M")
    ap.add_argument("--texts", default=str(ROOT / "data/samples/embedfilter_diagnostics_de.jsonl"))
    ap.add_argument("--embed-filter", default=None, help="EmbedFilter basis dir/.pt")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "outputs/v7-embedfilter/unembedding_lens.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(json.dumps(plan(args.model, args.texts, args.top_k, args.embed_filter, args.out),
                         ensure_ascii=False, indent=2))
        return 0

    # ----------------------------------------------------------------- real path (lazy ML)
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 output_hidden_states=True).to(dev).eval()
    oe = model.get_output_embeddings()
    W = (oe.weight if oe is not None and getattr(oe, "weight", None) is not None
         else model.get_input_embeddings().weight).detach().to(dev, torch.float32)  # [vocab, H]

    rows = _read_texts(args.texts)
    enc = tok([r["text"] for r in rows], padding=True, truncation=True, max_length=128,
              return_tensors="pt").to(dev)
    with torch.no_grad():
        hidden = model(**enc).hidden_states[-1]                      # [n, T, H]
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)     # [n, H]

    basis = None
    if args.embed_filter:
        b, _meta = EF.load_embed_filter_basis(args.embed_filter,
                                              expected_hidden_dim=pooled.shape[1])
        basis = b.to(dev, torch.float32)

    def _top_tokens(vecs: "torch.Tensor") -> List[List[str]]:
        logits = vecs @ W.t()                                        # [n, vocab]
        idx = torch.topk(logits, args.top_k, dim=1).indices.tolist()
        return [tok.convert_ids_to_tokens(row) for row in idx]

    before_tokens = _top_tokens(pooled)
    after_tokens = None
    if basis is not None:
        pooled_bulk = pooled @ basis @ basis.t()                     # project onto bulk subspace
        after_tokens = _top_tokens(pooled_bulk)

    def _mean_pairwise_cos(vecs: "torch.Tensor") -> float:
        x = F.normalize(vecs, dim=1)
        n = x.shape[0]
        sims = x @ x.t()
        return round(float((sims.sum() - n) / max(n * (n - 1), 1)), 4)

    per_text = []
    for i, r in enumerate(rows):
        entry = {"id": r.get("id", i), "top_before": before_tokens[i],
                 "noncontent_before": noncontent_ratio(before_tokens[i])}
        if after_tokens is not None:
            entry["top_after"] = after_tokens[i]
            entry["noncontent_after"] = noncontent_ratio(after_tokens[i])
        per_text.append(entry)

    summary = {
        "model": args.model, "embed_filter": args.embed_filter, "n_texts": len(rows),
        "top_k": args.top_k,
        "mean_noncontent_before": round(
            sum(e["noncontent_before"] for e in per_text) / max(len(per_text), 1), 4),
        "anisotropy_before": _mean_pairwise_cos(pooled),
        "note": "DIAGNOSTIC ONLY — token-lens / anisotropy, not a quality claim",
    }
    if basis is not None:
        summary["mean_noncontent_after"] = round(
            sum(e["noncontent_after"] for e in per_text) / max(len(per_text), 1), 4)
        summary["anisotropy_after"] = _mean_pairwise_cos(pooled @ basis)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_text": per_text}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    md = [f"# Unembedding-lens diagnostic ({args.model})", "",
          "_Diagnostic only — token lens + anisotropy before/after EmbedFilter (not a claim)._",
          "", f"- texts: {len(rows)}  · top-k: {args.top_k}  · embed-filter: {args.embed_filter}",
          f"- mean non-content top-token share — before: {summary['mean_noncontent_before']}"
          + (f", after: {summary['mean_noncontent_after']}" if basis is not None else ""),
          f"- anisotropy (mean pairwise cos) — before: {summary['anisotropy_before']}"
          + (f", after: {summary['anisotropy_after']}" if basis is not None else "")]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[unembedding-lens] {len(rows)} texts -> {out}  "
          f"(noncontent before={summary['mean_noncontent_before']}"
          + (f" after={summary['mean_noncontent_after']}" if basis is not None else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
