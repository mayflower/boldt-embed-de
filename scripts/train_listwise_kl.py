#!/usr/bin/env python3
"""v8 Stage 2 — listwise-KL distillation FT on top of a broad-pretrained dense student.

Per the 2025 listwise-distillation papers (2505.19274 / 2502.19712): plain contrastive FT degrades
broad retrieval, while teaching the student the cross-encoder's full ranking over a candidate list
improves it. Each cached list (data/processed/v6/reranker_train_lists_teacher_scored.jsonl) has a
query + K candidates with a precomputed `teacher_softmax_target` (Qwen3-Reranker distribution).

Loss = KL(teacher || student) over the top-K candidates (renormalized), where student logits are
cos(q, cand)/tau. A small contrastive term on the labelled positive keeps absolute geometry sane.
Custom loop (sentence-transformers has no listwise-KL); ST tokenize+forward keeps gradients.
Needs the [train] extra + GPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_lists(path, k):
    import math
    out = []
    for ln in Path(path).read_text(encoding="utf-8").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        q = (r.get("query") or "").strip()
        cands = r.get("candidates") or []
        rows = []
        for c in cands:
            t = (c.get("text") or "").strip()
            try:
                ts = float(c.get("teacher_score"))
            except (TypeError, ValueError):
                continue
            if t:
                rows.append((t, ts, str(c.get("label", "0")) == "1"))
        if not q or len(rows) < 2:
            continue
        rows.sort(key=lambda x: x[1], reverse=True)        # by teacher score desc
        rows = rows[:k]
        # renormalize teacher softmax over the kept top-k (temperature already baked via scores)
        mx = max(ts for _, ts, _ in rows)
        exps = [math.exp(ts - mx) for _, ts, _ in rows]
        s = sum(exps) or 1.0
        tgt = [e / s for e in exps]
        pos_idx = next((i for i, (_, _, lab) in enumerate(rows) if lab), 0)
        out.append({"query": q, "cands": [t for t, _, _ in rows], "target": tgt, "pos": pos_idx})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="broad-pretrained student checkpoint dir")
    ap.add_argument("--lists", default="data/processed/v6/reranker_train_lists_teacher_scored.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-queries", type=int, default=4)
    ap.add_argument("--list-k", type=int, default=24)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.05, help="student temperature")
    ap.add_argument("--contrastive-weight", type=float, default=0.1)
    ap.add_argument("--run-id", default="v8-listwise-kl")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import get_cosine_schedule_with_warmup

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lists = _load_lists(str(args.lists) if Path(args.lists).is_absolute()
                        else str(Path(__file__).resolve().parents[1] / args.lists), args.list_k)
    print(f"[lwkl] {len(lists)} teacher lists (top-{args.list_k})", flush=True)
    model = SentenceTransformer(args.base, device=dev)
    model.max_seq_length = args.max_seq_length
    try:  # cut activation memory: B x (1+K) texts encoded with grad is large
        model[0].auto_model.gradient_checkpointing_enable()
    except Exception as exc:
        print(f"[lwkl] grad-checkpointing not enabled: {exc}", flush=True)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * args.steps), args.steps)

    def embed(texts):
        feats = model.tokenize(texts)
        feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in feats.items()}
        emb = model(feats)["sentence_embedding"]
        return F.normalize(emb, p=2, dim=1)

    n = len(lists)
    bs = args.batch_queries
    losses = []
    started = time.time()
    for step in range(args.steps):
        batch = [lists[(step * bs + i) % n] for i in range(bs)]
        opt.zero_grad()
        loss = 0.0
        for ex in batch:
            texts = [ex["query"]] + ex["cands"]
            emb = embed(texts)                              # [1+K, d]
            q, cands = emb[0:1], emb[1:]                     # [1,d], [K,d]
            logits = (q @ cands.T).squeeze(0) / args.tau     # [K]
            logp = F.log_softmax(logits, dim=0)
            target = torch.tensor(ex["target"], device=dev, dtype=logp.dtype)
            kl = -(target * logp).sum()                      # KL(teacher||student) up to teacher entropy
            con = F.cross_entropy(logits.unsqueeze(0),
                                  torch.tensor([ex["pos"]], device=dev))
            loss = loss + kl + args.contrastive_weight * con
        loss = loss / bs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(float(loss.item()))
        if (step + 1) % 100 == 0 or step == 0:
            print(f"[lwkl] step {step+1}/{args.steps} loss={losses[-1]:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model.eval(); model.save(str(out))
    (out / "lwkl_run_card.json").write_text(json.dumps(
        {"run_id": args.run_id, "base": args.base, "lists": args.lists, "n_lists": n,
         "steps": args.steps, "batch_queries": bs, "list_k": args.list_k, "lr": args.lr,
         "tau": args.tau, "contrastive_weight": args.contrastive_weight,
         "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
         "elapsed_seconds": round(time.time() - started, 1)}, indent=2), encoding="utf-8")
    print(f"[lwkl] saved -> {out} (loss {losses[0]:.3f} -> {losses[-1]:.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
