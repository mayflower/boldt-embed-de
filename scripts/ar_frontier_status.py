#!/usr/bin/env python3
"""AutoResearch frontier-program state — the view the autonomous agent reads each round.

Scans every saved ``outputs/mteb/<label>/summary.json`` (ADR-005: a metric is only what was saved)
and reports, for the MTEB(deu) retrieval-core, the current state of the merge+train+distill program
so the agent can pick its NEXT MOVE deterministically:

  - per-candidate 4-task nDCG@10 + the aggregate (mean over the 4 tasks),
  - the same-size-peer FRONTIER (max over e5-base / lfm2.5) per task + aggregate, and the gap,
  - the ranking of OUR candidates by aggregate (the current frontier-best),
  - the per-task LEADERS among our candidates — i.e. which checkpoints are COMPLEMENTARY and thus
    worth MERGING (a wiki/MIRACL leader + a legal/GerDaLIR leader, etc.),
  - which candidates already beat the peer aggregate (promotable territory; still needs the gate).

Pure stdlib, read-only — never trains, never claims beyond the saved summaries. The autonomous
``/ar-frontier`` loop calls this between rounds; ``check_mteb_frontier_gate.py`` is the actual gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MTEB = ROOT / "outputs" / "mteb"
TASKS = ["GermanQuAD-Retrieval", "GerDaLIRSmall", "MIRACLRetrievalHardNegatives",
         "MultiLongDocRetrieval"]
SHORT = {"GermanQuAD-Retrieval": "GermanQuAD", "GerDaLIRSmall": "GerDaLIR",
         "MIRACLRetrievalHardNegatives": "MIRACL", "MultiLongDocRetrieval": "MLDR"}
# same-size peers are the gate; the larger models are stretch references only
DEFAULT_PEERS = ["e5-base", "lfm2.5"]
STRETCH = ["qwen3-0.6b", "gte-multilingual-base"]


def _scores(label: str) -> dict:
    p = MTEB / label / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("scores", {})


def _agg(s: dict):
    vals = [s[t] for t in TASKS if isinstance(s.get(t), (int, float))]
    return sum(vals) / len(vals) if vals else None


def _all_labels() -> list:
    if not MTEB.exists():
        return []
    return sorted(d.name for d in MTEB.iterdir()
                  if d.is_dir() and (d / "summary.json").exists())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peers", default=",".join(DEFAULT_PEERS),
                    help="same-size-peer MTEB labels that define the frontier to beat")
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = ap.parse_args(argv)

    peers = [p.strip() for p in args.peers.split(",") if p.strip()]
    peer_scores = {p: _scores(p) for p in peers}
    peer_front = {t: max((s[t] for s in peer_scores.values() if isinstance(s.get(t), (int, float))),
                         default=None) for t in TASKS}
    peer_agg = _agg(peer_front)

    known = set(peers) | set(STRETCH)
    candidates = []
    for label in _all_labels():
        if label in known:
            continue
        s = _scores(label)
        if not any(isinstance(s.get(t), (int, float)) for t in TASKS):
            continue
        candidates.append({"label": label, "scores": s, "aggregate": _agg(s)})
    candidates.sort(key=lambda c: (c["aggregate"] is not None, c["aggregate"] or 0), reverse=True)

    # per-task leaders among OUR candidates -> complementary merge inputs
    leaders = {}
    for t in TASKS:
        best, bestlbl = None, None
        for c in candidates:
            v = c["scores"].get(t)
            if isinstance(v, (int, float)) and (best is None or v > best):
                best, bestlbl = v, c["label"]
        leaders[SHORT[t]] = {"label": bestlbl, "score": best,
                             "peer_best": peer_front.get(t),
                             "gap_to_peer": (None if best is None or peer_front.get(t) is None
                                             else round(best - peer_front[t], 4))}
    distinct = sorted({v["label"] for v in leaders.values() if v["label"]})

    best = candidates[0] if candidates else None
    beats = [c["label"] for c in candidates
             if c["aggregate"] is not None and peer_agg is not None and c["aggregate"] >= peer_agg]

    verdict = {
        "peers": peers,
        "peer_frontier": {SHORT[t]: peer_front[t] for t in TASKS},
        "peer_aggregate": peer_agg,
        "n_candidates": len(candidates),
        "frontier_best": None if not best else {"label": best["label"], "aggregate": best["aggregate"],
                                                "gap_to_peer_aggregate": (None if best["aggregate"] is None or peer_agg is None
                                                                          else round(best["aggregate"] - peer_agg, 4))},
        "per_task_leaders": leaders,
        "complementary_merge_inputs": distinct,
        "candidates_beating_peer_aggregate": beats,
        "candidates": [{"label": c["label"], "aggregate": c["aggregate"],
                        **{SHORT[t]: c["scores"].get(t) for t in TASKS}} for c in candidates],
        "note": "MTEB(deu) retrieval-core, saved summaries only. Merging is most promising when the "
                "per-task leaders are DIFFERENT checkpoints (complementary) sharing a warm-start basin.",
    }

    if args.format == "json":
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        return 0

    def f(x):
        return "  .  " if not isinstance(x, (int, float)) else f"{x:.4f}"
    print("# AutoResearch frontier status — MTEB(deu) retrieval-core (nDCG@10)\n")
    print(f"Same-size peers: {', '.join(peers)}")
    print("\n| | " + " | ".join(SHORT[t] for t in TASKS) + " | **agg** |")
    print("|---|" + "---|" * (len(TASKS) + 1))
    print("| **peer frontier** | " + " | ".join(f(peer_front[t]) for t in TASKS)
          + f" | **{f(peer_agg)}** |")
    for c in candidates:
        star = " 🏆" if c["label"] in beats else ""
        print(f"| {c['label']}{star} | " + " | ".join(f(c['scores'].get(t)) for t in TASKS)
              + f" | {f(c['aggregate'])} |")
    print("\n**Per-task leaders (our candidates) — complementary merge inputs:**")
    for t in TASKS:
        L = leaders[SHORT[t]]
        print(f"- {SHORT[t]}: `{L['label']}` {f(L['score'])} "
              f"(peer {f(L['peer_best'])}, gap {L['gap_to_peer']})")
    if best:
        ga = verdict["frontier_best"]["gap_to_peer_aggregate"]
        print(f"\nFrontier-best: `{best['label']}` agg {f(best['aggregate'])} "
              f"(gap to peer aggregate {ga}). "
              + ("BEATS peer aggregate." if best['label'] in beats else "below peer aggregate."))
    if len(distinct) >= 2:
        print(f"\n→ Complementary checkpoints to MERGE: {', '.join('`'+d+'`' for d in distinct)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
