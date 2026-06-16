"""Tests for the v6 RAW RAG reranker trainer (stdlib; ML stays lazy). No torch imported here."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402


def _cand(doc_id, label, ts, *, hp=False, uncertain=False, src="bm25", rank=0, soft=None):
    return {"doc_id": doc_id, "text": f"doc {doc_id}", "candidate_source": src,
            "first_stage_rank": rank, "teacher_score": ts, "teacher_softmax_target": soft,
            "label": label, "high_precision_positive": hp, "uncertain": uncertain}


def _present_list(qid, domain="qa_passage_non_eval"):
    # gold present (label 1, high precision) + clear negs + one uncertain
    return {"query_id": qid, "query": f"q {qid}", "positive_doc_ids": [f"{qid}-gold"], "domain": domain,
            "candidates": [
                _cand(f"{qid}-gold", 1, 8.0, hp=True, rank=0),
                _cand(f"{qid}-n1", 0, 1.0, rank=1),
                _cand(f"{qid}-n2", 0, 0.5, rank=2),
                _cand(f"{qid}-u1", None, 5.0, uncertain=True, rank=3)]}


def _absent_list(qid, domain="web"):
    # positive NOT in the candidate set: no label 1, no high precision, gold id absent
    return {"query_id": qid, "query": f"q {qid}", "positive_doc_ids": [f"{qid}-gold-missing"],
            "domain": domain,
            "candidates": [_cand(f"{qid}-a", 0, 1.0, rank=0), _cand(f"{qid}-b", None, 2.0, rank=1)]}


class PositivePresenceTests(unittest.TestCase):
    def test_partition(self):
        present, absent = RM.partition_lists_by_positive_presence(
            [_present_list("q1"), _absent_list("q2"), _present_list("q3")])
        self.assertEqual({r["query_id"] for r in present}, {"q1", "q3"})
        self.assertEqual({r["query_id"] for r in absent}, {"q2"})

    def test_absent_positive_lists_excluded_from_bce_and_pairwise(self):
        rows = [_present_list("q1"), _absent_list("q2")]
        ds = RM.build_v6_reranker_dataset(rows)
        # no candidate from the absent list q2 may appear in pointwise BCE or pairwise
        pt_docs = {e["document"] for e in ds["pointwise"]}
        pw_docs = {e["positive"] for e in ds["pairwise"]} | {e["negative"] for e in ds["pairwise"]}
        self.assertFalse(any(d.startswith("doc q2-") for d in pt_docs), pt_docs)
        self.assertFalse(any(d.startswith("doc q2-") for d in pw_docs), pw_docs)
        self.assertEqual(ds["report"]["lists_positive_absent_excluded"], 1)
        self.assertEqual(ds["report"]["lists_positive_present"], 1)

    def test_uncertain_candidate_is_not_a_bce_negative(self):
        ds = RM.build_v6_reranker_dataset([_present_list("q1")])
        pt = {e["document"]: e["label"] for e in ds["pointwise"]}
        self.assertNotIn("doc q1-u1", pt)           # uncertain (label None) excluded from BCE
        self.assertEqual(pt.get("doc q1-gold"), 1.0)
        self.assertEqual(pt.get("doc q1-n1"), 0.0)

    def test_uncertain_still_in_listwise_target(self):
        ds = RM.build_v6_reranker_dataset([_present_list("q1")])
        lw = ds["listwise"][0]
        self.assertEqual(len(lw["documents"]), 4)   # listwise sees ALL candidates incl uncertain
        self.assertIn("doc q1-u1", lw["documents"])
        self.assertAlmostEqual(sum(lw["target"]), 1.0, places=4)

    def test_pairwise_respects_teacher_margin(self):
        ds = RM.build_v6_reranker_dataset([_present_list("q1")], min_teacher_margin=2.0)
        # gold ts 8.0 vs negs 1.0/0.5 -> margins 7.0/7.5 >= 2.0 -> pairs emitted
        self.assertTrue(ds["pairwise"])
        ds_hi = RM.build_v6_reranker_dataset([_present_list("q1")], min_teacher_margin=100.0)
        self.assertEqual(ds_hi["pairwise"], [])     # no pair clears an impossible margin


class LeakageTests(unittest.TestCase):
    def test_public_eval_leakage_blocked(self):
        leak = _present_list("q9")
        leak["domain"] = "germanquad"
        leak["public_benchmark"] = True
        ds = RM.build_v6_reranker_dataset([_present_list("q1"), leak])
        self.assertTrue(ds["errors"])
        self.assertTrue(any("leakage" in e for e in ds["errors"]))


class RunCardTests(unittest.TestCase):
    def test_run_card_records_exclusion_and_no_policy(self):
        ds = RM.build_v6_reranker_dataset([_present_list("q1"), _absent_list("q2")])
        plan = RM.plan_v5_reranker_loss("listwise_kl+pairwise+pointwise_confident")
        card = RM.v6_raw_reranker_run_card(ds["report"], plan, run_id="t", model_base="b",
                                           output="o", bf16=True, gradient_checkpointing=True)
        self.assertTrue(card["absent_excluded_from_bce_pairwise"])
        self.assertTrue(card["no_serving_policy_in_promotion"])
        self.assertEqual(card["lists_positive_absent_excluded"], 1)
        self.assertEqual(card["primary"], "listwise")
        self.assertIn("RAW reranker lift", card["evaluated_by"])


class DryRunNoTorchTests(unittest.TestCase):
    def test_cli_dry_run_imports_no_torch(self):
        d = pathlib.Path(tempfile.mkdtemp())
        rows = [_present_list(f"q{i}") for i in range(4)] + [_absent_list("qa")]
        cl = d / "lists.jsonl"
        cl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "train_v6_raw_rag_reranker.py"),
               "--candidate-lists", str(cl), "--output", str(d / "ckpt"),
               "--report", str(d / "report.json"), "--dry-run", "--run-id", "v6-rr-test"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry-run-ok", r.stdout)
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["dataset_report"]["lists_positive_present"], 4)
        self.assertEqual(rep["dataset_report"]["lists_positive_absent_excluded"], 1)
        self.assertEqual(rep["run_card"]["primary"], "listwise")


if __name__ == "__main__":
    unittest.main()
