"""Tests for the v6.1 dense top-50 trainer (stdlib; ML stays lazy). No torch imported here."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402
from boldt_embed.train_modern import MATRYOSHKA_DEFAULT  # noqa: E402


def _pairs(n=4, domain="web_nonfaq"):
    return [{"query": f"q{i}", "positive": f"p{i}", "domain": domain, "license": "cc-by-4.0"}
            for i in range(n)]


def _hardneg(qid, pr, n_neg=3, domain="faq_real", margins=None):
    negs = []
    for j in range(n_neg):
        negs.append({"doc_id": f"{qid}-n{j}", "text": f"blocker {qid} {j}",
                     "negative_rank_v6": j + 1, "source": "dense_top50_false_positive",
                     "teacher_score": -5.0, "margin_to_positive": (margins[j] if margins else None)})
    return {"query_id": qid, "query": f"q {qid}", "positive_doc_id": f"{qid}-pos",
            "positive": f"answer {qid}", "positive_rank_v6": pr, "negatives": negs, "domain": domain,
            "source": "webfaq_train"}


class DatasetTests(unittest.TestCase):
    def test_builds_pairs_and_rank_promotion_triplets(self):
        ds = TM.build_v6_1_dense_dataset(_pairs(4), [_hardneg("a", 73), _hardneg("b", 150)])
        self.assertEqual(ds["report"]["pair_examples"], 4)
        self.assertEqual(ds["report"]["rank_promotion_triplets"], 6)   # 2 queries x 3 negs
        self.assertEqual(ds["report"]["positive_rank_51_100"], 1)
        self.assertEqual(ds["report"]["positive_rank_101_200"], 1)
        self.assertEqual(ds["errors"], [])
        # triplets carry the blocker as the explicit negative
        t = ds["triplet_examples"][0]
        self.assertEqual(set(t), {"query", "positive", "negative", "domain", "positive_rank_v6",
                                  "negative_rank_v6", "teacher_score", "margin_to_positive"})

    def test_excludes_in_top50_and_beyond_window(self):
        ds = TM.build_v6_1_dense_dataset([], [_hardneg("a", 40), _hardneg("b", 250)])
        self.assertEqual(ds["report"]["rank_promotion_triplets"], 0)   # 40<=50, 250>200

    def test_no_eval_leakage(self):
        leak = _hardneg("gq1", 73, domain="germanquad"); leak["public_benchmark"] = True
        ds = TM.build_v6_1_dense_dataset(_pairs(2), [_hardneg("a", 73), leak])
        self.assertTrue(any("leakage" in e for e in ds["errors"]))
        # the leaking query produced no triplets
        self.assertTrue(all("gq1" not in t["query"] for t in ds["triplet_examples"]))

    def test_leaking_pair_excluded(self):
        bad = {"query": "x", "positive": "y", "domain": "germanquad", "public_benchmark": True}
        ds = TM.build_v6_1_dense_dataset(_pairs(2) + [bad], [_hardneg("a", 73)])
        self.assertTrue(any("leakage" in e for e in ds["errors"]))
        self.assertEqual(ds["report"]["pair_examples"], 2)             # leaking pair dropped


class LossPlanTests(unittest.TestCase):
    def test_matryoshka_dims_preserved(self):
        plan = TM.plan_v6_1_loss_stack(has_teacher_margins=True)
        self.assertEqual(plan["matryoshka_dims"], MATRYOSHKA_DEFAULT)
        self.assertEqual(plan["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        self.assertEqual(plan["batch_sampler"], "NO_DUPLICATES")

    def test_rank_promotion_in_loss_stack(self):
        plan = TM.plan_v6_1_loss_stack(has_teacher_margins=False)
        self.assertTrue(any("RankPromotion" in c for c in plan["loss_stack"]))
        self.assertTrue(any("CachedMultipleNegativesRankingLoss" == c for c in plan["loss_stack"]))
        self.assertFalse(plan["margin_mse_wired_as_separate_loss"])

    def test_run_card_records_rank_promotion_and_no_reranker(self):
        ds = TM.build_v6_1_dense_dataset(_pairs(3), [_hardneg("a", 73)])
        plan = TM.plan_v6_1_loss_stack(has_teacher_margins=False)
        card = TM.v6_1_dense_run_card(ds["report"], plan, run_id="t", base_model="b", output="o",
                                      max_steps=10, batch_size=8, bf16=True,
                                      gradient_checkpointing=False)
        self.assertEqual(card["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        self.assertFalse(card["reranker_trained"])
        self.assertGreater(card["rank_promotion_triplets"], 0)
        self.assertIn("Recall@50", card["target_metric"])


class DryRunTests(unittest.TestCase):
    def test_cli_dry_run_no_torch(self):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "pairs.jsonl").write_text("\n".join(json.dumps(p) for p in _pairs(5)), encoding="utf-8")
        (d / "hn.jsonl").write_text("\n".join(json.dumps(_hardneg(f"q{i}", 73)) for i in range(3)),
                                    encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "train_v6_1_dense_top50.py"),
               "--train-pairs", str(d / "pairs.jsonl"), "--hard-negatives", str(d / "hn.jsonl"),
               "--output", str(d / "ckpt"), "--report", str(d / "rep.json"),
               "--run-id", "v6-1-dryrun-test", "--dry-run"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry-run-ok", r.stdout)
        rep = json.loads((d / "rep.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["run_card"]["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        self.assertFalse(rep["run_card"]["reranker_trained"])
        self.assertEqual(rep["run_card"]["rank_promotion_triplets"], 9)   # 3 queries x 3 negs

    def test_cli_dry_run_fails_closed_on_leakage(self):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "pairs.jsonl").write_text(json.dumps(
            {"query": "x", "positive": "y", "domain": "germanquad", "public_benchmark": True}),
            encoding="utf-8")
        (d / "hn.jsonl").write_text(json.dumps(_hardneg("a", 73)), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "train_v6_1_dense_top50.py"),
               "--train-pairs", str(d / "pairs.jsonl"), "--hard-negatives", str(d / "hn.jsonl"),
               "--output", str(d / "ckpt"), "--report", str(d / "rep.json"),
               "--run-id", "v6-1-leak-test", "--dry-run"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("fail closed", r.stderr)


if __name__ == "__main__":
    unittest.main()
