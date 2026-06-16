"""Tests for the v6 dense RAG embedder trainer (stdlib; ML stays lazy). No torch imported here."""
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


def _pairs(domain, n, q="Frage", d="Dokument"):
    return [{"query": f"{q} {domain} {i}", "document": f"{d} {domain} {i}", "domain": domain,
             "license": "cc-by-4.0"} for i in range(n)]


class FaqCapTests(unittest.TestCase):
    def test_faq_cap_enforced(self):
        ex = _pairs("faq_real", 80) + _pairs("qa_passage_non_eval", 20)
        ordered, rep = TM.domain_balanced_examples(ex, faq_cap=0.5)
        self.assertTrue(rep["faq_capped"])
        self.assertLessEqual(rep["faq_share_after"], 0.5 + 1e-9)
        # with 20 non-faq and cap 0.5, at most 20 faq survive -> 40 total
        self.assertEqual(rep["examples_after"], 40)
        self.assertEqual(len(ordered), 40)

    def test_no_cap_when_already_balanced(self):
        ex = _pairs("faq_real", 10) + _pairs("qa_passage_non_eval", 30)
        _, rep = TM.domain_balanced_examples(ex, faq_cap=0.5)
        self.assertFalse(rep["faq_capped"])
        self.assertEqual(rep["examples_after"], 40)

    def test_batches_are_domain_balanced(self):
        # round-robin: the first len(domains) examples must cover distinct domains
        ex = _pairs("faq_real", 6) + _pairs("qa_passage_non_eval", 6) + _pairs("german_stress", 6)
        ordered, _ = TM.domain_balanced_examples(ex, faq_cap=0.9)
        head_domains = {e["domain"] for e in ordered[:3]}
        self.assertEqual(len(head_domains), 3)


class DatasetTests(unittest.TestCase):
    def test_public_eval_leakage_blocked(self):
        pairs = _pairs("faq_real", 5) + [
            {"query": "q", "document": "d", "domain": "germanquad", "public_benchmark": True}]
        out = TM.build_v6_dense_dataset(pairs, faq_cap=0.5)
        self.assertTrue(out["errors"])
        self.assertTrue(any("leakage" in e for e in out["errors"]))

    def test_hard_negative_margin_parsing(self):
        pairs = _pairs("qa_passage_non_eval", 4)
        hardnegs = [
            {"query": "qn", "positive": "p", "negative": "n1", "teacher_margin": 2.5,
             "negative_score": 1.0, "positive_score": 3.5, "domain": "qa_passage_non_eval"},
            {"query": "qn", "positive": "p", "negative": "n2", "teacher_margin": 6.0,
             "negative_score": -1.0, "positive_score": 5.0, "domain": "qa_passage_non_eval"},
        ]
        out = TM.build_v6_dense_dataset(pairs, hardnegs, faq_cap=0.9)
        hn = out["report"]["hard_negatives"]
        self.assertEqual(hn["triplets"], 2)
        self.assertEqual(hn["queries_with_hardnegs"], 1)
        self.assertAlmostEqual(hn["avg_margin"], (2.5 + 6.0) / 2, places=4)
        self.assertIn(">=5", hn["margin_distribution"])
        self.assertTrue(out["report"]["has_teacher_scores"])

    def test_matryoshka_dims_recorded(self):
        out = TM.build_v6_dense_dataset(_pairs("faq_real", 4), faq_cap=0.9)
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=out["report"]["has_teacher_scores"],
                                           has_distill_vectors=False)
        self.assertEqual(plan["matryoshka_dims"], MATRYOSHKA_DEFAULT)
        self.assertEqual(plan["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        card = TM.v6_dense_run_card(out["report"], plan, run_id="t", model="m", output="o",
                                    max_steps=10, batch_size=8, bf16=True, gradient_checkpointing=True)
        self.assertEqual(card["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        self.assertIn("Recall@50/100", card["target_metric"])
        self.assertEqual(plan["batch_sampler"], "NO_DUPLICATES")

    def test_run_card_records_domain_balance(self):
        out = TM.build_v6_dense_dataset(_pairs("faq_real", 30) + _pairs("german_stress", 10),
                                        faq_cap=0.5)
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=False, has_distill_vectors=False)
        card = TM.v6_dense_run_card(out["report"], plan, run_id="t", model="m", output="o",
                                    max_steps=10, batch_size=8, bf16=True, gradient_checkpointing=True)
        self.assertIsNotNone(card["domain_balance"])
        self.assertTrue(card["domain_balance"]["faq_capped"])


class DryRunNoTorchTests(unittest.TestCase):
    def test_cli_dry_run_imports_no_torch(self):
        # subprocess (per project rule): the full suite imports torch elsewhere, so an in-process
        # sys.modules check is unreliable. The CLI asserts no-torch internally on --dry-run.
        d = pathlib.Path(tempfile.mkdtemp())
        pairs = _pairs("faq_real", 5) + _pairs("qa_passage_non_eval", 5)
        pf = d / "pairs.jsonl"
        pf.write_text("\n".join(json.dumps(r) for r in pairs), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "train_v6_dense_rag_embedder.py"),
               "--train-pairs", str(pf), "--output", str(d / "ckpt"),
               "--report", str(d / "report.json"), "--max-steps", "10", "--dry-run",
               "--run-id", "v6-dense-test"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dry-run-ok", r.stdout)
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["run_card"]["matryoshka_dims"], [1024, 768, 512, 256, 128, 64])
        self.assertEqual(rep["loss_plan"]["batch_sampler"], "NO_DUPLICATES")

    def test_cli_dry_run_fails_closed_on_leakage(self):
        d = pathlib.Path(tempfile.mkdtemp())
        pairs = _pairs("faq_real", 3) + [
            {"query": "q", "document": "d", "domain": "germanquad", "public_benchmark": True}]
        pf = d / "pairs.jsonl"
        pf.write_text("\n".join(json.dumps(r) for r in pairs), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "train_v6_dense_rag_embedder.py"),
               "--train-pairs", str(pf), "--output", str(d / "ckpt"),
               "--report", str(d / "report.json"), "--dry-run", "--run-id", "v6-leak-test"]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("fail closed", r.stderr)


if __name__ == "__main__":
    unittest.main()
