"""Stdlib tests for v5 dense RAG embedder training (plan/dataset/leakage). No ML, no network."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import train_modern as TM  # noqa: E402

V5_CFG = ROOT / "configs" / "experiments" / "v5_small_rag.json"
MATRYOSHKA = [1024, 768, 512, 256, 128, 64]


def pair(q, p, *, domain="web_nonfaq", license="CC-BY-4.0", **kw):
    r = {"query": q, "positive": p, "domain": domain, "license": license,
         "synthetic_query": False, "source_id": f"src-{q[:6]}"}
    r.update(kw)
    return r


def triplet(q, p, n, margin, pos=8.0, neg=1.0, domain="webfaq2"):
    return {"query": q, "positive": p, "negative": n, "teacher_margin": margin,
            "positive_score": pos, "negative_score": neg, "domain": domain, "license": "CC-BY-4.0"}


class TestLossPlan(unittest.TestCase):
    def test_full_stack(self):
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=True, has_distill_vectors=True)
        joined = " | ".join(plan["loss_stack"])
        self.assertIn("CachedMultipleNegativesRankingLoss", joined)
        self.assertIn("MatryoshkaLoss", joined)
        self.assertIn("MarginMSELoss", joined)
        self.assertIn("EmbedDistillLoss", joined)
        self.assertEqual(plan["batch_sampler"], "NO_DUPLICATES")

    def test_matryoshka_dims(self):
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=False, has_distill_vectors=False)
        self.assertEqual(plan["matryoshka_dims"], MATRYOSHKA)

    def test_optional_losses_off_without_signals(self):
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=False, has_distill_vectors=False)
        self.assertEqual(plan["margin_mse"], [])
        self.assertEqual(plan["embed_distill"], [])
        # contrastive + matryoshka still present
        self.assertEqual(len(plan["loss_stack"]), 2)


class TestDataset(unittest.TestCase):
    def test_real_pairs_build(self):
        ds = TM.build_v5_dense_dataset([pair("Wie hoch?", "Hoechstens drei Mieten.")])
        self.assertEqual(ds["errors"], [])
        self.assertEqual(ds["report"]["examples"], 1)
        self.assertEqual(ds["report"]["teacher_validation"]["real_pairs"], 1)

    def test_provisional_pair_excluded_until_teacher_validated(self):
        provisional = pair("Frage?", "Antwort.", synthetic_query=True, must_teacher_validate=True)
        ds = TM.build_v5_dense_dataset([provisional])
        self.assertEqual(ds["report"]["examples"], 0)
        self.assertEqual(ds["report"]["teacher_validation"]["provisional_excluded"], 1)
        # same pair WITH a passing teacher score is kept
        ok = dict(provisional); ok["teacher_score"] = 4.5
        ds2 = TM.build_v5_dense_dataset([ok], teacher_threshold=4.0)
        self.assertEqual(ds2["report"]["examples"], 1)
        self.assertEqual(ds2["report"]["teacher_validation"]["validated_synthetic"], 1)

    def test_hard_negative_margin_labels(self):
        hn = [triplet("Q?", "Positiv.", "Negativ A.", 7.0),
              triplet("Q?", "Positiv.", "Negativ B.", 2.5)]
        ds = TM.build_v5_dense_dataset([], hardnegs=hn)
        rep = ds["report"]["hard_negatives"]
        self.assertEqual(rep["triplets"], 2)
        self.assertEqual(rep["queries_with_hardnegs"], 1)             # grouped by (query, positive)
        self.assertAlmostEqual(rep["avg_margin"], 4.75, 3)
        self.assertIn(">=5", rep["margin_distribution"])     # margin 7.0 caps into the >=5 bucket
        self.assertIn("2-3", rep["margin_distribution"])     # margin 2.5
        self.assertTrue(ds["report"]["has_teacher_scores"])
        # the grouped example carries both negatives
        ex = [e for e in ds["examples"] if e["query"] == "Q?"][0]
        self.assertEqual(len(ex["negatives"]), 2)

    def test_no_public_eval_train_leakage(self):
        leak_flag = TM.build_v5_dense_dataset([pair("q", "p", public_benchmark=True)])
        self.assertTrue(leak_flag["errors"])
        leak_token = TM.build_v5_dense_dataset(
            [pair("q", "p", source_id="germanquad-test-3")])
        self.assertTrue(any("leakage" in e for e in leak_token["errors"]))

    def test_distill_vectors_flag_from_teacher_scores(self):
        ts = [{"query": "q", "document": "p", "teacher_score": 5.0, "teacher_vector": [0.1, 0.2]}]
        ds = TM.build_v5_dense_dataset([pair("q", "p")], teacher_scores=ts)
        self.assertTrue(ds["report"]["has_distill_vectors"])
        self.assertTrue(ds["report"]["has_teacher_scores"])


class TestRunCard(unittest.TestCase):
    def test_run_card_fields(self):
        ds = TM.build_v5_dense_dataset([pair("q", "p")], hardnegs=[triplet("q2", "p2", "n", 3.0)])
        plan = TM.plan_v5_dense_loss_stack(has_teacher_scores=True, has_distill_vectors=False)
        card = TM.v5_dense_run_card(ds["report"], plan, run_id="v5-dense-boldt",
                                    model="boldt-v3", output="out", max_steps=2000,
                                    bf16=True, gradient_checkpointing=True)
        self.assertEqual(card["run_id"], "v5-dense-boldt")
        self.assertEqual(card["matryoshka_dims"], MATRYOSHKA)
        self.assertIn("domain_mix", card)
        self.assertIn("hard_negative_margins", card)
        self.assertIn("teacher_validation", card)


class TestDryRunNoMl(unittest.TestCase):
    def test_cli_dry_run_no_ml(self):
        with tempfile.TemporaryDirectory() as d:
            pairs = pathlib.Path(d) / "pairs.jsonl"
            pairs.write_text(json.dumps(pair("Wie hoch?", "Hoechstens drei Mieten.")) + "\n",
                             encoding="utf-8")
            report = pathlib.Path(d) / "card.json"
            code = (
                "import sys; sys.path.insert(0, %r); "
                "sys.argv=['x','--config', %r,'--train-pairs', %r,'--report', %r,'--run-id','t',"
                "'--dry-run']\n"
                "import runpy; rc=0\n"
                "try:\n runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n rc=e.code or 0\n"
                "assert 'torch' not in sys.modules, 'torch imported'\n"
                "print('RC', rc)\n"
                % (str(ROOT / "src"), str(V5_CFG), str(pairs), str(report),
                   str(ROOT / "scripts" / "train_v5_dense_rag_embedder.py"))
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("RC 0", r.stdout)
            self.assertTrue(report.exists())
            card = json.loads(report.read_text("utf-8"))["run_card"]
            self.assertEqual(card["matryoshka_dims"], MATRYOSHKA)
            self.assertEqual(card["batch_sampler"], "NO_DUPLICATES")


if __name__ == "__main__":
    unittest.main()
