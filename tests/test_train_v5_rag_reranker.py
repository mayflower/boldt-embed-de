"""Stdlib tests for v5 RAG reranker training data + loss plan (anti-FAQ-overfit). No ML, no net."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker_modern as RM  # noqa: E402

V5_CFG = ROOT / "configs" / "experiments" / "v5_small_rag.json"


def slist(qid, domain, *, gold_score=8.0, neg_score=1.0, uncertain_score=3.0):
    return {"query_id": qid, "query": f"q-{qid}", "domain": domain, "candidates": [
        {"doc_id": f"{qid}-p", "text": f"pos {qid}", "teacher_score": gold_score, "label": 1,
         "high_precision_positive": True},
        {"doc_id": f"{qid}-n", "text": f"neg {qid}", "teacher_score": neg_score, "label": 0},
        {"doc_id": f"{qid}-u", "text": f"unc {qid}", "teacher_score": uncertain_score,
         "label": None, "uncertain": True},
    ]}


class TestFaqShareCap(unittest.TestCase):
    def test_faq_heavy_capped(self):
        rows = [slist(f"f{i}", "faq_real") for i in range(80)] + \
               [slist(f"w{i}", "web_nonfaq") for i in range(20)]
        cap = RM.cap_faq_share(rows, max_faq_share=0.35)
        self.assertEqual(cap["status"], "pass")
        self.assertAlmostEqual(cap["faq_share_before"], 0.8)
        self.assertLessEqual(cap["faq_share_after"], 0.35)
        # all non-FAQ kept
        self.assertEqual(sum(1 for r in cap["kept"] if r["domain"] != "faq_real"), 20)
        self.assertGreater(cap["faq_dropped_for_cap"], 0)

    def test_already_balanced_unchanged(self):
        rows = [slist(f"f{i}", "faq_real") for i in range(10)] + \
               [slist(f"w{i}", "web_nonfaq") for i in range(30)]
        cap = RM.cap_faq_share(rows, max_faq_share=0.35)
        self.assertEqual(cap["faq_dropped_for_cap"], 0)
        self.assertEqual(cap["n_after"], 40)

    def test_faq_only_fails_closed(self):
        rows = [slist(f"f{i}", "faq_real") for i in range(10)]
        cap = RM.cap_faq_share(rows, max_faq_share=0.35)
        self.assertEqual(cap["status"], "fail")
        self.assertEqual(cap["n_after"], 0)

    def test_cap_is_deterministic(self):
        rows = [slist(f"f{i}", "faq_real") for i in range(50)] + \
               [slist(f"w{i}", "web_nonfaq") for i in range(10)]
        a = RM.cap_faq_share(rows, 0.35)["kept"]
        b = RM.cap_faq_share(list(reversed(rows)), 0.35)["kept"]
        self.assertEqual(sorted(r["query_id"] for r in a), sorted(r["query_id"] for r in b))


class TestSupervision(unittest.TestCase):
    def test_uncertain_never_bce_label(self):
        pw = RM.scored_lists_to_pointwise_high_confidence([slist("q1", "web_nonfaq")])
        texts = {p["document"] for p in pw}
        self.assertEqual(len(pw), 2)                 # gold + hard-neg only
        self.assertNotIn("unc q1", texts)            # uncertain excluded from BCE
        self.assertEqual({p["label"] for p in pw}, {1.0, 0.0})

    def test_listwise_target_sums_to_one(self):
        lw = RM.scored_lists_to_listwise([slist("q1", "web_nonfaq")])
        self.assertEqual(len(lw), 1)
        self.assertAlmostEqual(sum(lw[0]["target"]), 1.0, 6)
        self.assertEqual(len(lw[0]["target"]), 3)    # listwise over the FULL candidate set

    def test_domain_balanced_sampler(self):
        rows = ([slist(f"f{i}", "faq_real") for i in range(10)] +
                [slist(f"w{i}", "web_nonfaq") for i in range(3)] +
                [slist(f"q{i}", "qa_passage_non_eval") for i in range(5)])
        balanced = RM.domain_balanced_list_sampler(rows, max_per_domain=3)
        by_dom = {}
        for r in balanced:
            by_dom[r["domain"]] = by_dom.get(r["domain"], 0) + 1
        self.assertTrue(all(v <= 3 for v in by_dom.values()))


class TestLossPlan(unittest.TestCase):
    def test_full_spec(self):
        plan = RM.plan_v5_reranker_loss("listwise_kl+pairwise+pointwise_confident")
        joined = " | ".join(plan["components"])
        self.assertIn("KLDivLoss(listwise)", joined)
        self.assertIn("MarginRankingLoss", joined)
        self.assertIn("BCEWithLogitsLoss(high_confidence_only)", joined)
        self.assertEqual(plan["primary"], "listwise")
        self.assertTrue(plan["uncertain_listwise_only"])
        self.assertTrue(plan["pointwise_bce_high_confidence_only"])

    def test_listwise_is_always_primary(self):
        plan = RM.plan_v5_reranker_loss("pairwise")     # listwise omitted from spec
        self.assertEqual(plan["weights"].get("listwise"), 1.0)
        self.assertIn("KLDivLoss(listwise)", plan["components"])

    def test_optional_listwise_variants_and_ranknet(self):
        plan = RM.plan_v5_reranker_loss("lambdaloss+listnet+ranknet")
        self.assertIn("LambdaLoss", plan["optional_listwise_variants_if_available"])
        self.assertIn("ListNet", plan["optional_listwise_variants_if_available"])
        self.assertEqual(plan["pairwise"], "RankNet")


class TestReport(unittest.TestCase):
    def test_shares_separation_uncertain(self):
        rows = ([slist(f"f{i}", "faq_real") for i in range(2)] +
                [slist(f"w{i}", "web_nonfaq") for i in range(3)])
        rep = RM.v5_reranker_training_report(rows)
        self.assertAlmostEqual(rep["faq_share"], 0.4)
        self.assertAlmostEqual(rep["nonfaq_share"], 0.6)
        self.assertTrue(rep["not_faq_only"])
        self.assertIn("faq_real", rep["score_separation_by_domain"])
        self.assertIn("web_nonfaq", rep["score_separation_by_domain"])
        # each list: 1 uncertain of 3 candidates -> 1/3
        self.assertAlmostEqual(rep["uncertain_fraction"], 1 / 3, 3)
        self.assertAlmostEqual(rep["score_separation_by_domain"]["faq_real"]["separation"], 7.0)

    def test_run_card_fields(self):
        rows = [slist("f0", "faq_real"), slist("w0", "web_nonfaq")]
        cap = RM.cap_faq_share(rows, 0.5)
        rep = RM.v5_reranker_training_report(cap["kept"])
        plan = RM.plan_v5_reranker_loss("listwise_kl+pairwise+pointwise_confident")
        card = RM.v5_reranker_run_card(rep, plan, cap, run_id="v5-reranker-boldt",
                                       model_base="Boldt/Boldt-DC-350M", output="out",
                                       bf16=True, gradient_checkpointing=True)
        self.assertEqual(card["run_id"], "v5-reranker-boldt")
        self.assertIn("score_separation_by_domain", card)
        self.assertIn("hardness-aware", card["evaluated_by"])


class TestDryRunNoMl(unittest.TestCase):
    def test_cli_dry_run_no_ml(self):
        with tempfile.TemporaryDirectory() as d:
            lists = pathlib.Path(d) / "scored.jsonl"
            rows = ([slist(f"f{i}", "faq_real") for i in range(3)] +
                    [slist(f"w{i}", "web_nonfaq") for i in range(3)] +
                    [slist(f"q{i}", "qa_passage_non_eval") for i in range(2)])
            lists.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = pathlib.Path(d) / "card.json"
            code = (
                "import sys; sys.path.insert(0, %r); "
                "sys.argv=['x','--config', %r,'--candidate-lists', %r,'--report', %r,"
                "'--run-id','t','--max-faq-share','0.5','--dry-run']\n"
                "import runpy; rc=0\n"
                "try:\n runpy.run_path(%r, run_name='__main__')\n"
                "except SystemExit as e:\n rc=e.code or 0\n"
                "assert 'torch' not in sys.modules, 'torch imported'\n"
                "print('RC', rc)\n"
                % (str(ROOT / "src"), str(V5_CFG), str(lists), str(report),
                   str(ROOT / "scripts" / "train_v5_rag_reranker.py"))
            )
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("RC 0", r.stdout)
            card = json.loads(report.read_text("utf-8"))["run_card"]
            self.assertTrue(card["not_faq_only"])
            self.assertLessEqual(card["faq_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
