import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import reranker  # noqa: E402
from boldt_embed.textutil import jaccard, tokenize  # noqa: E402

CONFIG = ROOT / "configs" / "training_reranker.json"


def lexical_scorer(query: str, document: str) -> float:
    return jaccard(tokenize(query), tokenize(document))


class TestRerankerDryRun(unittest.TestCase):
    def setUp(self):
        self.rr = reranker.Reranker.from_config(CONFIG)

    def test_build_input_has_both_fields(self):
        s = self.rr.build_input("Wie hoch ist die Miete?", "Die Miete beträgt 800 Euro.")
        self.assertIn("Wie hoch ist die Miete?", s)
        self.assertIn("Die Miete beträgt 800 Euro.", s)

    def test_dry_run(self):
        report = self.rr.dry_run("frage", ["doc a", "doc b"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["labels"], ["Ja", "Nein"])
        self.assertEqual(report["num_pairs"], 2)


class TestHardNegativeMining(unittest.TestCase):
    def test_mines_high_scoring_non_positives(self):
        query = "kündigungsfrist mietwohnung drei monate"
        positive = "Die Kündigungsfrist für eine Mietwohnung beträgt drei Monate."
        candidates = [
            positive,
            "Die Kündigungsfrist für die Mietwohnung kann drei Monate betragen.",  # close distractor
            "Ein Rezept für Apfelkuchen mit Zimt und Zucker.",                      # unrelated
        ]
        negs = reranker.mine_hard_negatives(
            query, candidates, lexical_scorer, positives=[positive], k=1
        )
        self.assertEqual(len(negs), 1)
        self.assertNotEqual(negs[0], positive)
        self.assertIn("Kündigungsfrist", negs[0])  # the lexically close distractor, not the cake


class TestDistillation(unittest.TestCase):
    def test_softmax_sums_to_one_and_keeps_order(self):
        probs = reranker.softmax([2.0, 1.0, 0.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(probs[0] > probs[1] > probs[2])

    def test_temperature_sharpens(self):
        sharp = reranker.distillation_soft_labels([2.0, 1.0, 0.0], temperature=0.5)
        soft = reranker.distillation_soft_labels([2.0, 1.0, 0.0], temperature=2.0)
        self.assertGreater(sharp[0], soft[0])

    def test_margin_mse_target(self):
        self.assertAlmostEqual(reranker.margin_mse_target(0.9, 0.2), 0.7)

    def test_softmax_bad_temperature(self):
        with self.assertRaises(ValueError):
            reranker.softmax([1.0], temperature=0.0)


if __name__ == "__main__":
    unittest.main()
