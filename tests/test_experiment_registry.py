"""Stdlib tests for the experiment registry + run-card summarizer."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boldt_embed import experiment_registry as ER  # noqa: E402

SUMMARIZE = ROOT / "scripts" / "summarize_experiments.py"


def _card(**over):
    base = ER.new_run_card("rid-1", "eval", "scripts/x.py", model="m", dataset="d",
                           metrics={"ndcg@10": 0.5}, date="2026-06-09T00:00:00+00:00")
    base.update(over)
    return base


class TestRunCardSchema(unittest.TestCase):
    def test_new_run_card_has_fields(self):
        c = _card()
        for key in ("run_id", "run_type", "command", "commit", "date", "metrics",
                    "input_artifacts", "output_artifacts", "model", "dataset"):
            self.assertIn(key, c)

    def test_validate_good(self):
        self.assertEqual(ER.validate_run_card(_card()), [])

    def test_validate_missing_field(self):
        c = _card(); del c["command"]
        self.assertTrue(any("command" in e for e in ER.validate_run_card(c)))

    def test_validate_bad_run_type(self):
        self.assertTrue(any("run_type" in e for e in ER.validate_run_card(_card(run_type="bogus"))))

    def test_link_artifacts_dedups(self):
        c = ER.link_artifacts(_card(input_artifacts=["a"]), inputs=["a", "b"], outputs=["o"])
        self.assertEqual(c["input_artifacts"], ["a", "b"])
        self.assertEqual(c["output_artifacts"], ["o"])


class TestEnvMetadata(unittest.TestCase):
    def test_keys_present(self):
        meta = ER.collect_env_metadata()
        for key in ("commit", "python", "platform", "torch", "transformers", "sentence_transformers"):
            self.assertIn(key, meta)

    def test_no_ml_import_subprocess(self):
        code = ("import sys; sys.path.insert(0, %r);"
                "from boldt_embed import experiment_registry as ER; ER.collect_env_metadata();"
                "assert 'torch' not in sys.modules; print('clean')") % str(ROOT / "src")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


class TestWriteReadEmit(unittest.TestCase):
    def test_write_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = ER.write_run_card(_card(run_id="abc"), out_dir=pathlib.Path(d))
            self.assertTrue(pathlib.Path(p).exists())
            cards = ER.read_run_cards(d)
            self.assertEqual(cards[0]["run_id"], "abc")

    def test_emit_writes_card(self):
        with tempfile.TemporaryDirectory() as d:
            p = ER.emit_run_card("emit-1", "train_embedder", "cmd", model="m",
                                 metrics={"x": 1}, out_dir=pathlib.Path(d))
            self.assertTrue(pathlib.Path(p).exists())
            self.assertEqual(ER.read_run_cards(d)[0]["run_type"], "train_embedder")


class TestSummarizer(unittest.TestCase):
    def test_summary_and_filter(self):
        with tempfile.TemporaryDirectory() as d:
            cards_dir = pathlib.Path(d) / "cards"
            ER.write_run_card(_card(run_id="eval-1", run_type="eval", dataset="gerdalir"),
                              out_dir=cards_dir)
            ER.write_run_card(_card(run_id="train-1", run_type="train_reranker",
                                    metrics={"final_loss": 0.2}), out_dir=cards_dir)
            md_all = pathlib.Path(d) / "all.md"
            out = subprocess.run(
                [sys.executable, str(SUMMARIZE), "--run-cards-dir", str(cards_dir),
                 "--output", str(md_all)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            text = md_all.read_text()
            self.assertIn("eval-1", text)
            self.assertIn("train-1", text)
            # filter by run_type
            md_eval = pathlib.Path(d) / "eval.md"
            subprocess.run([sys.executable, str(SUMMARIZE), "--run-cards-dir", str(cards_dir),
                            "--output", str(md_eval), "--run-type", "eval"],
                           capture_output=True, text=True, check=True)
            etext = md_eval.read_text()
            self.assertIn("eval-1", etext)
            self.assertNotIn("train-1", etext)


if __name__ == "__main__":
    unittest.main()
