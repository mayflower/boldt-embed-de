"""Stdlib tests for the 2026 release gate — pure-function pass/fail cases + integration."""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_release_2026 as VR  # noqa: E402


class TestWeightChecks(unittest.TestCase):
    def test_clean_passes(self):
        self.assertEqual(VR.check_no_committed_weights(["src/a.py", "configs/x.json"]), [])

    def test_weight_file_flagged(self):
        self.assertTrue(VR.check_no_committed_weights(["model.safetensors"]))
        self.assertTrue(VR.check_no_committed_weights(["a/b/pytorch_model.bin"]))

    def test_checkpoint_dir_flagged(self):
        self.assertTrue(VR.check_no_committed_weights(["outputs/checkpoints/m/config.json"]))

    def test_teacher_cache_flagged(self):
        self.assertTrue(VR.check_no_committed_teacher_cache(["outputs/teacher-cache/x.jsonl"]))
        self.assertEqual(VR.check_no_committed_teacher_cache(["outputs/eval/r.json"]), [])


class TestConfigChecks(unittest.TestCase):
    def test_real_repo_has_required_configs(self):
        self.assertEqual(VR.check_required_configs(ROOT), [])

    def test_empty_dir_missing_configs(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(VR.check_required_configs(pathlib.Path(d)))


class TestCardChecks(unittest.TestCase):
    def test_overclaim_flagged(self):
        self.assertTrue(VR.check_overclaims("c.md", "This is state-of-the-art and unbeatable."))
        self.assertEqual(VR.check_overclaims("c.md", "An honest German embedding model."), [])

    def test_missing_sections_flagged(self):
        issues = VR.check_card_sections("c.md", "## Limitations\nsome text", "embedder")
        kinds = {k for k, _ in issues}
        self.assertIn("card_missing_section", kinds)
        self.assertIn("card_missing_non_legal_warning", kinds)

    def test_complete_embedder_card_passes(self):
        text = ("## Teacher distillation\n## Training data provenance\n## Leakage policy\n"
                "## German stress tests\n## Limitations\n## Production default\n"
                "## Known failure modes\n## Matryoshka dimensions\nthis is not legal advice\n")
        self.assertEqual(VR.check_card_sections("c.md", text, "embedder"), [])

    def test_reranker_requires_lift_not_matryoshka(self):
        text = ("## Teacher distillation\n## Training data provenance\n## Leakage policy\n"
                "## German stress tests\n## Limitations\n## Production default\n"
                "## Known failure modes\n## Reranker lift\nnot legal advice\n")
        self.assertEqual(VR.check_card_sections("r.md", text, "reranker"), [])

    def test_checklist_reference(self):
        self.assertEqual(VR.check_checklist_references_runcards("see run card for each number"), [])
        self.assertTrue(VR.check_checklist_references_runcards("no provenance mentioned here"))


class TestIntegration(unittest.TestCase):
    def test_repo_release_gate_passes(self):
        report = VR.run_checks(ROOT)
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
