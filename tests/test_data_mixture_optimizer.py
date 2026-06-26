"""Tests for boldt_embed.data_mixture_optimizer (PURE STDLIB).

These never touch the real big data files: every source is a tiny fixture JSONL written
into a tempdir, referenced by ABSOLUTE path from a tiny fake catalogue dict.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src/ is importable without an install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from boldt_embed import data_mixture_optimizer as dmo  # noqa: E402
from boldt_embed.data_mixture_optimizer import MixtureConfigError  # noqa: E402


class TestStdlibPurity(unittest.TestCase):
    def test_no_torch_imported(self):
        # Checked in a FRESH subprocess so unittest-discover's torch-using modules can't pollute
        # the shared sys.modules (a bare `'torch' not in sys.modules` is order-dependent).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from torch_free import module_is_torch_free
        self.assertTrue(module_is_torch_free("boldt_embed.data_mixture_optimizer"))


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_catalogue(tmp: Path):
    """Build a tiny fake catalogue + fixtures. Returns (catalogue_dict, paths)."""
    # alpha: clean, training_usable, web domain — short + long docs
    alpha = tmp / "alpha.jsonl"
    _write_jsonl(alpha, [
        {"query": "frage eins", "document": "kurz a", "domain": "web"},          # short
        {"query": "frage zwei", "document": "x" * 1500, "domain": "web"},         # long
        {"query": "frage drei", "document": "y" * 400, "domain": "web"},          # medium
        {"query": "frage eins", "document": "kurz a", "domain": "web"},           # EXACT dup
        {"query": "Frage Eins", "document": "Kurz   A", "domain": "web"},         # NORM dup of #1
    ])
    # beta: clean, training_usable, faq domain — short docs
    beta = tmp / "beta.jsonl"
    _write_jsonl(beta, [
        {"query": "faq frage", "positive": "antwort kurz", "domain": "faq"},      # short, uses 'positive'
        {"query": "faq frage zwei", "positive": "antwort zwei", "domain": "faq"}, # short
    ])
    # unscanned: clean=false leakage
    unscanned = tmp / "unscanned.jsonl"
    _write_jsonl(unscanned, [{"query": "q", "document": "doc text here", "domain": "web"}])
    # notusable: training_usable false
    notusable = tmp / "notusable.jsonl"
    _write_jsonl(notusable, [{"query": "q", "document": "doc text here", "domain": "web"}])

    catalogue = {
        "train_pairs_processed_unions": [
            {"id": "alpha", "path": str(alpha), "domain": "web",
             "leakage": "scanned_clean", "training_usable": True},
        ],
        "train_pairs_raw_sources": [
            {"id": "beta", "path": str(beta), "domain": "faq",
             "leakage": "clean", "training_usable": True},
            {"id": "unscanned_src", "path": str(unscanned), "domain": "web",
             "leakage": "unscanned", "training_usable": True},
            {"id": "notusable_src", "path": str(notusable), "domain": "web",
             "leakage": "scanned_clean", "training_usable": False},
        ],
        "eval_only_NEVER_TRAIN": [
            {"id": "germanquad_eval"},
        ],
    }
    return catalogue, {"alpha": alpha, "beta": beta}


class TestFailClosed(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cat, _ = _make_catalogue(self.tmp)
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, sources):
        cfg = {"name": "t", "total_rows": 10, "sources": sources,
               "constraints": {"dedupe": {"exact": True, "normalized_text": True}}}
        return dmo.plan_mixture(cfg, self.cat, out_dir=self.out, created_utc="FIXED")

    def test_unknown_source_fails(self):
        with self.assertRaises(MixtureConfigError) as ctx:
            self._run({"does_not_exist": 1.0})
        self.assertIn("does_not_exist", str(ctx.exception))

    def test_unscanned_source_fails(self):
        with self.assertRaises(MixtureConfigError) as ctx:
            self._run({"unscanned_src": 1.0})
        self.assertIn("unscanned_src", str(ctx.exception))
        self.assertIn("leakage", str(ctx.exception))

    def test_training_usable_false_fails(self):
        with self.assertRaises(MixtureConfigError) as ctx:
            self._run({"notusable_src": 1.0})
        self.assertIn("notusable_src", str(ctx.exception))
        self.assertIn("training_usable", str(ctx.exception))

    def test_eval_only_source_fails(self):
        with self.assertRaises(MixtureConfigError) as ctx:
            self._run({"germanquad_eval": 1.0})
        self.assertIn("germanquad_eval", str(ctx.exception))
        self.assertIn("eval-only", str(ctx.exception))

    def test_zero_weight_fails(self):
        with self.assertRaises(MixtureConfigError) as ctx:
            self._run({"alpha": 0.0})
        self.assertIn("alpha", str(ctx.exception))
        self.assertIn("> 0", str(ctx.exception))

    def test_negative_weight_fails(self):
        with self.assertRaises(MixtureConfigError):
            self._run({"alpha": -0.5})

    def test_empty_sources_fails(self):
        with self.assertRaises(MixtureConfigError):
            self._run({})

    def test_missing_file_fails(self):
        bad = dict(self.cat)
        bad["train_pairs_processed_unions"] = [
            {"id": "alpha", "path": str(self.tmp / "nope.jsonl"), "domain": "web",
             "leakage": "scanned_clean", "training_usable": True}]
        cfg = {"name": "t", "total_rows": 5, "sources": {"alpha": 1.0}, "constraints": {}}
        with self.assertRaises(MixtureConfigError) as ctx:
            dmo.plan_mixture(cfg, bad, out_dir=self.out, created_utc="FIXED")
        self.assertIn("alpha", str(ctx.exception))


class TestDryRunDeterministic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cat, _ = _make_catalogue(self.tmp)
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def _cfg(self):
        # total_rows is far larger than the fixtures so every source's budget exceeds its row
        # count → stride 1 → all rows visited (the 2 alpha dups get exercised by dedupe).
        return {
            "name": "tiny_balanced",
            "total_rows": 100,
            "sources": {"alpha": 0.9, "beta": 0.1},
            "constraints": {
                "faq_cap": 0.30,
                "length_buckets": {"short": 0.4, "medium": 0.4, "long": 0.2},
                "dedupe": {"exact": True, "normalized_text": True, "simhash": False},
            },
        }

    def test_dry_run_writes_no_train_jsonl(self):
        res = dmo.plan_mixture(self._cfg(), self.cat, out_dir=self.out, created_utc="2026-01-01T00:00:00Z")
        self.assertTrue(res["dry_run"])
        self.assertTrue((self.out / "manifest.json").exists())
        self.assertTrue((self.out / "report.md").exists())
        self.assertFalse((self.out / "train.jsonl").exists())

    def test_real_build_dedupe_and_mix(self):
        # Use the REAL build to assert dedupe + domain/length mixes on a known fixture.
        res = dmo.build_mixture(self._cfg(), self.cat, out_dir=self.out,
                                created_utc="2026-01-01T00:00:00Z")
        man = res["manifest"]
        # alpha has 5 lines: 1 exact dup + 1 normalized dup of line 1 → 2 dropped, 3 unique kept.
        self.assertEqual(man["dedupe"]["dropped_exact"], 1)
        self.assertEqual(man["dedupe"]["dropped_normalized_text"], 1)
        # 3 unique alpha (web) + 2 beta (faq) = 5 rows. faq_cap 0.30 → faq <= 0.30/0.70*3 = 1 row.
        self.assertEqual(man["rows_written"], 4)
        self.assertEqual(man["leakage"], {"status": "scanned_clean", "basis": "source_catalogue"})
        # domain mix: 3 web + 1 faq out of 4
        self.assertEqual(man["domain_mix"].get("web"), 0.75)
        self.assertEqual(man["domain_mix"].get("faq"), 0.25)
        # length mix: web rows = short(kurz a), long(1500), medium(400); kept faq row = short.
        lm = man["length_mix"]
        self.assertEqual(lm["short"], round(2 / 4, 4))
        self.assertEqual(lm["medium"], round(1 / 4, 4))
        self.assertEqual(lm["long"], round(1 / 4, 4))
        # train.jsonl exists in a real build and has rows_written lines.
        train = self.out / "train.jsonl"
        self.assertTrue(train.exists())
        lines = [ln for ln in train.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), man["rows_written"])

    def test_manifest_is_deterministic(self):
        out_a = self.tmp / "a"
        out_b = self.tmp / "b"
        man_a = dmo.build_mixture(self._cfg(), self.cat, out_dir=out_a,
                                  created_utc="FIXED")["manifest"]
        man_b = dmo.build_mixture(self._cfg(), self.cat, out_dir=out_b,
                                  created_utc="FIXED")["manifest"]
        self.assertEqual(json.dumps(man_a, sort_keys=True), json.dumps(man_b, sort_keys=True))
        # source_hashes are present and stable.
        self.assertIn("alpha", man_a["source_hashes"])
        self.assertEqual(man_a["source_hashes"], man_b["source_hashes"])

    def test_weights_normalized(self):
        cfg = self._cfg()
        cfg["sources"] = {"alpha": 4.0, "beta": 4.0}  # non-normalized, equal
        spec = dmo.validate_mixture_config(cfg, self.cat)
        self.assertAlmostEqual(sum(spec["weights"].values()), 1.0)
        self.assertAlmostEqual(spec["weights"]["alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()
