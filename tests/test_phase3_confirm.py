import json
import tempfile
import unittest
from pathlib import Path
from scripts.report_phase3_confirm import primary_succeeded


class Phase3ConfirmTests(unittest.TestCase):
    def test_positive_point_estimate_is_not_enough_to_test_the_secondary(self):
        result = {"delta": 0.03, "mcnemar_exact_p": 0.5}
        self.assertFalse(primary_succeeded(result, 0.05))
        self.assertTrue(primary_succeeded({"delta": 0.1, "mcnemar_exact_p": 0.01}, 0.05))
        self.assertFalse(primary_succeeded({"delta": -0.1, "mcnemar_exact_p": 0.01}, 0.05))

    def test_unfrozen_config_is_refused(self):
        from scripts.report_phase3_confirm import main
        import sys
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = {"status": "DEVELOPMENT_NOT_CONFIRMATION", "base_commit": "TO_FILL",
                      "study_id": "x", "population": "x", "n": 1, "primary": "x", "secondary": "x",
                      "alpha": 0.05, "read_k": 2, "split_seed": 1, "task_ids_hash": "x",
                      "source_hashes": {}, "observer_ledger_policy": "x", "reader_policy": "x",
                      "model_identity": "x", "failure_rules": "x", "exposure_manifest": "x"}
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "exposure.json").write_text(json.dumps({"unexposed_confirmation_available": False}), encoding="utf-8")
            (root / "gold.json").write_text("[]", encoding="utf-8")
            (root / "reader").mkdir()
            sys.argv = ["report", "--reader-root", str(root / "reader"), "--gold", str(root / "gold.json"),
                        "--official-evaluator", str(root / "missing.py"), "--config", str(root / "config.json"),
                        "--exposure", str(root / "exposure.json"), "--out", str(root / "out.json")]
            with self.assertRaises(ValueError):
                main()


if __name__ == "__main__":
    unittest.main()
