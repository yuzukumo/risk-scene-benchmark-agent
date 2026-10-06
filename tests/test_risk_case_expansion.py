import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_learned_retrieval import _write_fixture_db
from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.risk_case_expansion import expand_risk_cases


class RiskCaseExpansionTest(unittest.TestCase):
    def test_reuse_checks_artifacts_and_protocol_without_revalidating_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "index.db"
            db.write_text("database fixture")
            output = root / "output"
            output.mkdir()
            cases, report = output / "case_library.json", output / "expansion_report.json"
            cases.write_text("[]")
            report.write_text(json.dumps({"db": str(db), "scene_split": "val", "per_family_cap": 30,
                                          "candidate_limit_per_family": 180, "provenance": {"original": True}}))
            write_artifact_manifest(output, [build_artifact_entry(path, "input", "fixture", output)
                                             for path in [cases, report, db]])
            with patch("nusc_scene_agent.risk_case_expansion.validate_candidate", side_effect=AssertionError("revalidation")), \
                    patch("nusc_scene_agent.risk_case_expansion._write_benchmark_exports", side_effect=lambda _db, _out, result: result):
                result = expand_risk_cases(db, output, scene_split="val", reuse_validated=True)
                self.assertEqual(result["provenance"], {"original": True})
                self.assertIn("export_refresh", result)
                with self.assertRaisesRegex(ValueError, "different limits or scene split"):
                    expand_risk_cases(db, output, scene_split="all", reuse_validated=True)
                cases.write_text("changed")
                with self.assertRaisesRegex(ValueError, "failed verification"):
                    expand_risk_cases(db, output, scene_split="val", reuse_validated=True)

    def test_official_validation_filter_precedes_candidate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "index.db"
            _write_fixture_db(db)
            observed = []
            def validate(_conn, _query, candidate, **_kwargs):
                observed.append(candidate.scene_name)
                return Mock(passed=False, behavior_matches={})
            with patch("nuscenes.utils.splits.create_splits_scenes", return_value={"val": ["scene-ped"]}), \
                    patch("nusc_scene_agent.risk_case_expansion.validate_candidate", side_effect=validate), \
                    patch("nusc_scene_agent.risk_case_expansion._write_benchmark_exports", side_effect=lambda _db, _out, result: result):
                result = expand_risk_cases(db, root / "output", scene_split="val", reuse_validated=True)
            self.assertTrue(observed)
            self.assertEqual(set(observed), {"scene-ped"})
            self.assertEqual(result["scene_split"], "val")
