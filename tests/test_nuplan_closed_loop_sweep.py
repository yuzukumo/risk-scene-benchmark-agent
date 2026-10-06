import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.nuplan_closed_loop_sweep import run_nuplan_closed_loop_sweep
from test_nuplan_replay import _build_nuplan_db


class NuPlanClosedLoopSweepTest(unittest.TestCase):
    def test_run_closed_loop_sweep_aggregates_studies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_a = root / "split_a"
            split_b = root / "split_b"
            split_a.mkdir()
            split_b.mkdir()
            _build_nuplan_db(split_a / "a.db")
            _build_nuplan_db(split_b / "b.db")

            output_dir = root / "closed_loop_sweep"
            payload = run_nuplan_closed_loop_sweep(
                studies=[
                    {"name": "city_a", "split_dir": str(split_a)},
                    {"name": "city_b", "split_dir": str(split_b)},
                ],
                output_dir=output_dir,
                defaults={
                    "max_dbs": 1,
                    "max_cases": 1,
                    "max_cases_per_db": 1,
                    "history_s": 0.5,
                    "future_s": 1.0,
                    "frame_hz": 2.0,
                    "profiles": ["logged_ego_oracle", "history_kinematic"],
                },
            )

            self.assertEqual(payload["schema"], "nuplan_closed_loop_sweep_v1")
            self.assertEqual(payload["overview"]["study_count"], 2)
            self.assertEqual(payload["overview"]["case_count"], 2)
            self.assertEqual(payload["overview"]["profile_count"], 2)
            self.assertTrue((output_dir / "nuplan_closed_loop_sweep_summary.json").exists())
            self.assertTrue((output_dir / "nuplan_closed_loop_sweep_leaderboard.csv").exists())
            self.assertTrue((output_dir / "nuplan_closed_loop_sweep_family_matrix.csv").exists())
            self.assertTrue((output_dir / "nuplan_closed_loop_sweep_failure_taxonomy.csv").exists())
            self.assertTrue((output_dir / "artifact_manifest.json").exists())
            manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
            for entry in manifest["artifacts"]:
                path = output_dir / entry["path"]
                if path.is_file():
                    self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

            overall_rows = [
                row for row in payload["profile_leaderboard"] if row["study_name"] == "__overall__"
            ]
            self.assertEqual({row["profile_name"] for row in overall_rows}, {"logged_ego_oracle", "history_kinematic"})

    def test_experiment_config_supports_closed_loop_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            _build_nuplan_db(split_dir / "mini.db")
            config_path = root / "closed_loop_sweep.yaml"
            output_dir = root / "outputs"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment:",
                        "  id: closed_loop_sweep_test",
                        "  type: nuplan_closed_loop_sweep",
                        f"  output: {output_dir}",
                        f"  result_path: {output_dir / 'result.json'}",
                        "nuplan_closed_loop_sweep:",
                        f"  output: {output_dir}",
                        "  defaults:",
                        "    max_dbs: 1",
                        "    max_cases: 1",
                        "    max_cases_per_db: 1",
                        "    history_s: 0.5",
                        "    future_s: 1.0",
                        "    frame_hz: 2.0",
                        "    profiles:",
                        "      - logged_ego_oracle",
                        "  studies:",
                        "    - name: mini_fixture",
                        f"      split_dir: {split_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            result = run_experiment_config(config_path)
            self.assertEqual(result["experiment_type"], "nuplan_closed_loop_sweep")
            self.assertEqual(result["result"]["overview"]["case_count"], 1)
            written = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(written["experiment_id"], "closed_loop_sweep_test")


if __name__ == "__main__":
    unittest.main()
