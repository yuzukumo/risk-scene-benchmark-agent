import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nusc_scene_agent.carla_fixed_evaluation import run_fixed_carla_evaluation


class FixedCarlaEvaluationTest(unittest.TestCase):
    def test_keeps_driving_failures_and_errors_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fixture")
            config = {"output": str(root / "evaluation"), "checkpoint": str(checkpoint), "seeds": [7],
                      "min_completion": 0.9, "max_safety_override_ratio": 0.0,
                      "scenarios": [{"name": name, "type": "free_drive", "spawn_index": 0} for name in ["pass", "fail", "error"]]}
            good = {"metrics": {"collision_count": 0, "route_completion": 1.0},
                    "control_attribution": {"safety_override_ratio": 0.0}}
            bad = {"metrics": {"collision_count": 1, "route_completion": 0.3},
                   "control_attribution": {"safety_override_ratio": 0.0}}
            with patch("nusc_scene_agent.carla_fixed_evaluation._run_isolated_attempt",
                       side_effect=[good, bad, RuntimeError("connection failed")]) as runner:
                result = run_fixed_carla_evaluation(config)
                self.assertEqual(runner.call_count, 3)
                repeated = run_fixed_carla_evaluation(config)
                self.assertEqual(runner.call_count, 3)
            self.assertEqual(result["attempt_count"], 3)
            self.assertEqual(result["error_count"], 1)
            self.assertEqual(result["success_rate"], 1 / 3)
            self.assertEqual(result["attempts"], repeated["attempts"])
