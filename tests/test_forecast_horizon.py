import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import yaml

from test_world_model_benchmark import _build_test_db, _write_scenario_config
from nusc_scene_agent.perception_benchmark import generate_perception_benchmark_from_scenario_config
from nusc_scene_agent.world_model_benchmark import _FrameCountPredictHelper, _benchmark_horizon_seconds, _split_history_future, generate_world_model_benchmark_from_perception_benchmark


class ForecastHorizonTest(unittest.TestCase):
    def test_nominal_horizon_does_not_grow_from_timestamp_jitter(self):
        benchmark = {"cases": [{"motion_targets": {"horizon_s": 6.000943}}]}
        self.assertEqual(_benchmark_horizon_seconds(benchmark), 6.0)

    def test_oracle_alignment_tolerates_timestamp_jitter_without_extra_frames(self):
        helper = _FrameCountPredictHelper.__new__(_FrameCountPredictHelper)
        with patch("nusc_scene_agent.world_model_benchmark.PredictHelper.get_future_for_agent",
                   return_value=np.zeros((12, 2))) as future:
            result = helper.get_future_for_agent("actor", "anchor", 5.5, in_agent_frame=False)
        self.assertEqual(result.shape, (11, 2))
        self.assertEqual(future.call_args.args[2], 5.6)

    def test_missing_future_does_not_silently_move_anchor(self):
        frames = [{"sample_idx": i} for i in range(4)]
        history, future = _split_history_future({"anchor_sample_idx": 3, "frames": frames})
        self.assertEqual(history, frames)
        self.assertEqual(future, [])
        history, future = _split_history_future({"anchor_sample_idx": -1, "frames": frames})
        self.assertEqual(history, [])

    def test_reference_anchor_outside_event_window_keeps_its_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, query, perception, output = [root / name for name in ["index.db", "query.yaml", "perception.json", "forecast.json"]]
            _build_test_db(db)
            _write_scenario_config(query)
            config = yaml.safe_load(query.read_text())
            for spec in config["queries"]:
                spec["reference_event_sample_range"] = [0, 1]
                spec["reference_peak_sample_idx"] = 1
            query.write_text(yaml.safe_dump(config))
            generate_perception_benchmark_from_scenario_config(query, db, perception)
            payload = json.loads(perception.read_text())
            self.assertEqual(payload["cases"][0]["anchor_sample_idx"], 2)
            self.assertIn(payload["cases"][0]["anchor_sample_token"],
                          [frame["sample_token"] for frame in payload["cases"][0]["frames"]])
            payload["cases"][0]["anchor_sample_idx"] = 0
            perception.write_text(json.dumps(payload))
            generate_world_model_benchmark_from_perception_benchmark(perception, db, output)
            case = json.loads(output.read_text())["cases"][0]
            self.assertEqual(case["rollout_anchor_sample_token"], case["anchor_sample_token"])
            self.assertEqual(case["rollout_anchor_sample_idx"], 2)

    def test_forecast_loads_beyond_event_window_and_reports_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db, query, perception, output = [root / name for name in ["index.db", "query.yaml", "perception.json", "forecast.json"]]
            _build_test_db(db)
            _write_scenario_config(query)
            generate_perception_benchmark_from_scenario_config(query, db, perception)
            original = json.loads(perception.read_text())
            original["cases"][0]["frames"] = original["cases"][0]["frames"][:3]
            perception.write_text(json.dumps(original))
            generate_world_model_benchmark_from_perception_benchmark(perception, db, output)
            forecast = json.loads(output.read_text())
            self.assertGreater(forecast["cases"][0]["future_frame_count"], 0)
            self.assertEqual(forecast["cases"][0]["rollout_anchor_sample_idx"], 2)
            generate_world_model_benchmark_from_perception_benchmark(perception, db, output, min_future_s=3.0)
            forecast = json.loads(output.read_text())
            self.assertEqual(forecast["cases"], [])
            self.assertEqual(len(forecast["metadata"]["excluded_cases"]), 1)
