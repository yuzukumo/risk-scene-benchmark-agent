import json
import copy
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from nuscenes.utils.splits import create_splits_scenes

from nusc_scene_agent.artifact_manifest import build_artifact_entry, verify_artifact_manifest, write_artifact_manifest
from nusc_scene_agent.forecast_validation import audit_forecast_generalization
from nusc_scene_agent.perception_benchmark import generate_perception_benchmark_from_scenario_config
from nusc_scene_agent.world_model_benchmark import (
    adapt_and_evaluate_nuscenes_forecast_predictions,
    adapt_nuscenes_forecast_predictions,
    adapt_world_model_predictions,
    compare_world_model_evaluations,
    evaluate_world_model_predictions,
    export_world_model_replay,
    generate_nuscenes_forecast_baselines,
    generate_proxy_world_model_predictions,
    generate_world_model_benchmark_from_perception_benchmark,
    run_nuscenes_forecast_baselines,
    run_proxy_world_model_study,
)


def _write_scenario_config(path: Path) -> None:
    path.write_text(
        """
queries:
  - id: crossing_canonical
    description: Pedestrian crossing ahead of ego
    top_k: 3
    expect_match: true
    benchmark_group: crossing_anchor
    variant_type: canonical
    tags: [scenario_mining, crossing]
    query:
      natural_language: Pedestrian crossing ahead of ego
      actors: [pedestrian]
      behaviors: [crossing]
      reference_case_keys: [s2:inst-1]
      reference_scene_names: [scene-0001]
      reference_instance_tokens: [inst-1]
      reference_event_sample_range: [0, 4]
      reference_peak_sample_idx: 2
  - id: crossing_paraphrase
    description: Person moving through the ego lane
    top_k: 3
    expect_match: true
    benchmark_group: crossing_anchor
    variant_type: paraphrase
    tags: [scenario_mining, crossing, paraphrase]
    query:
      natural_language: Person moving through the ego lane
      actors: [person]
      behaviors: [crossing]
      reference_case_keys: [s2:inst-1]
      reference_scene_names: [scene-0001]
      reference_instance_tokens: [inst-1]
      reference_event_sample_range: [0, 4]
      reference_peak_sample_idx: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_test_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE samples (
            sample_token TEXT PRIMARY KEY,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            scene_description TEXT,
            sample_idx INTEGER NOT NULL,
            timestamp_us INTEGER NOT NULL,
            location TEXT NOT NULL,
            ego_x REAL NOT NULL,
            ego_y REAL NOT NULL,
            ego_yaw REAL NOT NULL
        );

        CREATE TABLE agents (
            ann_token TEXT PRIMARY KEY,
            sample_token TEXT NOT NULL,
            scene_token TEXT NOT NULL,
            scene_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            instance_token TEXT NOT NULL,
            category_name TEXT NOT NULL,
            category_group TEXT NOT NULL,
            x_ego REAL NOT NULL,
            y_ego REAL NOT NULL,
            distance REAL NOT NULL,
            visibility INTEGER NOT NULL,
            ttc REAL,
            speed REAL NOT NULL,
            rel_vx REAL NOT NULL,
            rel_vy REAL NOT NULL,
            heading_delta REAL NOT NULL,
            num_lidar_pts INTEGER NOT NULL,
            num_radar_pts INTEGER NOT NULL
        );
        """
    )

    sample_rows = [
        ("s0", "scene-token-1", "scene-0001", "test scene", 0, 0, "boston-seaport", 100.0, 50.0, 0.0),
        ("s1", "scene-token-1", "scene-0001", "test scene", 1, 500000, "boston-seaport", 102.0, 50.5, 0.0),
        ("s2", "scene-token-1", "scene-0001", "test scene", 2, 1000000, "boston-seaport", 104.0, 51.0, 0.0),
        ("s3", "scene-token-1", "scene-0001", "test scene", 3, 1500000, "boston-seaport", 106.0, 51.5, 0.0),
        ("s4", "scene-token-1", "scene-0001", "test scene", 4, 2000000, "boston-seaport", 108.0, 52.0, 0.0),
    ]
    agent_rows = [
        ("ann-0", "s0", "scene-token-1", "scene-0001", 0, "inst-1", "human.pedestrian.adult", "pedestrian", 8.0, 1.2, 8.1, 4, 4.0, 1.3, -1.0, 0.0, 0.0, 8, 0),
        ("ann-1", "s1", "scene-token-1", "scene-0001", 1, "inst-1", "human.pedestrian.adult", "pedestrian", 6.0, 0.7, 6.0, 4, 3.0, 1.2, -1.0, 0.0, 0.0, 8, 0),
        ("ann-2", "s2", "scene-token-1", "scene-0001", 2, "inst-1", "human.pedestrian.adult", "pedestrian", 4.0, 0.1, 4.0, 3, 2.0, 1.1, -1.0, 0.0, 0.0, 8, 0),
        ("ann-3", "s3", "scene-token-1", "scene-0001", 3, "inst-1", "human.pedestrian.adult", "pedestrian", 2.0, -0.5, 2.1, 2, 1.2, 1.0, -1.0, 0.0, 0.0, 8, 0),
        ("ann-4", "s4", "scene-token-1", "scene-0001", 4, "inst-1", "human.pedestrian.adult", "pedestrian", 0.5, -1.0, 1.1, 2, 1.0, 1.0, -1.0, 0.0, 0.0, 8, 0),
    ]
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("dataroot", str(path.parent / "data")))
    conn.executemany("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_rows)
    conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", agent_rows)
    conn.commit()
    conn.close()


class WorldModelBenchmarkTest(unittest.TestCase):
    def _create_world_model_benchmark(self, root: Path) -> Path:
        config_path = root / "scenario.yaml"
        db_path = root / "index.sqlite"
        perception_benchmark_path = root / "perception_benchmark.json"
        world_model_benchmark_path = root / "world_model_benchmark.json"
        _write_scenario_config(config_path)
        _build_test_db(db_path)
        generate_perception_benchmark_from_scenario_config(config_path, db_path, perception_benchmark_path)
        generate_world_model_benchmark_from_perception_benchmark(perception_benchmark_path, db_path, world_model_benchmark_path)
        return world_model_benchmark_path

    def test_validation_audit_excludes_entire_development_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text())
            cases = []
            for index, name in enumerate(create_splits_scenes()["val"][:2]):
                case = copy.deepcopy(benchmark["cases"][0])
                case.update(benchmark_group=f"case_{index}", scene_token=f"scene_{index}", scene_name=name)
                cases.append(case)
            benchmark["cases"] = cases
            benchmark_path.write_text(json.dumps(benchmark))
            predictions = root / "predictions.json"
            generate_proxy_world_model_predictions(benchmark_path, predictions, "oracle_rollout")
            study = root / "study"
            for directory in ["contextvae", "nuscenes_baselines/cv_heading", "nuscenes_baselines/physics_oracle"]:
                target = study / directory
                target.mkdir(parents=True)
                (target / "world_model_metrics.json").write_text(json.dumps({"predictions_path": str(predictions)}))
            (study / "contextvae_study_manifest.json").write_text(json.dumps({
                "preparation": {"subset_benchmark_path": str(benchmark_path)},
            }))
            write_artifact_manifest(study, [build_artifact_entry(predictions, "input", "forecast")])
            development = root / "development.json"
            development.write_text(json.dumps({"cases": [{"scene_token": "scene_0", "instance_token": "another_actor"}]}))
            output = root / "audit"
            result = audit_forecast_generalization(study, [development], output)
            self.assertEqual(result["case_count"], 1)
            self.assertEqual(result["excluded_case_count"], 1)
            self.assertEqual(result["shared_development_scene_count"], 0)
            retained = json.loads(Path(result["benchmark_path"]).read_text())["cases"]
            self.assertEqual(retained[0]["scene_token"], "scene_1")
            self.assertTrue(verify_artifact_manifest(output / "artifact_manifest.json")["valid"])

    def test_generate_world_model_benchmark_from_perception_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["case_count"], 1)
            case = payload["cases"][0]
            self.assertEqual(case["benchmark_group"], "crossing_anchor")
            self.assertEqual(case["history_frame_count"], 3)
            self.assertEqual(case["future_frame_count"], 2)
            self.assertEqual(case["motion_targets"]["horizon_s"], 1.0)
            self.assertEqual(case["motion_targets"]["closest_approach_sample_idx"], 4)
            self.assertEqual(len(case["future_occupancy"]), 2)
            self.assertIn("occupied_cells", case["future_occupancy"][0])
            self.assertIn("challenge/crossing_emergence", case["challenge_tracks"])
            self.assertTrue(payload["metadata"]["challenge_tracks"])
            self.assertIn("x_global", case["future_frames"][0])
            self.assertIn("ego_yaw", case["future_frames"][0])

    def test_evaluate_oracle_world_model_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            predictions_path = root / "oracle_predictions.json"
            output_dir = root / "oracle_eval"
            generate_proxy_world_model_predictions(benchmark_path, predictions_path, "oracle_rollout")

            summary = evaluate_world_model_predictions(benchmark_path, predictions_path, output_dir)

            self.assertEqual(summary["overview"]["case_count"], 1)
            self.assertEqual(summary["overview"]["full_horizon_count"], 1)
            self.assertEqual(summary["overview"]["mean_ade_m"], 0.0)
            self.assertEqual(summary["overview"]["mean_fde_m"], 0.0)
            self.assertEqual(summary["overview"]["mean_occupancy_iou"], 1.0)
            self.assertEqual(summary["overview"]["mean_primary_actor_iou"], 1.0)
            self.assertEqual(summary["overview"]["mean_risk_fidelity_score"], 1.0)
            self.assertEqual(summary["case_metrics"][0]["failure_tags"], [])
            self.assertEqual(summary["track_breakdown"][0]["track"], "challenge/critical_range")
            self.assertTrue((output_dir / "world_model_metrics.json").exists())
            self.assertTrue((output_dir / "world_model_metrics_summary.md").exists())
            self.assertTrue((output_dir / "world_model_metrics_summary.html").exists())
            self.assertTrue((output_dir / "world_model_case_metrics.csv").exists())

    def test_missing_predictions_remain_missing_in_exported_error_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark = self._create_world_model_benchmark(root)
            predictions = root / "empty.json"
            predictions.write_text(json.dumps({"predictions": []}))
            output = root / "evaluation"
            result = evaluate_world_model_predictions(benchmark, predictions, output)
            self.assertIsNone(result["overview"]["mean_ade_m"])
            self.assertIsNone(result["overview"]["mean_fde_m"])
            self.assertIsNone(result["forecast_metrics"]["mean_min_ade_at_5"])
            self.assertEqual(result["forecast_metrics"]["mean_miss_rate_at_5"], 1.0)
            comparison = compare_world_model_evaluations([output], root / "comparison")
            self.assertEqual(comparison["case_count"], 1)
            exported = json.loads((root / "comparison/world_model_comparison.json").read_text())
            self.assertIsNone(exported["profiles"][0]["mean_ade_m"])
            for path in [output / "world_model_metrics_summary.md", output / "world_model_metrics_summary.html",
                         root / "comparison/world_model_comparison_summary.md",
                         root / "comparison/world_model_comparison_summary.html"]:
                self.assertIn("n/a", path.read_text())

    def test_adapt_compact_world_model_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            raw_input_path = root / "raw_rollout.json"
            adapted_path = root / "adapted_rollout.json"
            eval_dir = root / "adapted_eval"
            raw_input_path.write_text(
                json.dumps(
                    {
                        "metadata": {"profile_name": "compact_oracle"},
                        "predictions": {
                            case["benchmark_group"]: {
                                "xy_ego": [[frame["x_ego"], frame["y_ego"]] for frame in case["future_frames"]],
                                "sample_indices": [frame["sample_idx"] for frame in case["future_frames"]],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            metadata = adapt_world_model_predictions(benchmark_path, raw_input_path, adapted_path)
            self.assertEqual(metadata["prediction_count"], 1)
            adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
            self.assertEqual(adapted["metadata"]["profile_name"], "compact_oracle")
            self.assertEqual(len(adapted["predictions"][0]["future_trajectory"]), 2)
            self.assertEqual(len(adapted["predictions"][0]["future_occupancy"]), 2)

            summary = evaluate_world_model_predictions(benchmark_path, adapted_path, eval_dir)
            self.assertEqual(summary["overview"]["mean_ade_m"], 0.0)
            self.assertEqual(summary["overview"]["mean_occupancy_iou"], 1.0)
            self.assertEqual(summary["overview"]["mean_risk_fidelity_score"], 1.0)

    def test_adapt_nuscenes_forecast_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            raw_input_path = root / "nuscenes_forecast.json"
            adapted_path = root / "adapted_nuscenes_forecast.json"

            good_mode = [[frame["x_global"], frame["y_global"]] for frame in case["future_frames"]]
            bad_mode = [[frame["x_global"] + 3.0, frame["y_global"] - 2.0] for frame in case["future_frames"]]
            raw_input_path.write_text(
                json.dumps(
                    [
                        {
                            "instance": case["instance_token"],
                            "sample": case["rollout_anchor_sample_token"],
                            "prediction": [good_mode, bad_mode],
                            "probabilities": [0.9, 0.1],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            metadata = adapt_nuscenes_forecast_predictions(
                benchmark_path=benchmark_path,
                input_path=raw_input_path,
                output_path=adapted_path,
                mode_selection="top_probability",
            )
            self.assertEqual(metadata["prediction_count"], 1)
            adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
            self.assertEqual(adapted["predictions"][0]["selected_mode_index"], 0)
            self.assertEqual(len(adapted["predictions"][0]["future_trajectory"]), 2)
            self.assertEqual(len(adapted["predictions"][0]["future_trajectory_modes"]), 2)

    def test_adapt_and_evaluate_nuscenes_forecast_predictions_oracle_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            raw_input_path = root / "nuscenes_forecast_oracle.json"
            output_dir = root / "nuscenes_forecast_eval"

            bad_mode = [[frame["x_global"] + 4.0, frame["y_global"] + 1.5] for frame in case["future_frames"]]
            good_mode = [[frame["x_global"], frame["y_global"]] for frame in case["future_frames"]]
            raw_input_path.write_text(
                json.dumps(
                    [
                        {
                            "instance": case["instance_token"],
                            "sample": case["rollout_anchor_sample_token"],
                            "prediction": [bad_mode, good_mode],
                            "probabilities": [0.95, 0.05],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            metadata = adapt_and_evaluate_nuscenes_forecast_predictions(
                benchmark_path=benchmark_path,
                input_path=raw_input_path,
                output_dir=output_dir,
                mode_selection="oracle_ade",
            )
            self.assertEqual(metadata["adapter"]["prediction_count"], 1)
            self.assertEqual(metadata["overview"]["mean_ade_m"], 0.0)
            self.assertEqual(metadata["overview"]["mean_risk_fidelity_score"], 1.0)
            adapted = json.loads((output_dir / "adapted_predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(adapted["predictions"][0]["selected_mode_index"], 1)

    def test_multimodal_forecast_topk_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            case = benchmark["cases"][0]
            raw_input_path = root / "nuscenes_forecast_multimodal.json"
            adapted_path = root / "adapted_multimodal.json"
            eval_dir = root / "multimodal_eval"

            bad_mode = [[frame["x_global"] + 5.0, frame["y_global"] + 0.5] for frame in case["future_frames"]]
            good_mode = [[frame["x_global"], frame["y_global"]] for frame in case["future_frames"]]
            raw_input_path.write_text(
                json.dumps(
                    [
                        {
                            "instance": case["instance_token"],
                            "sample": case["rollout_anchor_sample_token"],
                            "prediction": [bad_mode, good_mode],
                            "probabilities": [0.9, 0.1],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            adapt_nuscenes_forecast_predictions(
                benchmark_path=benchmark_path,
                input_path=raw_input_path,
                output_path=adapted_path,
                mode_selection="top_probability",
            )
            summary = evaluate_world_model_predictions(benchmark_path, adapted_path, eval_dir)

            self.assertGreater(summary["overview"]["mean_ade_m"], 1.0)
            self.assertGreater(summary["forecast_metrics"]["mean_min_ade_at_1"], 1.0)
            self.assertEqual(summary["forecast_metrics"]["mean_min_ade_at_5"], 0.0)
            self.assertEqual(summary["forecast_metrics"]["mean_miss_rate_at_1"], 1.0)
            self.assertEqual(summary["forecast_metrics"]["mean_miss_rate_at_5"], 0.0)

    def test_run_proxy_world_model_study_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            study_output = root / "study"

            metadata = run_proxy_world_model_study(benchmark_path, study_output)
            self.assertEqual(metadata["profile_count"], 3)
            self.assertTrue((study_output / "world_model_comparison.json").exists())

            comparison_output = root / "comparison"
            compare_metadata = compare_world_model_evaluations(
                [study_output / "oracle_rollout", study_output / "kinematic_rollout"],
                comparison_output,
            )
            self.assertEqual(compare_metadata["profile_count"], 2)
            comparison = json.loads((comparison_output / "world_model_comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["profiles"][0]["name"], "oracle_rollout")
            self.assertTrue(comparison["track_matrix"])
            self.assertEqual(comparison["overview"]["common_case_count"], 1)
            self.assertEqual(comparison["overview"]["comparison_basis"], "common_case_intersection")
            self.assertEqual(comparison["common_case_summary"][0]["case_count"], 1)
            self.assertEqual(comparison["profiles"][0]["case_count"], 1)
            self.assertTrue(comparison["full_set_profiles"])
            self.assertEqual(comparison["overview"]["bootstrap_replicates"], 5000)
            self.assertTrue(comparison["common_case_summary"][0]["uncertainty"])
            self.assertTrue(comparison["paired_profile_comparisons"])
            self.assertGreater(
                comparison["profiles"][0]["mean_risk_fidelity_score"],
                comparison["profiles"][1]["mean_risk_fidelity_score"],
            )

    def test_generate_nuscenes_forecast_baselines_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            output_dir = root / "baselines"

            class FakePrediction:
                def __init__(self, instance: str, sample: str) -> None:
                    self.instance = instance
                    self.sample = sample

                def serialize(self) -> dict:
                    return {
                        "instance": self.instance,
                        "sample": self.sample,
                        "prediction": [[[1.0, 2.0], [2.0, 3.0]]],
                        "probabilities": [1.0],
                    }

            class FakeModel:
                def __init__(self, horizon_s: float, helper: object) -> None:
                    self.horizon_s = horizon_s
                    self.helper = helper

                def __call__(self, token: str) -> FakePrediction:
                    instance, sample = token.split("_", 1)
                    return FakePrediction(instance, sample)

            with mock.patch("nusc_scene_agent.world_model_benchmark.NuScenes"), mock.patch(
                "nusc_scene_agent.world_model_benchmark.PredictHelper"
            ), mock.patch(
                "nusc_scene_agent.world_model_benchmark.ConstantVelocityHeading", FakeModel
            ), mock.patch(
                "nusc_scene_agent.world_model_benchmark.PhysicsOracle", FakeModel
            ):
                metadata = generate_nuscenes_forecast_baselines(
                    benchmark_path=benchmark_path,
                    dataroot=root / "data",
                    version="v1.0-trainval",
                    output_dir=output_dir,
                )

            self.assertEqual(metadata["case_count"], 1)
            self.assertTrue((output_dir / "cv_heading.json").exists())
            self.assertTrue((output_dir / "physics_oracle.json").exists())
            manifest = json.loads((output_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], "v1.0-trainval")

    def test_run_nuscenes_forecast_baselines_with_mocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            output_dir = root / "baseline_eval"

            def fake_generate(**kwargs: object) -> dict:
                predictions_dir = Path(kwargs["output_dir"])
                predictions_dir.mkdir(parents=True, exist_ok=True)
                profiles = {}
                for name in ["cv_heading", "physics_oracle"]:
                    path = predictions_dir / f"{name}.json"
                    path.write_text("[]", encoding="utf-8")
                    profiles[name] = str(path)
                return {"profiles": profiles, "case_count": 1, "horizon_s": 1.0, "output_dir": str(predictions_dir)}

            def fake_eval(**kwargs: object) -> dict:
                eval_dir = Path(kwargs["output_dir"])
                eval_dir.mkdir(parents=True, exist_ok=True)
                (eval_dir / "world_model_metrics.json").write_text(
                    json.dumps(
                        {
                            "profile_name": kwargs["profile_name"],
                            "overview": {
                                "case_count": 1,
                                "full_horizon_count": 1,
                                "full_horizon_rate": 1.0,
                                "mean_horizon_recall": 1.0,
                                "mean_ade_m": 0.0,
                                "mean_fde_m": 0.0,
                                "mean_occupancy_iou": 1.0,
                                "mean_primary_actor_iou": 1.0,
                                "mean_closest_approach_distance_error_m": 0.0,
                                "mean_closest_approach_time_error_s": 0.0,
                                "mean_risk_fidelity_score": 1.0,
                            },
                            "behavior_breakdown": [],
                            "track_breakdown": [],
                            "risk_breakdowns": {},
                        }
                    ),
                    encoding="utf-8",
                )
                return {"output_dir": str(eval_dir), "overview": {"case_count": 1}}

            def fake_compare(evaluation_dirs: object, output_dir: object) -> dict:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "world_model_comparison.json").write_text(
                    json.dumps({"overview": {"profile_count": 2, "case_count": 1}, "profiles": []}),
                    encoding="utf-8",
                )
                return {"output_dir": str(output_dir), "profile_count": 2, "case_count": 1}

            with mock.patch(
                "nusc_scene_agent.world_model_benchmark.generate_nuscenes_forecast_baselines", side_effect=fake_generate
            ), mock.patch(
                "nusc_scene_agent.world_model_benchmark.adapt_and_evaluate_nuscenes_forecast_predictions",
                side_effect=fake_eval,
            ), mock.patch(
                "nusc_scene_agent.world_model_benchmark.compare_world_model_evaluations",
                side_effect=fake_compare,
            ):
                metadata = run_nuscenes_forecast_baselines(
                    benchmark_path=benchmark_path,
                    dataroot=root / "data",
                    version="v1.0-trainval",
                    output_dir=output_dir,
                    mode_selection="top_probability",
                )

            self.assertEqual(metadata["comparison"]["profile_count"], 2)
            self.assertTrue((output_dir / "comparison" / "world_model_comparison.json").exists())

    def test_export_world_model_replay_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = self._create_world_model_benchmark(root)
            replay_dir = root / "replay"
            metadata = export_world_model_replay(benchmark_path, replay_dir, export_format="jsonl")

            self.assertEqual(metadata["case_count"], 1)
            manifest = json.loads((replay_dir / "replay_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["metadata"]["export_format"], "jsonl")
            case_path = replay_dir / "crossing_anchor.jsonl"
            self.assertTrue(case_path.exists())
            lines = [json.loads(line) for line in case_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            topics = {line["topic"] for line in lines}
            self.assertIn("/nusc_scene_agent/history_track", topics)
            self.assertIn("/nusc_scene_agent/future_track", topics)
            self.assertIn("/nusc_scene_agent/future_occupancy", topics)


if __name__ == "__main__":
    unittest.main()
