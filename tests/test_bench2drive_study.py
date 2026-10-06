import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nusc_scene_agent.bench2drive_e2e import VisionE2EModelConfig, _build_vision_e2e_model
from nusc_scene_agent.bench2drive_study import _evaluate_study_run, run_bench2drive_study, select_training_rows


class ControlledStudyTest(unittest.TestCase):
    def test_evaluation_cache_requires_matching_checkpoint_and_intact_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint = root / "manifest.jsonl", root / "model.pt"
            manifest.write_text("{}\n")
            checkpoint.write_bytes(b"first checkpoint")
            def evaluate(_manifest, _checkpoint, output, **_kwargs):
                output.mkdir(exist_ok=True)
                (output / "evaluation_report.json").write_text("{}")
                (output / "predictions.jsonl").write_text("{}\n")
            def replay(_manifest, _checkpoint, output, **_kwargs):
                output.mkdir(exist_ok=True)
                (output / "closed_loop_report.json").write_text("{}")
            with patch("nusc_scene_agent.bench2drive_study.evaluate_vision_e2e_planner", side_effect=evaluate) as evaluator, \
                    patch("nusc_scene_agent.bench2drive_study.run_bench2drive_vision_closed_loop", side_effect=replay):
                _evaluate_study_run(manifest, checkpoint, root, device="cpu", image_size=160)
                _evaluate_study_run(manifest, checkpoint, root, device="cpu", image_size=160)
                self.assertEqual(evaluator.call_count, 1)
                checkpoint.write_bytes(b"second checkpoint")
                _evaluate_study_run(manifest, checkpoint, root, device="cpu", image_size=160)
                self.assertEqual(evaluator.call_count, 2)
                (root / "eval_test/predictions.jsonl").write_text("truncated")
                _evaluate_study_run(manifest, checkpoint, root, device="cpu", image_size=160)
                self.assertEqual(evaluator.call_count, 3)
                (root / "evaluation_complete.json").write_text("{")
                _evaluate_study_run(manifest, checkpoint, root, device="cpu", image_size=160)
                self.assertEqual(evaluator.call_count, 4)

    def test_partial_seed_worker_cannot_publish_a_complete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "partial seed worker"):
                run_bench2drive_study({"output": str(root), "manifest": str(manifest),
                                      "seeds": [7, 17, 27], "arms": [{"name": "baseline"}]}, selected_seeds=[27])

    def test_sampling_keeps_budget_and_held_out_rows(self):
        rows = [{"split": "train", "frame_id": i, "scenario_family": "PedestrianCrossing" if i < 20 else "Unmapped"}
                for i in range(100)] + [{"split": "val", "frame_id": 101}, {"split": "test", "frame_id": 102}]
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "cases.json"
            library.write_text(json.dumps([{"passed": True, "matched_behaviors": ["crossing"]}]))
            uniform, _ = select_training_rows(rows, fraction=0.5, seed=7, policy="uniform")
            selected, report = select_training_rows(rows, fraction=0.5, seed=7, policy="risk_prior", case_library=library)
        self.assertEqual(len(uniform), len(selected))
        self.assertEqual(selected[-2:], rows[-2:])
        self.assertEqual(report["training_budget"], 50)
        self.assertEqual(report["mined_family_prior"], {"vru_crossing": 1})
        self.assertGreater(sum(row.get("scenario_family") == "PedestrianCrossing" for row in selected),
                           sum(row.get("scenario_family") == "PedestrianCrossing" for row in uniform))

    def test_route_only_checkpoint_cannot_use_image_content(self):
        torch.set_num_threads(2)
        model = _build_vision_e2e_model(VisionE2EModelConfig(
            model_size="tiny", architecture="trajectory_transformer", input_mode="route_only", trajectory_modes=4,
        )).eval()
        route = torch.rand(2, 8)
        with torch.no_grad():
            first = model(torch.rand(2, 6, 3, 32, 32), route)
            second = model(torch.randn(2, 6, 3, 32, 32), route)
        for key in ["future", "control", "brake_logits"]:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_spatial_resolution_preserves_prediction_contract(self):
        model = _build_vision_e2e_model(VisionE2EModelConfig(
            model_size="tiny", architecture="trajectory_transformer", spatial_pool_size=4, trajectory_modes=4,
        )).eval()
        with torch.no_grad():
            output = model(torch.rand(2, 6, 3, 32, 32), torch.rand(2, 8))
        self.assertEqual(tuple(output["future"].shape), (2, 10))
        self.assertEqual(model.spatial_embedding.shape[0], 16)
        self.assertTrue(torch.isfinite(output["future"]).all())
