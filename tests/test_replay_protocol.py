import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from nusc_scene_agent.bench2drive_closed_loop import (
    ClosedLoopControlConfig,
    _evaluate_closed_loop_rollout,
    _logged_route_from_rows,
    _polyline_length,
    _run_closed_loop_case,
    _render_case_rollout_figure,
    _select_closed_loop_cases,
)


def straight_rows(count=20):
    return [
        {"frame_id": index * 5, "clip_name": "straight", "scenario_family": "straight",
         "ego_state": {"x": 0.0, "y": -float(index), "theta": 0.0, "speed": 2.0},
         "future_waypoints_ego": [[0.0, -float(step)] for step in range(1, 6)]}
        for index in range(count)
    ]


class ReplayProtocolTest(unittest.TestCase):
    def test_perfect_replay_has_full_completion_without_future_tail(self):
        route = _logged_route_from_rows(straight_rows())
        self.assertEqual(_polyline_length(route), 19.0)
        metrics = _evaluate_closed_loop_rollout(closed_loop_states=route, logged_states=route, route_global=route)
        self.assertEqual(metrics["route_completion"], 1.0)
        self.assertEqual(metrics["closed_loop_ade_m"], 0.0)
        self.assertEqual(metrics["closed_loop_score"], 1.0)

    def test_integrated_states_match_next_observation_time(self):
        class Dataset:
            def __init__(self, *args, **kwargs):
                pass

            def __getitem__(self, index):
                return {"images": torch.zeros(1, 3, 8, 8), "route": torch.zeros(8),
                        "future": torch.zeros(10), "control": torch.zeros(3), "brake": torch.zeros(1)}

        class Model:
            def __call__(self, images, route):
                count = images.shape[0]
                return {"future": torch.tensor([[0.0, -float(i)] for i in range(1, 6)]).repeat(count, 1, 1),
                        "control": torch.zeros(count, 3), "brake_logits": torch.full((count, 1), -10.0)}

        with patch("nusc_scene_agent.bench2drive_closed_loop._Bench2DriveVisionDataset", Dataset), patch(
            "nusc_scene_agent.bench2drive_closed_loop._control_from_predicted_waypoints",
            return_value={"steer": 0.0, "throttle": 0.0, "brake": 0.0},
        ) as controller:
            report = _run_closed_loop_case(torch=torch, model=Model(), case_rows=straight_rows(),
                                          device=torch.device("cpu"), image_size=8,
                                          config=ClosedLoopControlConfig(horizon_s=2.0))
        self.assertEqual(report["metrics"]["closed_loop_ade_m"], 0.0)
        self.assertEqual(report["metrics"]["route_completion"], 1.0)
        self.assertEqual([row["t_s"] for row in report["closed_loop_states"]], [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(controller.call_args.kwargs["waypoint_reference_state"]["x"], 3.0)
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "rollout.png"
            _render_case_rollout_figure(report, figure)
            self.assertGreater(figure.stat().st_size, 0)

    def test_balanced_selection_covers_families_before_repeating(self):
        rows = []
        for family, clips, density in [("busy", 5, 100), ("quiet", 2, 1), ("rare", 1, 0)]:
            for clip in range(clips):
                rows.extend({**row, "clip_name": f"{family}_{clip}", "scenario_family": family,
                             "object_count": density} for row in straight_rows(2))
        selected = _select_closed_loop_cases(rows, max_cases=3, max_frames_per_clip=2, case_selection="balanced")
        self.assertEqual({case[0]["scenario_family"] for case in selected}, {"busy", "quiet", "rare"})


if __name__ == "__main__":
    unittest.main()
