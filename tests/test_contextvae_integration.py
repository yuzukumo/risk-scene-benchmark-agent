import math
import unittest
from unittest.mock import Mock

import numpy as np
from shapely.geometry import MultiLineString

from nusc_scene_agent.contextvae_integration import _CompatibleMapExplorer, _case_file_name, _contextvae_local_to_global, _prepare_case_records


class ContextVAEIntegrationTest(unittest.TestCase):
    def test_map_mask_renders_every_segment_with_shapely_two(self) -> None:
        lines = MultiLineString([[(2, 2), (8, 2)], [(2, 8), (8, 8)]])
        mask = _CompatibleMapExplorer.mask_for_lines(lines, np.zeros((12, 12), dtype=np.uint8))
        self.assertEqual(mask[2, 5], 1)
        self.assertEqual(mask[8, 5], 1)
        self.assertEqual(mask[5, 5], 0)

    def test_preparation_requires_the_actor_throughout_the_forecast(self) -> None:
        tables = {"sample": {}, "sample_annotation": {}, "scene": {"scene": {"name": "scene", "token": "scene", "log_token": "log"}},
                  "log": {"log": {"location": "map"}}, "sample_data": {"camera": {"ego_pose_token": "ego"}},
                  "ego_pose": {"ego": {"translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}}}
        for index in range(17):
            tables["sample"][f"s{index}"] = {"token": f"s{index}", "scene_token": "scene", "anns": [f"a{index}"],
                                                "data": {"CAM_FRONT": "camera"},
                                                "prev": f"s{index - 1}" if index else "",
                                                "next": f"s{index + 1}" if index < 16 else ""}
            tables["sample_annotation"][f"a{index}"] = {"category_name": "vehicle.car", "attribute_tokens": [],
                                                         "instance_token": "actor", "translation": [index, 0, 0],
                                                         "rotation": [1, 0, 0, 0]}
        nusc = Mock()
        nusc.get.side_effect = lambda table, token: tables[table][token]
        case = {"instance_token": "actor", "rollout_anchor_sample_token": "s4"}
        self.assertIsNotNone(_prepare_case_records(nusc, case, ob_horizon=5, pred_horizon=12))
        tables["sample"]["s10"]["anns"] = []
        self.assertIsNone(_prepare_case_records(nusc, case, ob_horizon=5, pred_horizon=12))

    def test_case_file_name_uses_rollout_anchor(self) -> None:
        name = _case_file_name(
            {
                "instance_token": "instance-123",
                "rollout_anchor_sample_token": "sample-456",
            }
        )
        self.assertEqual(name, "instance-123_sample-456")

    def test_contextvae_local_to_global_identity(self) -> None:
        restored = _contextvae_local_to_global(
            trajectory=[[10.0, 5.0], [12.0, 7.0]],
            origin_x=10.0,
            origin_y=5.0,
            heading_rad=0.0,
        )
        self.assertEqual(restored, [[10.0, 5.0], [12.0, 7.0]])

    def test_contextvae_local_to_global_rotates_back_to_global_frame(self) -> None:
        restored = _contextvae_local_to_global(
            trajectory=[[11.0, 5.0], [12.0, 5.0]],
            origin_x=10.0,
            origin_y=5.0,
            heading_rad=math.pi / 2.0,
        )
        self.assertEqual(restored, [[10.0, 6.0], [10.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
