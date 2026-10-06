"""
Generate sample BEV visualization using academic style.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nusc_scene_agent.models import ValidatedCase, CandidateAnchor
import pandas as pd
import numpy as np

# Create synthetic case for demonstration
def create_demo_case():
    """Create a synthetic scenario for visualization demo."""

    # Synthetic candidate
    candidate = CandidateAnchor(
        sample_token="demo_sample",
        instance_token="target_vehicle",
        behavior_label="cut_in_from_right",
        ego_width=1.8,
        ego_length=4.5,
        score=0.95,
    )

    # Synthetic timeline
    timeline_data = []
    for i in range(20):
        t = i * 0.5  # 0.5s intervals

        # Ego vehicle moving forward
        timeline_data.append({
            "sample_token": f"sample_{i}",
            "instance_token": "ego",
            "ego_x": t * 5.0,
            "ego_y": 0.0,
            "ego_yaw": 0.0,
            "x": t * 5.0,
            "y": 0.0,
            "yaw": 0.0,
            "width": 1.8,
            "length": 4.5,
        })

        # Target vehicle cutting in from right
        cut_in_progress = min(i / 10.0, 1.0)
        timeline_data.append({
            "sample_token": f"sample_{i}",
            "instance_token": "target_vehicle",
            "ego_x": t * 5.0,
            "ego_y": 0.0,
            "ego_yaw": 0.0,
            "x": t * 5.0 + 10.0 + cut_in_progress * 5.0,
            "y": 3.5 - cut_in_progress * 3.0,  # Moving from right lane to ego lane
            "yaw": -0.3 * cut_in_progress,
            "width": 1.9,
            "length": 4.8,
        })

        # Context vehicle ahead
        timeline_data.append({
            "sample_token": f"sample_{i}",
            "instance_token": "context_1",
            "ego_x": t * 5.0,
            "ego_y": 0.0,
            "ego_yaw": 0.0,
            "x": t * 5.0 + 25.0,
            "y": 0.0,
            "yaw": 0.0,
            "width": 1.8,
            "length": 4.6,
        })

    timeline = pd.DataFrame(timeline_data)

    # Synthetic ego window (past 2s)
    ego_window = pd.DataFrame({
        "ego_x": np.linspace(-10, 0, 5),
        "ego_y": np.zeros(5),
    })

    # Context agents at anchor frame
    context_agents = timeline[
        (timeline["sample_token"] == "sample_10") &
        (timeline["instance_token"] != "target_vehicle")
    ].copy()

    # Synthetic map geometries (simple lane structure)
    map_geometries = {
        "lane": [
            # Left lane
            np.array([[-20, -3.5], [40, -3.5], [40, 0], [-20, 0]]),
            # Right lane (where cut-in comes from)
            np.array([[-20, 0], [40, 0], [40, 3.5], [-20, 3.5]]),
        ],
        "drivable_area": [
            np.array([[-20, -3.5], [40, -3.5], [40, 3.5], [-20, 3.5]]),
        ],
        "ped_crossing": [
            np.array([[15, -3.5], [18, -3.5], [18, 3.5], [15, 3.5]]),
        ],
    }

    case = ValidatedCase(
        candidate=candidate,
        timeline=timeline,
        ego_window=ego_window,
        context_agents=context_agents,
        map_geometries=map_geometries,
        validation_scores={},
        metadata={},
    )

    return case


if __name__ == "__main__":
    from academic_visualization import render_bev_evidence

    case = create_demo_case()
    output_path = Path("assets/bev_scenario_example.png")
    render_bev_evidence(case, output_path)
    print(f"✓ Generated: {output_path}")
