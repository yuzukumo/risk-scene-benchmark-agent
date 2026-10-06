from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from jinja2 import Template
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.models.physics import ConstantVelocityHeading, PhysicsOracle

from nusc_scene_agent.geometry import ego_xy_to_global, global_xy_to_anchor_ego
from nusc_scene_agent.perception_benchmark import CATEGORY_ALIASES, RISK_FACET_FIELDS


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _metric_mean(values: Iterable[object]) -> float | None:
    finite = [number for value in values if (number := _optional_float(value)) is not None]
    return round(mean(finite), 4) if finite else None


def _format_metric(value: object) -> str:
    number = _optional_float(value)
    return f"{number:.3f}" if number is not None else "n/a"


def _profile_sort_key(row: Dict[str, object]) -> tuple:
    errors = [_optional_float(row.get(name)) for name in ["mean_min_ade_at_5", "mean_ade_m"]]
    return (float(row.get("mean_risk_fidelity_score") or 0.0), float(row["full_horizon_rate"]),
            *[-value if value is not None else -math.inf for value in errors],
            float(row.get("mean_occupancy_iou") or 0.0))


WORLD_MODEL_SUMMARY_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Scenario-Conditioned World-Model Evaluation</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 32px; background: #f5f1ea; color: #1f1f1f; }
    h1, h2 { margin-bottom: 0.4rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; background: white; }
    th, td { border-bottom: 1px solid #ddd4c7; padding: 10px 12px; text-align: left; vertical-align: top; }
    th { background: #ede5d8; }
    .meta { color: #665d54; margin-bottom: 18px; }
  </style>
</head>
<body>
  <h1>Scenario-Conditioned World-Model Evaluation</h1>
  <div class="meta">
    Profile: {{ summary.profile_name }} |
    Cases: {{ summary.overview.case_count }} |
    Full Horizon: {{ summary.overview.full_horizon_count }}/{{ summary.overview.case_count }} ({{ "%.1f"|format(summary.overview.full_horizon_rate * 100.0) }}%) |
    Mean Risk Fidelity: {{ "%.3f"|format(summary.overview.mean_risk_fidelity_score) }}
  </div>

  <h2>Overview</h2>
  <table>
    <tbody>
      <tr><th>Mean Horizon Recall</th><td>{{ "%.3f"|format(summary.overview.mean_horizon_recall) }}</td></tr>
      <tr><th>Mean ADE</th><td>{{ metric(summary.overview.mean_ade_m) }}</td></tr>
      <tr><th>Mean FDE</th><td>{{ metric(summary.overview.mean_fde_m) }}</td></tr>
      <tr><th>Mean Occupancy IoU</th><td>{{ "%.3f"|format(summary.overview.mean_occupancy_iou) }}</td></tr>
      <tr><th>Mean Primary Actor IoU</th><td>{{ "%.3f"|format(summary.overview.mean_primary_actor_iou) }}</td></tr>
      <tr><th>Mean Closest-Approach Distance Error</th><td>{{ metric(summary.overview.mean_closest_approach_distance_error_m) }}</td></tr>
      <tr><th>Mean Closest-Approach Time Error</th><td>{{ metric(summary.overview.mean_closest_approach_time_error_s) }}</td></tr>
      <tr><th>Perfect Cases</th><td>{{ summary.overview.perfect_case_count }}</td></tr>
    </tbody>
  </table>

  <h2>Forecast Metrics</h2>
  <table>
    <tbody>
      <tr><th>Mean Mode Count</th><td>{{ "%.2f"|format(summary.forecast_metrics.mean_mode_count) }}</td></tr>
      <tr><th>Mean MinADE@1</th><td>{{ metric(summary.forecast_metrics.mean_min_ade_at_1) }}</td></tr>
      <tr><th>Mean MinADE@5</th><td>{{ metric(summary.forecast_metrics.mean_min_ade_at_5) }}</td></tr>
      <tr><th>Mean MinFDE@1</th><td>{{ metric(summary.forecast_metrics.mean_min_fde_at_1) }}</td></tr>
      <tr><th>Mean MinFDE@5</th><td>{{ metric(summary.forecast_metrics.mean_min_fde_at_5) }}</td></tr>
      <tr><th>Mean MissRate@1</th><td>{{ "%.3f"|format(summary.forecast_metrics.mean_miss_rate_at_1) }}</td></tr>
      <tr><th>Mean MissRate@5</th><td>{{ "%.3f"|format(summary.forecast_metrics.mean_miss_rate_at_5) }}</td></tr>
    </tbody>
  </table>

  <h2>Behavior Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Behavior</th><th>Cases</th><th>Full Horizon</th><th>Risk Fidelity</th><th>ADE</th><th>Occupancy IoU</th><th>Top Failure Modes</th>
      </tr>
    </thead>
    <tbody>
    {% for row in summary.behavior_breakdown %}
      <tr>
        <td>{{ row.behavior }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.full_horizon_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_horizon_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_risk_fidelity_score) }}</td>
        <td>{{ metric(row.mean_ade_m) }}</td>
        <td>{{ "%.3f"|format(row.mean_occupancy_iou) }}</td>
        <td>{{ row.top_failure_summary }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Challenge Track Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Track</th><th>Cases</th><th>Full Horizon</th><th>Risk Fidelity</th><th>ADE</th><th>Occupancy IoU</th><th>Top Failure Modes</th>
      </tr>
    </thead>
    <tbody>
    {% for row in summary.track_breakdown %}
      <tr>
        <td>{{ row.track }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.full_horizon_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_horizon_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_risk_fidelity_score) }}</td>
        <td>{{ metric(row.mean_ade_m) }}</td>
        <td>{{ "%.3f"|format(row.mean_occupancy_iou) }}</td>
        <td>{{ row.top_failure_summary }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  {% for facet_name, rows in summary.risk_breakdowns.items() %}
  <h2>{{ facet_name|replace('_', ' ')|title }}</h2>
  <table>
    <thead>
      <tr>
        <th>Group</th><th>Cases</th><th>Full Horizon</th><th>Risk Fidelity</th><th>ADE</th><th>Occupancy IoU</th><th>Top Failure Modes</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row[facet_name] }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.full_horizon_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_horizon_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_risk_fidelity_score) }}</td>
        <td>{{ metric(row.mean_ade_m) }}</td>
        <td>{{ "%.3f"|format(row.mean_occupancy_iou) }}</td>
        <td>{{ row.top_failure_summary }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endfor %}
</body>
</html>
"""
)


WORLD_MODEL_COMPARISON_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Scenario-Conditioned World-Model Comparison</title>
  <style>
    body { font-family: Helvetica, Arial, sans-serif; margin: 32px; background: #f5f1ea; color: #1f1f1f; }
    h1, h2 { margin-bottom: 0.4rem; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; background: white; }
    th, td { border-bottom: 1px solid #ddd4c7; padding: 10px 12px; text-align: left; vertical-align: top; }
    th { background: #ede5d8; }
    .meta { color: #665d54; margin-bottom: 18px; }
  </style>
</head>
<body>
  <h1>Scenario-Conditioned World-Model Comparison</h1>
  <div class="meta">
    Profiles: {{ comparison.overview.profile_count }} |
    Cases: {{ comparison.overview.case_count }}
  </div>

  <h2>Profile Overview</h2>
  <table>
    <thead>
      <tr>
        <th>Profile</th><th>Full Horizon</th><th>Risk Fidelity</th><th>ADE</th><th>MinADE@1</th><th>MinADE@5</th><th>MissRate@5</th><th>Occupancy IoU</th><th>Closest-Approach Time Error</th>
      </tr>
    </thead>
    <tbody>
    {% for row in comparison.profiles %}
      <tr>
        <td>{{ row.label }}</td>
        <td>{{ row.full_horizon_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_horizon_rate * 100.0) }}%)</td>
        <td>{{ metric(row.mean_risk_fidelity_score) }}</td>
        <td>{{ metric(row.mean_ade_m) }}</td>
        <td>{{ metric(row.mean_min_ade_at_1) }}</td>
        <td>{{ metric(row.mean_min_ade_at_5) }}</td>
        <td>{{ metric(row.mean_miss_rate_at_5) }}</td>
        <td>{{ metric(row.mean_occupancy_iou) }}</td>
        <td>{{ metric(row.mean_closest_approach_time_error_s) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Behavior Matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Behavior</th>
        {% for row in comparison.profiles %}
        <th>{{ row.label }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
    {% for row in comparison.behavior_matrix %}
      <tr>
        <td>{{ row.behavior }}</td>
        {% for cell in row.cells %}
        <td>{{ "%.3f"|format(cell.mean_risk_fidelity_score) }} | {{ "%.3f"|format(cell.mean_occupancy_iou) }}</td>
        {% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Challenge Track Matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Track</th>
        {% for row in comparison.profiles %}
        <th>{{ row.label }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
    {% for row in comparison.track_matrix %}
      <tr>
        <td>{{ row.track }}</td>
        {% for cell in row.cells %}
        <td>{{ "%.3f"|format(cell.mean_risk_fidelity_score) }} | {{ "%.3f"|format(cell.mean_occupancy_iou) }}</td>
        {% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


WORLD_MODEL_PROXY_PROFILES = [
    "oracle_rollout",
    "kinematic_rollout",
    "risk_underreach_rollout",
]

NUSCENES_FORECAST_MODE_SELECTIONS = [
    "top_probability",
    "oracle_ade",
    "oracle_fde",
]

NUSCENES_FORECAST_BASELINE_PROFILES = [
    "cv_heading",
    "physics_oracle",
]

FORECAST_TOPK_REPORTS = [1, 5, 10]
FORECAST_MISS_TOLERANCE_M = 2.0

DEFAULT_GRID_SPEC = {
    "x_min_m": -24.0,
    "x_max_m": 24.0,
    "y_min_m": -24.0,
    "y_max_m": 24.0,
    "resolution_m": 1.0,
    "dilation_radius_cells": 1,
}

CHALLENGE_TRACK_DESCRIPTIONS = {
    "challenge/critical_range": "Cases whose minimum distance falls in the critical range.",
    "challenge/crossing_emergence": "Crossing cases that stress late emergence and crosswalk interactions.",
    "challenge/generic_risk_slice": "Cases that do not map to a more specific challenge track.",
    "challenge/large_lead_occluder": "Lead-vehicle cases with large-vehicle occlusion cues.",
    "challenge/lateral_merge": "Cut-in or lateral merge interactions near the ego path.",
    "challenge/opposite_direction_conflict": "Oncoming interactions with opposite-direction conflict.",
    "challenge/shared_lane_lead": "Shared-lane lead interactions with lane-supported geometry.",
    "challenge/visibility_limited": "Cases with limited visibility or partial observability cues.",
}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_category(value: str) -> str:
    key = str(value or "").strip().lower()
    return CATEGORY_ALIASES.get(key, key)


def _unique_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _distance_m(lhs: Dict[str, object], rhs: Dict[str, object]) -> float:
    return math.hypot(
        _safe_float(lhs.get("x_ego")) - _safe_float(rhs.get("x_ego")),
        _safe_float(lhs.get("y_ego")) - _safe_float(rhs.get("y_ego")),
    )


def _sorted_unique_cells(cells: Iterable[Tuple[int, int]]) -> List[List[int]]:
    ordered = sorted({(int(x), int(y)) for x, y in cells}, key=lambda item: (item[0], item[1]))
    return [[int(x), int(y)] for x, y in ordered]


def _frame_cell_set(frame: Dict[str, object], key: str) -> set[Tuple[int, int]]:
    return {
        (int(cell[0]), int(cell[1]))
        for cell in list(frame.get(key) or [])
        if isinstance(cell, (list, tuple)) and len(cell) >= 2
    }


def _build_union_cells(primary_cells: Sequence[Sequence[int]], context_cells: Sequence[Sequence[int]]) -> List[List[int]]:
    return _sorted_unique_cells(
        [(int(cell[0]), int(cell[1])) for cell in list(primary_cells) + list(context_cells) if len(cell) >= 2]
    )


def _cell_for_point(x_ego: float, y_ego: float, grid_spec: Dict[str, float]) -> Optional[Tuple[int, int]]:
    x_min = float(grid_spec["x_min_m"])
    x_max = float(grid_spec["x_max_m"])
    y_min = float(grid_spec["y_min_m"])
    y_max = float(grid_spec["y_max_m"])
    resolution = float(grid_spec["resolution_m"])
    if not (x_min <= x_ego < x_max and y_min <= y_ego < y_max):
        return None
    x_idx = int(math.floor((x_ego - x_min) / resolution))
    y_idx = int(math.floor((y_ego - y_min) / resolution))
    return (x_idx, y_idx)


def _dilate_cells(cells: Iterable[Tuple[int, int]], radius_cells: int) -> set[Tuple[int, int]]:
    expanded: set[Tuple[int, int]] = set()
    for x_idx, y_idx in cells:
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                expanded.add((int(x_idx + dx), int(y_idx + dy)))
    return expanded


def _rasterize_points(points: Sequence[Tuple[float, float]], grid_spec: Dict[str, float]) -> List[List[int]]:
    base_cells = []
    for x_ego, y_ego in points:
        cell = _cell_for_point(float(x_ego), float(y_ego), grid_spec)
        if cell is not None:
            base_cells.append(cell)
    expanded = _dilate_cells(base_cells, int(grid_spec["dilation_radius_cells"]))
    return _sorted_unique_cells(expanded)


def _bounded_score(error: Optional[float], scale: float) -> float:
    if error is None:
        return 0.0
    return max(0.0, 1.0 - float(error) / float(scale))


def _split_history_future(case: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    frames = list(case.get("frames") or [])
    anchor_sample_idx = int(case["anchor_sample_idx"])
    history = [dict(frame) for frame in frames if _safe_int(frame.get("sample_idx")) <= anchor_sample_idx]
    future = [dict(frame) for frame in frames if _safe_int(frame.get("sample_idx")) > anchor_sample_idx]
    return history, future


def _duration_s(frames: Sequence[Dict[str, object]]) -> float:
    if len(frames) < 2:
        return 0.0
    start = _safe_int(frames[0].get("timestamp_us"))
    end = _safe_int(frames[-1].get("timestamp_us"))
    return round(max(0.0, float(end - start) / 1_000_000.0), 3)


def _motion_targets(history_frames: Sequence[Dict[str, object]], future_frames: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not history_frames or not future_frames:
        return {
            "horizon_s": 0.0,
            "path_length_m": 0.0,
            "endpoint_displacement_m": 0.0,
            "min_future_distance_m": None,
            "closest_approach_sample_idx": None,
            "closest_approach_offset_s": None,
        }
    anchor = dict(history_frames[-1])
    future_rows = [dict(frame) for frame in future_frames]
    all_points = [anchor] + future_rows
    path_length = 0.0
    for lhs, rhs in zip(all_points[:-1], all_points[1:]):
        path_length += _distance_m(lhs, rhs)
    closest = min(future_rows, key=lambda row: _safe_float(row.get("distance"), float("inf")))
    return {
        "horizon_s": _duration_s([anchor] + future_rows),
        "path_length_m": round(path_length, 4),
        "endpoint_displacement_m": round(_distance_m(anchor, future_rows[-1]), 4),
        "min_future_distance_m": round(_safe_float(closest.get("distance")), 4),
        "closest_approach_sample_idx": _safe_int(closest.get("sample_idx")),
        "closest_approach_offset_s": round(
            max(0.0, (_safe_int(closest.get("timestamp_us")) - _safe_int(anchor.get("timestamp_us"))) / 1_000_000.0),
            4,
        ),
    }


def _challenge_tracks(case: Dict[str, object]) -> List[str]:
    risk_facets = dict(case.get("risk_facets") or {})
    primary_behavior = str(case.get("primary_behavior") or "")
    tracks: List[str] = []

    if str(risk_facets.get("distance_band") or "") == "critical_range":
        tracks.append("challenge/critical_range")
    if str(risk_facets.get("occlusion_proxy") or "") == "visibility_limited":
        tracks.append("challenge/visibility_limited")
    if str(risk_facets.get("occlusion_proxy") or "") == "large_lead_occluder":
        tracks.append("challenge/large_lead_occluder")
    if primary_behavior == "crossing":
        tracks.append("challenge/crossing_emergence")
    if primary_behavior == "cut_in":
        tracks.append("challenge/lateral_merge")
    if primary_behavior == "oncoming":
        tracks.append("challenge/opposite_direction_conflict")
    if primary_behavior == "stopped_lead" or (
        primary_behavior == "proximity" and str(risk_facets.get("map_relation") or "") == "shared_lane_supported"
    ):
        tracks.append("challenge/shared_lane_lead")
    if not tracks:
        tracks.append("challenge/generic_risk_slice")
    return _unique_strings(tracks)


def _build_track_catalog(cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    counts: Dict[str, int] = defaultdict(int)
    for case in cases:
        for track in list(case.get("challenge_tracks") or []):
            counts[str(track)] += 1
    rows = []
    for track in sorted(counts):
        rows.append(
            {
                "track": track,
                "case_count": int(counts[track]),
                "description": CHALLENGE_TRACK_DESCRIPTIONS.get(track, ""),
            }
        )
    return rows


def _build_context_index(conn: sqlite3.Connection, sample_tokens: Sequence[str]) -> Dict[str, List[Dict[str, object]]]:
    if not sample_tokens:
        return {}
    placeholders = ", ".join("?" for _ in sample_tokens)
    rows = conn.execute(
        """
        SELECT sample_token, sample_idx, instance_token, category_group, x_ego, y_ego
        FROM agents
        WHERE sample_token IN ({0})
        """.format(placeholders),
        tuple(sample_tokens),
    ).fetchall()
    by_sample: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_token"])].append(
            {
                "sample_token": str(row["sample_token"]),
                "sample_idx": _safe_int(row["sample_idx"]),
                "instance_token": str(row["instance_token"]),
                "category_group": _normalize_category(str(row["category_group"])),
                "x_ego": _safe_float(row["x_ego"]),
                "y_ego": _safe_float(row["y_ego"]),
            }
        )
    return by_sample


def _build_sample_pose_index(conn: sqlite3.Connection, sample_tokens: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not sample_tokens:
        return {}
    placeholders = ", ".join("?" for _ in sample_tokens)
    rows = conn.execute(
        """
        SELECT sample_token, ego_x, ego_y, ego_yaw
        FROM samples
        WHERE sample_token IN ({0})
        """.format(placeholders),
        tuple(sample_tokens),
    ).fetchall()
    return {
        str(row["sample_token"]): {
            "ego_x": _safe_float(row["ego_x"]),
            "ego_y": _safe_float(row["ego_y"]),
            "ego_yaw": _safe_float(row["ego_yaw"]),
        }
        for row in rows
    }


def _attach_global_frame_geometry(
    frames: Sequence[Dict[str, object]],
    sample_pose_index: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    enriched = []
    for frame in frames:
        sample_token = str(frame["sample_token"])
        pose = dict(sample_pose_index.get(sample_token) or {})
        row = dict(frame)
        if pose:
            global_xy = ego_xy_to_global(
                [_safe_float(frame["x_ego"]), _safe_float(frame["y_ego"])],
                [float(pose["ego_x"]), float(pose["ego_y"])],
                float(pose["ego_yaw"]),
            )
            row["ego_x_global"] = round(float(pose["ego_x"]), 4)
            row["ego_y_global"] = round(float(pose["ego_y"]), 4)
            row["ego_yaw"] = round(float(pose["ego_yaw"]), 6)
            row["x_global"] = round(float(global_xy[0]), 4)
            row["y_global"] = round(float(global_xy[1]), 4)
        enriched.append(row)
    return enriched


def _convert_frames_to_anchor_ego(
    frames: Sequence[Dict[str, object]],
    anchor_frame: Mapping[str, object],
) -> List[Dict[str, object]]:
    anchor_x = _safe_float(anchor_frame.get("ego_x_global"))
    anchor_y = _safe_float(anchor_frame.get("ego_y_global"))
    anchor_yaw = _safe_float(anchor_frame.get("ego_yaw"))
    converted = []
    for frame in frames:
        row = dict(frame)
        if row.get("x_global") is not None and row.get("y_global") is not None:
            local_xy = global_xy_to_anchor_ego(
                np.asarray([[_safe_float(row["x_global"]), _safe_float(row["y_global"])]], dtype=float),
                [anchor_x, anchor_y],
                anchor_yaw,
            )[0]
            row["x_current_ego"] = row.get("x_ego")
            row["y_current_ego"] = row.get("y_ego")
            row["current_ego_yaw"] = row.get("ego_yaw")
            row["x_ego"] = round(float(local_xy[0]), 4)
            row["y_ego"] = round(float(local_xy[1]), 4)
            row["ego_yaw"] = round(anchor_yaw, 6)
            row["x_anchor_ego"] = row["x_ego"]
            row["y_anchor_ego"] = row["y_ego"]
        row["coordinate_frame"] = "rollout_anchor_ego"
        row["anchor_ego_x_global"] = round(anchor_x, 4)
        row["anchor_ego_y_global"] = round(anchor_y, 4)
        row["anchor_ego_yaw"] = round(anchor_yaw, 6)
        converted.append(row)
    return converted


def generate_world_model_benchmark_from_perception_benchmark(
    perception_benchmark_path: Path,
    db_path: Path,
    output_path: Path,
    grid_spec: Optional[Dict[str, float]] = None,
    *,
    history_s: float = 2.0,
    future_s: float = 6.0,
    min_future_s: float = 0.0,
) -> Dict[str, object]:
    perception_payload = _load_json(perception_benchmark_path)
    grid = dict(DEFAULT_GRID_SPEC)
    if grid_spec:
        grid.update({key: float(value) for key, value in grid_spec.items()})

    all_window_tokens: List[str] = []
    split_cases = []
    excluded = []
    if history_s < 0.0 or future_s <= 0.0 or not 0.0 <= min_future_s <= future_s:
        raise ValueError("Forecast durations require history >= 0 and 0 <= min_future <= future.")
    conn = sqlite3.connect(str(db_path.resolve()))
    conn.row_factory = sqlite3.Row
    try:
        for original in list(perception_payload.get("cases") or []):
            raw_case = dict(original)
            anchor = conn.execute("SELECT timestamp_us, sample_idx FROM samples WHERE sample_token = ?",
                                  (raw_case["anchor_sample_token"],)).fetchone()
            if anchor is None:
                excluded.append({"reference_case_key": raw_case["reference_case_key"], "reason": "missing_anchor_sample"})
                continue
            anchor_us = int(anchor["timestamp_us"])
            raw_case["anchor_sample_idx"] = int(anchor["sample_idx"])
            rows = conn.execute(
                "SELECT a.sample_token, a.sample_idx, s.timestamp_us, a.x_ego, a.y_ego, a.distance "
                "FROM agents a JOIN samples s ON s.sample_token = a.sample_token "
                "WHERE a.scene_token = ? AND a.instance_token = ? AND s.timestamp_us BETWEEN ? AND ? "
                "ORDER BY s.timestamp_us",
                (raw_case["scene_token"], raw_case["instance_token"],
                 anchor_us - int((history_s + 0.05) * 1e6), anchor_us + int((future_s + 0.05) * 1e6)),
            ).fetchall()
            raw_case["frames"] = [dict(row) for row in rows]
            history_frames, future_frames = _split_history_future(raw_case)
            if not history_frames or history_frames[-1]["sample_token"] != raw_case["anchor_sample_token"]:
                excluded.append({"reference_case_key": raw_case["reference_case_key"], "reason": "missing_anchor_actor"})
                continue
            duration = ((int(future_frames[-1]["timestamp_us"]) - int(history_frames[-1]["timestamp_us"])) / 1e6
                        if history_frames and future_frames else 0.0)
            if not future_frames or duration + 0.05 < min_future_s:
                excluded.append({"reference_case_key": raw_case["reference_case_key"],
                                 "reason": "insufficient_future_track", "available_future_s": duration})
                continue
            split_cases.append((raw_case, history_frames, future_frames))
            all_window_tokens.extend(str(frame["sample_token"]) for frame in history_frames + future_frames)
        all_window_tokens = list(dict.fromkeys(all_window_tokens))
        context_index = _build_context_index(conn, all_window_tokens)
        sample_pose_index = _build_sample_pose_index(conn, all_window_tokens)
    finally:
        conn.close()

    cases: List[Dict[str, object]] = []
    for raw_case, history_frames, future_frames in split_cases:
        history_frames = _attach_global_frame_geometry(history_frames, sample_pose_index)
        future_frames = _attach_global_frame_geometry(future_frames, sample_pose_index)
        history_frames = _convert_frames_to_anchor_ego(history_frames, history_frames[-1])
        future_frames = _convert_frames_to_anchor_ego(future_frames, history_frames[-1])
        future_occupancy = []
        for frame in future_frames:
            sample_token = str(frame["sample_token"])
            sample_rows = list(context_index.get(sample_token) or [])
            pose = dict(sample_pose_index.get(sample_token) or {})
            context_anchor_points = []
            for row in sample_rows:
                if str(row["instance_token"]) == str(raw_case["instance_token"]):
                    continue
                global_xy = ego_xy_to_global(
                    [_safe_float(row["x_ego"]), _safe_float(row["y_ego"])],
                    [float(pose.get("ego_x", 0.0)), float(pose.get("ego_y", 0.0))],
                    float(pose.get("ego_yaw", 0.0)),
                )
                anchor_xy = global_xy_to_anchor_ego(
                    np.asarray([[float(global_xy[0]), float(global_xy[1])]], dtype=float),
                    [_safe_float(history_frames[-1]["ego_x_global"]), _safe_float(history_frames[-1]["ego_y_global"])],
                    _safe_float(history_frames[-1]["ego_yaw"]),
                )[0]
                context_anchor_points.append((float(anchor_xy[0]), float(anchor_xy[1])))
            primary_cells = _rasterize_points(
                [(_safe_float(frame["x_ego"]), _safe_float(frame["y_ego"]))],
                grid,
            )
            context_cells = _rasterize_points(
                context_anchor_points,
                grid,
            )
            future_occupancy.append(
                {
                    "sample_token": sample_token,
                    "sample_idx": _safe_int(frame["sample_idx"]),
                    "timestamp_us": _safe_int(frame["timestamp_us"]),
                    "primary_actor_cells": primary_cells,
                    "context_cells": context_cells,
                    "occupied_cells": _build_union_cells(primary_cells, context_cells),
                    "coordinate_frame": "rollout_anchor_ego",
                    "context_actor_count": sum(
                        1 for row in sample_rows if str(row["instance_token"]) != str(raw_case["instance_token"])
                    ),
                }
            )

        case = {
            "benchmark_group": str(raw_case["benchmark_group"]),
            "reference_case_key": str(raw_case["reference_case_key"]),
            "reference_scene_name": str(raw_case.get("reference_scene_name") or ""),
            "scene_name": str(raw_case["scene_name"]),
            "scene_token": str(raw_case["scene_token"]),
            "instance_token": str(raw_case["instance_token"]),
            "category_name": str(raw_case["category_name"]),
            "category_group": str(raw_case["category_group"]),
            "location": str(raw_case["location"]),
            "primary_behavior": str(raw_case["primary_behavior"]),
            "behaviors": list(raw_case.get("behaviors") or []),
            "actors": list(raw_case.get("actors") or []),
            "tags": list(raw_case.get("tags") or []),
            "source_query_ids": list(raw_case.get("source_query_ids") or []),
            "source_query_texts": list(raw_case.get("source_query_texts") or []),
            "anchor_sample_token": str(raw_case["anchor_sample_token"]),
            "anchor_sample_idx": _safe_int(raw_case["anchor_sample_idx"]),
            "rollout_anchor_sample_token": str(history_frames[-1]["sample_token"]),
            "rollout_anchor_sample_idx": _safe_int(history_frames[-1]["sample_idx"]),
            "event_start_sample_idx": _safe_int(raw_case["event_start_sample_idx"]),
            "event_end_sample_idx": _safe_int(raw_case["event_end_sample_idx"]),
            "event_peak_sample_idx": _safe_int(raw_case["event_peak_sample_idx"]),
            "history_frame_count": len(history_frames),
            "future_frame_count": len(future_frames),
            "future_duration_s": (int(future_frames[-1]["timestamp_us"]) - int(history_frames[-1]["timestamp_us"])) / 1e6,
            "history_duration_s": _duration_s(history_frames),
            "history_frames": history_frames,
            "future_frames": future_frames,
            "future_occupancy": future_occupancy,
            "motion_targets": _motion_targets(history_frames, future_frames),
            "risk_facets": dict(raw_case.get("risk_facets") or {}),
            "map_support": dict(raw_case.get("map_support") or {}),
            "min_distance_m": raw_case.get("min_distance_m"),
            "min_ttc_s": raw_case.get("min_ttc_s"),
            "anchor_visibility": raw_case.get("anchor_visibility"),
            "coordinate_frame": "rollout_anchor_ego",
            "uses_future_ego_pose_for_targets": False,
        }
        case["challenge_tracks"] = _challenge_tracks(case)
        cases.append(case)

    output = {
        "metadata": {
            "generator": "world_model_benchmark_generator_v1",
            "source_perception_benchmark": str(perception_benchmark_path),
            "db_path": str(db_path),
            "case_count": len(cases),
            "requested_history_s": history_s,
            "requested_future_s": future_s,
            "minimum_future_s": min_future_s,
            "anchor_policy": "preserve mined anchor; load the actor track beyond the event window",
            "excluded_cases": excluded,
            "full_horizon_case_count": sum(case["future_duration_s"] >= future_s - 0.1 for case in cases),
            "grid_spec": grid,
            "challenge_tracks": _build_track_catalog(cases),
        },
        "cases": cases,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "case_count": len(cases)}


def _copy_future_trajectory(case: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        {
            "sample_token": str(frame["sample_token"]),
            "sample_idx": _safe_int(frame["sample_idx"]),
            "timestamp_us": _safe_int(frame["timestamp_us"]),
            "x_ego": round(_safe_float(frame["x_ego"]), 4),
            "y_ego": round(_safe_float(frame["y_ego"]), 4),
            "coordinate_frame": "rollout_anchor_ego",
        }
        for frame in list(case.get("future_frames") or [])
    ]


def _constant_velocity_trajectory(case: Dict[str, object]) -> List[Dict[str, object]]:
    history = list(case.get("history_frames") or [])
    future = list(case.get("future_frames") or [])
    if not history or not future:
        return []
    anchor = dict(history[-1])
    if len(history) >= 2:
        previous = dict(history[-2])
    else:
        previous = dict(anchor)
    delta_x = _safe_float(anchor["x_ego"]) - _safe_float(previous["x_ego"])
    delta_y = _safe_float(anchor["y_ego"]) - _safe_float(previous["y_ego"])
    predictions = []
    for step_idx, frame in enumerate(future, start=1):
        predictions.append(
            {
                "sample_token": str(frame["sample_token"]),
                "sample_idx": _safe_int(frame["sample_idx"]),
                "timestamp_us": _safe_int(frame["timestamp_us"]),
                "x_ego": round(_safe_float(anchor["x_ego"]) + delta_x * step_idx, 4),
                "y_ego": round(_safe_float(anchor["y_ego"]) + delta_y * step_idx, 4),
            }
        )
    return predictions


def _risk_underreach_trajectory(case: Dict[str, object]) -> List[Dict[str, object]]:
    history = list(case.get("history_frames") or [])
    future = list(case.get("future_frames") or [])
    if not history or not future:
        return []
    anchor = dict(history[-1])
    behavior = str(case.get("primary_behavior") or "")
    if behavior not in {"crossing", "cut_in", "oncoming"}:
        return _constant_velocity_trajectory(case)
    predictions = []
    longitudinal_scale = 0.9 if behavior in {"crossing", "cut_in"} else 0.85
    lateral_scale = 0.45 if behavior in {"crossing", "cut_in"} else 0.75
    for frame in future:
        dx = _safe_float(frame["x_ego"]) - _safe_float(anchor["x_ego"])
        dy = _safe_float(frame["y_ego"]) - _safe_float(anchor["y_ego"])
        predictions.append(
            {
                "sample_token": str(frame["sample_token"]),
                "sample_idx": _safe_int(frame["sample_idx"]),
                "timestamp_us": _safe_int(frame["timestamp_us"]),
                "x_ego": round(_safe_float(anchor["x_ego"]) + longitudinal_scale * dx, 4),
                "y_ego": round(_safe_float(anchor["y_ego"]) + lateral_scale * dy, 4),
            }
        )
    return predictions


def _occupancy_from_trajectory(
    case: Dict[str, object],
    trajectory: Sequence[Dict[str, object]],
    use_gt_context: bool,
    grid_spec: Dict[str, float],
) -> List[Dict[str, object]]:
    gt_context_by_sample = {
        _safe_int(frame["sample_idx"]): dict(frame)
        for frame in list(case.get("future_occupancy") or [])
    }
    result = []
    for frame in trajectory:
        sample_idx = _safe_int(frame["sample_idx"])
        gt_context = gt_context_by_sample.get(sample_idx, {})
        primary_cells = _rasterize_points(
            [(_safe_float(frame["x_ego"]), _safe_float(frame["y_ego"]))],
            grid_spec,
        )
        context_cells = list(gt_context.get("context_cells") or []) if use_gt_context else []
        result.append(
            {
                "sample_token": str(frame["sample_token"]),
                "sample_idx": sample_idx,
                "timestamp_us": _safe_int(frame.get("timestamp_us")),
                "primary_actor_cells": primary_cells,
                "context_cells": context_cells,
                "occupied_cells": _build_union_cells(primary_cells, context_cells),
            }
        )
    return result


def generate_proxy_world_model_predictions(
    benchmark_path: Path,
    output_path: Path,
    profile_name: str,
) -> Dict[str, object]:
    if profile_name not in WORLD_MODEL_PROXY_PROFILES:
        raise ValueError(
            "Unknown proxy profile: {0}. Expected one of {1}.".format(
                profile_name,
                ", ".join(WORLD_MODEL_PROXY_PROFILES),
            )
        )

    benchmark = _load_json(benchmark_path)
    grid_spec = dict((benchmark.get("metadata") or {}).get("grid_spec") or DEFAULT_GRID_SPEC)
    predictions = []
    for case in list(benchmark.get("cases") or []):
        if profile_name == "oracle_rollout":
            future_trajectory = _copy_future_trajectory(case)
            occupancy = [
                {
                    "sample_token": str(frame["sample_token"]),
                    "sample_idx": _safe_int(frame["sample_idx"]),
                    "timestamp_us": _safe_int(frame["timestamp_us"]),
                    "primary_actor_cells": list(frame.get("primary_actor_cells") or []),
                    "context_cells": list(frame.get("context_cells") or []),
                    "occupied_cells": list(frame.get("occupied_cells") or []),
                }
                for frame in list(case.get("future_occupancy") or [])
            ]
        elif profile_name == "kinematic_rollout":
            future_trajectory = _constant_velocity_trajectory(case)
            occupancy = _occupancy_from_trajectory(case, future_trajectory, use_gt_context=True, grid_spec=grid_spec)
        else:
            future_trajectory = _risk_underreach_trajectory(case)
            occupancy = _occupancy_from_trajectory(case, future_trajectory, use_gt_context=True, grid_spec=grid_spec)

        predictions.append(
            {
                "benchmark_group": str(case["benchmark_group"]),
                "primary_behavior": str(case["primary_behavior"]),
                "category_group": str(case["category_group"]),
                "future_trajectory": future_trajectory,
                "future_occupancy": occupancy,
            }
        )

    output = {
        "metadata": {
            "generator": "proxy_world_model_predictions_v1",
            "profile_name": profile_name,
            "source_benchmark": str(benchmark_path),
            "coordinate_frame": "rollout_anchor_ego",
            "uses_future_ego_pose_for_targets": False,
        },
        "predictions": predictions,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "profile_name": profile_name, "prediction_count": len(predictions)}


def _frame_reference_index(case: Dict[str, object]) -> Tuple[Dict[int, Dict[str, object]], List[Dict[str, object]]]:
    frames = [dict(frame) for frame in list(case.get("future_frames") or [])]
    return ({_safe_int(frame["sample_idx"]): frame for frame in frames}, frames)


def _resolve_case_for_prediction(
    case_by_group: Dict[str, Dict[str, object]],
    case_by_reference: Dict[str, Dict[str, object]],
    record: Dict[str, object],
    fallback_key: str = "",
) -> Optional[Dict[str, object]]:
    benchmark_group = str(record.get("benchmark_group") or fallback_key or "").strip()
    if benchmark_group and benchmark_group in case_by_group:
        return dict(case_by_group[benchmark_group])
    reference_case_key = str(record.get("reference_case_key") or fallback_key or "").strip()
    if reference_case_key and reference_case_key in case_by_reference:
        return dict(case_by_reference[reference_case_key])
    return None


def _normalize_xy_entry(raw_frame: Dict[str, object]) -> Tuple[float, float]:
    if raw_frame.get("xy_ego") is not None:
        xy = list(raw_frame.get("xy_ego") or [])
        if len(xy) >= 2:
            return round(_safe_float(xy[0]), 4), round(_safe_float(xy[1]), 4)
    return round(_safe_float(raw_frame.get("x_ego")), 4), round(_safe_float(raw_frame.get("y_ego")), 4)


def _normalize_trajectory_frames(case: Dict[str, object], record: Dict[str, object]) -> List[Dict[str, object]]:
    ref_by_idx, ordered_refs = _frame_reference_index(case)
    rows: List[Dict[str, object]] = []
    future_trajectory = list(record.get("future_trajectory") or [])
    if future_trajectory:
        for step_idx, raw_frame in enumerate(future_trajectory):
            if not isinstance(raw_frame, dict):
                continue
            ref = dict(ordered_refs[min(step_idx, len(ordered_refs) - 1)]) if ordered_refs else {}
            sample_idx = _safe_int(raw_frame.get("sample_idx"), _safe_int(ref.get("sample_idx")))
            ref = dict(ref_by_idx.get(sample_idx) or ref)
            x_ego, y_ego = _normalize_xy_entry(raw_frame)
            rows.append(
                {
                    "sample_token": str(raw_frame.get("sample_token") or ref.get("sample_token") or ""),
                    "sample_idx": sample_idx,
                    "timestamp_us": _safe_int(raw_frame.get("timestamp_us"), _safe_int(ref.get("timestamp_us"))),
                    "x_ego": x_ego,
                    "y_ego": y_ego,
                }
            )
        return rows

    xy_sequence = list(record.get("xy_ego") or record.get("future_xy_ego") or [])
    sample_indices = list(record.get("sample_indices") or [])
    for step_idx, xy in enumerate(xy_sequence):
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        ref = dict(ordered_refs[min(step_idx, len(ordered_refs) - 1)]) if ordered_refs else {}
        sample_idx = _safe_int(sample_indices[step_idx], _safe_int(ref.get("sample_idx"))) if step_idx < len(sample_indices) else _safe_int(ref.get("sample_idx"))
        ref = dict(ref_by_idx.get(sample_idx) or ref)
        rows.append(
            {
                "sample_token": str(ref.get("sample_token") or ""),
                "sample_idx": sample_idx,
                "timestamp_us": _safe_int(ref.get("timestamp_us")),
                "x_ego": round(_safe_float(xy[0]), 4),
                "y_ego": round(_safe_float(xy[1]), 4),
            }
        )
    return rows


def _normalize_trajectory_modes(case: Dict[str, object], record: Dict[str, object]) -> List[List[Dict[str, object]]]:
    modes = []
    raw_modes = list(record.get("future_trajectory_modes") or [])
    if raw_modes:
        for raw_mode in raw_modes:
            if isinstance(raw_mode, list):
                modes.append(_normalize_trajectory_frames(case, {"future_trajectory": raw_mode}))
        return modes

    xy_modes = list(record.get("xy_ego_modes") or record.get("future_xy_ego_modes") or [])
    sample_indices = list(record.get("sample_indices") or [])
    for raw_mode in xy_modes:
        if not isinstance(raw_mode, list):
            continue
        mode_record = {
            "xy_ego": raw_mode,
            "sample_indices": sample_indices,
        }
        modes.append(_normalize_trajectory_frames(case, mode_record))
    return modes


def _normalize_cell_sequence(value: object) -> List[List[List[int]]]:
    rows = []
    for frame_cells in list(value or []):
        cells = []
        for cell in list(frame_cells or []):
            if isinstance(cell, (list, tuple)) and len(cell) >= 2:
                cells.append([_safe_int(cell[0]), _safe_int(cell[1])])
        rows.append(cells)
    return rows


def _normalize_occupancy_frames(
    case: Dict[str, object],
    record: Dict[str, object],
    future_trajectory: Sequence[Dict[str, object]],
    rasterize_trajectory: bool,
    grid_spec: Dict[str, float],
) -> List[Dict[str, object]]:
    ref_by_idx, ordered_refs = _frame_reference_index(case)
    future_occupancy = list(record.get("future_occupancy") or [])
    if future_occupancy:
        rows: List[Dict[str, object]] = []
        for step_idx, raw_frame in enumerate(future_occupancy):
            if not isinstance(raw_frame, dict):
                continue
            ref = dict(ordered_refs[min(step_idx, len(ordered_refs) - 1)]) if ordered_refs else {}
            sample_idx = _safe_int(raw_frame.get("sample_idx"), _safe_int(ref.get("sample_idx")))
            ref = dict(ref_by_idx.get(sample_idx) or ref)
            primary_cells = _normalize_cell_sequence([raw_frame.get("primary_actor_cells")])[0]
            context_cells = _normalize_cell_sequence([raw_frame.get("context_cells")])[0]
            occupied_cells = _normalize_cell_sequence([raw_frame.get("occupied_cells")])[0]
            if not occupied_cells:
                occupied_cells = _build_union_cells(primary_cells, context_cells)
            rows.append(
                {
                    "sample_token": str(raw_frame.get("sample_token") or ref.get("sample_token") or ""),
                    "sample_idx": sample_idx,
                    "timestamp_us": _safe_int(raw_frame.get("timestamp_us"), _safe_int(ref.get("timestamp_us"))),
                    "primary_actor_cells": primary_cells,
                    "context_cells": context_cells,
                    "occupied_cells": occupied_cells,
                }
            )
        return rows

    primary_sequences = _normalize_cell_sequence(record.get("primary_actor_cells"))
    context_sequences = _normalize_cell_sequence(record.get("context_cells"))
    occupied_sequences = _normalize_cell_sequence(record.get("occupied_cells"))
    if primary_sequences or context_sequences or occupied_sequences:
        rows = []
        frame_count = max(len(primary_sequences), len(context_sequences), len(occupied_sequences), len(future_trajectory))
        for step_idx in range(frame_count):
            ref = dict(ordered_refs[min(step_idx, len(ordered_refs) - 1)]) if ordered_refs else {}
            traj_ref = dict(future_trajectory[min(step_idx, len(future_trajectory) - 1)]) if future_trajectory else {}
            sample_idx = _safe_int(traj_ref.get("sample_idx"), _safe_int(ref.get("sample_idx")))
            ref = dict(ref_by_idx.get(sample_idx) or ref)
            primary_cells = primary_sequences[step_idx] if step_idx < len(primary_sequences) else []
            context_cells = context_sequences[step_idx] if step_idx < len(context_sequences) else []
            occupied_cells = occupied_sequences[step_idx] if step_idx < len(occupied_sequences) else _build_union_cells(primary_cells, context_cells)
            rows.append(
                {
                    "sample_token": str(traj_ref.get("sample_token") or ref.get("sample_token") or ""),
                    "sample_idx": sample_idx,
                    "timestamp_us": _safe_int(traj_ref.get("timestamp_us"), _safe_int(ref.get("timestamp_us"))),
                    "primary_actor_cells": primary_cells,
                    "context_cells": context_cells,
                    "occupied_cells": occupied_cells,
                }
            )
        return rows

    if rasterize_trajectory:
        return _occupancy_from_trajectory(case, future_trajectory, use_gt_context=False, grid_spec=grid_spec)
    return []


def adapt_world_model_predictions(
    benchmark_path: Path,
    input_path: Path,
    output_path: Path,
    rasterize_trajectory: bool = True,
) -> Dict[str, object]:
    benchmark = _load_json(benchmark_path)
    raw_payload = _load_json(input_path)
    grid_spec = dict((benchmark.get("metadata") or {}).get("grid_spec") or DEFAULT_GRID_SPEC)
    case_by_group = {
        str(case.get("benchmark_group") or ""): dict(case)
        for case in list(benchmark.get("cases") or [])
        if str(case.get("benchmark_group") or "")
    }
    case_by_reference = {
        str(case.get("reference_case_key") or ""): dict(case)
        for case in list(benchmark.get("cases") or [])
        if str(case.get("reference_case_key") or "")
    }

    raw_predictions = raw_payload.get("predictions")
    records: List[Tuple[str, Dict[str, object]]] = []
    if isinstance(raw_predictions, dict):
        for key, value in raw_predictions.items():
            if isinstance(value, dict):
                records.append((str(key), dict(value)))
    else:
        for item in list(raw_predictions or []):
            if isinstance(item, dict):
                records.append(("", dict(item)))

    adapted_predictions = []
    skipped_count = 0
    for fallback_key, record in records:
        case = _resolve_case_for_prediction(case_by_group, case_by_reference, record, fallback_key=fallback_key)
        if case is None:
            skipped_count += 1
            continue
        future_trajectory = _normalize_trajectory_frames(case, record)
        future_trajectory_modes = _normalize_trajectory_modes(case, record)
        if not future_trajectory_modes and future_trajectory:
            future_trajectory_modes = [future_trajectory]
        future_occupancy = _normalize_occupancy_frames(
            case=case,
            record=record,
            future_trajectory=future_trajectory,
            rasterize_trajectory=rasterize_trajectory,
            grid_spec=grid_spec,
        )
        adapted_predictions.append(
            {
                "benchmark_group": str(case["benchmark_group"]),
                "reference_case_key": str(case["reference_case_key"]),
                "primary_behavior": str(case["primary_behavior"]),
                "category_group": str(case["category_group"]),
                "mode_probabilities": [round(_safe_float(value), 6) for value in list(record.get("mode_probabilities") or [])],
                "future_trajectory_modes": future_trajectory_modes,
                "future_trajectory": future_trajectory,
                "future_occupancy": future_occupancy,
            }
        )

    output = {
        "metadata": {
            "generator": "world_model_prediction_adapter_v1",
            "source_benchmark": str(benchmark_path),
            "source_predictions": str(input_path),
            "rasterize_trajectory": bool(rasterize_trajectory),
            "prediction_count": len(adapted_predictions),
            "skipped_count": skipped_count,
            "profile_name": str((raw_payload.get("metadata") or {}).get("profile_name") or input_path.stem),
        },
        "predictions": adapted_predictions,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(output_path),
        "prediction_count": len(adapted_predictions),
        "skipped_count": skipped_count,
    }


def _future_global_xy(case: Dict[str, object]) -> List[List[float]]:
    rows = []
    for frame in list(case.get("future_frames") or []):
        if frame.get("x_global") is None or frame.get("y_global") is None:
            continue
        rows.append([_safe_float(frame["x_global"]), _safe_float(frame["y_global"])])
    return rows


def _benchmark_prediction_tokens(benchmark: Dict[str, object]) -> List[str]:
    tokens = []
    for case in list(benchmark.get("cases") or []):
        instance_token = str(case.get("instance_token") or "")
        sample_token = str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or "")
        if instance_token and sample_token:
            tokens.append("{0}_{1}".format(instance_token, sample_token))
    return _unique_strings(tokens)


def _benchmark_case_index_by_token(benchmark: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    mapping = {}
    for case in list(benchmark.get("cases") or []):
        instance_token = str(case.get("instance_token") or "")
        sample_token = str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or "")
        if instance_token and sample_token:
            mapping["{0}_{1}".format(instance_token, sample_token)] = dict(case)
    return mapping


def _benchmark_horizon_seconds(benchmark: Dict[str, object]) -> float:
    horizon_s = 0.5
    for case in list(benchmark.get("cases") or []):
        value = _safe_float((case.get("motion_targets") or {}).get("horizon_s"), 0.0)
        if value > horizon_s:
            horizon_s = value
    rounded = max(0.5, round(horizon_s / 0.5) * 0.5)
    return round(float(rounded), 3)


def _trajectory_errors_for_mode(
    case: Dict[str, object],
    mode_trajectory: Sequence[Sequence[object]],
) -> Tuple[List[float], Optional[float]]:
    anchor_sample_idx = _safe_int(case.get("rollout_anchor_sample_idx"), _safe_int(case.get("anchor_sample_idx")))
    errors = []
    final_error = None
    for frame in list(case.get("future_frames") or []):
        step_idx = _safe_int(frame.get("sample_idx")) - anchor_sample_idx - 1
        if step_idx < 0 or step_idx >= len(mode_trajectory):
            continue
        point = list(mode_trajectory[step_idx] or [])
        if len(point) < 2:
            continue
        error = math.hypot(
            float(point[0]) - _safe_float(frame.get("x_global")),
            float(point[1]) - _safe_float(frame.get("y_global")),
        )
        errors.append(error)
        final_error = error
    return errors, final_error


def _select_forecast_mode_index(
    case: Dict[str, object],
    prediction_modes: Sequence[Sequence[Sequence[object]]],
    probabilities: Sequence[object],
    mode_selection: str,
) -> int:
    if not prediction_modes:
        return 0
    if mode_selection == "top_probability":
        if probabilities:
            return int(max(range(len(prediction_modes)), key=lambda idx: _safe_float(probabilities[idx])))
        return 0
    if mode_selection not in {"oracle_ade", "oracle_fde"}:
        raise ValueError("Unsupported mode_selection: {0}".format(mode_selection))

    best_idx = 0
    best_score = None
    for idx, mode in enumerate(prediction_modes):
        errors, final_error = _trajectory_errors_for_mode(case, mode)
        if mode_selection == "oracle_ade":
            score = mean(errors) if errors else float("inf")
        else:
            score = float(final_error) if final_error is not None else float("inf")
        if best_score is None or score < best_score:
            best_idx = idx
            best_score = score
    return int(best_idx)


def _local_trajectory_from_global_mode(
    case: Dict[str, object],
    mode_trajectory: Sequence[Sequence[object]],
) -> List[Dict[str, object]]:
    future_trajectory = []
    anchor_sample_idx = _safe_int(case.get("rollout_anchor_sample_idx"), _safe_int(case.get("anchor_sample_idx")))
    for frame in list(case.get("future_frames") or []):
        step_idx = _safe_int(frame.get("sample_idx")) - anchor_sample_idx - 1
        if step_idx < 0 or step_idx >= len(mode_trajectory):
            continue
        global_xy = list(mode_trajectory[step_idx] or [])
        if len(global_xy) < 2:
            continue
        anchor_frame = dict(list(case.get("history_frames") or [{}])[-1])
        local_xy = global_xy_to_anchor_ego(
            np.asarray([[float(global_xy[0]), float(global_xy[1])]], dtype=float),
            [_safe_float(anchor_frame.get("ego_x_global")), _safe_float(anchor_frame.get("ego_y_global"))],
            _safe_float(anchor_frame.get("ego_yaw")),
        )[0]
        future_trajectory.append(
            {
                "sample_token": str(frame.get("sample_token") or ""),
                "sample_idx": _safe_int(frame.get("sample_idx")),
                "timestamp_us": _safe_int(frame.get("timestamp_us")),
                "x_ego": round(float(local_xy[0]), 4),
                "y_ego": round(float(local_xy[1]), 4),
                "x_global": round(float(global_xy[0]), 4),
                "y_global": round(float(global_xy[1]), 4),
                "coordinate_frame": "rollout_anchor_ego",
            }
        )
    return future_trajectory


def _multimodal_mode_metrics(
    case: Dict[str, object],
    mode_frames: Sequence[Sequence[Dict[str, object]]],
    mode_probabilities: Sequence[object],
) -> Dict[str, object]:
    ranked_indices = list(range(len(mode_frames)))
    if mode_probabilities:
        ranked_indices = sorted(
            ranked_indices,
            key=lambda idx: _safe_float(mode_probabilities[idx]),
            reverse=True,
        )

    mode_metrics = []
    gt_by_idx = _trajectory_by_sample_idx(list(case.get("future_frames") or []))
    gt_frame_count = len(gt_by_idx)
    for idx, frames in enumerate(mode_frames):
        pred_by_idx = _trajectory_by_sample_idx(list(frames or []))
        matched_sample_indices = sorted(set(gt_by_idx) & set(pred_by_idx))
        errors = [_distance_m(gt_by_idx[sample_idx], pred_by_idx[sample_idx]) for sample_idx in matched_sample_indices]
        ade_m = mean(errors) if errors else float("inf")
        final_sample_idx = max(gt_by_idx) if gt_by_idx else None
        fde_m = _distance_m(gt_by_idx[final_sample_idx], pred_by_idx[final_sample_idx]) if final_sample_idx in pred_by_idx else float("inf")
        max_error = max(errors) if errors else float("inf")
        full_horizon = len(matched_sample_indices) == gt_frame_count
        miss = (not full_horizon) or (max_error >= FORECAST_MISS_TOLERANCE_M)
        mode_metrics.append(
            {
                "index": idx,
                "ade_m": float(ade_m),
                "fde_m": float(fde_m),
                "max_error_m": float(max_error),
                "miss": bool(miss),
            }
        )

    topk = {}
    for report_k in FORECAST_TOPK_REPORTS:
        considered = [mode_metrics[idx] for idx in ranked_indices[: min(report_k, len(ranked_indices))]]
        if considered:
            min_ade = min(item["ade_m"] for item in considered)
            min_fde = min(item["fde_m"] for item in considered)
            miss_rate = 0.0 if any(not item["miss"] for item in considered) else 1.0
        else:
            min_ade = float("inf")
            min_fde = float("inf")
            miss_rate = 1.0
        topk[report_k] = {
            "min_ade": float(min_ade),
            "min_fde": float(min_fde),
            "miss_rate": float(miss_rate),
        }
    return {
        "mode_count": len(mode_frames),
        "topk": topk,
    }


def adapt_nuscenes_forecast_predictions(
    benchmark_path: Path,
    input_path: Path,
    output_path: Path,
    mode_selection: str = "top_probability",
    rasterize_trajectory: bool = True,
) -> Dict[str, object]:
    if mode_selection not in NUSCENES_FORECAST_MODE_SELECTIONS:
        raise ValueError(
            "Unsupported mode_selection: {0}. Expected one of {1}.".format(
                mode_selection,
                ", ".join(NUSCENES_FORECAST_MODE_SELECTIONS),
            )
        )

    benchmark = _load_json(benchmark_path)
    raw_payload = _load_json(input_path)
    grid_spec = dict((benchmark.get("metadata") or {}).get("grid_spec") or DEFAULT_GRID_SPEC)
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("predictions"), list):
        records = [dict(item) for item in list(raw_payload.get("predictions") or []) if isinstance(item, dict)]
    elif isinstance(raw_payload, list):
        records = [dict(item) for item in raw_payload if isinstance(item, dict)]
    else:
        raise ValueError("Expected a prediction JSON list or a dictionary with a 'predictions' list.")

    case_by_anchor = {
        (
            str(case.get("instance_token") or ""),
            str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or ""),
        ): dict(case)
        for case in list(benchmark.get("cases") or [])
    }

    adapted_predictions = []
    skipped_count = 0
    for record in records:
        instance_token = str(record.get("instance") or record.get("instance_token") or "")
        sample_token = str(record.get("sample") or record.get("sample_token") or "")
        case = case_by_anchor.get((instance_token, sample_token))
        if case is None:
            skipped_count += 1
            continue

        prediction_modes = list(record.get("prediction") or [])
        probabilities = list(record.get("probabilities") or [])
        if not prediction_modes:
            skipped_count += 1
            continue

        selected_mode_idx = _select_forecast_mode_index(
            case=case,
            prediction_modes=prediction_modes,
            probabilities=probabilities,
            mode_selection=mode_selection,
        )
        selected_mode = list(prediction_modes[selected_mode_idx] or [])
        future_trajectory = _local_trajectory_from_global_mode(case, selected_mode)
        future_trajectory_modes = [
            _local_trajectory_from_global_mode(case, mode_trajectory)
            for mode_trajectory in prediction_modes
        ]

        future_occupancy = _occupancy_from_trajectory(
            case=case,
            trajectory=future_trajectory,
            use_gt_context=False,
            grid_spec=grid_spec,
        ) if rasterize_trajectory else []

        adapted_predictions.append(
            {
                "benchmark_group": str(case["benchmark_group"]),
                "reference_case_key": str(case["reference_case_key"]),
                "primary_behavior": str(case["primary_behavior"]),
                "category_group": str(case["category_group"]),
                "mode_selection": mode_selection,
                "selected_mode_index": selected_mode_idx,
                "selected_mode_probability": round(_safe_float(probabilities[selected_mode_idx]), 6) if selected_mode_idx < len(probabilities) else None,
                "mode_probabilities": [round(_safe_float(value), 6) for value in probabilities],
                "future_trajectory_modes": future_trajectory_modes,
                "future_trajectory": future_trajectory,
                "future_occupancy": future_occupancy,
            }
        )

    output = {
        "metadata": {
            "generator": "nuscenes_forecast_adapter_v1",
            "source_benchmark": str(benchmark_path),
            "source_predictions": str(input_path),
            "mode_selection": mode_selection,
            "rasterize_trajectory": bool(rasterize_trajectory),
            "prediction_count": len(adapted_predictions),
            "skipped_count": skipped_count,
            "profile_name": "{0}_{1}".format(input_path.stem, mode_selection),
        },
        "predictions": adapted_predictions,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(output_path),
        "prediction_count": len(adapted_predictions),
        "skipped_count": skipped_count,
        "mode_selection": mode_selection,
    }


def adapt_and_evaluate_nuscenes_forecast_predictions(
    benchmark_path: Path,
    input_path: Path,
    output_dir: Path,
    mode_selection: str = "top_probability",
    profile_name: str = "",
    rasterize_trajectory: bool = True,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = output_dir / "adapted_predictions.json"
    adapter_metadata = adapt_nuscenes_forecast_predictions(
        benchmark_path=benchmark_path,
        input_path=input_path,
        output_path=adapted_path,
        mode_selection=mode_selection,
        rasterize_trajectory=rasterize_trajectory,
    )
    summary = evaluate_world_model_predictions(
        benchmark_path=benchmark_path,
        predictions_path=adapted_path,
        output_dir=output_dir,
        profile_name=profile_name or "{0}_{1}".format(Path(input_path).stem, mode_selection),
    )
    return {
        "adapter": adapter_metadata,
        "overview": dict(summary["overview"]),
        "output_dir": str(output_dir),
    }


class _FrameCountPredictHelper(PredictHelper):
    def get_future_for_agent(self, instance_token, sample_token, seconds, **kwargs):
        # Keyframe timestamps jitter around 2 Hz; the oracle requires exactly K frames.
        future = super().get_future_for_agent(instance_token, sample_token, seconds + 0.1, **kwargs)
        return future[:int(round(seconds * 2))]


def generate_nuscenes_forecast_baselines(
    benchmark_path: Path,
    dataroot: Path,
    version: str,
    output_dir: Path,
) -> Dict[str, object]:
    benchmark = _load_json(benchmark_path)
    case_by_token = _benchmark_case_index_by_token(benchmark)
    tokens = list(case_by_token.keys())
    horizon_s = _benchmark_horizon_seconds(benchmark)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    helper = _FrameCountPredictHelper(nusc)

    output_paths = {}
    for profile_name, model_cls in {
        "cv_heading": ConstantVelocityHeading,
        "physics_oracle": PhysicsOracle,
    }.items():
        predictions = []
        by_horizon: Dict[float, List[str]] = defaultdict(list)
        for token, case in case_by_token.items():
            case_horizon = max(0.5, round(_safe_float((case.get("motion_targets") or {}).get("horizon_s"), 0.5) * 2.0) / 2.0)
            by_horizon[case_horizon].append(token)
        for case_horizon, horizon_tokens in sorted(by_horizon.items()):
            model = model_cls(case_horizon, helper)
            for token in horizon_tokens:
                predictions.append(model(token).serialize())
        path = output_dir / "{0}.json".format(profile_name)
        path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
        output_paths[profile_name] = str(path)

    manifest = {
        "generator": "nuscenes_forecast_baselines_v1",
        "source_benchmark": str(benchmark_path),
        "dataroot": str(dataroot),
        "version": version,
        "case_count": len(tokens),
        "horizon_s": horizon_s,
        "profiles": output_paths,
        "oracle_ground_truth_alignment": "2 Hz frame count with 0.1 s timestamp tolerance",
    }
    (output_dir / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "case_count": len(tokens),
        "horizon_s": horizon_s,
        "profiles": output_paths,
    }


def run_nuscenes_forecast_baselines(
    benchmark_path: Path,
    dataroot: Path,
    version: str,
    output_dir: Path,
    mode_selection: str = "top_probability",
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = generate_nuscenes_forecast_baselines(
        benchmark_path=benchmark_path,
        dataroot=dataroot,
        version=version,
        output_dir=output_dir / "predictions",
    )

    evaluation_dirs = []
    for profile_name in NUSCENES_FORECAST_BASELINE_PROFILES:
        prediction_path = Path(generation["profiles"][profile_name])
        profile_output = output_dir / profile_name
        adapt_and_evaluate_nuscenes_forecast_predictions(
            benchmark_path=benchmark_path,
            input_path=prediction_path,
            output_dir=profile_output,
            mode_selection=mode_selection,
            profile_name=profile_name,
            rasterize_trajectory=True,
        )
        evaluation_dirs.append(profile_output)

    comparison_dir = output_dir / "comparison"
    comparison = compare_world_model_evaluations(evaluation_dirs=evaluation_dirs, output_dir=comparison_dir)
    return {
        "generation": generation,
        "comparison": comparison,
        "output_dir": str(output_dir),
    }


def _prediction_index(predictions: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        str(item.get("benchmark_group") or ""): dict(item)
        for item in predictions
        if str(item.get("benchmark_group") or "")
    }


def _trajectory_by_sample_idx(frames: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    return {_safe_int(frame["sample_idx"]): dict(frame) for frame in frames}


def _iou(lhs: set[Tuple[int, int]], rhs: set[Tuple[int, int]]) -> float:
    if not lhs and not rhs:
        return 1.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs)) / float(len(union))


def _closest_approach_from_trajectory(
    anchor_timestamp_us: int,
    frames: Sequence[Dict[str, object]],
) -> Tuple[Optional[float], Optional[float]]:
    if not frames:
        return None, None
    closest = min(frames, key=lambda frame: math.hypot(_safe_float(frame["x_ego"]), _safe_float(frame["y_ego"])))
    closest_distance = math.hypot(_safe_float(closest["x_ego"]), _safe_float(closest["y_ego"]))
    closest_offset = max(0.0, (_safe_int(closest["timestamp_us"]) - anchor_timestamp_us) / 1_000_000.0)
    return round(closest_distance, 4), round(closest_offset, 4)


def _evaluate_world_model_case(case: Dict[str, object], prediction: Optional[Dict[str, object]]) -> Dict[str, object]:
    gt_future = list(case.get("future_frames") or [])
    gt_occ = list(case.get("future_occupancy") or [])
    if not gt_future:
        return {
            "benchmark_group": str(case["benchmark_group"]),
            "primary_behavior": str(case["primary_behavior"]),
            "category_group": str(case["category_group"]),
            "full_horizon_success": False,
            "horizon_recall": 0.0,
            "ade_m": None,
            "fde_m": None,
            "occupancy_iou": 0.0,
            "primary_actor_iou": 0.0,
            "context_iou": 0.0,
            "closest_approach_distance_error_m": None,
            "closest_approach_time_error_s": None,
            "risk_fidelity_score": 0.0,
            "mode_count": 0,
            "min_ade_at_1": None,
            "min_ade_at_5": None,
            "min_ade_at_10": None,
            "min_fde_at_1": None,
            "min_fde_at_5": None,
            "min_fde_at_10": None,
            "miss_rate_at_1": 1.0,
            "miss_rate_at_5": 1.0,
            "miss_rate_at_10": 1.0,
            "failure_tags": ["missing_ground_truth"],
        }

    if prediction is None:
        return {
            "benchmark_group": str(case["benchmark_group"]),
            "reference_case_key": str(case["reference_case_key"]),
            "primary_behavior": str(case["primary_behavior"]),
            "category_group": str(case["category_group"]),
            "location": str(case["location"]),
            "future_frame_count": len(gt_future),
            "predicted_frame_count": 0,
            "full_horizon_success": False,
            "horizon_recall": 0.0,
            "ade_m": None,
            "fde_m": None,
            "occupancy_iou": 0.0,
            "primary_actor_iou": 0.0,
            "context_iou": 0.0,
            "closest_approach_distance_error_m": None,
            "closest_approach_time_error_s": None,
            "risk_fidelity_score": 0.0,
            "mode_count": 0,
            "min_ade_at_1": None,
            "min_ade_at_5": None,
            "min_ade_at_10": None,
            "min_fde_at_1": None,
            "min_fde_at_5": None,
            "min_fde_at_10": None,
            "miss_rate_at_1": 1.0,
            "miss_rate_at_5": 1.0,
            "miss_rate_at_10": 1.0,
            "failure_tags": ["missing_prediction", "short_horizon", "occupancy_error", "trajectory_error"],
        }

    pred_future = list(prediction.get("future_trajectory") or [])
    pred_occ = list(prediction.get("future_occupancy") or [])
    gt_by_idx = _trajectory_by_sample_idx(gt_future)
    pred_by_idx = _trajectory_by_sample_idx(pred_future)
    gt_occ_by_idx = {_safe_int(frame["sample_idx"]): dict(frame) for frame in gt_occ}
    pred_occ_by_idx = {_safe_int(frame["sample_idx"]): dict(frame) for frame in pred_occ}

    matched_sample_indices = sorted(set(gt_by_idx) & set(pred_by_idx))
    errors = [_distance_m(gt_by_idx[idx], pred_by_idx[idx]) for idx in matched_sample_indices]
    horizon_recall = _ratio(len(matched_sample_indices), len(gt_future))
    ade_m = mean(errors) if errors else None
    last_sample_idx = _safe_int(gt_future[-1]["sample_idx"])
    fde_m = _distance_m(gt_by_idx[last_sample_idx], pred_by_idx[last_sample_idx]) if last_sample_idx in pred_by_idx else None

    occupancy_ious = []
    primary_ious = []
    context_ious = []
    for frame in gt_occ:
        sample_idx = _safe_int(frame["sample_idx"])
        pred_frame = pred_occ_by_idx.get(sample_idx, {})
        gt_primary = _frame_cell_set(frame, "primary_actor_cells")
        gt_context = _frame_cell_set(frame, "context_cells")
        gt_union = _frame_cell_set(frame, "occupied_cells") or (gt_primary | gt_context)
        pred_primary = _frame_cell_set(pred_frame, "primary_actor_cells")
        pred_context = _frame_cell_set(pred_frame, "context_cells")
        pred_union = _frame_cell_set(pred_frame, "occupied_cells") or (pred_primary | pred_context)
        primary_ious.append(_iou(gt_primary, pred_primary))
        context_ious.append(_iou(gt_context, pred_context))
        occupancy_ious.append(_iou(gt_union, pred_union))

    anchor_timestamp_us = _safe_int(list(case.get("history_frames") or [{}])[-1].get("timestamp_us"))
    gt_closest_distance, gt_closest_time = _closest_approach_from_trajectory(anchor_timestamp_us, gt_future)
    pred_closest_distance, pred_closest_time = _closest_approach_from_trajectory(anchor_timestamp_us, pred_future)
    closest_distance_error = (
        abs(float(pred_closest_distance) - float(gt_closest_distance))
        if pred_closest_distance is not None and gt_closest_distance is not None
        else None
    )
    closest_time_error = (
        abs(float(pred_closest_time) - float(gt_closest_time))
        if pred_closest_time is not None and gt_closest_time is not None
        else None
    )

    mean_occupancy_iou = mean(occupancy_ious) if occupancy_ious else 0.0
    mean_primary_iou = mean(primary_ious) if primary_ious else 0.0
    mean_context_iou = mean(context_ious) if context_ious else 0.0

    mode_frames = list(prediction.get("future_trajectory_modes") or [])
    if not mode_frames and pred_future:
        mode_frames = [pred_future]
    mode_probabilities = list(prediction.get("mode_probabilities") or [])
    multimodal_metrics = _multimodal_mode_metrics(case, mode_frames, mode_probabilities) if mode_frames else {
        "mode_count": 0,
        "topk": {k: {"min_ade": float("inf"), "min_fde": float("inf"), "miss_rate": 1.0} for k in FORECAST_TOPK_REPORTS},
    }

    trajectory_score = 0.6 * _bounded_score(ade_m, 4.0) + 0.4 * _bounded_score(fde_m, 6.0)
    occupancy_score = 0.5 * mean_occupancy_iou + 0.5 * mean_primary_iou
    risk_score = 0.5 * _bounded_score(closest_distance_error, 5.0) + 0.5 * _bounded_score(closest_time_error, 2.0)
    risk_fidelity_score = round(0.2 * horizon_recall + 0.3 * trajectory_score + 0.3 * occupancy_score + 0.2 * risk_score, 4)

    failure_tags: List[str] = []
    if horizon_recall < 1.0:
        failure_tags.append("short_horizon")
    if ade_m is None or fde_m is None or ade_m > 1.5 or fde_m > 2.5:
        failure_tags.append("trajectory_error")
    if mean_occupancy_iou < 0.55 or mean_primary_iou < 0.45:
        failure_tags.append("occupancy_error")
    if closest_distance_error is None or closest_time_error is None or closest_distance_error > 1.5 or closest_time_error > 0.75:
        failure_tags.append("risk_alignment_error")

    return {
        "benchmark_group": str(case["benchmark_group"]),
        "reference_case_key": str(case["reference_case_key"]),
        "primary_behavior": str(case["primary_behavior"]),
        "category_group": str(case["category_group"]),
        "location": str(case["location"]),
        "future_frame_count": len(gt_future),
        "predicted_frame_count": len(pred_future),
        "full_horizon_success": horizon_recall == 1.0,
        "horizon_recall": round(horizon_recall, 4),
        "ade_m": round(ade_m, 4) if ade_m is not None else None,
        "fde_m": round(fde_m, 4) if fde_m is not None else None,
        "occupancy_iou": round(mean_occupancy_iou, 4),
        "primary_actor_iou": round(mean_primary_iou, 4),
        "context_iou": round(mean_context_iou, 4),
        "closest_approach_distance_error_m": round(closest_distance_error, 4) if closest_distance_error is not None else None,
        "closest_approach_time_error_s": round(closest_time_error, 4) if closest_time_error is not None else None,
        "risk_fidelity_score": risk_fidelity_score,
        "mode_count": int(multimodal_metrics["mode_count"]),
        "min_ade_at_1": round(multimodal_metrics["topk"][1]["min_ade"], 4) if math.isfinite(multimodal_metrics["topk"][1]["min_ade"]) else None,
        "min_ade_at_5": round(multimodal_metrics["topk"][5]["min_ade"], 4) if math.isfinite(multimodal_metrics["topk"][5]["min_ade"]) else None,
        "min_ade_at_10": round(multimodal_metrics["topk"][10]["min_ade"], 4) if math.isfinite(multimodal_metrics["topk"][10]["min_ade"]) else None,
        "min_fde_at_1": round(multimodal_metrics["topk"][1]["min_fde"], 4) if math.isfinite(multimodal_metrics["topk"][1]["min_fde"]) else None,
        "min_fde_at_5": round(multimodal_metrics["topk"][5]["min_fde"], 4) if math.isfinite(multimodal_metrics["topk"][5]["min_fde"]) else None,
        "min_fde_at_10": round(multimodal_metrics["topk"][10]["min_fde"], 4) if math.isfinite(multimodal_metrics["topk"][10]["min_fde"]) else None,
        "miss_rate_at_1": round(multimodal_metrics["topk"][1]["miss_rate"], 4),
        "miss_rate_at_5": round(multimodal_metrics["topk"][5]["miss_rate"], 4),
        "miss_rate_at_10": round(multimodal_metrics["topk"][10]["miss_rate"], 4),
        "failure_tags": failure_tags,
    }


def _build_group_breakdown(
    case_metrics: Sequence[Dict[str, object]],
    field_name: str,
    output_key: str,
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "case_count": 0,
            "full_horizon_count": 0,
            "risk_scores": [],
            "ade_values": [],
            "occupancy_values": [],
            "failure_counts": defaultdict(int),
        }
    )

    for row in case_metrics:
        value = str(row.get(field_name) or "unknown")
        bucket = grouped[value]
        bucket["case_count"] += 1
        bucket["full_horizon_count"] += int(bool(row["full_horizon_success"]))
        bucket["risk_scores"].append(float(row["risk_fidelity_score"]))
        bucket["occupancy_values"].append(float(row["occupancy_iou"]))
        if row.get("ade_m") is not None:
            bucket["ade_values"].append(float(row["ade_m"]))
        for tag in list(row.get("failure_tags") or []):
            bucket["failure_counts"][str(tag)] += 1

    rows = []
    for value, bucket in grouped.items():
        failure_rows = [{"name": name, "count": int(count)} for name, count in dict(bucket["failure_counts"]).items()]
        failure_rows.sort(key=lambda item: (int(item["count"]), str(item["name"])), reverse=True)
        rows.append(
            {
                output_key: value,
                "case_count": int(bucket["case_count"]),
                "full_horizon_count": int(bucket["full_horizon_count"]),
                "full_horizon_rate": round(_ratio(int(bucket["full_horizon_count"]), int(bucket["case_count"])), 4),
                "mean_risk_fidelity_score": round(mean(bucket["risk_scores"]), 4) if bucket["risk_scores"] else 0.0,
                "mean_ade_m": _metric_mean(bucket["ade_values"]),
                "mean_occupancy_iou": round(mean(bucket["occupancy_values"]), 4) if bucket["occupancy_values"] else 0.0,
                "top_failure_modes": failure_rows[:3],
                "top_failure_summary": ", ".join(
                    "{0}:{1}".format(item["name"], item["count"]) for item in failure_rows[:3]
                )
                or "none",
            }
        )
    rows.sort(key=lambda item: (float(item["mean_risk_fidelity_score"]), int(item["case_count"])), reverse=True)
    return rows


def _build_multi_group_breakdown(
    case_metrics: Sequence[Dict[str, object]],
    field_name: str,
    output_key: str,
) -> List[Dict[str, object]]:
    expanded_rows = []
    for row in case_metrics:
        values = _unique_strings(list(row.get(field_name) or []))
        if not values:
            values = ["unknown"]
        for value in values:
            expanded = dict(row)
            expanded[output_key] = value
            expanded_rows.append(expanded)
    return _build_group_breakdown(expanded_rows, field_name=output_key, output_key=output_key)


def evaluate_world_model_predictions(
    benchmark_path: Path,
    predictions_path: Path,
    output_dir: Path,
    profile_name: str = "",
) -> Dict[str, object]:
    benchmark = _load_json(benchmark_path)
    predictions_payload = _load_json(predictions_path)
    prediction_index = _prediction_index(list(predictions_payload.get("predictions") or []))
    profile_name = profile_name or str((predictions_payload.get("metadata") or {}).get("profile_name") or predictions_path.stem)

    case_metrics = []
    for case in list(benchmark.get("cases") or []):
        row = _evaluate_world_model_case(case, prediction_index.get(str(case["benchmark_group"])))
        row["scene_token"] = str(case.get("scene_token") or "")
        row["scene_name"] = str(case.get("scene_name") or "")
        support = dict(case.get("risk_facets") or {})
        row.update({name: str(support.get(name) or "unknown") for name in RISK_FACET_FIELDS})
        row["challenge_tracks"] = list(case.get("challenge_tracks") or [])
        row["min_distance_m"] = case.get("min_distance_m")
        row["min_ttc_s"] = case.get("min_ttc_s")
        case_metrics.append(row)

    behavior_breakdown = _build_group_breakdown(case_metrics, field_name="primary_behavior", output_key="behavior")
    track_breakdown = _build_multi_group_breakdown(case_metrics, field_name="challenge_tracks", output_key="track")
    risk_breakdowns = {
        field_name: _build_group_breakdown(case_metrics, field_name=field_name, output_key=field_name)
        for field_name in RISK_FACET_FIELDS
    }

    full_horizon_count = sum(1 for row in case_metrics if row["full_horizon_success"])
    horizon_recalls = [float(row["horizon_recall"]) for row in case_metrics]
    ade_values = [float(row["ade_m"]) for row in case_metrics if row.get("ade_m") is not None]
    fde_values = [float(row["fde_m"]) for row in case_metrics if row.get("fde_m") is not None]
    occupancy_values = [float(row["occupancy_iou"]) for row in case_metrics]
    primary_values = [float(row["primary_actor_iou"]) for row in case_metrics]
    context_values = [float(row["context_iou"]) for row in case_metrics]
    closest_distance_values = [
        float(row["closest_approach_distance_error_m"])
        for row in case_metrics
        if row.get("closest_approach_distance_error_m") is not None
    ]
    closest_time_values = [
        float(row["closest_approach_time_error_s"])
        for row in case_metrics
        if row.get("closest_approach_time_error_s") is not None
    ]
    mode_counts = [float(row.get("mode_count") or 0.0) for row in case_metrics]
    min_ade_at_1_values = [float(row["min_ade_at_1"]) for row in case_metrics if row.get("min_ade_at_1") is not None]
    min_ade_at_5_values = [float(row["min_ade_at_5"]) for row in case_metrics if row.get("min_ade_at_5") is not None]
    min_ade_at_10_values = [float(row["min_ade_at_10"]) for row in case_metrics if row.get("min_ade_at_10") is not None]
    min_fde_at_1_values = [float(row["min_fde_at_1"]) for row in case_metrics if row.get("min_fde_at_1") is not None]
    min_fde_at_5_values = [float(row["min_fde_at_5"]) for row in case_metrics if row.get("min_fde_at_5") is not None]
    min_fde_at_10_values = [float(row["min_fde_at_10"]) for row in case_metrics if row.get("min_fde_at_10") is not None]
    miss_rate_at_1_values = [float(row["miss_rate_at_1"]) for row in case_metrics]
    miss_rate_at_5_values = [float(row["miss_rate_at_5"]) for row in case_metrics]
    miss_rate_at_10_values = [float(row["miss_rate_at_10"]) for row in case_metrics]
    risk_scores = [float(row["risk_fidelity_score"]) for row in case_metrics]
    perfect_case_count = sum(
        1
        for row in case_metrics
        if row["full_horizon_success"] and not list(row.get("failure_tags") or []) and float(row["risk_fidelity_score"]) >= 0.999
    )

    summary = {
        "profile_name": profile_name,
        "benchmark_path": str(benchmark_path),
        "predictions_path": str(predictions_path),
        "overview": {
            "case_count": len(case_metrics),
            "full_horizon_count": full_horizon_count,
            "full_horizon_rate": round(_ratio(full_horizon_count, len(case_metrics)), 4),
            "mean_horizon_recall": round(mean(horizon_recalls), 4) if horizon_recalls else 0.0,
            "mean_ade_m": _metric_mean(ade_values),
            "mean_fde_m": _metric_mean(fde_values),
            "mean_occupancy_iou": round(mean(occupancy_values), 4) if occupancy_values else 0.0,
            "mean_primary_actor_iou": round(mean(primary_values), 4) if primary_values else 0.0,
            "mean_context_iou": round(mean(context_values), 4) if context_values else 0.0,
            "mean_closest_approach_distance_error_m": _metric_mean(closest_distance_values),
            "mean_closest_approach_time_error_s": _metric_mean(closest_time_values),
            "mean_risk_fidelity_score": round(mean(risk_scores), 4) if risk_scores else 0.0,
            "perfect_case_count": perfect_case_count,
        },
        "forecast_metrics": {
            "mean_mode_count": round(mean(mode_counts), 4) if mode_counts else 0.0,
            "mean_min_ade_at_1": _metric_mean(min_ade_at_1_values),
            "mean_min_ade_at_5": _metric_mean(min_ade_at_5_values),
            "mean_min_ade_at_10": _metric_mean(min_ade_at_10_values),
            "mean_min_fde_at_1": _metric_mean(min_fde_at_1_values),
            "mean_min_fde_at_5": _metric_mean(min_fde_at_5_values),
            "mean_min_fde_at_10": _metric_mean(min_fde_at_10_values),
            "mean_miss_rate_at_1": round(mean(miss_rate_at_1_values), 4) if miss_rate_at_1_values else 0.0,
            "mean_miss_rate_at_5": round(mean(miss_rate_at_5_values), 4) if miss_rate_at_5_values else 0.0,
            "mean_miss_rate_at_10": round(mean(miss_rate_at_10_values), 4) if miss_rate_at_10_values else 0.0,
        },
        "behavior_breakdown": behavior_breakdown,
        "track_breakdown": track_breakdown,
        "risk_breakdowns": risk_breakdowns,
        "case_metrics": case_metrics,
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "world_model_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Scenario-Conditioned World-Model Evaluation",
        "",
        "- Profile: {0}".format(profile_name),
        "- Cases: {0}".format(summary["overview"]["case_count"]),
        "- Full-horizon success: {0}/{1} ({2:.1%})".format(
            summary["overview"]["full_horizon_count"],
            summary["overview"]["case_count"],
            summary["overview"]["full_horizon_rate"],
        ),
        "- Mean horizon recall: {0:.3f}".format(summary["overview"]["mean_horizon_recall"]),
        "- Mean ADE: {0}".format(_format_metric(summary["overview"]["mean_ade_m"])),
        "- Mean FDE: {0}".format(_format_metric(summary["overview"]["mean_fde_m"])),
        "- Mean occupancy IoU: {0:.3f}".format(summary["overview"]["mean_occupancy_iou"]),
        "- Mean risk fidelity: {0:.3f}".format(summary["overview"]["mean_risk_fidelity_score"]),
        "- Mean MinADE@1: {0}".format(_format_metric(summary["forecast_metrics"]["mean_min_ade_at_1"])),
        "- Mean MinADE@5: {0}".format(_format_metric(summary["forecast_metrics"]["mean_min_ade_at_5"])),
        "- Mean MissRate@5: {0:.3f}".format(summary["forecast_metrics"]["mean_miss_rate_at_5"]),
        "",
        "## Behavior Breakdown",
        "",
        "| Behavior | Cases | Full Horizon | Risk Fidelity | ADE | Occupancy IoU | Top Failure Modes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in behavior_breakdown:
        lines.append(
            "| {0} | {1} | {2}/{1} ({3:.1%}) | {4:.3f} | {5} | {6:.3f} | {7} |".format(
                row["behavior"],
                row["case_count"],
                row["full_horizon_count"],
                row["full_horizon_rate"],
                row["mean_risk_fidelity_score"],
                _format_metric(row["mean_ade_m"]),
                row["mean_occupancy_iou"],
                row["top_failure_summary"],
            )
        )
    lines.extend(["", "## Challenge Track Breakdown", "", "| Track | Cases | Full Horizon | Risk Fidelity | ADE | Occupancy IoU | Top Failure Modes |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in track_breakdown:
        lines.append(
            "| {0} | {1} | {2}/{1} ({3:.1%}) | {4:.3f} | {5} | {6:.3f} | {7} |".format(
                row["track"],
                row["case_count"],
                row["full_horizon_count"],
                row["full_horizon_rate"],
                row["mean_risk_fidelity_score"],
                _format_metric(row["mean_ade_m"]),
                row["mean_occupancy_iou"],
                row["top_failure_summary"],
            )
        )
    lines.extend(["", "## Risk Breakdown", ""])
    for field_name in RISK_FACET_FIELDS:
        lines.extend(
            [
                "### {0}".format(field_name.replace("_", " ").title()),
                "",
                "| Group | Cases | Full Horizon | Risk Fidelity | ADE | Occupancy IoU | Top Failure Modes |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in risk_breakdowns[field_name]:
            lines.append(
                "| {0} | {1} | {2}/{1} ({3:.1%}) | {4:.3f} | {5} | {6:.3f} | {7} |".format(
                    row[field_name],
                    row["case_count"],
                    row["full_horizon_count"],
                    row["full_horizon_rate"],
                    row["mean_risk_fidelity_score"],
                    _format_metric(row["mean_ade_m"]),
                    row["mean_occupancy_iou"],
                    row["top_failure_summary"],
                )
            )
        lines.append("")
    (output_dir / "world_model_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "world_model_metrics_summary.html").write_text(
        WORLD_MODEL_SUMMARY_TEMPLATE.render(summary=summary, metric=_format_metric),
        encoding="utf-8",
    )
    with (output_dir / "world_model_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "benchmark_group",
            "reference_case_key",
            "scene_token",
            "scene_name",
            "primary_behavior",
            "category_group",
            "location",
            "full_horizon_success",
            "horizon_recall",
            "ade_m",
            "fde_m",
            "mode_count",
            "min_ade_at_1",
            "min_ade_at_5",
            "min_ade_at_10",
            "min_fde_at_1",
            "min_fde_at_5",
            "min_fde_at_10",
            "miss_rate_at_1",
            "miss_rate_at_5",
            "miss_rate_at_10",
            "occupancy_iou",
            "primary_actor_iou",
            "context_iou",
            "closest_approach_distance_error_m",
            "closest_approach_time_error_s",
            "risk_fidelity_score",
            "distance_band",
            "ttc_band",
            "visibility_band",
            "map_relation",
            "occlusion_proxy",
            "challenge_tracks",
            "min_distance_m",
            "min_ttc_s",
            "failure_tags",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_metrics:
            payload = dict(row)
            payload["challenge_tracks"] = "|".join(str(item) for item in list(row.get("challenge_tracks") or []))
            payload["failure_tags"] = "|".join(str(item) for item in list(row.get("failure_tags") or []))
            writer.writerow({field: payload.get(field, "") for field in fieldnames})
    return summary


def build_world_model_comparison(run_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    profiles: List[Dict[str, object]] = []
    behavior_index: Dict[str, Dict[str, object]] = defaultdict(dict)
    track_index: Dict[str, Dict[str, object]] = defaultdict(dict)
    risk_index: Dict[str, Dict[str, Dict[str, object]]] = {field_name: defaultdict(dict) for field_name in RISK_FACET_FIELDS}
    for item in run_summaries:
        overview = dict(item.get("overview") or {})
        profile_name = str(item.get("profile_name") or "profile")
        profiles.append(
            {
                "name": profile_name,
                "label": profile_name.replace("_", "-").title(),
                "case_count": int(overview.get("case_count") or 0),
                "full_horizon_count": int(overview.get("full_horizon_count") or 0),
                "full_horizon_rate": float(overview.get("full_horizon_rate") or 0.0),
                "mean_horizon_recall": float(overview.get("mean_horizon_recall") or 0.0),
                "mean_ade_m": _optional_float(overview.get("mean_ade_m")),
                "mean_fde_m": _optional_float(overview.get("mean_fde_m")),
                "mean_occupancy_iou": float(overview.get("mean_occupancy_iou") or 0.0),
                "mean_primary_actor_iou": float(overview.get("mean_primary_actor_iou") or 0.0),
                "mean_closest_approach_distance_error_m": _optional_float(overview.get("mean_closest_approach_distance_error_m")),
                "mean_closest_approach_time_error_s": _optional_float(overview.get("mean_closest_approach_time_error_s")),
                "mean_risk_fidelity_score": float(overview.get("mean_risk_fidelity_score") or 0.0),
                "mean_min_ade_at_1": _optional_float((item.get("forecast_metrics") or {}).get("mean_min_ade_at_1")),
                "mean_min_ade_at_5": _optional_float((item.get("forecast_metrics") or {}).get("mean_min_ade_at_5")),
                "mean_min_fde_at_1": _optional_float((item.get("forecast_metrics") or {}).get("mean_min_fde_at_1")),
                "mean_min_fde_at_5": _optional_float((item.get("forecast_metrics") or {}).get("mean_min_fde_at_5")),
                "mean_miss_rate_at_1": _optional_float((item.get("forecast_metrics") or {}).get("mean_miss_rate_at_1")),
                "mean_miss_rate_at_5": _optional_float((item.get("forecast_metrics") or {}).get("mean_miss_rate_at_5")),
            }
        )
        for row in list(item.get("behavior_breakdown") or []):
            behavior_index[str(row["behavior"])][profile_name] = dict(row)
        for row in list(item.get("track_breakdown") or []):
            track_index[str(row["track"])][profile_name] = dict(row)
        for field_name, rows in dict(item.get("risk_breakdowns") or {}).items():
            if field_name not in risk_index:
                continue
            for row in list(rows or []):
                risk_value = str(row.get(field_name) or "unknown")
                risk_index[field_name][risk_value][profile_name] = dict(row)

    profiles.sort(key=_profile_sort_key, reverse=True)
    profile_order = [str(row["name"]) for row in profiles]
    full_set_profiles = [dict(row) for row in profiles]

    case_maps = {}
    for item in run_summaries:
        profile_name = str(item.get("profile_name") or "profile")
        case_map = {}
        for row in list(item.get("case_metrics") or []):
            key = str(row.get("benchmark_group") or row.get("reference_case_key") or "")
            if key:
                case_map[key] = dict(row)
        case_maps[profile_name] = case_map
    common_case_keys = set.intersection(*(set(mapping) for mapping in case_maps.values())) if case_maps else set()

    def _common_metric_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        full_horizon_count = sum(1 for row in rows if bool(row.get("full_horizon_success")))
        return {
            "case_count": len(rows),
            "full_horizon_count": full_horizon_count,
            "full_horizon_rate": round(_ratio(full_horizon_count, len(rows)), 4),
            **{"mean_" + name: _metric_mean(row.get(name) for row in rows) for name in [
                "horizon_recall", "ade_m", "fde_m", "occupancy_iou", "primary_actor_iou", "risk_fidelity_score",
                "closest_approach_distance_error_m", "closest_approach_time_error_s",
                "min_ade_at_1", "min_ade_at_5", "min_fde_at_1", "min_fde_at_5", "miss_rate_at_1", "miss_rate_at_5",
            ]},
        }

    common_case_summary = []
    bootstrap_seed = 7
    bootstrap_replicates = 5000
    ordered_common_keys = sorted(common_case_keys)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    clusters = []
    for key in ordered_common_keys:
        scene_ids = {str(case_maps[name][key].get("scene_token") or case_maps[name][key].get("scene_name") or "")
                     for name in profile_order}
        scene_ids.discard("")
        if len(scene_ids) > 1:
            raise ValueError(f"Paired forecast case has inconsistent scene identities: {key}")
        clusters.append(next(iter(scene_ids), key))
    cluster_names = sorted(set(clusters))
    cluster_index = {name: index for index, name in enumerate(cluster_names)}
    cluster_ids = np.asarray([cluster_index[name] for name in clusters], dtype=int)
    weights = np.zeros((bootstrap_replicates, len(ordered_common_keys)), dtype=int)
    if cluster_names:
        sampled_clusters = bootstrap_rng.integers(0, len(cluster_names), size=(bootstrap_replicates, len(cluster_names)))
        cluster_counts = np.zeros((bootstrap_replicates, len(cluster_names)), dtype=int)
        np.add.at(cluster_counts, (np.arange(bootstrap_replicates)[:, None], sampled_clusters), 1)
        weights = cluster_counts[:, cluster_ids]

    def _metric_values(rows: Sequence[Dict[str, object]], name: str) -> np.ndarray:
        return np.asarray([float(row[name]) if row.get(name) is not None else np.nan for row in rows])

    def _interval(values: np.ndarray) -> Optional[Dict[str, object]]:
        finite = np.isfinite(values)
        if not finite.any():
            return None
        valid_weights = weights[:, finite]
        counts = valid_weights.sum(axis=1)
        means = (valid_weights @ values[finite])[counts > 0] / counts[counts > 0]
        low, high = np.percentile(means, [2.5, 97.5])
        return {"mean": round(float(np.mean(values[finite])), 6), "ci95_low": round(float(low), 6),
                "ci95_high": round(float(high), 6), "case_count": int(finite.sum()),
                "cluster_count": len(set(cluster_ids[finite]))}
    for profile_name in profile_order:
        common_rows = [case_maps.get(profile_name, {}).get(key, {}) for key in ordered_common_keys]
        summary_row = {
            "name": profile_name,
            "label": profile_name.replace("_", "-").title(),
            **_common_metric_rows(common_rows),
        }
        metric_uncertainty: Dict[str, Dict[str, float]] = {}
        for metric_name in ["ade_m", "fde_m", "risk_fidelity_score", "occupancy_iou"]:
            interval = _interval(_metric_values(common_rows, metric_name))
            if interval is not None:
                metric_uncertainty[metric_name] = interval
        summary_row["uncertainty"] = metric_uncertainty
        common_case_summary.append(summary_row)

    paired_profile_comparisons = []
    if ordered_common_keys:
        for left_idx, left_name in enumerate(profile_order):
            for right_name in profile_order[left_idx + 1 :]:
                comparison_row: Dict[str, object] = {
                    "profile_a": left_name,
                    "profile_b": right_name,
                    "case_count": len(ordered_common_keys),
                    "deltas": {},
                }
                for metric_name in ["ade_m", "fde_m", "risk_fidelity_score", "occupancy_iou"]:
                    left_values = _metric_values([case_maps[left_name][key] for key in ordered_common_keys], metric_name)
                    right_values = _metric_values([case_maps[right_name][key] for key in ordered_common_keys], metric_name)
                    interval = _interval(right_values - left_values)
                    if interval is not None:
                        interval["profile_b_minus_profile_a"] = interval.pop("mean")
                        comparison_row["deltas"][metric_name] = interval
                paired_profile_comparisons.append(comparison_row)

    if common_case_keys:
        common_by_name = {str(row["name"]): dict(row) for row in common_case_summary}
        profiles = [
            {
                **row,
                **common_by_name[str(row["name"])],
                "full_set_case_count": int(row.get("case_count") or 0),
            }
            for row in full_set_profiles
        ]
        profiles.sort(key=_profile_sort_key, reverse=True)
        profile_order = [str(row["name"]) for row in profiles]

    behavior_matrix = []
    for behavior in sorted(behavior_index):
        behavior_matrix.append(
            {
                "behavior": behavior,
                "cells": [
                    dict(
                        behavior_index[behavior].get(
                            name,
                            {
                                "case_count": 0,
                                "mean_risk_fidelity_score": 0.0,
                                "mean_occupancy_iou": 0.0,
                            },
                        )
                    )
                    for name in profile_order
                ],
            }
        )

    track_matrix = []
    for track in sorted(track_index):
        track_matrix.append(
            {
                "track": track,
                "cells": [
                    dict(
                        track_index[track].get(
                            name,
                            {
                                "case_count": 0,
                                "mean_risk_fidelity_score": 0.0,
                                "mean_occupancy_iou": 0.0,
                            },
                        )
                    )
                    for name in profile_order
                ],
            }
        )

    risk_matrices = {}
    for field_name in RISK_FACET_FIELDS:
        rows = []
        for risk_value in sorted(risk_index[field_name]):
            rows.append(
                {
                    "group": risk_value,
                    "cells": [
                        dict(
                            risk_index[field_name][risk_value].get(
                                name,
                                {
                                    "case_count": 0,
                                    "mean_risk_fidelity_score": 0.0,
                                    "mean_occupancy_iou": 0.0,
                                },
                            )
                        )
                        for name in profile_order
                    ],
                }
            )
        risk_matrices[field_name] = rows

    return {
        "overview": {
            "profile_count": len(profiles),
            "case_count": len(common_case_keys) if common_case_keys else (int(profiles[0]["case_count"]) if profiles else 0),
            "common_case_count": len(common_case_keys),
            "comparison_basis": "common_case_intersection" if common_case_keys else "profile_specific_cases",
            "uncertainty_method": "paired cluster-level percentile bootstrap" if common_case_keys else "none",
            "bootstrap_cluster_key": "scene_token, scene_name, then case_key",
            "bootstrap_cluster_count": len(cluster_names),
            "bootstrap_seed": bootstrap_seed if common_case_keys else None,
            "bootstrap_replicates": bootstrap_replicates if common_case_keys else 0,
        },
        "profiles": profiles,
        "full_set_profiles": full_set_profiles,
        "common_case_summary": common_case_summary,
        "paired_profile_comparisons": paired_profile_comparisons,
        "behavior_matrix": behavior_matrix,
        "track_matrix": track_matrix,
        "risk_matrices": risk_matrices,
    }


def write_world_model_comparison(comparison: Dict[str, object], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "world_model_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Scenario-Conditioned World-Model Comparison",
        "",
        "- Profiles: {0}".format(comparison["overview"]["profile_count"]),
        "- Cases: {0}".format(comparison["overview"]["case_count"]),
        "- Common cases across all profiles: {0}".format(comparison["overview"].get("common_case_count", 0)),
        "- Primary comparison basis: `{0}`".format(comparison["overview"].get("comparison_basis", "")),
        "",
        "## Profile Overview",
        "",
        "| Profile | Full Horizon | Risk Fidelity | ADE | MinADE@1 | MinADE@5 | MissRate@5 | Occupancy IoU | Closest-Approach Time Error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(comparison.get("profiles") or []):
        lines.append(
            "| {0} | {1}/{2} ({3:.1%}) | {4} | {5} | {6} | {7} | {8} | {9} | {10} |".format(
                row["label"],
                row["full_horizon_count"],
                row["case_count"],
                row["full_horizon_rate"],
                *[_format_metric(row[name]) for name in [
                    "mean_risk_fidelity_score", "mean_ade_m", "mean_min_ade_at_1", "mean_min_ade_at_5",
                    "mean_miss_rate_at_5", "mean_occupancy_iou", "mean_closest_approach_time_error_s",
                ]],
            )
        )
    lines.extend(
        [
            "",
            "## Common-Case Comparison",
            "",
            "All rows below use the intersection of case keys across profiles.",
            "",
            "| Profile | Cases | Full Horizon | ADE | FDE | Risk Fidelity | Occupancy IoU |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(comparison.get("common_case_summary") or []):
        lines.append(
            "| {0} | {1} | {2}/{1} ({3:.1%}) | {4} | {5} | {6} | {7} |".format(
                row["label"],
                row["case_count"],
                row["full_horizon_count"],
                row["full_horizon_rate"],
                *[_format_metric(row[name]) for name in ["mean_ade_m", "mean_fde_m", "mean_risk_fidelity_score", "mean_occupancy_iou"]],
            )
        )
    lines.extend(
        [
            "",
            "## Paired Uncertainty",
            "",
            "Deltas are profile B minus profile A on the same cases. Lower ADE is better; higher risk fidelity is better.",
            "",
            "| Profile A | Profile B | ADE Delta [95% CI] | Risk-Fidelity Delta [95% CI] |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in list(comparison.get("paired_profile_comparisons") or []):
        ade = dict(dict(row.get("deltas") or {}).get("ade_m") or {})
        risk = dict(dict(row.get("deltas") or {}).get("risk_fidelity_score") or {})
        lines.append(
            "| {0} | {1} | {2} [{3}, {4}] | {5} [{6}, {7}] |".format(
                str(row.get("profile_a") or "").replace("_", "-").title(),
                str(row.get("profile_b") or "").replace("_", "-").title(),
                *[_format_metric(values.get(name)) for values in [ade, risk]
                  for name in ["profile_b_minus_profile_a", "ci95_low", "ci95_high"]],
            )
        )
    lines.extend(["", "## Challenge Track Matrix", "", "| Track | " + " | ".join(str(row["label"]) for row in list(comparison.get("profiles") or [])) + " |", "| --- | " + " | ".join("---" for _ in list(comparison.get("profiles") or [])) + " |"])
    for row in list(comparison.get("track_matrix") or []):
        lines.append(
            "| {0} | {1} |".format(
                row["track"],
                " | ".join(
                    "{0:.3f}, {1:.3f}".format(
                        float(cell.get("mean_risk_fidelity_score") or 0.0),
                        float(cell.get("mean_occupancy_iou") or 0.0),
                    )
                    for cell in list(row.get("cells") or [])
                ),
            )
        )
    (output_dir / "world_model_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "world_model_comparison_summary.html").write_text(
        WORLD_MODEL_COMPARISON_TEMPLATE.render(comparison=comparison, metric=_format_metric),
        encoding="utf-8",
    )

    with (output_dir / "world_model_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "label",
                "case_count",
                "full_horizon_count",
                "full_horizon_rate",
                "mean_horizon_recall",
                "mean_ade_m",
                "mean_fde_m",
                "mean_occupancy_iou",
                "mean_primary_actor_iou",
                "mean_closest_approach_distance_error_m",
                "mean_closest_approach_time_error_s",
                "mean_risk_fidelity_score",
                "mean_min_ade_at_1",
                "mean_min_ade_at_5",
                "mean_min_fde_at_1",
                "mean_min_fde_at_5",
                "mean_miss_rate_at_1",
                "mean_miss_rate_at_5",
            ],
        )
        writer.writeheader()
        for row in list(comparison.get("profiles") or []):
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def compare_world_model_evaluations(evaluation_dirs: Sequence[Path], output_dir: Path) -> Dict[str, object]:
    summaries = []
    for eval_dir in evaluation_dirs:
        payload = _load_json(Path(eval_dir).resolve() / "world_model_metrics.json")
        payload["output_dir"] = str(Path(eval_dir).resolve())
        summaries.append(payload)
    comparison = build_world_model_comparison(summaries)
    write_world_model_comparison(comparison, output_dir)
    return {
        "output_dir": str(output_dir.resolve()),
        "profile_count": int(comparison["overview"]["profile_count"]),
        "case_count": int(comparison["overview"]["case_count"]),
    }


def run_proxy_world_model_study(benchmark_path: Path, output_dir: Path) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    for profile_name in WORLD_MODEL_PROXY_PROFILES:
        prediction_path = output_dir / "predictions" / "{0}.json".format(profile_name)
        generate_proxy_world_model_predictions(benchmark_path, prediction_path, profile_name)
        profile_output = output_dir / profile_name
        summary = evaluate_world_model_predictions(
            benchmark_path=benchmark_path,
            predictions_path=prediction_path,
            output_dir=profile_output,
            profile_name=profile_name,
        )
        run_summaries.append(summary)
    comparison = build_world_model_comparison(run_summaries)
    write_world_model_comparison(comparison, output_dir)
    return {
        "output_dir": str(output_dir),
        "profile_count": len(run_summaries),
        "case_count": int(comparison["overview"]["case_count"]),
    }


def _json_bytes(payload: Dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _write_jsonl_replay(case: Dict[str, object], case_output_path: Path) -> int:
    message_count = 0
    lines = []
    metadata_message = {
        "topic": "/nusc_scene_agent/metadata",
        "log_time_us": _safe_int(case.get("history_frames", [{}])[-1].get("timestamp_us")),
        "data": {
            "benchmark_group": str(case["benchmark_group"]),
            "scene_name": str(case["scene_name"]),
            "primary_behavior": str(case["primary_behavior"]),
            "category_group": str(case["category_group"]),
            "risk_facets": dict(case.get("risk_facets") or {}),
        },
    }
    lines.append(json.dumps(metadata_message, ensure_ascii=False))
    message_count += 1

    for topic, frames in [
        ("/nusc_scene_agent/history_track", list(case.get("history_frames") or [])),
        ("/nusc_scene_agent/future_track", list(case.get("future_frames") or [])),
        ("/nusc_scene_agent/future_occupancy", list(case.get("future_occupancy") or [])),
    ]:
        for frame in frames:
            lines.append(
                json.dumps(
                    {
                        "topic": topic,
                        "log_time_us": _safe_int(frame.get("timestamp_us")),
                        "data": frame,
                    },
                    ensure_ascii=False,
                )
            )
            message_count += 1
    case_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return message_count


def _write_mcap_replay(case: Dict[str, object], case_output_path: Path) -> int:
    try:
        from mcap.writer import Writer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MCAP export requires the optional 'mcap' package.") from exc

    message_count = 0
    with case_output_path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        schema_id = writer.register_schema(
            name="nusc_scene_agent_json",
            encoding="jsonschema",
            data=_json_bytes({"type": "object"}),
        )
        channel_ids = {
            topic: writer.register_channel(topic=topic, message_encoding="json", schema_id=schema_id)
            for topic in [
                "/nusc_scene_agent/metadata",
                "/nusc_scene_agent/history_track",
                "/nusc_scene_agent/future_track",
                "/nusc_scene_agent/future_occupancy",
            ]
        }
        metadata = {
            "benchmark_group": str(case["benchmark_group"]),
            "scene_name": str(case["scene_name"]),
            "primary_behavior": str(case["primary_behavior"]),
            "category_group": str(case["category_group"]),
            "risk_facets": dict(case.get("risk_facets") or {}),
        }
        log_time = _safe_int(case.get("history_frames", [{}])[-1].get("timestamp_us"))
        writer.add_message(
            channel_id=channel_ids["/nusc_scene_agent/metadata"],
            log_time=log_time,
            publish_time=log_time,
            data=_json_bytes(metadata),
        )
        message_count += 1
        for topic, frames in [
            ("/nusc_scene_agent/history_track", list(case.get("history_frames") or [])),
            ("/nusc_scene_agent/future_track", list(case.get("future_frames") or [])),
            ("/nusc_scene_agent/future_occupancy", list(case.get("future_occupancy") or [])),
        ]:
            for frame in frames:
                frame_time = _safe_int(frame.get("timestamp_us"))
                writer.add_message(
                    channel_id=channel_ids[topic],
                    log_time=frame_time,
                    publish_time=frame_time,
                    data=_json_bytes(frame),
                )
                message_count += 1
        writer.finish()
    return message_count


def export_world_model_replay(
    benchmark_path: Path,
    output_dir: Path,
    export_format: str = "jsonl",
) -> Dict[str, object]:
    if export_format not in {"jsonl", "mcap"}:
        raise ValueError("Unsupported export_format: {0}".format(export_format))
    benchmark = _load_json(benchmark_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases = []
    total_messages = 0
    for case in list(benchmark.get("cases") or []):
        stem = str(case["benchmark_group"])
        suffix = ".mcap" if export_format == "mcap" else ".jsonl"
        case_output_path = output_dir / "{0}{1}".format(stem, suffix)
        if export_format == "mcap":
            message_count = _write_mcap_replay(case, case_output_path)
        else:
            message_count = _write_jsonl_replay(case, case_output_path)
        manifest_cases.append(
            {
                "benchmark_group": stem,
                "scene_name": str(case["scene_name"]),
                "primary_behavior": str(case["primary_behavior"]),
                "path": str(case_output_path),
                "message_count": message_count,
            }
        )
        total_messages += message_count

    manifest = {
        "metadata": {
            "generator": "world_model_replay_export_v1",
            "source_benchmark": str(benchmark_path),
            "export_format": export_format,
            "case_count": len(manifest_cases),
            "message_count": total_messages,
        },
        "cases": manifest_cases,
    }
    (output_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "case_count": len(manifest_cases),
        "message_count": total_messages,
        "export_format": export_format,
    }
