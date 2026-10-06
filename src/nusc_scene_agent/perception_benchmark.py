from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from jinja2 import Template

from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.geometry import global_xy_to_anchor_ego
from nusc_scene_agent.map_context import build_case_map_context
from nusc_scene_agent.models import RetrievalCandidate


PERCEPTION_SUMMARY_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Scenario-Conditioned Perception Evaluation</title>
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
  <h1>Scenario-Conditioned Perception Evaluation</h1>
  <div class="meta">
    Profile: {{ summary.profile_name }} |
    Cases: {{ summary.overview.case_count }} |
    Anchor Recall: {{ summary.overview.anchor_recall_count }}/{{ summary.overview.case_count }} ({{ "%.1f"|format(summary.overview.anchor_recall_rate * 100.0) }}%) |
    Full Track: {{ summary.overview.full_track_count }}/{{ summary.overview.case_count }} ({{ "%.1f"|format(summary.overview.full_track_rate * 100.0) }}%)
  </div>

  <h2>Overview</h2>
  <table>
    <tbody>
      <tr><th>Mean Event Recall</th><td>{{ "%.3f"|format(summary.overview.mean_event_recall) }}</td></tr>
      <tr><th>Mean Contiguous Coverage</th><td>{{ "%.3f"|format(summary.overview.mean_contiguous_coverage) }}</td></tr>
      <tr><th>Mean Center Error</th><td>{{ "%.3f"|format(summary.overview.mean_center_error_m) }}</td></tr>
      <tr><th>Mean First Match Lag</th><td>{{ "%.2f"|format(summary.overview.mean_first_match_lag_frames) }}</td></tr>
      <tr><th>Perfect Cases</th><td>{{ summary.overview.perfect_case_count }}</td></tr>
    </tbody>
  </table>

  <h2>Behavior Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Behavior</th><th>Cases</th><th>Anchor Recall</th><th>Full Track</th><th>Mean Event Recall</th><th>Mean Center Error</th><th>Top Failure Modes</th>
      </tr>
    </thead>
    <tbody>
    {% for row in summary.behavior_breakdown %}
      <tr>
        <td>{{ row.behavior }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.anchor_recall_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.anchor_recall_rate * 100.0) }}%)</td>
        <td>{{ row.full_track_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_track_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_event_recall) }}</td>
        <td>{{ "%.3f"|format(row.mean_center_error_m) }}</td>
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
        <th>Group</th><th>Cases</th><th>Anchor Recall</th><th>Full Track</th><th>Mean Event Recall</th><th>Mean Center Error</th><th>Top Failure Modes</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row[facet_name] }}</td>
        <td>{{ row.case_count }}</td>
        <td>{{ row.anchor_recall_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.anchor_recall_rate * 100.0) }}%)</td>
        <td>{{ row.full_track_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_track_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_event_recall) }}</td>
        <td>{{ "%.3f"|format(row.mean_center_error_m) }}</td>
        <td>{{ row.top_failure_summary }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endfor %}

  <h2>Case Breakdown</h2>
  <table>
    <thead>
      <tr>
        <th>Group</th><th>Behavior</th><th>Actor</th><th>Anchor</th><th>Event Recall</th><th>Contiguous Coverage</th><th>Center Error</th><th>Distance Band</th><th>TTC Band</th><th>Map Relation</th><th>Occlusion Proxy</th><th>Failure Tags</th>
      </tr>
    </thead>
    <tbody>
    {% for row in summary.case_metrics %}
      <tr>
        <td>{{ row.benchmark_group }}</td>
        <td>{{ row.primary_behavior }}</td>
        <td>{{ row.category_group }}</td>
        <td>{{ row.anchor_detected }}</td>
        <td>{{ "%.3f"|format(row.event_recall) }}</td>
        <td>{{ "%.3f"|format(row.contiguous_coverage) }}</td>
        <td>{{ "%.3f"|format(row.mean_center_error_m) if row.mean_center_error_m is not none else "-" }}</td>
        <td>{{ row.distance_band }}</td>
        <td>{{ row.ttc_band }}</td>
        <td>{{ row.map_relation }}</td>
        <td>{{ row.occlusion_proxy }}</td>
        <td>{{ ", ".join(row.failure_tags) if row.failure_tags else "none" }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"""
)


PERCEPTION_COMPARISON_TEMPLATE = Template(
    """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Scenario-Conditioned Perception Comparison</title>
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
  <h1>Scenario-Conditioned Perception Comparison</h1>
  <div class="meta">
    Profiles: {{ comparison.overview.profile_count }} |
    Cases: {{ comparison.overview.case_count }}
  </div>

  <h2>Profile Overview</h2>
  <table>
    <thead>
      <tr>
        <th>Profile</th><th>Anchor Recall</th><th>Full Track</th><th>Mean Event Recall</th><th>Mean Contiguous Coverage</th><th>Mean Center Error</th>
      </tr>
    </thead>
    <tbody>
    {% for row in comparison.profiles %}
      <tr>
        <td>{{ row.label }}</td>
        <td>{{ row.anchor_recall_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.anchor_recall_rate * 100.0) }}%)</td>
        <td>{{ row.full_track_count }}/{{ row.case_count }} ({{ "%.1f"|format(row.full_track_rate * 100.0) }}%)</td>
        <td>{{ "%.3f"|format(row.mean_event_recall) }}</td>
        <td>{{ "%.3f"|format(row.mean_contiguous_coverage) }}</td>
        <td>{{ "%.3f"|format(row.mean_center_error_m) }}</td>
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
        <td>{{ cell.anchor_recall_count }}/{{ cell.case_count }} | {{ "%.3f"|format(cell.mean_event_recall) }}</td>
        {% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>

  {% for facet_name, rows in comparison.risk_matrices.items() %}
  <h2>{{ facet_name|replace('_', ' ')|title }}</h2>
  <table>
    <thead>
      <tr>
        <th>Group</th>
        {% for row in comparison.profiles %}
        <th>{{ row.label }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.group }}</td>
        {% for cell in row.cells %}
        <td>{{ cell.anchor_recall_count }}/{{ cell.case_count }} | {{ "%.3f"|format(cell.mean_event_recall) }}</td>
        {% endfor %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% endfor %}
</body>
</html>
"""
)


CATEGORY_ALIASES = {
    "car": "vehicle",
    "vehicle": "vehicle",
    "construction_vehicle": "truck",
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "trailer": "truck",
}

PROXY_PERCEPTION_PROFILES = [
    "oracle_tracking",
    "delayed_track",
    "crossing_sparse_track",
]

PREDICTION_COVERAGE_MODES = [
    "anchor",
    "any_frame",
    "full_window",
]

RISK_FACET_FIELDS = [
    "distance_band",
    "ttc_band",
    "visibility_band",
    "map_relation",
    "occlusion_proxy",
]


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


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_case_key(case_key: str) -> Tuple[str, str]:
    sample_token, instance_token = str(case_key).split(":", 1)
    return sample_token, instance_token


def _unique(items: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        value = str(item)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _load_anchor_specs(config_path: Path) -> List[Dict[str, object]]:
    specs = load_benchmark_config(config_path)
    grouped: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        if not spec.reference_case_keys or spec.expect_match is not True:
            continue
        group = str(spec.benchmark_group or spec.id)
        bucket = grouped.setdefault(
            group,
            {
                "benchmark_group": group,
                "reference_case_key": str(spec.reference_case_keys[0]),
                "reference_scene_name": str(spec.reference_scene_names[0]) if spec.reference_scene_names else "",
                "reference_instance_token": str(spec.reference_instance_tokens[0]) if spec.reference_instance_tokens else "",
                "reference_event_sample_range": list(spec.reference_event_sample_range),
                "reference_peak_sample_idx": spec.reference_peak_sample_idx,
                "source_query_ids": [],
                "source_query_texts": [],
                "behaviors": [],
                "actors": [],
                "tags": [],
            },
        )
        bucket["source_query_ids"].append(str(spec.id))
        bucket["source_query_texts"].append(str(spec.natural_language))
        bucket["behaviors"] = _unique(list(bucket["behaviors"]) + list(spec.behaviors))
        bucket["actors"] = _unique(list(bucket["actors"]) + list(spec.actors))
        bucket["tags"] = _unique(list(bucket["tags"]) + list(spec.tags))
    return [grouped[key] for key in sorted(grouped)]


def _primary_behavior(behaviors: Sequence[str]) -> str:
    ordered = _unique(behaviors)
    if ordered:
        return ordered[0]
    return "proximity"


def _profile_label(profile_name: str) -> str:
    return profile_name.replace("_", "-").title()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute("PRAGMA table_info({0})".format(table_name)).fetchall()
    return {str(row[1]) for row in rows}


def _distance_band(min_distance_m: float) -> str:
    if min_distance_m <= 3.0:
        return "critical_range"
    if min_distance_m <= 8.0:
        return "near_range"
    return "extended_range"


def _ttc_band(min_ttc_s: Optional[float]) -> str:
    if min_ttc_s is None or not np.isfinite(float(min_ttc_s)):
        return "ttc_unavailable"
    if float(min_ttc_s) <= 1.5:
        return "urgent_ttc"
    if float(min_ttc_s) <= 3.0:
        return "elevated_ttc"
    return "moderate_ttc"


def _visibility_band(visibility: Optional[int]) -> str:
    if visibility is None or int(visibility) <= 0:
        return "visibility_unknown"
    if int(visibility) <= 2:
        return "low_visibility"
    if int(visibility) == 3:
        return "partial_visibility"
    return "clear_visibility"


def _map_relation(primary_behavior: str, map_context: Dict[str, object]) -> str:
    if bool(map_context.get("actor_on_crosswalk_any")):
        return "crosswalk_supported"
    if bool(map_context.get("shares_lane_at_anchor")):
        return "shared_lane_supported"
    if bool(map_context.get("actor_uses_ego_lane_any")):
        return "lane_overlap_supported"
    if primary_behavior == "crossing":
        return "crosswalk_like"
    if primary_behavior == "cut_in":
        return "merge_like"
    if primary_behavior == "oncoming":
        return "opposite_direction_like"
    if primary_behavior == "stopped_lead":
        return "lead_lane_like"
    return "generic_proximity"


def _occlusion_proxy(
    category_group: str,
    primary_behavior: str,
    visibility_band: str,
    map_relation: str,
) -> str:
    normalized_category = _normalize_category(category_group)
    if visibility_band in {"low_visibility", "partial_visibility"}:
        return "visibility_limited"
    if normalized_category in {"bus", "truck"} and primary_behavior == "stopped_lead":
        return "large_lead_occluder"
    if normalized_category in {"bus", "truck"}:
        return "large_actor"
    if primary_behavior == "crossing" and map_relation in {"crosswalk_supported", "crosswalk_like"}:
        return "crossing_emergence"
    return "nominal_exposure"


def _minimal_case_candidate(
    anchor_row: Dict[str, object],
    sample_token: str,
    instance_token: str,
    location: str,
    category_name: str,
    category_group: str,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        ann_token=str(anchor_row.get("ann_token") or ""),
        sample_token=sample_token,
        scene_token=str(anchor_row.get("scene_token") or ""),
        scene_name=str(anchor_row.get("scene_name") or ""),
        sample_idx=_safe_int(anchor_row.get("sample_idx")),
        instance_token=instance_token,
        category_name=category_name,
        category_group=category_group,
        location=location,
        distance=_safe_float(anchor_row.get("distance")),
        ttc=(
            float(anchor_row["ttc"])
            if anchor_row.get("ttc") is not None and np.isfinite(_safe_float(anchor_row.get("ttc"), float("nan")))
            else float("inf")
        ),
        x_ego=_safe_float(anchor_row.get("x_ego")),
        y_ego=_safe_float(anchor_row.get("y_ego")),
        speed=_safe_float(anchor_row.get("speed")),
        rel_vx=_safe_float(anchor_row.get("rel_vx")),
        rel_vy=_safe_float(anchor_row.get("rel_vy")),
        heading_delta=_safe_float(anchor_row.get("heading_delta")),
        retrieval_score=0.0,
        scene_description=str(anchor_row.get("scene_description") or ""),
        num_lidar_pts=_safe_int(anchor_row.get("num_lidar_pts")),
        num_radar_pts=_safe_int(anchor_row.get("num_radar_pts")),
    )


def _load_case_support(
    conn: sqlite3.Connection,
    case: Dict[str, object],
    agent_columns: set[str],
    sample_columns: set[str],
) -> Dict[str, object]:
    select_terms = [
        "a.sample_token",
        "a.sample_idx",
        "a.x_ego",
        "a.y_ego",
        "a.distance",
    ]
    optional_agent_columns = [
        "ann_token",
        "ttc",
        "speed",
        "rel_vx",
        "rel_vy",
        "heading_delta",
        "visibility",
        "num_lidar_pts",
        "num_radar_pts",
    ]
    optional_sample_columns = [
        "timestamp_us",
        "ego_x",
        "ego_y",
        "ego_yaw",
        "scene_description",
    ]
    for column in optional_agent_columns:
        if column in agent_columns:
            select_terms.append("a.{0}".format(column))
    for column in optional_sample_columns:
        if column in sample_columns:
            select_terms.append("s.{0}".format(column))

    timeline_rows = conn.execute(
        """
        SELECT {0}
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE a.scene_token = ?
          AND a.instance_token = ?
          AND a.sample_idx BETWEEN ? AND ?
        ORDER BY a.sample_idx ASC
        """.format(", ".join(select_terms)),
        (
            str(case["scene_token"]),
            str(case["instance_token"]),
            int(case["event_start_sample_idx"]),
            int(case["event_end_sample_idx"]),
        ),
    ).fetchall()
    if not timeline_rows:
        return {}

    timeline_frame_rows = [dict(row) for row in timeline_rows]
    min_distance = min(_safe_float(row.get("distance")) for row in timeline_frame_rows)
    finite_ttc_values = [
        _safe_float(row.get("ttc"), float("nan"))
        for row in timeline_frame_rows
        if row.get("ttc") is not None and np.isfinite(_safe_float(row.get("ttc"), float("nan")))
    ]
    min_ttc = min(finite_ttc_values) if finite_ttc_values else None

    anchor_row = next(
        (
            row
            for row in timeline_frame_rows
            if str(row.get("sample_token") or "") == str(case["anchor_sample_token"])
        ),
        timeline_frame_rows[min(len(timeline_frame_rows) - 1, max(0, len(timeline_frame_rows) // 2))],
    )

    visibility_band = _visibility_band(
        _safe_int(anchor_row.get("visibility")) if anchor_row.get("visibility") is not None else None
    )
    map_context: Dict[str, object] = {"available": False}
    if {"ego_x", "ego_y", "ego_yaw"}.issubset(sample_columns):
        timeline_df = pd.DataFrame(timeline_frame_rows)
        ego_rows = conn.execute(
            """
            SELECT sample_token, sample_idx, ego_x, ego_y, ego_yaw
            FROM samples
            WHERE scene_token = ?
              AND sample_idx BETWEEN ? AND ?
            ORDER BY sample_idx ASC
            """,
            (
                str(case["scene_token"]),
                int(case["event_start_sample_idx"]),
                int(case["event_end_sample_idx"]),
            ),
        ).fetchall()
        ego_window = pd.DataFrame([dict(row) for row in ego_rows])
        candidate = _minimal_case_candidate(
            anchor_row=anchor_row,
            sample_token=str(case["anchor_sample_token"]),
            instance_token=str(case["instance_token"]),
            location=str(case["location"]),
            category_name=str(case["category_name"]),
            category_group=str(case["category_group"]),
        )
        if not timeline_df.empty and not ego_window.empty:
            try:
                map_context, _ = build_case_map_context(
                    conn,
                    candidate,
                    timeline_df,
                    ego_window,
                    query_behaviors=list(case.get("behaviors") or []),
                    include_patch_geometries=False,
                )
            except Exception:  # noqa: BLE001
                map_context = {"available": False, "reason": "map_context_error"}

    primary_behavior = str(case["primary_behavior"])
    map_relation = _map_relation(primary_behavior, map_context)
    risk_facets = {
        "distance_band": _distance_band(min_distance),
        "ttc_band": _ttc_band(min_ttc),
        "visibility_band": visibility_band,
        "map_relation": map_relation,
        "occlusion_proxy": _occlusion_proxy(
            category_group=str(case["category_group"]),
            primary_behavior=primary_behavior,
            visibility_band=visibility_band,
            map_relation=map_relation,
        ),
    }
    map_support = {
        "available": bool(map_context.get("available")),
        "actor_on_crosswalk_any": bool(map_context.get("actor_on_crosswalk_any")),
        "actor_on_walkway_any": bool(map_context.get("actor_on_walkway_any")),
        "actor_uses_ego_lane_any": bool(map_context.get("actor_uses_ego_lane_any")),
        "shares_lane_at_anchor": bool(map_context.get("shares_lane_at_anchor")),
    }
    return {
        "anchor_visibility": _safe_int(anchor_row.get("visibility")) if anchor_row.get("visibility") is not None else None,
        "min_distance_m": round(min_distance, 3),
        "min_ttc_s": round(min_ttc, 3) if min_ttc is not None else None,
        "map_support": map_support,
        "risk_facets": risk_facets,
    }


def generate_perception_benchmark_from_scenario_config(
    config_path: Path,
    db_path: Path,
    output_path: Path,
) -> Dict[str, object]:
    anchors = _load_anchor_specs(config_path)
    conn = sqlite3.connect(str(db_path.resolve()))
    conn.row_factory = sqlite3.Row
    agent_columns = _table_columns(conn, "agents")
    sample_columns = _table_columns(conn, "samples")
    cases: List[Dict[str, object]] = []
    try:
        for anchor in anchors:
            sample_token, instance_token = _parse_case_key(str(anchor["reference_case_key"]))
            anchor_row = conn.execute(
                """
                SELECT a.scene_token, a.scene_name, a.sample_idx, a.category_name, a.category_group, s.location
                FROM agents a
                JOIN samples s ON s.sample_token = a.sample_token
                WHERE a.sample_token = ? AND a.instance_token = ?
                LIMIT 1
                """,
                (sample_token, instance_token),
            ).fetchone()
            if anchor_row is None:
                continue

            event_range = list(anchor["reference_event_sample_range"])
            if len(event_range) != 2:
                continue

            rows = conn.execute(
                """
                SELECT
                    a.sample_token,
                    a.sample_idx,
                    s.timestamp_us,
                    a.x_ego,
                    a.y_ego,
                    a.distance
                FROM agents a
                JOIN samples s ON s.sample_token = a.sample_token
                WHERE a.scene_token = ?
                  AND a.instance_token = ?
                  AND a.sample_idx BETWEEN ? AND ?
                ORDER BY a.sample_idx ASC
                """,
                (
                    str(anchor_row["scene_token"]),
                    instance_token,
                    min(int(event_range[0]), int(anchor_row["sample_idx"])),
                    max(int(event_range[1]), int(anchor_row["sample_idx"])),
                ),
            ).fetchall()
            if not rows:
                continue

            frames = [
                {
                    "sample_token": str(row["sample_token"]),
                    "sample_idx": int(row["sample_idx"]),
                    "timestamp_us": int(row["timestamp_us"]),
                    "x_ego": _safe_float(row["x_ego"]),
                    "y_ego": _safe_float(row["y_ego"]),
                    "distance": _safe_float(row["distance"]),
                }
                for row in rows
            ]
            anchor_sample_idx = int(anchor_row["sample_idx"])
            case = {
                "benchmark_group": str(anchor["benchmark_group"]),
                "reference_case_key": str(anchor["reference_case_key"]),
                "reference_scene_name": str(anchor["reference_scene_name"]),
                "scene_name": str(anchor_row["scene_name"]),
                "scene_token": str(anchor_row["scene_token"]),
                "instance_token": instance_token,
                "category_name": str(anchor_row["category_name"]),
                "category_group": _normalize_category(str(anchor_row["category_group"])),
                "location": str(anchor_row["location"]),
                "primary_behavior": _primary_behavior(list(anchor["behaviors"])),
                "behaviors": list(anchor["behaviors"]),
                "actors": list(anchor["actors"]),
                "tags": list(anchor["tags"]),
                "source_query_ids": list(anchor["source_query_ids"]),
                "source_query_texts": list(anchor["source_query_texts"]),
                "anchor_sample_token": sample_token,
                "anchor_sample_idx": anchor_sample_idx,
                "event_start_sample_idx": int(event_range[0]),
                "event_end_sample_idx": int(event_range[1]),
                "event_peak_sample_idx": _safe_int(anchor["reference_peak_sample_idx"]),
                "frame_count": len(frames),
                "frames": frames,
            }
            case.update(_load_case_support(conn, case, agent_columns=agent_columns, sample_columns=sample_columns))
            cases.append(case)
    finally:
        conn.close()

    payload = {
        "metadata": {
            "generator": "perception_slice_benchmark_generator_v1",
            "source_scenario_benchmark": str(config_path),
            "db_path": str(db_path),
            "case_count": len(cases),
        },
        "cases": cases,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "case_count": len(cases)}


def generate_proxy_perception_predictions(
    benchmark_path: Path,
    output_path: Path,
    profile_name: str,
) -> Dict[str, object]:
    if profile_name not in PROXY_PERCEPTION_PROFILES:
        raise ValueError(
            "Unknown proxy profile: {0}. Expected one of {1}.".format(
                profile_name,
                ", ".join(PROXY_PERCEPTION_PROFILES),
            )
        )
    payload = _load_json(benchmark_path)
    cases = list(payload.get("cases") or [])
    predictions: List[Dict[str, object]] = []

    for case in cases:
        behavior = str(case["primary_behavior"])
        track_id = "track_" + str(case["benchmark_group"])
        frames = list(case["frames"])
        for idx, frame in enumerate(frames):
            keep = True
            x_offset = 0.0
            y_offset = 0.0

            if profile_name == "oracle_tracking":
                keep = True
            elif profile_name == "delayed_track":
                keep = idx >= max(1, len(frames) // 3)
                x_offset = 0.35
                y_offset = -0.2
            elif profile_name == "crossing_sparse_track":
                if behavior == "crossing":
                    keep = idx % 2 == 0
                    x_offset = 0.55
                    y_offset = 0.25
                else:
                    keep = True

            if not keep:
                continue

            predictions.append(
                {
                    "sample_token": str(frame["sample_token"]),
                    "sample_idx": int(frame["sample_idx"]),
                    "track_id": track_id,
                    "category_group": str(case["category_group"]),
                    "x_ego": round(_safe_float(frame["x_ego"]) + x_offset, 3),
                    "y_ego": round(_safe_float(frame["y_ego"]) + y_offset, 3),
                    "score": 0.95,
                }
            )

    output = {
        "metadata": {
            "generator": "proxy_perception_predictions_v1",
            "profile_name": profile_name,
            "source_benchmark": str(benchmark_path),
        },
        "predictions": predictions,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output_path": str(output_path), "prediction_count": len(predictions), "profile_name": profile_name}


def _sample_pose_index(conn: sqlite3.Connection, sample_tokens: Sequence[str]) -> Dict[str, Dict[str, float]]:
    if not sample_tokens:
        return {}
    placeholders = ", ".join("?" for _ in sample_tokens)
    rows = conn.execute(
        """
        SELECT sample_token, sample_idx, scene_token, ego_x, ego_y, ego_yaw
        FROM samples
        WHERE sample_token IN ({0})
        """.format(placeholders),
        tuple(sample_tokens),
    ).fetchall()
    return {
        str(row["sample_token"]): {
            "sample_idx": _safe_int(row["sample_idx"]),
            "ego_x": _safe_float(row["ego_x"]),
            "ego_y": _safe_float(row["ego_y"]),
            "ego_yaw": _safe_float(row["ego_yaw"]),
            "scene_token": str(row["scene_token"]),
        }
        for row in rows
    }


def _link_detection_predictions(
    rows: Sequence[Dict[str, object]],
    max_link_distance_m: float = 3.0,
) -> List[Dict[str, object]]:
    by_scene: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row.get("scene_token") or "")].append(dict(row))

    linked: List[Dict[str, object]] = []
    for scene_token, scene_rows in by_scene.items():
        ordered = sorted(
            scene_rows,
            key=lambda item: (
                _safe_int(item.get("sample_idx")),
                str(item.get("category_group") or ""),
                -_safe_float(item.get("score")),
            ),
        )
        active_tracks: List[Dict[str, object]] = []
        next_track_id = 0
        for row in ordered:
            row_copy = dict(row)
            sample_idx = _safe_int(row_copy.get("sample_idx"))
            category_group = _normalize_category(str(row_copy.get("category_group") or ""))
            best_match = None
            best_distance = None
            for track in active_tracks:
                if str(track["category_group"]) != category_group:
                    continue
                gap = sample_idx - int(track["sample_idx"])
                if gap < 1 or gap > 1:
                    continue
                distance = _distance_m(track, row_copy)
                if distance > max_link_distance_m:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_match = track
            if best_match is None:
                track_id = "det_track:{0}:{1}".format(scene_token or "scene", next_track_id)
                next_track_id += 1
            else:
                track_id = str(best_match["track_id"])
                best_match.update(
                    {
                        "sample_idx": sample_idx,
                        "x_ego": _safe_float(row_copy.get("x_ego")),
                        "y_ego": _safe_float(row_copy.get("y_ego")),
                    }
                )
            if best_match is None:
                active_tracks.append(
                    {
                        "track_id": track_id,
                        "sample_idx": sample_idx,
                        "x_ego": _safe_float(row_copy.get("x_ego")),
                        "y_ego": _safe_float(row_copy.get("y_ego")),
                        "category_group": category_group,
                    }
                )
            row_copy["track_id"] = track_id
            linked.append(row_copy)
    return linked


def adapt_nuscenes_predictions(
    benchmark_path: Path,
    db_path: Path,
    input_path: Path,
    output_path: Path,
    task_type: str = "tracking",
) -> Dict[str, object]:
    if task_type not in {"tracking", "detection"}:
        raise ValueError("Unsupported task_type: {0}".format(task_type))

    benchmark = _load_json(benchmark_path)
    cases = list(benchmark.get("cases") or [])
    sample_tokens = _unique(
        [str(frame["sample_token"]) for case in cases for frame in list(case.get("frames") or [])]
    )
    raw_payload = _load_json(input_path)
    raw_results = raw_payload.get("results") if isinstance(raw_payload.get("results"), dict) else raw_payload
    if not isinstance(raw_results, dict):
        raise ValueError("Expected a nuScenes prediction JSON with a top-level 'results' dictionary.")

    conn = sqlite3.connect(str(db_path.resolve()))
    conn.row_factory = sqlite3.Row
    try:
        sample_pose = _sample_pose_index(conn, sample_tokens)
    finally:
        conn.close()

    adapted_rows: List[Dict[str, object]] = []
    for sample_token in sample_tokens:
        pose = sample_pose.get(sample_token)
        if pose is None:
            continue
        boxes = list(raw_results.get(sample_token) or [])
        for box_idx, box in enumerate(boxes):
            translation = list(box.get("translation") or [])
            if len(translation) < 2:
                continue
            ego_point = global_xy_to_anchor_ego(
                np.asarray([[float(translation[0]), float(translation[1])]], dtype=float),
                [float(pose["ego_x"]), float(pose["ego_y"])],
                float(pose["ego_yaw"]),
            )[0]
            category_group = _normalize_category(
                str(
                    box.get("tracking_name")
                    or box.get("detection_name")
                    or box.get("category_name")
                    or box.get("category_group")
                    or ""
                )
            )
            adapted_rows.append(
                {
                    "sample_token": sample_token,
                    "sample_idx": int(pose["sample_idx"]),
                    "scene_token": str(pose["scene_token"]),
                    "track_id": str(box.get("tracking_id") or "det:{0}:{1}".format(sample_token, box_idx)),
                    "category_group": category_group,
                    "x_ego": round(float(ego_point[0]), 3),
                    "y_ego": round(float(ego_point[1]), 3),
                    "score": round(
                        _safe_float(
                            box.get("tracking_score")
                            if task_type == "tracking"
                            else box.get("detection_score")
                            if box.get("detection_score") is not None
                            else box.get("tracking_score")
                            if box.get("tracking_score") is not None
                            else box.get("score"),
                            0.0,
                        ),
                        4,
                    ),
                }
            )

    if task_type == "detection":
        adapted_rows = _link_detection_predictions(adapted_rows)

    predictions = [
        {
            key: value
            for key, value in row.items()
            if key in {"sample_token", "sample_idx", "track_id", "category_group", "x_ego", "y_ego", "score"}
        }
        for row in adapted_rows
    ]
    output = {
        "metadata": {
            "generator": "nuscenes_prediction_adapter_v1",
            "task_type": task_type,
            "source_benchmark": str(benchmark_path),
            "source_predictions": str(input_path),
        },
        "predictions": predictions,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(output_path),
        "prediction_count": len(predictions),
        "task_type": task_type,
    }


def filter_perception_benchmark_by_predictions(
    benchmark_path: Path,
    predictions_path: Path,
    output_path: Path,
    coverage_mode: str = "full_window",
) -> Dict[str, object]:
    if coverage_mode not in PREDICTION_COVERAGE_MODES:
        raise ValueError(
            "Unsupported coverage_mode: {0}. Expected one of {1}.".format(
                coverage_mode,
                ", ".join(PREDICTION_COVERAGE_MODES),
            )
        )

    benchmark = _load_json(benchmark_path)
    predictions_payload = _load_json(predictions_path)
    if isinstance(predictions_payload.get("results"), dict):
        prediction_sample_tokens = set(str(token) for token in predictions_payload["results"].keys())
    else:
        prediction_sample_tokens = set(
            str(item.get("sample_token") or "")
            for item in list(predictions_payload.get("predictions") or [])
            if item.get("sample_token")
        )

    filtered_cases = []
    coverage_rows = []
    for case in list(benchmark.get("cases") or []):
        frame_tokens = [str(frame["sample_token"]) for frame in list(case.get("frames") or [])]
        anchor_covered = str(case.get("anchor_sample_token") or "") in prediction_sample_tokens
        any_frame_covered = any(token in prediction_sample_tokens for token in frame_tokens)
        full_window_covered = all(token in prediction_sample_tokens for token in frame_tokens)
        should_keep = {
            "anchor": anchor_covered,
            "any_frame": any_frame_covered,
            "full_window": full_window_covered,
        }[coverage_mode]
        coverage_rows.append(
            {
                "benchmark_group": str(case.get("benchmark_group") or ""),
                "scene_name": str(case.get("scene_name") or ""),
                "anchor_covered": anchor_covered,
                "any_frame_covered": any_frame_covered,
                "full_window_covered": full_window_covered,
                "covered_frame_count": sum(token in prediction_sample_tokens for token in frame_tokens),
                "frame_count": len(frame_tokens),
            }
        )
        if should_keep:
            filtered_cases.append(case)

    output = {
        "metadata": {
            **dict(benchmark.get("metadata") or {}),
            "generator": "filtered_perception_slice_benchmark_v1",
            "source_benchmark": str(benchmark_path),
            "source_predictions": str(predictions_path),
            "coverage_mode": coverage_mode,
            "original_case_count": len(list(benchmark.get("cases") or [])),
            "case_count": len(filtered_cases),
        },
        "cases": filtered_cases,
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_path": str(output_path),
        "original_case_count": len(list(benchmark.get("cases") or [])),
        "filtered_case_count": len(filtered_cases),
        "coverage_mode": coverage_mode,
        "coverage_summary": {
            "anchor_covered_case_count": sum(int(row["anchor_covered"]) for row in coverage_rows),
            "any_frame_covered_case_count": sum(int(row["any_frame_covered"]) for row in coverage_rows),
            "full_window_covered_case_count": sum(int(row["full_window_covered"]) for row in coverage_rows),
        },
    }


def adapt_and_evaluate_nuscenes_predictions(
    benchmark_path: Path,
    db_path: Path,
    input_path: Path,
    output_dir: Path,
    task_type: str = "tracking",
    profile_name: str = "",
    match_distance_m: float = 2.0,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = output_dir / "adapted_predictions.json"
    adapter_metadata = adapt_nuscenes_predictions(
        benchmark_path=benchmark_path,
        db_path=db_path,
        input_path=input_path,
        output_path=adapted_path,
        task_type=task_type,
    )
    summary = evaluate_perception_predictions(
        benchmark_path=benchmark_path,
        predictions_path=adapted_path,
        output_dir=output_dir,
        profile_name=profile_name or Path(input_path).stem,
        match_distance_m=match_distance_m,
    )
    return {
        "adapter": adapter_metadata,
        "overview": dict(summary["overview"]),
        "output_dir": str(output_dir),
    }


def adapt_filter_and_evaluate_nuscenes_predictions(
    benchmark_path: Path,
    db_path: Path,
    input_path: Path,
    output_dir: Path,
    task_type: str = "tracking",
    profile_name: str = "",
    match_distance_m: float = 2.0,
    coverage_mode: str = "full_window",
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = output_dir / "adapted_predictions.json"
    adapter_metadata = adapt_nuscenes_predictions(
        benchmark_path=benchmark_path,
        db_path=db_path,
        input_path=input_path,
        output_path=adapted_path,
        task_type=task_type,
    )
    filtered_benchmark_path = output_dir / "filtered_benchmark.json"
    filter_metadata = filter_perception_benchmark_by_predictions(
        benchmark_path=benchmark_path,
        predictions_path=adapted_path,
        output_path=filtered_benchmark_path,
        coverage_mode=coverage_mode,
    )
    summary = evaluate_perception_predictions(
        benchmark_path=filtered_benchmark_path,
        predictions_path=adapted_path,
        output_dir=output_dir,
        profile_name=profile_name or Path(input_path).stem,
        match_distance_m=match_distance_m,
    )
    return {
        "adapter": adapter_metadata,
        "filter": filter_metadata,
        "overview": dict(summary["overview"]),
        "output_dir": str(output_dir),
    }


def _build_prediction_index(predictions: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    by_sample: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in predictions:
        normalized = dict(item)
        normalized["category_group"] = _normalize_category(str(item.get("category_group") or ""))
        normalized["track_id"] = str(item.get("track_id") or "track")
        normalized["x_ego"] = _safe_float(item.get("x_ego"))
        normalized["y_ego"] = _safe_float(item.get("y_ego"))
        by_sample[str(item.get("sample_token") or "")].append(normalized)
    return by_sample


def _distance_m(lhs: Dict[str, object], rhs: Dict[str, object]) -> float:
    return math.hypot(_safe_float(lhs["x_ego"]) - _safe_float(rhs["x_ego"]), _safe_float(lhs["y_ego"]) - _safe_float(rhs["y_ego"]))


def _evaluate_case_tracking(
    case: Dict[str, object],
    predictions_by_sample: Dict[str, List[Dict[str, object]]],
    match_distance_m: float,
) -> Dict[str, object]:
    frame_matches: Dict[str, Dict[int, float]] = defaultdict(dict)
    gt_frames = list(case["frames"])
    category_group = _normalize_category(str(case["category_group"]))
    anchor_sample_idx = int(case["anchor_sample_idx"])

    for frame_idx, frame in enumerate(gt_frames):
        sample_token = str(frame["sample_token"])
        for prediction in predictions_by_sample.get(sample_token, []):
            if _normalize_category(str(prediction.get("category_group") or "")) != category_group:
                continue
            error = _distance_m(prediction, frame)
            if error > match_distance_m:
                continue
            track_id = str(prediction["track_id"])
            current = frame_matches[track_id].get(frame_idx)
            if current is None or error < current:
                frame_matches[track_id][frame_idx] = error

    if not frame_matches:
        failure_tags = ["anchor_miss", "event_sparse", "track_fragmented"]
        return {
            "benchmark_group": str(case["benchmark_group"]),
            "reference_case_key": str(case["reference_case_key"]),
            "primary_behavior": str(case["primary_behavior"]),
            "category_group": str(case["category_group"]),
            "location": str(case["location"]),
            "event_frame_count": len(gt_frames),
            "matched_frame_count": 0,
            "anchor_detected": False,
            "event_recall": 0.0,
            "full_track_success": False,
            "contiguous_coverage": 0.0,
            "mean_center_error_m": None,
            "first_match_lag_frames": len(gt_frames),
            "matched_track_id": "",
            "failure_tags": failure_tags,
        }

    def track_priority(item: Tuple[str, Dict[int, float]]) -> Tuple[int, int, float]:
        track_id, matches = item
        matched_frames = sorted(matches)
        anchor_detected = any(int(gt_frames[idx]["sample_idx"]) == anchor_sample_idx for idx in matched_frames)
        mean_error = mean(matches.values()) if matches else float("inf")
        return (len(matched_frames), int(anchor_detected), -mean_error)

    best_track_id, best_matches = max(frame_matches.items(), key=track_priority)
    matched_indices = sorted(best_matches)
    matched_sample_indices = [int(gt_frames[idx]["sample_idx"]) for idx in matched_indices]
    anchor_detected = anchor_sample_idx in matched_sample_indices
    event_recall = _ratio(len(matched_indices), len(gt_frames))
    full_track_success = len(matched_indices) == len(gt_frames)

    longest = 0
    current = 0
    previous = None
    for idx in matched_indices:
        if previous is None or idx == previous + 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        previous = idx
    longest = max(longest, current)
    contiguous_coverage = _ratio(longest, len(gt_frames))

    first_match_lag = (
        int(gt_frames[matched_indices[0]]["sample_idx"]) - int(case["event_start_sample_idx"])
        if matched_indices
        else len(gt_frames)
    )
    mean_center_error_m = mean(best_matches.values()) if best_matches else None

    failure_tags: List[str] = []
    if not anchor_detected:
        failure_tags.append("anchor_miss")
    if event_recall < 0.75:
        failure_tags.append("event_sparse")
    if contiguous_coverage < 1.0:
        failure_tags.append("track_fragmented")
    if mean_center_error_m is not None and mean_center_error_m > 1.5:
        failure_tags.append("localization_error")

    return {
        "benchmark_group": str(case["benchmark_group"]),
        "reference_case_key": str(case["reference_case_key"]),
        "primary_behavior": str(case["primary_behavior"]),
        "category_group": str(case["category_group"]),
        "location": str(case["location"]),
        "event_frame_count": len(gt_frames),
        "matched_frame_count": len(matched_indices),
        "anchor_detected": anchor_detected,
        "event_recall": round(event_recall, 4),
        "full_track_success": full_track_success,
        "contiguous_coverage": round(contiguous_coverage, 4),
        "mean_center_error_m": round(mean_center_error_m, 4) if mean_center_error_m is not None else None,
        "first_match_lag_frames": int(first_match_lag),
        "matched_track_id": best_track_id,
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
            "anchor_recall_count": 0,
            "full_track_count": 0,
            "event_recalls": [],
            "center_errors": [],
            "failure_counts": defaultdict(int),
        }
    )

    for row in case_metrics:
        value = str(row.get(field_name) or "unknown")
        bucket = grouped[value]
        bucket["case_count"] += 1
        bucket["anchor_recall_count"] += int(bool(row["anchor_detected"]))
        bucket["full_track_count"] += int(bool(row["full_track_success"]))
        bucket["event_recalls"].append(float(row["event_recall"]))
        if row.get("mean_center_error_m") is not None:
            bucket["center_errors"].append(float(row["mean_center_error_m"]))
        for tag in list(row.get("failure_tags") or []):
            bucket["failure_counts"][str(tag)] += 1

    rows: List[Dict[str, object]] = []
    for value, bucket in grouped.items():
        failure_rows = [{"name": name, "count": int(count)} for name, count in dict(bucket["failure_counts"]).items()]
        failure_rows.sort(key=lambda item: (int(item["count"]), str(item["name"])), reverse=True)
        rows.append(
            {
                output_key: value,
                "case_count": int(bucket["case_count"]),
                "anchor_recall_count": int(bucket["anchor_recall_count"]),
                "anchor_recall_rate": round(_ratio(int(bucket["anchor_recall_count"]), int(bucket["case_count"])), 4),
                "full_track_count": int(bucket["full_track_count"]),
                "full_track_rate": round(_ratio(int(bucket["full_track_count"]), int(bucket["case_count"])), 4),
                "mean_event_recall": round(mean(bucket["event_recalls"]), 4) if bucket["event_recalls"] else 0.0,
                "mean_center_error_m": round(mean(bucket["center_errors"]), 4) if bucket["center_errors"] else 0.0,
                "top_failure_modes": failure_rows[:3],
                "top_failure_summary": ", ".join(
                    "{0}:{1}".format(item["name"], item["count"]) for item in failure_rows[:3]
                )
                or "none",
            }
        )
    rows.sort(key=lambda item: (int(item["case_count"]), str(item[output_key])), reverse=True)
    return rows


def _build_behavior_breakdown(case_metrics: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return _build_group_breakdown(case_metrics, field_name="primary_behavior", output_key="behavior")


def evaluate_perception_predictions(
    benchmark_path: Path,
    predictions_path: Path,
    output_dir: Path,
    profile_name: str = "",
    match_distance_m: float = 2.0,
) -> Dict[str, object]:
    benchmark = _load_json(benchmark_path)
    predictions = _load_json(predictions_path)
    cases = list(benchmark.get("cases") or [])
    prediction_rows = list(predictions.get("predictions") or [])
    predictions_by_sample = _build_prediction_index(prediction_rows)
    profile_name = profile_name or str((predictions.get("metadata") or {}).get("profile_name") or predictions_path.stem)

    case_metrics = []
    for case in cases:
        row = _evaluate_case_tracking(case, predictions_by_sample, match_distance_m=match_distance_m)
        support = dict(case.get("risk_facets") or {})
        row.update({name: str(support.get(name) or "unknown") for name in RISK_FACET_FIELDS})
        row["min_distance_m"] = case.get("min_distance_m")
        row["min_ttc_s"] = case.get("min_ttc_s")
        case_metrics.append(row)

    behavior_breakdown = _build_behavior_breakdown(case_metrics)
    risk_breakdowns = {
        field_name: _build_group_breakdown(case_metrics, field_name=field_name, output_key=field_name)
        for field_name in RISK_FACET_FIELDS
    }

    anchor_recall_count = sum(1 for row in case_metrics if row["anchor_detected"])
    full_track_count = sum(1 for row in case_metrics if row["full_track_success"])
    event_recalls = [float(row["event_recall"]) for row in case_metrics]
    contiguous_coverages = [float(row["contiguous_coverage"]) for row in case_metrics]
    center_errors = [float(row["mean_center_error_m"]) for row in case_metrics if row.get("mean_center_error_m") is not None]
    first_match_lags = [int(row["first_match_lag_frames"]) for row in case_metrics]
    perfect_case_count = sum(
        1
        for row in case_metrics
        if row["anchor_detected"] and row["full_track_success"] and not row["failure_tags"]
    )

    summary = {
        "profile_name": profile_name,
        "benchmark_path": str(benchmark_path),
        "predictions_path": str(predictions_path),
        "overview": {
            "case_count": len(case_metrics),
            "anchor_recall_count": anchor_recall_count,
            "anchor_recall_rate": round(_ratio(anchor_recall_count, len(case_metrics)), 4),
            "full_track_count": full_track_count,
            "full_track_rate": round(_ratio(full_track_count, len(case_metrics)), 4),
            "mean_event_recall": round(mean(event_recalls), 4) if event_recalls else 0.0,
            "mean_contiguous_coverage": round(mean(contiguous_coverages), 4) if contiguous_coverages else 0.0,
            "mean_center_error_m": round(mean(center_errors), 4) if center_errors else 0.0,
            "mean_first_match_lag_frames": round(mean(first_match_lags), 4) if first_match_lags else 0.0,
            "perfect_case_count": perfect_case_count,
        },
        "behavior_breakdown": behavior_breakdown,
        "risk_breakdowns": risk_breakdowns,
        "case_metrics": case_metrics,
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perception_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Scenario-Conditioned Perception Evaluation",
        "",
        "- Profile: {0}".format(profile_name),
        "- Cases: {0}".format(summary["overview"]["case_count"]),
        "- Anchor recall: {0}/{1} ({2:.1%})".format(
            summary["overview"]["anchor_recall_count"],
            summary["overview"]["case_count"],
            summary["overview"]["anchor_recall_rate"],
        ),
        "- Full-track success: {0}/{1} ({2:.1%})".format(
            summary["overview"]["full_track_count"],
            summary["overview"]["case_count"],
            summary["overview"]["full_track_rate"],
        ),
        "- Mean event recall: {0:.3f}".format(summary["overview"]["mean_event_recall"]),
        "- Mean contiguous coverage: {0:.3f}".format(summary["overview"]["mean_contiguous_coverage"]),
        "- Mean center error: {0:.3f}".format(summary["overview"]["mean_center_error_m"]),
        "",
        "## Behavior Breakdown",
        "",
        "| Behavior | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Center Error | Top Failure Modes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in behavior_breakdown:
        lines.append(
            "| {0} | {1} | {2}/{1} ({3:.1%}) | {4}/{1} ({5:.1%}) | {6:.3f} | {7:.3f} | {8} |".format(
                row["behavior"],
                row["case_count"],
                row["anchor_recall_count"],
                row["anchor_recall_rate"],
                row["full_track_count"],
                row["full_track_rate"],
                row["mean_event_recall"],
                row["mean_center_error_m"],
                row["top_failure_summary"],
            )
        )
    lines.extend(
        [
            "",
            "## Risk Breakdown",
            "",
        ]
    )
    for field_name in RISK_FACET_FIELDS:
        lines.extend(
            [
                "### {0}".format(field_name.replace("_", " ").title()),
                "",
                "| Group | Cases | Anchor Recall | Full Track | Mean Event Recall | Mean Center Error | Top Failure Modes |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in risk_breakdowns[field_name]:
            lines.append(
                "| {0} | {1} | {2}/{1} ({3:.1%}) | {4}/{1} ({5:.1%}) | {6:.3f} | {7:.3f} | {8} |".format(
                    row[field_name],
                    row["case_count"],
                    row["anchor_recall_count"],
                    row["anchor_recall_rate"],
                    row["full_track_count"],
                    row["full_track_rate"],
                    row["mean_event_recall"],
                    row["mean_center_error_m"],
                    row["top_failure_summary"],
                )
            )
        lines.extend([""])
    lines.extend(
        [
            "",
            "## Case Metrics",
            "",
            "| Group | Behavior | Actor | Anchor | Event Recall | Contiguous Coverage | Center Error | Distance Band | TTC Band | Map Relation | Occlusion Proxy | Failure Tags |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in case_metrics:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4:.3f} | {5:.3f} | {6} | {7} | {8} | {9} | {10} | {11} |".format(
                row["benchmark_group"],
                row["primary_behavior"],
                row["category_group"],
                row["anchor_detected"],
                row["event_recall"],
                row["contiguous_coverage"],
                "{0:.3f}".format(float(row["mean_center_error_m"])) if row.get("mean_center_error_m") is not None else "-",
                row["distance_band"],
                row["ttc_band"],
                row["map_relation"],
                row["occlusion_proxy"],
                ", ".join(row["failure_tags"]) or "none",
            )
        )
    (output_dir / "perception_metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "perception_metrics_summary.html").write_text(
        PERCEPTION_SUMMARY_TEMPLATE.render(summary=summary),
        encoding="utf-8",
    )
    with (output_dir / "perception_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "benchmark_group",
            "reference_case_key",
            "primary_behavior",
            "category_group",
            "location",
            "anchor_detected",
            "event_recall",
            "full_track_success",
            "contiguous_coverage",
            "mean_center_error_m",
            "first_match_lag_frames",
            "distance_band",
            "ttc_band",
            "visibility_band",
            "map_relation",
            "occlusion_proxy",
            "min_distance_m",
            "min_ttc_s",
            "failure_tags",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_metrics:
            payload = dict(row)
            payload["failure_tags"] = "|".join(str(item) for item in list(row.get("failure_tags") or []))
            writer.writerow({field: payload.get(field, "") for field in fieldnames})
    return summary


def build_perception_comparison(run_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    profiles: List[Dict[str, object]] = []
    behavior_index: Dict[str, Dict[str, object]] = defaultdict(dict)
    risk_index: Dict[str, Dict[str, Dict[str, object]]] = {field_name: defaultdict(dict) for field_name in RISK_FACET_FIELDS}
    for item in run_summaries:
        overview = dict(item.get("overview") or {})
        profile_name = str(item.get("profile_name") or "profile")
        profiles.append(
            {
                "name": profile_name,
                "label": _profile_label(profile_name),
                "case_count": int(overview.get("case_count") or 0),
                "anchor_recall_count": int(overview.get("anchor_recall_count") or 0),
                "anchor_recall_rate": float(overview.get("anchor_recall_rate") or 0.0),
                "full_track_count": int(overview.get("full_track_count") or 0),
                "full_track_rate": float(overview.get("full_track_rate") or 0.0),
                "mean_event_recall": float(overview.get("mean_event_recall") or 0.0),
                "mean_contiguous_coverage": float(overview.get("mean_contiguous_coverage") or 0.0),
                "mean_center_error_m": float(overview.get("mean_center_error_m") or 0.0),
            }
        )
        for row in list(item.get("behavior_breakdown") or []):
            behavior_index[str(row["behavior"])][profile_name] = dict(row)
        for field_name, rows in dict(item.get("risk_breakdowns") or {}).items():
            if field_name not in risk_index:
                continue
            for row in list(rows or []):
                risk_value = str(row.get(field_name) or "unknown")
                risk_index[field_name][risk_value][profile_name] = dict(row)

    profiles.sort(
        key=lambda row: (
            float(row["anchor_recall_rate"]),
            float(row["full_track_rate"]),
            float(row["mean_event_recall"]),
            -float(row["mean_center_error_m"]),
        ),
        reverse=True,
    )
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
                                "anchor_recall_count": 0,
                                "mean_event_recall": 0.0,
                            },
                        )
                    )
                    for name in profile_order
                ],
            }
        )

    risk_matrices: Dict[str, List[Dict[str, object]]] = {}
    for field_name in RISK_FACET_FIELDS:
        field_rows: List[Dict[str, object]] = []
        for risk_value in sorted(risk_index[field_name]):
            field_rows.append(
                {
                    "group": risk_value,
                    "cells": [
                        dict(
                            risk_index[field_name][risk_value].get(
                                name,
                                {
                                    "case_count": 0,
                                    "anchor_recall_count": 0,
                                    "mean_event_recall": 0.0,
                                },
                            )
                        )
                        for name in profile_order
                    ],
                }
            )
        risk_matrices[field_name] = field_rows

    return {
        "overview": {
            "profile_count": len(profiles),
            "case_count": int(profiles[0]["case_count"]) if profiles else 0,
        },
        "profiles": profiles,
        "behavior_matrix": behavior_matrix,
        "risk_matrices": risk_matrices,
    }


def load_perception_evaluation_summary(eval_dir: Path) -> Dict[str, object]:
    eval_dir = eval_dir.resolve()
    payload = _load_json(eval_dir / "perception_metrics.json")
    payload["output_dir"] = str(eval_dir)
    return payload


def compare_perception_evaluations(
    evaluation_dirs: Sequence[Path],
    output_dir: Path,
) -> Dict[str, object]:
    summaries = [load_perception_evaluation_summary(Path(path)) for path in evaluation_dirs]
    comparison = build_perception_comparison(summaries)
    write_perception_comparison(comparison, output_dir)
    return {
        "output_dir": str(output_dir.resolve()),
        "profile_count": int(comparison["overview"]["profile_count"]),
        "case_count": int(comparison["overview"]["case_count"]),
    }


def write_perception_comparison(comparison: Dict[str, object], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perception_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Scenario-Conditioned Perception Comparison",
        "",
        "- Profiles: {0}".format(comparison["overview"]["profile_count"]),
        "- Cases: {0}".format(comparison["overview"]["case_count"]),
        "",
        "## Profile Overview",
        "",
        "| Profile | Anchor Recall | Full Track | Mean Event Recall | Mean Contiguous Coverage | Mean Center Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(comparison.get("profiles") or []):
        lines.append(
            "| {0} | {1}/{2} ({3:.1%}) | {4}/{2} ({5:.1%}) | {6:.3f} | {7:.3f} | {8:.3f} |".format(
                row["label"],
                row["anchor_recall_count"],
                row["case_count"],
                row["anchor_recall_rate"],
                row["full_track_count"],
                row["full_track_rate"],
                row["mean_event_recall"],
                row["mean_contiguous_coverage"],
                row["mean_center_error_m"],
            )
        )
    for field_name in RISK_FACET_FIELDS:
        lines.extend(
            [
                "",
                "## {0}".format(field_name.replace("_", " ").title()),
                "",
                "| Group | " + " | ".join(str(row["label"]) for row in list(comparison.get("profiles") or [])) + " |",
                "| --- | " + " | ".join("---" for _ in list(comparison.get("profiles") or [])) + " |",
            ]
        )
        for row in list((comparison.get("risk_matrices") or {}).get(field_name) or []):
            lines.append(
                "| {0} | {1} |".format(
                    row["group"],
                    " | ".join(
                        "{0}/{1}, {2:.3f}".format(
                            cell.get("anchor_recall_count", 0),
                            cell.get("case_count", 0),
                            float(cell.get("mean_event_recall") or 0.0),
                        )
                        for cell in list(row.get("cells") or [])
                    ),
                )
            )
    (output_dir / "perception_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "perception_comparison_summary.html").write_text(
        PERCEPTION_COMPARISON_TEMPLATE.render(comparison=comparison),
        encoding="utf-8",
    )

    with (output_dir / "perception_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "label",
                "case_count",
                "anchor_recall_count",
                "anchor_recall_rate",
                "full_track_count",
                "full_track_rate",
                "mean_event_recall",
                "mean_contiguous_coverage",
                "mean_center_error_m",
            ],
        )
        writer.writeheader()
        for row in list(comparison.get("profiles") or []):
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def run_proxy_perception_study(
    benchmark_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries = []

    for profile_name in PROXY_PERCEPTION_PROFILES:
        prediction_path = output_dir / "predictions" / "{0}.json".format(profile_name)
        generate_proxy_perception_predictions(benchmark_path, prediction_path, profile_name=profile_name)
        profile_output = output_dir / profile_name
        summary = evaluate_perception_predictions(
            benchmark_path=benchmark_path,
            predictions_path=prediction_path,
            output_dir=profile_output,
            profile_name=profile_name,
        )
        run_summaries.append(summary)

    comparison = build_perception_comparison(run_summaries)
    write_perception_comparison(comparison, output_dir)
    return {
        "output_dir": str(output_dir),
        "profile_count": len(run_summaries),
        "case_count": int(comparison["overview"]["case_count"]),
    }
