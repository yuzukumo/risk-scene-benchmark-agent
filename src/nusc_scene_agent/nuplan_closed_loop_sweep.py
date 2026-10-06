from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.nuplan_closed_loop import (
    DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES,
    run_nuplan_closed_loop_study,
)
from nusc_scene_agent.nuplan_replay import DEFAULT_NUPLAN_SPLIT


NUPLAN_CLOSED_LOOP_SWEEP_SCHEMA = "nuplan_closed_loop_sweep_v1"
DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT = Path("outputs/nuplan_closed_loop_sweep_v1")

PROFILE_MEAN_METRICS = [
    "mean_ego_ade_m",
    "mean_ego_fde_m",
    "mean_min_distance_error_m",
    "mean_min_ttc_error_s",
    "mean_progress_ratio",
    "mean_raw_progress_ratio",
    "mean_closed_loop_score",
]

PROFILE_COUNT_METRICS = [
    "collision_proxy_mismatch_count",
    "comfort_violation_count",
    "closed_loop_drift_count",
]

CASE_MEAN_METRICS = [
    "ego_ade_m",
    "ego_fde_m",
    "min_distance_error_m",
    "min_ttc_error_s",
    "progress_ratio",
    "raw_progress_ratio",
    "closed_loop_score",
]


def run_nuplan_closed_loop_sweep(
    studies: Sequence[Mapping[str, Any]],
    output_dir: Path = DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT,
    *,
    defaults: Optional[Mapping[str, Any]] = None,
    profiles: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not studies:
        raise ValueError("At least one nuPlan closed-loop sweep study is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_config = dict(defaults or {})
    if profiles:
        default_config["profiles"] = list(profiles)

    study_results: List[Dict[str, Any]] = []
    for raw_study in studies:
        study = _normalize_study_config(raw_study, output_dir=output_dir, defaults=default_config)
        manifest = run_nuplan_closed_loop_study(
            split_dir=Path(study["split_dir"]),
            output_dir=Path(study["output"]),
            max_dbs=int(study["max_dbs"]),
            max_cases=int(study["max_cases"]),
            max_cases_per_db=int(study["max_cases_per_db"]),
            history_s=float(study["history_s"]),
            future_s=float(study["future_s"]),
            frame_hz=float(study["frame_hz"]),
            min_anchor_gap_s=float(study["min_anchor_gap_s"]),
            scenario_tags=study.get("scenario_tags") or None,
            profiles=study.get("profiles") or None,
        )
        study_results.append(_load_completed_study(study, manifest))

    payload = build_nuplan_closed_loop_sweep_summary(study_results, output_dir=output_dir)
    payload["artifact_manifest"] = {"path": str(output_dir / "artifact_manifest.json")}
    _write_sweep_outputs(payload, output_dir)
    _write_sweep_artifact_manifest(output_dir, study_results)
    return payload


def build_nuplan_closed_loop_sweep_summary(
    study_results: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    leaderboard = _build_profile_leaderboard(study_results)
    family_matrix = _build_family_matrix(study_results)
    failure_taxonomy, top_failure_tags = _build_failure_taxonomy(study_results)
    study_rows = [_study_summary_row(study) for study in study_results]
    profile_names = sorted({str(row["profile_name"]) for row in leaderboard if row["study_name"] != "__overall__"})
    return {
        "schema": NUPLAN_CLOSED_LOOP_SWEEP_SCHEMA,
        "output_dir": str(output_dir),
        "overview": {
            "study_count": len(study_results),
            "profile_count": len(profile_names),
            "profiles": profile_names,
            "db_count_scanned": sum(int(row.get("db_count_scanned") or 0) for row in study_rows),
            "candidate_case_count": sum(int(row.get("candidate_case_count") or 0) for row in study_rows),
            "case_count": sum(int(row.get("case_count") or 0) for row in study_rows),
            "failure_tag_count": sum(int(row.get("count") or 0) for row in failure_taxonomy),
        },
        "studies": study_rows,
        "profile_leaderboard": leaderboard,
        "scenario_family_matrix": family_matrix,
        "failure_taxonomy": failure_taxonomy,
        "top_failure_tags": top_failure_tags,
    }


def _normalize_study_config(
    raw_study: Mapping[str, Any],
    *,
    output_dir: Path,
    defaults: Mapping[str, Any],
) -> Dict[str, Any]:
    config = {**dict(defaults), **dict(raw_study)}
    name = _safe_name(str(config.get("name") or config.get("id") or f"study_{len(str(config))}"))
    split_dir = Path(str(config.get("split_dir") or DEFAULT_NUPLAN_SPLIT))
    return {
        "name": name,
        "split_dir": str(split_dir),
        "output": str(Path(str(config.get("output") or (output_dir / name)))),
        "max_dbs": int(config.get("max_dbs") or 4),
        "max_cases": int(config.get("max_cases") or 16),
        "max_cases_per_db": int(config.get("max_cases_per_db") or 4),
        "history_s": float(config.get("history_s") if config.get("history_s") is not None else 2.0),
        "future_s": float(config.get("future_s") if config.get("future_s") is not None else 4.0),
        "frame_hz": float(config.get("frame_hz") if config.get("frame_hz") is not None else 2.0),
        "min_anchor_gap_s": float(
            config.get("min_anchor_gap_s") if config.get("min_anchor_gap_s") is not None else 4.0
        ),
        "scenario_tags": list(config.get("scenario_tags") or []),
        "profiles": list(config.get("profiles") or DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES),
    }


def _load_completed_study(study: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    study_output = Path(str(study["output"]))
    benchmark_path = study_output / "nuplan_closed_loop_benchmark.json"
    comparison_path = study_output / "comparison/closed_loop_comparison.json"
    benchmark = _read_json(benchmark_path)
    comparison = _read_json(comparison_path)
    evaluations = []
    for item in manifest.get("evaluations", []):
        metrics_path = Path(str(item["evaluation_dir"])) / "closed_loop_metrics.json"
        evaluations.append(
            {
                "profile_name": str(item.get("profile_name") or metrics_path.parent.name),
                "evaluation_dir": str(metrics_path.parent),
                "metrics": _read_json(metrics_path),
            }
        )
    return {
        "config": dict(study),
        "manifest": dict(manifest),
        "benchmark": benchmark,
        "comparison": comparison,
        "evaluations": evaluations,
    }


def _study_summary_row(study: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(study.get("config") or {})
    metadata = dict(study.get("benchmark", {}).get("metadata") or {})
    distribution = dict(metadata.get("case_distribution") or {})
    return {
        "study_name": str(config.get("name") or ""),
        "split_dir": str(config.get("split_dir") or metadata.get("split_dir") or ""),
        "output_dir": str(config.get("output") or ""),
        "db_count_scanned": int(metadata.get("db_count_scanned") or 0),
        "candidate_case_count": int(metadata.get("candidate_case_count") or 0),
        "case_count": int(metadata.get("case_count") or 0),
        "scenario_families": dict(distribution.get("by_family") or {}),
        "difficulty": dict(distribution.get("by_difficulty") or {}),
        "skipped": dict(metadata.get("skipped") or {}),
    }


def _build_profile_leaderboard(study_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for study in study_results:
        study_name = str(dict(study.get("config") or {}).get("name") or "")
        for profile in study.get("comparison", {}).get("profiles", []):
            row = {"study_name": study_name}
            row.update({key: profile.get(key) for key in _profile_leaderboard_fieldnames() if key != "study_name"})
            rows.append(row)

    for profile_name in sorted({str(row.get("profile_name") or "") for row in rows if row.get("profile_name")}):
        profile_rows = [row for row in rows if str(row.get("profile_name") or "") == profile_name]
        rows.append({"study_name": "__overall__", "profile_name": profile_name, **_aggregate_profile_rows(profile_rows)})
    return rows


def _aggregate_profile_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_count = sum(int(row.get("case_count") or 0) for row in rows)
    full_horizon_count = sum(int(row.get("full_horizon_count") or 0) for row in rows)
    payload: Dict[str, Any] = {
        "case_count": case_count,
        "full_horizon_count": full_horizon_count,
        "full_horizon_rate": full_horizon_count / case_count if case_count else 0.0,
    }
    for key in PROFILE_MEAN_METRICS:
        payload[key] = _weighted_mean(rows, key, "case_count")
    for key in PROFILE_COUNT_METRICS:
        payload[key] = sum(int(row.get(key) or 0) for row in rows)
    return payload


def _build_family_matrix(study_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for study in study_results:
        study_name = str(dict(study.get("config") or {}).get("name") or "")
        for evaluation in study.get("evaluations", []):
            profile_name = str(evaluation.get("profile_name") or "")
            for row in evaluation.get("metrics", {}).get("case_metrics", []):
                family = str(row.get("scenario_family") or "unknown")
                grouped[(study_name, profile_name, family)].append(row)
                grouped[("__overall__", profile_name, family)].append(row)

    matrix = []
    for (study_name, profile_name, family), rows in sorted(grouped.items()):
        matrix.append(
            {
                "study_name": study_name,
                "profile_name": profile_name,
                "scenario_family": family,
                **_aggregate_case_rows(rows),
            }
        )
    return matrix


def _build_failure_taxonomy(
    study_results: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Counter[Tuple[str, str, str, str, str, str]] = Counter()
    top_counter: Counter[str] = Counter()
    for study in study_results:
        study_name = str(dict(study.get("config") or {}).get("name") or "")
        for evaluation in study.get("evaluations", []):
            profile_name = str(evaluation.get("profile_name") or "")
            for row in evaluation.get("metrics", {}).get("case_metrics", []):
                tags = [str(tag) for tag in row.get("failure_tags", []) if str(tag)]
                for tag in tags:
                    key = (
                        study_name,
                        profile_name,
                        str(row.get("scenario_family") or "unknown"),
                        str(row.get("scenario_tag") or "unknown"),
                        str(row.get("difficulty_label") or "unknown"),
                        tag,
                    )
                    grouped[key] += 1
                    top_counter[tag] += 1

    rows = [
        {
            "study_name": key[0],
            "profile_name": key[1],
            "scenario_family": key[2],
            "scenario_tag": key[3],
            "difficulty_label": key[4],
            "failure_tag": key[5],
            "count": count,
        }
        for key, count in sorted(grouped.items())
    ]
    top_rows = [{"failure_tag": tag, "count": count} for tag, count in top_counter.most_common()]
    return rows, top_rows


def _aggregate_case_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_count = len(rows)
    full_horizon_count = sum(1 for row in rows if bool(row.get("full_horizon")))
    payload: Dict[str, Any] = {
        "case_count": case_count,
        "full_horizon_count": full_horizon_count,
        "full_horizon_rate": full_horizon_count / case_count if case_count else 0.0,
        "collision_proxy_mismatch_count": sum(
            1 for row in rows if "collision_proxy_mismatch" in set(row.get("failure_tags", []))
        ),
        "comfort_violation_count": sum(1 for row in rows if "comfort_violation" in set(row.get("failure_tags", []))),
        "closed_loop_drift_count": sum(1 for row in rows if "closed_loop_drift" in set(row.get("failure_tags", []))),
        "risk_distance_error_count": sum(1 for row in rows if "risk_distance_error" in set(row.get("failure_tags", []))),
        "ttc_error_count": sum(1 for row in rows if "ttc_error" in set(row.get("failure_tags", []))),
    }
    for key in CASE_MEAN_METRICS:
        payload[f"mean_{key}"] = _mean(rows, key)
    return payload


def _write_sweep_outputs(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "nuplan_closed_loop_sweep_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_csv(payload["profile_leaderboard"], output_dir / "nuplan_closed_loop_sweep_leaderboard.csv")
    _write_csv(payload["scenario_family_matrix"], output_dir / "nuplan_closed_loop_sweep_family_matrix.csv")
    _write_csv(payload["failure_taxonomy"], output_dir / "nuplan_closed_loop_sweep_failure_taxonomy.csv")
    markdown = _render_sweep_markdown(payload)
    (output_dir / "nuplan_closed_loop_sweep_summary.md").write_text(markdown, encoding="utf-8")
    (output_dir / "nuplan_closed_loop_sweep_summary.html").write_text(
        _markdown_to_basic_html(markdown),
        encoding="utf-8",
    )


def _write_sweep_artifact_manifest(output_dir: Path, study_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    artifact_specs = [
        (output_dir / "nuplan_closed_loop_sweep_summary.json", "summary", "sweep_summary"),
        (output_dir / "nuplan_closed_loop_sweep_summary.md", "summary", "sweep_summary"),
        (output_dir / "nuplan_closed_loop_sweep_summary.html", "summary", "sweep_summary"),
        (output_dir / "nuplan_closed_loop_sweep_leaderboard.csv", "comparison", "cross_study_leaderboard"),
        (output_dir / "nuplan_closed_loop_sweep_family_matrix.csv", "analysis", "scenario_family_matrix"),
        (output_dir / "nuplan_closed_loop_sweep_failure_taxonomy.csv", "analysis", "failure_taxonomy"),
    ]
    for study in study_results:
        study_output = Path(str(dict(study.get("config") or {}).get("output") or ""))
        artifact_specs.extend(
            [
                (study_output / "nuplan_closed_loop_benchmark.json", "benchmark", "study_benchmark"),
                (study_output / "nuplan_closed_loop_study_manifest.json", "summary", "study_manifest"),
                (study_output / "nuplan_closed_loop_study_summary.md", "summary", "study_summary"),
                (study_output / "comparison/closed_loop_leaderboard.csv", "comparison", "study_leaderboard"),
                (study_output / "case_studies/nuplan_closed_loop_case_studies.png", "evidence", "case_study_figure"),
            ]
        )

    artifacts = [
        build_artifact_entry(path=path, role=role, kind=kind, output_root=output_dir)
        for path, role, kind in artifact_specs
    ]
    return write_artifact_manifest(
        output_dir=output_dir,
        artifacts=artifacts,
        metadata={"benchmark_layer": "nuplan_closed_loop_replay", "sweep_schema": NUPLAN_CLOSED_LOOP_SWEEP_SCHEMA},
    )


def _render_sweep_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# nuPlan Closed-Loop Replay Sweep",
        "",
        f"- Studies: `{overview.get('study_count', 0)}`",
        f"- Cases: `{overview.get('case_count', 0)}`",
        f"- Candidate anchors: `{overview.get('candidate_case_count', 0)}`",
        f"- Scanned DBs: `{overview.get('db_count_scanned', 0)}`",
        "",
        "## Studies",
        "",
        "| Study | DBs | Candidates | Cases | Split |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("studies", []):
        lines.append(
            f"| `{row.get('study_name', '')}` | `{row.get('db_count_scanned', 0)}` | "
            f"`{row.get('candidate_case_count', 0)}` | `{row.get('case_count', 0)}` | "
            f"`{row.get('split_dir', '')}` |"
        )

    lines.extend(
        [
            "",
            "## Overall Profile Leaderboard",
            "",
            "| Profile | Cases | Full Horizon | Ego ADE | Ego FDE | Distance Error | TTC Error | Progress Ratio | Raw Progress Ratio | Closed-Loop Score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    overall_rows = [row for row in payload.get("profile_leaderboard", []) if row.get("study_name") == "__overall__"]
    for row in overall_rows:
        lines.append(
            f"| `{row.get('profile_name', '')}` | `{row.get('case_count', 0)}` | "
            f"`{row.get('full_horizon_count', 0)}/{row.get('case_count', 0)}` | "
            f"`{_format_optional_float(row.get('mean_ego_ade_m'))}` | "
            f"`{_format_optional_float(row.get('mean_ego_fde_m'))}` | "
            f"`{_format_optional_float(row.get('mean_min_distance_error_m'))}` | "
            f"`{_format_optional_float(row.get('mean_min_ttc_error_s'))}` | "
            f"`{_format_optional_float(row.get('mean_progress_ratio'))}` | "
            f"`{_format_optional_float(row.get('mean_raw_progress_ratio'))}` | "
            f"`{_format_optional_float(row.get('mean_closed_loop_score'))}` |"
        )

    lines.extend(
        [
            "",
            "## Frequent Failure Tags",
            "",
            "| Failure Tag | Count |",
            "| --- | --- |",
        ]
    )
    top_failure_tags = list(payload.get("top_failure_tags", []))
    if not top_failure_tags:
        lines.append("| `none` | `0` |")
    for row in top_failure_tags[:12]:
        lines.append(f"| `{row.get('failure_tag', '')}` | `{row.get('count', 0)}` |")
    lines.append("")
    return "\n".join(lines)


def _profile_leaderboard_fieldnames() -> List[str]:
    return [
        "study_name",
        "profile_name",
        "case_count",
        "full_horizon_count",
        "full_horizon_rate",
        *PROFILE_MEAN_METRICS,
        *PROFILE_COUNT_METRICS,
    ]


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return cleaned.strip("_") or "study"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if not rows:
        return ["empty"]
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _weighted_mean(rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str) -> Optional[float]:
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        weight = int(row.get(weight_key) or 0)
        if weight <= 0:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else None


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _markdown_to_basic_html(markdown: str) -> str:
    escaped_lines = []
    for line in markdown.splitlines():
        safe_line = html.escape(line)
        if line.startswith("# "):
            escaped_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            escaped_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            escaped_lines.append(f"<p>{safe_line}</p>")
        else:
            escaped_lines.append(f"<p>{safe_line}</p>")
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head><meta charset=\"utf-8\"><title>nuPlan Closed-Loop Replay Sweep</title></head>",
            "<body>",
            *escaped_lines,
            "</body></html>",
        ]
    )
