from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.benchmark_catalog import build_default_benchmark_catalog


RESULT_REGISTRY_SCHEMA = "benchmark_result_registry_v1"
DEFAULT_RESULT_REGISTRY_OUTPUT = Path("outputs/full_benchmark_suite_v1/result_registry")


DEFAULT_RESULT_SOURCES = [
    Path("outputs/risk_benchmark_suite_v1/experiment_result.json"),
    Path("outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_summary.json"),
    Path("outputs/nuplan_closed_loop_sweep_v1/nuplan_closed_loop_sweep_summary.json"),
    Path("outputs/bench2drive_controlled_study_v2/study_summary.json"),
    Path("outputs/carla_fixed_evaluation_v3/fixed_evaluation.json"),
    Path("outputs/contextvae_world_model_study_v3/contextvae_study_manifest.json"),
    Path("outputs/contextvae_expanded_study_v1/contextvae_study_manifest.json"),
    Path("outputs/forecast_validation_study_v1/contextvae/contextvae_study_manifest.json"),
    Path("outputs/forecast_validation_study_v1/scene_disjoint_audit/generalization_audit.json"),
    Path("outputs/learned_retriever_trainval_v3/training_report.json"),
    Path("outputs/risk_case_expansion_v2/expansion_report.json"),
    Path("outputs/failure_aware_reranking_eval_v4/failure_aware_reranking_eval.json"),
    Path("outputs/carla_semantic_demo_final/carla_semantic_demo_mining_report.json"),
    Path("outputs/carla_semantic_demo_final/carla_semantic_demo_report.json"),
    Path("outputs/carla_semantic_demo_final/carla_semantic_demo_audit.json"),
    Path("outputs/model_in_the_loop_failure_mining_v2/failure_mining_report.json"),
]


def write_result_registry(
    output_dir: Path = DEFAULT_RESULT_REGISTRY_OUTPUT,
    *,
    sources: Optional[Sequence[Path]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(path) for path in (sources or DEFAULT_RESULT_SOURCES)]
    source_status = [
        {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
        for path in source_paths
    ]
    entries = [entry for path in source_paths if (entry := _build_result_entry(path)) is not None]
    missing_sources = [item["path"] for item in source_status if not item["exists"]]
    layer_catalog = build_default_benchmark_catalog()
    payload = {
        "schema": RESULT_REGISTRY_SCHEMA,
        "metadata": dict(metadata or {}),
        "overview": {
            "result_count": len(entries),
            "layer_count": len({entry["layer_id"] for entry in entries}),
            "source_count": len(source_paths),
            "existing_source_count": sum(1 for path in source_paths if path.exists()),
            "missing_source_count": len(missing_sources),
            "complete": not missing_sources,
        },
        "source_status": source_status,
        "missing_sources": missing_sources,
        "catalog_schema": layer_catalog.get("schema"),
        "layers": layer_catalog.get("layers", {}),
        "results": entries,
    }
    (output_dir / "result_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_registry_csv(entries, output_dir / "result_registry.csv")
    (output_dir / "result_registry.md").write_text(_render_registry_markdown(payload), encoding="utf-8")
    artifact_manifest = write_artifact_manifest(
        output_dir=output_dir,
        artifacts=[
            build_artifact_entry(output_dir / "result_registry.csv", "summary", "result_registry_table", output_dir),
            build_artifact_entry(output_dir / "result_registry.md", "summary", "result_registry_report", output_dir),
            *[
                build_artifact_entry(path, "source", _infer_source_kind(path), output_dir)
                for path in source_paths
            ],
        ],
        metadata={
            "schema": RESULT_REGISTRY_SCHEMA,
            "registry_json": str(output_dir / "result_registry.json"),
            "registry_json_excluded_from_manifest": True,
        },
    )
    payload["artifact_manifest"] = {
        "path": str(output_dir / "artifact_manifest.json"),
        "overview": artifact_manifest.get("overview", {}),
    }
    (output_dir / "result_registry.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_result_entry(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = _read_json(path)
    schema = str(payload.get("schema") or "")
    if schema == "experiment_result_v1" and payload.get("experiment_type") == "risk_benchmark_suite":
        return _risk_suite_entry(path, payload)
    if schema == "experiment_result_v1" and isinstance(payload.get("result"), dict):
        payload = payload["result"]
        schema = str(payload.get("schema") or "")
    if schema == "nuplan_replay_sweep_v1":
        return _nuplan_replay_sweep_entry(path, payload)
    if schema == "nuplan_closed_loop_sweep_v1":
        return _nuplan_closed_loop_sweep_entry(path, payload)
    if schema == "bench2drive_vision_e2e_training_v1":
        return _bench2drive_training_entry(path, payload)
    if schema == "bench2drive_vision_e2e_eval_v1":
        return _bench2drive_eval_entry(path, payload)
    if schema == "bench2drive_vision_planner_diagnostics_v1":
        return _bench2drive_planner_diagnostics_entry(path, payload)
    if schema == "bench2drive_vision_closed_loop_v1":
        return _bench2drive_closed_loop_entry(path, payload)
    if schema == "bench2drive_controlled_study_v1":
        return {
            "layer_id": "bench2drive_controlled_study", "source_path": str(path), "schema": schema,
            "summary": {"selection": payload.get("selection"), "test_policy": payload.get("test_policy"),
                        "seeds": dict(payload.get("protocol") or {}).get("seeds", [])},
            "metrics": {"run_count": len(payload.get("runs", [])), "comparison_count": len(payload.get("comparisons", []))},
            "aggregate": payload.get("aggregate", []),
        }
    if schema == "carla_fixed_evaluation_v1":
        return {
            "layer_id": "carla_fixed_evaluation", "source_path": str(path), "schema": schema,
            "summary": {"evaluation_boundary": payload.get("evaluation_boundary")},
            "metrics": {key: payload.get(key) for key in ["attempt_count", "completed_count", "error_count",
                                                         "passed_count", "success_rate", "mean_route_completion", "collision_count"]},
        }
    if schema == "contextvae_world_model_study_v1":
        return {
            "layer_id": "external_forecast", "source_path": str(path), "schema": schema,
            "summary": payload.get("protocol", {}),
            "metrics": {**dict(payload.get("preparation") or {}),
                        **dict(payload.get("evaluation", {}).get("overview") or {})},
        }
    if schema == "forecast_generalization_audit_v1":
        return {
            "layer_id": "forecast_generalization", "source_path": str(path), "schema": schema,
            "summary": {"selection_policy": payload.get("selection_policy")},
            "metrics": {key: payload.get(key) for key in ["source_case_count", "case_count", "scene_count",
                                                         "excluded_case_count", "shared_development_scene_count"]},
        }
    if schema == "learned_scene_retriever_training_report_v1":
        return {
            "layer_id": "learned_retrieval", "source_path": str(path), "schema": schema,
            "summary": {"label_semantics": payload.get("label_semantics"), "split_integrity": payload.get("split_integrity")},
            "metrics": {key: payload.get(key) for key in ["group_count", "train_group_count", "validation_group_count"]},
        }
    if schema == "failure_aware_reranking_eval_v1":
        return {
            "layer_id": "failure_aware_reranking", "source_path": str(path), "schema": schema,
            "summary": {"query_config": payload.get("query_config"), "checkpoint": payload.get("learned_checkpoint")},
            "metrics": payload.get("overview", {}),
        }
    if schema == "risk_case_expansion_v1":
        return {
            "layer_id": "risk_case_coverage", "source_path": str(path), "schema": schema,
            "summary": {"label_semantics": payload.get("label_semantics"), "family_counts": payload.get("family_counts")},
            "metrics": {"case_count": payload.get("case_count"), "scene_count": payload.get("scene_count"),
                        "forecast_case_count": payload.get("forecast", {}).get("case_count")},
        }
    if schema == "carla_semantic_demo_mining_v1":
        return _carla_semantic_demo_mining_entry(path, payload)
    if schema == "carla_vision_closed_loop_batch_v1":
        return _carla_vision_closed_loop_batch_entry(path, payload)
    if schema == "carla_vision_closed_loop_v1":
        return _carla_vision_closed_loop_entry(path, payload)
    if schema == "carla_vision_video_audit_v1":
        return _carla_video_audit_entry(path, payload)
    if schema == "model_in_the_loop_failure_mining_report_v1":
        return _failure_mining_entry(path, payload)
    return {
        "layer_id": "unknown",
        "source_path": str(path),
        "schema": schema,
        "summary": {},
        "metrics": {},
    }


def _risk_suite_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload.get("result") or {})
    outputs = dict(result.get("outputs") or {})
    proxy = dict(result.get("proxy_studies") or {})
    return {
        "layer_id": "risk_benchmark_suite",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "experiment_id": payload.get("experiment_id"),
            "scenario_mining_output": outputs.get("scenario_mining"),
            "perception_output": outputs.get("perception"),
            "world_model_output": outputs.get("world_model"),
            "bev_occupancy_output": outputs.get("bev_occupancy"),
        },
        "metrics": {
            "scenario_anchors": _nested_int(result, ["scenario_mining", "anchor_case_count"]),
            "scenario_queries": _nested_int(result, ["scenario_mining", "query_count"]),
            "perception_cases": _nested_int(result, ["perception", "case_count"]),
            "world_model_cases": _nested_int(result, ["world_model", "case_count"]),
            "bev_occupancy_cases": _nested_int(result, ["bev_occupancy", "case_count"]),
            "perception_proxy_cases": _nested_int(proxy, ["perception", "case_count"]),
            "world_model_proxy_cases": _nested_int(proxy, ["world_model", "case_count"]),
            "bev_occupancy_proxy_cases": _nested_int(proxy, ["bev_occupancy", "case_count"]),
        },
    }


def _nuplan_replay_sweep_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    overview = dict(payload.get("overview") or {})
    overall = _overall_rows(payload)
    best_non_oracle = _best_profile(
        overall,
        metric="mean_risk_fidelity_score",
        exclude={"logged_ego"},
    )
    return {
        "layer_id": "nuplan_replay_regression",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "studies": overview.get("study_count"),
            "profiles": overview.get("profiles", []),
            "best_non_oracle_profile": best_non_oracle.get("profile_name"),
        },
        "metrics": {
            "db_count_scanned": overview.get("db_count_scanned"),
            "candidate_case_count": overview.get("candidate_case_count"),
            "case_count": overview.get("case_count"),
            "best_non_oracle_risk_fidelity": best_non_oracle.get("mean_risk_fidelity_score"),
            "best_non_oracle_ego_ade_m": best_non_oracle.get("mean_ego_ade_m"),
        },
    }


def _nuplan_closed_loop_sweep_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    overview = dict(payload.get("overview") or {})
    overall = _overall_rows(payload)
    best_non_oracle = _best_profile(
        overall,
        metric="mean_closed_loop_score",
        exclude={"logged_ego_oracle"},
    )
    return {
        "layer_id": "nuplan_closed_loop_replay",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "studies": overview.get("study_count"),
            "profiles": overview.get("profiles", []),
            "best_non_oracle_profile": best_non_oracle.get("profile_name"),
        },
        "metrics": {
            "db_count_scanned": overview.get("db_count_scanned"),
            "candidate_case_count": overview.get("candidate_case_count"),
            "case_count": overview.get("case_count"),
            "best_non_oracle_closed_loop_score": best_non_oracle.get("mean_closed_loop_score"),
            "best_non_oracle_ego_ade_m": best_non_oracle.get("mean_ego_ade_m"),
            "best_non_oracle_progress_ratio": best_non_oracle.get("mean_progress_ratio"),
            "best_non_oracle_raw_progress_ratio": best_non_oracle.get(
                "mean_raw_progress_ratio"
            ),
        },
    }


def _failure_mining_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "layer_id": "failure_mining",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "report_md": str(path.with_suffix(".md")),
        },
        "metrics": {
            "source_count": payload.get("source_count") or payload.get("sources"),
            "failure_record_count": payload.get("failure_record_count")
            or payload.get("failure_records")
            or payload.get("record_count"),
            "cluster_count": payload.get("cluster_count") or payload.get("clusters"),
            "update_query_count": payload.get("update_query_count") or len(list(payload.get("update_queries") or [])),
        },
    }


def _bench2drive_training_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    history = list(payload.get("history") or [])
    latest = dict((history[-1] if history else {}).get("val") or {})
    train = dict((history[-1] if history else {}).get("train") or {})
    calibration = dict(payload.get("calibration") or {})
    return {
        "layer_id": "bench2drive_vision_planner",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "checkpoint_path": payload.get("checkpoint_path"),
            "model_size": payload.get("model_size"),
            "architecture": payload.get("architecture"),
            "trajectory_selection": payload.get("trajectory_selection"),
            "trajectory_temperature": payload.get("trajectory_temperature"),
            "calibration_method": calibration.get("method") or calibration.get("rationale"),
            "distributed_world_size": payload.get("distributed_world_size"),
            "precision": payload.get("precision"),
        },
        "metrics": {
            "train_sample_count": payload.get("train_sample_count"),
            "val_sample_count": payload.get("val_sample_count"),
            "runtime_s": payload.get("runtime_s"),
            "train_samples_per_s": train.get("samples_per_s"),
            "uncalibrated_val_ade_m": latest.get("ade_m"),
            "uncalibrated_val_fde_m": latest.get("fde_m"),
            "uncalibrated_val_brake_accuracy": latest.get("brake_accuracy"),
        },
    }


def _bench2drive_eval_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    return {
        "layer_id": "bench2drive_vision_planner",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "checkpoint_path": payload.get("checkpoint_path"),
            "split": payload.get("split"),
            "sample_count": payload.get("sample_count"),
        },
        "metrics": {
            "ade_m": metrics.get("ade_m"),
            "fde_m": metrics.get("fde_m"),
            "loss": metrics.get("loss"),
            "lateral_mae_m": metrics.get("lateral_mae_m"),
            "turn_lateral_mae_m": metrics.get("turn_lateral_mae_m"),
            "oracle_ade_m": metrics.get("oracle_ade_m"),
            "oracle_fde_m": metrics.get("oracle_fde_m"),
            "brake_accuracy": metrics.get("brake_accuracy"),
            "brake_f1": metrics.get("brake_f1"),
        },
    }


def _bench2drive_planner_diagnostics_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = dict(payload.get("aggregate") or {})
    readiness = dict(payload.get("readiness") or {})
    return {
        "layer_id": "bench2drive_vision_planner",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "predictions_path": payload.get("predictions_path"),
            "evaluation_report_path": payload.get("evaluation_report_path"),
            "status": readiness.get("status"),
            "findings": readiness.get("findings", []),
        },
        "metrics": {
            "sample_count": payload.get("sample_count"),
            "mean_ade_m": aggregate.get("mean_ade_m"),
            "mean_fde_m": aggregate.get("mean_fde_m"),
            "mean_path_length_ratio": aggregate.get("mean_path_length_ratio"),
            "underreach_rate": aggregate.get("underreach_rate"),
            "severe_underreach_rate": aggregate.get("severe_underreach_rate"),
            "high_lateral_error_rate": aggregate.get("high_lateral_error_rate"),
            "brake_f1": aggregate.get("brake_f1"),
            "predicted_to_target_speed_ratio": readiness.get("mean_predicted_to_target_speed_ratio"),
        },
    }


def _bench2drive_closed_loop_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    comparison = dict(payload.get("comparison") or {})
    metrics = dict(comparison.get("metrics") or {})
    return {
        "layer_id": "bench2drive_vision_closed_loop",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "checkpoint_path": payload.get("checkpoint_path"),
            "split": payload.get("split"),
            "case_count": payload.get("case_count"),
            "output_dir": payload.get("output_dir"),
        },
        "metrics": {
            "case_count": payload.get("case_count"),
            "closed_loop_ade_m": metrics.get("mean_closed_loop_ade_m"),
            "closed_loop_fde_m": metrics.get("mean_closed_loop_fde_m"),
            "mean_lateral_error_m": metrics.get("mean_mean_lateral_error_m"),
            "route_completion": metrics.get("mean_route_completion"),
            "closed_loop_score": metrics.get("mean_closed_loop_score"),
        },
    }


def _carla_vision_closed_loop_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(payload.get("metrics") or {})
    media = dict(payload.get("media") or {})
    scenario = dict(payload.get("scenario") or {})
    scenario_type = str(scenario.get("type") or "")
    layer_id = "carla_vision_closed_loop"
    return {
        "layer_id": layer_id,
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "town": payload.get("town"),
            "scenario_name": scenario.get("name"),
            "scenario_type": scenario_type,
            "checkpoint_path": payload.get("checkpoint_path"),
            "output_dir": payload.get("output_dir"),
            "gif_path": media.get("gif_path"),
            "video_path": media.get("video_path"),
        },
        "metrics": {
            "route_length_m": payload.get("route_length_m"),
            "frame_count": metrics.get("frame_count"),
            "duration_s": metrics.get("duration_s"),
            "route_completion": metrics.get("route_completion"),
            "driving_score": metrics.get("driving_score"),
            "collision_count": metrics.get("collision_count"),
            "mean_lateral_error_m": metrics.get("mean_lateral_error_m"),
            "max_lateral_error_m": metrics.get("max_lateral_error_m"),
            "comfort_violation_count": metrics.get("comfort_violation_count"),
        },
    }


def _carla_vision_closed_loop_batch_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = dict(payload.get("aggregate") or {})
    scenarios = list(payload.get("scenarios") or [])
    layer_id = "carla_semantic_demo" if payload.get("semantic_targets") or "carla_semantic_demo" in str(path) else "carla_vision_closed_loop"
    return {
        "layer_id": layer_id,
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "town": payload.get("town"),
            "scenario_count": payload.get("scenario_count"),
            "output_dir": payload.get("output_dir"),
            "scenario_names": [dict(row).get("name") for row in scenarios],
        },
        "metrics": {
            "scenario_count": payload.get("scenario_count"),
            "total_frames": aggregate.get("total_frames"),
            "mean_route_completion": aggregate.get("mean_route_completion"),
            "mean_driving_score": aggregate.get("mean_driving_score"),
            "total_collisions": aggregate.get("total_collisions"),
            "total_traffic_manager_vehicles": aggregate.get("total_traffic_manager_vehicles"),
            "total_scripted_vehicles": aggregate.get("total_scripted_vehicles"),
            "total_crosswalk_pedestrians": aggregate.get("total_crosswalk_pedestrians"),
        },
    }


def _carla_video_audit_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    report_path = str(payload.get("report_path") or "")
    layer_id = "carla_semantic_demo" if "carla_semantic_demo" in f"{path} {report_path}" else "carla_vision_closed_loop"
    return {
        "layer_id": layer_id,
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "status": payload.get("status"),
            "report_path": payload.get("report_path"),
            "failures": payload.get("failures", []),
            "warnings": payload.get("warnings", []),
        },
        "metrics": {
            "scenario_count": payload.get("scenario_count"),
            "failure_count": payload.get("failure_count"),
            "warning_count": payload.get("warning_count"),
            "passed_scenarios": summary.get("passed_scenarios"),
            "failed_scenarios": summary.get("failed_scenarios"),
            "total_video_frames": summary.get("total_video_frames"),
            "total_state_rows": summary.get("total_state_rows"),
            "total_collisions": summary.get("total_collisions"),
            "total_traffic_manager_vehicles": summary.get("total_traffic_manager_vehicles"),
            "mean_nearby_actor_ratio": summary.get("mean_nearby_actor_ratio"),
        },
    }


def _carla_semantic_demo_mining_entry(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    final_report = dict(payload.get("final_report") or {})
    final_audit = dict(payload.get("final_audit") or {})
    aggregate = dict(final_report.get("aggregate") or {})
    return {
        "layer_id": "carla_semantic_demo",
        "source_path": str(path),
        "schema": str(payload.get("schema") or ""),
        "summary": {
            "status": payload.get("status"),
            "output_dir": payload.get("output_dir"),
            "report_path": payload.get("report_path"),
            "audit_path": payload.get("audit_path"),
        },
        "metrics": {
            "target_count": payload.get("target_count"),
            "passed_target_count": payload.get("passed_target_count"),
            "attempt_count": payload.get("attempt_count"),
            "scenario_count": final_report.get("scenario_count"),
            "semantic_audit_status": final_audit.get("status"),
            "semantic_audit_failure_count": final_audit.get("failure_count"),
            "semantic_audit_warning_count": final_audit.get("warning_count"),
            "total_frames": aggregate.get("total_frames"),
            "total_traffic_manager_vehicles": aggregate.get("total_traffic_manager_vehicles"),
            "total_scripted_vehicles": aggregate.get("total_scripted_vehicles"),
            "total_crosswalk_pedestrians": aggregate.get("total_crosswalk_pedestrians"),
            "total_collisions": aggregate.get("total_collisions"),
        },
    }


def _overall_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [
        dict(row)
        for row in payload.get("profile_leaderboard", [])
        if str(row.get("study_name") or "") == "__overall__"
    ]


def _best_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    exclude: set[str],
) -> Dict[str, Any]:
    candidates = [dict(row) for row in rows if str(row.get("profile_name") or "") not in exclude]
    candidates = [row for row in candidates if row.get(metric) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row.get(metric) or 0.0))


def _nested_int(payload: Mapping[str, Any], path: Sequence[str]) -> Optional[int]:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if value is None:
        return None
    return int(value)


def _write_registry_csv(entries: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = ["layer_id", "source_path", "schema", "summary", "metrics"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "layer_id": entry.get("layer_id", ""),
                    "source_path": entry.get("source_path", ""),
                    "schema": entry.get("schema", ""),
                    "summary": json.dumps(entry.get("summary", {}), sort_keys=True),
                    "metrics": json.dumps(entry.get("metrics", {}), sort_keys=True),
                }
            )


def _render_registry_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# Benchmark Result Registry",
        "",
        f"- Results: `{overview.get('result_count', 0)}`",
        f"- Layers: `{overview.get('layer_count', 0)}`",
        f"- Existing sources: `{overview.get('existing_source_count', 0)}/{overview.get('source_count', 0)}`",
        f"- Complete: `{overview.get('complete', False)}`",
        "",
        "| Layer | Source | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for entry in payload.get("results", []):
        metrics = ", ".join(f"{key}={_format_metric(value)}" for key, value in dict(entry.get("metrics") or {}).items())
        lines.append(
            f"| `{entry.get('layer_id', '')}` | `{entry.get('source_path', '')}` | {metrics} |"
        )
    missing_sources = list(payload.get("missing_sources") or [])
    if missing_sources:
        lines.extend(["", "## Missing Sources", ""])
        lines.extend(f"- `{path}`" for path in missing_sources)
    lines.append("")
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _infer_source_kind(path: Path) -> str:
    name = str(path).lower()
    if "replay_sweep" in name:
        return "nuplan_replay_sweep"
    if "closed_loop_sweep" in name:
        return "nuplan_closed_loop_sweep"
    if "failure_mining" in name:
        return "failure_mining_report"
    if "bench2drive" in name and "evaluation_report" in name:
        return "bench2drive_eval_report"
    if "bench2drive" in name and "planner_diagnostics" in name:
        return "bench2drive_planner_diagnostics"
    if "bench2drive" in name and "training_report" in name:
        return "bench2drive_training_report"
    if "bench2drive" in name and "closed_loop_report" in name:
        return "bench2drive_closed_loop_report"
    if "carla_semantic_demo" in name:
        if "audit" in name:
            return "carla_semantic_demo_audit"
        if "mining" in name:
            return "carla_semantic_demo_mining_report"
        return "carla_semantic_demo_report"
    if "carla_vision_batch" in name:
        if "video_audit" in name:
            return "carla_vision_video_audit"
        return "carla_vision_batch_report"
    if "carla_vision_closed_loop" in name:
        return "carla_vision_closed_loop_report"
    if "risk_benchmark_suite" in name:
        return "risk_benchmark_suite_result"
    return "result_source"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
