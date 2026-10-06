from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from nusc_scene_agent.artifact_manifest import build_artifact_entry
from nusc_scene_agent.benchmark_exports import write_benchmark_exports
from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.bev_occupancy_benchmark import (
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.benchmark_catalog import build_default_benchmark_catalog, write_benchmark_catalog
from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.bench2drive_closed_loop import (
    DEFAULT_BENCH2DRIVE_CLOSED_LOOP_OUTPUT,
    ClosedLoopControlConfig,
    run_bench2drive_vision_closed_loop,
)
from nusc_scene_agent.bench2drive_e2e import DEFAULT_BENCH2DRIVE_OUTPUT
from nusc_scene_agent.carla_semantic_demo_mining import mine_carla_semantic_demos
from nusc_scene_agent.carla_vision_closed_loop import run_carla_vision_closed_loop
from nusc_scene_agent.case_library import build_case_library, write_case_library
from nusc_scene_agent.case_library_enrichment import enrich_case_library
from nusc_scene_agent.dataset_backends import inspect_dataset_backends, write_dataset_backend_inventory
from nusc_scene_agent.failure_mining import DEFAULT_FAILURE_MINING_OUTPUT, mine_model_failures
from nusc_scene_agent.failure_aware_reranking import DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT, DEFAULT_FAILURE_UPDATE_QUERIES, run_failure_aware_reranking_eval
from nusc_scene_agent.learned_retrieval import (
    DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
    DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT,
    LearnedRetrieverConfig,
    train_weakly_supervised_scene_retriever,
)
from nusc_scene_agent.llm_client import (
    DEFAULT_TIMEOUT_S,
    LLMConfig,
    inspect_ollama_model,
    verify_ollama_model,
)
from nusc_scene_agent.nuplan_closed_loop import run_nuplan_closed_loop_study
from nusc_scene_agent.nuplan_closed_loop_sweep import run_nuplan_closed_loop_sweep
from nusc_scene_agent.nuplan_replay import run_nuplan_replay_study
from nusc_scene_agent.nuplan_sweep import run_nuplan_replay_sweep
from nusc_scene_agent.perception_benchmark import (
    generate_perception_benchmark_from_scenario_config,
    run_proxy_perception_study,
)
from nusc_scene_agent.pipeline import run_query_pipeline
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.result_registry import write_result_registry
from nusc_scene_agent.validation import ValidationConfig
from nusc_scene_agent.world_model_benchmark import (
    generate_world_model_benchmark_from_perception_benchmark,
    run_proxy_world_model_study,
)


EXPERIMENT_RESULT_SCHEMA = "experiment_result_v1"
DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT = Path("outputs/trainval_case_library_v1")
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"


def run_experiment_config(config_path: Path, *, reuse_case_library: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict(config.get("experiment") or {})
    experiment_id = str(experiment.get("id") or config_path.stem)
    experiment_type = str(experiment.get("type") or "")
    runtime_overrides = {}
    if reuse_case_library:
        if experiment_type != "full_benchmark_suite":
            raise ValueError("Case-library reuse is supported only for a full benchmark suite.")
        suite = config.setdefault("full_benchmark_suite", {})
        risk = config.setdefault("risk_benchmark_suite", {})
        generation = dict(config.get("case_library_generation") or {})
        library = Path(str(risk.get("case_library") or Path(str(generation.get("output") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT))
                           / ("case_library_enriched.json" if generation.get("enrich", True) else "case_library.json")))
        if not library.is_file():
            raise FileNotFoundError(f"Validated case library is unavailable: {library}")
        entries = json.loads(library.read_text())
        if not isinstance(entries, list) or not any(
            isinstance(entry, Mapping) and entry.get("passed") is True for entry in entries
        ):
            raise ValueError("Case-library reuse requires existing validated cases.")
        suite.setdefault("stages", {})["case_library_generation"] = False
        risk["case_library"] = str(library)
        runtime_overrides["reused_case_library"] = build_artifact_entry(library, "input", "validated_case_library").to_dict()

    if experiment_type == "nuplan_replay_study":
        result = _run_nuplan_replay_experiment(config)
    elif experiment_type == "nuplan_closed_loop_study":
        result = _run_nuplan_closed_loop_experiment(config)
    elif experiment_type == "nuplan_replay_sweep":
        result = _run_nuplan_replay_sweep_experiment(config)
    elif experiment_type == "nuplan_closed_loop_sweep":
        result = _run_nuplan_closed_loop_sweep_experiment(config)
    elif experiment_type == "carla_vision_closed_loop":
        result = _run_carla_vision_closed_loop_experiment(config)
    elif experiment_type == "carla_semantic_demo_mining":
        result = _run_carla_semantic_demo_mining_experiment(config)
    elif experiment_type == "bench2drive_vision_closed_loop":
        result = _run_bench2drive_vision_closed_loop_experiment(config)
    elif experiment_type == "bench2drive_study":
        from nusc_scene_agent.bench2drive_study import run_bench2drive_study
        result = run_bench2drive_study(dict(config.get("bench2drive_study") or {}))
    elif experiment_type == "carla_fixed_evaluation":
        from nusc_scene_agent.carla_fixed_evaluation import run_fixed_carla_evaluation
        result = run_fixed_carla_evaluation(dict(config.get("carla_fixed_evaluation") or {}))
    elif experiment_type == "risk_case_expansion":
        from nusc_scene_agent.risk_case_expansion import expand_risk_cases
        stage = dict(config.get("risk_case_expansion") or {})
        result = expand_risk_cases(Path(stage["db"]), Path(stage["output"]),
                                   per_family=int(stage.get("per_family", 30)), candidate_limit=int(stage.get("candidate_limit", 180)),
                                   scene_split=str(stage.get("scene_split", "all")),
                                   reuse_validated=bool(stage.get("reuse_validated", False)))
    elif experiment_type == "full_benchmark_suite":
        result = _run_full_benchmark_suite_experiment(config, config_path)
    elif experiment_type == "case_library_generation":
        result = _run_case_library_generation_experiment(config)
    elif experiment_type == "risk_benchmark_suite":
        result = _run_risk_benchmark_suite_experiment(config)
    elif experiment_type == "bev_occupancy_study":
        result = _run_bev_occupancy_experiment(config)
    elif experiment_type == "failure_mining":
        result = _run_failure_mining_experiment(config)
    elif experiment_type == "failure_aware_reranking":
        result = _run_failure_aware_reranking_experiment(config)
    elif experiment_type == "learned_retriever_training":
        result = _run_learned_retriever_training_experiment(config)
    elif experiment_type == "contextvae_world_model_study":
        result = _run_contextvae_experiment(config)
    elif experiment_type == "forecast_validation_study":
        result = _run_forecast_validation_experiment(config)
    elif experiment_type == "catalog_export":
        result = _run_catalog_export_experiment(config)
    elif experiment_type == "result_registry":
        result = _run_result_registry_experiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    payload = {
        "schema": EXPERIMENT_RESULT_SCHEMA,
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "config_path": str(config_path),
        "runtime_overrides": runtime_overrides,
        "result": result,
    }
    output_path = _experiment_result_path(config, config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_nuplan_replay_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    replay = dict(config.get("nuplan_replay") or {})
    manifest = run_nuplan_replay_study(
        split_dir=Path(str(replay.get("split_dir") or "data/nuplan/dataset/nuplan-v1.1/splits/mini")),
        output_dir=Path(str(replay.get("output") or "outputs/nuplan_replay_study_v1")),
        max_dbs=int(replay.get("max_dbs") or 4),
        max_cases=int(replay.get("max_cases") or 16),
        max_cases_per_db=int(replay.get("max_cases_per_db") or 4),
        history_s=float(replay.get("history_s") if replay.get("history_s") is not None else 2.0),
        future_s=float(replay.get("future_s") if replay.get("future_s") is not None else 4.0),
        frame_hz=float(replay.get("frame_hz") if replay.get("frame_hz") is not None else 2.0),
        min_anchor_gap_s=float(
            replay.get("min_anchor_gap_s") if replay.get("min_anchor_gap_s") is not None else 4.0
        ),
        scenario_tags=replay.get("scenario_tags") or None,
        profiles=replay.get("profiles") or None,
    )
    return {
        "output_dir": str(replay.get("output") or "outputs/nuplan_replay_study_v1"),
        "benchmark": manifest.get("benchmark", {}),
        "comparison": manifest.get("comparison", {}),
        "case_studies": manifest.get("case_studies", {}),
        "unified_cases": manifest.get("unified_cases", {}),
        "artifact_manifest": manifest.get("artifact_manifest", {}),
    }


def _run_nuplan_replay_sweep_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    sweep = dict(config.get("nuplan_replay_sweep") or {})
    output_dir = Path(str(sweep.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/nuplan_replay_sweep_v1"))
    result = run_nuplan_replay_sweep(
        studies=list(sweep.get("studies") or []),
        output_dir=output_dir,
        defaults=dict(sweep.get("defaults") or {}),
        profiles=sweep.get("profiles") or None,
    )
    return {
        "output_dir": str(output_dir),
        "overview": result.get("overview", {}),
        "studies": result.get("studies", []),
        "artifact_manifest": result.get("artifact_manifest", {}),
    }


def _run_nuplan_closed_loop_sweep_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    sweep = dict(config.get("nuplan_closed_loop_sweep") or {})
    output_dir = Path(
        str(sweep.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/nuplan_closed_loop_sweep_v1")
    )
    result = run_nuplan_closed_loop_sweep(
        studies=list(sweep.get("studies") or []),
        output_dir=output_dir,
        defaults=dict(sweep.get("defaults") or {}),
        profiles=sweep.get("profiles") or None,
    )
    return {
        "output_dir": str(output_dir),
        "overview": result.get("overview", {}),
        "studies": result.get("studies", []),
        "artifact_manifest": result.get("artifact_manifest", {}),
    }


def _run_nuplan_closed_loop_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    closed_loop = dict(config.get("nuplan_closed_loop") or {})
    manifest = run_nuplan_closed_loop_study(
        split_dir=Path(str(closed_loop.get("split_dir") or "data/nuplan/dataset/nuplan-v1.1/splits/mini")),
        output_dir=Path(str(closed_loop.get("output") or "outputs/nuplan_closed_loop_study_v1")),
        max_dbs=int(closed_loop.get("max_dbs") or 64),
        max_cases=int(closed_loop.get("max_cases") or 16),
        max_cases_per_db=int(closed_loop.get("max_cases_per_db") or 4),
        history_s=float(closed_loop.get("history_s") if closed_loop.get("history_s") is not None else 2.0),
        future_s=float(closed_loop.get("future_s") if closed_loop.get("future_s") is not None else 4.0),
        frame_hz=float(closed_loop.get("frame_hz") if closed_loop.get("frame_hz") is not None else 2.0),
        min_anchor_gap_s=float(
            closed_loop.get("min_anchor_gap_s") if closed_loop.get("min_anchor_gap_s") is not None else 4.0
        ),
        scenario_tags=closed_loop.get("scenario_tags") or None,
        profiles=closed_loop.get("profiles") or None,
    )
    return {
        "output_dir": str(closed_loop.get("output") or "outputs/nuplan_closed_loop_study_v1"),
        "benchmark": manifest.get("benchmark", {}),
        "comparison": manifest.get("comparison", {}),
        "case_studies": manifest.get("case_studies", {}),
        "artifact_manifest": manifest.get("artifact_manifest", {}),
    }


def _run_bench2drive_vision_closed_loop_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    closed_loop = dict(config.get("bench2drive_vision_closed_loop") or {})
    output_dir = Path(str(closed_loop.get("output") or "outputs/bench2drive_vision_closed_loop_v1"))
    report = run_bench2drive_vision_closed_loop(
        manifest_path=Path(str(closed_loop.get("manifest") or "artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl")),
        checkpoint_path=Path(str(closed_loop.get("checkpoint") or DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt")),
        output_dir=output_dir,
        split=str(closed_loop.get("split") or "val"),
        max_cases=int(closed_loop.get("max_cases") or 64),
        max_frames_per_clip=int(closed_loop.get("max_frames_per_clip") or 20),
        image_size=int(closed_loop.get("image_size") or 160),
        device=str(closed_loop.get("device") or ""),
        video_fps=int(closed_loop.get("video_fps") or 6),
        render_case_media=bool(closed_loop.get("render_case_media", False)),
        case_selection=str(closed_loop.get("case_selection") or "balanced"),
        control_config=ClosedLoopControlConfig(
            dt_s=float(closed_loop.get("dt_s") if closed_loop.get("dt_s") is not None else 0.5),
            horizon_s=float(closed_loop.get("horizon_s") if closed_loop.get("horizon_s") is not None else 10.0),
            target_speed_mps=float(closed_loop.get("target_speed_mps") if closed_loop.get("target_speed_mps") is not None else 5.5),
            brake_probability_threshold=float(closed_loop.get("brake_threshold") if closed_loop.get("brake_threshold") is not None else 0.85),
            lookahead_m=float(closed_loop.get("lookahead_m") if closed_loop.get("lookahead_m") is not None else 9.0),
            speed_kp=float(closed_loop.get("speed_kp") if closed_loop.get("speed_kp") is not None else 0.45),
        ),
    )
    return {
        "output_dir": str(output_dir),
        "case_count": report.get("case_count"),
        "comparison": report.get("comparison", {}),
        "cases": report.get("cases", []),
    }


def _run_carla_vision_closed_loop_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = dict(config.get("carla_vision_closed_loop") or {})
    output_dir = Path(str(stage.get("output") or "outputs/carla_vision_closed_loop_v1"))
    report = run_carla_vision_closed_loop(
        carla_root=Path(str(stage.get("carla_root") or "external/carla/latest")),
        checkpoint_path=Path(str(stage.get("checkpoint") or DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt")),
        output_dir=output_dir,
        host=str(stage.get("host") or "127.0.0.1"),
        port=int(stage.get("port") or 2000),
        town=str(stage.get("town") or ""),
        spawn_index=int(stage.get("spawn_index") or 0),
        destination_index=int(stage.get("destination_index") if stage.get("destination_index") is not None else -1),
        route_sampling_resolution_m=float(
            stage.get("route_sampling_resolution_m")
            if stage.get("route_sampling_resolution_m") is not None
            else 2.0
        ),
        route_min_length_m=float(
            stage.get("route_min_length_m") if stage.get("route_min_length_m") is not None else 40.0
        ),
        route_max_length_m=float(
            stage.get("route_max_length_m") if stage.get("route_max_length_m") is not None else 220.0
        ),
        route_preferred_length_m=float(
            stage.get("route_preferred_length_m") if stage.get("route_preferred_length_m") is not None else 0.0
        ),
        fps=int(stage.get("fps") or 10),
        horizon_s=float(stage.get("horizon_s") if stage.get("horizon_s") is not None else 30.0),
        image_size=int(stage.get("image_size") or 160),
        camera_width=int(stage.get("camera_width") or 320),
        camera_height=int(stage.get("camera_height") or 180),
        video_width=int(stage.get("video_width") or 0),
        video_height=int(stage.get("video_height") or 0),
        camera_fov=float(stage.get("camera_fov") if stage.get("camera_fov") is not None else 90.0),
        carla_quality_level=str(stage.get("carla_quality_level") or stage.get("quality_level") or "Epic"),
        scenario_type=str(stage.get("scenario_type") or "free_drive"),
        scenario_name=str(stage.get("scenario_name") or ""),
        scenario_params=dict(stage.get("scenario_params") or stage.get("parameters") or {}),
        scenarios=list(stage.get("scenarios") or []),
        target_speed_mps=float(stage.get("target_speed_mps") if stage.get("target_speed_mps") is not None else 7.0),
        brake_probability_threshold=float(stage.get("brake_threshold") if stage.get("brake_threshold") is not None else 0.75),
        enable_scenario_safety_override=bool(stage.get("enable_scenario_safety_override", True)),
        enable_lane_departure_guard=bool(stage.get("enable_lane_departure_guard", True)),
        condition_ego_route_traffic_lights=bool(stage.get("condition_ego_route_traffic_lights", False)),
        device=str(stage.get("device") or ""),
        auto_launch=bool(stage.get("auto_launch", False)),
        cuda_visible_devices=str(stage.get("cuda_visible_devices") or ""),
        graphics_adapter=int(stage.get("graphics_adapter", -1)),
        traffic_manager_port=int(stage.get("traffic_manager_port") or 8000),
        seed=int(stage.get("seed", 7)),
        launch_timeout_s=float(stage.get("launch_timeout_s") if stage.get("launch_timeout_s") is not None else 90.0),
        rpc_timeout_s=float(stage.get("rpc_timeout_s") if stage.get("rpc_timeout_s") is not None else 30.0),
        keep_server=bool(stage.get("keep_server", False)),
        video_fps=int(stage.get("video_fps") or 10),
        video_encoder=str(stage.get("video_encoder") or "hevc_nvenc"),
        video_nvenc_preset=str(stage.get("video_nvenc_preset") or "p4"),
        video_quality=int(stage.get("video_quality") or 23),
        render_gif=bool(stage.get("render_gif", True)),
    )
    if str(report.get("schema") or "") == "carla_vision_closed_loop_batch_v1":
        aggregate = dict(report.get("aggregate") or {})
        return {
            "schema": report.get("schema"),
            "output_dir": str(output_dir),
            "metrics": aggregate,
            "media": {},
            "route_length_m": None,
            "scenario_count": report.get("scenario_count"),
            "scenarios": report.get("scenarios", []),
        }
    return {
        "schema": report.get("schema"),
        "output_dir": str(output_dir),
        "metrics": report.get("metrics", {}),
        "media": report.get("media", {}),
        "route_length_m": report.get("route_length_m"),
    }


def _run_carla_semantic_demo_mining_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = dict(config.get("carla_semantic_demo") or config.get("carla_semantic_demo_mining") or {})
    output_dir = Path(str(stage.get("output") or "outputs/carla_semantic_demo_final"))
    result = mine_carla_semantic_demos(
        carla_root=Path(str(stage.get("carla_root") or "external/carla/latest")),
        checkpoint_path=Path(str(stage.get("checkpoint") or DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt")),
        output_dir=output_dir,
        trials_output_dir=Path(str(stage.get("trials_output") or "outputs/carla_semantic_demo_trials")),
        host=str(stage.get("host") or "127.0.0.1"),
        port_start=int(stage.get("port_start") or 2040),
        town=str(stage.get("town") or "Town10HD_Opt"),
        fps=int(stage.get("fps") or 10),
        horizon_s=float(stage.get("horizon_s") if stage.get("horizon_s") is not None else 18.0),
        image_size=int(stage.get("image_size") or 160),
        camera_width=int(stage.get("camera_width") or 640),
        camera_height=int(stage.get("camera_height") or 360),
        video_width=int(stage.get("video_width") or 1920),
        video_height=int(stage.get("video_height") or 1080),
        camera_fov=float(stage.get("camera_fov") if stage.get("camera_fov") is not None else 90.0),
        carla_quality_level=str(stage.get("carla_quality_level") or stage.get("quality_level") or "Epic"),
        brake_probability_threshold=float(stage.get("brake_threshold") if stage.get("brake_threshold") is not None else 0.82),
        device=str(stage.get("device") or "cuda"),
        auto_launch=bool(stage.get("auto_launch", True)),
        cuda_visible_devices=str(stage.get("cuda_visible_devices") or ""),
        traffic_manager_port_start=int(stage.get("traffic_manager_port_start") or 8040),
        launch_timeout_s=float(stage.get("launch_timeout_s") if stage.get("launch_timeout_s") is not None else 180.0),
        rpc_timeout_s=float(stage.get("rpc_timeout_s") if stage.get("rpc_timeout_s") is not None else 90.0),
        keep_server=bool(stage.get("keep_server", False)),
        reuse_carla_server=bool(stage.get("reuse_carla_server", True)),
        video_fps=int(stage.get("video_fps") or 10),
        video_encoder=str(stage.get("video_encoder") or "hevc_nvenc"),
        video_nvenc_preset=str(stage.get("video_nvenc_preset") or "p4"),
        video_quality=int(stage.get("video_quality") or 23),
        render_gif=bool(stage.get("render_gif", False)),
        enable_scenario_safety_override=bool(stage.get("enable_scenario_safety_override", True)),
        enable_lane_departure_guard=bool(stage.get("enable_lane_departure_guard", False)),
        condition_ego_route_traffic_lights=bool(stage.get("condition_ego_route_traffic_lights", True)),
        route_sampling_resolution_m=float(
            stage.get("route_sampling_resolution_m")
            if stage.get("route_sampling_resolution_m") is not None
            else 2.0
        ),
        route_min_length_m=float(
            stage.get("route_min_length_m") if stage.get("route_min_length_m") is not None else 55.0
        ),
        route_max_length_m=float(
            stage.get("route_max_length_m") if stage.get("route_max_length_m") is not None else 160.0
        ),
        route_preferred_length_m=float(
            stage.get("route_preferred_length_m") if stage.get("route_preferred_length_m") is not None else 95.0
        ),
        max_attempts_per_target=int(stage.get("max_attempts_per_target") or 4),
        targets=list(stage.get("targets") or []),
        min_resolution_width=int(stage.get("min_resolution_width") or 1920),
        min_resolution_height=int(stage.get("min_resolution_height") or 1080),
        min_fps=float(stage.get("min_fps") if stage.get("min_fps") is not None else 8.0),
        min_frames=int(stage.get("min_frames") or 80),
        min_traffic_manager_vehicles=int(
            stage.get("min_traffic_manager_vehicles")
            if stage.get("min_traffic_manager_vehicles") is not None
            else 1
        ),
        min_route_completion=float(
            stage.get("min_route_completion") if stage.get("min_route_completion") is not None else 0.05
        ),
        max_mean_lateral_error_m=float(
            stage.get("max_mean_lateral_error_m") if stage.get("max_mean_lateral_error_m") is not None else 2.5
        ),
        max_lateral_error_m=float(
            stage.get("max_lateral_error_m") if stage.get("max_lateral_error_m") is not None else 6.0
        ),
        max_safety_override_ratio=float(
            stage.get("max_safety_override_ratio") if stage.get("max_safety_override_ratio") is not None else 0.45
        ),
        max_nearest_actor_distance_m=float(
            stage.get("max_nearest_actor_distance_m")
            if stage.get("max_nearest_actor_distance_m") is not None
            else 60.0
        ),
        nearby_actor_distance_m=float(
            stage.get("nearby_actor_distance_m") if stage.get("nearby_actor_distance_m") is not None else 30.0
        ),
        min_nearby_actor_ratio=float(
            stage.get("min_nearby_actor_ratio") if stage.get("min_nearby_actor_ratio") is not None else 0.30
        ),
        require_hevc=bool(stage.get("require_hevc", True)),
    )
    return result


def _run_case_library_generation_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = dict(config.get("case_library_generation") or {})
    experiment = dict(config.get("experiment") or {})
    output_dir = Path(str(stage.get("output") or experiment.get("output") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT))
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(str(stage.get("benchmark") or "benchmarks/trainval_suite_v1.yaml"))
    db_path = Path(str(stage.get("db") or "artifacts/index/v1.0-trainval.sqlite"))
    candidate_pool = int(stage.get("candidate_pool") or 12)
    query_mode = str(stage.get("query_mode") or "hybrid")
    rerank_mode = str(stage.get("rerank_mode") or "llm")
    llm_config = _stage_llm_config(stage, query_mode=query_mode, rerank_mode=rerank_mode)
    llm_model_metadata: Dict[str, Any] = {}
    llm_model_metadata_path = output_dir / "ollama_model_metadata.json"
    if llm_config is not None:
        llm_model_metadata = (
            verify_ollama_model(llm_config)
            if llm_config.digest or llm_config.require_digest
            else inspect_ollama_model(llm_config)
        )
        llm_config.resolved_digest = str(llm_model_metadata.get("digest") or "")
        llm_model_metadata_path.write_text(
            json.dumps(llm_model_metadata, indent=2),
            encoding="utf-8",
        )

    summaries: List[Dict[str, Any]] = []
    benchmark_results: List[Dict[str, Any]] = []
    validation_config = ValidationConfig()
    for spec in load_benchmark_config(benchmark_path):
        result = run_query_pipeline(
            db_path=db_path,
            query_text=spec.natural_language,
            output_root=output_dir,
            top_k=spec.top_k,
            candidate_pool=spec.candidate_pool or candidate_pool,
            benchmark_spec=spec,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            validation_config=validation_config,
        )
        result["id"] = spec.id
        benchmark_results.append(result)
        summaries.append(
            {
                "id": spec.id,
                "description": spec.description,
                "query_dir": str(result["query_dir"]),
                "candidate_count": result["candidate_count"],
                "selected_count": result["selected_count"],
                "tags": spec.tags,
                "behaviors": spec.behaviors,
                "actors": spec.actors,
            }
        )

    (output_dir / "benchmark_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    case_library_entries = build_case_library(benchmark_results)
    write_case_library(case_library_entries, output_dir)
    write_benchmark_metrics(build_benchmark_metrics(benchmark_results, case_library_entries), output_dir)
    write_benchmark_exports(benchmark_results, case_library_entries, output_dir)

    case_library_path = output_dir / "case_library.json"
    enriched_path = output_dir / "case_library_enriched.json"
    enrichment: Optional[Dict[str, Any]] = None
    if bool(stage.get("enrich", True)):
        enrichment = enrich_case_library(
            case_library_path=case_library_path,
            db_path=db_path,
            output_path=Path(str(stage.get("enriched_output") or enriched_path)),
        )
        enriched_path = Path(str(enrichment["output_path"]))

    return {
        "output_dir": str(output_dir),
        "benchmark": str(benchmark_path),
        "db": str(db_path),
        "query_mode": query_mode,
        "rerank_mode": rerank_mode,
        "llm": llm_config.to_dict() if llm_config is not None else None,
        "ollama_model_metadata": str(llm_model_metadata_path) if llm_model_metadata else "",
        "query_count": len(summaries),
        "case_count": len(case_library_entries),
        "case_library": str(case_library_path),
        "enriched_case_library": str(enriched_path) if enriched_path.exists() else "",
        "enrichment": enrichment or {},
    }


def _run_full_benchmark_suite_experiment(config: Mapping[str, Any], config_path: Path) -> Dict[str, Any]:
    suite = dict(config.get("full_benchmark_suite") or {})
    experiment = dict(config.get("experiment") or {})
    output_dir = Path(str(suite.get("output") or experiment.get("output") or "outputs/full_benchmark_suite_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_results: Dict[str, Any] = {}
    stage_result_paths: Dict[str, str] = {}
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")

    if _stage_enabled(suite, "case_library_generation", default=bool(config.get("case_library_generation"))):
        stage_config = _merged_stage_config(config, suite, "case_library_generation")
        stage_output = Path(str(stage_config.get("output") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT))
        stage_experiment = {
            "id": "full_case_library_generation",
            "type": "case_library_generation",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_case_library_generation_experiment(
            {"experiment": stage_experiment, "case_library_generation": stage_config}
        )
        stage_results["case_library_generation"] = result
        stage_result_paths["case_library_generation"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#case_library_generation", result)

    if _stage_enabled(suite, "risk_benchmark_suite", default=True):
        stage_config = _merged_stage_config(config, suite, "risk_benchmark_suite")
        generated_library = dict(stage_results.get("case_library_generation") or {})
        if generated_library and not stage_config.get("case_library"):
            stage_config["case_library"] = (
                generated_library.get("enriched_case_library")
                or generated_library.get("case_library")
                or str(DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT / "case_library_enriched.json")
            )
        stage_experiment = {
            "id": "full_risk_benchmark_suite",
            "type": "risk_benchmark_suite",
            "output": str(stage_config.get("output") or "outputs/risk_benchmark_suite_v1"),
            "result_path": str(Path(str(stage_config.get("result_path") or "outputs/risk_benchmark_suite_v1/experiment_result.json"))),
        }
        result = _run_risk_benchmark_suite_experiment(
            {"experiment": stage_experiment, "risk_benchmark_suite": stage_config}
        )
        stage_results["risk_benchmark_suite"] = result
        stage_result_paths["risk_benchmark_suite"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#risk_benchmark_suite", result)

    if _stage_enabled(suite, "nuplan_replay_sweep", default=True):
        stage_config = _merged_stage_config(config, suite, "nuplan_replay_sweep")
        stage_output = Path(str(stage_config.get("output") or "outputs/nuplan_replay_sweep_v1"))
        stage_experiment = {
            "id": "full_nuplan_replay_sweep",
            "type": "nuplan_replay_sweep",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_nuplan_replay_sweep_experiment(
            {"experiment": stage_experiment, "nuplan_replay_sweep": stage_config}
        )
        stage_results["nuplan_replay_sweep"] = result
        stage_result_paths["nuplan_replay_sweep"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#nuplan_replay_sweep", result)

    if _stage_enabled(suite, "nuplan_closed_loop_sweep", default=True):
        stage_config = _merged_stage_config(config, suite, "nuplan_closed_loop_sweep")
        stage_output = Path(str(stage_config.get("output") or "outputs/nuplan_closed_loop_sweep_v1"))
        stage_experiment = {
            "id": "full_nuplan_closed_loop_sweep",
            "type": "nuplan_closed_loop_sweep",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_nuplan_closed_loop_sweep_experiment(
            {"experiment": stage_experiment, "nuplan_closed_loop_sweep": stage_config}
        )
        stage_results["nuplan_closed_loop_sweep"] = result
        stage_result_paths["nuplan_closed_loop_sweep"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#nuplan_closed_loop_sweep", result)

    for stage_name, runner in [
        ("learned_retriever_training", _run_learned_retriever_training_experiment),
        ("contextvae_world_model_study", _run_contextvae_experiment),
        ("bench2drive_study", None),
        ("bench2drive_vision_closed_loop", _run_bench2drive_vision_closed_loop_experiment),
        ("carla_vision_closed_loop", _run_carla_vision_closed_loop_experiment),
        ("carla_fixed_evaluation", None),
        ("risk_case_expansion", None),
        ("contextvae_expanded_study", None),
        ("forecast_validation_study", _run_forecast_validation_experiment),
    ]:
        if not _stage_enabled(suite, stage_name, default=False):
            continue
        stage_config = _merged_stage_config(config, suite, stage_name)
        if stage_name == "contextvae_world_model_study" and not stage_config.get("benchmark"):
            stage_config["benchmark"] = dict(stage_results.get("risk_benchmark_suite", {}).get("outputs", {})).get("world_model")
        if stage_config.get("config_path"):
            result_payload = run_experiment_config(Path(str(stage_config["config_path"])))
            result = result_payload["result"]
        elif stage_name == "bench2drive_study":
            from nusc_scene_agent.bench2drive_study import run_bench2drive_study
            result = run_bench2drive_study(stage_config)
        elif stage_name == "carla_fixed_evaluation":
            from nusc_scene_agent.carla_fixed_evaluation import run_fixed_carla_evaluation
            result = run_fixed_carla_evaluation(stage_config)
        elif stage_name == "risk_case_expansion":
            from nusc_scene_agent.risk_case_expansion import expand_risk_cases
            result = expand_risk_cases(Path(stage_config["db"]), Path(stage_config["output"]),
                                       per_family=int(stage_config.get("per_family", 30)),
                                       candidate_limit=int(stage_config.get("candidate_limit", 180)),
                                       scene_split=str(stage_config.get("scene_split", "all")),
                                       reuse_validated=bool(stage_config.get("reuse_validated", False)))
        elif stage_name == "contextvae_expanded_study":
            result = _run_contextvae_experiment({"contextvae_world_model_study": stage_config})
        else:
            result = runner({stage_name: stage_config})
        stage_results[stage_name] = result
        stage_experiment = {"id": f"full_{stage_name}", "type": stage_name,
                            "result_path": str(output_dir / f"{stage_name}_result.json")}
        stage_result_paths[stage_name] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#{stage_name}", result)

    if _stage_enabled(suite, "failure_mining", default=True):
        stage_config = _merged_stage_config(config, suite, "failure_mining")
        if not stage_config.get("inputs"):
            stage_config["inputs"] = _full_suite_failure_inputs(stage_results)
        stage_output = Path(str(stage_config.get("output") or DEFAULT_FAILURE_MINING_OUTPUT))
        stage_experiment = {
            "id": "full_failure_mining",
            "type": "failure_mining",
            "output": str(stage_output),
            "result_path": str(stage_output / "experiment_result.json"),
        }
        result = _run_failure_mining_experiment({"experiment": stage_experiment, "failure_mining": stage_config})
        stage_results["failure_mining"] = result
        stage_result_paths["failure_mining"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#failure_mining", result)

    if _stage_enabled(suite, "failure_aware_reranking", default=False):
        stage_config = _merged_stage_config(config, suite, "failure_aware_reranking")
        failure = dict(stage_results.get("failure_mining") or {})
        if not stage_config.get("query_config") and failure.get("output_dir"):
            stage_config["query_config"] = str(Path(failure["output_dir"]) / "failure_update_queries.yaml")
        training = dict(stage_results.get("learned_retriever_training") or {})
        if not stage_config.get("learned_checkpoint") and training.get("output_dir"):
            stage_config["learned_checkpoint"] = str(Path(training["output_dir"]) / "learned_retriever.pt")
        result = _run_failure_aware_reranking_experiment({"failure_aware_reranking": stage_config})
        stage_results["failure_aware_reranking"] = result
        stage_experiment = {"id": "full_failure_aware_reranking", "type": "failure_aware_reranking",
                            "result_path": str(output_dir / "failure_aware_reranking_result.json")}
        stage_result_paths["failure_aware_reranking"] = stage_experiment["result_path"]
        _write_stage_result(stage_experiment, f"{config_path}#failure_aware_reranking", result)

    if _stage_enabled(suite, "result_registry", default=True):
        registry_config = _merged_stage_config(config, suite, "result_registry")
        registry_output = Path(str(registry_config.get("output") or output_dir / "result_registry"))
        sources = registry_config.get("sources") or _full_suite_result_sources(stage_results, stage_result_paths)
        registry = write_result_registry(
            output_dir=registry_output,
            sources=[Path(str(path)) for path in sources],
            metadata={"suite_output": str(output_dir), "config_path": str(config_path)},
        )
        stage_results["result_registry"] = {
            "output_dir": str(registry_output),
            "overview": registry.get("overview", {}),
            "artifact_manifest": registry.get("artifact_manifest", {}),
        }

    summary = {
        "output_dir": str(output_dir),
        "stages": stage_results,
        "stage_result_paths": stage_result_paths,
    }
    (output_dir / "full_benchmark_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "full_benchmark_suite_summary.md").write_text(_render_full_suite_markdown(summary), encoding="utf-8")
    return summary


def _run_bev_occupancy_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    bev = dict(config.get("bev_occupancy") or {})
    benchmark_output = Path(str(bev.get("benchmark_output") or "benchmarks/trainval_bev_occupancy_slices_v1.json"))
    output_dir = Path(str(bev.get("output") or dict(config.get("experiment") or {}).get("output") or "outputs/trainval_bev_occupancy_proxy_study_v1"))
    benchmark = generate_bev_occupancy_benchmark_from_perception_benchmark(
        perception_benchmark_path=Path(str(bev.get("perception_benchmark") or "benchmarks/trainval_perception_slices_v1.json")),
        db_path=Path(str(bev.get("db") or "artifacts/index/v1.0-trainval.sqlite")),
        output_path=benchmark_output,
        grid_spec=bev.get("grid_spec") or None,
    )
    study = run_proxy_bev_occupancy_study(benchmark_output, output_dir)
    return {
        "benchmark": benchmark,
        "study": study,
    }


def _run_risk_benchmark_suite_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    suite = dict(config.get("risk_benchmark_suite") or {})
    experiment = dict(config.get("experiment") or {})
    output_root = Path(str(suite.get("output") or experiment.get("output") or "outputs/risk_benchmark_suite_v1"))
    db_path = Path(str(suite.get("db") or "artifacts/index/v1.0-trainval.sqlite"))
    case_library_path = Path(
        str(suite.get("case_library") or DEFAULT_TRAINVAL_CASE_LIBRARY_OUTPUT / "case_library_enriched.json")
    )
    max_cases = int(suite.get("max_cases") or 24)
    grid_spec = suite.get("grid_spec") or None

    scenario_output = Path(str(suite.get("scenario_output") or "benchmarks/trainval_scenario_mining_v1.yaml"))
    perception_output = Path(str(suite.get("perception_output") or "benchmarks/trainval_perception_slices_v1.json"))
    world_model_output = Path(str(suite.get("world_model_output") or "benchmarks/trainval_world_model_slices_v2.json"))
    bev_output = Path(str(suite.get("bev_occupancy_output") or "benchmarks/trainval_bev_occupancy_slices_v1.json"))

    scenario = generate_scenario_mining_benchmark_from_case_library(
        case_library_path=case_library_path,
        output_path=scenario_output,
        max_cases=max_cases,
    )
    perception = generate_perception_benchmark_from_scenario_config(
        config_path=scenario_output,
        db_path=db_path,
        output_path=perception_output,
    )
    world_model = generate_world_model_benchmark_from_perception_benchmark(
        perception_benchmark_path=perception_output,
        db_path=db_path,
        output_path=world_model_output,
        grid_spec=suite.get("world_model_grid_spec") or grid_spec,
        history_s=float(suite.get("history_s", 2.0)),
        future_s=float(suite.get("future_s", 6.0)),
        min_future_s=float(suite.get("min_future_s", 0.0)),
    )
    bev_occupancy = generate_bev_occupancy_benchmark_from_perception_benchmark(
        perception_benchmark_path=perception_output,
        db_path=db_path,
        output_path=bev_output,
        grid_spec=grid_spec,
    )

    proxy_studies: Dict[str, Any] = {}
    if bool(suite.get("run_proxy_studies", True)):
        proxy_outputs = dict(suite.get("proxy_outputs") or {})
        proxy_studies["perception"] = run_proxy_perception_study(
            benchmark_path=perception_output,
            output_dir=Path(str(proxy_outputs.get("perception") or output_root / "perception_proxy_study")),
        )
        proxy_studies["world_model"] = run_proxy_world_model_study(
            benchmark_path=world_model_output,
            output_dir=Path(str(proxy_outputs.get("world_model") or output_root / "world_model_proxy_study")),
        )
        proxy_studies["bev_occupancy"] = run_proxy_bev_occupancy_study(
            benchmark_path=bev_output,
            output_dir=Path(str(proxy_outputs.get("bev_occupancy") or output_root / "bev_occupancy_proxy_study")),
        )

    return {
        "output_root": str(output_root),
        "db": str(db_path),
        "case_library": str(case_library_path),
        "max_cases": max_cases,
        "outputs": {
            "scenario_mining": str(scenario_output),
            "perception": str(perception_output),
            "world_model": str(world_model_output),
            "bev_occupancy": str(bev_output),
        },
        "scenario_mining": scenario,
        "perception": perception,
        "world_model": world_model,
        "bev_occupancy": bev_occupancy,
        "proxy_studies": proxy_studies,
    }


def _run_catalog_export_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(dict(config.get("experiment") or {}).get("output") or "outputs/catalog_export"))
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = write_benchmark_catalog(output_dir / "benchmark_catalog.json", build_default_benchmark_catalog())
    inventory = write_dataset_backend_inventory(inspect_dataset_backends(), output_dir / "dataset_backends.json")
    return {
        "output_dir": str(output_dir),
        "benchmark_catalog": {
            "path": str(output_dir / "benchmark_catalog.json"),
            "layer_count": len(catalog.get("layers", {})),
        },
        "dataset_backends": {
            "path": str(output_dir / "dataset_backends.json"),
            "backend_count": len(inventory.get("backends", {})),
        },
    }


def _run_result_registry_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    result_registry = dict(config.get("result_registry") or {})
    output_dir = Path(
        str(
            result_registry.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or "outputs/full_benchmark_suite_v1/result_registry"
        )
    )
    sources = [Path(str(item)) for item in list(result_registry.get("sources") or [])]
    payload = write_result_registry(
        output_dir=output_dir,
        sources=sources or None,
        metadata={"experiment": dict(config.get("experiment") or {})},
    )
    return {
        "output_dir": str(output_dir),
        "overview": payload.get("overview", {}),
        "artifact_manifest": payload.get("artifact_manifest", {}),
    }


def _run_failure_mining_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    failure_mining = dict(config.get("failure_mining") or {})
    output_dir = Path(
        str(
            failure_mining.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or DEFAULT_FAILURE_MINING_OUTPUT
        )
    )
    inputs = [Path(str(item)) for item in list(failure_mining.get("inputs") or [])]
    if not inputs:
        inputs = [
            Path("outputs/trainval_bev_occupancy_proxy_study_v1"),
            Path("outputs/trainval_world_model_proxy_study_v2"),
            Path("outputs/contextvae_world_model_study_v3"),
            Path("outputs/contextvae_expanded_study_v1"),
            Path("outputs/nuplan_replay_sweep_v1"),
            Path("outputs/nuplan_closed_loop_sweep_v1"),
        ]
    return mine_model_failures(
        inputs=inputs,
        output_dir=output_dir,
        max_queries=int(failure_mining.get("max_queries") or 24),
        min_count=int(failure_mining.get("min_count") or 1),
    )


def _run_learned_retriever_training_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    stage = dict(config.get("learned_retriever_training") or {})
    output = Path(str(stage.get("output") or DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT))
    training = LearnedRetrieverConfig(**{"epochs": 20, **dict(stage.get("training") or {})})
    report = train_weakly_supervised_scene_retriever(
        db_path=Path(str(stage.get("db") or "artifacts/index/v1.0-trainval.sqlite")),
        output_dir=output, config=training,
        max_groups_per_family=int(stage.get("max_groups_per_family", 1000)),
    )
    return {"output_dir": str(output), "report": report}


def _run_contextvae_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    from nusc_scene_agent.contextvae_integration import run_contextvae_world_model_study

    stage = dict(config.get("contextvae_world_model_study") or {})
    return run_contextvae_world_model_study(
        benchmark_path=Path(str(stage.get("benchmark") or "benchmarks/trainval_world_model_slices_v2.json")),
        dataroot=Path(str(stage.get("dataroot") or "data/sets/nuscenes")),
        output_dir=Path(str(stage.get("output") or "outputs/contextvae_world_model_study_v3")),
        version=str(stage.get("version", "v1.0-trainval")),
        repo_dir=Path(str(stage.get("repo") or "external/ContextVAE")),
        checkpoint_path=Path(str(stage.get("checkpoint") or "external/ContextVAE/models/nuscenes_res18")),
        device=str(stage.get("device", "")), batch_size=int(stage.get("batch_size", 8)),
        mode_count=int(stage.get("mode_count", 5)), clustering_samples=int(stage.get("clustering_samples", 2000)),
        map_scale=int(stage.get("map_scale", 1)), seed=int(stage.get("seed", 1)),
    )


def _run_forecast_validation_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    from nusc_scene_agent.forecast_validation import audit_forecast_generalization
    from nusc_scene_agent.risk_case_expansion import expand_risk_cases

    stage = dict(config.get("forecast_validation_study") or {})
    output = Path(str(stage.get("output") or "outputs/forecast_validation_study_v1"))
    expand_risk_cases(Path(str(stage.get("db") or "artifacts/index/v1.0-trainval.sqlite")), output / "cases",
                      scene_split="val", per_family=int(stage.get("per_family", 30)),
                      candidate_limit=int(stage.get("candidate_limit", 180)),
                      reuse_validated=bool(stage.get("reuse_validated", False)))
    result = _run_contextvae_experiment({"contextvae_world_model_study": {
        **dict(stage.get("forecast") or {}), "benchmark": str(output / "cases/forecast.json"),
        "output": str(output / "contextvae"),
    }})
    development = stage.get("development_benchmarks") or []
    if development:
        result["generalization_audit"] = audit_forecast_generalization(
            output / "contextvae", [Path(str(path)) for path in development], output / "scene_disjoint_audit")
    return result


def _run_failure_aware_reranking_experiment(config: Mapping[str, Any]) -> Dict[str, Any]:
    reranking = dict(config.get("failure_aware_reranking") or {})
    output_dir = Path(
        str(
            reranking.get("output")
            or dict(config.get("experiment") or {}).get("output")
            or DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT
        )
    )
    payload = run_failure_aware_reranking_eval(
        query_config=Path(str(reranking.get("query_config") or DEFAULT_FAILURE_UPDATE_QUERIES)),
        db_path=Path(str(reranking.get("db") or "artifacts/index/v1.0-trainval.sqlite")),
        output_dir=output_dir,
        learned_checkpoint=Path(str(reranking.get("learned_checkpoint") or DEFAULT_LEARNED_RETRIEVER_CHECKPOINT)),
        candidate_pool=int(reranking.get("candidate_pool") or 48),
        top_k=int(reranking.get("top_k") or 3),
        max_queries=int(reranking.get("max_queries") or 24),
    )
    return {
        "output_dir": str(output_dir),
        "overview": payload.get("overview", {}),
        "artifact_manifest": payload.get("artifact_manifest", {}),
    }


def _experiment_result_path(config: Mapping[str, Any], config_path: Path) -> Path:
    experiment = dict(config.get("experiment") or {})
    output = experiment.get("result_path")
    if output:
        return Path(str(output))
    output_dir = Path(str(experiment.get("output") or "outputs/experiments"))
    return output_dir / f"{config_path.stem}_result.json"


def _stage_llm_config(stage: Mapping[str, Any], *, query_mode: str, rerank_mode: str) -> Optional[LLMConfig]:
    requires_llm = query_mode in {"llm", "hybrid"} or rerank_mode == "llm"
    if not requires_llm:
        return None

    llm = dict(stage.get("llm") or {})
    base_url = str(
        llm.get("base_url")
        or os.getenv("NUSC_SCENE_AGENT_OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA_BASE_URL
    ).strip()
    model = str(
        llm.get("model")
        or os.getenv("NUSC_SCENE_AGENT_OLLAMA_MODEL")
        or DEFAULT_OLLAMA_MODEL
    ).strip()
    digest = str(
        llm.get("digest")
        or os.getenv("NUSC_SCENE_AGENT_OLLAMA_DIGEST")
        or ""
    ).strip()
    timeout_s = float(llm.get("timeout_s") or os.getenv("NUSC_SCENE_AGENT_OLLAMA_TIMEOUT_S") or DEFAULT_TIMEOUT_S)
    require_digest = bool(
        llm.get("require_digest")
        or str(os.getenv("NUSC_SCENE_AGENT_OLLAMA_REQUIRE_DIGEST") or "").strip().lower()
        in {"1", "true", "yes"}
    )
    if not base_url or not model:
        raise ValueError("Ollama base URL and model are required for LLM-backed experiment stages.")
    return LLMConfig(
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        digest=digest,
        require_digest=require_digest,
    )


def _stage_enabled(suite: Mapping[str, Any], name: str, *, default: bool) -> bool:
    stages = dict(suite.get("stages") or {})
    if name not in stages:
        return default
    return bool(stages.get(name))


def _merged_stage_config(config: Mapping[str, Any], suite: Mapping[str, Any], name: str) -> Dict[str, Any]:
    base = dict(config.get(name) or {})
    override = dict(suite.get(name) or {})
    return {**base, **override}


def _write_stage_result(experiment: Mapping[str, Any], config_path: str, result: Mapping[str, Any]) -> None:
    output_path = Path(str(experiment["result_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": EXPERIMENT_RESULT_SCHEMA,
        "experiment_id": str(experiment.get("id") or ""),
        "experiment_type": str(experiment.get("type") or ""),
        "config_path": config_path,
        "result": dict(result),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _full_suite_failure_inputs(stage_results: Mapping[str, Any]) -> list[str]:
    risk = dict(stage_results.get("risk_benchmark_suite") or {})
    proxies = dict(risk.get("proxy_studies") or {})
    inputs = [str(study["output_dir"]) for study in proxies.values() if study.get("output_dir")]
    contextvae = dict(stage_results.get("contextvae_world_model_study") or {})
    if contextvae.get("output_dir"):
        inputs.append(str(contextvae["output_dir"]))
    expanded = dict(stage_results.get("contextvae_expanded_study") or {})
    if expanded.get("output_dir"):
        inputs.append(str(expanded["output_dir"]))
    replay = dict(stage_results.get("nuplan_replay_sweep") or {})
    closed_loop = dict(stage_results.get("nuplan_closed_loop_sweep") or {})
    if replay.get("output_dir"):
        inputs.append(str(replay["output_dir"]))
    if closed_loop.get("output_dir"):
        inputs.append(str(closed_loop["output_dir"]))
    if not inputs:
        raise ValueError("Failure mining requires current evaluation stages or explicit input paths.")
    return inputs


def _full_suite_result_sources(stage_results: Mapping[str, Any], stage_result_paths: Mapping[str, str]) -> list[str]:
    sources = []
    if stage_result_paths.get("risk_benchmark_suite"):
        sources.append(stage_result_paths["risk_benchmark_suite"])
    replay = dict(stage_results.get("nuplan_replay_sweep") or {})
    if replay.get("output_dir"):
        sources.append(str(Path(str(replay["output_dir"])) / "nuplan_replay_sweep_summary.json"))
    closed_loop = dict(stage_results.get("nuplan_closed_loop_sweep") or {})
    if closed_loop.get("output_dir"):
        sources.append(str(Path(str(closed_loop["output_dir"])) / "nuplan_closed_loop_sweep_summary.json"))
    failure = dict(stage_results.get("failure_mining") or {})
    if failure.get("report_json"):
        sources.append(str(failure["report_json"]))
    artifact_names = {
        "bench2drive_study": "study_summary.json",
        "carla_fixed_evaluation": "fixed_evaluation.json",
        "contextvae_world_model_study": "contextvae_study_manifest.json",
        "contextvae_expanded_study": "contextvae_study_manifest.json",
        "forecast_validation_study": "contextvae_study_manifest.json",
        "learned_retriever_training": "training_report.json",
        "risk_case_expansion": "expansion_report.json",
        "failure_aware_reranking": "failure_aware_reranking_eval.json",
    }
    for name, filename in artifact_names.items():
        result = dict(stage_results.get(name) or {})
        if result.get("output_dir"):
            sources.append(str(Path(result["output_dir"]) / filename))
        elif stage_result_paths.get(name):
            sources.append(stage_result_paths[name])
    return sources


def _render_full_suite_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Full Benchmark Suite",
        "",
        f"- Output: `{summary.get('output_dir', '')}`",
        "",
        "| Stage | Output |",
        "| --- | --- |",
    ]
    for name, result in dict(summary.get("stages") or {}).items():
        if isinstance(result, Mapping):
            lines.append(f"| `{name}` | `{result.get('output_dir', '')}` |")
    lines.append("")
    return "\n".join(lines)
