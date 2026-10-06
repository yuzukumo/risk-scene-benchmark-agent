from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from nusc_scene_agent.benchmark_comparison import (
    build_benchmark_comparison,
    default_benchmark_profiles,
    write_benchmark_comparison,
)
from nusc_scene_agent.ablation_study import default_ablation_profiles, write_ablation_manifest
from nusc_scene_agent.case_library_enrichment import enrich_case_library
from nusc_scene_agent.counterfactual_benchmark import generate_counterfactual_benchmark_from_case_library
from nusc_scene_agent.benchmark_exports import write_benchmark_exports
from nusc_scene_agent.benchmark_metrics import build_benchmark_metrics, write_benchmark_metrics
from nusc_scene_agent.bev_occupancy_benchmark import (
    BEV_OCCUPANCY_PROXY_PROFILES,
    DEFAULT_BEV_OCCUPANCY_BENCHMARK,
    adapt_perception_predictions_to_bev_occupancy,
    compare_bev_occupancy_evaluations,
    evaluate_bev_occupancy_predictions,
    generate_bev_occupancy_benchmark_from_perception_benchmark,
    generate_proxy_bev_occupancy_predictions,
    run_proxy_bev_occupancy_study,
)
from nusc_scene_agent.bench2drive_e2e import (
    DEFAULT_BENCH2DRIVE_CACHE_ROOT,
    DEFAULT_BENCH2DRIVE_MANIFEST,
    DEFAULT_BENCH2DRIVE_OUTPUT,
    DEFAULT_BENCH2DRIVE_ROOT,
    DEFAULT_BENCH2DRIVE_TENSOR_CACHE_ROOT,
    DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST,
    build_bench2drive_vision_manifest,
    build_bench2drive_vision_tensor_cache,
    calibrate_trajectory_selection,
    compare_vision_e2e_prediction_sets,
    diagnose_vision_e2e_predictions,
    evaluate_vision_e2e_planner,
    inspect_bench2drive_dataset,
    resplit_bench2drive_vision_manifest,
    train_vision_e2e_planner,
)
from nusc_scene_agent.bench2drive_closed_loop import (
    DEFAULT_BENCH2DRIVE_CLOSED_LOOP_OUTPUT,
    ClosedLoopControlConfig,
    compare_bench2drive_closed_loop_reports,
    run_bench2drive_vision_closed_loop,
)
from nusc_scene_agent.carla_closed_loop import (
    DEFAULT_CARLA_ROOT,
    build_carla_launch_command,
    format_carla_launch_command,
    inspect_carla_runtime,
    run_carla_connection_smoke,
)
from nusc_scene_agent.carla_vision_closed_loop import (
    DEFAULT_CARLA_VISION_OUTPUT,
    run_carla_vision_closed_loop,
)
from nusc_scene_agent.carla_semantic_demo_mining import (
    DEFAULT_CARLA_SEMANTIC_DEMO_OUTPUT,
    DEFAULT_CARLA_SEMANTIC_DEMO_TRIALS_OUTPUT,
    mine_carla_semantic_demos,
)
from nusc_scene_agent.carla_video_audit import audit_carla_vision_rollouts
from nusc_scene_agent.case_library import build_case_library, write_case_library
from nusc_scene_agent.benchmark_schema import load_benchmark_config
from nusc_scene_agent.benchmark_catalog import build_default_benchmark_catalog, write_benchmark_catalog
from nusc_scene_agent.contextvae_integration import (
    DEFAULT_CONTEXTVAE_CHECKPOINT,
    DEFAULT_CONTEXTVAE_OUTPUT,
    DEFAULT_CONTEXTVAE_REPO,
    run_contextvae_world_model_study,
)
from nusc_scene_agent.data_utils import DEFAULT_DATAROOT, PREPARE_PROFILES, discover_archive_inventory, prepare_data
from nusc_scene_agent.dataset_backends import inspect_dataset_backends, write_dataset_backend_inventory
from nusc_scene_agent.experiment_config import run_experiment_config
from nusc_scene_agent.failure_mining import mine_model_failures
from nusc_scene_agent.failure_aware_reranking import (
    DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT,
    DEFAULT_FAILURE_UPDATE_QUERIES,
    run_failure_aware_reranking_eval,
)
from nusc_scene_agent.gallery import build_benchmark_gallery, build_comparison_browser
from nusc_scene_agent.human_audit import evaluate_human_audit_set, generate_human_audit_set
from nusc_scene_agent.indexing import build_index, migrate_index_schema
from nusc_scene_agent.langgraph_agent import run_langgraph_query_pipeline
from nusc_scene_agent.learned_retrieval import (
    DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT,
    DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
    DEFAULT_LEARNED_RETRIEVER_OUTPUT,
    LearnedRetrieverConfig,
    run_learned_retrieval_report,
    train_learned_scene_retriever,
    train_weakly_supervised_scene_retriever,
)
from nusc_scene_agent.llm_client import DEFAULT_TIMEOUT_S, LLMConfig, inspect_ollama_model
from nusc_scene_agent.multimodal_retrieval import run_multimodal_retrieval_report
from nusc_scene_agent.nuplan_closed_loop import (
    DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT,
    DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES,
    run_nuplan_closed_loop_study,
)
from nusc_scene_agent.nuplan_closed_loop_sweep import DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT
from nusc_scene_agent.nuplan_replay import (
    DEFAULT_NUPLAN_REPLAY_BENCHMARK,
    DEFAULT_NUPLAN_SPLIT,
    NUPLAN_REPLAY_PROFILES,
    compare_nuplan_replay_evaluations,
    evaluate_nuplan_rollouts,
    generate_nuplan_proxy_rollouts,
    generate_nuplan_replay_benchmark,
    inspect_nuplan_dataset,
    render_nuplan_replay_case_studies,
    run_nuplan_replay_study,
)
from nusc_scene_agent.nuplan_sweep import DEFAULT_NUPLAN_REPLAY_SWEEP_OUTPUT
from nusc_scene_agent.perception_benchmark import (
    PROXY_PERCEPTION_PROFILES,
    PREDICTION_COVERAGE_MODES,
    adapt_filter_and_evaluate_nuscenes_predictions,
    adapt_and_evaluate_nuscenes_predictions,
    adapt_nuscenes_predictions,
    compare_perception_evaluations,
    evaluate_perception_predictions,
    filter_perception_benchmark_by_predictions,
    generate_perception_benchmark_from_scenario_config,
    generate_proxy_perception_predictions,
    run_proxy_perception_study,
)
from nusc_scene_agent.pipeline import run_query_pipeline
from nusc_scene_agent.retrieval import RETRIEVAL_SCORE_PROFILES, RetrievalScoreConfig
from nusc_scene_agent.research_agent import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_RESEARCH_AGENT_OUTPUT,
    run_research_agent,
)
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.unified_schema import load_unified_case_source, write_unified_case_collection
from nusc_scene_agent.world_model_benchmark import (
    NUSCENES_FORECAST_MODE_SELECTIONS,
    NUSCENES_FORECAST_BASELINE_PROFILES,
    WORLD_MODEL_PROXY_PROFILES,
    adapt_and_evaluate_nuscenes_forecast_predictions,
    adapt_nuscenes_forecast_predictions,
    adapt_world_model_predictions,
    compare_world_model_evaluations,
    evaluate_world_model_predictions,
    export_world_model_replay,
    generate_proxy_world_model_predictions,
    generate_nuscenes_forecast_baselines,
    generate_world_model_benchmark_from_perception_benchmark,
    run_nuscenes_forecast_baselines,
    run_proxy_world_model_study,
)
from nusc_scene_agent.validation import HEURISTIC_THRESHOLDS, VALIDATION_SCORE_PROFILES, ValidationConfig
from nusc_scene_agent.world_model_case_studies import render_world_model_case_studies


DEFAULT_DB = Path("artifacts/index/v1.0-trainval.sqlite")
DEFAULT_BENCHMARK = Path("benchmarks/smoke_queries.yaml")
DEFAULT_TRAINVAL_BENCHMARK = Path("benchmarks/trainval_suite_v1.yaml")
DEFAULT_COMPARE_BENCHMARK = Path("benchmarks/trainval_language_stress_v1.yaml")
DEFAULT_SCENARIO_MINING_BENCHMARK = Path("benchmarks/trainval_scenario_mining_v1.yaml")
DEFAULT_PERCEPTION_BENCHMARK = Path("benchmarks/trainval_perception_slices_v1.json")
DEFAULT_WORLD_MODEL_BENCHMARK = Path("benchmarks/trainval_world_model_slices_v2.json")


def _add_llm_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-base-url", default="")
    parser.add_argument("--ollama-model", default="")
    parser.add_argument(
        "--ollama-digest",
        default="",
        help="Expected Ollama model digest. A mismatch aborts LLM requests.",
    )
    parser.add_argument(
        "--ollama-require-digest",
        action="store_true",
        help="Require an expected Ollama digest before issuing LLM requests.",
    )
    parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT_S)


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--query-mode", choices=["rule", "llm", "hybrid"], default="rule")
    parser.add_argument("--rerank-mode", choices=["none", "llm", "multimodal", "learned"], default="none")
    parser.add_argument("--learned-reranker-checkpoint", default="")
    _add_llm_connection_args(parser)


def _add_retrieval_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-score-profile",
        choices=sorted(RETRIEVAL_SCORE_PROFILES),
        default="default",
        help="Retrieval scoring profile used before validation.",
    )


def _resolve_llm_config(
    args: argparse.Namespace,
    require: bool = False,
    default_base_url: str = "",
    default_model: str = "",
) -> Optional[LLMConfig]:
    env_config = LLMConfig.from_env()
    base_url = str(
        getattr(args, "ollama_base_url", "")
        or (env_config.base_url if env_config else "")
        or default_base_url
    ).strip()
    model = str(getattr(args, "ollama_model", "") or (env_config.model if env_config else "") or default_model).strip()
    digest = str(
        getattr(args, "ollama_digest", "")
        or (env_config.digest if env_config else "")
    ).strip()
    timeout_s = float(getattr(args, "ollama_timeout", DEFAULT_TIMEOUT_S))
    require_digest = bool(
        getattr(args, "ollama_require_digest", False)
        or (env_config.require_digest if env_config else False)
    )
    if not (base_url and model):
        if require:
            raise ValueError("Ollama configuration requires --ollama-base-url and --ollama-model.")
        return None
    return LLMConfig(
        base_url=base_url,
        model=model,
        timeout_s=timeout_s,
        digest=digest,
        require_digest=require_digest,
    )


def _optional_path(value: str) -> Optional[Path]:
    value = str(value or "").strip()
    return Path(value) if value else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nuScenes scene mining and benchmark generation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-archives", help="Inspect local nuScenes archives and readiness.")
    inspect_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")

    prepare_parser = subparsers.add_parser("prepare-data", help="Unpack nuScenes archives into a dataroot.")
    prepare_parser.add_argument("--archive", action="append", default=[], help="Archive path, repeatable.")
    prepare_parser.add_argument("--workspace", default=".", help="Workspace root that contains the archives.")
    prepare_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT), help="Extraction target.")
    prepare_parser.add_argument("--profile", choices=PREPARE_PROFILES, default="trainval-full")

    build_parser = subparsers.add_parser("build-index", help="Build a SQLite index from nuScenes annotations.")
    build_parser.add_argument("--version", default="v1.0-trainval")
    build_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    build_parser.add_argument("--db", default=str(DEFAULT_DB))
    build_parser.add_argument("--scene-limit", type=int, default=0)

    migrate_index_parser = subparsers.add_parser(
        "migrate-index",
        help="Validate and stamp a structurally compatible pre-versioned SQLite index.",
    )
    migrate_index_parser.add_argument("--db", default=str(DEFAULT_DB))

    query_parser = subparsers.add_parser("query", help="Run one natural-language risk query.")
    query_parser.add_argument("text", help="Natural-language risk description.")
    query_parser.add_argument("--db", default=str(DEFAULT_DB))
    query_parser.add_argument("--output", default="outputs/query")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_retrieval_score_args(query_parser)
    _add_llm_args(query_parser)

    multimodal_retrieval_parser = subparsers.add_parser(
        "run-multimodal-retrieval",
        help="Score one risk query with the structural multimodal retrieval model.",
    )
    multimodal_retrieval_parser.add_argument("text", help="Natural-language risk description.")
    multimodal_retrieval_parser.add_argument("--db", default=str(DEFAULT_DB))
    multimodal_retrieval_parser.add_argument("--output", default="outputs/multimodal_retrieval")
    multimodal_retrieval_parser.add_argument("--top-k", type=int, default=20)
    multimodal_retrieval_parser.add_argument("--candidate-pool", type=int, default=64)
    multimodal_retrieval_parser.add_argument("--query-mode", choices=["rule", "llm"], default="rule")
    _add_llm_connection_args(multimodal_retrieval_parser)

    train_learned_parser = subparsers.add_parser(
        "train-learned-retriever",
        help="Train a compact query-scene pairwise reranker from reference-aware scenario-mining anchors.",
    )
    train_learned_parser.add_argument("--benchmark", default=str(DEFAULT_SCENARIO_MINING_BENCHMARK))
    train_learned_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    train_learned_parser.add_argument("--output", default=str(DEFAULT_LEARNED_RETRIEVER_OUTPUT))
    train_learned_parser.add_argument("--epochs", type=int, default=80)
    train_learned_parser.add_argument("--negatives-per-query", type=int, default=12)
    train_learned_parser.add_argument("--candidate-pool", type=int, default=64)
    train_learned_parser.add_argument("--text-hash-dim", type=int, default=256)
    train_learned_parser.add_argument("--hidden-dim", type=int, default=128)
    train_learned_parser.add_argument("--embedding-dim", type=int, default=64)
    train_learned_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_learned_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_learned_parser.add_argument("--validation-fraction", type=float, default=0.25)
    train_learned_parser.add_argument("--seed", type=int, default=7)
    train_learned_parser.add_argument("--device", default="")

    train_large_learned_parser = subparsers.add_parser(
        "train-large-learned-retriever",
        help="Train a learned reranker from weakly labeled trainval scenario families.",
    )
    train_large_learned_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    train_large_learned_parser.add_argument("--output", default=str(DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT))
    train_large_learned_parser.add_argument("--max-groups-per-family", type=int, default=1000)
    train_large_learned_parser.add_argument("--epochs", type=int, default=20)
    train_large_learned_parser.add_argument("--negatives-per-query", type=int, default=12)
    train_large_learned_parser.add_argument("--text-hash-dim", type=int, default=256)
    train_large_learned_parser.add_argument("--hidden-dim", type=int, default=128)
    train_large_learned_parser.add_argument("--embedding-dim", type=int, default=64)
    train_large_learned_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_large_learned_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_large_learned_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_large_learned_parser.add_argument("--seed", type=int, default=7)
    train_large_learned_parser.add_argument("--device", default="")

    learned_retrieval_parser = subparsers.add_parser(
        "run-learned-retrieval",
        help="Score one risk query with a trained query-scene reranker checkpoint.",
    )
    learned_retrieval_parser.add_argument("text", help="Natural-language risk description.")
    learned_retrieval_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    learned_retrieval_parser.add_argument("--checkpoint", default=str(DEFAULT_LEARNED_RETRIEVER_CHECKPOINT))
    learned_retrieval_parser.add_argument("--output", default="outputs/learned_retrieval_v1")
    learned_retrieval_parser.add_argument("--top-k", type=int, default=20)
    learned_retrieval_parser.add_argument("--candidate-pool", type=int, default=64)

    failure_aware_reranking_parser = subparsers.add_parser(
        "run-failure-aware-reranking",
        help="Evaluate learned reranking on failure-mined scene queries.",
    )
    failure_aware_reranking_parser.add_argument("--query-config", default=str(DEFAULT_FAILURE_UPDATE_QUERIES))
    failure_aware_reranking_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    failure_aware_reranking_parser.add_argument("--checkpoint", default=str(DEFAULT_LEARNED_RETRIEVER_CHECKPOINT))
    failure_aware_reranking_parser.add_argument("--output", default=str(DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT))
    failure_aware_reranking_parser.add_argument("--candidate-pool", type=int, default=48)
    failure_aware_reranking_parser.add_argument("--top-k", type=int, default=3)
    failure_aware_reranking_parser.add_argument("--max-queries", type=int, default=24)

    langgraph_query_parser = subparsers.add_parser(
        "langgraph-query",
        help="Run one natural-language risk query through a LangGraph orchestration layer.",
    )
    langgraph_query_parser.add_argument("text", help="Natural-language risk description.")
    langgraph_query_parser.add_argument("--db", default=str(DEFAULT_DB))
    langgraph_query_parser.add_argument("--output", default="outputs/langgraph_query")
    langgraph_query_parser.add_argument("--top-k", type=int, default=5)
    langgraph_query_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_args(langgraph_query_parser)

    research_agent_parser = subparsers.add_parser(
        "run-research-agent",
        help="Run an Ollama-backed research agent over local benchmark artifacts.",
    )
    research_agent_parser.add_argument("--output", default=str(DEFAULT_RESEARCH_AGENT_OUTPUT))
    research_agent_parser.add_argument("--artifact", action="append", default=[], help="Benchmark artifact path, repeatable.")
    research_agent_parser.add_argument(
        "--focus",
        default="Assess benchmark gaps and propose evidence-based next actions.",
        help="Focused instruction for the research agent.",
    )
    research_agent_parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    research_agent_parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    research_agent_parser.add_argument("--ollama-digest", default="")
    research_agent_parser.add_argument("--ollama-require-digest", action="store_true")
    research_agent_parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT_S)

    inspect_ollama_parser = subparsers.add_parser(
        "inspect-ollama-model",
        help="Export local Ollama model metadata for experiment reproducibility.",
    )
    inspect_ollama_parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    inspect_ollama_parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    inspect_ollama_parser.add_argument("--ollama-digest", default="")
    inspect_ollama_parser.add_argument("--ollama-require-digest", action="store_true")
    inspect_ollama_parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_TIMEOUT_S)
    inspect_ollama_parser.add_argument("--output", default="outputs/ollama_model_metadata.json")

    failure_mining_parser = subparsers.add_parser(
        "run-failure-mining",
        help="Mine model-evaluation failures and generate benchmark update queries.",
    )
    failure_mining_parser.add_argument(
        "--input",
        action="append",
        default=[],
        required=True,
        help="Evaluation file or directory. Repeatable.",
    )
    failure_mining_parser.add_argument("--output", default="outputs/model_in_the_loop_failure_mining_v1")
    failure_mining_parser.add_argument("--max-queries", type=int, default=24)
    failure_mining_parser.add_argument("--min-count", type=int, default=1)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a YAML benchmark query suite.")
    benchmark_parser.add_argument("--config", default=str(DEFAULT_BENCHMARK))
    benchmark_parser.add_argument("--db", default=str(DEFAULT_DB))
    benchmark_parser.add_argument("--output", default="outputs/benchmark")
    benchmark_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_retrieval_score_args(benchmark_parser)
    _add_llm_args(benchmark_parser)

    score_sweep_parser = subparsers.add_parser(
        "benchmark-score-sweep",
        help="Run the same benchmark with multiple retrieval scoring profiles.",
    )
    score_sweep_parser.add_argument("--config", default=str(DEFAULT_TRAINVAL_BENCHMARK))
    score_sweep_parser.add_argument("--db", default=str(DEFAULT_DB))
    score_sweep_parser.add_argument("--output", default="outputs/retrieval_score_profile_sweep_v1")
    score_sweep_parser.add_argument("--candidate-pool", type=int, default=12)
    score_sweep_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=sorted(RETRIEVAL_SCORE_PROFILES),
        help="Retrieval scoring profile. Repeatable. Defaults to all profiles.",
    )
    _add_llm_args(score_sweep_parser)

    threshold_sweep_parser = subparsers.add_parser(
        "benchmark-threshold-sweep",
        help="Run the same benchmark with scaled behavior-detection thresholds.",
    )
    threshold_sweep_parser.add_argument("--config", default=str(DEFAULT_TRAINVAL_BENCHMARK))
    threshold_sweep_parser.add_argument("--db", default=str(DEFAULT_DB))
    threshold_sweep_parser.add_argument("--output", default="outputs/validation_threshold_sweep_v1")
    threshold_sweep_parser.add_argument("--candidate-pool", type=int, default=12)
    threshold_sweep_parser.add_argument(
        "--scale",
        action="append",
        type=float,
        default=[],
        help="Positive multiplier for metric-valued behavior thresholds. Repeatable; defaults to 0.85, 1.0, 1.15.",
    )
    _add_llm_args(threshold_sweep_parser)

    validation_score_sweep_parser = subparsers.add_parser(
        "benchmark-validation-score-sweep",
        help="Run the same benchmark with multiple validation-quality weighting profiles.",
    )
    validation_score_sweep_parser.add_argument("--config", default=str(DEFAULT_TRAINVAL_BENCHMARK))
    validation_score_sweep_parser.add_argument("--db", default=str(DEFAULT_DB))
    validation_score_sweep_parser.add_argument("--output", default="outputs/validation_score_profile_sweep_v1")
    validation_score_sweep_parser.add_argument("--candidate-pool", type=int, default=12)
    validation_score_sweep_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=sorted(VALIDATION_SCORE_PROFILES),
        help="Validation-quality weighting profile. Repeatable. Defaults to all profiles.",
    )
    _add_llm_args(validation_score_sweep_parser)

    generate_counterfactual_parser = subparsers.add_parser(
        "generate-counterfactual-benchmark",
        help="Generate a contrastive counterfactual benchmark from a case library.",
    )
    generate_counterfactual_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Path to case_library.json used as benchmark anchors.",
    )
    generate_counterfactual_parser.add_argument(
        "--output",
        default="benchmarks/trainval_counterfactual_reference_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_counterfactual_parser.add_argument("--max-cases", type=int, default=6)

    generate_scenario_parser = subparsers.add_parser(
        "generate-scenario-mining-benchmark",
        help="Generate a reference-aware scenario mining benchmark from a case library.",
    )
    generate_scenario_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Path to case_library.json used as scenario anchors.",
    )
    generate_scenario_parser.add_argument(
        "--output",
        default="benchmarks/trainval_scenario_mining_v1.yaml",
        help="Output YAML path for the generated benchmark.",
    )
    generate_scenario_parser.add_argument("--max-cases", type=int, default=8)

    generate_perception_parser = subparsers.add_parser(
        "generate-perception-benchmark",
        help="Generate a scenario-conditioned perception benchmark from a scenario mining benchmark.",
    )
    generate_perception_parser.add_argument("--config", default=str(DEFAULT_SCENARIO_MINING_BENCHMARK))
    generate_perception_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_perception_parser.add_argument("--output", default=str(DEFAULT_PERCEPTION_BENCHMARK))

    generate_world_model_parser = subparsers.add_parser(
        "generate-world-model-benchmark",
        help="Generate a scenario-conditioned world-model benchmark from a perception benchmark.",
    )
    generate_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    generate_world_model_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_world_model_parser.add_argument("--output", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))

    proxy_predictions_parser = subparsers.add_parser(
        "generate-proxy-perception-predictions",
        help="Generate proxy prediction tracks for a perception benchmark.",
    )
    proxy_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    proxy_predictions_parser.add_argument("--output", default="outputs/proxy_perception_predictions.json")
    proxy_predictions_parser.add_argument("--profile", choices=PROXY_PERCEPTION_PROFILES, default="oracle_tracking")

    adapt_predictions_parser = subparsers.add_parser(
        "adapt-nuscenes-predictions",
        help="Adapt official nuScenes detection or tracking outputs to the local perception-evaluation schema.",
    )
    adapt_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    adapt_predictions_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    adapt_predictions_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    adapt_predictions_parser.add_argument("--output", default="outputs/adapted_nuscenes_predictions.json")
    adapt_predictions_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")

    filter_perception_parser = subparsers.add_parser(
        "filter-perception-benchmark",
        help="Filter a perception benchmark to cases covered by a prediction file.",
    )
    filter_perception_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    filter_perception_parser.add_argument("--predictions", required=True)
    filter_perception_parser.add_argument("--output", default="outputs/filtered_perception_benchmark.json")
    filter_perception_parser.add_argument("--coverage-mode", choices=PREDICTION_COVERAGE_MODES, default="full_window")

    evaluate_perception_parser = subparsers.add_parser(
        "evaluate-perception-predictions",
        help="Evaluate prediction tracks on the scenario-conditioned perception benchmark.",
    )
    evaluate_perception_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_perception_parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON with per-sample actor tracks.",
    )
    evaluate_perception_parser.add_argument("--output", default="outputs/perception_evaluation")
    evaluate_perception_parser.add_argument("--profile-name", default="")
    evaluate_perception_parser.add_argument("--match-distance-m", type=float, default=2.0)

    proxy_study_parser = subparsers.add_parser(
        "run-proxy-perception-study",
        help="Run a proxy perception comparison on the scenario-conditioned benchmark.",
    )
    proxy_study_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    proxy_study_parser.add_argument("--output", default="outputs/trainval_perception_proxy_study_v1")

    proxy_world_model_predictions_parser = subparsers.add_parser(
        "generate-proxy-world-model-predictions",
        help="Generate proxy world-model rollouts for a world-model benchmark.",
    )
    proxy_world_model_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    proxy_world_model_predictions_parser.add_argument("--output", default="outputs/proxy_world_model_predictions.json")
    proxy_world_model_predictions_parser.add_argument("--profile", choices=WORLD_MODEL_PROXY_PROFILES, default="oracle_rollout")

    adapt_world_model_predictions_parser = subparsers.add_parser(
        "adapt-world-model-predictions",
        help="Adapt compact external world-model rollouts to the local evaluation schema.",
    )
    adapt_world_model_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    adapt_world_model_predictions_parser.add_argument("--input", required=True)
    adapt_world_model_predictions_parser.add_argument("--output", default="outputs/adapted_world_model_predictions.json")
    adapt_world_model_predictions_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    adapt_nuscenes_forecast_parser = subparsers.add_parser(
        "adapt-nuscenes-forecast-predictions",
        help="Adapt nuScenes prediction-challenge style trajectory forecasts to the local world-model schema.",
    )
    adapt_nuscenes_forecast_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    adapt_nuscenes_forecast_parser.add_argument("--input", required=True)
    adapt_nuscenes_forecast_parser.add_argument("--output", default="outputs/adapted_nuscenes_forecasts.json")
    adapt_nuscenes_forecast_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")
    adapt_nuscenes_forecast_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    evaluate_nuscenes_forecast_parser = subparsers.add_parser(
        "evaluate-nuscenes-forecast-predictions",
        help="Adapt nuScenes prediction-challenge forecasts and evaluate them on the world-model benchmark.",
    )
    evaluate_nuscenes_forecast_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    evaluate_nuscenes_forecast_parser.add_argument("--input", required=True)
    evaluate_nuscenes_forecast_parser.add_argument("--output", default="outputs/nuscenes_forecast_eval")
    evaluate_nuscenes_forecast_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")
    evaluate_nuscenes_forecast_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_forecast_parser.add_argument("--no-rasterize-trajectory", action="store_true")

    generate_nuscenes_forecast_baselines_parser = subparsers.add_parser(
        "generate-nuscenes-forecast-baselines",
        help="Generate official nuScenes physics forecast baselines directly on the world-model benchmark anchors.",
    )
    generate_nuscenes_forecast_baselines_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    generate_nuscenes_forecast_baselines_parser.add_argument("--dataroot", default="data/sets/nuscenes")
    generate_nuscenes_forecast_baselines_parser.add_argument("--version", default="v1.0-trainval")
    generate_nuscenes_forecast_baselines_parser.add_argument("--output", default="outputs/nuscenes_forecast_baselines")

    run_nuscenes_forecast_baselines_parser = subparsers.add_parser(
        "run-nuscenes-forecast-baselines",
        help="Generate, evaluate, and compare official nuScenes physics forecast baselines on the world-model benchmark.",
    )
    run_nuscenes_forecast_baselines_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    run_nuscenes_forecast_baselines_parser.add_argument("--dataroot", default="data/sets/nuscenes")
    run_nuscenes_forecast_baselines_parser.add_argument("--version", default="v1.0-trainval")
    run_nuscenes_forecast_baselines_parser.add_argument("--output", default="outputs/nuscenes_forecast_baselines_eval")
    run_nuscenes_forecast_baselines_parser.add_argument("--mode-selection", choices=NUSCENES_FORECAST_MODE_SELECTIONS, default="top_probability")

    run_contextvae_world_model_parser = subparsers.add_parser(
        "run-contextvae-world-model-study",
        help="Run the ContextVAE multimodal forecasting baseline on the forecast-compatible world-model benchmark subset.",
    )
    run_contextvae_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    run_contextvae_world_model_parser.add_argument("--dataroot", default=str(DEFAULT_DATAROOT))
    run_contextvae_world_model_parser.add_argument("--version", default="v1.0-trainval")
    run_contextvae_world_model_parser.add_argument("--output", default=str(DEFAULT_CONTEXTVAE_OUTPUT))
    run_contextvae_world_model_parser.add_argument("--repo", default=str(DEFAULT_CONTEXTVAE_REPO))
    run_contextvae_world_model_parser.add_argument("--checkpoint", default=str(DEFAULT_CONTEXTVAE_CHECKPOINT))
    run_contextvae_world_model_parser.add_argument("--device", default="")
    run_contextvae_world_model_parser.add_argument("--batch-size", type=int, default=8)
    run_contextvae_world_model_parser.add_argument("--mode-count", type=int, default=5)
    run_contextvae_world_model_parser.add_argument("--clustering-samples", type=int, default=2000)
    run_contextvae_world_model_parser.add_argument("--map-scale", type=int, default=1)
    run_contextvae_world_model_parser.add_argument("--seed", type=int, default=1)

    evaluate_world_model_parser = subparsers.add_parser(
        "evaluate-world-model-predictions",
        help="Evaluate future trajectory and occupancy predictions on the scenario-conditioned world-model benchmark.",
    )
    evaluate_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    evaluate_world_model_parser.add_argument("--predictions", required=True)
    evaluate_world_model_parser.add_argument("--output", default="outputs/world_model_evaluation")
    evaluate_world_model_parser.add_argument("--profile-name", default="")

    proxy_world_model_study_parser = subparsers.add_parser(
        "run-proxy-world-model-study",
        help="Run a proxy comparison on the scenario-conditioned world-model benchmark.",
    )
    proxy_world_model_study_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    proxy_world_model_study_parser.add_argument("--output", default="outputs/trainval_world_model_proxy_study_v1")

    compare_world_model_parser = subparsers.add_parser(
        "compare-world-model-evaluations",
        help="Compare multiple world-model evaluation output directories.",
    )
    compare_world_model_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains world_model_metrics.json. Repeatable.",
    )
    compare_world_model_parser.add_argument("--output", default="outputs/world_model_comparison")

    export_world_model_parser = subparsers.add_parser(
        "export-world-model-replay",
        help="Export world-model benchmark cases as replay-ready JSONL or MCAP artifacts.",
    )
    export_world_model_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    export_world_model_parser.add_argument("--output", default="outputs/world_model_replay")
    export_world_model_parser.add_argument("--format", choices=["jsonl", "mcap"], default="jsonl")

    render_world_model_case_studies_parser = subparsers.add_parser(
        "render-world-model-case-studies",
        help="Render qualitative case studies from world-model evaluation outputs.",
    )
    render_world_model_case_studies_parser.add_argument("--benchmark", default=str(DEFAULT_WORLD_MODEL_BENCHMARK))
    render_world_model_case_studies_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains world_model_metrics.json and adapted_predictions.json. Repeatable.",
    )
    render_world_model_case_studies_parser.add_argument("--output", default="outputs/world_model_case_studies")
    render_world_model_case_studies_parser.add_argument("--max-cases", type=int, default=4)

    inspect_nuplan_parser = subparsers.add_parser(
        "inspect-nuplan",
        help="Inspect local nuPlan dataset layout and split readiness.",
    )
    inspect_nuplan_parser.add_argument("--dataset-root", default="data/nuplan/dataset")

    generate_nuplan_replay_parser = subparsers.add_parser(
        "generate-nuplan-replay-benchmark",
        help="Generate a compact nuPlan replay-regression benchmark from SQLite scenario tags.",
    )
    generate_nuplan_replay_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    generate_nuplan_replay_parser.add_argument("--output", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    generate_nuplan_replay_parser.add_argument("--max-dbs", type=int, default=4)
    generate_nuplan_replay_parser.add_argument("--max-cases", type=int, default=16)
    generate_nuplan_replay_parser.add_argument("--max-cases-per-db", type=int, default=4)
    generate_nuplan_replay_parser.add_argument("--history-s", type=float, default=2.0)
    generate_nuplan_replay_parser.add_argument("--future-s", type=float, default=4.0)
    generate_nuplan_replay_parser.add_argument("--frame-hz", type=float, default=2.0)
    generate_nuplan_replay_parser.add_argument("--scenario-tag", action="append", default=[])

    generate_nuplan_rollout_parser = subparsers.add_parser(
        "generate-nuplan-proxy-rollouts",
        help="Generate simple ego rollout baselines for a nuPlan replay benchmark.",
    )
    generate_nuplan_rollout_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    generate_nuplan_rollout_parser.add_argument("--output", default="outputs/nuplan_proxy_rollouts.json")
    generate_nuplan_rollout_parser.add_argument(
        "--profile",
        choices=NUPLAN_REPLAY_PROFILES,
        default="constant_velocity",
    )

    evaluate_nuplan_rollout_parser = subparsers.add_parser(
        "evaluate-nuplan-rollouts",
        help="Evaluate ego rollout predictions on a nuPlan replay benchmark.",
    )
    evaluate_nuplan_rollout_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    evaluate_nuplan_rollout_parser.add_argument("--predictions", required=True)
    evaluate_nuplan_rollout_parser.add_argument("--output", default="outputs/nuplan_replay_evaluation")
    evaluate_nuplan_rollout_parser.add_argument("--profile-name", default="")
    evaluate_nuplan_rollout_parser.add_argument("--collision-distance-m", type=float, default=2.0)

    run_nuplan_replay_parser = subparsers.add_parser(
        "run-nuplan-replay-study",
        help="Generate a nuPlan replay benchmark, proxy rollouts, and evaluation summaries.",
    )
    run_nuplan_replay_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    run_nuplan_replay_parser.add_argument("--output", default="outputs/nuplan_replay_study_v1")
    run_nuplan_replay_parser.add_argument("--max-dbs", type=int, default=4)
    run_nuplan_replay_parser.add_argument("--max-cases", type=int, default=16)
    run_nuplan_replay_parser.add_argument("--max-cases-per-db", type=int, default=4)
    run_nuplan_replay_parser.add_argument("--history-s", type=float, default=2.0)
    run_nuplan_replay_parser.add_argument("--future-s", type=float, default=4.0)
    run_nuplan_replay_parser.add_argument("--frame-hz", type=float, default=2.0)
    run_nuplan_replay_parser.add_argument("--min-anchor-gap-s", type=float, default=4.0)
    run_nuplan_replay_parser.add_argument("--scenario-tag", action="append", default=[])
    run_nuplan_replay_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=NUPLAN_REPLAY_PROFILES,
        help="Proxy rollout profile. Repeatable. Defaults to all profiles.",
    )

    run_nuplan_sweep_parser = subparsers.add_parser(
        "run-nuplan-replay-sweep",
        help="Run a cross-split nuPlan replay-regression sweep.",
    )
    run_nuplan_sweep_parser.add_argument("--config", default="configs/nuplan_replay_sweep_medium.yaml")

    run_nuplan_closed_loop_sweep_parser = subparsers.add_parser(
        "run-nuplan-closed-loop-sweep",
        help="Run a cross-split nuPlan closed-loop replay sweep.",
    )
    run_nuplan_closed_loop_sweep_parser.add_argument("--config", default="configs/nuplan_closed_loop_sweep_medium.yaml")

    run_nuplan_closed_loop_parser = subparsers.add_parser(
        "run-nuplan-closed-loop-study",
        help="Run closed-loop replay simulation on compact nuPlan cases.",
    )
    run_nuplan_closed_loop_parser.add_argument("--split-dir", default=str(DEFAULT_NUPLAN_SPLIT))
    run_nuplan_closed_loop_parser.add_argument("--output", default=str(DEFAULT_NUPLAN_CLOSED_LOOP_OUTPUT))
    run_nuplan_closed_loop_parser.add_argument("--max-dbs", type=int, default=64)
    run_nuplan_closed_loop_parser.add_argument("--max-cases", type=int, default=16)
    run_nuplan_closed_loop_parser.add_argument("--max-cases-per-db", type=int, default=4)
    run_nuplan_closed_loop_parser.add_argument("--history-s", type=float, default=2.0)
    run_nuplan_closed_loop_parser.add_argument("--future-s", type=float, default=4.0)
    run_nuplan_closed_loop_parser.add_argument("--frame-hz", type=float, default=2.0)
    run_nuplan_closed_loop_parser.add_argument("--min-anchor-gap-s", type=float, default=4.0)
    run_nuplan_closed_loop_parser.add_argument("--scenario-tag", action="append", default=[])
    run_nuplan_closed_loop_parser.add_argument(
        "--profile",
        action="append",
        default=[],
        choices=DEFAULT_NUPLAN_CLOSED_LOOP_PROFILES,
        help="Closed-loop planner profile. Repeatable. Defaults to all profiles.",
    )

    inspect_bench2drive_parser = subparsers.add_parser(
        "inspect-bench2drive",
        help="Inspect the local Bench2Drive training archive layout.",
    )
    inspect_bench2drive_parser.add_argument("--dataset-root", default=str(DEFAULT_BENCH2DRIVE_ROOT))
    inspect_bench2drive_parser.add_argument("--sample-archives", type=int, default=3)

    build_bench2drive_manifest_parser = subparsers.add_parser(
        "build-bench2drive-vision-manifest",
        help="Build a sampled Bench2Drive manifest for vision planner training.",
    )
    build_bench2drive_manifest_parser.add_argument("--dataset-root", default=str(DEFAULT_BENCH2DRIVE_ROOT))
    build_bench2drive_manifest_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_MANIFEST))
    build_bench2drive_manifest_parser.add_argument("--max-archives", type=int, default=0)
    build_bench2drive_manifest_parser.add_argument("--frame-stride", type=int, default=5)
    build_bench2drive_manifest_parser.add_argument("--future-steps", type=int, default=5)
    build_bench2drive_manifest_parser.add_argument("--future-frame-stride", type=int, default=5)
    build_bench2drive_manifest_parser.add_argument("--train-fraction", type=float, default=0.8)
    build_bench2drive_manifest_parser.add_argument("--val-fraction", type=float, default=0.1)
    build_bench2drive_manifest_parser.add_argument("--seed", type=int, default=7)
    build_bench2drive_manifest_parser.add_argument("--cache-root", default=str(DEFAULT_BENCH2DRIVE_CACHE_ROOT))
    build_bench2drive_manifest_parser.add_argument("--no-cache", action="store_true")
    build_bench2drive_manifest_parser.add_argument("--verbose", action="store_true")

    split_bench2drive_manifest_parser = subparsers.add_parser(
        "split-bench2drive-vision-manifest",
        help="Create a scenario-stratified archive-disjoint train/val/test manifest.",
    )
    split_bench2drive_manifest_parser.add_argument("--manifest", default=str(DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST))
    split_bench2drive_manifest_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST))
    split_bench2drive_manifest_parser.add_argument("--train-fraction", type=float, default=0.8)
    split_bench2drive_manifest_parser.add_argument("--val-fraction", type=float, default=0.1)
    split_bench2drive_manifest_parser.add_argument("--seed", type=int, default=7)

    build_bench2drive_tensor_cache_parser = subparsers.add_parser(
        "build-bench2drive-vision-tensor-cache",
        help="Build a predecoded Bench2Drive multi-camera tensor cache for faster vision training.",
    )
    build_bench2drive_tensor_cache_parser.add_argument("--manifest", default=str(DEFAULT_BENCH2DRIVE_MANIFEST))
    build_bench2drive_tensor_cache_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST))
    build_bench2drive_tensor_cache_parser.add_argument("--cache-dir", default=str(DEFAULT_BENCH2DRIVE_TENSOR_CACHE_ROOT))
    build_bench2drive_tensor_cache_parser.add_argument("--image-size", type=int, default=160)
    build_bench2drive_tensor_cache_parser.add_argument("--max-rows", type=int, default=0)
    build_bench2drive_tensor_cache_parser.add_argument("--num-workers", type=int, default=0)
    build_bench2drive_tensor_cache_parser.add_argument("--chunk-rows", type=int, default=512)
    build_bench2drive_tensor_cache_parser.add_argument("--verbose", action="store_true")

    train_bench2drive_parser = subparsers.add_parser(
        "train-bench2drive-vision-planner",
        help="Train the Bench2Drive vision planner from the sampled manifest.",
    )
    train_bench2drive_parser.add_argument("--manifest", default=str(DEFAULT_BENCH2DRIVE_MANIFEST))
    train_bench2drive_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_OUTPUT))
    train_bench2drive_parser.add_argument("--epochs", type=int, default=3)
    train_bench2drive_parser.add_argument("--batch-size", type=int, default=32)
    train_bench2drive_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_bench2drive_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_bench2drive_parser.add_argument("--image-size", type=int, default=160)
    train_bench2drive_parser.add_argument("--model-size", choices=["tiny", "base", "large", "research"], default="base")
    train_bench2drive_parser.add_argument(
        "--architecture",
        choices=["conv_mlp", "trajectory_transformer"],
        default="conv_mlp",
    )
    train_bench2drive_parser.add_argument("--camera-pooling", choices=["mean", "attention", "transformer"], default="mean")
    train_bench2drive_parser.add_argument("--dropout", type=float, default=0.0)
    train_bench2drive_parser.add_argument("--trajectory-modes", type=int, default=1)
    train_bench2drive_parser.add_argument(
        "--trajectory-selection",
        choices=["argmax", "expected", "topk_expected", "top2_expected"],
        default="argmax",
    )
    train_bench2drive_parser.add_argument("--trajectory-top-k", type=int, default=2)
    train_bench2drive_parser.add_argument("--trajectory-temperature", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--waypoint-loss-weight", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--selected-waypoint-loss-weight", type=float, default=0.5)
    train_bench2drive_parser.add_argument("--input-mode", choices=["vision_route", "route_only"], default="vision_route")
    train_bench2drive_parser.add_argument("--spatial-pool-size", type=int, choices=[1, 2, 4], default=2)
    train_bench2drive_parser.add_argument("--displacement-loss-weight", type=float, default=0.0)
    train_bench2drive_parser.add_argument("--endpoint-loss-weight", type=float, default=0.0)
    train_bench2drive_parser.add_argument("--path-length-loss-weight", type=float, default=0.0)
    train_bench2drive_parser.add_argument("--control-loss-weight", type=float, default=0.25)
    train_bench2drive_parser.add_argument("--brake-loss-weight", type=float, default=0.1)
    train_bench2drive_parser.add_argument("--brake-positive-weight", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--risk-sample-weight", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--lateral-loss-weight", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--turn-sample-weight", type=float, default=1.0)
    train_bench2drive_parser.add_argument("--turn-lateral-threshold-m", type=float, default=2.0)
    train_bench2drive_parser.add_argument("--mode-classification-weight", type=float, default=0.05)
    train_bench2drive_parser.add_argument(
        "--selection-metric",
        choices=["ade_m", "fde_m", "brake_f1", "risk_aware", "lateral_aware"],
        default="ade_m",
    )
    train_bench2drive_parser.add_argument("--max-train-samples", type=int, default=0)
    train_bench2drive_parser.add_argument("--max-val-samples", type=int, default=0)
    train_bench2drive_parser.add_argument("--num-workers", type=int, default=4)
    train_bench2drive_parser.add_argument("--prefetch-factor", type=int, default=4)
    train_bench2drive_parser.add_argument("--device", default="")
    train_bench2drive_parser.add_argument("--no-data-parallel", action="store_true")
    train_bench2drive_parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    train_bench2drive_parser.add_argument("--disable-tf32", action="store_true")
    train_bench2drive_parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    train_bench2drive_parser.add_argument("--nonfinite-check-interval", type=int, default=0)
    train_bench2drive_parser.add_argument("--seed", type=int, default=7)
    train_bench2drive_parser.add_argument("--verbose", action="store_true")

    eval_bench2drive_parser = subparsers.add_parser(
        "evaluate-bench2drive-vision-planner",
        help="Evaluate a Bench2Drive vision planner checkpoint.",
    )
    eval_bench2drive_parser.add_argument("--manifest", default=str(DEFAULT_BENCH2DRIVE_MANIFEST))
    eval_bench2drive_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"),
    )
    eval_bench2drive_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "eval"))
    eval_bench2drive_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    eval_bench2drive_parser.add_argument("--batch-size", type=int, default=32)
    eval_bench2drive_parser.add_argument("--image-size", type=int, default=160)
    eval_bench2drive_parser.add_argument("--max-samples", type=int, default=0)
    eval_bench2drive_parser.add_argument("--num-workers", type=int, default=4)
    eval_bench2drive_parser.add_argument("--prefetch-factor", type=int, default=4)

    compare_bench2drive_predictions_parser = subparsers.add_parser(
        "compare-bench2drive-vision-predictions",
        help="Compare two Bench2Drive vision prediction files on paired clips.",
    )
    compare_bench2drive_predictions_parser.add_argument("--baseline", required=True)
    compare_bench2drive_predictions_parser.add_argument("--candidate", required=True)
    compare_bench2drive_predictions_parser.add_argument(
        "--output",
        default="outputs/bench2drive_prediction_comparison",
    )
    compare_bench2drive_predictions_parser.add_argument("--baseline-label", default="baseline")
    compare_bench2drive_predictions_parser.add_argument("--candidate-label", default="candidate")
    compare_bench2drive_predictions_parser.add_argument("--seed", type=int, default=7)
    compare_bench2drive_predictions_parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    compare_bench2drive_predictions_parser.add_argument("--allow-case-intersection", action="store_true")
    eval_bench2drive_parser.add_argument("--device", default="")

    calibrate_bench2drive_parser = subparsers.add_parser(
        "calibrate-bench2drive-trajectory-selection",
        help="Calibrate multimodal trajectory selection on the validation split.",
    )
    calibrate_bench2drive_parser.add_argument(
        "--predictions",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "val" / "predictions.jsonl"),
    )
    calibrate_bench2drive_parser.add_argument(
        "--evaluation-report",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "val" / "evaluation_report.json"),
    )
    calibrate_bench2drive_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"),
    )
    calibrate_bench2drive_parser.add_argument(
        "--output-checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_calibrated.pt"),
    )
    calibrate_bench2drive_parser.add_argument(
        "--output-report",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "trajectory_selection_calibration_report.json"),
    )
    calibrate_bench2drive_parser.add_argument("--temperature", type=float, action="append", default=[])
    calibrate_bench2drive_parser.add_argument("--top-k", type=int, default=2)

    diagnose_bench2drive_parser = subparsers.add_parser(
        "diagnose-bench2drive-vision-planner",
        help="Analyze Bench2Drive vision planner predictions for closed-loop transfer readiness.",
    )
    diagnose_bench2drive_parser.add_argument(
        "--predictions",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "eval" / "predictions.jsonl"),
    )
    diagnose_bench2drive_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "diagnostics"))
    diagnose_bench2drive_parser.add_argument("--evaluation-report", default="")
    diagnose_bench2drive_parser.add_argument("--brake-threshold", type=float, default=0.5)

    bench2drive_closed_loop_parser = subparsers.add_parser(
        "run-bench2drive-vision-closed-loop",
        help="Run model-in-the-loop closed-loop evaluation for the Bench2Drive vision planner.",
    )
    bench2drive_closed_loop_parser.add_argument("--manifest", default=str(DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST))
    bench2drive_closed_loop_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"),
    )
    bench2drive_closed_loop_parser.add_argument("--output", default=str(DEFAULT_BENCH2DRIVE_CLOSED_LOOP_OUTPUT))
    bench2drive_closed_loop_parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    bench2drive_closed_loop_parser.add_argument("--max-cases", type=int, default=64)
    bench2drive_closed_loop_parser.add_argument("--max-frames-per-clip", type=int, default=20)
    bench2drive_closed_loop_parser.add_argument("--image-size", type=int, default=160)
    bench2drive_closed_loop_parser.add_argument("--device", default="")
    bench2drive_closed_loop_parser.add_argument("--video-fps", type=int, default=6)
    bench2drive_closed_loop_parser.add_argument(
        "--render-case-media",
        action="store_true",
        help="Render per-case GIF/MP4 media in addition to JSON, CSV, and static figures.",
    )
    bench2drive_closed_loop_parser.add_argument(
        "--case-selection",
        choices=["balanced", "qualitative", "stress"],
        default="balanced",
    )
    bench2drive_closed_loop_parser.add_argument("--dt-s", type=float, default=0.5)
    bench2drive_closed_loop_parser.add_argument("--horizon-s", type=float, default=10.0)
    bench2drive_closed_loop_parser.add_argument("--target-speed-mps", type=float, default=5.5)
    bench2drive_closed_loop_parser.add_argument("--brake-threshold", type=float, default=0.85)
    bench2drive_closed_loop_parser.add_argument("--lookahead-m", type=float, default=9.0)
    bench2drive_closed_loop_parser.add_argument("--speed-kp", type=float, default=0.45)

    compare_bench2drive_closed_loop_parser = subparsers.add_parser(
        "compare-bench2drive-closed-loop",
        help="Compare two Bench2Drive closed-loop reports on paired case IDs.",
    )
    compare_bench2drive_closed_loop_parser.add_argument("--baseline", required=True)
    compare_bench2drive_closed_loop_parser.add_argument("--candidate", required=True)
    compare_bench2drive_closed_loop_parser.add_argument(
        "--output",
        default="outputs/bench2drive_closed_loop_comparison",
    )
    compare_bench2drive_closed_loop_parser.add_argument("--baseline-label", default="baseline")
    compare_bench2drive_closed_loop_parser.add_argument("--candidate-label", default="candidate")
    compare_bench2drive_closed_loop_parser.add_argument("--seed", type=int, default=7)
    compare_bench2drive_closed_loop_parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    compare_bench2drive_closed_loop_parser.add_argument("--allow-case-intersection", action="store_true")

    inspect_carla_parser = subparsers.add_parser(
        "inspect-carla",
        help="Inspect the local CARLA runtime and PythonAPI readiness.",
    )
    inspect_carla_parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT))

    carla_launch_parser = subparsers.add_parser(
        "print-carla-launch-command",
        help="Print a headless CARLA server launch command.",
    )
    carla_launch_parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT))
    carla_launch_parser.add_argument("--port", type=int, default=2000)
    carla_launch_parser.add_argument("--quality-level", default="Low")
    carla_launch_parser.add_argument("--fps", type=int, default=20)
    carla_launch_parser.add_argument("--cuda-visible-devices", default="")
    carla_launch_parser.add_argument("--null-rhi", action="store_true")
    carla_launch_parser.add_argument("--no-render-offscreen", action="store_true")

    carla_smoke_parser = subparsers.add_parser(
        "carla-connection-smoke",
        help="Connect to a running CARLA server and report map/spawn-point metadata.",
    )
    carla_smoke_parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT))
    carla_smoke_parser.add_argument("--host", default="127.0.0.1")
    carla_smoke_parser.add_argument("--port", type=int, default=2000)
    carla_smoke_parser.add_argument("--timeout-s", type=float, default=10.0)
    carla_smoke_parser.add_argument("--town", default="")
    carla_smoke_parser.add_argument("--load-town", action="store_true")

    carla_vision_parser = subparsers.add_parser(
        "run-carla-vision-closed-loop",
        help="Run a CARLA camera-stream closed-loop rollout with the Bench2Drive vision planner.",
    )
    carla_vision_parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT))
    carla_vision_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"),
    )
    carla_vision_parser.add_argument("--output", default=str(DEFAULT_CARLA_VISION_OUTPUT))
    carla_vision_parser.add_argument("--host", default="127.0.0.1")
    carla_vision_parser.add_argument("--port", type=int, default=2000)
    carla_vision_parser.add_argument("--town", default="")
    carla_vision_parser.add_argument("--spawn-index", type=int, default=0)
    carla_vision_parser.add_argument("--destination-index", type=int, default=-1)
    carla_vision_parser.add_argument("--route-sampling-resolution-m", type=float, default=2.0)
    carla_vision_parser.add_argument("--route-min-length-m", type=float, default=40.0)
    carla_vision_parser.add_argument("--route-max-length-m", type=float, default=220.0)
    carla_vision_parser.add_argument("--route-preferred-length-m", type=float, default=0.0)
    carla_vision_parser.add_argument("--fps", type=int, default=10)
    carla_vision_parser.add_argument("--horizon-s", type=float, default=30.0)
    carla_vision_parser.add_argument("--image-size", type=int, default=160)
    carla_vision_parser.add_argument("--camera-width", type=int, default=320)
    carla_vision_parser.add_argument("--camera-height", type=int, default=180)
    carla_vision_parser.add_argument("--video-width", type=int, default=0)
    carla_vision_parser.add_argument("--video-height", type=int, default=0)
    carla_vision_parser.add_argument("--carla-quality-level", default="Epic")
    carla_vision_parser.add_argument("--scenario-type", default="free_drive")
    carla_vision_parser.add_argument("--scenario-name", default="")
    carla_vision_parser.add_argument("--target-speed-mps", type=float, default=7.0)
    carla_vision_parser.add_argument("--brake-threshold", type=float, default=0.75)
    carla_vision_parser.add_argument("--no-scenario-safety-override", action="store_true")
    carla_vision_parser.add_argument("--no-lane-departure-guard", action="store_true")
    carla_vision_parser.add_argument("--condition-ego-route-traffic-lights", action="store_true")
    carla_vision_parser.add_argument("--device", default="")
    carla_vision_parser.add_argument("--auto-launch", action="store_true")
    carla_vision_parser.add_argument("--cuda-visible-devices", default="")
    carla_vision_parser.add_argument("--graphics-adapter", type=int, default=-1)
    carla_vision_parser.add_argument("--traffic-manager-port", type=int, default=8000)
    carla_vision_parser.add_argument("--seed", type=int, default=7)
    carla_vision_parser.add_argument("--launch-timeout-s", type=float, default=90.0)
    carla_vision_parser.add_argument("--rpc-timeout-s", type=float, default=30.0)
    carla_vision_parser.add_argument("--keep-server", action="store_true")
    carla_vision_parser.add_argument("--video-fps", type=int, default=10)
    carla_vision_parser.add_argument("--video-encoder", default="hevc_nvenc")
    carla_vision_parser.add_argument("--video-nvenc-preset", default="p4")
    carla_vision_parser.add_argument("--video-quality", type=int, default=23)

    carla_semantic_demo_parser = subparsers.add_parser(
        "mine-carla-semantic-demos",
        help="Search CARLA routes and keep only semantic-audit-passing vision closed-loop demos.",
    )
    carla_semantic_demo_parser.add_argument("--carla-root", default=str(DEFAULT_CARLA_ROOT))
    carla_semantic_demo_parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt"),
    )
    carla_semantic_demo_parser.add_argument("--output", default=str(DEFAULT_CARLA_SEMANTIC_DEMO_OUTPUT))
    carla_semantic_demo_parser.add_argument("--trials-output", default=str(DEFAULT_CARLA_SEMANTIC_DEMO_TRIALS_OUTPUT))
    carla_semantic_demo_parser.add_argument("--host", default="127.0.0.1")
    carla_semantic_demo_parser.add_argument("--port-start", type=int, default=2040)
    carla_semantic_demo_parser.add_argument("--town", default="Town10HD_Opt")
    carla_semantic_demo_parser.add_argument("--fps", type=int, default=10)
    carla_semantic_demo_parser.add_argument("--horizon-s", type=float, default=18.0)
    carla_semantic_demo_parser.add_argument("--image-size", type=int, default=160)
    carla_semantic_demo_parser.add_argument("--camera-width", type=int, default=640)
    carla_semantic_demo_parser.add_argument("--camera-height", type=int, default=360)
    carla_semantic_demo_parser.add_argument("--video-width", type=int, default=1920)
    carla_semantic_demo_parser.add_argument("--video-height", type=int, default=1080)
    carla_semantic_demo_parser.add_argument("--carla-quality-level", default="Epic")
    carla_semantic_demo_parser.add_argument("--brake-threshold", type=float, default=0.82)
    carla_semantic_demo_parser.add_argument("--device", default="cuda")
    carla_semantic_demo_parser.add_argument("--no-auto-launch", action="store_true")
    carla_semantic_demo_parser.add_argument("--cuda-visible-devices", default="")
    carla_semantic_demo_parser.add_argument("--traffic-manager-port-start", type=int, default=8040)
    carla_semantic_demo_parser.add_argument("--launch-timeout-s", type=float, default=180.0)
    carla_semantic_demo_parser.add_argument("--rpc-timeout-s", type=float, default=90.0)
    carla_semantic_demo_parser.add_argument("--keep-server", action="store_true")
    carla_semantic_demo_parser.add_argument("--no-reuse-carla-server", action="store_true")
    carla_semantic_demo_parser.add_argument("--video-fps", type=int, default=10)
    carla_semantic_demo_parser.add_argument("--video-encoder", default="hevc_nvenc")
    carla_semantic_demo_parser.add_argument("--video-nvenc-preset", default="p4")
    carla_semantic_demo_parser.add_argument("--video-quality", type=int, default=23)
    carla_semantic_demo_parser.add_argument(
        "--allow-non-hevc",
        action="store_true",
        help="Keep a rollout when the output video is not confirmed as HEVC.",
    )
    carla_semantic_demo_parser.add_argument("--no-scenario-safety-override", action="store_true")
    carla_semantic_demo_parser.add_argument("--enable-lane-departure-guard", action="store_true")
    carla_semantic_demo_parser.add_argument("--no-condition-ego-route-traffic-lights", action="store_true")
    carla_semantic_demo_parser.add_argument("--max-attempts-per-target", type=int, default=4)

    carla_audit_parser = subparsers.add_parser(
        "audit-carla-vision-rollout",
        help="Audit generated CARLA vision rollout videos and control evidence.",
    )
    carla_audit_parser.add_argument(
        "--report",
        default="outputs/carla_semantic_demo_final/carla_semantic_demo_report.json",
    )
    carla_audit_parser.add_argument("--output", default="")
    carla_audit_parser.add_argument("--min-resolution-width", type=int, default=1920)
    carla_audit_parser.add_argument("--min-resolution-height", type=int, default=1080)
    carla_audit_parser.add_argument("--min-fps", type=float, default=8.0)
    carla_audit_parser.add_argument("--min-frames", type=int, default=80)
    carla_audit_parser.add_argument("--min-traffic-manager-vehicles", type=int, default=1)
    carla_audit_parser.add_argument("--max-scripted-vehicles", type=int, default=0)
    carla_audit_parser.add_argument("--max-collision-count", type=int, default=0)
    carla_audit_parser.add_argument("--min-route-completion", type=float, default=0.05)
    carla_audit_parser.add_argument("--max-mean-lateral-error-m", type=float, default=2.5)
    carla_audit_parser.add_argument("--max-lateral-error-m", type=float, default=6.0)
    carla_audit_parser.add_argument("--max-safety-override-ratio", type=float, default=0.35)
    carla_audit_parser.add_argument("--max-nearest-actor-distance-m", type=float, default=60.0)
    carla_audit_parser.add_argument("--nearby-actor-distance-m", type=float, default=30.0)
    carla_audit_parser.add_argument("--min-nearby-actor-ratio", type=float, default=0.30)
    carla_audit_parser.add_argument("--require-semantic-match", action="store_true")
    carla_audit_parser.add_argument("--allow-non-model-control", action="store_true")
    carla_audit_parser.add_argument("--require-hevc", action="store_true")

    compare_nuplan_replay_parser = subparsers.add_parser(
        "compare-nuplan-replay-evaluations",
        help="Compare multiple nuPlan replay evaluation output directories.",
    )
    compare_nuplan_replay_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains nuplan_replay_metrics.json. Repeatable.",
    )
    compare_nuplan_replay_parser.add_argument("--output", default="outputs/nuplan_replay_comparison")

    render_nuplan_case_studies_parser = subparsers.add_parser(
        "render-nuplan-replay-case-studies",
        help="Render qualitative case studies from nuPlan replay evaluation outputs.",
    )
    render_nuplan_case_studies_parser.add_argument("--benchmark", default=str(DEFAULT_NUPLAN_REPLAY_BENCHMARK))
    render_nuplan_case_studies_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains nuplan_replay_metrics.json. Repeatable.",
    )
    render_nuplan_case_studies_parser.add_argument("--output", default="outputs/nuplan_replay_case_studies")
    render_nuplan_case_studies_parser.add_argument("--max-cases", type=int, default=4)

    inspect_backends_parser = subparsers.add_parser(
        "inspect-dataset-backends",
        help="Inspect local nuScenes and nuPlan backend readiness.",
    )
    inspect_backends_parser.add_argument("--nuscenes-root", default="data/sets/nuscenes")
    inspect_backends_parser.add_argument("--nuplan-root", default="data/nuplan/dataset")
    inspect_backends_parser.add_argument("--index-root", default="artifacts/index")
    inspect_backends_parser.add_argument("--output", default="")

    catalog_parser = subparsers.add_parser(
        "export-benchmark-catalog",
        help="Export the static benchmark-layer catalog.",
    )
    catalog_parser.add_argument("--output", default="outputs/benchmark_catalog.json")

    full_suite_parser = subparsers.add_parser(
        "run-full-benchmark-suite",
        help="Run the full benchmark suite config.",
    )
    full_suite_parser.add_argument("--config", default="configs/full_benchmark_suite.yaml")
    full_suite_parser.add_argument("--reuse-case-library", action="store_true",
                                   help="Reuse and hash an existing validated case library; skip LLM case generation.")

    unified_cases_parser = subparsers.add_parser(
        "export-unified-cases",
        help="Convert a supported case source to the unified risk-case schema.",
    )
    unified_cases_parser.add_argument("--source", required=True)
    unified_cases_parser.add_argument(
        "--source-type",
        choices=["nuplan_replay_benchmark", "nuscenes_case_library"],
        required=True,
    )
    unified_cases_parser.add_argument("--output", default="outputs/unified_risk_cases.json")

    experiment_config_parser = subparsers.add_parser(
        "run-experiment-config",
        help="Run a structured experiment YAML config.",
    )
    experiment_config_parser.add_argument("--config", required=True)

    generate_bev_occupancy_parser = subparsers.add_parser(
        "generate-bev-occupancy-benchmark",
        help="Generate a sparse BEV occupancy benchmark from perception slices.",
    )
    generate_bev_occupancy_parser.add_argument("--perception-benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    generate_bev_occupancy_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    generate_bev_occupancy_parser.add_argument("--output", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))

    generate_bev_occupancy_predictions_parser = subparsers.add_parser(
        "generate-proxy-bev-occupancy-predictions",
        help="Generate proxy BEV occupancy predictions.",
    )
    generate_bev_occupancy_predictions_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    generate_bev_occupancy_predictions_parser.add_argument("--output", default="outputs/bev_occupancy_predictions.json")
    generate_bev_occupancy_predictions_parser.add_argument(
        "--profile",
        choices=BEV_OCCUPANCY_PROXY_PROFILES,
        default="risk_actor_only",
    )

    adapt_bev_occupancy_parser = subparsers.add_parser(
        "adapt-perception-to-bev-occupancy",
        help="Rasterize adapted perception predictions into sparse BEV occupancy predictions.",
    )
    adapt_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    adapt_bev_occupancy_parser.add_argument("--predictions", required=True)
    adapt_bev_occupancy_parser.add_argument("--output", default="outputs/adapted_bev_occupancy_predictions.json")
    adapt_bev_occupancy_parser.add_argument("--profile-name", default="")

    evaluate_bev_occupancy_parser = subparsers.add_parser(
        "evaluate-bev-occupancy",
        help="Evaluate sparse BEV occupancy predictions on risk slices.",
    )
    evaluate_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    evaluate_bev_occupancy_parser.add_argument("--predictions", required=True)
    evaluate_bev_occupancy_parser.add_argument("--output", default="outputs/bev_occupancy_evaluation")
    evaluate_bev_occupancy_parser.add_argument("--profile-name", default="")

    run_bev_occupancy_parser = subparsers.add_parser(
        "run-proxy-bev-occupancy-study",
        help="Run proxy BEV occupancy profiles on a sparse occupancy benchmark.",
    )
    run_bev_occupancy_parser.add_argument("--benchmark", default=str(DEFAULT_BEV_OCCUPANCY_BENCHMARK))
    run_bev_occupancy_parser.add_argument("--output", default="outputs/trainval_bev_occupancy_proxy_study_v1")

    compare_bev_occupancy_parser = subparsers.add_parser(
        "compare-bev-occupancy-evaluations",
        help="Compare multiple BEV occupancy evaluation directories.",
    )
    compare_bev_occupancy_parser.add_argument("--eval-dir", action="append", default=[], required=True)
    compare_bev_occupancy_parser.add_argument("--output", default="outputs/bev_occupancy_comparison")

    compare_perception_parser = subparsers.add_parser(
        "compare-perception-evaluations",
        help="Compare multiple perception evaluation output directories.",
    )
    compare_perception_parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        required=True,
        help="Evaluation directory that contains perception_metrics.json. Repeatable.",
    )
    compare_perception_parser.add_argument("--output", default="outputs/perception_comparison")

    evaluate_nuscenes_parser = subparsers.add_parser(
        "evaluate-nuscenes-predictions",
        help="Adapt official nuScenes detection or tracking outputs and evaluate them on the perception benchmark.",
    )
    evaluate_nuscenes_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_nuscenes_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    evaluate_nuscenes_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    evaluate_nuscenes_parser.add_argument("--output", default="outputs/nuscenes_perception_eval")
    evaluate_nuscenes_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")
    evaluate_nuscenes_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_parser.add_argument("--match-distance-m", type=float, default=2.0)

    evaluate_nuscenes_filtered_parser = subparsers.add_parser(
        "evaluate-nuscenes-predictions-covered",
        help="Adapt official nuScenes outputs, filter the benchmark to covered cases, and evaluate the aligned subset.",
    )
    evaluate_nuscenes_filtered_parser.add_argument("--benchmark", default=str(DEFAULT_PERCEPTION_BENCHMARK))
    evaluate_nuscenes_filtered_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    evaluate_nuscenes_filtered_parser.add_argument("--input", required=True, help="Path to the official nuScenes prediction JSON.")
    evaluate_nuscenes_filtered_parser.add_argument("--output", default="outputs/nuscenes_perception_eval_covered")
    evaluate_nuscenes_filtered_parser.add_argument("--task", choices=["tracking", "detection"], default="tracking")
    evaluate_nuscenes_filtered_parser.add_argument("--profile-name", default="")
    evaluate_nuscenes_filtered_parser.add_argument("--match-distance-m", type=float, default=2.0)
    evaluate_nuscenes_filtered_parser.add_argument("--coverage-mode", choices=PREDICTION_COVERAGE_MODES, default="full_window")

    enrich_case_library_parser = subparsers.add_parser(
        "enrich-case-library",
        help="Re-validate a case library to populate actor grounding and event localization fields.",
    )
    enrich_case_library_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library.json",
        help="Input case library JSON path.",
    )
    enrich_case_library_parser.add_argument(
        "--db",
        default="artifacts/index/v1.0-trainval.sqlite",
        help="SQLite index used for re-validation.",
    )
    enrich_case_library_parser.add_argument(
        "--output",
        default="outputs/trainval_case_library_v1/case_library_enriched.json",
        help="Output JSON path for the enriched case library.",
    )

    human_audit_parser = subparsers.add_parser(
        "generate-human-audit-set",
        help="Sample a fixed human-audit subset from a case library.",
    )
    human_audit_parser.add_argument(
        "--case-library",
        default="outputs/trainval_case_library_v1/case_library_enriched.json",
        help="Input case library JSON path.",
    )
    human_audit_parser.add_argument("--output", default="audits/human_audit_v1")
    human_audit_parser.add_argument("--sample-size", type=int, default=100)
    human_audit_parser.add_argument("--seed", type=int, default=7)
    human_audit_parser.add_argument("--max-per-behavior", type=int, default=0)

    evaluate_human_audit_parser = subparsers.add_parser(
        "evaluate-human-audit-set",
        help="Evaluate a completed human-audit JSONL or CSV file.",
    )
    evaluate_human_audit_parser.add_argument(
        "--annotations",
        default="audits/human_audit_v1/human_audit_items.jsonl",
        help="Completed human-audit annotation file.",
    )
    evaluate_human_audit_parser.add_argument("--output", default="audits/human_audit_v1/evaluation")

    compare_parser = subparsers.add_parser(
        "benchmark-compare",
        help="Run the benchmark across rule, llm, and hybrid profiles and export a comparison summary.",
    )
    compare_parser.add_argument("--config", default=str(DEFAULT_COMPARE_BENCHMARK))
    compare_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    compare_parser.add_argument("--output", default="outputs/trainval_profile_comparison_v1")
    compare_parser.add_argument("--candidate-pool", type=int, default=12)
    _add_llm_connection_args(compare_parser)

    ablate_parser = subparsers.add_parser(
        "benchmark-ablate",
        help="Run an ablation study over the benchmark and export comparison artifacts.",
    )
    ablate_parser.add_argument("--config", default="benchmarks/trainval_scenario_mining_v1.yaml")
    ablate_parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    ablate_parser.add_argument("--output", default="outputs/trainval_hybrid_ablation_v1")
    ablate_parser.add_argument("--candidate-pool", type=int, default=12)
    ablate_parser.add_argument("--base-query-mode", choices=["rule", "llm", "hybrid"], default="hybrid")
    ablate_parser.add_argument("--base-rerank-mode", choices=["none", "llm"], default="llm")
    ablate_parser.add_argument("--reuse-existing", action="store_true")
    _add_llm_connection_args(ablate_parser)

    gallery_parser = subparsers.add_parser(
        "build-gallery",
        help="Build a static query gallery for one benchmark output or a side-by-side browser for a comparison output.",
    )
    gallery_group = gallery_parser.add_mutually_exclusive_group(required=True)
    gallery_group.add_argument(
        "--benchmark-output",
        default="",
        help="Benchmark output directory that contains benchmark_summary.json.",
    )
    gallery_group.add_argument(
        "--comparison-output",
        default="",
        help="Comparison output directory that contains benchmark_profile_comparison.json.",
    )
    gallery_parser.add_argument("--title", default="")

    return parser


def _run_benchmark(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
    validation_config: Optional[ValidationConfig] = None,
    retrieval_score_config: Optional[RetrievalScoreConfig] = None,
) -> List[dict]:
    retrieval_score_config = retrieval_score_config or RetrievalScoreConfig()
    queries = load_benchmark_config(config_path)
    summaries: List[dict] = []
    benchmark_results: List[dict] = []
    for spec in queries:
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
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            validation_config=validation_config,
            retrieval_score_config=retrieval_score_config,
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

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    case_library_entries = build_case_library(benchmark_results)
    write_case_library(case_library_entries, output_dir)
    write_benchmark_metrics(build_benchmark_metrics(benchmark_results, case_library_entries), output_dir)
    write_benchmark_exports(benchmark_results, case_library_entries, output_dir)
    return summaries


def _rate_string(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return "{0}/{1} ({2:.1%})".format(count, total, count / total)


def _retrieval_score_sweep_row(profile_name: str, profile_output: Path) -> Dict[str, object]:
    metrics_path = profile_output / "benchmark_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing benchmark metrics for profile {0}: {1}".format(profile_name, metrics_path))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    overview = dict(metrics.get("overview") or {})
    reference = dict(metrics.get("reference_metrics") or {})
    query_count = int(overview.get("query_count") or 0)
    pass_at_1_count = int(
        overview.get("validation_acceptance_at_1_count", overview.get("pass_at_1_count")) or 0
    )
    pass_at_k_count = int(
        overview.get("validation_acceptance_at_k_count", overview.get("pass_at_k_count")) or 0
    )
    pass_at_1_rate = float(
        overview.get("validation_acceptance_at_1_rate", overview.get("pass_at_1_rate")) or 0.0
    )
    pass_at_k_rate = float(
        overview.get("validation_acceptance_at_k_rate", overview.get("pass_at_k_rate")) or 0.0
    )
    reference_query_count = int(reference.get("query_count") or 0)
    reference_objective_at_1_count = int(reference.get("objective_at_1_count") or 0)
    reference_objective_at_k_count = int(reference.get("objective_at_k_count") or 0)
    mean_quality = float(
        overview.get("mean_best_validation_quality_score", overview.get("mean_best_validation_score")) or 0.0
    )
    return {
        "profile": profile_name,
        "output_dir": str(profile_output),
        "query_count": query_count,
        "validation_acceptance_at_1_count": pass_at_1_count,
        "validation_acceptance_at_1_rate": pass_at_1_rate,
        "validation_acceptance_at_k_count": pass_at_k_count,
        "validation_acceptance_at_k_rate": pass_at_k_rate,
        "validation_acceptance_at_1": _rate_string(pass_at_1_count, query_count),
        "validation_acceptance_at_k": _rate_string(pass_at_k_count, query_count),
        # Compatibility aliases for sweep artifacts produced before the
        # acceptance/quality distinction was explicit.
        "anchor_pass_at_1_count": pass_at_1_count,
        "anchor_pass_at_1_rate": pass_at_1_rate,
        "anchor_pass_at_k_count": pass_at_k_count,
        "anchor_pass_at_k_rate": pass_at_k_rate,
        "mean_best_validation_quality_score": mean_quality,
        "mean_best_validation_score": mean_quality,
        "unique_case_count": int(overview.get("unique_case_count") or 0),
        "unique_passed_case_count": int(overview.get("unique_passed_case_count") or 0),
        "reference_query_count": reference_query_count,
        "reference_objective_at_1_count": reference_objective_at_1_count,
        "reference_objective_at_1_rate": float(reference.get("objective_at_1_rate") or 0.0),
        "reference_objective_at_k_count": reference_objective_at_k_count,
        "reference_objective_at_k_rate": float(reference.get("objective_at_k_rate") or 0.0),
        "anchor_pass_at_1": _rate_string(pass_at_1_count, query_count),
        "anchor_pass_at_k": _rate_string(pass_at_k_count, query_count),
        "reference_objective_at_1": _rate_string(reference_objective_at_1_count, reference_query_count),
        "reference_objective_at_k": _rate_string(reference_objective_at_k_count, reference_query_count),
    }


def _with_baseline_deltas(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    baseline = dict(rows[0])
    baseline.setdefault("validation_acceptance_at_1_rate", float(baseline.get("anchor_pass_at_1_rate") or 0.0))
    baseline.setdefault("validation_acceptance_at_k_rate", float(baseline.get("anchor_pass_at_k_rate") or 0.0))
    baseline.setdefault("mean_best_validation_quality_score", float(baseline.get("mean_best_validation_score") or 0.0))
    baseline_metrics = {
        key: float(baseline.get(key) or 0.0)
        for key in [
            "validation_acceptance_at_1_rate",
            "validation_acceptance_at_k_rate",
            "mean_best_validation_quality_score",
            "reference_objective_at_1_rate",
            "reference_objective_at_k_rate",
        ]
    }
    enriched: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("validation_acceptance_at_1_rate", float(item.get("anchor_pass_at_1_rate") or 0.0))
        item.setdefault("validation_acceptance_at_k_rate", float(item.get("anchor_pass_at_k_rate") or 0.0))
        item.setdefault("mean_best_validation_quality_score", float(item.get("mean_best_validation_score") or 0.0))
        for key, baseline_value in baseline_metrics.items():
            item["delta_{0}".format(key)] = round(float(item.get(key) or 0.0) - baseline_value, 6)
        item["mean_best_validation_score"] = item["mean_best_validation_quality_score"]
        item["delta_mean_best_validation_score"] = item["delta_mean_best_validation_quality_score"]
        item["delta_anchor_pass_at_1_rate"] = item["delta_validation_acceptance_at_1_rate"]
        item["delta_anchor_pass_at_k_rate"] = item["delta_validation_acceptance_at_k_rate"]
        enriched.append(item)
    return enriched


def _write_retrieval_score_sweep_outputs(payload: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "retrieval_score_profile_sweep.json"
    csv_path = output_dir / "retrieval_score_profile_sweep.csv"
    md_path = output_dir / "retrieval_score_profile_sweep.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = list(payload.get("profiles") or [])
    fieldnames = [
        "profile",
        "query_count",
        "validation_acceptance_at_1",
        "validation_acceptance_at_k",
        "mean_best_validation_quality_score",
        "reference_objective_at_1",
        "reference_objective_at_k",
        "unique_case_count",
        "unique_passed_case_count",
        "delta_validation_acceptance_at_1_rate",
        "delta_validation_acceptance_at_k_rate",
        "delta_mean_best_validation_quality_score",
        "delta_reference_objective_at_1_rate",
        "delta_reference_objective_at_k_rate",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    lines = [
        "# Retrieval Score Profile Sweep",
        "",
        "This sweep reruns the same benchmark queries with different retrieval scoring profiles. "
        "The outputs measure deterministic anchor consistency, not independent human-label recall.",
        "",
        "| Profile | Validation Acceptance@1 | Validation Acceptance@K | Mean Best Validation Quality | Reference Objective@1 | Unique Cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {validation_acceptance_at_1} | {validation_acceptance_at_k} | {mean_best_validation_quality_score:.2f} | "
            "{reference_objective_at_1} | {unique_case_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- JSON: `retrieval_score_profile_sweep.json`",
            "- CSV: `retrieval_score_profile_sweep.csv`",
            "- Per-profile benchmark artifacts: `profiles/<profile>/`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _run_retrieval_score_sweep(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    profiles: Sequence[str],
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_profiles = list(profiles) or ["default", "equal"]
    raw_rows: List[Dict[str, object]] = []
    for profile_name in selected_profiles:
        profile_output = output_dir / "profiles" / profile_name
        _run_benchmark(
            config_path=config_path,
            db_path=db_path,
            output_dir=profile_output,
            candidate_pool=candidate_pool,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            retrieval_score_config=RetrievalScoreConfig(profile_name=profile_name),
        )
        raw_rows.append(_retrieval_score_sweep_row(profile_name, profile_output))

    payload = {
        "schema": "retrieval_score_profile_sweep_v1",
        "config": str(config_path),
        "db": str(db_path),
        "candidate_pool": int(candidate_pool),
        "query_mode": query_mode,
        "rerank_mode": rerank_mode,
        "profiles": _with_baseline_deltas(raw_rows),
        "limitations": [
            "Reference anchors are deterministic weakly supervised targets, not independent human annotations.",
            "The sweep isolates retrieval score weights; validation thresholds and benchmark anchors are unchanged.",
        ],
    }
    _write_retrieval_score_sweep_outputs(payload, output_dir)
    return payload


def _write_validation_score_sweep_outputs(payload: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_score_profile_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = list(payload.get("profiles") or [])
    fieldnames = [
        "profile",
        "query_count",
        "validation_acceptance_at_1",
        "validation_acceptance_at_k",
        "mean_best_validation_quality_score",
        "reference_objective_at_1",
        "reference_objective_at_k",
        "unique_case_count",
        "unique_passed_case_count",
        "delta_mean_best_validation_quality_score",
        "output_dir",
    ]
    with (output_dir / "validation_score_profile_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    lines = [
        "# Validation Score Profile Sweep",
        "",
        "This sweep changes only the weights of the continuous validation-quality score. "
        "The binary acceptance gate, query definitions, and retrieval order remain fixed.",
        "The reported values are deterministic anchor-consistency diagnostics, not independent human-label recall.",
        "",
        "| Profile | Validation Acceptance@1 | Validation Acceptance@K | Mean Best Validation Quality | Reference Objective@1 | Unique Cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {validation_acceptance_at_1} | {validation_acceptance_at_k} | {mean_best_validation_quality_score:.2f} | "
            "{reference_objective_at_1} | {unique_case_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "- JSON: `validation_score_profile_sweep.json`",
            "- CSV: `validation_score_profile_sweep.csv`",
            "- Per-profile benchmark artifacts: `profiles/<profile>/`",
            "",
        ]
    )
    (output_dir / "validation_score_profile_sweep.md").write_text("\n".join(lines), encoding="utf-8")


def _run_validation_score_sweep(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    profiles: Sequence[str],
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_profiles = list(profiles) or ["default", "equal"]
    rows: List[Dict[str, object]] = []
    for profile_name in selected_profiles:
        profile_output = output_dir / "profiles" / profile_name
        _run_benchmark(
            config_path=config_path,
            db_path=db_path,
            output_dir=profile_output,
            candidate_pool=candidate_pool,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            validation_config=ValidationConfig(
                name="validation_score_{0}".format(profile_name),
                score_profile=profile_name,
            ),
        )
        rows.append(_retrieval_score_sweep_row(profile_name, profile_output))

    rows = _with_baseline_deltas(rows)
    for row in rows:
        row["delta_mean_best_validation_quality_score"] = row.get("delta_mean_best_validation_quality_score", 0.0)
    payload = {
        "schema": "validation_score_profile_sweep_v1",
        "config": str(config_path),
        "db": str(db_path),
        "candidate_pool": int(candidate_pool),
        "query_mode": query_mode,
        "rerank_mode": rerank_mode,
        "profiles": rows,
        "protocol": {
            "varied": "continuous validation-quality score weights",
            "fixed": ["retrieval_score_weights", "query_benchmark", "pass_gate_thresholds", "behavior_thresholds"],
            "interpretation": "anchor consistency and quality-score sensitivity, not independent semantic recall",
        },
    }
    _write_validation_score_sweep_outputs(payload, output_dir)
    return payload


def _threshold_scale_overrides(scale: float) -> Dict[str, float]:
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("Validation threshold scale must be positive.")
    return {
        name: float(value) * scale
        for name, value in HEURISTIC_THRESHOLDS.items()
        if name.endswith(("_m", "_mps"))
    }


def _write_threshold_sweep_outputs(payload: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_threshold_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = list(payload.get("profiles") or [])
    fieldnames = [
        "profile",
        "threshold_scale",
        "query_count",
        "validation_acceptance_at_1",
        "validation_acceptance_at_k",
        "mean_best_validation_quality_score",
        "reference_objective_at_1",
        "reference_objective_at_k",
        "unique_case_count",
        "unique_passed_case_count",
        "output_dir",
    ]
    with (output_dir / "validation_threshold_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    lines = [
        "# Validation Threshold Sweep",
        "",
        "The sweep changes only metric-valued behavior thresholds; retrieval weights, query definitions, and validation gates remain fixed.",
        "The reported values are deterministic anchor consistency metrics, not independent human-label recall.",
        "",
        "| Profile | Scale | Validation Acceptance@1 | Validation Acceptance@K | Mean Best Validation Quality | Reference Objective@1 | Unique Cases |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {threshold_scale:.2f} | {validation_acceptance_at_1} | {validation_acceptance_at_k} | "
            "{mean_best_validation_quality_score:.2f} | {reference_objective_at_1} | {unique_case_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "- JSON: `validation_threshold_sweep.json`",
            "- CSV: `validation_threshold_sweep.csv`",
            "- Per-profile artifacts: `profiles/<profile>/`",
            "",
        ]
    )
    (output_dir / "validation_threshold_sweep.md").write_text("\n".join(lines), encoding="utf-8")


def _run_validation_threshold_sweep(
    config_path: Path,
    db_path: Path,
    output_dir: Path,
    candidate_pool: int,
    scales: Sequence[float],
    query_mode: str = "rule",
    rerank_mode: str = "none",
    llm_config: Optional[LLMConfig] = None,
    learned_reranker_checkpoint: Optional[Path] = None,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_scales = [float(value) for value in (scales or (0.85, 1.0, 1.15))]
    rows: List[Dict[str, object]] = []
    for scale in selected_scales:
        scale_label = "{0:.2f}".format(scale).replace(".", "p")
        profile_name = "geometry_scale_{0}".format(scale_label)
        profile_output = output_dir / "profiles" / profile_name
        _run_benchmark(
            config_path=config_path,
            db_path=db_path,
            output_dir=profile_output,
            candidate_pool=candidate_pool,
            query_mode=query_mode,
            rerank_mode=rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=learned_reranker_checkpoint,
            validation_config=ValidationConfig(
                name=profile_name,
                threshold_overrides=_threshold_scale_overrides(scale),
            ),
        )
        row = _retrieval_score_sweep_row(profile_name, profile_output)
        row["threshold_scale"] = scale
        rows.append(row)

    payload = {
        "schema": "validation_threshold_sweep_v1",
        "config": str(config_path),
        "db": str(db_path),
        "candidate_pool": int(candidate_pool),
        "query_mode": query_mode,
        "rerank_mode": rerank_mode,
        "profiles": rows,
        "protocol": {
            "thresholds": "metric-valued heuristic thresholds scaled together",
            "fixed": ["retrieval_score_weights", "query_benchmark", "pass_gate_thresholds"],
            "interpretation": "anchor consistency, not independent semantic recall",
        },
    }
    _write_threshold_sweep_outputs(payload, output_dir)
    return payload


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "inspect-archives":
        inventory = discover_archive_inventory(Path(args.workspace))
        print(json.dumps(inventory.to_dict(), indent=2, ensure_ascii=False))
        return

    if args.command == "prepare-data":
        extracted = prepare_data(
            workspace=Path(args.workspace),
            dataroot=Path(args.dataroot),
            archives=args.archive or None,
            profile=args.profile,
        )
        print("Prepared dataroot:", Path(args.dataroot).resolve())
        for item in extracted:
            print("  extracted:", item)
        return

    if args.command == "build-index":
        stats = build_index(
            version=args.version,
            dataroot=Path(args.dataroot),
            db_path=Path(args.db),
            scene_limit=args.scene_limit,
            verbose=True,
        )
        print("Built index:", Path(args.db).resolve())
        print(json.dumps(stats, indent=2))
        return

    if args.command == "migrate-index":
        metadata = migrate_index_schema(Path(args.db))
        print("Migrated index:", Path(args.db).resolve())
        print(json.dumps(metadata, indent=2))
        return

    if args.command == "query":
        llm_config = _resolve_llm_config(args)
        result = run_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
            retrieval_score_config=RetrievalScoreConfig(profile_name=args.retrieval_score_profile),
        )
        print("Query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "run-multimodal-retrieval":
        llm_config = _resolve_llm_config(args) if args.query_mode == "llm" else None
        result = run_multimodal_retrieval_report(
            db_path=Path(args.db),
            query_text=args.text,
            output_dir=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            llm_config=llm_config,
        )
        print("Multimodal retrieval report:", Path(result["markdown"]).resolve())
        print("Candidates:", result["candidate_count"])
        return

    if args.command == "train-learned-retriever":
        report = train_learned_scene_retriever(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            config=LearnedRetrieverConfig(
                text_hash_dim=args.text_hash_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                negatives_per_query=args.negatives_per_query,
                candidate_pool=args.candidate_pool,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device=args.device,
            ),
        )
        print("Learned retriever checkpoint:", Path(report["checkpoint_path"]).resolve())
        print(
            "Training groups:",
            report["train_group_count"],
            "Validation groups:",
            report["validation_group_count"],
        )
        print(
            "Validation weak-anchor consistency@1:",
            dict(report.get("validation_metrics") or {}).get("recall_at_1"),
        )
        return

    if args.command == "train-large-learned-retriever":
        report = train_weakly_supervised_scene_retriever(
            db_path=Path(args.db),
            output_dir=Path(args.output),
            max_groups_per_family=args.max_groups_per_family,
            config=LearnedRetrieverConfig(
                text_hash_dim=args.text_hash_dim,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                negatives_per_query=args.negatives_per_query,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device=args.device,
            ),
        )
        print("Large learned retriever checkpoint:", Path(report["checkpoint_path"]).resolve())
        print("Training groups:", report["train_group_count"], "Validation groups:", report["validation_group_count"])
        print(
            "Validation weak-anchor consistency@1:",
            dict(report.get("validation_metrics") or {}).get("recall_at_1"),
        )
        return

    if args.command == "run-learned-retrieval":
        result = run_learned_retrieval_report(
            db_path=Path(args.db),
            query_text=args.text,
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
        )
        print("Learned retrieval report:", Path(result["markdown"]).resolve())
        print("Candidates:", result["candidate_count"])
        return

    if args.command == "run-failure-aware-reranking":
        result = run_failure_aware_reranking_eval(
            query_config=Path(args.query_config),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            learned_checkpoint=Path(args.checkpoint),
            candidate_pool=args.candidate_pool,
            top_k=args.top_k,
            max_queries=args.max_queries,
        )
        print("Failure-aware reranking:", Path(args.output).resolve())
        print(json.dumps(result["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "langgraph-query":
        llm_config = _resolve_llm_config(args)
        result = run_langgraph_query_pipeline(
            db_path=Path(args.db),
            query_text=args.text,
            output_root=Path(args.output),
            top_k=args.top_k,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
            retrieval_score_config=RetrievalScoreConfig(profile_name=args.retrieval_score_profile),
        )
        print("LangGraph query output:", result["query_dir"])
        print("Candidates:", result["candidate_count"], "Selected:", result["selected_count"])
        return

    if args.command == "run-research-agent":
        llm_config = _resolve_llm_config(
            args,
            require=True,
            default_base_url=DEFAULT_OLLAMA_BASE_URL,
            default_model=DEFAULT_OLLAMA_MODEL,
        )
        result = run_research_agent(
            output_dir=Path(args.output),
            llm_config=llm_config,
            artifact_paths=[Path(path) for path in args.artifact] if args.artifact else None,
            focus=args.focus,
        )
        print("Research agent report:", result["report_path"])
        print("Output:", result["output_dir"])
        return

    if args.command == "inspect-ollama-model":
        llm_config = _resolve_llm_config(
            args,
            require=True,
            default_base_url=DEFAULT_OLLAMA_BASE_URL,
            default_model=DEFAULT_OLLAMA_MODEL,
        )
        metadata = inspect_ollama_model(llm_config)
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Ollama model metadata:", output_path)
        print(json.dumps({"model": metadata["model"], "digest": metadata.get("digest", "")}, indent=2))
        return

    if args.command == "run-failure-mining":
        result = mine_model_failures(
            inputs=[Path(path) for path in args.input],
            output_dir=Path(args.output),
            max_queries=args.max_queries,
            min_count=args.min_count,
        )
        print("Failure mining report:", result["report_md"])
        print("Benchmark update YAML:", result["benchmark_yaml"])
        print("Failures:", result["failure_record_count"], "Update queries:", result["update_query_count"])
        return

    if args.command == "benchmark":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        summaries = _run_benchmark(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=output_dir,
            candidate_pool=args.candidate_pool,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
            retrieval_score_config=RetrievalScoreConfig(profile_name=args.retrieval_score_profile),
        )
        print("Benchmark output:", output_dir)
        print("Queries:", len(summaries))
        return

    if args.command == "benchmark-score-sweep":
        llm_config = _resolve_llm_config(args)
        payload = _run_retrieval_score_sweep(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            candidate_pool=args.candidate_pool,
            profiles=args.profile,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("Retrieval score sweep:", Path(args.output).resolve())
        print(json.dumps({"profiles": [row["profile"] for row in payload["profiles"]]}, indent=2))
        return

    if args.command == "benchmark-threshold-sweep":
        llm_config = _resolve_llm_config(args)
        payload = _run_validation_threshold_sweep(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            candidate_pool=args.candidate_pool,
            scales=args.scale,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("Validation threshold sweep:", Path(args.output).resolve())
        print(json.dumps({"profiles": [row["profile"] for row in payload["profiles"]]}, indent=2))
        return

    if args.command == "benchmark-validation-score-sweep":
        llm_config = _resolve_llm_config(args)
        payload = _run_validation_score_sweep(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_dir=Path(args.output),
            candidate_pool=args.candidate_pool,
            profiles=args.profile,
            query_mode=args.query_mode,
            rerank_mode=args.rerank_mode,
            llm_config=llm_config,
            learned_reranker_checkpoint=_optional_path(args.learned_reranker_checkpoint),
        )
        print("Validation score sweep:", Path(args.output).resolve())
        print(json.dumps({"profiles": [row["profile"] for row in payload["profiles"]]}, indent=2))
        return

    if args.command == "generate-counterfactual-benchmark":
        metadata = generate_counterfactual_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Counterfactual benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-scenario-mining-benchmark":
        metadata = generate_scenario_mining_benchmark_from_case_library(
            case_library_path=Path(args.case_library),
            output_path=Path(args.output),
            max_cases=args.max_cases,
        )
        print("Scenario mining benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-perception-benchmark":
        metadata = generate_perception_benchmark_from_scenario_config(
            config_path=Path(args.config),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("Perception benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-world-model-benchmark":
        metadata = generate_world_model_benchmark_from_perception_benchmark(
            perception_benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("World-model benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-perception-predictions":
        metadata = generate_proxy_perception_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("Proxy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-nuscenes-predictions":
        metadata = adapt_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_path=Path(args.output),
            task_type=args.task,
        )
        print("Adapted predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "filter-perception-benchmark":
        metadata = filter_perception_benchmark_by_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_path=Path(args.output),
            coverage_mode=args.coverage_mode,
        )
        print("Filtered perception benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-perception-predictions":
        summary = evaluate_perception_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
        )
        print("Perception evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-predictions":
        metadata = adapt_and_evaluate_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            task_type=args.task,
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
        )
        print("nuScenes perception evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-predictions-covered":
        metadata = adapt_filter_and_evaluate_nuscenes_predictions(
            benchmark_path=Path(args.benchmark),
            db_path=Path(args.db),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            task_type=args.task,
            profile_name=args.profile_name,
            match_distance_m=args.match_distance_m,
            coverage_mode=args.coverage_mode,
        )
        print("Covered nuScenes perception evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-perception-study":
        metadata = run_proxy_perception_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("Proxy perception study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-world-model-predictions":
        metadata = generate_proxy_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("Proxy world-model predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-world-model-predictions":
        metadata = adapt_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_path=Path(args.output),
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("Adapted world-model predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-nuscenes-forecast-predictions":
        metadata = adapt_nuscenes_forecast_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_path=Path(args.output),
            mode_selection=args.mode_selection,
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("Adapted nuScenes forecasts:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-world-model-predictions":
        summary = evaluate_world_model_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
        )
        print("World-model evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuscenes-forecast-predictions":
        metadata = adapt_and_evaluate_nuscenes_forecast_predictions(
            benchmark_path=Path(args.benchmark),
            input_path=Path(args.input),
            output_dir=Path(args.output),
            mode_selection=args.mode_selection,
            profile_name=args.profile_name,
            rasterize_trajectory=not bool(args.no_rasterize_trajectory),
        )
        print("nuScenes forecast evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuscenes-forecast-baselines":
        metadata = generate_nuscenes_forecast_baselines(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
        )
        print("nuScenes forecast baselines:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuscenes-forecast-baselines":
        metadata = run_nuscenes_forecast_baselines(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
            mode_selection=args.mode_selection,
        )
        print("nuScenes forecast baseline evaluation:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-contextvae-world-model-study":
        metadata = run_contextvae_world_model_study(
            benchmark_path=Path(args.benchmark),
            dataroot=Path(args.dataroot),
            version=args.version,
            output_dir=Path(args.output),
            repo_dir=Path(args.repo),
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
            batch_size=args.batch_size,
            mode_count=args.mode_count,
            clustering_samples=args.clustering_samples,
            map_scale=args.map_scale,
            seed=args.seed,
        )
        print("ContextVAE world-model study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-world-model-study":
        metadata = run_proxy_world_model_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("Proxy world-model study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-world-model-evaluations":
        metadata = compare_world_model_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("World-model comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "export-world-model-replay":
        metadata = export_world_model_replay(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
            export_format=args.format,
        )
        print("World-model replay export:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "render-world-model-case-studies":
        metadata = render_world_model_case_studies(
            benchmark_path=Path(args.benchmark),
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
            max_cases=args.max_cases,
        )
        print("World-model case studies:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "inspect-nuplan":
        inventory = inspect_nuplan_dataset(dataset_root=Path(args.dataset_root))
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuplan-replay-benchmark":
        metadata = generate_nuplan_replay_benchmark(
            split_dir=Path(args.split_dir),
            output_path=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            scenario_tags=args.scenario_tag or None,
        )
        print("nuPlan replay benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-nuplan-proxy-rollouts":
        metadata = generate_nuplan_proxy_rollouts(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("nuPlan proxy rollouts:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-nuplan-rollouts":
        summary = evaluate_nuplan_rollouts(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
            collision_distance_m=args.collision_distance_m,
        )
        print("nuPlan replay evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-replay-study":
        manifest = run_nuplan_replay_study(
            split_dir=Path(args.split_dir),
            output_dir=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            min_anchor_gap_s=args.min_anchor_gap_s,
            scenario_tags=args.scenario_tag or None,
            profiles=args.profile or None,
        )
        print("nuPlan replay study:", Path(args.output).resolve())
        print(json.dumps(manifest["benchmark"]["metadata"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-replay-sweep":
        result = run_experiment_config(Path(args.config))
        print("nuPlan replay sweep:", Path(result["result"].get("output_dir", DEFAULT_NUPLAN_REPLAY_SWEEP_OUTPUT)).resolve())
        print(json.dumps(result["result"].get("overview", {}), indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-closed-loop-sweep":
        result = run_experiment_config(Path(args.config))
        print(
            "nuPlan closed-loop sweep:",
            Path(result["result"].get("output_dir", DEFAULT_NUPLAN_CLOSED_LOOP_SWEEP_OUTPUT)).resolve(),
        )
        print(json.dumps(result["result"].get("overview", {}), indent=2, ensure_ascii=False))
        return

    if args.command == "run-nuplan-closed-loop-study":
        manifest = run_nuplan_closed_loop_study(
            split_dir=Path(args.split_dir),
            output_dir=Path(args.output),
            max_dbs=args.max_dbs,
            max_cases=args.max_cases,
            max_cases_per_db=args.max_cases_per_db,
            history_s=args.history_s,
            future_s=args.future_s,
            frame_hz=args.frame_hz,
            min_anchor_gap_s=args.min_anchor_gap_s,
            scenario_tags=args.scenario_tag or None,
            profiles=args.profile or None,
        )
        print("nuPlan closed-loop study:", Path(args.output).resolve())
        print(json.dumps(manifest["comparison"]["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "inspect-bench2drive":
        inventory = inspect_bench2drive_dataset(
            dataset_root=Path(args.dataset_root),
            sample_archives=args.sample_archives,
        )
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "build-bench2drive-vision-manifest":
        metadata = build_bench2drive_vision_manifest(
            dataset_root=Path(args.dataset_root),
            output_path=Path(args.output),
            max_archives=args.max_archives,
            frame_stride=args.frame_stride,
            future_steps=args.future_steps,
            future_frame_stride=args.future_frame_stride,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
            cache_root=None if args.no_cache else Path(args.cache_root),
            verbose=bool(args.verbose),
        )
        print("Bench2Drive vision manifest:", Path(args.output).resolve())
        print(json.dumps({k: metadata[k] for k in ["archive_count", "row_count", "split_counts"]}, indent=2))
        return

    if args.command == "split-bench2drive-vision-manifest":
        metadata = resplit_bench2drive_vision_manifest(
            manifest_path=Path(args.manifest),
            output_path=Path(args.output),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        print("Bench2Drive split manifest:", Path(args.output).resolve())
        print(json.dumps({k: metadata[k] for k in ["archive_count", "row_count", "split_counts"]}, indent=2))
        return

    if args.command == "build-bench2drive-vision-tensor-cache":
        metadata = build_bench2drive_vision_tensor_cache(
            manifest_path=Path(args.manifest),
            output_manifest_path=Path(args.output),
            cache_dir=Path(args.cache_dir),
            image_size=args.image_size,
            max_rows=args.max_rows,
            num_workers=args.num_workers,
            chunk_rows=args.chunk_rows,
            verbose=bool(args.verbose),
        )
        print("Bench2Drive vision tensor manifest:", Path(args.output).resolve())
        print(json.dumps({k: metadata[k] for k in ["row_count", "invalid_sample_count", "split_counts"]}, indent=2))
        return

    if args.command == "train-bench2drive-vision-planner":
        report = train_vision_e2e_planner(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            model_size=args.model_size,
            architecture=args.architecture,
            camera_pooling=args.camera_pooling,
            dropout=args.dropout,
            trajectory_modes=args.trajectory_modes,
            trajectory_selection=args.trajectory_selection,
            trajectory_top_k=args.trajectory_top_k,
            trajectory_temperature=args.trajectory_temperature,
            input_mode=args.input_mode,
            spatial_pool_size=args.spatial_pool_size,
            waypoint_loss_weight=args.waypoint_loss_weight,
            selected_waypoint_loss_weight=args.selected_waypoint_loss_weight,
            displacement_loss_weight=args.displacement_loss_weight,
            endpoint_loss_weight=args.endpoint_loss_weight,
            path_length_loss_weight=args.path_length_loss_weight,
            control_loss_weight=args.control_loss_weight,
            brake_loss_weight=args.brake_loss_weight,
            brake_positive_weight=args.brake_positive_weight,
            risk_sample_weight=args.risk_sample_weight,
            lateral_loss_weight=args.lateral_loss_weight,
            turn_sample_weight=args.turn_sample_weight,
            turn_lateral_threshold_m=args.turn_lateral_threshold_m,
            mode_classification_weight=args.mode_classification_weight,
            selection_metric=args.selection_metric,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            device=args.device,
            use_data_parallel=not bool(args.no_data_parallel),
            precision=args.precision,
            allow_tf32=not bool(args.disable_tf32),
            cudnn_benchmark=not bool(args.disable_cudnn_benchmark),
            nonfinite_check_interval=args.nonfinite_check_interval,
            seed=args.seed,
            verbose=bool(args.verbose),
        )
        if bool(report.get("is_main_process", _is_main_process())):
            print("Bench2Drive vision planner checkpoint:", Path(report["checkpoint_path"]).resolve())
            print(
                json.dumps(
                    {"train_samples": report["train_sample_count"], "val_samples": report["val_sample_count"]},
                    indent=2,
                )
            )
        return

    if args.command == "evaluate-bench2drive-vision-planner":
        report = evaluate_vision_e2e_planner(
            manifest_path=Path(args.manifest),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            split=args.split,
            batch_size=args.batch_size,
            image_size=args.image_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            device=args.device,
        )
        print("Bench2Drive vision planner evaluation:", Path(args.output).resolve())
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.command == "compare-bench2drive-vision-predictions":
        report = compare_vision_e2e_prediction_sets(
            baseline_predictions_path=Path(args.baseline),
            candidate_predictions_path=Path(args.candidate),
            output_dir=Path(args.output),
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
            allow_case_intersection=args.allow_case_intersection,
        )
        print("Bench2Drive paired prediction comparison:", Path(args.output).resolve())
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.command == "calibrate-bench2drive-trajectory-selection":
        report = calibrate_trajectory_selection(
            predictions_path=Path(args.predictions),
            evaluation_report_path=Path(args.evaluation_report),
            checkpoint_path=Path(args.checkpoint),
            output_checkpoint_path=Path(args.output_checkpoint),
            output_report_path=Path(args.output_report),
            temperatures=args.temperature or (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
            top_k=args.top_k,
        )
        print("Bench2Drive calibrated checkpoint:", Path(args.output_checkpoint).resolve())
        print(json.dumps(report["selected"], indent=2))
        return

    if args.command == "diagnose-bench2drive-vision-planner":
        report = diagnose_vision_e2e_predictions(
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            evaluation_report_path=Path(args.evaluation_report) if args.evaluation_report else None,
            brake_threshold=args.brake_threshold,
        )
        print("Bench2Drive vision planner diagnostics:", Path(args.output).resolve())
        print(json.dumps(report["readiness"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-bench2drive-vision-closed-loop":
        report = run_bench2drive_vision_closed_loop(
            manifest_path=Path(args.manifest),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            split=args.split,
            max_cases=args.max_cases,
            max_frames_per_clip=args.max_frames_per_clip,
            image_size=args.image_size,
            device=args.device,
            video_fps=args.video_fps,
            render_case_media=args.render_case_media,
            case_selection=args.case_selection,
            control_config=ClosedLoopControlConfig(
                dt_s=args.dt_s,
                horizon_s=args.horizon_s,
                target_speed_mps=args.target_speed_mps,
                brake_probability_threshold=args.brake_threshold,
                lookahead_m=args.lookahead_m,
                speed_kp=args.speed_kp,
            ),
        )
        print("Bench2Drive vision closed-loop evaluation:", Path(args.output).resolve())
        print(json.dumps(report["comparison"], indent=2))
        return

    if args.command == "compare-bench2drive-closed-loop":
        report = compare_bench2drive_closed_loop_reports(
            baseline_report_path=Path(args.baseline),
            candidate_report_path=Path(args.candidate),
            output_dir=Path(args.output),
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
            allow_case_intersection=args.allow_case_intersection,
        )
        print("Bench2Drive paired closed-loop comparison:", Path(args.output).resolve())
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.command == "inspect-carla":
        inventory = inspect_carla_runtime(carla_root=Path(args.carla_root))
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "print-carla-launch-command":
        command = build_carla_launch_command(
            carla_root=Path(args.carla_root),
            render_offscreen=not bool(args.no_render_offscreen),
            null_rhi=bool(args.null_rhi),
            port=args.port,
            quality_level=args.quality_level,
            fps=args.fps,
        )
        print(format_carla_launch_command(command, cuda_visible_devices=args.cuda_visible_devices))
        return

    if args.command == "carla-connection-smoke":
        result = run_carla_connection_smoke(
            carla_root=Path(args.carla_root),
            host=args.host,
            port=args.port,
            timeout_s=args.timeout_s,
            town=args.town,
            load_town=args.load_town,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "run-carla-vision-closed-loop":
        report = run_carla_vision_closed_loop(
            carla_root=Path(args.carla_root),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            host=args.host,
            port=args.port,
            town=args.town,
            spawn_index=args.spawn_index,
            destination_index=args.destination_index,
            route_sampling_resolution_m=args.route_sampling_resolution_m,
            route_min_length_m=args.route_min_length_m,
            route_max_length_m=args.route_max_length_m,
            route_preferred_length_m=args.route_preferred_length_m,
            fps=args.fps,
            horizon_s=args.horizon_s,
            image_size=args.image_size,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            video_width=args.video_width,
            video_height=args.video_height,
            carla_quality_level=args.carla_quality_level,
            scenario_type=args.scenario_type,
            scenario_name=args.scenario_name,
            target_speed_mps=args.target_speed_mps,
            brake_probability_threshold=args.brake_threshold,
            enable_scenario_safety_override=not bool(args.no_scenario_safety_override),
            enable_lane_departure_guard=not bool(args.no_lane_departure_guard),
            condition_ego_route_traffic_lights=bool(args.condition_ego_route_traffic_lights),
            device=args.device,
            auto_launch=bool(args.auto_launch),
            cuda_visible_devices=args.cuda_visible_devices,
            graphics_adapter=args.graphics_adapter,
            traffic_manager_port=args.traffic_manager_port,
            seed=args.seed,
            launch_timeout_s=args.launch_timeout_s,
            rpc_timeout_s=args.rpc_timeout_s,
            keep_server=bool(args.keep_server),
            video_fps=args.video_fps,
            video_encoder=args.video_encoder,
            video_nvenc_preset=args.video_nvenc_preset,
            video_quality=args.video_quality,
        )
        print("CARLA vision closed-loop rollout:", Path(args.output).resolve())
        print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
        return

    if args.command == "mine-carla-semantic-demos":
        report = mine_carla_semantic_demos(
            carla_root=Path(args.carla_root),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output),
            trials_output_dir=Path(args.trials_output),
            host=args.host,
            port_start=args.port_start,
            town=args.town,
            fps=args.fps,
            horizon_s=args.horizon_s,
            image_size=args.image_size,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            video_width=args.video_width,
            video_height=args.video_height,
            carla_quality_level=args.carla_quality_level,
            brake_probability_threshold=args.brake_threshold,
            device=args.device,
            auto_launch=not bool(args.no_auto_launch),
            cuda_visible_devices=args.cuda_visible_devices,
            traffic_manager_port_start=args.traffic_manager_port_start,
            launch_timeout_s=args.launch_timeout_s,
            rpc_timeout_s=args.rpc_timeout_s,
            keep_server=bool(args.keep_server),
            video_fps=args.video_fps,
            video_encoder=args.video_encoder,
            video_nvenc_preset=args.video_nvenc_preset,
            video_quality=args.video_quality,
            enable_scenario_safety_override=not bool(args.no_scenario_safety_override),
            enable_lane_departure_guard=bool(args.enable_lane_departure_guard),
            condition_ego_route_traffic_lights=not bool(args.no_condition_ego_route_traffic_lights),
            max_attempts_per_target=args.max_attempts_per_target,
            require_hevc=not bool(args.allow_non_hevc),
        )
        print("CARLA semantic demo mining:", Path(args.output).resolve())
        print(
            json.dumps(
                {
                    "status": report.get("status"),
                    "passed_target_count": report.get("passed_target_count"),
                    "target_count": report.get("target_count"),
                    "attempt_count": report.get("attempt_count"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "audit-carla-vision-rollout":
        report = audit_carla_vision_rollouts(
            report_path=Path(args.report),
            output_path=Path(args.output) if args.output else None,
            min_resolution_width=args.min_resolution_width,
            min_resolution_height=args.min_resolution_height,
            min_fps=args.min_fps,
            min_frames=args.min_frames,
            min_traffic_manager_vehicles=args.min_traffic_manager_vehicles,
            max_scripted_vehicles=args.max_scripted_vehicles,
            max_collision_count=args.max_collision_count,
            min_route_completion=args.min_route_completion,
            max_mean_lateral_error_m=args.max_mean_lateral_error_m,
            max_lateral_error_m=args.max_lateral_error_m,
            max_safety_override_ratio=args.max_safety_override_ratio,
            max_nearest_actor_distance_m=args.max_nearest_actor_distance_m,
            nearby_actor_distance_m=args.nearby_actor_distance_m,
            min_nearby_actor_ratio=args.min_nearby_actor_ratio,
            require_semantic_match=bool(args.require_semantic_match),
            require_model_control=not bool(args.allow_non_model_control),
            require_hevc=bool(args.require_hevc),
        )
        default_audit_name = (
            "carla_semantic_demo_audit.json"
            if Path(args.report).name == "carla_semantic_demo_report.json"
            else "carla_vision_video_audit.json"
        )
        print("CARLA vision rollout audit:", Path(args.output or Path(args.report).with_name(default_audit_name)).resolve())
        print(json.dumps({k: report[k] for k in ["status", "failure_count", "warning_count", "summary"]}, indent=2))
        return

    if args.command == "compare-nuplan-replay-evaluations":
        comparison = compare_nuplan_replay_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("nuPlan replay comparison:", Path(args.output).resolve())
        print(json.dumps(comparison["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "render-nuplan-replay-case-studies":
        metadata = render_nuplan_replay_case_studies(
            benchmark_path=Path(args.benchmark),
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
            max_cases=args.max_cases,
        )
        print("nuPlan replay case studies:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "inspect-dataset-backends":
        inventory = inspect_dataset_backends(
            nuscenes_root=Path(args.nuscenes_root),
            nuplan_root=Path(args.nuplan_root),
            index_root=Path(args.index_root),
        )
        if args.output:
            write_dataset_backend_inventory(inventory, Path(args.output))
            print("Dataset backend inventory:", Path(args.output).resolve())
        print(json.dumps(inventory, indent=2, ensure_ascii=False))
        return

    if args.command == "export-benchmark-catalog":
        catalog = write_benchmark_catalog(Path(args.output), build_default_benchmark_catalog())
        print("Benchmark catalog:", Path(args.output).resolve())
        print(json.dumps({"schema": catalog["schema"], "layer_count": len(catalog["layers"])}, indent=2))
        return

    if args.command == "export-unified-cases":
        cases = load_unified_case_source(Path(args.source), args.source_type)
        payload = write_unified_case_collection(
            cases,
            Path(args.output),
            metadata={"source": str(args.source), "source_type": args.source_type},
        )
        print("Unified risk cases:", Path(args.output).resolve())
        print(json.dumps({"schema": payload["schema"], "case_count": len(payload["cases"])}, indent=2))
        return

    if args.command == "run-experiment-config":
        result = run_experiment_config(Path(args.config))
        print("Experiment result:", result["experiment_id"])
        print(json.dumps(result["result"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-full-benchmark-suite":
        result = run_experiment_config(Path(args.config), reuse_case_library=args.reuse_case_library)
        print("Full benchmark suite:", Path(result["result"].get("output_dir", "outputs/full_benchmark_suite_v1")).resolve())
        print(json.dumps({"stages": list(result["result"].get("stages", {}).keys())}, indent=2))
        return

    if args.command == "generate-bev-occupancy-benchmark":
        metadata = generate_bev_occupancy_benchmark_from_perception_benchmark(
            perception_benchmark_path=Path(args.perception_benchmark),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("BEV occupancy benchmark:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-proxy-bev-occupancy-predictions":
        metadata = generate_proxy_bev_occupancy_predictions(
            benchmark_path=Path(args.benchmark),
            output_path=Path(args.output),
            profile_name=args.profile,
        )
        print("BEV occupancy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "adapt-perception-to-bev-occupancy":
        metadata = adapt_perception_predictions_to_bev_occupancy(
            benchmark_path=Path(args.benchmark),
            perception_predictions_path=Path(args.predictions),
            output_path=Path(args.output),
            profile_name=args.profile_name,
        )
        print("Adapted BEV occupancy predictions:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate-bev-occupancy":
        summary = evaluate_bev_occupancy_predictions(
            benchmark_path=Path(args.benchmark),
            predictions_path=Path(args.predictions),
            output_dir=Path(args.output),
            profile_name=args.profile_name,
        )
        print("BEV occupancy evaluation:", Path(args.output).resolve())
        print(json.dumps(summary["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-proxy-bev-occupancy-study":
        metadata = run_proxy_bev_occupancy_study(
            benchmark_path=Path(args.benchmark),
            output_dir=Path(args.output),
        )
        print("BEV occupancy study:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-bev-occupancy-evaluations":
        metadata = compare_bev_occupancy_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("BEV occupancy comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "compare-perception-evaluations":
        metadata = compare_perception_evaluations(
            evaluation_dirs=[Path(path) for path in args.eval_dir],
            output_dir=Path(args.output),
        )
        print("Perception comparison:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "enrich-case-library":
        metadata = enrich_case_library(
            case_library_path=Path(args.case_library),
            db_path=Path(args.db),
            output_path=Path(args.output),
        )
        print("Enriched case library:", Path(args.output).resolve())
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    if args.command == "generate-human-audit-set":
        manifest = generate_human_audit_set(
            case_library_path=Path(args.case_library),
            output_dir=Path(args.output),
            sample_size=args.sample_size,
            seed=args.seed,
            max_per_behavior=args.max_per_behavior,
        )
        print("Human audit set:", Path(args.output).resolve())
        print(json.dumps({"audit_item_count": manifest["audit_item_count"], "jsonl": manifest["jsonl"]}, indent=2))
        return

    if args.command == "evaluate-human-audit-set":
        metrics = evaluate_human_audit_set(
            annotation_path=Path(args.annotations),
            output_dir=Path(args.output),
        )
        print("Human audit evaluation:", Path(args.output).resolve())
        print(json.dumps(metrics["overview"], indent=2, ensure_ascii=False))
        return

    if args.command == "benchmark-compare":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        profile_runs = default_benchmark_profiles(include_llm=llm_config is not None)

        for profile in profile_runs:
            profile_output = output_dir / profile["name"]
            profile_output.mkdir(parents=True, exist_ok=True)
            _run_benchmark(
                config_path=Path(args.config),
                db_path=Path(args.db),
                output_dir=profile_output,
                candidate_pool=args.candidate_pool,
                query_mode=str(profile["query_mode"]),
                rerank_mode=str(profile["rerank_mode"]),
                llm_config=llm_config,
            )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(profile_runs)
        write_benchmark_comparison(comparison, output_dir)
        print("Benchmark comparison output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in profile_runs))
        return

    if args.command == "benchmark-ablate":
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_config = _resolve_llm_config(args)
        ablation_runs = default_ablation_profiles(
            base_query_mode=args.base_query_mode,
            base_rerank_mode=args.base_rerank_mode,
        )
        requires_llm = any(
            str(profile["query_mode"]) != "rule" or str(profile["rerank_mode"]) == "llm"
            for profile in ablation_runs
        )
        if requires_llm and llm_config is None:
            raise ValueError("The selected ablation profiles require Ollama configuration.")

        for profile in ablation_runs:
            profile_output = output_dir / str(profile["name"])
            profile_output.mkdir(parents=True, exist_ok=True)
            benchmark_metrics_path = profile_output / "benchmark_metrics.json"
            if not (args.reuse_existing and benchmark_metrics_path.exists()):
                _run_benchmark(
                    config_path=Path(args.config),
                    db_path=Path(args.db),
                    output_dir=profile_output,
                    candidate_pool=args.candidate_pool,
                    query_mode=str(profile["query_mode"]),
                    rerank_mode=str(profile["rerank_mode"]),
                    llm_config=llm_config,
                    validation_config=profile.get("validation_config"),
                )
            profile["output_dir"] = str(profile_output)

        comparison = build_benchmark_comparison(ablation_runs)
        write_benchmark_comparison(comparison, output_dir)
        write_ablation_manifest(ablation_runs, output_dir)
        build_comparison_browser(
            comparison_output_dir=output_dir,
            title="nuScenes Ablation Browser",
        )
        print("Benchmark ablation output:", output_dir)
        print("Profiles:", ", ".join(str(profile["name"]) for profile in ablation_runs))
        return

    if args.command == "build-gallery":
        if args.benchmark_output:
            metadata = build_benchmark_gallery(
                benchmark_output_dir=Path(args.benchmark_output),
                title=args.title or "nuScenes Benchmark Query Gallery",
            )
        else:
            metadata = build_comparison_browser(
                comparison_output_dir=Path(args.comparison_output),
                title=args.title or "nuScenes Benchmark Comparison Browser",
            )
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return


def _is_main_process() -> bool:
    return int(os.environ.get("RANK") or "0") == 0


if __name__ == "__main__":
    main()
