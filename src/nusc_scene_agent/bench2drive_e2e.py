from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import json
import math
import os
import random
import tempfile
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nusc_scene_agent.scenario_taxonomy import build_taxonomy_coverage


DEFAULT_BENCH2DRIVE_ROOT = Path("data/bench2drive/Bench2Drive-Base")
DEFAULT_BENCH2DRIVE_MANIFEST = Path("artifacts/bench2drive/vision_e2e_manifest.jsonl")
DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST = Path("artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl")
DEFAULT_BENCH2DRIVE_OUTPUT = Path("outputs/bench2drive_vision_e2e_final")
DEFAULT_BENCH2DRIVE_CACHE_ROOT = Path("data/bench2drive/cache/vision_e2e")
DEFAULT_BENCH2DRIVE_TENSOR_CACHE_ROOT = Path("data/bench2drive/cache/vision_e2e_tensor_cache")
DEFAULT_BENCH2DRIVE_CAMERAS = [
    "rgb_front",
    "rgb_front_left",
    "rgb_front_right",
    "rgb_back",
    "rgb_back_left",
    "rgb_back_right",
]
BENCH2DRIVE_PLANNER_DT_S = 0.5
BENCH2DRIVE_PREDICTION_COMPARISON_SCHEMA = "bench2drive_vision_prediction_comparison_v1"

BENCH2DRIVE_DATASET_SCHEMA = "bench2drive_dataset_inventory_v1"
BENCH2DRIVE_MANIFEST_SCHEMA = "bench2drive_vision_e2e_manifest_v1"
BENCH2DRIVE_TENSOR_CACHE_SCHEMA = "bench2drive_vision_tensor_cache_v1"
BENCH2DRIVE_TRAINING_SCHEMA = "bench2drive_vision_e2e_training_v1"
BENCH2DRIVE_EVAL_SCHEMA = "bench2drive_vision_e2e_eval_v1"
BENCH2DRIVE_DIAGNOSTIC_SCHEMA = "bench2drive_vision_planner_diagnostics_v1"
BENCH2DRIVE_CALIBRATION_SCHEMA = "bench2drive_trajectory_selection_calibration_v3"
_NORMALIZATION_TENSOR_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


@dataclass(frozen=True)
class VisionE2EModelConfig:
    model_size: str = "base"
    architecture: str = "conv_mlp"
    image_size: int = 160
    future_steps: int = 5
    route_feature_dim: int = 8
    camera_count: int = len(DEFAULT_BENCH2DRIVE_CAMERAS)
    camera_pooling: str = "mean"
    dropout: float = 0.0
    trajectory_modes: int = 1
    trajectory_selection: str = "argmax"
    trajectory_top_k: int = 2
    trajectory_temperature: float = 1.0
    trajectory_mode_calibrator: Optional[Dict[str, Any]] = None
    input_mode: str = "vision_route"
    spatial_pool_size: int = 2


@dataclass(frozen=True)
class VisionE2ELossConfig:
    waypoint_weight: float = 1.0
    selected_waypoint_weight: float = 0.5
    displacement_weight: float = 0.0
    endpoint_weight: float = 0.0
    path_length_weight: float = 0.0
    control_weight: float = 0.25
    brake_weight: float = 0.1
    brake_positive_weight: float = 1.0
    risk_sample_weight: float = 1.0
    lateral_loss_weight: float = 1.0
    turn_sample_weight: float = 1.0
    turn_lateral_threshold_m: float = 2.0
    brake_threshold: float = 0.5
    mode_classification_weight: float = 0.05


def inspect_bench2drive_dataset(
    dataset_root: Path = DEFAULT_BENCH2DRIVE_ROOT,
    *,
    sample_archives: int = 3,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    archives = sorted(dataset_root.glob("*.tar.gz"))
    archive_sizes = [path.stat().st_size for path in archives if path.exists()]
    samples = []
    for archive_path in archives[: max(int(sample_archives), 0)]:
        samples.append(_inspect_bench2drive_archive(archive_path))
    scenario_counts = Counter(_scenario_family_from_name(path.stem.replace(".tar", "")) for path in archives)
    return {
        "schema": BENCH2DRIVE_DATASET_SCHEMA,
        "dataset_root": str(dataset_root),
        "exists": dataset_root.exists(),
        "archive_count": len(archives),
        "total_size_gb": round(sum(archive_sizes) / (1024**3), 3),
        "scenario_family_count": len(scenario_counts),
        "scenario_families": dict(sorted(scenario_counts.items())),
        "camera_set": DEFAULT_BENCH2DRIVE_CAMERAS,
        "sample_archives": samples,
    }


def build_bench2drive_vision_manifest(
    dataset_root: Path = DEFAULT_BENCH2DRIVE_ROOT,
    output_path: Path = DEFAULT_BENCH2DRIVE_MANIFEST,
    *,
    max_archives: int = 0,
    frame_stride: int = 5,
    future_steps: int = 5,
    future_frame_stride: int = 5,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 7,
    cameras: Sequence[str] = DEFAULT_BENCH2DRIVE_CAMERAS,
    cache_root: Optional[Path] = DEFAULT_BENCH2DRIVE_CACHE_ROOT,
    verbose: bool = False,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    cache_path = Path(cache_root) if cache_root is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    archives = sorted(dataset_root.glob("*.tar.gz"))
    if max_archives > 0:
        archives = archives[: int(max_archives)]

    split_by_archive = _stratified_archive_splits(
        [(path.name, _scenario_family_from_name(path.name.replace(".tar.gz", ""))) for path in archives],
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        seed=int(seed),
    )

    rows_written = 0
    split_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    archive_summaries = []
    with output_path.open("w", encoding="utf-8") as handle:
        for archive_idx, archive_path in enumerate(archives, start=1):
            rows, summary = _build_manifest_rows_for_archive(
                archive_path,
                split=split_by_archive.get(archive_path.name, "train"),
                frame_stride=max(int(frame_stride), 1),
                future_steps=max(int(future_steps), 1),
                future_frame_stride=max(int(future_frame_stride), 1),
                cameras=list(cameras),
                cache_root=cache_path,
            )
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows_written += len(rows)
            split_counts.update(row["split"] for row in rows)
            scenario_counts.update(row["scenario_family"] for row in rows)
            archive_summaries.append(summary)
            if verbose:
                print(
                    f"[{archive_idx}/{len(archives)}] {archive_path.name}: {len(rows)} rows",
                    flush=True,
                )

    metadata = {
        "schema": BENCH2DRIVE_MANIFEST_SCHEMA,
        "dataset_root": str(dataset_root),
        "manifest_path": str(output_path),
        "archive_count": len(archives),
        "row_count": rows_written,
        "split_counts": dict(sorted(split_counts.items())),
        "scenario_family_counts": dict(sorted(scenario_counts.items())),
        "frame_stride": int(frame_stride),
        "future_steps": int(future_steps),
        "future_frame_stride": int(future_frame_stride),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": round(max(0.0, 1.0 - float(train_fraction) - float(val_fraction)), 6),
        "seed": int(seed),
        "split_protocol": "scenario-stratified archive-disjoint train/val/test",
        "split_integrity": _split_integrity_from_rows(archive_summaries),
        "taxonomy_coverage": build_taxonomy_coverage("bench2drive", scenario_counts),
        "cameras": list(cameras),
        "cache_root": str(cache_path) if cache_path is not None else "",
        "archive_summaries": archive_summaries[:64],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def resplit_bench2drive_vision_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 7,
) -> Dict[str, Any]:
    rows = _read_manifest_rows(Path(manifest_path))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    archive_families = {
        str(row.get("archive") or row.get("clip_name") or ""): str(row.get("scenario_family") or "unknown")
        for row in rows
    }
    if "" in archive_families:
        raise ValueError("Every Bench2Drive row must include an archive or clip_name for disjoint splitting.")
    split_by_archive = _stratified_archive_splits(
        sorted(archive_families.items()),
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        seed=int(seed),
    )
    split_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    split_scenario_counts: Dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(output_path.parent),
        prefix=output_path.name + ".",
        suffix=".building",
        delete=False,
    ) as handle:
        build_path = Path(handle.name)
        for source_row in rows:
            row = dict(source_row)
            archive = str(row.get("archive") or row.get("clip_name") or "")
            family = str(row.get("scenario_family") or "unknown")
            split = split_by_archive[archive]
            row["split"] = split
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            split_counts[split] += 1
            scenario_counts[family] += 1
            split_scenario_counts[split][family] += 1
    os.replace(build_path, output_path)
    archive_counts = Counter(split_by_archive.values())
    metadata = {
        "schema": BENCH2DRIVE_MANIFEST_SCHEMA,
        "source_manifest_path": str(manifest_path),
        "manifest_path": str(output_path),
        "row_count": len(rows),
        "archive_count": len(split_by_archive),
        "split_counts": dict(sorted(split_counts.items())),
        "split_archive_counts": dict(sorted(archive_counts.items())),
        "scenario_family_counts": dict(sorted(scenario_counts.items())),
        "split_scenario_family_counts": {
            split: dict(sorted(counts.items())) for split, counts in split_scenario_counts.items()
        },
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": round(max(0.0, 1.0 - float(train_fraction) - float(val_fraction)), 6),
        "seed": int(seed),
        "split_protocol": "scenario-stratified archive-disjoint train/val/test",
        "split_integrity": _split_integrity_from_manifest_rows(rows, split_by_archive),
        "taxonomy_coverage": build_taxonomy_coverage("bench2drive", scenario_counts),
    }
    source_metadata_path = Path(manifest_path).with_suffix(Path(manifest_path).suffix + ".metadata.json")
    if source_metadata_path.exists():
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        for key in ["tensor_cache_path", "image_size", "dtype", "shape", "cameras", "tensor_cache_size_gb"]:
            if key in source_metadata:
                metadata[key] = source_metadata[key]
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _stratified_archive_splits(
    archives: Sequence[Tuple[str, str]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Dict[str, str]:
    if float(train_fraction) >= 1.0:
        val_fraction = 0.0
    test_fraction = 1.0 - float(train_fraction) - float(val_fraction)
    if train_fraction <= 0.0 or val_fraction < 0.0 or test_fraction < -1e-9:
        raise ValueError("Split fractions must satisfy train > 0, val >= 0, and train + val <= 1.")
    fractions = {"train": float(train_fraction), "val": float(val_fraction), "test": max(test_fraction, 0.0)}
    grouped: Dict[str, List[str]] = {}
    for archive, family in archives:
        grouped.setdefault(str(family), []).append(str(archive))

    assignments: Dict[str, str] = {}
    for family in sorted(grouped):
        members = sorted(grouped[family])
        random.Random(f"{int(seed)}:{family}").shuffle(members)
        counts = _allocate_split_counts(len(members), fractions)
        cursor = 0
        for split in ["train", "val", "test"]:
            for archive in members[cursor : cursor + counts[split]]:
                assignments[archive] = split
            cursor += counts[split]
    return assignments


def _allocate_split_counts(size: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    names = ["train", "val", "test"]
    raw = {name: size * float(fractions[name]) for name in names}
    counts = {name: int(math.floor(raw[name])) for name in names}
    for name in sorted(names, key=lambda item: (raw[item] - counts[item], fractions[item]), reverse=True):
        if sum(counts.values()) >= size:
            break
        counts[name] += 1
    positive = [name for name in names if float(fractions[name]) > 0.0]
    if size >= len(positive):
        for name in positive:
            if counts[name] > 0:
                continue
            donor = max((item for item in positive if counts[item] > 1), key=lambda item: counts[item], default="")
            if donor:
                counts[donor] -= 1
                counts[name] += 1
    return counts


def _split_integrity_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    assignments = {str(row.get("archive") or ""): str(row.get("split") or "") for row in rows}
    return _split_integrity(assignments)


def _split_integrity_from_manifest_rows(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> Dict[str, Any]:
    observed = {
        str(row.get("archive") or row.get("clip_name") or ""): assignments[
            str(row.get("archive") or row.get("clip_name") or "")
        ]
        for row in rows
    }
    return _split_integrity(observed)


def _split_integrity(assignments: Mapping[str, str]) -> Dict[str, Any]:
    split_archives = {
        split: {archive for archive, assigned in assignments.items() if assigned == split}
        for split in ["train", "val", "test"]
    }
    overlaps = {
        "train_val": sorted(split_archives["train"] & split_archives["val"]),
        "train_test": sorted(split_archives["train"] & split_archives["test"]),
        "val_test": sorted(split_archives["val"] & split_archives["test"]),
    }
    return {
        "archive_disjoint": not any(overlaps.values()),
        "overlaps": overlaps,
    }


def train_vision_e2e_planner(
    manifest_path: Path = DEFAULT_BENCH2DRIVE_MANIFEST,
    output_dir: Path = DEFAULT_BENCH2DRIVE_OUTPUT,
    *,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    image_size: int = 160,
    model_size: str = "base",
    architecture: str = "conv_mlp",
    camera_pooling: str = "mean",
    dropout: float = 0.0,
    trajectory_modes: int = 1,
    trajectory_selection: str = "argmax",
    trajectory_top_k: int = 2,
    trajectory_temperature: float = 1.0,
    input_mode: str = "vision_route",
    spatial_pool_size: int = 2,
    waypoint_loss_weight: float = 1.0,
    selected_waypoint_loss_weight: float = 0.5,
    displacement_loss_weight: float = 0.0,
    endpoint_loss_weight: float = 0.0,
    path_length_loss_weight: float = 0.0,
    control_loss_weight: float = 0.25,
    brake_loss_weight: float = 0.1,
    brake_positive_weight: float = 1.0,
    risk_sample_weight: float = 1.0,
    lateral_loss_weight: float = 1.0,
    turn_sample_weight: float = 1.0,
    turn_lateral_threshold_m: float = 2.0,
    mode_classification_weight: float = 0.05,
    selection_metric: str = "ade_m",
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    device: str = "",
    use_data_parallel: bool = True,
    precision: str = "fp32",
    allow_tf32: bool = True,
    cudnn_benchmark: bool = True,
    nonfinite_check_interval: int = 0,
    seed: int = 7,
    verbose: bool = False,
) -> Dict[str, Any]:
    torch, _, _ = _require_torch_stack(require_pillow=True)
    dist_ctx = _setup_distributed_training(torch, device)
    target_device = dist_ctx["device"]
    distributed = bool(dist_ctx["distributed"])
    is_main_process = bool(dist_ctx["is_main_process"])
    _set_torch_seed(torch, int(seed))
    _configure_torch_precision(
        torch,
        allow_tf32=bool(allow_tf32),
        cudnn_benchmark=bool(cudnn_benchmark),
    )
    autocast_dtype = _resolve_autocast_dtype(torch, str(precision), target_device)
    use_amp = autocast_dtype is not None
    output_dir = Path(output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier(torch, distributed)
    rows = _read_manifest_rows(Path(manifest_path))
    if not rows:
        raise ValueError(f"Manifest contains no rows: {manifest_path}")
    rows, invalid_sample_count = _filter_valid_training_rows(rows)
    if not rows:
        raise ValueError(f"Manifest contains no valid rows after finite-value filtering: {manifest_path}")
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if max_train_samples > 0:
        train_rows = train_rows[: int(max_train_samples)]
    if max_val_samples > 0:
        val_rows = val_rows[: int(max_val_samples)]
    if not train_rows:
        raise ValueError("No training rows found in manifest.")
    if not _manifest_split_integrity(rows)["archive_disjoint"]:
        raise ValueError("Training manifest contains archives shared across splits.")
    if input_mode not in {"vision_route", "route_only"}:
        raise ValueError("input_mode must be vision_route or route_only")
    if int(spatial_pool_size) not in {1, 2, 4}:
        raise ValueError("spatial_pool_size must be 1, 2 or 4")

    config = VisionE2EModelConfig(
        model_size=str(model_size),
        architecture=str(architecture),
        image_size=int(image_size),
        future_steps=len(train_rows[0].get("future_waypoints_ego") or []),
        camera_count=len(train_rows[0].get("cameras") or DEFAULT_BENCH2DRIVE_CAMERAS),
        camera_pooling=str(camera_pooling),
        dropout=float(dropout),
        trajectory_modes=max(int(trajectory_modes), 1),
        trajectory_selection=str(trajectory_selection),
        trajectory_top_k=max(int(trajectory_top_k), 1),
        trajectory_temperature=max(float(trajectory_temperature), 1e-6),
        input_mode=str(input_mode),
        spatial_pool_size=int(spatial_pool_size),
    )
    loss_config = VisionE2ELossConfig(
        waypoint_weight=float(waypoint_loss_weight),
        selected_waypoint_weight=float(selected_waypoint_loss_weight),
        displacement_weight=float(displacement_loss_weight),
        endpoint_weight=float(endpoint_loss_weight),
        path_length_weight=float(path_length_loss_weight),
        control_weight=float(control_loss_weight),
        brake_weight=float(brake_loss_weight),
        brake_positive_weight=float(brake_positive_weight),
        risk_sample_weight=float(risk_sample_weight),
        lateral_loss_weight=float(lateral_loss_weight),
        turn_sample_weight=float(turn_sample_weight),
        turn_lateral_threshold_m=float(turn_lateral_threshold_m),
        mode_classification_weight=float(mode_classification_weight),
    )
    train_loader, val_loader = _build_dataloaders(
        train_rows=train_rows,
        val_rows=val_rows,
        batch_size=int(batch_size),
        image_size=int(image_size),
        num_workers=int(num_workers),
        prefetch_factor=int(prefetch_factor),
        distributed_rank=int(dist_ctx["rank"]),
        distributed_world_size=int(dist_ctx["world_size"]),
    )
    model = _build_vision_e2e_model(config)
    model = model.to(target_device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(dist_ctx["local_rank"])])
    data_parallel = False
    if (
        not distributed
        and
        bool(use_data_parallel)
        and target_device.type == "cuda"
        and torch.cuda.device_count() > 1
        and int(batch_size) >= torch.cuda.device_count()
    ):
        model = torch.nn.DataParallel(model)
        data_parallel = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and autocast_dtype == torch.float16))

    history = []
    best_metric = float("inf")
    best_path = output_dir / "vision_e2e_planner_best.pt"
    last_path = output_dir / "vision_e2e_planner_last.pt"
    started_at = time.time()
    for epoch in range(1, max(int(epochs), 1) + 1):
        _set_distributed_sampler_epoch(train_loader, epoch)
        train_metrics = _run_training_epoch(
            torch,
            model,
            train_loader,
            optimizer,
            target_device,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            loss_config=loss_config,
            nonfinite_check_interval=int(nonfinite_check_interval),
        )
        train_metrics = _reduce_epoch_metrics(torch, train_metrics, target_device, distributed)
        train_metrics = _add_brake_derived_metrics(train_metrics)
        eval_model = _unwrap_parallel_model(model) if distributed else model
        val_metrics = (
            _run_eval_epoch(torch, eval_model, val_loader, target_device, autocast_dtype=autocast_dtype, loss_config=loss_config)
            if val_loader is not None
            else {}
        )
        val_metrics = _reduce_epoch_metrics(torch, val_metrics, target_device, distributed) if val_metrics else {}
        val_metrics = _add_brake_derived_metrics(val_metrics) if val_metrics else {}
        epoch_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        if is_main_process:
            history.append(epoch_row)
        if verbose and is_main_process:
            _print_training_epoch(epoch, max(int(epochs), 1), train_metrics, val_metrics)
        score = _checkpoint_selection_score(val_metrics or train_metrics, str(selection_metric))
        if is_main_process:
            _save_planner_checkpoint(torch, last_path, model, config, epoch_row, loss_config=loss_config, data_parallel=data_parallel)
        if is_main_process and score <= best_metric:
            best_metric = score
            _save_planner_checkpoint(torch, best_path, model, config, epoch_row, loss_config=loss_config, data_parallel=data_parallel)

    from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance

    report = {
        "schema": BENCH2DRIVE_TRAINING_SCHEMA,
        "provenance": collect_runtime_provenance() if is_main_process else {},
        "manifest_sha256": build_artifact_entry(Path(manifest_path), "input", "manifest").sha256 if is_main_process else "",
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "input_mode": str(input_mode),
        "spatial_pool_size": int(spatial_pool_size),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "test_sample_count": len(test_rows),
        "split_protocol": "scenario-stratified archive-disjoint train/val/test",
        "split_integrity": _manifest_split_integrity(rows),
        "epochs": max(int(epochs), 1),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "precision": str(precision),
        "allow_tf32": bool(allow_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "nonfinite_check_interval": int(nonfinite_check_interval),
        "model_size": str(model_size),
        "architecture": str(architecture),
        "camera_pooling": str(camera_pooling),
        "dropout": float(dropout),
        "trajectory_modes": max(int(trajectory_modes), 1),
        "trajectory_selection": str(trajectory_selection),
        "trajectory_top_k": max(int(trajectory_top_k), 1),
        "trajectory_temperature": max(float(trajectory_temperature), 1e-6),
        "loss_config": dict(loss_config.__dict__),
        "selection_metric": str(selection_metric),
        "image_size": int(image_size),
        "device": str(target_device),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "data_parallel": data_parallel,
        "distributed": distributed,
        "distributed_world_size": int(dist_ctx["world_size"]),
        "rank": int(dist_ctx["rank"]),
        "is_main_process": is_main_process,
        "invalid_sample_count": int(invalid_sample_count),
        "history": history,
        "runtime_s": round(time.time() - started_at, 3),
    }
    if is_main_process:
        _write_training_outputs(report, output_dir)
    _distributed_barrier(torch, distributed)
    _destroy_distributed_training(torch, distributed)
    return report


def evaluate_vision_e2e_planner(
    manifest_path: Path = DEFAULT_BENCH2DRIVE_MANIFEST,
    checkpoint_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "vision_e2e_planner_best.pt",
    output_dir: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "eval",
    *,
    split: str = "test",
    batch_size: int = 32,
    image_size: int = 160,
    max_samples: int = 0,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    device: str = "",
) -> Dict[str, Any]:
    torch, _, _ = _require_torch_stack(require_pillow=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_manifest_rows(Path(manifest_path))
    rows, invalid_sample_count = _filter_valid_training_rows(rows)
    if split != "all":
        rows = [row for row in rows if str(row.get("split")) == split]
    if max_samples > 0:
        rows = rows[: int(max_samples)]
    if not rows:
        raise ValueError(f"No rows found for split={split!r}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = VisionE2EModelConfig(**dict(checkpoint.get("model_config") or {}))
    loss_config = VisionE2ELossConfig(**dict(checkpoint.get("loss_config") or {}))
    model = _build_vision_e2e_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_device = _select_torch_device(torch, device)
    model = model.to(target_device)
    dataset = _Bench2DriveVisionDataset(rows, image_size=int(image_size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=target_device.type == "cuda",
        **_dataloader_worker_kwargs(int(num_workers), int(prefetch_factor)),
    )
    metrics, predictions = _run_prediction_epoch(torch, model, loader, target_device, loss_config=loss_config)
    scenario_family_metrics = _scenario_family_metrics_from_predictions(predictions)
    scenario_counts = Counter(str(row.get("scenario_family") or "unknown") for row in rows)
    report = {
        "schema": BENCH2DRIVE_EVAL_SCHEMA,
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "split": split,
        "sample_count": len(rows),
        "batch_size": int(batch_size),
        "image_size": int(image_size),
        "num_workers": int(num_workers),
        "prefetch_factor": int(prefetch_factor),
        "invalid_sample_count": int(invalid_sample_count),
        "device": str(target_device),
        "metrics": metrics,
        "scenario_family_metrics": scenario_family_metrics,
        "taxonomy_coverage": build_taxonomy_coverage("bench2drive", scenario_counts),
        "uncertainty": _bootstrap_prediction_metrics(predictions, seed=7, replicates=1000),
        "trajectory_calibration_comparison": _trajectory_calibration_comparison(
            predictions,
            seed=7,
            replicates=2000,
        ),
    }
    _write_eval_outputs(report, predictions, output_dir)
    return report


def compare_vision_e2e_prediction_sets(
    baseline_predictions_path: Path,
    candidate_predictions_path: Path,
    output_dir: Path,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    seed: int = 7,
    bootstrap_replicates: int = 5000,
    allow_case_intersection: bool = False,
) -> Dict[str, Any]:
    """Compare two prediction JSONL files with clip-level paired bootstrap."""
    import numpy as np

    baseline_path = Path(baseline_predictions_path)
    candidate_path = Path(candidate_predictions_path)
    baseline_rows = _read_prediction_rows(baseline_path)
    candidate_rows = _read_prediction_rows(candidate_path)
    baseline_map = _index_prediction_rows(baseline_rows)
    candidate_map = _index_prediction_rows(candidate_rows)
    baseline_ids = set(baseline_map)
    candidate_ids = set(candidate_map)
    common_ids = sorted(baseline_ids & candidate_ids)
    baseline_only = sorted(baseline_ids - candidate_ids)
    candidate_only = sorted(candidate_ids - baseline_ids)
    if not common_ids:
        raise ValueError("Prediction files do not share any case IDs.")
    if (baseline_only or candidate_only) and not allow_case_intersection:
        raise ValueError(
            "Prediction files use different case sets; pass allow_case_intersection=True "
            "to compare only their intersection."
        )

    paired_rows: List[Dict[str, Any]] = []
    for case_id in common_ids:
        baseline = baseline_map[case_id]
        candidate = candidate_map[case_id]
        baseline_target = np.asarray(baseline.get("target_future_waypoints_ego") or [], dtype=float)
        candidate_target = np.asarray(candidate.get("target_future_waypoints_ego") or [], dtype=float)
        if baseline_target.shape != candidate_target.shape or not np.allclose(baseline_target, candidate_target):
            raise ValueError(f"Targets differ for paired case_id: {case_id}")
        baseline_pred = np.asarray(baseline.get("predicted_future_waypoints_ego") or [], dtype=float)
        candidate_pred = np.asarray(candidate.get("predicted_future_waypoints_ego") or [], dtype=float)
        if baseline_pred.shape != baseline_target.shape or candidate_pred.shape != candidate_target.shape:
            raise ValueError(f"Prediction horizon differs for paired case_id: {case_id}")
        baseline_distance = np.linalg.norm(baseline_pred - baseline_target, axis=-1)
        candidate_distance = np.linalg.norm(candidate_pred - candidate_target, axis=-1)
        baseline_lateral = np.abs(baseline_pred[:, 0] - baseline_target[:, 0])
        candidate_lateral = np.abs(candidate_pred[:, 0] - candidate_target[:, 0])
        target_path_length = _waypoint_path_length(baseline_target)
        baseline_path_error = abs(_waypoint_path_length(baseline_pred) / max(target_path_length, 1e-6) - 1.0)
        candidate_path_error = abs(_waypoint_path_length(candidate_pred) / max(target_path_length, 1e-6) - 1.0)
        baseline_brake = bool(baseline.get("predicted_should_brake"))
        candidate_brake = bool(candidate.get("predicted_should_brake"))
        target_brake = bool(baseline.get("target_should_brake"))
        paired_rows.append(
            {
                "case_id": case_id,
                "clip_name": str(baseline.get("clip_name") or candidate.get("clip_name") or case_id.rsplit(":", 1)[0]),
                "scenario_family": str(baseline.get("scenario_family") or candidate.get("scenario_family") or "unknown"),
                "baseline_ade_m": float(np.mean(baseline_distance)),
                "candidate_ade_m": float(np.mean(candidate_distance)),
                "baseline_fde_m": float(baseline_distance[-1]),
                "candidate_fde_m": float(candidate_distance[-1]),
                "baseline_lateral_mae_m": float(np.mean(baseline_lateral)),
                "candidate_lateral_mae_m": float(np.mean(candidate_lateral)),
                "baseline_path_length_error": float(baseline_path_error),
                "candidate_path_length_error": float(candidate_path_error),
                "target_should_brake": target_brake,
                "baseline_predicted_should_brake": baseline_brake,
                "candidate_predicted_should_brake": candidate_brake,
            }
        )

    rng = np.random.default_rng(int(seed))
    clusters: Dict[str, List[Mapping[str, Any]]] = {}
    for row in paired_rows:
        clusters.setdefault(str(row["clip_name"]), []).append(row)
    cluster_rows = list(clusters.values())
    cluster_count = len(cluster_rows)
    replicate_count = max(int(bootstrap_replicates), 1)
    sampled_cluster_indices = rng.integers(0, cluster_count, size=(replicate_count, cluster_count))

    def _continuous_metric(row: Mapping[str, Any], prefix: str, metric: str) -> float:
        return float(row[f"{prefix}_{metric}"])

    metric_specs = (
        ("ade_m", "ADE", "lower"),
        ("fde_m", "FDE", "lower"),
        ("lateral_mae_m", "Lateral MAE", "lower"),
        ("path_length_error", "Path-length error", "lower"),
    )
    metric_rows: List[Dict[str, Any]] = []
    for metric, display_name, direction in metric_specs:
        baseline_cluster_sums = np.asarray(
            [sum(_continuous_metric(row, "baseline", metric) for row in rows) for rows in cluster_rows],
            dtype=float,
        )
        candidate_cluster_sums = np.asarray(
            [sum(_continuous_metric(row, "candidate", metric) for row in rows) for rows in cluster_rows],
            dtype=float,
        )
        cluster_sizes = np.asarray([len(rows) for rows in cluster_rows], dtype=float)
        baseline_samples = np.sum(baseline_cluster_sums[sampled_cluster_indices], axis=1) / np.sum(
            cluster_sizes[sampled_cluster_indices], axis=1
        )
        candidate_samples = np.sum(candidate_cluster_sums[sampled_cluster_indices], axis=1) / np.sum(
            cluster_sizes[sampled_cluster_indices], axis=1
        )
        baseline_values = np.asarray([row[f"baseline_{metric}"] for row in paired_rows], dtype=float)
        candidate_values = np.asarray([row[f"candidate_{metric}"] for row in paired_rows], dtype=float)
        improvements = baseline_samples - candidate_samples if direction == "lower" else candidate_samples - baseline_samples
        point_improvement = float(np.mean(baseline_values - candidate_values)) if direction == "lower" else float(np.mean(candidate_values - baseline_values))
        ci_low, ci_high = np.percentile(improvements, [2.5, 97.5])
        baseline_mean = float(np.mean(baseline_values))
        candidate_mean = float(np.mean(candidate_values))
        metric_rows.append(
            {
                "metric": metric,
                "display_name": display_name,
                "direction": direction,
                "case_count": len(paired_rows),
                "cluster_count": cluster_count,
                "baseline_mean": round(baseline_mean, 6),
                "candidate_mean": round(candidate_mean, 6),
                "candidate_minus_baseline": round(candidate_mean - baseline_mean, 6),
                "oriented_improvement": round(point_improvement, 6),
                "improvement_ci95_low": round(float(ci_low), 6),
                "improvement_ci95_high": round(float(ci_high), 6),
                "candidate_win_rate": round(
                    float(np.mean((baseline_values - candidate_values if direction == "lower" else candidate_values - baseline_values) > 0.0)),
                    6,
                ),
                "bootstrap_probability_of_improvement": round(float(np.mean(improvements > 0.0)), 6),
                "ci95_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
            }
        )

    baseline_tp, baseline_fp, baseline_fn = _cluster_brake_counts(cluster_rows, "baseline")
    candidate_tp, candidate_fp, candidate_fn = _cluster_brake_counts(cluster_rows, "candidate")
    baseline_f1_samples = _f1_from_counts(
        np.sum(baseline_tp[sampled_cluster_indices], axis=1),
        np.sum(baseline_fp[sampled_cluster_indices], axis=1),
        np.sum(baseline_fn[sampled_cluster_indices], axis=1),
    )
    candidate_f1_samples = _f1_from_counts(
        np.sum(candidate_tp[sampled_cluster_indices], axis=1),
        np.sum(candidate_fp[sampled_cluster_indices], axis=1),
        np.sum(candidate_fn[sampled_cluster_indices], axis=1),
    )
    baseline_f1 = float(_f1_from_counts(np.sum(baseline_tp), np.sum(baseline_fp), np.sum(baseline_fn)))
    candidate_f1 = float(_f1_from_counts(np.sum(candidate_tp), np.sum(candidate_fp), np.sum(candidate_fn)))
    f1_improvements = candidate_f1_samples - baseline_f1_samples
    f1_low, f1_high = np.percentile(f1_improvements, [2.5, 97.5])
    metric_rows.append(
        {
            "metric": "brake_f1",
            "display_name": "Brake F1",
            "direction": "higher",
            "case_count": len(paired_rows),
            "cluster_count": cluster_count,
            "baseline_mean": round(baseline_f1, 6),
            "candidate_mean": round(candidate_f1, 6),
            "candidate_minus_baseline": round(candidate_f1 - baseline_f1, 6),
            "oriented_improvement": round(candidate_f1 - baseline_f1, 6),
            "improvement_ci95_low": round(float(f1_low), 6),
            "improvement_ci95_high": round(float(f1_high), 6),
            "candidate_win_rate": round(float(np.mean(f1_improvements > 0.0)), 6),
            "bootstrap_probability_of_improvement": round(float(np.mean(f1_improvements > 0.0)), 6),
            "ci95_excludes_zero": bool(f1_low > 0.0 or f1_high < 0.0),
        }
    )
    payload = {
        "schema": BENCH2DRIVE_PREDICTION_COMPARISON_SCHEMA,
        "method": "paired clip-level percentile bootstrap",
        "baseline": {"label": str(baseline_label), "predictions_path": str(baseline_path)},
        "candidate": {"label": str(candidate_label), "predictions_path": str(candidate_path)},
        "case_alignment": {
            "paired_case_count": len(common_ids),
            "identical_case_sets": not baseline_only and not candidate_only,
            "baseline_only_case_ids": baseline_only,
            "candidate_only_case_ids": candidate_only,
        },
        "bootstrap": {"seed": int(seed), "replicates": replicate_count, "cluster_key": "clip_name"},
        "metrics": metric_rows,
        "paired_cases": paired_rows,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_prediction_comparison_csv(metric_rows, output_dir / "prediction_comparison.csv")
    _write_prediction_comparison_csv(paired_rows, output_dir / "prediction_paired_cases.csv")
    (output_dir / "prediction_comparison.md").write_text(_render_prediction_comparison_markdown(payload), encoding="utf-8")
    _render_prediction_comparison_figure(metric_rows, output_dir / "prediction_comparison.png")
    return payload


def _index_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in indexed:
            raise ValueError(f"Prediction rows contain an invalid or duplicate case_id: {case_id}")
        indexed[case_id] = dict(row)
    return indexed


def _waypoint_path_length(points: Any) -> float:
    import numpy as np

    values = np.asarray(points, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def _cluster_brake_counts(cluster_rows: Sequence[Sequence[Mapping[str, Any]]], prefix: str) -> Tuple[Any, Any, Any]:
    import numpy as np

    tp, fp, fn = [], [], []
    for rows in cluster_rows:
        true_positive = false_positive = false_negative = 0
        for row in rows:
            target = bool(row.get("target_should_brake"))
            predicted = bool(row.get(f"{prefix}_predicted_should_brake"))
            true_positive += int(target and predicted)
            false_positive += int(not target and predicted)
            false_negative += int(target and not predicted)
        tp.append(true_positive)
        fp.append(false_positive)
        fn.append(false_negative)
    return np.asarray(tp, dtype=float), np.asarray(fp, dtype=float), np.asarray(fn, dtype=float)


def _f1_from_counts(tp: Any, fp: Any, fn: Any) -> Any:
    import numpy as np

    return 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)


def _write_prediction_comparison_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        Path(output_path).write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def _render_prediction_comparison_markdown(payload: Mapping[str, Any]) -> str:
    alignment = dict(payload.get("case_alignment") or {})
    lines = [
        "# Bench2Drive Vision Prediction Comparison",
        "",
        f"- Baseline: `{dict(payload.get('baseline') or {}).get('label', '')}`",
        f"- Candidate: `{dict(payload.get('candidate') or {}).get('label', '')}`",
        f"- Paired cases: `{alignment.get('paired_case_count', 0)}`",
        "- Positive improvement favors the candidate.",
        "",
        "| Metric | Baseline | Candidate | Improvement | 95% CI | Win Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in list(payload.get("metrics") or []):
        lines.append(
            "| {0} | {1:.4f} | {2:.4f} | {3:.4f} | [{4:.4f}, {5:.4f}] | {6:.3f} |".format(
                row.get("display_name", row.get("metric", "")),
                float(row.get("baseline_mean") or 0.0),
                float(row.get("candidate_mean") or 0.0),
                float(row.get("oriented_improvement") or 0.0),
                float(row.get("improvement_ci95_low") or 0.0),
                float(row.get("improvement_ci95_high") or 0.0),
                float(row.get("candidate_win_rate") or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def _render_prediction_comparison_figure(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    labels = [str(row.get("display_name") or row.get("metric") or "") for row in rows]
    values = [float(row.get("oriented_improvement") or 0.0) for row in rows]
    low = [float(row.get("improvement_ci95_low") or 0.0) for row in rows]
    high = [float(row.get("improvement_ci95_high") or 0.0) for row in rows]
    errors = [[max(value - bound, 0.0) for value, bound in zip(values, low)], [max(bound - value, 0.0) for value, bound in zip(values, high)]]
    colors = ["#26734d" if value >= 0.0 else "#b24a3b" for value in values]
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=180)
    positions = list(range(len(rows)))
    ax.barh(positions, values, xerr=errors, color=colors, alpha=0.9, capsize=3)
    ax.axvline(0.0, color="#222222", linewidth=0.9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate improvement; positive is better")
    ax.set_title("Bench2Drive paired open-loop comparison")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def calibrate_trajectory_selection(
    predictions_path: Path,
    evaluation_report_path: Path,
    checkpoint_path: Path,
    output_checkpoint_path: Path,
    output_report_path: Path,
    *,
    temperatures: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    selections: Sequence[str] = ("argmax", "expected", "topk_expected"),
    top_k: int = 2,
) -> Dict[str, Any]:
    evaluation_report = json.loads(Path(evaluation_report_path).read_text(encoding="utf-8"))
    if str(evaluation_report.get("split") or "") != "val":
        raise ValueError("Trajectory calibration requires predictions from the validation split.")
    predictions = _read_prediction_rows(Path(predictions_path))
    if not predictions:
        raise ValueError(f"No validation predictions found: {predictions_path}")
    if any(not row.get("predicted_future_modes_ego") or not row.get("mode_logits") for row in predictions):
        raise ValueError("Validation predictions must include trajectory modes and mode logits.")

    search_rows = []
    for selection in selections:
        candidate_temperatures = [1.0] if selection == "argmax" else list(temperatures)
        for temperature in candidate_temperatures:
            selected_predictions = [
                _select_trajectory_from_prediction(
                    row,
                    selection=str(selection),
                    temperature=float(temperature),
                    top_k=int(top_k),
                )
                for row in predictions
            ]
            metrics = _trajectory_metrics_from_pairs(
                selected_predictions,
                [row["target_future_waypoints_ego"] for row in predictions],
            )
            search_rows.append(
                {
                    "selection": str(selection),
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    **metrics,
                }
            )
    selected = min(search_rows, key=lambda row: (float(row["ade_m"]), float(row["fde_m"])))
    mode_calibrator_config, mode_calibrator_report = _fit_mode_calibrator(
        predictions,
        selection=str(selected["selection"]),
        temperature=float(selected["temperature"]),
        top_k=int(selected["top_k"]),
    )

    torch, _, _ = _require_torch_stack(require_pillow=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = dict(checkpoint.get("model_config") or {})
    model_config.update(
        {
            "trajectory_selection": selected["selection"],
            "trajectory_temperature": float(selected["temperature"]),
            "trajectory_top_k": int(selected["top_k"]),
            "trajectory_mode_calibrator": mode_calibrator_config,
        }
    )
    checkpoint["model_config"] = model_config
    checkpoint["trajectory_calibration"] = {
        "schema": BENCH2DRIVE_CALIBRATION_SCHEMA,
        "validation_evaluation_report": str(evaluation_report_path),
        "selected": dict(selected),
        "mode_calibrator": mode_calibrator_report,
    }
    output_checkpoint_path = Path(output_checkpoint_path)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_checkpoint_path))

    report = {
        "schema": BENCH2DRIVE_CALIBRATION_SCHEMA,
        "source_checkpoint": str(checkpoint_path),
        "output_checkpoint": str(output_checkpoint_path),
        "validation_evaluation_report": str(evaluation_report_path),
        "validation_predictions": str(predictions_path),
        "validation_sample_count": len(predictions),
        "selection_metric": "ADE with FDE tie-break",
        "selected": dict(selected),
        "search": sorted(search_rows, key=lambda row: (float(row["ade_m"]), float(row["fde_m"]))),
        "mode_calibrator": mode_calibrator_report,
        "calibrated_model_config": model_config,
    }
    output_report_path = Path(output_report_path)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def diagnose_vision_e2e_predictions(
    predictions_path: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "eval" / "predictions.jsonl",
    output_dir: Path = DEFAULT_BENCH2DRIVE_OUTPUT / "diagnostics",
    *,
    evaluation_report_path: Optional[Path] = None,
    brake_threshold: float = 0.5,
) -> Dict[str, Any]:
    predictions = _read_prediction_rows(Path(predictions_path))
    if not predictions:
        raise ValueError(f"No prediction rows found: {predictions_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_prediction_diagnostic_row(row, brake_threshold=float(brake_threshold)) for row in predictions]
    aggregate = _aggregate_prediction_diagnostics(rows)
    by_family = {
        family: _aggregate_prediction_diagnostics(family_rows)
        for family, family_rows in sorted(_group_rows_by_key(rows, "scenario_family").items())
    }
    readiness = _planner_readiness_assessment(aggregate)
    evaluation_metrics = {}
    if evaluation_report_path is not None and Path(evaluation_report_path).exists():
        evaluation_metrics = dict(json.loads(Path(evaluation_report_path).read_text(encoding="utf-8")).get("metrics") or {})
    report = {
        "schema": BENCH2DRIVE_DIAGNOSTIC_SCHEMA,
        "predictions_path": str(predictions_path),
        "evaluation_report_path": str(evaluation_report_path or ""),
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "brake_threshold": float(brake_threshold),
        "evaluation_metrics": evaluation_metrics,
        "aggregate": aggregate,
        "scenario_family_diagnostics": by_family,
        "readiness": readiness,
    }
    _write_prediction_diagnostic_outputs(report, rows, output_dir)
    return report


def build_bench2drive_vision_tensor_cache(
    manifest_path: Path = DEFAULT_BENCH2DRIVE_MANIFEST,
    output_manifest_path: Path = DEFAULT_BENCH2DRIVE_TENSOR_MANIFEST,
    cache_dir: Path = DEFAULT_BENCH2DRIVE_TENSOR_CACHE_ROOT,
    *,
    image_size: int = 160,
    max_rows: int = 0,
    num_workers: int = 0,
    chunk_rows: int = 512,
    verbose: bool = False,
) -> Dict[str, Any]:
    import numpy as np

    rows = _read_manifest_rows(Path(manifest_path))
    rows, invalid_sample_count = _filter_valid_training_rows(rows)
    if max_rows > 0:
        rows = rows[: int(max_rows)]
    if not rows:
        raise ValueError(f"No valid rows found in manifest: {manifest_path}")

    output_manifest_path = Path(output_manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(image_size)
    camera_names = list(rows[0].get("camera_names") or rows[0].get("cameras") or DEFAULT_BENCH2DRIVE_CAMERAS)
    tensor_path = cache_dir / f"images_{image_size}px_uint8.npy"
    shape = (len(rows), len(camera_names), 3, image_size, image_size)
    tensor = np.lib.format.open_memmap(str(tensor_path), mode="w+", dtype=np.uint8, shape=shape)
    tensor.flush()

    started_at = time.time()
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        for row_idx, source_row in enumerate(rows):
            row = dict(source_row)
            row_camera_names = list(row.get("camera_names") or row.get("cameras") or camera_names)
            if row_camera_names != camera_names:
                raise ValueError(f"Inconsistent camera set at row {row_idx}: {row_camera_names}")
            row["tensor_cache_path"] = str(tensor_path)
            row["tensor_cache_index"] = int(row_idx)
            row["tensor_cache_image_size"] = image_size
            row["tensor_cache_dtype"] = "uint8"
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    written = 0
    worker_count = max(int(num_workers), 0)
    if worker_count <= 1:
        for row_idx, row in enumerate(rows):
            tensor[row_idx] = _load_multicamera_uint8_array(row, camera_names=camera_names, image_size=image_size)
            written += 1
            if verbose and (written == len(rows) or written % 1000 == 0):
                elapsed = max(time.time() - started_at, 1e-6)
                print(
                    f"[tensor-cache] {written}/{len(rows)} rows "
                    f"({written / elapsed:.1f} rows/s)",
                    flush=True,
                )
    else:
        tasks = []
        chunk_size = max(int(chunk_rows), 1)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for start_idx in range(0, len(rows), chunk_size):
                chunk = rows[start_idx : start_idx + chunk_size]
                tasks.append(
                    executor.submit(
                        _write_tensor_cache_chunk,
                        str(tensor_path),
                        chunk,
                        list(camera_names),
                        image_size,
                        start_idx,
                    )
                )
            for future in as_completed(tasks):
                written += int(future.result())
                if verbose:
                    elapsed = max(time.time() - started_at, 1e-6)
                    print(
                        f"[tensor-cache] {written}/{len(rows)} rows "
                        f"({written / elapsed:.1f} rows/s)",
                        flush=True,
                    )
    tensor.flush()

    split_counts = Counter(str(row.get("split") or "") for row in rows)
    scenario_counts = Counter(str(row.get("scenario_family") or "") for row in rows)
    metadata = {
        "schema": BENCH2DRIVE_TENSOR_CACHE_SCHEMA,
        "source_manifest_path": str(manifest_path),
        "manifest_path": str(output_manifest_path),
        "tensor_cache_path": str(tensor_path),
        "row_count": len(rows),
        "invalid_sample_count": int(invalid_sample_count),
        "split_counts": dict(sorted(split_counts.items())),
        "scenario_family_counts": dict(sorted(scenario_counts.items())),
        "split_integrity": _manifest_split_integrity(rows),
        "taxonomy_coverage": build_taxonomy_coverage("bench2drive", scenario_counts),
        "image_size": image_size,
        "dtype": "uint8",
        "shape": list(shape),
        "cameras": camera_names,
        "runtime_s": round(time.time() - started_at, 3),
        "num_workers": worker_count,
        "chunk_rows": max(int(chunk_rows), 1),
        "tensor_cache_size_gb": round(tensor_path.stat().st_size / (1024**3), 3) if tensor_path.exists() else 0.0,
    }
    metadata_path = output_manifest_path.with_suffix(output_manifest_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _manifest_split_integrity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    archive_splits: Dict[str, set[str]] = {}
    for row in rows:
        archive = str(row.get("archive") or row.get("clip_name") or "")
        archive_splits.setdefault(archive, set()).add(str(row.get("split") or ""))
    overlap = {archive: sorted(splits) for archive, splits in archive_splits.items() if len(splits) > 1}
    return {
        "archive_disjoint": not overlap,
        "archive_count": len(archive_splits),
        "overlapping_archives": overlap,
    }


def _inspect_bench2drive_archive(archive_path: Path) -> Dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as tar:
        names = tar.getnames()
    camera_dirs = sorted(
        {
            fragment.split("/", 1)[0]
            for name in names
            if "/camera/" in name
            for fragment in [name.split("/camera/", 1)[1]]
            if "/" in fragment
        }
    )
    anno_frames = _frame_ids_from_names(names, "/anno/", ".json.gz")
    rgb_front_frames = _frame_ids_from_names(names, "/camera/rgb_front/", ".jpg")
    return {
        "archive": str(archive_path),
        "clip_name": archive_path.name.replace(".tar.gz", ""),
        "scenario_family": _scenario_family_from_name(archive_path.name.replace(".tar.gz", "")),
        "size_gb": round(archive_path.stat().st_size / (1024**3), 3),
        "camera_dirs": camera_dirs,
        "annotation_frame_count": len(anno_frames),
        "rgb_front_frame_count": len(rgb_front_frames),
        "first_frame": min(anno_frames) if anno_frames else None,
        "last_frame": max(anno_frames) if anno_frames else None,
    }


def _build_manifest_rows_for_archive(
    archive_path: Path,
    *,
    split: str,
    frame_stride: int,
    future_steps: int,
    future_frame_stride: int,
    cameras: Sequence[str],
    cache_root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotations: Dict[int, Dict[str, Any]] = {}
    camera_by_frame: Dict[str, Dict[int, str]] = {camera: {} for camera in cameras}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            frame = _frame_id_from_path(member.name)
            if frame is None:
                continue
            if "/anno/" in member.name and member.name.endswith(".json.gz"):
                annotations[frame] = _read_json_gz_from_tar(tar, member)
                continue
            for camera in cameras:
                if f"/camera/{camera}/" in member.name and member.name.endswith(".jpg"):
                    camera_by_frame[camera][frame] = member.name
                    break

    frame_ids = sorted(annotations)
    selected = frame_ids[::frame_stride]
    rows = []
    clip_name = archive_path.name.replace(".tar.gz", "")
    scenario_family = _scenario_family_from_name(clip_name)
    cache_jobs: Dict[str, Path] = {}
    for frame_id in selected:
        future_ids = [frame_id + future_frame_stride * step for step in range(1, future_steps + 1)]
        if not all(future_id in annotations for future_id in future_ids):
            continue
        if not all(frame_id in camera_by_frame[camera] for camera in cameras):
            continue
        current = annotations[frame_id]
        future_annos = [annotations[future_id] for future_id in future_ids]
        route_features = _build_route_features(current)
        camera_members = {camera: camera_by_frame[camera][frame_id] for camera in cameras}
        camera_cache_paths = {}
        if cache_root is not None:
            for camera, member_name in camera_members.items():
                output_path = _camera_cache_path(
                    cache_root=cache_root,
                    clip_name=clip_name,
                    camera_name=camera,
                    frame_id=frame_id,
                    member_name=member_name,
                )
                camera_cache_paths[camera] = str(output_path)
                if not output_path.exists():
                    cache_jobs[member_name] = output_path
        row = {
            "schema": BENCH2DRIVE_MANIFEST_SCHEMA,
            "archive": str(archive_path),
            "clip_name": clip_name,
            "scenario_family": scenario_family,
            "split": split,
            "frame_id": int(frame_id),
            "camera_names": list(cameras),
            "cameras": camera_members,
            "camera_cache_paths": camera_cache_paths,
            "ego_state": _ego_state_from_annotation(current),
            "route_features": route_features,
            "control": {
                "steer": _float_field(current, "steer"),
                "throttle": _float_field(current, "throttle"),
                "brake": _float_field(current, "brake"),
            },
            "should_brake": bool(current.get("should_brake") or current.get("only_ap_brake") or False),
            "future_waypoints_ego": _future_waypoints_in_ego(current, future_annos),
            "future_frame_ids": future_ids,
            "object_count": len(current.get("bounding_boxes") or []),
            "weather": _compact_weather(current.get("weather") or {}),
        }
        rows.append(row)

    if cache_jobs:
        _materialize_camera_cache(
            archive_path=archive_path,
            cache_jobs=cache_jobs,
        )
    summary = {
        "archive": str(archive_path),
        "clip_name": archive_path.name.replace(".tar.gz", ""),
        "split": split,
        "row_count": len(rows),
        "scenario_family": _scenario_family_from_name(archive_path.name.replace(".tar.gz", "")),
    }
    return rows, summary


def _frame_ids_from_names(names: Sequence[str], marker: str, suffix: str) -> List[int]:
    frames = []
    for name in names:
        if marker not in name or not name.endswith(suffix):
            continue
        frame_id = _frame_id_from_path(name)
        if frame_id is not None:
            frames.append(frame_id)
    return sorted(set(frames))


def _frame_id_from_path(name: str) -> Optional[int]:
    stem = Path(name).name
    if stem.endswith(".json.gz"):
        stem = stem[: -len(".json.gz")]
    else:
        stem = Path(stem).stem
    try:
        return int(stem)
    except ValueError:
        return None


def _read_json_gz_from_tar(tar: tarfile.TarFile, member: tarfile.TarInfo) -> Dict[str, Any]:
    stream = tar.extractfile(member)
    if stream is None:
        raise FileNotFoundError(member.name)
    return json.loads(gzip.decompress(stream.read()).decode("utf-8"))


def _materialize_camera_cache(*, archive_path: Path, cache_jobs: Mapping[str, Path]) -> None:
    pending = dict(cache_jobs)
    if not pending:
        return
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if member.name not in pending:
                continue
            output_path = pending.pop(member.name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_path.exists():
                stream = tar.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(member.name)
                output_path.write_bytes(stream.read())
            if not pending:
                break
    if pending:
        missing = ", ".join(sorted(pending)[:5])
        raise FileNotFoundError(f"Missing camera cache members in {archive_path}: {missing}")


def _camera_cache_path(
    *,
    cache_root: Path,
    clip_name: str,
    camera_name: str,
    frame_id: int,
    member_name: str,
) -> Path:
    suffix = Path(member_name).suffix or ".jpg"
    return Path(cache_root) / clip_name / camera_name / f"{int(frame_id):05d}{suffix}"


def _scenario_family_from_name(name: str) -> str:
    text = name.replace(".tar", "")
    if "_Town" in text:
        return text.split("_Town", 1)[0]
    return text.split("_Route", 1)[0]


def _ego_state_from_annotation(annotation: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "x": _float_field(annotation, "x"),
        "y": _float_field(annotation, "y"),
        "theta": _float_field(annotation, "theta"),
        "speed": _float_field(annotation, "speed"),
    }


def _build_route_features(annotation: Mapping[str, Any]) -> List[float]:
    x = _float_field(annotation, "x")
    y = _float_field(annotation, "y")
    theta = _float_field(annotation, "theta")
    target = _point_to_ego(
        x0=x,
        y0=y,
        theta=theta,
        x1=_float_field(annotation, "x_target", x),
        y1=_float_field(annotation, "y_target", y),
    )
    near = _point_to_ego(
        x0=x,
        y0=y,
        theta=theta,
        x1=_float_field(annotation, "x_command_near", x),
        y1=_float_field(annotation, "y_command_near", y),
    )
    return [
        target[0] / 50.0,
        target[1] / 50.0,
        near[0] / 50.0,
        near[1] / 50.0,
        _float_field(annotation, "speed") / 20.0,
        _float_field(annotation, "next_command") / 6.0,
        _float_field(annotation, "command_near") / 6.0,
        _float_field(annotation, "command_far") / 6.0,
    ]


def _future_waypoints_in_ego(current: Mapping[str, Any], future_annos: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    x0 = _float_field(current, "x")
    y0 = _float_field(current, "y")
    theta = _float_field(current, "theta")
    return [
        list(_point_to_ego(x0=x0, y0=y0, theta=theta, x1=_float_field(item, "x"), y1=_float_field(item, "y")))
        for item in future_annos
    ]


def _point_to_ego(*, x0: float, y0: float, theta: float, x1: float, y1: float) -> Tuple[float, float]:
    dx = x1 - x0
    dy = y1 - y0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


def _compact_weather(weather: Mapping[str, Any]) -> Dict[str, float]:
    keys = ["cloudiness", "precipitation", "fog_density", "wetness", "sun_altitude_angle"]
    return {key: _float_field(weather, key) for key in keys}


def _float_field(payload: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _read_manifest_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _filter_valid_training_rows(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    valid = []
    invalid_count = 0
    for row in rows:
        row_dict = dict(row)
        if _is_valid_training_row(row_dict):
            valid.append(row_dict)
        else:
            invalid_count += 1
    return valid, invalid_count


def _is_valid_training_row(row: Mapping[str, Any]) -> bool:
    values: List[Any] = []
    values.extend(row.get("route_features") or [])
    values.extend((row.get("control") or {}).values())
    for waypoint in row.get("future_waypoints_ego") or []:
        values.extend(waypoint)
    return bool(values) and all(_is_finite_number(value) for value in values)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _load_multicamera_uint8_array(row: Mapping[str, Any], *, camera_names: Sequence[str], image_size: int) -> Any:
    import numpy as np

    camera_members = dict(row["cameras"])
    camera_cache_paths = dict(row.get("camera_cache_paths") or {})
    images = []
    tar_handle = None
    try:
        for camera_name in camera_names:
            camera_key = str(camera_name)
            cache_path = Path(str(camera_cache_paths.get(camera_key) or ""))
            if cache_path.exists():
                images.append(_load_image_file_uint8_array(cache_path, image_size=image_size))
                continue
            if tar_handle is None:
                tar_handle = tarfile.open(str(row["archive"]), "r:gz")
            member_name = camera_members[camera_key]
            stream = tar_handle.extractfile(member_name)
            if stream is None:
                raise FileNotFoundError(member_name)
            images.append(_load_image_bytes_uint8_array(stream.read(), image_size=image_size))
    finally:
        if tar_handle is not None:
            tar_handle.close()
    return np.stack(images, axis=0)


def _write_tensor_cache_chunk(
    tensor_path: str,
    rows: Sequence[Mapping[str, Any]],
    camera_names: Sequence[str],
    image_size: int,
    start_idx: int,
) -> int:
    import numpy as np

    tensor = np.load(tensor_path, mmap_mode="r+")
    for offset, row in enumerate(rows):
        tensor[int(start_idx) + offset] = _load_multicamera_uint8_array(
            row,
            camera_names=camera_names,
            image_size=int(image_size),
        )
    tensor.flush()
    return len(rows)


def _load_image_file_uint8_array(path: Path, *, image_size: int) -> Any:
    return _load_image_bytes_uint8_array(Path(path).read_bytes(), image_size=image_size)


def _load_image_bytes_uint8_array(payload: bytes, *, image_size: int) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(payload)).convert("RGB").resize((int(image_size), int(image_size)))
    array = np.asarray(image, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))


def _require_torch_stack(*, require_pillow: bool = False) -> Tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover - depends on local training env
        raise RuntimeError(
            "PyTorch is required for Bench2Drive vision training/evaluation. "
            "Install it in the training environment before running this command."
        ) from exc
    if require_pillow:
        try:
            import PIL.Image  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on local training env
            raise RuntimeError("Pillow is required for image loading.") from exc
    return torch, nn, functional


def _set_torch_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_precision(torch: Any, *, allow_tf32: bool, cudnn_benchmark: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def _resolve_autocast_dtype(torch: Any, precision: str, device: Any) -> Any:
    normalized = str(precision or "fp32").lower()
    if normalized in {"fp32", "float32", "none"}:
        return None
    if device.type != "cuda":
        raise ValueError(f"Mixed precision requires a CUDA device, got {device}.")
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 autocast is not supported by the current CUDA device.")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision!r}. Expected fp32, fp16, or bf16.")


def _select_torch_device(torch: Any, device: str) -> Any:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_distributed_training(torch: Any, device: str) -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE") or "1")
    rank = int(os.environ.get("RANK") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK") or "0")
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            target_device = torch.device("cuda", local_rank)
        else:
            target_device = torch.device("cpu")
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if torch.cuda.is_available():
                try:
                    torch.distributed.init_process_group(backend=backend, device_id=target_device)
                except TypeError:
                    torch.distributed.init_process_group(backend=backend)
            else:
                torch.distributed.init_process_group(backend=backend)
    else:
        target_device = _select_torch_device(torch, device)
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main_process": rank == 0,
        "device": target_device,
    }


def _distributed_barrier(torch: Any, distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK") or "0")])
        else:
            torch.distributed.barrier()


def _destroy_distributed_training(torch: Any, distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _build_dataloaders(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    image_size: int,
    num_workers: int,
    prefetch_factor: int,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> Tuple[Any, Any]:
    torch, _, _ = _require_torch_stack(require_pillow=True)
    worker_kwargs = _dataloader_worker_kwargs(int(num_workers), int(prefetch_factor))
    train_dataset = _Bench2DriveVisionDataset(train_rows, image_size=image_size)
    train_sampler = None
    if int(distributed_world_size) > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=int(distributed_world_size),
            rank=int(distributed_rank),
            shuffle=True,
            drop_last=len(train_dataset) >= int(distributed_world_size),
        )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= batch_size,
        **worker_kwargs,
    )
    val_loader = None
    if val_rows:
        val_dataset = _Bench2DriveVisionDataset(val_rows, image_size=image_size)
        val_sampler = None
        if int(distributed_world_size) > 1:
            val_sampler = _DistributedShardSamplerNoPadding(
                dataset_length=len(val_dataset),
                rank=int(distributed_rank),
                world_size=int(distributed_world_size),
            )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **worker_kwargs,
        )
    return train_loader, val_loader


def _set_distributed_sampler_epoch(loader: Any, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))


def _dataloader_worker_kwargs(num_workers: int, prefetch_factor: int) -> Dict[str, Any]:
    if int(num_workers) <= 0:
        return {}
    return {
        "persistent_workers": True,
        "prefetch_factor": max(int(prefetch_factor), 1),
    }


class _Bench2DriveVisionDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]], *, image_size: int) -> None:
        torch, _, _ = _require_torch_stack(require_pillow=True)
        self._torch = torch
        self.rows = [dict(row) for row in rows]
        self.image_size = int(image_size)
        self._tar_handles: Dict[str, tarfile.TarFile] = {}
        self._tensor_cache_handles: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __del__(self) -> None:
        for handle in self._tar_handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[int(idx)]
        torch = self._torch
        images = self._load_images(row)
        future = torch.tensor(row["future_waypoints_ego"], dtype=torch.float32).reshape(-1)
        control = torch.tensor(
            [
                float(row["control"]["steer"]),
                float(row["control"]["throttle"]),
                float(row["control"]["brake"]),
            ],
            dtype=torch.float32,
        )
        route = torch.tensor(row["route_features"], dtype=torch.float32)
        brake = torch.tensor([1.0 if row.get("should_brake") else 0.0], dtype=torch.float32)
        return {
            "images": images,
            "route": route,
            "future": future,
            "control": control,
            "brake": brake,
            "case_id": f"{row['clip_name']}:{int(row['frame_id']):05d}",
            "scenario_family": str(row.get("scenario_family") or ""),
        }

    def _load_images(self, row: Mapping[str, Any]) -> Any:
        tensor_cache_path = str(row.get("tensor_cache_path") or "")
        tensor_cache_index = row.get("tensor_cache_index")
        tensor_cache_image_size = int(row.get("tensor_cache_image_size") or 0)
        if tensor_cache_path and tensor_cache_index is not None and tensor_cache_image_size == self.image_size:
            cache_path = Path(tensor_cache_path)
            if cache_path.exists():
                return self._load_tensor_cache_sample(cache_path, int(tensor_cache_index))

        images = []
        camera_members = dict(row["cameras"])
        camera_cache_paths = dict(row.get("camera_cache_paths") or {})
        camera_names = list(row.get("camera_names") or camera_members.keys())
        for camera_name in camera_names:
            camera_key = str(camera_name)
            cache_path = Path(str(camera_cache_paths.get(camera_key) or ""))
            if cache_path.exists():
                images.append(self._load_image_file_uint8_tensor(cache_path))
            else:
                member_name = camera_members[camera_key]
                images.append(self._load_image_uint8_tensor(str(row["archive"]), str(member_name)))
        return self._torch.stack(images, dim=0)

    def _load_tensor_cache_sample(self, tensor_cache_path: Path, tensor_cache_index: int) -> Any:
        import numpy as np

        cache_key = str(tensor_cache_path)
        tensor = self._tensor_cache_handles.get(cache_key)
        if tensor is None:
            tensor = np.load(cache_key, mmap_mode="r")
            self._tensor_cache_handles[cache_key] = tensor
        array = np.asarray(tensor[int(tensor_cache_index)])
        return self._torch.from_numpy(array.copy())

    def _load_image_uint8_tensor(self, archive_path: str, member_name: str) -> Any:
        torch = self._torch
        tar = self._tar_handles.get(archive_path)
        if tar is None:
            tar = tarfile.open(archive_path, "r:gz")
            self._tar_handles[archive_path] = tar
        stream = tar.extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(member_name)
        return torch.from_numpy(_load_image_bytes_uint8_array(stream.read(), image_size=self.image_size).copy())

    def _load_image_file_uint8_tensor(self, path: Path) -> Any:
        return self._torch.from_numpy(_load_image_file_uint8_array(path, image_size=self.image_size).copy())


class _DistributedShardSamplerNoPadding:
    def __init__(self, *, dataset_length: int, rank: int, world_size: int) -> None:
        self.indices = list(range(int(rank), int(dataset_length), max(int(world_size), 1)))

    def __iter__(self) -> Any:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _build_vision_e2e_model(config: VisionE2EModelConfig) -> Any:
    torch, nn, _ = _require_torch_stack()
    size_to_channels = {
        "tiny": [24, 48, 96],
        "base": [32, 64, 128, 256],
        "large": [64, 128, 256, 512],
        "research": [64, 128, 256, 384],
    }
    channels = size_to_channels.get(config.model_size, size_to_channels["base"])
    architecture = str(getattr(config, "architecture", "conv_mlp") or "conv_mlp")

    class ConvEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            in_channels = 3
            for out_channels in channels:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            self.features = nn.Sequential(*layers)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.spatial_pool = nn.AdaptiveAvgPool2d((config.spatial_pool_size, config.spatial_pool_size))
            self.out_dim = channels[-1]

        def forward(self, images: Any) -> Any:
            return self.global_pool(self.features(images)).flatten(1)

        def spatial_tokens(self, images: Any) -> Any:
            features = self.features(images)
            pooled = self.spatial_pool(features)
            return pooled.flatten(2).transpose(1, 2)

    class VisionE2EPlanner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ConvEncoder()
            hidden = channels[-1]
            self.camera_embedding = nn.Parameter(torch.randn(config.camera_count, hidden) * 0.02)
            self.camera_pooling = str(config.camera_pooling or "mean")
            self.route_mlp = nn.Sequential(
                nn.Linear(config.route_feature_dim, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
            if self.camera_pooling == "attention":
                self.camera_attention = nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden, 1),
                )
            dropout_p = max(min(float(config.dropout), 0.8), 0.0)
            fusion_layers = [
                nn.Linear(hidden * 2, hidden),
                nn.SiLU(inplace=True),
            ]
            if dropout_p > 0.0:
                fusion_layers.append(nn.Dropout(p=dropout_p))
            fusion_layers.extend(
                [
                    nn.Linear(hidden, hidden),
                    nn.SiLU(inplace=True),
                ]
            )
            if dropout_p > 0.0:
                fusion_layers.append(nn.Dropout(p=dropout_p))
            self.fusion = nn.Sequential(*fusion_layers)
            self.future_head = nn.Linear(hidden, config.future_steps * 2)
            self.control_head = nn.Linear(hidden, 3)
            self.brake_head = nn.Linear(hidden, 1)

        def forward(self, images: Any, route: Any) -> Dict[str, Any]:
            if config.input_mode == "route_only":
                images = torch.zeros_like(images)
            batch, camera_count, channels_in, height, width = images.shape
            encoded = self.encoder(images.reshape(batch * camera_count, channels_in, height, width))
            encoded = encoded.reshape(batch, camera_count, -1)
            camera_bias = self.camera_embedding[:camera_count].unsqueeze(0)
            route_token = self.route_mlp(route)
            camera_tokens = encoded + camera_bias
            if self.camera_pooling == "attention":
                route_context = route_token.unsqueeze(1).expand(-1, camera_count, -1)
                attention_logits = self.camera_attention(torch.cat([camera_tokens, route_context], dim=2))
                attention = torch.softmax(attention_logits, dim=1)
                scene = (camera_tokens * attention).sum(dim=1)
            else:
                scene = camera_tokens.mean(dim=1)
            fused = self.fusion(torch.cat([scene, route_token], dim=1))
            return {
                "future": self.future_head(fused),
                "control": self.control_head(fused),
                "brake_logits": self.brake_head(fused),
            }

    class TrajectoryTransformerPlanner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ConvEncoder()
            hidden = channels[-1]
            self.hidden = hidden
            self.mode_count = max(int(getattr(config, "trajectory_modes", 1) or 1), 1)
            self.future_steps = int(config.future_steps)
            self.camera_embedding = nn.Parameter(torch.randn(config.camera_count, hidden) * 0.02)
            self.spatial_token_count = int(config.spatial_pool_size) ** 2
            self.spatial_embedding = nn.Parameter(torch.randn(self.spatial_token_count, hidden) * 0.02)
            self.mode_queries = nn.Parameter(torch.randn(self.mode_count, hidden) * 0.02)
            self.trajectory_selection = str(getattr(config, "trajectory_selection", "argmax") or "argmax")
            self.trajectory_top_k = max(int(getattr(config, "trajectory_top_k", 2) or 2), 1)
            self.trajectory_temperature = max(float(getattr(config, "trajectory_temperature", 1.0) or 1.0), 1e-6)
            calibrator = dict(getattr(config, "trajectory_mode_calibrator", None) or {})
            self.mode_calibrator_enabled = bool(calibrator.get("enabled"))
            if self.mode_calibrator_enabled:
                self.register_buffer(
                    "mode_calibrator_mean",
                    torch.tensor(calibrator["feature_mean"], dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "mode_calibrator_scale",
                    torch.tensor(calibrator["feature_scale"], dtype=torch.float32).clamp_min(1e-6),
                    persistent=False,
                )
                self.register_buffer(
                    "mode_calibrator_coef",
                    torch.tensor(calibrator["coef"], dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "mode_calibrator_intercept",
                    torch.tensor(calibrator["intercept"], dtype=torch.float32),
                    persistent=False,
                )
            self.route_mlp = nn.Sequential(
                nn.Linear(config.route_feature_dim, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
            dropout_p = max(min(float(config.dropout), 0.8), 0.0)
            head_count = max(4, min(8, hidden // 64))
            layer_count = 4 if str(config.model_size) == "research" else 3
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=head_count,
                dim_feedforward=hidden * 4,
                dropout=dropout_p,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layer_count)
            self.future_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.future_steps * 2),
            )
            self.mode_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            self.control_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            )
            self.brake_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, images: Any, route: Any) -> Dict[str, Any]:
            if config.input_mode == "route_only":
                images = torch.zeros_like(images)
            batch, camera_count, channels_in, height, width = images.shape
            spatial_tokens = self.encoder.spatial_tokens(images.reshape(batch * camera_count, channels_in, height, width))
            spatial_tokens = spatial_tokens.reshape(batch, camera_count, self.spatial_token_count, self.hidden)
            camera_bias = self.camera_embedding[:camera_count].view(1, camera_count, 1, self.hidden)
            spatial_bias = self.spatial_embedding.view(1, 1, self.spatial_token_count, self.hidden)
            camera_tokens = (spatial_tokens + camera_bias + spatial_bias).reshape(batch, camera_count * self.spatial_token_count, self.hidden)
            route_token = self.route_mlp(route).unsqueeze(1)
            mode_queries = self.mode_queries.unsqueeze(0).expand(batch, -1, -1)
            tokens = torch.cat([route_token, camera_tokens, mode_queries], dim=1)
            encoded = self.transformer(tokens)
            scene_token = encoded[:, 0]
            mode_states = encoded[:, -self.mode_count :]
            future_modes = self.future_head(mode_states)
            mode_logits = self.mode_head(mode_states).squeeze(-1)
            selection_logits = mode_logits / float(self.trajectory_temperature)
            mode_probabilities = torch.softmax(selection_logits, dim=1)
            selected_future = self._select_future(future_modes, mode_logits, mode_probabilities)
            uncalibrated_future = selected_future
            mode_context = (mode_states * mode_probabilities.unsqueeze(-1)).sum(dim=1)
            fused = scene_token + mode_context
            control_output = self.control_head(fused)
            brake_output = self.brake_head(fused)
            calibrated_probabilities = None
            if self.mode_calibrator_enabled:
                selected_future, calibrated_probabilities = self._apply_mode_calibrator(
                    selected_future=selected_future,
                    future_modes=future_modes,
                    mode_logits=mode_logits,
                    control_output=control_output,
                    brake_output=brake_output,
                )
            return {
                "future": selected_future,
                "uncalibrated_future": uncalibrated_future,
                "future_modes": future_modes,
                "mode_logits": mode_logits,
                "calibrated_mode_probabilities": calibrated_probabilities,
                "control": control_output,
                "brake_logits": brake_output,
            }

        def _apply_mode_calibrator(
            self,
            *,
            selected_future: Any,
            future_modes: Any,
            mode_logits: Any,
            control_output: Any,
            brake_output: Any,
        ) -> Tuple[Any, Any]:
            features = torch.cat(
                [
                    selected_future,
                    future_modes.flatten(1),
                    mode_logits,
                    control_output,
                    torch.sigmoid(brake_output),
                ],
                dim=1,
            )
            normalized = (features - self.mode_calibrator_mean) / self.mode_calibrator_scale
            calibrated_logits = normalized @ self.mode_calibrator_coef.transpose(0, 1)
            calibrated_logits = calibrated_logits + self.mode_calibrator_intercept
            calibrated_probabilities = torch.softmax(calibrated_logits, dim=1)
            calibrated_future = (future_modes * calibrated_probabilities.unsqueeze(-1)).sum(dim=1)
            return calibrated_future, calibrated_probabilities

        def _select_future(self, future_modes: Any, mode_logits: Any, mode_probabilities: Any) -> Any:
            batch = int(future_modes.shape[0])
            selection = self.trajectory_selection
            if selection == "expected":
                return (future_modes * mode_probabilities.unsqueeze(-1)).sum(dim=1)
            if selection in {"topk_expected", "top2_expected"}:
                top_k = min(max(int(self.trajectory_top_k), 1), int(future_modes.shape[1]))
                top_probs, top_indices = mode_probabilities.topk(k=top_k, dim=1)
                gather_index = top_indices.unsqueeze(-1).expand(-1, -1, self.future_steps * 2)
                top_futures = future_modes.gather(1, gather_index)
                top_probs = top_probs / top_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
                return (top_futures * top_probs.unsqueeze(-1)).sum(dim=1)
            selected_index = mode_logits.argmax(dim=1)
            gather_index = selected_index.view(batch, 1, 1).expand(-1, 1, self.future_steps * 2)
            return future_modes.gather(1, gather_index).squeeze(1)

    if architecture == "trajectory_transformer":
        return TrajectoryTransformerPlanner()
    return VisionE2EPlanner()


def _run_training_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    *,
    autocast_dtype: Any = None,
    scaler: Any = None,
    loss_config: Optional[VisionE2ELossConfig] = None,
    nonfinite_check_interval: int = 0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, Any] = {}
    sample_count = 0
    started_at = time.time()
    check_interval = max(int(nonfinite_check_interval), 0)
    for step_idx, batch in enumerate(loader, start=1):
        images, route, future, control, brake = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            prediction = model(images, route)
            losses = _planner_losses(torch, prediction, future, control, brake, loss_config=loss_config)
        if check_interval > 0 and step_idx % check_interval == 0 and not bool(torch.isfinite(losses["loss"]).detach().cpu()):
            raise ValueError("Non-finite training loss detected. Check manifest labels and model outputs.")
        if scaler is not None and bool(scaler.is_enabled()):
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            optimizer.step()
        batch_size = int(images.shape[0])
        sample_count += batch_size
        for key, value in losses.items():
            contribution = value.detach() * batch_size
            totals[key] = contribution if key not in totals else totals[key] + contribution
    metrics = _finalize_weighted_tensor_metrics(totals, sample_count)
    elapsed = max(time.time() - started_at, 1e-6)
    metrics["epoch_runtime_s"] = round(elapsed, 3)
    metrics["samples_per_s"] = float(sample_count / elapsed)
    metrics["sample_count"] = float(sample_count)
    return _add_brake_derived_metrics(metrics)


def _run_eval_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    *,
    autocast_dtype: Any = None,
    loss_config: Optional[VisionE2ELossConfig] = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, Any] = {}
    sample_count = 0
    started_at = time.time()
    with torch.no_grad():
        for batch in loader:
            images, route, future, control, brake = _batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                prediction = model(images, route)
                losses = _planner_losses(torch, prediction, future, control, brake, loss_config=loss_config)
            batch_size = int(images.shape[0])
            sample_count += batch_size
            for key, value in losses.items():
                contribution = value.detach() * batch_size
                totals[key] = contribution if key not in totals else totals[key] + contribution
    metrics = _finalize_weighted_tensor_metrics(totals, sample_count)
    elapsed = max(time.time() - started_at, 1e-6)
    metrics["epoch_runtime_s"] = round(elapsed, 3)
    metrics["samples_per_s"] = float(sample_count / elapsed)
    metrics["sample_count"] = float(sample_count)
    return _add_brake_derived_metrics(metrics)


def _run_prediction_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    *,
    loss_config: Optional[VisionE2ELossConfig] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    totals: Dict[str, float] = {}
    sample_count = 0
    predictions = []
    with torch.no_grad():
        for batch in loader:
            images, route, future, control, brake = _batch_to_device(batch, device)
            prediction = model(images, route)
            losses = _planner_losses(torch, prediction, future, control, brake, loss_config=loss_config)
            batch_size = int(images.shape[0])
            sample_count += batch_size
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
            pred_future = prediction["future"].detach().cpu().reshape(images.shape[0], -1, 2).tolist()
            uncalibrated_future = None
            if prediction.get("uncalibrated_future") is not None:
                uncalibrated_future = (
                    prediction["uncalibrated_future"]
                    .detach()
                    .cpu()
                    .reshape(images.shape[0], -1, 2)
                    .tolist()
                )
            true_future = future.detach().cpu().reshape(images.shape[0], -1, 2).tolist()
            pred_control = prediction["control"].detach().cpu().tolist()
            true_control = control.detach().cpu().tolist()
            target_brake = brake.detach().cpu().reshape(-1).tolist()
            brake_prob = torch.sigmoid(prediction["brake_logits"]).detach().cpu().reshape(-1).tolist()
            mode_futures = None
            mode_logits = None
            if prediction.get("future_modes") is not None and prediction.get("mode_logits") is not None:
                mode_futures = (
                    prediction["future_modes"]
                    .detach()
                    .cpu()
                    .reshape(images.shape[0], prediction["future_modes"].shape[1], -1, 2)
                    .tolist()
                )
                mode_logits = prediction["mode_logits"].detach().cpu().tolist()
            for idx, case_id in enumerate(batch["case_id"]):
                row = {
                        "case_id": str(case_id),
                        "scenario_family": str(batch["scenario_family"][idx]),
                        "predicted_future_waypoints_ego": pred_future[idx],
                        "target_future_waypoints_ego": true_future[idx],
                        "predicted_control": pred_control[idx],
                        "target_control": true_control[idx],
                        "predicted_brake_probability": float(brake_prob[idx]),
                        "predicted_should_brake": bool(float(brake_prob[idx]) >= float((loss_config or VisionE2ELossConfig()).brake_threshold)),
                        "target_should_brake": bool(float(target_brake[idx]) >= 0.5),
                    }
                if mode_futures is not None and mode_logits is not None:
                    row["predicted_future_modes_ego"] = mode_futures[idx]
                    row["mode_logits"] = mode_logits[idx]
                if uncalibrated_future is not None:
                    row["uncalibrated_future_waypoints_ego"] = uncalibrated_future[idx]
                predictions.append(row)
    return _add_brake_derived_metrics(_finalize_weighted_metrics(totals, sample_count)), predictions


def _select_trajectory_from_prediction(
    row: Mapping[str, Any],
    *,
    selection: str,
    temperature: float,
    top_k: int,
) -> List[List[float]]:
    import numpy as np

    modes = np.asarray(row["predicted_future_modes_ego"], dtype=float)
    logits = np.asarray(row["mode_logits"], dtype=float)
    if selection == "argmax":
        return modes[int(np.argmax(logits))].tolist()
    scaled = logits / max(float(temperature), 1e-6)
    scaled = scaled - float(np.max(scaled))
    probabilities = np.exp(scaled)
    probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
    if selection == "topk_expected":
        count = min(max(int(top_k), 1), len(probabilities))
        indices = np.argsort(probabilities)[-count:]
        probabilities = probabilities[indices]
        probabilities = probabilities / max(float(probabilities.sum()), 1e-12)
        modes = modes[indices]
    elif selection != "expected":
        raise ValueError(f"Unknown trajectory selection: {selection}")
    return np.sum(modes * probabilities[:, None, None], axis=0).tolist()


def _trajectory_metrics_from_pairs(
    predicted: Sequence[Sequence[Sequence[float]]],
    target: Sequence[Sequence[Sequence[float]]],
) -> Dict[str, float]:
    import numpy as np

    pred = np.asarray(predicted, dtype=float)
    truth = np.asarray(target, dtype=float)
    distances = np.linalg.norm(pred - truth, axis=-1)
    return {
        "ade_m": round(float(np.mean(distances)), 6),
        "fde_m": round(float(np.mean(distances[:, -1])), 6),
    }


def _mode_calibrator_feature_arrays(
    predictions: Sequence[Mapping[str, Any]],
    *,
    selection: str,
    temperature: float,
    top_k: int,
) -> Tuple[Any, Any, Any, Any, List[str]]:
    import numpy as np

    selected = np.asarray(
        [
            _select_trajectory_from_prediction(
                row,
                selection=selection,
                temperature=temperature,
                top_k=top_k,
            )
            for row in predictions
        ],
        dtype=float,
    )
    modes = np.asarray([row["predicted_future_modes_ego"] for row in predictions], dtype=float)
    logits = np.asarray([row["mode_logits"] for row in predictions], dtype=float)
    control = np.asarray([row["predicted_control"] for row in predictions], dtype=float)
    brake_probability = np.asarray(
        [row["predicted_brake_probability"] for row in predictions],
        dtype=float,
    ).reshape(-1, 1)
    target = np.asarray([row["target_future_waypoints_ego"] for row in predictions], dtype=float)
    features = np.concatenate(
        [
            selected.reshape(len(predictions), -1),
            modes.reshape(len(predictions), -1),
            logits,
            control,
            brake_probability,
        ],
        axis=1,
    )
    clusters = [str(row.get("clip_name") or str(row.get("case_id") or "").rsplit(":", 1)[0]) for row in predictions]
    return features, selected, modes, target, clusters


def _calibrator_probabilities(model: Any, features: Any, mode_count: int) -> Any:
    import numpy as np

    raw = np.asarray(model.predict_proba(features), dtype=float)
    probabilities = np.zeros((len(features), int(mode_count)), dtype=float)
    for source_idx, class_id in enumerate(model.named_steps["logisticregression"].classes_):
        probabilities[:, int(class_id)] = raw[:, source_idx]
    row_sum = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.maximum(row_sum, 1e-12)


def _fit_mode_calibrator(
    predictions: Sequence[Mapping[str, Any]],
    *,
    selection: str,
    temperature: float,
    top_k: int,
    regularization_values: Sequence[float] = (0.1, 1.0, 10.0),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import numpy as np

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Trajectory mode calibration requires scikit-learn. Install the vision extra."
        ) from exc

    features, selected, modes, target, clusters = _mode_calibrator_feature_arrays(
        predictions,
        selection=selection,
        temperature=temperature,
        top_k=top_k,
    )
    mode_count = int(modes.shape[1])
    mode_ade = np.linalg.norm(modes - target[:, None, :, :], axis=-1).mean(axis=-1)
    target_mode = mode_ade.argmin(axis=1)
    unique_target_modes = np.unique(target_mode)
    if mode_count < 2 or len(unique_target_modes) < 2:
        return {"enabled": False}, {
            "enabled": False,
            "reason": "insufficient_mode_diversity",
            "mode_count": mode_count,
            "target_mode_count": int(len(unique_target_modes)),
        }

    unique_clusters = sorted(set(clusters))
    holdout_clusters = {
        name
        for name in unique_clusters
        if int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0
    }
    if not holdout_clusters and unique_clusters:
        holdout_clusters = {unique_clusters[-1]}
    holdout_mask = np.asarray([name in holdout_clusters for name in clusters], dtype=bool)
    train_mask = ~holdout_mask
    if not bool(train_mask.any()) or not bool(holdout_mask.any()):
        raise ValueError("Mode calibration requires at least one calibration-train and holdout cluster.")

    baseline_holdout = _trajectory_metrics_from_pairs(selected[holdout_mask], target[holdout_mask])
    search_rows: List[Dict[str, Any]] = []
    for regularization in regularization_values:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=max(float(regularization), 1e-6),
                max_iter=3000,
                solver="lbfgs",
            ),
        )
        model.fit(features[train_mask], target_mode[train_mask])
        probabilities = _calibrator_probabilities(model, features[holdout_mask], mode_count)
        calibrated = np.sum(modes[holdout_mask] * probabilities[:, :, None, None], axis=1)
        metrics = _trajectory_metrics_from_pairs(calibrated, target[holdout_mask])
        search_rows.append(
            {
                "regularization_c": float(regularization),
                **metrics,
                "ade_delta_m": round(float(metrics["ade_m"]) - float(baseline_holdout["ade_m"]), 6),
                "fde_delta_m": round(float(metrics["fde_m"]) - float(baseline_holdout["fde_m"]), 6),
            }
        )

    selected_idx = min(
        range(len(search_rows)),
        key=lambda idx: (float(search_rows[idx]["ade_m"]), float(search_rows[idx]["fde_m"])),
    )
    selected_search = search_rows[selected_idx]
    # Require both trajectory metrics to improve on held-out clips. A small ADE
    # gain should not mask a degraded endpoint forecast.
    if any(
        float(selected_search[key]) >= 0.0
        for key in ("ade_delta_m", "fde_delta_m")
    ):
        return {"enabled": False}, {
            "enabled": False,
            "reason": "no_clip_holdout_improvement",
            "baseline_holdout": baseline_holdout,
            "selected": dict(selected_search),
            "search": sorted(search_rows, key=lambda row: (float(row["ade_m"]), float(row["fde_m"]))),
        }
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(selected_search["regularization_c"]),
            max_iter=3000,
            solver="lbfgs",
        ),
    )
    final_model.fit(features, target_mode)
    scaler = final_model.named_steps["standardscaler"]
    classifier = final_model.named_steps["logisticregression"]

    full_coef = np.zeros((mode_count, features.shape[1]), dtype=float)
    full_intercept = np.full(mode_count, -20.0, dtype=float)
    for source_idx, class_id in enumerate(classifier.classes_):
        full_coef[int(class_id)] = classifier.coef_[source_idx]
        full_intercept[int(class_id)] = classifier.intercept_[source_idx]

    config = {
        "schema": "bench2drive_mode_calibrator_v1",
        "enabled": True,
        "method": "validation_fitted_multinomial_expected",
        "feature_order": [
            "selected_future_flat",
            "future_modes_flat",
            "mode_logits",
            "predicted_control",
            "predicted_brake_probability",
        ],
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coef": full_coef.tolist(),
        "intercept": full_intercept.tolist(),
        "mode_count": mode_count,
        "regularization_c": float(selected_search["regularization_c"]),
    }
    report = {
        "enabled": True,
        "protocol": "clip-disjoint 80/20 calibration holdout; refit on all validation clips",
        "calibration_sample_count": int(train_mask.sum()),
        "holdout_sample_count": int(holdout_mask.sum()),
        "calibration_cluster_count": int(len(unique_clusters) - len(holdout_clusters)),
        "holdout_cluster_count": int(len(holdout_clusters)),
        "mode_count": mode_count,
        "target_mode_distribution": {
            str(mode_idx): int(np.sum(target_mode == mode_idx)) for mode_idx in range(mode_count)
        },
        "baseline_holdout": baseline_holdout,
        "selected": dict(selected_search),
        "search": sorted(search_rows, key=lambda row: (float(row["ade_m"]), float(row["fde_m"]))),
    }
    return config, report


def _bootstrap_prediction_metrics(
    predictions: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> Dict[str, Any]:
    import numpy as np

    if not predictions:
        return {"method": "cluster-level percentile bootstrap", "replicates": 0, "metrics": {}}

    rows_by_cluster: Dict[str, List[Mapping[str, Any]]] = {}
    for row in predictions:
        case_id = str(row.get("case_id") or "unknown")
        cluster = str(row.get("clip_name") or case_id.rsplit(":", 1)[0] or case_id)
        rows_by_cluster.setdefault(cluster, []).append(row)

    def row_metrics(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[float], List[bool], List[bool]]:
        ade: List[float] = []
        fde: List[float] = []
        brake_target: List[bool] = []
        brake_predicted: List[bool] = []
        for row in rows:
            pred = np.asarray(row["predicted_future_waypoints_ego"], dtype=float)
            target = np.asarray(row["target_future_waypoints_ego"], dtype=float)
            distance = np.linalg.norm(pred - target, axis=-1)
            ade.append(float(np.mean(distance)))
            fde.append(float(distance[-1]))
            brake_target.append(bool(row.get("target_should_brake")))
            brake_predicted.append(bool(row.get("predicted_should_brake")))
        return ade, fde, brake_target, brake_predicted

    ade, fde, brake_target, brake_predicted = row_metrics(predictions)
    ade_array = np.asarray(ade, dtype=float)
    fde_array = np.asarray(fde, dtype=float)
    target_array = np.asarray(brake_target, dtype=bool)
    predicted_array = np.asarray(brake_predicted, dtype=bool)
    rng = np.random.default_rng(int(seed))
    replicate_count = max(int(replicates), 1)
    ade_samples = np.empty(replicate_count, dtype=float)
    fde_samples = np.empty(replicate_count, dtype=float)
    f1_samples = np.empty(replicate_count, dtype=float)
    clusters = list(rows_by_cluster.values())
    cluster_count = len(clusters)
    for idx in range(replicate_count):
        cluster_indices = rng.integers(0, cluster_count, size=cluster_count)
        sampled_rows = [row for cluster_idx in cluster_indices for row in clusters[int(cluster_idx)]]
        sampled_ade, sampled_fde, sampled_target, sampled_predicted = row_metrics(sampled_rows)
        ade_samples[idx] = float(np.mean(sampled_ade))
        fde_samples[idx] = float(np.mean(sampled_fde))
        sampled_target_array = np.asarray(sampled_target, dtype=bool)
        sampled_predicted_array = np.asarray(sampled_predicted, dtype=bool)
        tp = int(np.sum(sampled_target_array & sampled_predicted_array))
        fp = int(np.sum(~sampled_target_array & sampled_predicted_array))
        fn = int(np.sum(sampled_target_array & ~sampled_predicted_array))
        f1_samples[idx] = 2.0 * tp / max(2 * tp + fp + fn, 1)

    def summary(values: Any, point: float) -> Dict[str, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return {"point": round(float(point), 6), "ci95_low": round(float(low), 6), "ci95_high": round(float(high), 6)}

    tp = int(np.sum(target_array & predicted_array))
    fp = int(np.sum(~target_array & predicted_array))
    fn = int(np.sum(target_array & ~predicted_array))
    f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
    return {
        "method": "cluster-level percentile bootstrap",
        "seed": int(seed),
        "replicates": replicate_count,
        "sample_count": len(predictions),
        "cluster_key": "clip_name or case_id prefix",
        "cluster_count": cluster_count,
        "metrics": {
            "ade_m": summary(ade_samples, float(np.mean(ade_array))),
            "fde_m": summary(fde_samples, float(np.mean(fde_array))),
            "brake_f1": summary(f1_samples, f1),
        },
    }


def _trajectory_calibration_comparison(
    predictions: Sequence[Mapping[str, Any]],
    *,
    seed: int = 7,
    replicates: int = 2000,
) -> Dict[str, Any]:
    import numpy as np

    rows = [row for row in predictions if row.get("uncalibrated_future_waypoints_ego")]
    if not rows:
        return {"enabled": False, "sample_count": 0}

    baseline_ade = []
    baseline_fde = []
    calibrated_ade = []
    calibrated_fde = []
    clusters = []
    for row in rows:
        target = np.asarray(row["target_future_waypoints_ego"], dtype=float)
        baseline = np.asarray(row["uncalibrated_future_waypoints_ego"], dtype=float)
        calibrated = np.asarray(row["predicted_future_waypoints_ego"], dtype=float)
        baseline_distance = np.linalg.norm(baseline - target, axis=-1)
        calibrated_distance = np.linalg.norm(calibrated - target, axis=-1)
        baseline_ade.append(float(np.mean(baseline_distance)))
        baseline_fde.append(float(baseline_distance[-1]))
        calibrated_ade.append(float(np.mean(calibrated_distance)))
        calibrated_fde.append(float(calibrated_distance[-1]))
        case_id = str(row.get("case_id") or "unknown")
        clusters.append(str(row.get("clip_name") or case_id.rsplit(":", 1)[0] or case_id))

    baseline_ade_array = np.asarray(baseline_ade, dtype=float)
    baseline_fde_array = np.asarray(baseline_fde, dtype=float)
    calibrated_ade_array = np.asarray(calibrated_ade, dtype=float)
    calibrated_fde_array = np.asarray(calibrated_fde, dtype=float)
    cluster_names = sorted(set(clusters))
    cluster_indices = {
        name: np.asarray([idx for idx, cluster in enumerate(clusters) if cluster == name], dtype=int)
        for name in cluster_names
    }
    rng = np.random.default_rng(int(seed))
    ade_deltas = np.zeros(max(int(replicates), 1), dtype=float)
    fde_deltas = np.zeros(max(int(replicates), 1), dtype=float)
    for replicate_idx in range(len(ade_deltas)):
        sampled = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        indices = np.concatenate([cluster_indices[name] for name in sampled])
        ade_deltas[replicate_idx] = float(
            np.mean(calibrated_ade_array[indices] - baseline_ade_array[indices])
        )
        fde_deltas[replicate_idx] = float(
            np.mean(calibrated_fde_array[indices] - baseline_fde_array[indices])
        )

    def metric_summary(baseline: Any, calibrated: Any, deltas: Any) -> Dict[str, float]:
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        return {
            "baseline": round(float(np.mean(baseline)), 6),
            "calibrated": round(float(np.mean(calibrated)), 6),
            "delta": round(float(np.mean(calibrated - baseline)), 6),
            "delta_ci95_low": round(float(ci_low), 6),
            "delta_ci95_high": round(float(ci_high), 6),
        }

    return {
        "enabled": True,
        "method": "paired clip-level percentile bootstrap",
        "sample_count": len(rows),
        "cluster_count": len(cluster_names),
        "replicates": len(ade_deltas),
        "metrics": {
            "ade_m": metric_summary(baseline_ade_array, calibrated_ade_array, ade_deltas),
            "fde_m": metric_summary(baseline_fde_array, calibrated_fde_array, fde_deltas),
        },
    }


def _batch_to_device(batch: Mapping[str, Any], device: Any) -> Tuple[Any, Any, Any, Any, Any]:
    images = batch["images"].to(device, non_blocking=True)
    if str(images.dtype) == "torch.uint8":
        images = images.float().div_(255.0)
    images = _normalize_images_on_device(images)
    return (
        images,
        batch["route"].to(device, non_blocking=True),
        batch["future"].to(device, non_blocking=True),
        batch["control"].to(device, non_blocking=True),
        batch["brake"].to(device, non_blocking=True),
    )


def _normalize_images_on_device(images: Any) -> Any:
    mean_tensor, std_tensor = _normalization_tensors(images)
    return (images - mean_tensor) / std_tensor


def _normalization_tensors(images: Any) -> Tuple[Any, Any]:
    key = (str(images.device), str(images.dtype))
    cached = _NORMALIZATION_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    mean_tensor = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std_tensor = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    cached = (mean_tensor, std_tensor)
    _NORMALIZATION_TENSOR_CACHE[key] = cached
    return cached


def _planner_losses(
    torch: Any,
    prediction: Mapping[str, Any],
    future: Any,
    control: Any,
    brake: Any,
    *,
    loss_config: Optional[VisionE2ELossConfig] = None,
) -> Dict[str, Any]:
    import torch.nn.functional as functional

    config = loss_config or VisionE2ELossConfig()
    future_pred = prediction["future"]
    control_pred = prediction["control"]
    brake_logits = prediction["brake_logits"]
    pred_xy = future_pred.reshape(future.shape[0], -1, 2)
    true_xy = future.reshape(future.shape[0], -1, 2)
    max_target_lateral = true_xy[:, :, 0].abs().max(dim=1).values
    turn_positive = max_target_lateral >= float(config.turn_lateral_threshold_m)
    risk_weight = 1.0 + (max(float(config.risk_sample_weight), 0.0) - 1.0) * brake.reshape(-1)
    turn_weight = 1.0 + (max(float(config.turn_sample_weight), 0.0) - 1.0) * turn_positive.float()
    sample_weight = risk_weight * turn_weight
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
    waypoint_axis_weight = future.new_tensor([max(float(config.lateral_loss_weight), 1e-6), 1.0]).view(1, 1, 2)
    mode_loss = future.new_tensor(0.0)
    oracle_ade = future.new_tensor(0.0)
    oracle_fde = future.new_tensor(0.0)
    future_modes = prediction.get("future_modes")
    mode_logits = prediction.get("mode_logits")
    if future_modes is not None:
        mode_xy = future_modes.reshape(future.shape[0], -1, true_xy.shape[1], 2)
        mode_element_loss = functional.smooth_l1_loss(
            mode_xy,
            true_xy.unsqueeze(1).expand_as(mode_xy),
            reduction="none",
        )
        mode_per_sample = (mode_element_loss * waypoint_axis_weight.unsqueeze(1)).mean(dim=(2, 3))
        best_mode = mode_per_sample.argmin(dim=1)
        waypoint_per_sample = mode_per_sample.gather(1, best_mode.view(-1, 1)).reshape(-1)
        if mode_logits is not None and mode_logits.shape[:2] == mode_per_sample.shape:
            mode_loss = functional.cross_entropy(mode_logits, best_mode, reduction="none")
            mode_loss = (mode_loss * sample_weight).mean()
        mode_distances = torch.linalg.norm(mode_xy - true_xy.unsqueeze(1), dim=3)
        oracle_mode_ade = mode_distances.mean(dim=2)
        best_oracle_mode = oracle_mode_ade.argmin(dim=1)
        oracle_ade = oracle_mode_ade.gather(1, best_oracle_mode.view(-1, 1)).mean()
        oracle_fde = mode_distances[:, :, -1].gather(1, best_oracle_mode.view(-1, 1)).mean()
    else:
        waypoint_element_loss = functional.smooth_l1_loss(future_pred, future, reduction="none").reshape(future.shape[0], -1, 2)
        waypoint_per_sample = (waypoint_element_loss * waypoint_axis_weight).mean(dim=(1, 2))
    selected_element_loss = functional.smooth_l1_loss(
        future_pred,
        future,
        reduction="none",
    ).reshape(future.shape[0], -1, 2)
    selected_waypoint_per_sample = (selected_element_loss * waypoint_axis_weight).mean(dim=(1, 2))
    origin = true_xy.new_zeros((true_xy.shape[0], 1, 2))
    pred_displacements = torch.diff(torch.cat([origin, pred_xy], dim=1), dim=1)
    true_displacements = torch.diff(torch.cat([origin, true_xy], dim=1), dim=1)
    displacement_element_loss = functional.smooth_l1_loss(
        pred_displacements,
        true_displacements,
        reduction="none",
    )
    displacement_per_sample = (displacement_element_loss * waypoint_axis_weight).mean(dim=(1, 2))
    endpoint_element_loss = functional.smooth_l1_loss(
        pred_xy[:, -1],
        true_xy[:, -1],
        reduction="none",
    )
    endpoint_per_sample = (endpoint_element_loss * waypoint_axis_weight.reshape(2)).mean(dim=1)
    pred_path_length = torch.linalg.norm(pred_displacements, dim=2).sum(dim=1)
    target_path_length = torch.linalg.norm(true_displacements, dim=2).sum(dim=1)
    path_length_per_sample = functional.smooth_l1_loss(
        pred_path_length,
        target_path_length,
        reduction="none",
    )
    control_per_sample = functional.smooth_l1_loss(control_pred, control, reduction="none").reshape(control.shape[0], -1).mean(dim=1)
    pos_weight = brake.new_tensor([max(float(config.brake_positive_weight), 1e-6)])
    brake_per_sample = functional.binary_cross_entropy_with_logits(
        brake_logits,
        brake,
        pos_weight=pos_weight,
        reduction="none",
    ).reshape(-1)
    waypoint_loss = (waypoint_per_sample * sample_weight).mean()
    selected_waypoint_loss = (selected_waypoint_per_sample * sample_weight).mean()
    displacement_loss = (displacement_per_sample * sample_weight).mean()
    endpoint_loss = (endpoint_per_sample * sample_weight).mean()
    path_length_loss = (path_length_per_sample * sample_weight).mean()
    control_loss = (control_per_sample * sample_weight).mean()
    brake_loss = (brake_per_sample * sample_weight).mean()
    loss = (
        float(config.waypoint_weight) * waypoint_loss
        + float(config.selected_waypoint_weight) * selected_waypoint_loss
        + float(config.displacement_weight) * displacement_loss
        + float(config.endpoint_weight) * endpoint_loss
        + float(config.path_length_weight) * path_length_loss
        + float(config.control_weight) * control_loss
        + float(config.brake_weight) * brake_loss
        + float(config.mode_classification_weight) * mode_loss
    )
    distances = torch.linalg.norm(pred_xy - true_xy, dim=2)
    ade = distances.mean()
    fde = distances[:, -1].mean()
    lateral_errors = (pred_xy[:, :, 0] - true_xy[:, :, 0]).abs()
    final_lateral_error = lateral_errors[:, -1]
    turn_lateral_error = lateral_errors[turn_positive].mean() if bool(turn_positive.any()) else lateral_errors.new_tensor(0.0)
    turn_final_lateral_error = (
        final_lateral_error[turn_positive].mean() if bool(turn_positive.any()) else final_lateral_error.new_tensor(0.0)
    )
    pred_final_lateral_abs = pred_xy[:, -1, 0].abs().mean()
    target_final_lateral_abs = true_xy[:, -1, 0].abs().mean()
    brake_prob = torch.sigmoid(brake_logits)
    pred_positive = brake_prob >= float(config.brake_threshold)
    target_positive = brake >= 0.5
    brake_acc = (pred_positive == target_positive).float().mean()
    true_positive = (pred_positive & target_positive).float().mean()
    false_positive = (pred_positive & ~target_positive).float().mean()
    false_negative = (~pred_positive & target_positive).float().mean()
    true_negative = (~pred_positive & ~target_positive).float().mean()
    return {
        "loss": loss,
        "waypoint_loss": waypoint_loss,
        "selected_waypoint_loss": selected_waypoint_loss,
        "displacement_loss": displacement_loss,
        "endpoint_loss": endpoint_loss,
        "path_length_loss": path_length_loss,
        "control_loss": control_loss,
        "brake_loss": brake_loss,
        "mode_loss": mode_loss,
        "ade_m": ade,
        "fde_m": fde,
        "oracle_ade_m": oracle_ade,
        "oracle_fde_m": oracle_fde,
        "lateral_mae_m": lateral_errors.mean(),
        "final_lateral_mae_m": final_lateral_error.mean(),
        "turn_lateral_mae_m": turn_lateral_error,
        "turn_final_lateral_mae_m": turn_final_lateral_error,
        "pred_final_lateral_abs_m": pred_final_lateral_abs,
        "target_final_lateral_abs_m": target_final_lateral_abs,
        "pred_path_length_m": pred_path_length.mean(),
        "target_path_length_m": target_path_length.mean(),
        "path_length_ratio": (pred_path_length / target_path_length.clamp_min(1e-6)).mean(),
        "turn_sample_rate": turn_positive.float().mean(),
        "brake_accuracy": brake_acc,
        "brake_tp_rate": true_positive,
        "brake_fp_rate": false_positive,
        "brake_fn_rate": false_negative,
        "brake_tn_rate": true_negative,
        "brake_positive_rate": target_positive.float().mean(),
        "brake_predicted_positive_rate": pred_positive.float().mean(),
    }


def _add_brake_derived_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    payload = dict(metrics)
    tp = float(payload.get("brake_tp_rate") or 0.0)
    fp = float(payload.get("brake_fp_rate") or 0.0)
    fn = float(payload.get("brake_fn_rate") or 0.0)
    tn = float(payload.get("brake_tn_rate") or 0.0)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    if "brake_accuracy" not in payload:
        payload["brake_accuracy"] = (tp + tn) / max(tp + fp + fn + tn, 1e-12)
    payload["brake_precision"] = precision
    payload["brake_recall"] = recall
    payload["brake_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return payload


def _checkpoint_selection_score(metrics: Mapping[str, float], selection_metric: str) -> float:
    name = str(selection_metric or "ade_m")
    if name == "risk_aware":
        ade = float(metrics.get("ade_m") or 0.0)
        fde = float(metrics.get("fde_m") or 0.0)
        brake_f1 = float(metrics.get("brake_f1") or 0.0)
        return ade + 0.15 * fde - 0.75 * brake_f1
    if name == "lateral_aware":
        ade = float(metrics.get("ade_m") or 0.0)
        lateral = float(metrics.get("lateral_mae_m") or 0.0)
        turn_lateral = float(metrics.get("turn_lateral_mae_m") or lateral)
        return ade + 0.35 * lateral + 0.65 * turn_lateral
    if name == "brake_f1":
        return -float(metrics.get("brake_f1") or 0.0)
    return float(metrics.get(name, metrics.get("ade_m", 0.0)) or 0.0)


def _scenario_family_metrics_from_predictions(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, List[Dict[str, float]]] = {}
    for row in predictions:
        family = str(row.get("scenario_family") or "unknown")
        target = row.get("target_future_waypoints_ego") or []
        pred = row.get("predicted_future_waypoints_ego") or []
        if not target or not pred:
            continue
        distances = [
            math.hypot(float(px[0]) - float(tx[0]), float(px[1]) - float(tx[1]))
            for px, tx in zip(pred, target)
        ]
        if not distances:
            continue
        target_brake = bool(row.get("target_should_brake"))
        pred_brake = bool(row.get("predicted_should_brake"))
        rows.setdefault(family, []).append(
            {
                "sample_count": 1.0,
                "ade_m": float(mean(distances)),
                "fde_m": float(distances[-1]),
                "brake_tp_rate": 1.0 if pred_brake and target_brake else 0.0,
                "brake_fp_rate": 1.0 if pred_brake and not target_brake else 0.0,
                "brake_fn_rate": 1.0 if (not pred_brake) and target_brake else 0.0,
                "brake_tn_rate": 1.0 if (not pred_brake) and (not target_brake) else 0.0,
                "brake_positive_rate": 1.0 if target_brake else 0.0,
                "brake_predicted_positive_rate": 1.0 if pred_brake else 0.0,
            }
        )
    metrics = {}
    for family, family_rows in rows.items():
        count = len(family_rows)
        aggregate = {
            key: sum(float(row.get(key) or 0.0) for row in family_rows) / max(count, 1)
            for key in family_rows[0]
            if key != "sample_count"
        }
        aggregate["sample_count"] = float(count)
        metrics[family] = _add_brake_derived_metrics(aggregate)
    return dict(sorted(metrics.items()))


def _read_prediction_rows(predictions_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(predictions_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prediction_diagnostic_row(row: Mapping[str, Any], *, brake_threshold: float) -> Dict[str, Any]:
    pred = [list(point) for point in list(row.get("predicted_future_waypoints_ego") or [])]
    target = [list(point) for point in list(row.get("target_future_waypoints_ego") or [])]
    pred_geom = _waypoint_path_geometry(pred)
    target_geom = _waypoint_path_geometry(target)
    distances = [
        math.hypot(float(px[0]) - float(tx[0]), float(px[1]) - float(tx[1]))
        for px, tx in zip(pred, target)
        if len(px) >= 2 and len(tx) >= 2
    ]
    lateral_errors = [
        abs(_waypoint_forward_right(px)[1] - _waypoint_forward_right(tx)[1])
        for px, tx in zip(pred, target)
        if len(px) >= 2 and len(tx) >= 2
    ]
    target_control = list(row.get("target_control") or [])
    pred_control = list(row.get("predicted_control") or [])
    target_should_brake = bool(row.get("target_should_brake"))
    if "target_should_brake" not in row and len(target_control) >= 3:
        target_should_brake = float(target_control[2]) >= 0.5
    pred_brake_probability = float(row.get("predicted_brake_probability") or 0.0)
    predicted_should_brake = bool(row.get("predicted_should_brake"))
    if "predicted_should_brake" not in row:
        predicted_should_brake = pred_brake_probability >= float(brake_threshold)
    target_final_forward = float(target_geom["final_forward_m"])
    pred_final_forward = float(pred_geom["final_forward_m"])
    target_path_length = float(target_geom["path_length_m"])
    pred_path_length = float(pred_geom["path_length_m"])
    path_length_ratio = pred_path_length / max(target_path_length, 1e-6)
    if abs(target_final_forward) >= 1.0:
        final_forward_ratio = pred_final_forward / target_final_forward
    else:
        final_forward_ratio = path_length_ratio
    horizon_s = max(len(pred), len(target), 1) * BENCH2DRIVE_PLANNER_DT_S
    return {
        "case_id": str(row.get("case_id") or ""),
        "scenario_family": str(row.get("scenario_family") or "unknown"),
        "ade_m": mean(distances) if distances else 0.0,
        "fde_m": distances[-1] if distances else 0.0,
        "mean_abs_lateral_error_m": mean(lateral_errors) if lateral_errors else 0.0,
        "pred_path_length_m": pred_path_length,
        "target_path_length_m": target_path_length,
        "path_length_ratio": path_length_ratio,
        "pred_final_forward_m": pred_final_forward,
        "target_final_forward_m": target_final_forward,
        "final_forward_ratio": final_forward_ratio,
        "pred_final_right_m": float(pred_geom["final_right_m"]),
        "target_final_right_m": float(target_geom["final_right_m"]),
        "final_right_error_m": float(pred_geom["final_right_m"]) - float(target_geom["final_right_m"]),
        "pred_mean_abs_right_m": float(pred_geom["mean_abs_right_m"]),
        "target_mean_abs_right_m": float(target_geom["mean_abs_right_m"]),
        "pred_speed_mps": pred_path_length / max(horizon_s, 1e-6),
        "target_speed_mps": target_path_length / max(horizon_s, 1e-6),
        "predicted_brake_probability": pred_brake_probability,
        "target_should_brake": target_should_brake,
        "predicted_should_brake": predicted_should_brake,
        "target_throttle": float(target_control[1]) if len(target_control) >= 2 else 0.0,
        "predicted_throttle": float(pred_control[1]) if len(pred_control) >= 2 else 0.0,
        "target_brake": float(target_control[2]) if len(target_control) >= 3 else (1.0 if target_should_brake else 0.0),
        "predicted_brake": float(pred_control[2]) if len(pred_control) >= 3 else pred_brake_probability,
    }


def _waypoint_path_geometry(points: Sequence[Sequence[float]]) -> Dict[str, float]:
    converted = [_waypoint_forward_right(point) for point in points if len(point) >= 2]
    if not converted:
        return {
            "path_length_m": 0.0,
            "final_forward_m": 0.0,
            "final_right_m": 0.0,
            "mean_abs_right_m": 0.0,
        }
    path_length = 0.0
    previous = (0.0, 0.0)
    for point in converted:
        path_length += math.hypot(float(point[0]) - previous[0], float(point[1]) - previous[1])
        previous = (float(point[0]), float(point[1]))
    return {
        "path_length_m": path_length,
        "final_forward_m": float(converted[-1][0]),
        "final_right_m": float(converted[-1][1]),
        "mean_abs_right_m": mean(abs(float(point[1])) for point in converted),
    }


def _waypoint_forward_right(point: Sequence[float]) -> Tuple[float, float]:
    return -float(point[1]), float(point[0])


def _aggregate_prediction_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [
        "ade_m",
        "fde_m",
        "mean_abs_lateral_error_m",
        "pred_path_length_m",
        "target_path_length_m",
        "path_length_ratio",
        "pred_final_forward_m",
        "target_final_forward_m",
        "final_forward_ratio",
        "pred_mean_abs_right_m",
        "target_mean_abs_right_m",
        "pred_speed_mps",
        "target_speed_mps",
        "predicted_brake_probability",
    ]
    payload = {
        "sample_count": float(len(rows)),
        **{f"mean_{key}": mean(float(row.get(key) or 0.0) for row in rows) for key in numeric_keys},
        "underreach_rate": _fraction(rows, lambda row: float(row.get("path_length_ratio") or 0.0) < 0.65),
        "severe_underreach_rate": _fraction(rows, lambda row: float(row.get("path_length_ratio") or 0.0) < 0.40),
        "overshoot_rate": _fraction(rows, lambda row: float(row.get("path_length_ratio") or 0.0) > 1.35),
        "near_stop_prediction_rate": _fraction(rows, lambda row: float(row.get("pred_path_length_m") or 0.0) < 1.0),
        "high_lateral_error_rate": _fraction(rows, lambda row: float(row.get("mean_abs_lateral_error_m") or 0.0) > 1.0),
        "target_brake_positive_rate": _fraction(rows, lambda row: bool(row.get("target_should_brake"))),
        "predicted_brake_positive_rate": _fraction(rows, lambda row: bool(row.get("predicted_should_brake"))),
    }
    tp = sum(1 for row in rows if bool(row.get("predicted_should_brake")) and bool(row.get("target_should_brake")))
    fp = sum(1 for row in rows if bool(row.get("predicted_should_brake")) and not bool(row.get("target_should_brake")))
    fn = sum(1 for row in rows if not bool(row.get("predicted_should_brake")) and bool(row.get("target_should_brake")))
    tn = sum(1 for row in rows if not bool(row.get("predicted_should_brake")) and not bool(row.get("target_should_brake")))
    payload.update(
        {
            "brake_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "brake_precision": tp / max(tp + fp, 1),
            "brake_recall": tp / max(tp + fn, 1),
            "brake_f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
            "path_length_ratio_p10": _percentile([float(row.get("path_length_ratio") or 0.0) for row in rows], 0.10),
            "path_length_ratio_p50": _percentile([float(row.get("path_length_ratio") or 0.0) for row in rows], 0.50),
            "path_length_ratio_p90": _percentile([float(row.get("path_length_ratio") or 0.0) for row in rows], 0.90),
        }
    )
    return payload


def _group_rows_by_key(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return grouped


def _fraction(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(predicate(row))) / float(len(rows))


def _percentile(values: Sequence[float], q: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return 0.0
    index = max(min(int(round(float(q) * (len(finite) - 1))), len(finite) - 1), 0)
    return finite[index]


def _planner_readiness_assessment(aggregate: Mapping[str, float]) -> Dict[str, Any]:
    findings = []
    underreach = float(aggregate.get("underreach_rate") or 0.0)
    lateral = float(aggregate.get("high_lateral_error_rate") or 0.0)
    brake_f1 = float(aggregate.get("brake_f1") or 0.0)
    speed_ratio = float(aggregate.get("mean_pred_speed_mps") or 0.0) / max(float(aggregate.get("mean_target_speed_mps") or 0.0), 1e-6)
    if underreach > 0.35:
        findings.append("predicted_horizon_underreach")
    if lateral > 0.25:
        findings.append("large_lateral_prediction_error")
    if brake_f1 < 0.25 and float(aggregate.get("target_brake_positive_rate") or 0.0) > 0.05:
        findings.append("weak_brake_event_detection")
    if speed_ratio < 0.70:
        findings.append("low_predicted_speed_profile")
    if speed_ratio > 1.30:
        findings.append("high_predicted_speed_profile")
    status = "ready_for_closed_loop_diagnostics"
    if findings:
        status = "requires_planner_improvement_before_carla_rollout"
    return {
        "status": status,
        "findings": findings,
        "mean_predicted_to_target_speed_ratio": speed_ratio,
    }


def _write_prediction_diagnostic_outputs(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "planner_diagnostics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_prediction_diagnostic_csv(rows, output_dir / "planner_diagnostics_samples.csv")
    _write_family_diagnostic_csv(
        dict(report.get("scenario_family_diagnostics") or {}),
        output_dir / "planner_diagnostics_by_family.csv",
    )
    _plot_planner_diagnostics(rows, output_dir / "planner_diagnostics_overview.png")
    markdown = _render_prediction_diagnostic_markdown(report)
    (output_dir / "planner_diagnostics_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "planner_diagnostics_report.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_prediction_diagnostic_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "scenario_family",
        "ade_m",
        "fde_m",
        "mean_abs_lateral_error_m",
        "pred_path_length_m",
        "target_path_length_m",
        "path_length_ratio",
        "pred_final_forward_m",
        "target_final_forward_m",
        "final_forward_ratio",
        "pred_speed_mps",
        "target_speed_mps",
        "predicted_brake_probability",
        "target_should_brake",
        "predicted_should_brake",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_family_diagnostic_csv(metrics: Mapping[str, Mapping[str, float]], output_path: Path) -> None:
    fieldnames = [
        "scenario_family",
        "sample_count",
        "mean_ade_m",
        "mean_fde_m",
        "mean_final_forward_ratio",
        "underreach_rate",
        "severe_underreach_rate",
        "high_lateral_error_rate",
        "brake_f1",
        "target_brake_positive_rate",
        "predicted_brake_positive_rate",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family, row in sorted(metrics.items()):
            payload = {"scenario_family": family}
            payload.update({key: row.get(key, "") for key in fieldnames if key != "scenario_family"})
            writer.writerow(payload)


def _plot_planner_diagnostics(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=160)
    axes[0].hist([float(row.get("final_forward_ratio") or 0.0) for row in rows], bins=30, color="#2a9d8f", alpha=0.85)
    axes[0].axvline(1.0, color="#303030", lw=1.0)
    axes[0].set_title("final forward ratio")
    axes[0].set_xlabel("pred / target")
    axes[1].hist([float(row.get("mean_abs_lateral_error_m") or 0.0) for row in rows], bins=30, color="#e9c46a", alpha=0.9)
    axes[1].set_title("lateral error")
    axes[1].set_xlabel("m")
    axes[2].hist([float(row.get("predicted_brake_probability") or 0.0) for row in rows], bins=20, color="#e76f51", alpha=0.9)
    axes[2].set_title("brake probability")
    axes[2].set_xlabel("p")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_prediction_diagnostic_markdown(report: Mapping[str, Any]) -> str:
    aggregate = dict(report.get("aggregate") or {})
    readiness = dict(report.get("readiness") or {})
    lines = [
        "# Bench2Drive Vision Planner Diagnostics",
        "",
        f"- Samples: `{report.get('sample_count')}`",
        f"- Readiness status: `{readiness.get('status', '')}`",
        f"- Findings: `{', '.join(readiness.get('findings') or []) or 'none'}`",
        f"- Mean ADE: `{float(aggregate.get('mean_ade_m') or 0.0):.4f}` m",
        f"- Mean FDE: `{float(aggregate.get('mean_fde_m') or 0.0):.4f}` m",
        f"- Mean final-forward ratio: `{float(aggregate.get('mean_final_forward_ratio') or 0.0):.4f}`",
        f"- Underreach rate: `{float(aggregate.get('underreach_rate') or 0.0):.4f}`",
        f"- Severe underreach rate: `{float(aggregate.get('severe_underreach_rate') or 0.0):.4f}`",
        f"- High lateral-error rate: `{float(aggregate.get('high_lateral_error_rate') or 0.0):.4f}`",
        f"- Brake F1: `{float(aggregate.get('brake_f1') or 0.0):.4f}`",
        "",
        "| Scenario family | Samples | ADE | FDE | Forward ratio | Underreach | Lateral error rate | Brake F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, row in sorted(dict(report.get("scenario_family_diagnostics") or {}).items()):
        metrics = dict(row)
        lines.append(
            "| {family} | {samples:.0f} | {ade:.3f} | {fde:.3f} | {ratio:.3f} | {underreach:.3f} | {lateral:.3f} | {brake:.3f} |".format(
                family=family,
                samples=float(metrics.get("sample_count") or 0.0),
                ade=float(metrics.get("mean_ade_m") or 0.0),
                fde=float(metrics.get("mean_fde_m") or 0.0),
                ratio=float(metrics.get("mean_final_forward_ratio") or 0.0),
                underreach=float(metrics.get("underreach_rate") or 0.0),
                lateral=float(metrics.get("high_lateral_error_rate") or 0.0),
                brake=float(metrics.get("brake_f1") or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def _mean_metric_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(mean(float(row[key]) for row in rows if key in row)) for key in keys}


def _finalize_weighted_metrics(totals: Mapping[str, float], sample_count: int) -> Dict[str, float]:
    if sample_count <= 0:
        return {}
    return {key: float(value) / float(sample_count) for key, value in sorted(totals.items())}


def _finalize_weighted_tensor_metrics(totals: Mapping[str, Any], sample_count: int) -> Dict[str, float]:
    if sample_count <= 0:
        return {}
    return {
        key: float((value / float(sample_count)).detach().cpu())
        for key, value in sorted(totals.items())
    }


def _reduce_epoch_metrics(torch: Any, metrics: Mapping[str, float], device: Any, distributed: bool) -> Dict[str, float]:
    if not distributed or not metrics:
        return dict(metrics)
    reduced = dict(metrics)
    mean_keys = sorted(key for key in metrics if key not in {"sample_count", "samples_per_s", "epoch_runtime_s"})
    local_sample_count = float(metrics.get("sample_count", 0.0))

    sample_tensor = torch.tensor([local_sample_count], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(sample_tensor, op=torch.distributed.ReduceOp.SUM)
    sample_count = float(sample_tensor.item())

    if mean_keys:
        values = torch.tensor(
            [float(metrics[key]) * local_sample_count for key in mean_keys],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values = values / max(sample_count, 1e-6)
        for idx, key in enumerate(mean_keys):
            reduced[key] = float(values[idx].detach().cpu())

    runtime_tensor = torch.tensor([float(metrics.get("epoch_runtime_s", 0.0))], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(runtime_tensor, op=torch.distributed.ReduceOp.MAX)
    runtime_s = max(float(runtime_tensor.item()), 1e-6)
    reduced["sample_count"] = sample_count
    reduced["epoch_runtime_s"] = runtime_s
    reduced["samples_per_s"] = sample_count / runtime_s
    return reduced


def _save_planner_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    config: VisionE2EModelConfig,
    epoch_row: Mapping[str, Any],
    *,
    loss_config: VisionE2ELossConfig,
    data_parallel: bool,
) -> None:
    wrapped = data_parallel or model.__class__.__name__ == "DistributedDataParallel"
    checkpoint_model = _unwrap_parallel_model(model) if wrapped else model
    state_dict = checkpoint_model.state_dict()
    torch.save(
        {
            "schema": "bench2drive_vision_e2e_checkpoint_v1",
            "model_config": dict(config.__dict__),
            "loss_config": dict(loss_config.__dict__),
            "model_state_dict": state_dict,
            "epoch": int(epoch_row.get("epoch") or 0),
            "metrics": dict(epoch_row),
        },
        str(path),
    )


def _unwrap_parallel_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _print_training_epoch(
    epoch: int,
    epoch_count: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    train_text = _format_metric_summary(train_metrics)
    val_text = _format_metric_summary(val_metrics)
    print(f"[epoch {epoch}/{epoch_count}] train {train_text} | val {val_text}", flush=True)


def _format_metric_summary(metrics: Mapping[str, float]) -> str:
    if not metrics:
        return "n/a"
    return (
        f"loss={float(metrics.get('loss', 0.0)):.4f} "
        f"ade={float(metrics.get('ade_m', 0.0)):.4f} "
        f"fde={float(metrics.get('fde_m', 0.0)):.4f} "
        f"brake_f1={float(metrics.get('brake_f1', 0.0)):.4f}"
    )


def _write_training_outputs(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    history = list(report.get("history") or [])
    csv_path = output_dir / "training_history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ade_m",
        "train_fde_m",
        "val_loss",
        "val_ade_m",
        "val_fde_m",
        "val_brake_accuracy",
        "val_brake_precision",
        "val_brake_recall",
        "val_brake_f1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            train = dict(row.get("train") or {})
            val = dict(row.get("val") or {})
            writer.writerow(
                {
                    "epoch": row.get("epoch"),
                    "train_loss": train.get("loss"),
                    "train_ade_m": train.get("ade_m"),
                    "train_fde_m": train.get("fde_m"),
                    "val_loss": val.get("loss"),
                    "val_ade_m": val.get("ade_m"),
                    "val_fde_m": val.get("fde_m"),
                    "val_brake_accuracy": val.get("brake_accuracy"),
                    "val_brake_precision": val.get("brake_precision"),
                    "val_brake_recall": val.get("brake_recall"),
                    "val_brake_f1": val.get("brake_f1"),
                }
            )
    _plot_training_history(history, output_dir / "training_curves.png")
    markdown = _render_training_markdown(report)
    (output_dir / "training_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "training_report.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_eval_outputs(report: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pred_path = output_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    _write_scenario_family_metrics_csv(
        dict(report.get("scenario_family_metrics") or {}),
        output_dir / "scenario_family_metrics.csv",
    )
    _plot_prediction_examples(predictions[:6], output_dir / "prediction_examples.png")
    markdown = _render_eval_markdown(report)
    (output_dir / "evaluation_report.md").write_text(markdown, encoding="utf-8")
    (output_dir / "evaluation_report.html").write_text(_markdown_to_basic_html(markdown), encoding="utf-8")


def _write_scenario_family_metrics_csv(metrics: Mapping[str, Mapping[str, float]], output_path: Path) -> None:
    fieldnames = [
        "scenario_family",
        "sample_count",
        "ade_m",
        "fde_m",
        "brake_accuracy",
        "brake_precision",
        "brake_recall",
        "brake_f1",
        "brake_positive_rate",
        "brake_predicted_positive_rate",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family, row in sorted(metrics.items()):
            payload = {"scenario_family": family}
            payload.update({key: row.get(key, "") for key in fieldnames if key != "scenario_family"})
            writer.writerow(payload)


def _plot_training_history(history: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not history:
        return
    epochs = [int(row.get("epoch") or 0) for row in history]
    train_ade = [float(dict(row.get("train") or {}).get("ade_m") or 0.0) for row in history]
    val_ade = [float(dict(row.get("val") or {}).get("ade_m") or 0.0) for row in history]
    plt.figure(figsize=(8, 4.5), dpi=160)
    plt.plot(epochs, train_ade, marker="o", label="train ADE")
    if any(value > 0.0 for value in val_ade):
        plt.plot(epochs, val_ade, marker="o", label="val ADE")
    plt.xlabel("epoch")
    plt.ylabel("ADE (m)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_prediction_examples(predictions: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not predictions:
        return
    cols = 3
    rows = math.ceil(len(predictions) / cols)
    plt.figure(figsize=(cols * 4.0, rows * 3.5), dpi=160)
    for idx, row in enumerate(predictions, start=1):
        plt.subplot(rows, cols, idx)
        target = row.get("target_future_waypoints_ego") or []
        pred = row.get("predicted_future_waypoints_ego") or []
        if target:
            plt.plot([p[0] for p in target], [p[1] for p in target], marker="o", label="target")
        if pred:
            plt.plot([p[0] for p in pred], [p[1] for p in pred], marker="x", label="pred")
        plt.axhline(0, color="#999999", lw=0.8)
        plt.axvline(0, color="#999999", lw=0.8)
        plt.title(str(row.get("scenario_family") or ""), fontsize=9)
        plt.axis("equal")
        plt.grid(alpha=0.25)
        if idx == 1:
            plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _render_training_markdown(report: Mapping[str, Any]) -> str:
    latest = dict((list(report.get("history") or [{}])[-1]).get("val") or {})
    lines = [
        "# Bench2Drive Vision E2E Training",
        "",
        f"- Training samples: `{report.get('train_sample_count')}`",
        f"- Validation samples: `{report.get('val_sample_count')}`",
        f"- Model size: `{report.get('model_size')}`",
        f"- Architecture: `{report.get('architecture')}`",
        f"- Camera pooling: `{report.get('camera_pooling')}`",
        f"- Trajectory modes: `{report.get('trajectory_modes')}`",
        f"- Trajectory selection: `{report.get('trajectory_selection')}`",
        f"- Trajectory top-k: `{report.get('trajectory_top_k')}`",
        f"- Trajectory temperature: `{report.get('trajectory_temperature')}`",
        f"- Dropout: `{report.get('dropout')}`",
        f"- Selection metric: `{report.get('selection_metric')}`",
        f"- Device: `{report.get('device')}`",
        f"- Distributed: `{report.get('distributed')}`",
        f"- Distributed world size: `{report.get('distributed_world_size')}`",
        f"- Precision: `{report.get('precision')}`",
        f"- TF32 enabled: `{report.get('allow_tf32')}`",
        f"- cuDNN benchmark: `{report.get('cudnn_benchmark')}`",
        f"- Nonfinite check interval: `{report.get('nonfinite_check_interval')}`",
        f"- Runtime: `{report.get('runtime_s')}` seconds",
    ]
    if latest:
        lines.extend(
            [
                f"- Validation ADE: `{latest.get('ade_m'):.4f}`",
                f"- Validation FDE: `{latest.get('fde_m'):.4f}`",
                f"- Brake accuracy: `{latest.get('brake_accuracy'):.4f}`",
                f"- Brake F1: `{latest.get('brake_f1', 0.0):.4f}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_eval_markdown(report: Mapping[str, Any]) -> str:
    metrics = dict(report.get("metrics") or {})
    lines = [
        "# Bench2Drive Vision E2E Evaluation",
        "",
        f"- Split: `{report.get('split')}`",
        f"- Samples: `{report.get('sample_count')}`",
        f"- ADE: `{metrics.get('ade_m', 0.0):.4f}`",
        f"- FDE: `{metrics.get('fde_m', 0.0):.4f}`",
        f"- Brake accuracy: `{metrics.get('brake_accuracy', 0.0):.4f}`",
        f"- Brake precision: `{metrics.get('brake_precision', 0.0):.4f}`",
        f"- Brake recall: `{metrics.get('brake_recall', 0.0):.4f}`",
        f"- Brake F1: `{metrics.get('brake_f1', 0.0):.4f}`",
    ]
    return "\n".join(lines) + "\n"


def _markdown_to_basic_html(markdown: str) -> str:
    body = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    return "<!doctype html><html><body>" + "\n".join(body) + "</body></html>\n"
