from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import math
import pickle
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from nusc_scene_agent.data_utils import DEFAULT_DATAROOT
from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, write_artifact_manifest
from nusc_scene_agent.world_model_benchmark import (
    adapt_and_evaluate_nuscenes_forecast_predictions,
    compare_world_model_evaluations,
    run_nuscenes_forecast_baselines,
)


DEFAULT_CONTEXTVAE_REPO = Path("external/ContextVAE")
DEFAULT_CONTEXTVAE_CHECKPOINT = DEFAULT_CONTEXTVAE_REPO / "models" / "nuscenes_res18"
DEFAULT_CONTEXTVAE_OUTPUT = Path("outputs/contextvae_world_model_study")
CONTEXTVAE_NUSCENES_CHECKPOINT_URL = (
    "https://github.com/xupei0610/ContextVAE/releases/download/pretrained/nuscenes_res18"
)
CONTEXTVAE_DEFAULT_OB_HORIZON = 5
CONTEXTVAE_DEFAULT_MIN_OB_HORIZON = 2
CONTEXTVAE_DEFAULT_PRED_HORIZON = 12
CONTEXTVAE_DEFAULT_MODE_COUNT = 5
CONTEXTVAE_DEFAULT_CLUSTERING_SAMPLES = 2000


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _agent_group(category_name: str) -> Optional[str]:
    if "cycle" in category_name:
        return "CYCLE"
    if "vehicle" in category_name:
        return "VEHICLE"
    if "human" in category_name:
        return "PEDESTRIAN"
    return None


def _load_contextvae_config(config_path: Path):
    package_name = "_contextvae_config_pkg"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(config_path.parent)]
        sys.modules[package_name] = package

    train_path = config_path.parent / "nuscenes_train.py"
    if train_path.exists():
        train_module_name = "{0}.nuscenes_train".format(package_name)
        if train_module_name not in sys.modules:
            train_spec = importlib.util.spec_from_file_location(train_module_name, train_path)
            if train_spec is None or train_spec.loader is None:
                raise RuntimeError("Failed to load ContextVAE base config: {0}".format(train_path))
            train_module = importlib.util.module_from_spec(train_spec)
            sys.modules[train_module_name] = train_module
            train_spec.loader.exec_module(train_module)

    module_name = "{0}.{1}".format(package_name, config_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load ContextVAE config: {0}".format(config_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _previous_samples(nusc: NuScenes, sample_token: str, count: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample = nusc.get("sample", sample_token)
    while sample.get("prev") and len(rows) < count:
        sample = nusc.get("sample", sample["prev"])
        rows.append(sample)
    rows.reverse()
    return rows


def _next_samples(nusc: NuScenes, sample_token: str, count: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample = nusc.get("sample", sample_token)
    while sample.get("next") and len(rows) < count:
        sample = nusc.get("sample", sample["next"])
        rows.append(sample)
    return rows


class _CompatibleMapExplorer(NuScenesMapExplorer):
    @staticmethod
    def mask_for_lines(lines: object, mask: np.ndarray) -> np.ndarray:
        # The SDK still iterates MultiLineString directly, which Shapely 2 removed.
        parts = lines.geoms if lines.geom_type == "MultiLineString" else [lines]
        for line in parts:
            if not line.is_empty:
                NuScenesMapExplorer.mask_for_lines(line, mask)
        return mask


def _semantic_map_patch(
    dataroot: Path,
    map_name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    map_scale: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_map = NuScenesMap(dataroot=str(dataroot), map_name=map_name)
    scene_map.explorer = _CompatibleMapExplorer(scene_map)
    x_min = int(math.floor(x_min - 300.0))
    x_max = int(math.ceil(x_max + 300.0))
    y_min = int(math.floor(y_min - 300.0))
    y_max = int(math.ceil(y_max + 300.0))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    canvas_size = (
        int(math.ceil((y_max - y_min) * map_scale)),
        int(math.ceil((x_max - x_min) * map_scale)),
    )
    height = float(canvas_size[0]) / float(map_scale)
    width = float(canvas_size[1]) / float(map_scale)
    patch_box = (x_center, y_center, height, width)
    mask = scene_map.get_map_mask(
        patch_box,
        0.0,
        ["lane", "road_segment", "drivable_area", "road_divider", "lane_divider", "ped_crossing"],
        canvas_size,
    )
    semantic_map = np.stack((np.max(mask[:3], axis=0) * 0.75 + mask[5] * 0.25, mask[3], mask[4]))
    semantic_map = (semantic_map * 2.0 - 1.0).clip(-1.0, 1.0)
    homography = np.array(
        [
            [0.0, -map_scale, (y_center + 0.5 * height) * map_scale],
            [map_scale, 0.0, -(x_center - 0.5 * width) * map_scale],
            [0.0, 0.0, 1.0],
        ]
    )
    semantic_map = semantic_map[:, ::-1, :]
    return semantic_map, homography


def _download_file(url: str, output_path: Path) -> Path:
    output_path = output_path.resolve()
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "nuscenes-scene-agent/0.1"})
    with urlopen(request) as response, output_path.open("wb") as handle:  # noqa: S310
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return output_path


def ensure_contextvae_checkpoint(
    checkpoint_path: Path = DEFAULT_CONTEXTVAE_CHECKPOINT,
    url: str = CONTEXTVAE_NUSCENES_CHECKPOINT_URL,
) -> Path:
    return _download_file(url, checkpoint_path)


def _case_file_name(case: Dict[str, object]) -> str:
    return "{0}_{1}".format(
        str(case.get("instance_token") or ""),
        str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or ""),
    )


def _prepare_case_records(
    nusc: NuScenes,
    case: Dict[str, object],
    ob_horizon: int,
    pred_horizon: int,
) -> Optional[Dict[str, object]]:
    rollout_anchor_sample_token = str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or "")
    if not rollout_anchor_sample_token:
        return None
    instance_token = str(case.get("instance_token") or "")
    anchor_sample = nusc.get("sample", rollout_anchor_sample_token)
    previous_samples = _previous_samples(nusc, rollout_anchor_sample_token, ob_horizon - 1)
    future_samples = _next_samples(nusc, rollout_anchor_sample_token, pred_horizon)
    observation_samples = previous_samples + [anchor_sample]
    if len(observation_samples) < CONTEXTVAE_DEFAULT_MIN_OB_HORIZON or len(future_samples) < pred_horizon:
        return None

    scene = nusc.get("scene", anchor_sample["scene_token"])
    map_name = nusc.get("log", scene["log_token"])["location"]
    samples = observation_samples + future_samples
    anchor_frame_idx = len(observation_samples) - 1

    records: Dict[str, List[Tuple[int, float, float, float, str]]] = defaultdict(list)
    bounds = {
        "x_min": float("inf"),
        "x_max": float("-inf"),
        "y_min": float("inf"),
        "y_max": float("-inf"),
    }
    agent_order: List[str] = []

    for frame_idx, sample in enumerate(samples):
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category_name = str(ann["category_name"])
            if not ann["attribute_tokens"] and "vehicle" not in category_name:
                continue
            group = _agent_group(category_name)
            if group is None:
                continue
            agent_token = str(ann["instance_token"])
            if agent_token not in records:
                agent_order.append(agent_token)
            x_global = float(ann["translation"][0])
            y_global = float(ann["translation"][1])
            heading = float(Quaternion(ann["rotation"]).yaw_pitch_roll[0])
            records[agent_token].append((frame_idx, x_global, y_global, heading, group))
            bounds["x_min"] = min(bounds["x_min"], x_global)
            bounds["x_max"] = max(bounds["x_max"], x_global)
            bounds["y_min"] = min(bounds["y_min"], y_global)
            bounds["y_max"] = max(bounds["y_max"], y_global)

        camera_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])
        ego_x = float(ego_pose["translation"][0])
        ego_y = float(ego_pose["translation"][1])
        ego_heading = float(Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0])
        if "EGO" not in records:
            agent_order.append("EGO")
        records["EGO"].append((frame_idx, ego_x, ego_y, ego_heading, "EGO"))
        bounds["x_min"] = min(bounds["x_min"], ego_x)
        bounds["x_max"] = max(bounds["x_max"], ego_x)
        bounds["y_min"] = min(bounds["y_min"], ego_y)
        bounds["y_max"] = max(bounds["y_max"], ego_y)

    if instance_token not in records:
        return None

    target_track = {frame_idx: (x_global, y_global, heading) for frame_idx, x_global, y_global, heading, _ in records[instance_token]}
    if set(target_track) != set(range(len(samples))):
        return None
    transform = target_track.get(0)
    if transform is None:
        return None

    file_lines: List[str] = []
    agent_ids: Dict[str, int] = {}
    next_agent_id = 1
    for agent_token in agent_order:
        rows = records.get(agent_token) or []
        if len(rows) <= 1:
            continue
        if agent_token not in agent_ids:
            agent_ids[agent_token] = next_agent_id
            next_agent_id += 1
        agent_id = agent_ids[agent_token]
        for frame_idx, x_global, y_global, heading, group in rows:
            row_group = group
            if agent_token == instance_token:
                row_group = "{0}/CHALLENGE".format(group)
            file_lines.append(
                "{0} {1} {2:.6f} {3:.6f} {4:.6f} {5}".format(
                    frame_idx,
                    agent_id,
                    x_global,
                    y_global,
                    heading,
                    row_group,
                )
            )

    window_entries = []
    for current_idx in range(1, len(observation_samples)):
        current_sample = observation_samples[current_idx]
        window_entries.append(
            {
                "current_sample_token": str(current_sample["token"]),
                "is_rollout_anchor": bool(current_idx == anchor_frame_idx),
                "available_observation_steps": int(current_idx + 1),
                "transform_origin_x_global": round(float(transform[0]), 6),
                "transform_origin_y_global": round(float(transform[1]), 6),
                "transform_heading_rad": round(float(transform[2]), 6),
                "instance_token": instance_token,
                "rollout_anchor_sample_token": rollout_anchor_sample_token,
                "file_name": _case_file_name(case),
            }
        )
    return {
        "scene_name": str(scene["name"]),
        "scene_token": str(scene["token"]),
        "map_name": str(map_name),
        "rollout_anchor_sample_token": rollout_anchor_sample_token,
        "anchor_sample_token": str(case.get("anchor_sample_token") or ""),
        "instance_token": instance_token,
        "file_name": _case_file_name(case),
        "file_lines": file_lines,
        "info_line": "0 {0}".format(map_name),
        "window_entries": window_entries,
        "bounds": {key: round(float(value), 6) for key, value in bounds.items()},
        "available_observation_steps": len(observation_samples),
        "available_future_steps": len(future_samples),
        "observation_sample_tokens": [str(sample["token"]) for sample in observation_samples],
        "future_sample_tokens": [str(sample["token"]) for sample in future_samples],
        "reference_scene_name": str(case.get("reference_scene_name") or ""),
        "benchmark_group": str(case.get("benchmark_group") or ""),
    }


def prepare_contextvae_world_model_dataset(
    benchmark_path: Path,
    dataroot: Path,
    output_dir: Path,
    version: str = "v1.0-trainval",
    ob_horizon: int = CONTEXTVAE_DEFAULT_OB_HORIZON,
    pred_horizon: int = CONTEXTVAE_DEFAULT_PRED_HORIZON,
    map_scale: int = 1,
) -> Dict[str, object]:
    benchmark = _load_json(benchmark_path)
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    output_dir = output_dir.resolve()
    preparation_key = hashlib.sha256(
        Path(benchmark_path).read_bytes() + Path(__file__).read_bytes()
        + json.dumps([version, ob_horizon, pred_horizon, map_scale]).encode()
    ).hexdigest()[:16]
    val_root = output_dir / "prepared" / preparation_key / "val"
    map_root = output_dir / "prepared" / preparation_key / "map"
    val_root.mkdir(parents=True, exist_ok=True)
    map_root.mkdir(parents=True, exist_ok=True)

    prepared_cases: List[Dict[str, object]] = []
    skipped_cases: List[Dict[str, object]] = []
    compatible_cases: List[Dict[str, object]] = []
    map_bounds: Dict[str, Dict[str, float]] = {}
    seen_forecast_keys = set()

    for case in list(benchmark.get("cases") or []):
        forecast_key = _case_file_name(case)
        if forecast_key in seen_forecast_keys:
            skipped_cases.append({"reference_case_key": str(case.get("reference_case_key") or ""),
                                  "reason": "duplicate_instance_anchor", "forecast_key": forecast_key})
            continue
        prepared = _prepare_case_records(
            nusc=nusc,
            case=dict(case),
            ob_horizon=ob_horizon,
            pred_horizon=pred_horizon,
        )
        if prepared is None:
            skipped_cases.append(
                {
                    "reference_case_key": str(case.get("reference_case_key") or ""),
                    "benchmark_group": str(case.get("benchmark_group") or ""),
                    "instance_token": str(case.get("instance_token") or ""),
                    "rollout_anchor_sample_token": str(case.get("rollout_anchor_sample_token") or case.get("anchor_sample_token") or ""),
                    "reason": "insufficient_or_discontinuous_actor_context_or_future",
                }
            )
            continue

        seen_forecast_keys.add(forecast_key)

        scene_dir = val_root / prepared["scene_name"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        txt_path = scene_dir / "{0}.txt".format(prepared["file_name"])
        info_path = scene_dir / "{0}.info".format(prepared["file_name"])
        txt_path.write_text("\n".join(prepared["file_lines"]) + "\n", encoding="utf-8")
        info_path.write_text(str(prepared["info_line"]) + "\n", encoding="utf-8")

        prepared["relative_txt_path"] = str(txt_path.relative_to(output_dir))
        prepared["relative_info_path"] = str(info_path.relative_to(output_dir))
        prepared_cases.append(prepared)
        compatible_cases.append(dict(case))
        bounds = map_bounds.setdefault(
            prepared["map_name"],
            {"x_min": float("inf"), "x_max": float("-inf"), "y_min": float("inf"), "y_max": float("-inf")},
        )
        prepared_bounds = dict(prepared["bounds"])
        bounds["x_min"] = min(bounds["x_min"], _safe_float(prepared_bounds["x_min"]))
        bounds["x_max"] = max(bounds["x_max"], _safe_float(prepared_bounds["x_max"]))
        bounds["y_min"] = min(bounds["y_min"], _safe_float(prepared_bounds["y_min"]))
        bounds["y_max"] = max(bounds["y_max"], _safe_float(prepared_bounds["y_max"]))

    for map_name, bounds in map_bounds.items():
        semantic_map, homography = _semantic_map_patch(
            dataroot=dataroot,
            map_name=map_name,
            x_min=bounds["x_min"],
            x_max=bounds["x_max"],
            y_min=bounds["y_min"],
            y_max=bounds["y_max"],
            map_scale=map_scale,
        )
        with (map_root / "{0}.pkl".format(map_name)).open("wb") as handle:
            pickle.dump((semantic_map, homography), handle)

    ordered_windows: List[Dict[str, object]] = []
    for case_info in sorted(prepared_cases, key=lambda item: str(item["relative_txt_path"])):
        anchor_window = next(
            window for window in list(case_info["window_entries"]) if bool(window.get("is_rollout_anchor"))
        )
        ordered_windows.append(
            {
                "relative_txt_path": str(case_info["relative_txt_path"]),
                "scene_name": str(case_info["scene_name"]),
                "scene_token": str(case_info["scene_token"]),
                "map_name": str(case_info["map_name"]),
                "file_name": str(case_info["file_name"]),
                "instance_token": str(anchor_window["instance_token"]),
                "rollout_anchor_sample_token": str(anchor_window["rollout_anchor_sample_token"]),
                "current_sample_token": str(anchor_window["current_sample_token"]),
                "is_rollout_anchor": True,
                "available_observation_steps": int(anchor_window["available_observation_steps"]),
                "transform_origin_x_global": _safe_float(anchor_window["transform_origin_x_global"]),
                "transform_origin_y_global": _safe_float(anchor_window["transform_origin_y_global"]),
                "transform_heading_rad": _safe_float(anchor_window["transform_heading_rad"]),
            }
        )

    subset_path = output_dir / "forecast_compatible_world_model_benchmark.json"
    subset = {
        "metadata": {
            **dict(benchmark.get("metadata") or {}),
            "generator": "contextvae_forecast_compatible_subset_v1",
            "source_benchmark": str(Path(benchmark_path).resolve()),
            "case_count": len(compatible_cases),
            "skipped_case_count": len(skipped_cases),
            "required_future_steps": pred_horizon,
        },
        "cases": compatible_cases,
    }
    _write_json(subset_path, subset)

    manifest = {
        "metadata": {
            "generator": "contextvae_dataset_prep_v1",
            "source_benchmark": str(Path(benchmark_path).resolve()),
            "subset_benchmark": str(subset_path.resolve()),
            "version": version,
            "case_count": len(prepared_cases),
            "window_count": len(ordered_windows),
            "skipped_case_count": len(skipped_cases),
            "ob_horizon": ob_horizon,
            "pred_horizon": pred_horizon,
            "map_scale": map_scale,
            "preparation_key": preparation_key,
            "val_root": str(val_root),
            "map_root": str(map_root),
        },
        "cases": [
            {
                key: value
                for key, value in case_info.items()
                if key
                not in {
                    "file_lines",
                    "info_line",
                }
            }
            for case_info in sorted(prepared_cases, key=lambda item: str(item["relative_txt_path"]))
        ],
        "windows": ordered_windows,
        "skipped_cases": skipped_cases,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "subset_benchmark_path": str(subset_path),
        "case_count": len(prepared_cases),
        "window_count": len(ordered_windows),
        "skipped_case_count": len(skipped_cases),
    }


def _contextvae_local_to_global(
    trajectory: Sequence[Sequence[object]],
    origin_x: float,
    origin_y: float,
    heading_rad: float,
) -> List[List[float]]:
    cos_yaw = math.cos(float(heading_rad))
    sin_yaw = math.sin(float(heading_rad))
    origin = np.array([float(origin_x), float(origin_y)], dtype=float)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=float)
    restored = []
    for point in trajectory:
        xy = np.array([_safe_float(point[0]), _safe_float(point[1])], dtype=float)
        global_xy = rotation @ (xy - origin) + origin
        restored.append([round(float(global_xy[0]), 4), round(float(global_xy[1]), 4)])
    return restored


def export_contextvae_nuscenes_forecasts(
    benchmark_path: Path,
    dataroot: Path,
    output_path: Path,
    dataset_dir: Path,
    repo_dir: Path = DEFAULT_CONTEXTVAE_REPO,
    checkpoint_path: Path = DEFAULT_CONTEXTVAE_CHECKPOINT,
    device: str = "",
    batch_size: int = 8,
    mode_count: int = CONTEXTVAE_DEFAULT_MODE_COUNT,
    clustering_samples: int = CONTEXTVAE_DEFAULT_CLUSTERING_SAMPLES,
    seed: int = 1,
) -> Dict[str, object]:
    repo_dir = repo_dir.resolve()
    checkpoint_path = ensure_contextvae_checkpoint(checkpoint_path)
    if not (repo_dir / "context_vae.py").exists():
        raise FileNotFoundError(
            "ContextVAE repo not found at {0}. Clone https://github.com/xupei0610/ContextVAE first.".format(repo_dir)
        )

    manifest = _load_json(dataset_dir / "manifest.json")
    windows = [dict(item) for item in list(manifest.get("windows") or []) if bool(item.get("is_rollout_anchor"))]
    subset_benchmark_path = Path(str((manifest.get("metadata") or {}).get("subset_benchmark") or benchmark_path))
    if not windows:
        raise RuntimeError("No forecast-compatible rollout anchors were prepared for ContextVAE export.")

    import torch

    repo_dir_str = str(repo_dir)
    if repo_dir_str not in sys.path:
        sys.path.insert(0, repo_dir_str)
    context_vae_module = importlib.import_module("context_vae")
    data_module = importlib.import_module("data")
    utils_module = importlib.import_module("utils")

    class _ThreadProcessPoolAdapter(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            kwargs.pop("mp_context", None)
            super().__init__(*args, **kwargs)

    data_module.ProcessPoolExecutor = _ThreadProcessPoolAdapter
    config = _load_contextvae_config(repo_dir / "config" / "nuscenes_eval.py")
    config.test_dataloader = dict(config.test_dataloader)
    config.test_dataloader["batch_size"] = min(max(1, batch_size), len(windows))
    config.pred_samples = int(mode_count)
    config.clustering = int(clustering_samples)
    utils_module.seed(seed)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = data_module.Dataloader(
        [str(Path(manifest.get("metadata", {}).get("val_root") or dataset_dir / "val").resolve())],
        map_dir=str(Path(manifest.get("metadata", {}).get("map_root") or dataset_dir / "map").resolve()),
        batch_first=False,
        device="cpu",
        shuffle=False,
        **config.test_dataloader,
    )
    if len(dataset) != len(manifest.get("windows") or []):
        raise RuntimeError(
            "Prepared manifest/window count mismatch: dataset yielded {0} windows, manifest recorded {1}.".format(
                len(dataset),
                len(manifest.get("windows") or []),
            )
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_sampler=dataset.batch_sampler,
        num_workers=0,
    )

    model = context_vae_module.ContextVAE(**config.model)
    checkpoint = torch.load(str(checkpoint_path), map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(torch_device)
    model.eval()

    predictions: List[Dict[str, object]] = []
    window_index = 0
    with torch.no_grad():
        for batch in dataloader:
            batch_tensors = [tensor.to(torch_device) for tensor in batch]
            x, _y, neighbor, *extras = batch_tensors
            if int(config.clustering) > 0:
                predicted = model(x, neighbor, *extras, n_predictions=int(config.clustering))
            else:
                predicted = model(x, neighbor, *extras, n_predictions=int(config.pred_samples))

            batch_modes: List[List[List[List[float]]]] = []
            batch_probabilities: List[List[float]] = []
            for batch_item_idx in range(predicted.size(-2)):
                raw_modes = predicted[..., batch_item_idx, :].detach().cpu().numpy()
                if int(config.clustering) > 0:
                    clustered_modes, counts = utils_module.clustering(raw_modes, n_samples=int(config.pred_samples))
                    local_modes = [clustered_modes[mode_idx].tolist() for mode_idx in range(clustered_modes.shape[0])]
                    total = float(sum(counts)) or 1.0
                    probabilities = [round(float(count) / total, 6) for count in counts]
                else:
                    local_modes = [raw_modes[mode_idx].tolist() for mode_idx in range(raw_modes.shape[0])]
                    probability = round(1.0 / max(len(local_modes), 1), 6)
                    probabilities = [probability for _ in local_modes]
                batch_modes.append(local_modes)
                batch_probabilities.append(probabilities)

            batch_size_actual = len(batch_modes)
            for batch_item_idx in range(batch_size_actual):
                window = dict(manifest["windows"][window_index + batch_item_idx])
                if not bool(window.get("is_rollout_anchor")):
                    continue
                global_modes = [
                    _contextvae_local_to_global(
                        trajectory=mode,
                        origin_x=_safe_float(window.get("transform_origin_x_global")),
                        origin_y=_safe_float(window.get("transform_origin_y_global")),
                        heading_rad=_safe_float(window.get("transform_heading_rad")),
                    )
                    for mode in batch_modes[batch_item_idx]
                ]
                predictions.append(
                    {
                        "instance": str(window["instance_token"]),
                        "sample": str(window["rollout_anchor_sample_token"]),
                        "prediction": global_modes,
                        "probabilities": batch_probabilities[batch_item_idx],
                    }
                )
            window_index += batch_size_actual

    payload = {
        "metadata": {
            "generator": "contextvae_nuscenes_export_v1",
            "source_benchmark": str(Path(benchmark_path).resolve()),
            "subset_benchmark": str(subset_benchmark_path.resolve()),
            "dataset_dir": str(dataset_dir.resolve()),
            "repo_dir": str(repo_dir),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "device": str(torch_device),
            "prediction_count": len(predictions),
            "mode_count": int(mode_count),
            "clustering_samples": int(clustering_samples),
        },
        "predictions": predictions,
    }
    _write_json(output_path, payload)
    return {
        "output_path": str(output_path.resolve()),
        "subset_benchmark_path": str(subset_benchmark_path.resolve()),
        "prediction_count": len(predictions),
        "mode_count": int(mode_count),
    }


def run_contextvae_world_model_study(
    benchmark_path: Path,
    dataroot: Path = DEFAULT_DATAROOT,
    version: str = "v1.0-trainval",
    output_dir: Path = DEFAULT_CONTEXTVAE_OUTPUT,
    repo_dir: Path = DEFAULT_CONTEXTVAE_REPO,
    checkpoint_path: Path = DEFAULT_CONTEXTVAE_CHECKPOINT,
    device: str = "",
    batch_size: int = 8,
    mode_count: int = CONTEXTVAE_DEFAULT_MODE_COUNT,
    clustering_samples: int = CONTEXTVAE_DEFAULT_CLUSTERING_SAMPLES,
    map_scale: int = 1,
    seed: int = 1,
) -> Dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "contextvae_dataset"
    prep_metadata = prepare_contextvae_world_model_dataset(
        benchmark_path=benchmark_path,
        dataroot=dataroot,
        output_dir=dataset_dir,
        version=version,
        map_scale=map_scale,
    )
    subset_benchmark_path = Path(prep_metadata["subset_benchmark_path"])
    prediction_path = output_dir / "contextvae_nuscenes_forecasts.json"
    export_metadata = export_contextvae_nuscenes_forecasts(
        benchmark_path=benchmark_path,
        dataroot=dataroot,
        output_path=prediction_path,
        dataset_dir=dataset_dir,
        repo_dir=repo_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        mode_count=mode_count,
        clustering_samples=clustering_samples,
        seed=seed,
    )

    evaluation_dir = output_dir / "contextvae"
    evaluation = adapt_and_evaluate_nuscenes_forecast_predictions(
        benchmark_path=subset_benchmark_path,
        input_path=prediction_path,
        output_dir=evaluation_dir,
        mode_selection="top_probability",
        profile_name="contextvae",
    )

    baseline_root = output_dir / "nuscenes_baselines"
    run_nuscenes_forecast_baselines(
        benchmark_path=subset_benchmark_path,
        dataroot=dataroot,
        version=version,
        output_dir=baseline_root,
        mode_selection="top_probability",
    )
    comparison = compare_world_model_evaluations(
        evaluation_dirs=[
            evaluation_dir,
            baseline_root / "cv_heading",
            baseline_root / "physics_oracle",
        ],
        output_dir=output_dir / "comparison",
    )
    summary = {
        "schema": "contextvae_world_model_study_v1",
        "provenance": collect_runtime_provenance(),
        "protocol": {"benchmark": str(benchmark_path), "checkpoint": str(checkpoint_path),
                     "seed": seed, "mode_count": mode_count, "clustering_samples": clustering_samples,
                     "evaluation_subset": "unique actor-anchor pairs with continuous observations"},
        "preparation": prep_metadata,
        "export": export_metadata,
        "evaluation": evaluation,
        "comparison": comparison,
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "contextvae_study_manifest.json", summary)
    write_artifact_manifest(output_dir, [
        build_artifact_entry(path, "result", "contextvae_study", output_dir)
        for path in [output_dir / "contextvae_study_manifest.json", prediction_path, subset_benchmark_path,
                     output_dir / "comparison/world_model_comparison.json", benchmark_path, checkpoint_path]
    ])
    return summary
