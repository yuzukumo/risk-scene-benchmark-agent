"""Measure model-only inference cost for the controlled planner checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, write_artifact_manifest
from nusc_scene_agent.bench2drive_e2e import VisionE2EModelConfig, _build_vision_e2e_model


def benchmark(config_path: Path, output: Path, *, seed: int, warmup: int, repeats: int) -> None:
    if warmup < 1 or repeats < 1:
        raise ValueError("Warmup and repeat counts must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Select an otherwise idle CUDA GPU for the inference measurement.")
    config = yaml.safe_load(config_path.read_text())["bench2drive_study"]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    rows, inputs = [], []
    for arm in config["arms"]:
        checkpoint_path = Path(config["output"]) / f"seed_{seed}" / arm["name"] / "vision_e2e_planner_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = VisionE2EModelConfig(**checkpoint["model_config"])
        model = _build_vision_e2e_model(model_config).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        images = torch.randn(1, model_config.camera_count, 3, model_config.image_size, model_config.image_size, device=device)
        route = torch.zeros(1, model_config.route_feature_dim, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(images, route)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(repeats):
                start = perf_counter()
                prediction = model(images, route)
                torch.cuda.synchronize()
                times.append((perf_counter() - start) * 1000)
            if not torch.isfinite(prediction["future"]).all():
                raise ValueError(f"Nonfinite trajectory output: {checkpoint_path}")
        row = {"arm": arm["name"], "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
               "mean_ms": float(np.mean(times)), "p50_ms": float(np.percentile(times, 50)),
               "p95_ms": float(np.percentile(times, 95)),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2)}
        rows.append(row)
        inputs.append(build_artifact_entry(checkpoint_path.resolve(), "model", "inference_cost"))
        print(json.dumps(row), flush=True)
        del model, images, route, prediction, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "planner_inference_cost_v1", "provenance": collect_runtime_provenance(),
              "protocol": {"device": torch.cuda.get_device_name(), "precision": "float32", "batch_size": 1,
                           "warmup": warmup, "repeats": repeats, "checkpoint_seed": seed,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "timing": "synchronized wall time; synthetic on-device inputs; excludes camera capture, loading and control"},
              "results": rows}
    report_path = output / "inference_cost.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    write_artifact_manifest(output, [*inputs, build_artifact_entry(report_path, "result", "inference_cost", output),
                                     build_artifact_entry(Path(__file__).resolve(), "source", "inference_cost")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/bench2drive_controlled_study.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/planner_inference_v1"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    benchmark(args.config, args.output, seed=args.seed, warmup=args.warmup, repeats=args.repeats)
