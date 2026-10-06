from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, verify_artifact_manifest, write_artifact_manifest
from nusc_scene_agent.bench2drive_closed_loop import compare_bench2drive_closed_loop_reports, run_bench2drive_vision_closed_loop
from nusc_scene_agent.bench2drive_e2e import (
    _read_manifest_rows,
    compare_vision_e2e_prediction_sets,
    evaluate_vision_e2e_planner,
)
from nusc_scene_agent.scenario_taxonomy import load_scenario_taxonomy, map_scenario_labels


def select_training_rows(rows: Sequence[Mapping[str, Any]], *, fraction: float, seed: int,
                         policy: str, case_library: Path | None = None) -> tuple[list[dict], dict]:
    if not 0 < fraction <= 1 or policy not in {"uniform", "risk_prior"}:
        raise ValueError("Sampling requires 0 < fraction <= 1 and a supported policy.")
    train = [dict(row) for row in rows if row.get("split") == "train"]
    unchanged = [dict(row) for row in rows if row.get("split") != "train"]
    budget = max(1, int(len(train) * fraction))
    weights = np.ones(len(train), dtype=float)
    family_prior: Counter[str] = Counter()
    if policy == "risk_prior":
        if case_library is None:
            raise ValueError("Risk-prior sampling requires the mined nuScenes case library.")
        taxonomy = load_scenario_taxonomy()
        cases = json.loads(case_library.read_text())
        for case in cases:
            if case.get("passed"):
                labels = case.get("matched_behaviors") or case.get("all_behaviors") or []
                family_prior.update(map_scenario_labels("nuscenes", labels, taxonomy)["family_ids"])
        if not family_prior:
            raise ValueError("No validated mined families are available for risk-prior sampling.")
        labels = [map_scenario_labels("bench2drive", [str(row.get("scenario_family") or "")], taxonomy)["family_ids"]
                  for row in train]
        availability = Counter(family for families in labels for family in families)
        risk_mass = np.asarray([sum(family_prior[family] / max(availability[family], 1) for family in families)
                                for families in labels], dtype=float)
        if risk_mass.sum() <= 0:
            raise ValueError("The mined risk prior has no matching training scenarios.")
        # Half uniform mass preserves coverage of unmapped scenario families.
        weights = 0.5 / len(train) + 0.5 * risk_mass / risk_mass.sum()
    indices = np.random.default_rng(seed).choice(len(train), budget, replace=False, p=weights / weights.sum())
    selected = [train[int(index)] for index in sorted(indices)]
    return selected + unchanged, {
        "policy": policy, "seed": seed, "fraction": fraction, "training_budget": budget,
        "available_train_samples": len(train), "mined_family_prior": dict(family_prior),
        "selected_family_counts": dict(Counter(str(row.get("scenario_family") or "") for row in selected)),
        "validation_and_test_unchanged": True,
        "interpretation": "taxonomy-guided data selection; no claim of simulator-to-real transfer",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _training_command(settings: Mapping[str, Any], *, manifest: Path, output: Path, seed: int, workers: int) -> list[str]:
    command = [sys.executable]
    if workers > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={workers}"]
    command += ["-m", "nusc_scene_agent", "train-bench2drive-vision-planner", "--manifest", str(manifest),
                "--output", str(output), "--seed", str(seed), "--verbose"]
    for key, value in settings.items():
        if isinstance(value, bool):
            if value:
                command.append("--" + key.replace("_", "-"))
        else:
            command += ["--" + key.replace("_", "-"), str(value)]
    return command


def _evaluate_study_run(manifest: Path, checkpoint: Path, run_dir: Path, *, device: str, image_size: int) -> None:
    source_dir = Path(__file__).parent
    source_files = [source_dir / name for name in ["bench2drive_e2e.py", "bench2drive_closed_loop.py",
                                                  "carla_closed_loop.py", "geometry.py"]]
    request = {"manifest_sha256": build_artifact_entry(manifest, "input", "manifest").sha256,
               "checkpoint_sha256": build_artifact_entry(checkpoint, "model", "checkpoint").sha256,
               "evaluation_source_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest(),
               "split": "test", "max_cases": 0, "max_frames_per_clip": 21, "image_size": image_size}
    stamp = run_dir / "evaluation_complete.json"
    if stamp.exists():
        try:
            previous = json.loads(stamp.read_text())
            verification = verify_artifact_manifest(stamp)
            if previous.get("request") == request and verification["valid"] and verification["checked_files"] == 3:
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
    evaluate_vision_e2e_planner(manifest, checkpoint, run_dir / "eval_test", split="test", batch_size=64,
                               image_size=image_size, device=device, num_workers=4)
    run_bench2drive_vision_closed_loop(manifest, checkpoint, run_dir / "replay", split="test", max_cases=0,
                                      max_frames_per_clip=21, image_size=image_size, device=device)
    _write_json(stamp, {"request": request, "artifacts": [
        build_artifact_entry(run_dir / filename, "evaluation", "controlled_study", run_dir).to_dict()
        for filename in ["eval_test/evaluation_report.json", "eval_test/predictions.jsonl", "replay/closed_loop_report.json"]
    ]})


def run_bench2drive_study(config: Mapping[str, Any], *, selected_seeds: Sequence[int] | None = None,
                         finalize: bool = True) -> dict[str, Any]:
    import torch

    torch.set_num_threads(4)
    output = Path(str(config["output"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(str(config["manifest"]))
    rows = _read_manifest_rows(manifest)
    manifest_hash = build_artifact_entry(manifest, "input", "manifest").sha256
    source_hash = hashlib.sha256(Path(__file__).with_name("bench2drive_e2e.py").read_bytes()).hexdigest()
    seeds = [int(seed) for seed in config.get("seeds", [7, 17, 27])]
    arms = list(config["arms"])
    if len(set(seeds)) != len(seeds) or len({str(arm["name"]) for arm in arms}) != len(arms):
        raise ValueError("Study seeds and arm names must be unique.")
    active_seeds = seeds if selected_seeds is None else list(selected_seeds)
    if not active_seeds or len(set(active_seeds)) != len(active_seeds) or not set(active_seeds).issubset(seeds):
        raise ValueError("Selected seeds must be a nonempty, unique subset of the study seeds.")
    if finalize and set(active_seeds) != set(seeds):
        raise ValueError("A partial seed worker cannot publish the complete study summary.")
    progress_path = output / ("study_progress.json" if finalize else
                              "study_progress_seeds_" + "_".join(map(str, active_seeds)) + ".json")
    results = []
    for seed in active_seeds:
        for arm in arms:
            name = str(arm["name"])
            run_dir = output / f"seed_{seed}" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            settings = {**dict(config.get("training") or {}), **dict(arm.get("training") or {})}
            selected, sampling = select_training_rows(
                rows, fraction=float(arm.get("fraction", 1.0)), seed=seed,
                policy=str(arm.get("sampling", "uniform")),
                case_library=Path(config["case_library"]) if config.get("case_library") else None,
            )
            request = {"manifest_sha256": manifest_hash, "training_source_sha256": source_hash, "seed": seed,
                       "settings": settings, "sampling": sampling, "world_size": int(config.get("world_size", 1))}
            request_path = run_dir / "request.json"
            if request_path.exists() and json.loads(request_path.read_text()) != request:
                raise ValueError(f"Existing run has a different protocol: {run_dir}; use a new output directory.")
            _write_json(request_path, request)
            run_manifest = run_dir / "manifest.jsonl"
            with run_manifest.open("w", encoding="utf-8") as stream:
                for row in selected:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            if not (run_dir / "training_report.json").exists():
                print(f"Training {name}, seed {seed}, {sampling['training_budget']} samples", flush=True)
                env = {**os.environ, "OMP_NUM_THREADS": "4"}
                with (run_dir / "train.log").open("w") as log:
                    subprocess.run(_training_command(settings, manifest=run_manifest, output=run_dir, seed=seed,
                                                      workers=int(config.get("world_size", 1))),
                                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            checkpoint = run_dir / "vision_e2e_planner_best.pt"
            evaluation_dir = run_dir / "eval_test"
            replay_dir = run_dir / "replay"
            _evaluate_study_run(manifest, checkpoint, run_dir, device=str(config.get("device", "cuda:0")),
                                 image_size=int(settings.get("image_size", 160)))
            training = json.loads((run_dir / "training_report.json").read_text())
            evaluation = json.loads((evaluation_dir / "evaluation_report.json").read_text())
            replay = json.loads((replay_dir / "closed_loop_report.json").read_text())
            results.append({"arm": name, "seed": seed, "output": str(run_dir), "sampling": sampling,
                            "training_epochs": training["epochs"], "open_loop": evaluation["metrics"],
                            "replay": replay["comparison"]["metrics"]})
            _write_json(progress_path, {"completed_runs": results, "expected_run_count": len(active_seeds) * len(arms)})
    if not finalize:
        return {"output_dir": str(output), "selected_seeds": active_seeds, "runs": results, "finalized": False}
    comparisons = []
    for pair in config.get("comparisons", []):
        for seed in seeds:
            baseline, candidate = [output / f"seed_{seed}" / str(name) for name in pair]
            pair_dir = output / "comparisons" / f"{pair[1]}_vs_{pair[0]}" / f"seed_{seed}"
            open_loop = compare_vision_e2e_prediction_sets(
                baseline / "eval_test/predictions.jsonl", candidate / "eval_test/predictions.jsonl", pair_dir / "open_loop",
                baseline_label=pair[0], candidate_label=pair[1], bootstrap_replicates=10000,
            )
            replay = compare_bench2drive_closed_loop_reports(
                baseline / "replay/closed_loop_report.json", candidate / "replay/closed_loop_report.json", pair_dir / "replay",
                baseline_label=pair[0], candidate_label=pair[1], bootstrap_replicates=10000,
            )
            comparisons.append({"baseline": pair[0], "candidate": pair[1], "seed": seed,
                                "open_loop": open_loop["metrics"], "replay": replay["metrics"]})
    aggregate = []
    for arm in arms:
        runs = [run for run in results if run["arm"] == arm["name"]]
        for domain in ("open_loop", "replay"):
            for metric in runs[0][domain]:
                if not all(isinstance(run[domain][metric], (int, float)) for run in runs):
                    continue
                values = [float(run[domain][metric]) for run in runs]
                aggregate.append({"arm": arm["name"], "domain": domain, "metric": metric,
                                  "mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                  "values": values})
    summary = {"schema": "bench2drive_controlled_study_v1", "output_dir": str(output), "protocol": dict(config),
               "provenance": collect_runtime_provenance(), "selection": "validation checkpoint; fixed selection, no fitted calibration",
               "test_policy": "all held-out clips; no result-based filtering or checkpoint promotion",
               "uncertainty": "per-seed paired clip/case bootstrap; cross-seed mean and standard deviation",
               "runs": results, "aggregate": aggregate, "comparisons": comparisons}
    _write_json(output / "study_summary.json", summary)
    lines = ["# Controlled Bench2Drive Study", "", summary["selection"], "", summary["test_policy"], "",
             "| Arm | Metric | Mean | Seed SD |", "| --- | --- | ---: | ---: |"]
    for row in aggregate:
        if row["metric"] in {"ade_m", "fde_m", "brake_f1", "lateral_mae_m", "mean_closed_loop_ade_m", "mean_route_completion"}:
            sd = f"{row['seed_std']:.4f}" if row["seed_std"] is not None else "n/a"
            lines.append(f"| {row['arm']} | {row['domain']}: {row['metric']} | {row['mean']:.4f} | {sd} |")
    (output / "study_summary.md").write_text("\n".join(lines) + "\n")
    artifact_paths = [output / "study_summary.json", output / "study_summary.md"]
    for run in results:
        artifact_paths.extend(Path(run["output"]) / name for name in [
            "request.json", "manifest.jsonl", "training_report.json", "vision_e2e_planner_best.pt",
            "evaluation_complete.json", "eval_test/evaluation_report.json", "eval_test/predictions.jsonl",
            "replay/closed_loop_report.json",
        ])
    artifact_paths.extend(sorted((output / "comparisons").rglob("*comparison.json")))
    write_artifact_manifest(output, [build_artifact_entry(path, "result", "controlled_study", output) for path in artifact_paths])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--defer-summary", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    run_bench2drive_study(config.get("bench2drive_study", config), selected_seeds=args.seeds, finalize=not args.defer_summary)
