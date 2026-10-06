from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from nuscenes.utils.splits import create_splits_scenes

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, verify_artifact_manifest, write_artifact_manifest
from nusc_scene_agent.world_model_benchmark import compare_world_model_evaluations, evaluate_world_model_predictions


def audit_forecast_generalization(study: Path, development_benchmarks: Sequence[Path], output: Path) -> dict:
    study, output = study.resolve(), output.resolve()
    if not development_benchmarks:
        raise ValueError("Development benchmark paths are required for a scene-disjoint audit.")
    if not verify_artifact_manifest(study / "artifact_manifest.json")["valid"]:
        raise ValueError("The archived forecast artifacts failed verification.")
    manifest_path = study / "contextvae_study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_benchmark = Path(manifest["preparation"]["subset_benchmark_path"])
    benchmark = json.loads(source_benchmark.read_text())
    development_scenes = set()
    for path in development_benchmarks:
        for case in json.loads(path.read_text())["cases"]:
            if not case.get("scene_token"):
                raise ValueError(f"Development case has no scene identity: {path}")
            development_scenes.add(case["scene_token"])
    validation_names = set(create_splits_scenes()["val"])
    cases = benchmark["cases"]
    if any(not case.get("scene_token") or case["scene_name"] not in validation_names for case in cases):
        raise ValueError("The source forecasts must have explicit official-validation scene identities.")
    excluded = [case for case in cases if case["scene_token"] in development_scenes]
    retained = [case for case in cases if case["scene_token"] not in development_scenes]
    if not retained:
        raise ValueError("No forecast cases remain after excluding development scenes.")
    output.mkdir(parents=True, exist_ok=True)
    benchmark = {**benchmark, "cases": retained, "metadata": {
        **benchmark.get("metadata", {}), "case_count": len(retained),
        "source_benchmark": str(source_benchmark),
        "subset_policy": "official validation; exclude every scene in the declared development benchmarks",
    }}
    benchmark_path = output / "forecast_benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    inputs = [manifest_path, source_benchmark, *[path.resolve() for path in development_benchmarks]]
    result_paths = [benchmark_path]
    directories = []
    for name, directory in [("contextvae", "contextvae"), ("cv_heading", "nuscenes_baselines/cv_heading"),
                            ("physics_oracle", "nuscenes_baselines/physics_oracle")]:
        source_metrics = json.loads((study / directory / "world_model_metrics.json").read_text())
        predictions = Path(source_metrics["predictions_path"])
        target = output / name
        evaluate_world_model_predictions(benchmark_path, predictions, target, profile_name=name)
        directories.append(target)
        inputs.append(predictions)
        result_paths.append(target / "world_model_metrics.json")
    compare_world_model_evaluations(directories, output / "comparison")
    report = {
        "schema": "forecast_generalization_audit_v1", "output_dir": str(output),
        "provenance": collect_runtime_provenance(), "source_case_count": len(cases),
        "case_count": len(retained), "scene_count": len({case["scene_token"] for case in retained}),
        "development_scene_count": len(development_scenes), "excluded_case_count": len(excluded),
        "excluded_scene_count": len({case["scene_token"] for case in excluded}),
        "excluded_cases": [{"benchmark_group": case["benchmark_group"], "scene_token": case["scene_token"],
                            "reason": "development_scene"} for case in excluded],
        "shared_development_scene_count": 0,
        "selection_policy": "scene identity only; independent of prediction quality; archived predictions unchanged",
        "benchmark_path": str(benchmark_path),
        "comparison_path": str(output / "comparison/world_model_comparison.json"),
    }
    report_path = output / "generalization_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    result_paths.extend([report_path, Path(report["comparison_path"])])
    write_artifact_manifest(output, [*[build_artifact_entry(path, "input", "forecast_audit") for path in inputs],
                                     *[build_artifact_entry(path, "result", "forecast_audit", output) for path in result_paths]])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_forecast_generalization(args.study, args.development, args.output), indent=2))
