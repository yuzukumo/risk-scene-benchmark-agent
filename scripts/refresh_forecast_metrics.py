"""Re-evaluate archived predictions without drawing new stochastic trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, verify_artifact_manifest, write_artifact_manifest
from nusc_scene_agent.world_model_benchmark import compare_world_model_evaluations, evaluate_world_model_predictions


def refresh(study_dir: Path) -> None:
    study_dir = study_dir.resolve()
    verification = verify_artifact_manifest(study_dir / "artifact_manifest.json")
    if not verification["valid"]:
        raise ValueError(f"Forecast inputs or outputs changed: {verification['failures']}")
    manifest_path = study_dir / "contextvae_study_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    directories = [study_dir / "contextvae", study_dir / "nuscenes_baselines/cv_heading",
                   study_dir / "nuscenes_baselines/physics_oracle"]
    metrics_paths = []
    for output in directories:
        metrics_path = output / "world_model_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        updated = evaluate_world_model_predictions(Path(metrics["benchmark_path"]), Path(metrics["predictions_path"]),
                                                   output, profile_name=metrics["profile_name"])
        if output == directories[0]:
            manifest["evaluation"]["overview"] = updated["overview"]
        metrics_paths.extend([metrics_path, Path(metrics["predictions_path"])])
    compare_world_model_evaluations(directories[1:], study_dir / "nuscenes_baselines/comparison")
    manifest["comparison"] = compare_world_model_evaluations(directories, study_dir / "comparison")
    manifest["evaluation_refresh"] = {"provenance": collect_runtime_provenance(),
                                       "scope": "metrics and uncertainty only; archived predictions are unchanged"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    artifact_paths = [manifest_path, study_dir / "contextvae_nuscenes_forecasts.json",
                      Path(manifest["preparation"]["subset_benchmark_path"]),
                      study_dir / "comparison/world_model_comparison.json",
                      Path(manifest["protocol"]["benchmark"]), Path(manifest["protocol"]["checkpoint"]), *metrics_paths]
    write_artifact_manifest(study_dir, [build_artifact_entry(path, "result", "contextvae_study", study_dir)
                                        for path in dict.fromkeys(artifact_paths)])
    print(f"Refreshed {study_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    refresh(parser.parse_args().study)
