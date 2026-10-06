"""Publish compact, traceable results from completed local experiments."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.splits import create_splits_scenes

from nusc_scene_agent.artifact_manifest import build_artifact_entry, verify_artifact_manifest


ARM_LABELS = {
    "baseline": "Baseline", "geometric_supervision": "Geometric supervision",
    "route_only": "Navigation only", "spatial_4x4": "Spatial 4x4",
    "random_half": "Random half data", "risk_prior_half": "Risk-prior half data",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _replace_section(path: Path, name: str, content: str) -> None:
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    document = path.read_text()
    before, remainder = document.split(start, 1)
    _, after = remainder.split(end, 1)
    path.write_text(before + start + "\n" + content.strip() + "\n" + end + after)


def _planner_table(study: dict) -> str:
    indexed = {(row["arm"], row["domain"], row["metric"]): row for row in study["aggregate"]}
    columns = [("open_loop", "ade_m"), ("open_loop", "fde_m"), ("open_loop", "brake_f1"),
               ("replay", "mean_route_completion")]
    lines = ["Means +/- sample standard deviation across three training seeds.", "",
             "| Arm | Test ADE (m) | Test FDE (m) | Brake F1 | Replay completion |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for arm in study["protocol"]["arms"]:
        values = [indexed[(arm["name"], domain, metric)] for domain, metric in columns]
        lines.append("| " + ARM_LABELS[arm["name"]] + " | " + " | ".join(
            f"{row['mean']:.3f} +/- {row['seed_std']:.3f}" for row in values) + " |")
    lines += ["", "![Controlled planner results](../assets/planner_controlled_results.png)", "",
              "The figure shows individual training seeds and their mean. Its error bars are seed standard deviations, "
              "not confidence intervals from independent new driving scenes. Per-seed paired comparisons remain in the study outputs."]
    return "\n".join(lines)


def _planner_figure(study: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    metrics = [("open_loop", "ade_m", "Test ADE (m)", "#19758b"),
               ("open_loop", "brake_f1", "Brake F1", "#32855b"),
               ("replay", "mean_route_completion", "Replay completion", "#a04362")]
    names = [arm["name"] for arm in study["protocol"]["arms"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    for ax, (domain, metric, label, color) in zip(axes, metrics):
        for index, name in enumerate(names):
            values = [run[domain][metric] for run in study["runs"] if run["arm"] == name]
            ax.errorbar(np.mean(values), index, xerr=np.std(values, ddof=1), color=color,
                        fmt="o", capsize=4, markersize=6, linewidth=1.5)
            ax.scatter(values, np.full(len(values), index) + np.linspace(-0.13, 0.13, len(values)),
                       color=color, s=18, alpha=0.45)
        ax.set_xlabel(label)
        ax.set_yticks(range(len(names)), [ARM_LABELS[name] for name in names])
        ax.grid(axis="x", color="#e6e9eb", linewidth=0.7)
        ax.set_axisbelow(True)
        if metric != "ade_m":
            ax.set_xlim(0, 1)
    axes[0].invert_yaxis()
    fig.suptitle("Controlled planner study | 3 seeds | 97 held-out clips", fontsize=13, y=0.98)
    fig.text(0.52, 0.015, "Logged-sensor replay has fixed images and no visual feedback.", ha="center", color="#535b62", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=2.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _forecast_snapshot(root: Path, paths: list[Path], *, audit: bool = False) -> dict:
    verification = verify_artifact_manifest(root / "artifact_manifest.json")
    if not verification["valid"]:
        raise ValueError(f"Forecast artifacts failed verification: {root}")
    manifest_path = root / ("generalization_audit.json" if audit else "contextvae_study_manifest.json")
    comparison_path = root / "comparison/world_model_comparison.json"
    manifest, comparison = _read(manifest_path), _read(comparison_path)
    benchmark_path = Path(manifest["benchmark_path"] if audit else manifest["preparation"]["subset_benchmark_path"])
    benchmark = _read(benchmark_path)
    cases = benchmark["cases"]
    source_benchmark = Path(benchmark["metadata"]["source_benchmark"] if audit else manifest["protocol"]["benchmark"])
    durations = [case["future_duration_s"] for case in cases]
    splits = create_splits_scenes()
    scene_splits = {name: split for split in ["train", "val"] for name in splits[split]}
    paths.extend([manifest_path, comparison_path, benchmark_path, source_benchmark])
    return {"case_count": len(cases), "scene_count": len({case["scene_name"] for case in cases}),
            "source_case_count": len(_read(source_benchmark)["cases"]),
            "future_duration_s": {"min": min(durations), "max": max(durations),
                                  "approximately_six_second_count": sum(value >= 5.95 for value in durations)},
            "official_split_counts": dict(Counter(scene_splits.get(case["scene_name"], "unknown") for case in cases)),
            "comparison_protocol": comparison["overview"],
            "development_audit": {key: manifest[key] for key in ["excluded_case_count", "excluded_scene_count",
                                                                 "shared_development_scene_count", "selection_policy"]} if audit else None,
            "profiles": comparison["profiles"], "paired_comparisons": comparison["paired_profile_comparisons"]}


def _forecast_table(forecasts: dict) -> str:
    validation = forecasts["scene_disjoint_validation"]
    full_validation = forecasts["official_validation"]
    full_profiles = {row["name"]: row for row in full_validation["profiles"]}
    audit = validation["development_audit"]
    lines = [f"Official-validation mining retains {full_validation['case_count']} compatible actor-anchor pairs from "
             f"{full_validation['source_case_count']} forecast cases after continuity checks. "
             f"A scene-identity audit excludes {audit['excluded_case_count']} cases from {audit['excluded_scene_count']} "
             "scenes already present in the declared development benchmarks. The primary comparison below uses the remaining "
             f"{validation['case_count']} cases from {validation['scene_count']} scenes, with zero development-scene overlap. "
             "Exclusion depends only on scene identity; the archived predictions are unchanged. "
             "This validation slice is excluded from the development failure-query feedback loop.", "",
             "| Model | Common cases | ADE (m) | FDE (m) | Risk fidelity |",
             "| --- | ---: | ---: | ---: | ---: |"]
    labels = {"cv_heading": "Constant velocity and heading", "contextvae": "ContextVAE",
              "physics_oracle": "Physics oracle, GT-selected upper bound"}
    profiles = {row["name"]: row for row in validation["profiles"]}
    for name, label in labels.items():
        row = profiles[name]
        lines.append(f"| {label} | {row['case_count']} | {row['mean_ade_m']:.3f} | {row['mean_fde_m']:.3f} "
                     f"| {row['mean_risk_fidelity_score']:.3f} |")
    paired = next(row for row in validation["paired_comparisons"]
                  if row["profile_a"] == "cv_heading" and row["profile_b"] == "contextvae")
    delta = paired["deltas"]["ade_m"]
    horizon = validation["future_duration_s"]
    lines += ["", f"ContextVAE minus constant-velocity ADE is {delta['profile_b_minus_profile_a']:+.3f} m "
              f"(paired scene-bootstrap 95% interval [{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}] m; "
              f"{delta['cluster_count']} scene clusters). Lower ADE is better. "
              "Physics oracle selects a motion model using future ground truth and is not deployable.", "",
              f"Target durations range from {horizon['min']:.2f} to {horizon['max']:.2f} seconds; "
              f"{horizon['approximately_six_second_count']} cases cover approximately 6 seconds. "
              "Full-horizon coverage means covering each case's available targets, not a uniform 6-second benchmark.", "",
              f"The complete {full_validation['case_count']}-case official-validation slice is retained as a secondary result: "
              f"CV ADE {full_profiles['cv_heading']['mean_ade_m']:.3f} m and "
              f"ContextVAE ADE {full_profiles['contextvae']['mean_ade_m']:.3f} m. It includes development-scene overlap.", "",
              "The other slices are development diagnostics because they include official training scenes. "
              "They are not pooled with the primary validation slice:", "",
              "| Development slice | Cases / scenes | Train / val cases | CV ADE (m) | ContextVAE ADE (m) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, label in [("development", "Original"), ("expanded_development", "Expanded")]:
        result = forecasts[name]
        rows = {row["name"]: row for row in result["profiles"]}
        splits = result["official_split_counts"]
        lines.append(f"| {label} | {result['case_count']} / {result['scene_count']} | "
                     f"{splits.get('train', 0)} / {splits.get('val', 0)} | {rows['cv_heading']['mean_ade_m']:.3f} | "
                     f"{rows['contextvae']['mean_ade_m']:.3f} |")
    lines += ["", "Sources and paired intervals are retained in [results_snapshot.json](results_snapshot.json). "
              "These are mined slices with substantial exclusions, not the official nuScenes prediction benchmark."]
    return "\n".join(lines)


def _retrieval_table(retriever: dict, reranking: dict) -> str:
    integrity, overview = retriever["split_integrity"], reranking["overview"]
    count = overview["query_count"]
    rows = [("Weak-rule groups", f"{retriever['group_count']:,}"),
            ("Train / validation groups", f"{retriever['train_group_count']:,} / {retriever['validation_group_count']:,}"),
            ("Train / validation candidate scenes", f"{integrity['train_scene_count']} / {integrity['validation_scene_count']}"),
            ("Shared candidate scenes", len(integrity["shared_scene_tokens"])),
            ("Shared annotations", integrity["shared_annotation_count"]),
            ("Validation weak-rule consistency@1", f"{retriever['validation_metrics']['recall_at_1']:.3f}"),
            ("Failure-query acceptance@1, rule / learned", f"{overview['rule_pass_at_1']}/{count} / {overview['learned_pass_at_1']}/{count}"),
            ("Failure-query acceptance@K, rule / learned", f"{overview['rule_pass_at_k']}/{count} / {overview['learned_pass_at_k']}/{count}")]
    lines = ["| Quantity | Corrected result |", "| --- | ---: |", *[f"| {label} | {value} |" for label, value in rows]]
    lines += ["", f"Learned-minus-rule mean top-1 validation quality is {overview['mean_top1_score_delta']:+.4f}; "
              f"mean best-candidate quality is {overview['mean_best_score_delta']:+.4f}. "
              f"The recorded selection policy is `{overview['selection_policy']}` "
              f"(learned final ranker selected: {str(overview['final_ranker_selected']).lower()}; "
              f"learned candidate generator selected: {str(overview['candidate_generator_selected']).lower()}).", "",
              "Sources: `outputs/learned_retriever_trainval_v3/training_report.json` and "
              "`outputs/failure_aware_reranking_eval_v4/failure_aware_reranking_eval.json`."]
    return "\n".join(lines)


def publish(root: Path) -> None:
    study_dir = root / "outputs/bench2drive_controlled_study_v2"
    study_path = study_dir / "study_summary.json"
    study = _read(study_path)
    expected = {(seed, arm["name"]) for seed in study["protocol"]["seeds"] for arm in study["protocol"]["arms"]}
    actual = {(run["seed"], run["arm"]) for run in study["runs"]}
    if actual != expected or len(study["runs"]) != len(expected) or len(study["protocol"]["seeds"]) != 3:
        raise ValueError("Only a complete three-seed study can be published.")
    for run in study["runs"]:
        checked = verify_artifact_manifest(root / run["output"] / "evaluation_complete.json")
        if not checked["valid"] or checked["checked_files"] != 3:
            raise ValueError(f"Evaluation artifacts failed verification: {run['output']}")
    carla_path = root / "outputs/carla_fixed_evaluation_v3/fixed_evaluation.json"
    coverage_path = root / "outputs/risk_case_expansion_v2/expansion_report.json"
    retriever_path = root / "outputs/learned_retriever_trainval_v3/training_report.json"
    reranking_path = root / "outputs/failure_aware_reranking_eval_v4/failure_aware_reranking_eval.json"
    carla, coverage, retriever, reranking = [_read(path) for path in [carla_path, coverage_path, retriever_path, reranking_path]]
    for directory in [study_dir, carla_path.parent, coverage_path.parent, retriever_path.parent, reranking_path.parent]:
        if not verify_artifact_manifest(directory / "artifact_manifest.json")["valid"]:
            raise ValueError(f"Result artifacts failed verification: {directory}")
    source_paths = [study_path, carla_path, coverage_path, retriever_path, reranking_path]
    inference_path = root / "outputs/planner_inference_v1/inference_cost.json"
    inference = None
    if inference_path.exists():
        if not verify_artifact_manifest(inference_path.parent / "artifact_manifest.json")["valid"]:
            raise ValueError("Inference-cost artifacts failed verification.")
        inference = _read(inference_path)
        source_paths.append(inference_path)
    forecasts = {name: _forecast_snapshot(root / path, source_paths) for name, path in {
        "development": "outputs/contextvae_world_model_study_v3",
        "expanded_development": "outputs/contextvae_expanded_study_v1",
        "official_validation": "outputs/forecast_validation_study_v1/contextvae",
    }.items()}
    forecasts["scene_disjoint_validation"] = _forecast_snapshot(
        root / "outputs/forecast_validation_study_v1/scene_disjoint_audit", source_paths, audit=True)
    validation = forecasts["official_validation"]
    if validation["official_split_counts"] != {"val": validation["case_count"]}:
        raise ValueError("The official-validation forecast result contains non-validation scenes.")
    audit = forecasts["scene_disjoint_validation"]
    if audit["official_split_counts"] != {"val": audit["case_count"]} or audit["development_audit"]["shared_development_scene_count"]:
        raise ValueError("The audited validation result contains non-validation or development scenes.")
    snapshot = {
        "schema": "project_evaluation_snapshot_v1",
        "planner": {"protocol": study["protocol"], "aggregate": study["aggregate"], "comparisons": study["comparisons"]},
        "carla": {key: value for key, value in carla.items() if key not in {"provenance", "attempts"}},
        "coverage": {key: coverage[key] for key in ["case_count", "scene_count", "family_counts", "forecast"]},
        "retriever": {key: retriever[key] for key in ["group_count", "train_group_count", "validation_group_count", "split_integrity", "label_semantics"]},
        "failure_reranking": reranking["overview"], "forecasts": forecasts,
        "inference_cost": inference,
        "sources": [build_artifact_entry(path, "source", "evaluation_snapshot", root).to_dict() for path in source_paths],
    }
    (root / "docs/results_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    _planner_figure(study, root / "assets/planner_controlled_results.png")
    document = root / "docs/benchmark_snapshot.md"
    planner_content = _planner_table(study)
    if inference is not None:
        protocol = inference["protocol"]
        planner_content += (f"\n\nModel-only inference on {protocol['device']}, batch size 1, FP32, "
                            f"{protocol['warmup']} warmup passes and {protocol['repeats']} timed passes:\n\n"
                            "| Arm | Parameters (M) | Median (ms) | P95 (ms) |\n| --- | ---: | ---: | ---: |\n")
        planner_content += "\n".join(
            f"| {ARM_LABELS[row['arm']]} | {row['parameter_count'] / 1e6:.3f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} |"
            for row in inference["results"])
        planner_content += "\n\nInputs are synthetic and already on the device. These timings exclude camera capture, decoding, data loading and control."
    _replace_section(document, "CONTROLLED_RESULTS", planner_content)
    _replace_section(document, "FORECAST_RESULTS", _forecast_table(forecasts))
    _replace_section(document, "RETRIEVAL_RESULTS", _retrieval_table(retriever, reranking))
    collision_scenarios = Counter(item["scenario"] for item in carla["attempts"]
                                  if item.get("metrics", {}).get("collision_count", 0) > 0)
    collision_detail = ", ".join(f"`{name}`: {count} attempts" for name, count in collision_scenarios.items()) or "none"
    _replace_section(document, "CARLA_RESULTS", (
        f"The fixed suite completed {carla['completed_count']}/{carla['attempt_count']} attempts with {carla['error_count']} simulator errors. "
        f"It passed {carla['passed_count']}/{carla['attempt_count']} attempts ({carla['success_rate']:.0%}), "
        f"with {carla['collision_count']} recorded collisions and mean route completion {carla['mean_route_completion']:.1%}.\n\n"
        f"Collision scenarios: {collision_detail}. These are results for the retained legacy checkpoint under this local protocol. "
        "They are not results for the newly trained ablation arms or for the official Bench2Drive routes."
    ))
    _replace_section(document, "COVERAGE_RESULTS", (
        f"The expanded library retains {coverage['case_count']} cases from {coverage['scene_count']} scenes: "
        + ", ".join(f"{name}: {count}" for name, count in coverage["family_counts"].items()) + ". "
        + f"{coverage['forecast']['case_count']} cases have at least 3 seconds of future targets. "
        + "The uneven family counts and every rejected candidate are retained in `outputs/risk_case_expansion_v2/expansion_report.json`."
    ))
    print(f"Published {root / 'docs/results_snapshot.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    publish(parser.parse_args().root.resolve())
