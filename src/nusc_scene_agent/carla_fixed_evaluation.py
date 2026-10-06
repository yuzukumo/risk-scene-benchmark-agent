from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import yaml

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, write_artifact_manifest
from nusc_scene_agent.carla_vision_closed_loop import run_carla_vision_closed_loop


def _run_isolated_attempt(parameters: Mapping[str, Any], run_dir: Path) -> dict:
    request = run_dir / "request.json"
    request.write_text(json.dumps(parameters, default=str, indent=2))
    try:
        with (run_dir / "client.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "nusc_scene_agent.carla_fixed_evaluation", "--attempt", str(request)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        return json.loads((run_dir / "carla_vision_closed_loop_report.json").read_text())
    finally:
        process_file = run_dir / "carla_server_process.json"
        if process_file.exists():
            pid = int(json.loads(process_file.read_text())["pid"])
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"CarlaUE4" in command and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass


def run_fixed_carla_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(config["output"]))
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(config)
    protocol["checkpoint_sha256"] = build_artifact_entry(Path(config["checkpoint"]), "model", "checkpoint").sha256
    protocol["source_sha256"] = hashlib.sha256(Path(__file__).with_name("carla_vision_closed_loop.py").read_bytes()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing CARLA output uses a different protocol; choose a new output directory.")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    attempts = []
    for seed in config["seeds"]:
        for scenario in config["scenarios"]:
            run_dir = output / f"seed_{seed}" / str(scenario["name"])
            run_dir.mkdir(parents=True, exist_ok=True)
            record_path = run_dir / "attempt.json"
            if record_path.exists():
                attempts.append(json.loads(record_path.read_text()))
                continue
            print(f"CARLA fixed evaluation: seed {seed}, {scenario['name']}", flush=True)
            try:
                report = _run_isolated_attempt({
                    **dict(config.get("settings") or {}), "checkpoint_path": str(config["checkpoint"]),
                    "output_dir": str(run_dir), "seed": int(seed), "scenario_name": str(scenario["name"]),
                    "scenario_type": str(scenario["type"]), "spawn_index": int(scenario["spawn_index"]),
                    "scenario_params": dict(scenario.get("parameters") or {}),
                }, run_dir)
                metrics = report["metrics"]
                attribution = report["control_attribution"]
                passed = (metrics["collision_count"] == 0 and metrics["route_completion"] >= float(config["min_completion"])
                          and attribution["safety_override_ratio"] <= float(config["max_safety_override_ratio"]))
                record = {"seed": seed, "scenario": scenario["name"], "status": "completed", "passed": passed,
                          "metrics": metrics, "control_attribution": attribution,
                          "report": str(run_dir / "carla_vision_closed_loop_report.json")}
            except Exception as error:
                # Infrastructure failures remain in the denominator and are never promoted to successes.
                record = {"seed": seed, "scenario": scenario["name"], "status": "error", "passed": False,
                          "error": f"{type(error).__name__}: {error}"}
            record_path.write_text(json.dumps(record, indent=2))
            attempts.append(record)
            (output / "progress.json").write_text(json.dumps({"attempts": attempts}, indent=2))
    completed = [item for item in attempts if item["status"] == "completed"]
    summary = {"schema": "carla_fixed_evaluation_v1", "output_dir": str(output), "protocol": protocol,
               "provenance": collect_runtime_provenance(),
               "evaluation_boundary": "fixed local scenarios; not the official Bench2Drive benchmark",
               "attempt_count": len(attempts), "completed_count": len(completed),
               "error_count": len(attempts) - len(completed), "passed_count": sum(item["passed"] for item in attempts),
               "success_rate": mean(item["passed"] for item in attempts) if attempts else None,
               "mean_route_completion": mean(item["metrics"]["route_completion"] for item in completed) if completed else None,
               "collision_count": sum(item["metrics"]["collision_count"] for item in completed),
               "attempts": attempts}
    (output / "fixed_evaluation.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Fixed CARLA Evaluation", "", summary["evaluation_boundary"], "",
             "All planned attempts are retained, including errors and driving failures.", "",
             "| Seed | Scenario | Status | Completion | Collisions | Safety override | Pass |",
             "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for item in attempts:
        metrics, attribution = item.get("metrics", {}), item.get("control_attribution", {})
        lines.append(f"| {item['seed']} | {item['scenario']} | {item['status']} | {metrics.get('route_completion', 'n/a')} | "
                     f"{metrics.get('collision_count', 'n/a')} | {attribution.get('safety_override_ratio', 'n/a')} | {item['passed']} |")
    (output / "fixed_evaluation.md").write_text("\n".join(lines) + "\n")
    artifact_paths = [protocol_path, output / "fixed_evaluation.json", output / "fixed_evaluation.md"]
    for item in attempts:
        run_dir = output / f"seed_{item['seed']}" / str(item["scenario"])
        artifact_paths.append(run_dir / "attempt.json")
        if item["status"] == "completed":
            artifact_paths.extend(run_dir / name for name in ["carla_vision_closed_loop_report.json",
                                                             "carla_vision_closed_loop_states.csv",
                                                             "carla_vision_route.json", "carla_vision_closed_loop.mp4"])
    write_artifact_manifest(output, [build_artifact_entry(path, "evaluation", "fixed_carla", output) for path in artifact_paths])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config")
    group.add_argument("--attempt")
    args = parser.parse_args()
    if args.attempt:
        import torch
        torch.set_num_threads(4)
        run_carla_vision_closed_loop(**json.loads(Path(args.attempt).read_text()))
    else:
        payload = yaml.safe_load(Path(args.config).read_text())
        run_fixed_carla_evaluation(payload.get("carla_fixed_evaluation", payload))
