from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, verify_artifact_manifest, write_artifact_manifest
from nusc_scene_agent.case_library import _entry_from_case, write_case_library
from nusc_scene_agent.learned_retrieval import _load_candidates_for_where, _weak_definition_query, _weak_scenario_definitions
from nusc_scene_agent.perception_benchmark import generate_perception_benchmark_from_scenario_config
from nusc_scene_agent.scenario_mining_benchmark import generate_scenario_mining_benchmark_from_case_library
from nusc_scene_agent.validation import validate_candidate
from nusc_scene_agent.world_model_benchmark import generate_world_model_benchmark_from_perception_benchmark


def expand_risk_cases(db: Path, output: Path, *, per_family: int = 30, candidate_limit: int = 180,
                      scene_split: str = "all", reuse_validated: bool = False) -> dict[str, Any]:
    if per_family < 1 or candidate_limit < 1:
        raise ValueError("Risk-case limits must be positive.")
    if scene_split not in {"all", "train", "val"}:
        raise ValueError("scene_split must be all, train or val.")
    report_path = output / "expansion_report.json"
    if reuse_validated and report_path.exists():
        previous = json.loads(report_path.read_text())
        if (previous.get("per_family_cap") != per_family or previous.get("candidate_limit_per_family") != candidate_limit
                or previous.get("scene_split", "all") != scene_split):
            raise ValueError("Cached case mining uses different limits or scene split; disable reuse or choose a new output.")
        return refresh_risk_case_exports(db, output)
    split_scenes = []
    if scene_split != "all":
        from nuscenes.utils.splits import create_splits_scenes
        split_scenes = create_splits_scenes()[scene_split]
    output.mkdir(parents=True, exist_ok=True)
    entries, attempts = [], []
    seen_actors = set()
    with sqlite3.connect(db) as conn:
        for definition in _weak_scenario_definitions():
            query = _weak_definition_query(definition)
            where, parameters = definition.positive_where, list(definition.positive_params)
            if split_scenes:
                where += " AND a.scene_name IN (" + ",".join("?" for _ in split_scenes) + ")"
                parameters += split_scenes
            candidates = _load_candidates_for_where(conn, where, parameters,
                                                    candidate_limit, balanced_scenes=True)
            family_entries = []
            scenes = set()
            for candidate in candidates:
                actor = (candidate.scene_token, candidate.instance_token)
                if actor in seen_actors or candidate.scene_token in scenes:
                    continue
                case = validate_candidate(conn, query, candidate, include_map_geometries=False)
                attempts.append({"family": definition.name, "ann_token": candidate.ann_token,
                                 "scene_token": candidate.scene_token, "passed": bool(case.passed),
                                 "behavior_matches": case.behavior_matches})
                if case.passed and all(case.behavior_matches.get(behavior, False) for behavior in query.behaviors):
                    entry = _entry_from_case(definition.name, query.original_text, [definition.name], 1, case)
                    family_entries.append(entry)
                    seen_actors.add(actor)
                    scenes.add(candidate.scene_token)
                if len(family_entries) >= per_family:
                    break
                if len(attempts) % 20 == 0:
                    print(f"{definition.name}: {len(family_entries)} validated, {len(attempts)} total attempts", flush=True)
            entries.extend(family_entries)
            write_case_library(entries, output)
            print(f"{definition.name}: {len(family_entries)} retained from {len(candidates)} candidates", flush=True)
    summary = {"schema": "risk_case_expansion_v1", "output_dir": str(output), "provenance": collect_runtime_provenance(),
               "db": str(db), "per_family_cap": per_family, "candidate_limit_per_family": candidate_limit,
               "scene_split": scene_split,
               "sampling": "scene-balanced candidates; distinct actors, at most one accepted scene per family",
               "label_semantics": "temporal and map-supported deterministic validation; no independent human labels",
               "case_count": len(entries), "scene_count": len({entry["scene_token"] for entry in entries}),
               "family_counts": dict(Counter(str(entry["event_primary_behavior"]) for entry in entries)),
               "attempts": attempts}
    return _write_benchmark_exports(db, output, summary)


def _write_benchmark_exports(db: Path, output: Path, summary: dict[str, Any]) -> dict[str, Any]:
    scenario = output / "scenario_mining.yaml"
    perception = output / "perception.json"
    forecast = output / "forecast.json"
    generate_scenario_mining_benchmark_from_case_library(output / "case_library.json", scenario,
                                                         max_cases=int(summary["case_count"]))
    generate_perception_benchmark_from_scenario_config(scenario, db, perception)
    summary["forecast"] = generate_world_model_benchmark_from_perception_benchmark(perception, db, forecast, min_future_s=3.0)
    (output / "expansion_report.json").write_text(json.dumps(summary, indent=2))
    write_artifact_manifest(output, [build_artifact_entry(path, "benchmark", "risk_case_expansion", output)
                                     for path in [output / "case_library.json", scenario, perception, forecast,
                                                  output / "expansion_report.json"]]
                            + [build_artifact_entry(db.resolve(), "input", "risk_case_expansion")])
    return summary


def refresh_risk_case_exports(db: Path, output: Path) -> dict[str, Any]:
    summary = json.loads((output / "expansion_report.json").read_text())
    if Path(summary["db"]).resolve() != db.resolve():
        raise ValueError("The cached validated cases belong to a different database.")
    if not verify_artifact_manifest(output / "artifact_manifest.json")["valid"]:
        raise ValueError("Cached risk-case artifacts failed verification.")
    summary["output_dir"] = str(output)
    summary["export_refresh"] = {"provenance": collect_runtime_provenance(),
                                 "scope": "re-export existing validated cases; mining acceptance is unchanged"}
    return _write_benchmark_exports(db, output, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="artifacts/index/v1.0-trainval.sqlite")
    parser.add_argument("--output", default="outputs/risk_case_expansion_v2")
    parser.add_argument("--per-family", type=int, default=30)
    parser.add_argument("--candidate-limit", type=int, default=180)
    parser.add_argument("--refresh-exports", action="store_true")
    parser.add_argument("--reuse-validated", action="store_true")
    parser.add_argument("--scene-split", choices=["all", "train", "val"], default="all")
    args = parser.parse_args()
    if args.refresh_exports:
        refresh_risk_case_exports(Path(args.db), Path(args.output))
    else:
        expand_risk_cases(Path(args.db), Path(args.output), per_family=args.per_family,
                          candidate_limit=args.candidate_limit, scene_split=args.scene_split,
                          reuse_validated=args.reuse_validated)
