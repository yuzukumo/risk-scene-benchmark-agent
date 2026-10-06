from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DEFAULT_FAILURE_MINING_OUTPUT = Path("outputs/model_in_the_loop_failure_mining_v2")
DEFAULT_FAILURE_SOURCES = [
    Path("outputs/trainval_bev_occupancy_proxy_study_v1"),
    Path("outputs/trainval_world_model_proxy_study_v2"),
    Path("outputs/contextvae_world_model_study_v3"),
    Path("outputs/contextvae_expanded_study_v1"),
    Path("outputs/nuplan_replay_sweep_v1/nuplan_replay_sweep_failure_taxonomy.csv"),
    Path("outputs/nuplan_closed_loop_sweep_v1"),
]


@dataclass(frozen=True)
class FailureRecord:
    source_path: str
    source_type: str
    profile_name: str
    case_key: str
    failure_tag: str
    severity: float
    primary_behavior: str = ""
    category_group: str = ""
    location: str = ""
    scenario_family: str = ""
    scenario_tag: str = ""
    difficulty_label: str = ""
    evidence: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = dict(self.evidence or {})
        return payload


def run_failure_mining(
    sources: Sequence[Path] | None = None,
    output_dir: Path = DEFAULT_FAILURE_MINING_OUTPUT,
    top_k: int = 24,
    min_count: int = 1,
) -> Dict[str, Any]:
    source_paths = [Path(path) for path in (sources or DEFAULT_FAILURE_SOURCES)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_sources(source_paths)
    records: List[FailureRecord] = []
    for path in discovered:
        if path.suffix.lower() == ".json":
            records.extend(_load_metric_json_failures(path))
        elif path.suffix.lower() == ".csv":
            records.extend(_load_taxonomy_csv_failures(path))

    records.sort(key=lambda item: (item.severity, item.failure_tag, item.profile_name), reverse=True)
    clusters = _cluster_failures(records, min_count=min_count)
    update_queries = _build_update_queries(clusters, limit=top_k)

    payload = {
        "schema": "model_in_the_loop_failure_mining_report_v1",
        "source_count": len(discovered),
        "record_count": len(records),
        "cluster_count": len(clusters),
        "sources": [str(path) for path in discovered],
        "top_failure_tags": _counter_rows(Counter(record.failure_tag for record in records)),
        "top_profiles": _counter_rows(Counter(record.profile_name for record in records if record.profile_name)),
        "clusters": clusters,
        "update_queries": update_queries,
    }

    json_path = output_dir / "failure_mining_report.json"
    md_path = output_dir / "failure_mining_report.md"
    csv_path = output_dir / "failure_records.csv"
    benchmark_path = output_dir / "failure_update_queries.yaml"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_records_csv(records, csv_path)
    benchmark_path.write_text(_render_benchmark_yaml(update_queries), encoding="utf-8")

    return {
        "schema": payload["schema"],
        "output_dir": str(output_dir),
        "source_count": len(discovered),
        "record_count": len(records),
        "failure_record_count": len(records),
        "cluster_count": len(clusters),
        "update_query_count": len(update_queries),
        "report_json": str(json_path),
        "report_md": str(md_path),
        "records_csv": str(csv_path),
        "benchmark_yaml": str(benchmark_path),
    }


def mine_model_failures(
    inputs: Sequence[Path] | None = None,
    output_dir: Path = DEFAULT_FAILURE_MINING_OUTPUT,
    max_queries: int = 24,
    min_count: int = 1,
) -> Dict[str, Any]:
    return run_failure_mining(
        sources=inputs,
        output_dir=output_dir,
        top_k=max_queries,
        min_count=min_count,
    )


def _discover_sources(paths: Sequence[Path]) -> List[Path]:
    discovered: List[Path] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        candidates: Iterable[Path]
        if path.is_dir():
            candidates = list(path.rglob("*_metrics.json")) + list(path.rglob("*failure_taxonomy.csv"))
        else:
            candidates = [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if candidate.suffix.lower() not in {".json", ".csv"}:
                continue
            seen.add(resolved)
            discovered.append(candidate)
    return sorted(discovered, key=lambda item: str(item))


def _load_metric_json_failures(path: Path) -> List[FailureRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_metrics = list(payload.get("case_metrics") or [])
    profile = _profile_name_from_metric_path(path, payload)
    source_type = _infer_source_type(path, payload)
    records: List[FailureRecord] = []
    for row in case_metrics:
        if not isinstance(row, Mapping):
            continue
        failure_tags = [str(tag) for tag in list(row.get("failure_tags") or []) if str(tag)]
        if not failure_tags:
            continue
        for tag in failure_tags:
            scenario_family = str(row.get("scenario_family") or "")
            scenario_tag = str(row.get("scenario_tag") or "")
            records.append(
                FailureRecord(
                    source_path=str(path),
                    source_type=source_type,
                    profile_name=profile,
                    case_key=str(row.get("case_id") or row.get("reference_case_key") or row.get("benchmark_group") or ""),
                    failure_tag=tag,
                    severity=round(_metric_row_severity(row, tag), 4),
                    primary_behavior=str(row.get("primary_behavior") or _behavior_from_context(scenario_family, scenario_tag)),
                    category_group=str(row.get("category_group") or _actors_from_context(scenario_tag or scenario_family)[0]),
                    location=str(row.get("location") or ""),
                    scenario_family=scenario_family,
                    scenario_tag=scenario_tag,
                    difficulty_label=str(row.get("difficulty_label") or ""),
                    evidence=_compact_evidence(row),
                )
            )
    return records


def _load_taxonomy_csv_failures(path: Path) -> List[FailureRecord]:
    records: List[FailureRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count = _safe_float(row.get("count"), default=1.0)
            difficulty = str(row.get("difficulty_label") or "")
            failure_tag = str(row.get("failure_tag") or "")
            if not failure_tag:
                continue
            records.append(
                FailureRecord(
                    source_path=str(path),
                    source_type="nuplan_replay_sweep",
                    profile_name=str(row.get("profile_name") or ""),
                    case_key="{0}:{1}:{2}:{3}".format(
                        row.get("study_name") or "",
                        row.get("profile_name") or "",
                        row.get("scenario_tag") or "",
                        failure_tag,
                    ),
                    failure_tag=failure_tag,
                    severity=round(count * _difficulty_weight(difficulty), 4),
                    primary_behavior=_behavior_from_context(
                        str(row.get("scenario_family") or ""),
                        str(row.get("scenario_tag") or ""),
                    ),
                    category_group=_actors_from_context(str(row.get("scenario_tag") or ""))[0],
                    scenario_family=str(row.get("scenario_family") or ""),
                    scenario_tag=str(row.get("scenario_tag") or ""),
                    difficulty_label=difficulty,
                    evidence={"count": count, "study_name": row.get("study_name") or ""},
                )
            )
    return records


def _cluster_failures(records: Sequence[FailureRecord], min_count: int) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        context, actor_or_scenario = _cluster_context(record)
        key = (
            record.source_type,
            record.failure_tag,
            context,
            actor_or_scenario,
        )
        bucket = buckets.setdefault(
            key,
            {
                "source_type": key[0],
                "failure_tag": key[1],
                "context": key[2],
                "actor_or_scenario": key[3],
                "count": 0,
                "total_severity": 0.0,
                "profiles": Counter(),
                "locations": Counter(),
                "case_keys": [],
            },
        )
        bucket["count"] += 1
        bucket["total_severity"] += float(record.severity)
        if record.profile_name:
            bucket["profiles"][record.profile_name] += 1
        if record.location:
            bucket["locations"][record.location] += 1
        if record.case_key and len(bucket["case_keys"]) < 8:
            bucket["case_keys"].append(record.case_key)

    clusters: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        if int(bucket["count"]) < int(min_count):
            continue
        clusters.append(
            {
                "source_type": bucket["source_type"],
                "failure_tag": bucket["failure_tag"],
                "context": bucket["context"],
                "actor_or_scenario": bucket["actor_or_scenario"],
                "count": int(bucket["count"]),
                "mean_severity": round(float(bucket["total_severity"]) / max(int(bucket["count"]), 1), 4),
                "top_profiles": _counter_rows(bucket["profiles"], limit=5),
                "top_locations": _counter_rows(bucket["locations"], limit=5),
                "example_case_keys": list(bucket["case_keys"]),
            }
        )
    clusters.sort(key=lambda item: (int(item["count"]), float(item["mean_severity"])), reverse=True)
    return clusters


def _build_update_queries(clusters: Sequence[Mapping[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    seen = set()
    for cluster in clusters:
        failure_tag = str(cluster.get("failure_tag") or "failure")
        context = str(cluster.get("context") or "")
        actor_or_scenario = str(cluster.get("actor_or_scenario") or "")
        dedupe_key = (_slug(failure_tag), _slug(context), _slug(actor_or_scenario))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        spec = _query_spec_from_failure(failure_tag, context, actor_or_scenario)
        query_index = len(queries) + 1
        query_id = "failure_mining_{0:02d}_{1}_{2}".format(
            query_index,
            _slug(failure_tag),
            _slug(actor_or_scenario or context),
        )
        queries.append(
            {
                "id": query_id,
                "description": spec["natural_language"],
                "top_k": 3,
                "candidate_pool": 16,
                "apply_query_overrides": False,
                "tags": ["failure_mining", _slug(failure_tag), _slug(context), _slug(actor_or_scenario)],
                "query": spec,
                "source_cluster": dict(cluster),
            }
        )
        if limit is not None and len(queries) >= int(limit):
            break
    return queries


def _query_spec_from_failure(failure_tag: str, context: str, actor_or_scenario: str) -> Dict[str, Any]:
    behavior = _behavior_from_context(context, actor_or_scenario)
    actors = _actors_from_context(actor_or_scenario)
    risk_terms = ["risky"]
    map_constraints: Dict[str, bool] = {}
    natural_language = _natural_language_from_failure(failure_tag, behavior, actors, actor_or_scenario)
    if behavior == "crossing":
        map_constraints["prefer_crosswalk"] = True
    if behavior == "stopped_lead":
        map_constraints["prefer_same_lane"] = True
    return {
        "natural_language": natural_language,
        "actors": actors,
        "positions": ["front"],
        "behaviors": [behavior] if behavior else [],
        "risk_terms": risk_terms,
        "map_constraints": map_constraints,
        "thresholds": {
            "near_distance_m": 18,
            "max_ttc_s": 4,
        },
    }


def _natural_language_from_failure(
    failure_tag: str,
    behavior: str,
    actors: Sequence[str],
    actor_or_scenario: str,
) -> str:
    actor = "vulnerable road user" if set(actors).intersection({"pedestrian", "bicycle", "motorcycle"}) else actors[0]
    if failure_tag in {"context_undercoverage", "low_occupancy_iou", "occupancy_error"}:
        if behavior == "crossing":
            return "{0} crossing near ego with dense surrounding traffic context".format(actor)
        if behavior == "stopped_lead":
            return "stopped {0} ahead with nearby context actors around ego lane".format(actor)
        if behavior == "oncoming":
            return "oncoming {0} close to ego with nearby context actors".format(actor)
        return "close {0} near ego with sparse BEV occupancy failure context".format(actor)
    if failure_tag in {"risk_distance_error", "ttc_error", "collision_proxy_mismatch"}:
        if "pedestrian" in actor_or_scenario:
            return "pedestrian close ahead with short time-to-collision risk"
        if "long_vehicle" in actor_or_scenario:
            return "large vehicle close ahead with replay risk-distance error"
        if "high_speed" in actor_or_scenario:
            return "high-speed vehicle close ahead with replay risk-distance error"
        if "trafficcone" in actor_or_scenario or "construction" in actor_or_scenario:
            return "static obstacle close ahead in construction context"
        return "close front actor with risk-distance and time-to-collision error"
    if failure_tag in {"miss_rate", "trajectory_error", "risk_underreach"}:
        return "{0} ahead with inaccurate future trajectory near ego".format(actor)
    if failure_tag == "closed_loop_drift":
        return "{0} ahead while ego closed-loop rollout drifts from logged behavior".format(actor)
    if failure_tag == "comfort_violation":
        return "{0} ahead with closed-loop comfort violation".format(actor)
    return "{0} ahead in a mined model failure case".format(actor)


def _behavior_from_context(context: str, actor_or_scenario: str) -> str:
    text = "{0} {1}".format(context, actor_or_scenario)
    if "cross" in text or "pedestrian" in text:
        return "crossing"
    if "stopped" in text or "long_vehicle" in text or "large_vehicle" in text:
        return "stopped_lead"
    if "oncoming" in text or "opposite" in text or "high_speed" in text:
        return "oncoming"
    if "cut_in" in text or "merge" in text:
        return "cut_in"
    return ""


def _actors_from_context(actor_or_scenario: str) -> List[str]:
    text = actor_or_scenario.lower()
    if "pedestrian" in text or "vru" in text:
        return ["pedestrian"]
    if "bicycle" in text or "cyclist" in text:
        return ["bicycle"]
    if "bus" in text:
        return ["bus"]
    if "truck" in text or "long_vehicle" in text or "large_vehicle" in text:
        return ["truck"]
    if "trafficcone" in text or "construction" in text:
        return ["vehicle"]
    return ["vehicle"]


def _cluster_context(record: FailureRecord) -> tuple[str, str]:
    if record.source_type.startswith("nuplan_replay") or record.source_type.startswith("nuplan_closed_loop"):
        return (
            record.scenario_family or record.primary_behavior or "unspecified",
            record.scenario_tag or record.category_group or "unspecified",
        )
    return (
        record.primary_behavior or record.scenario_family or "unspecified",
        record.category_group or record.scenario_tag or "unspecified",
    )


def _metric_row_severity(row: Mapping[str, Any], failure_tag: str) -> float:
    severity = 1.0
    severity += max(0.0, 1.0 - _safe_float(row.get("risk_fidelity_score"), default=1.0)) * 3.0
    severity += max(0.0, 1.0 - _safe_float(row.get("closed_loop_score"), default=1.0)) * 3.0
    severity += max(0.0, 1.0 - _safe_float(row.get("mean_occupancy_iou") or row.get("occupancy_iou"), default=1.0)) * 1.5
    severity += min(2.0, _safe_float(row.get("ade_m"), default=0.0) / 2.0)
    severity += min(2.0, _safe_float(row.get("ego_ade_m"), default=0.0) / 4.0)
    severity += 1.0 if failure_tag in {"missed_primary_actor", "collision_proxy_mismatch", "ttc_error"} else 0.0
    return severity


def _compact_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "risk_fidelity_score",
        "closed_loop_score",
        "mean_occupancy_iou",
        "occupancy_iou",
        "mean_primary_actor_recall",
        "mean_context_recall",
        "ade_m",
        "fde_m",
        "min_ade_at_5",
        "min_distance_m",
        "min_ttc_s",
        "logged_min_distance_m",
        "logged_min_ttc_s",
        "predicted_min_distance_m",
        "predicted_min_ttc_s",
        "min_distance_error_m",
        "min_ttc_error_s",
        "ego_ade_m",
        "ego_fde_m",
        "closed_loop_progress_m",
        "logged_progress_m",
        "progress_ratio",
        "distance_band",
        "ttc_band",
        "map_relation",
    ]
    return {key: row[key] for key in keys if key in row}


def _infer_source_type(path: Path, payload: Mapping[str, Any]) -> str:
    schema = str(payload.get("schema") or "")
    name = path.name
    text = "{0} {1} {2}".format(path, schema, name).lower()
    if "closed_loop" in text or schema == "nuplan_closed_loop_metrics_v1":
        return "nuplan_closed_loop"
    if "nuplan_replay" in text or "scenario_family_breakdown" in payload:
        return "nuplan_replay"
    if "bev_occupancy" in text:
        return "bev_occupancy"
    if "world_model" in text or "contextvae" in text:
        return "world_model"
    if "perception" in text:
        return "perception"
    return "model_metrics"


def _profile_name_from_metric_path(path: Path, payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    profile = str(payload.get("profile_name") or overview.get("profile_name") or path.parent.name)
    if profile.endswith("_evaluation"):
        return profile[: -len("_evaluation")]
    if profile.endswith("_closed_loop"):
        return profile[: -len("_closed_loop")]
    return profile


def _difficulty_weight(label: str) -> float:
    return {"hard": 3.0, "medium": 2.0, "easy": 1.0}.get(str(label).lower(), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _counter_rows(counter: Counter[str], limit: int = 10) -> List[Dict[str, Any]]:
    return [
        {"name": key, "count": int(value)}
        for key, value in counter.most_common(limit)
    ]


def _write_records_csv(records: Sequence[FailureRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "source_type",
            "profile_name",
            "case_key",
            "failure_tag",
            "severity",
            "primary_behavior",
            "category_group",
            "location",
            "scenario_family",
            "scenario_tag",
            "difficulty_label",
            "source_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = record.to_dict()
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def _render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Model-in-the-loop Failure Mining",
        "",
        "Schema: `{0}`".format(payload.get("schema")),
        "",
        "- Sources: `{0}`".format(payload.get("source_count")),
        "- Failure records: `{0}`".format(payload.get("record_count")),
        "- Clusters: `{0}`".format(payload.get("cluster_count")),
        "",
        "## Top Clusters",
        "",
        "| Rank | Source | Failure | Context | Actor/Scenario | Count | Mean Severity |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for rank, cluster in enumerate(list(payload.get("clusters") or [])[:20], start=1):
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6:.3f} |".format(
                rank,
                cluster.get("source_type", ""),
                cluster.get("failure_tag", ""),
                cluster.get("context", ""),
                cluster.get("actor_or_scenario", ""),
                int(cluster.get("count") or 0),
                float(cluster.get("mean_severity") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Update Queries",
            "",
            "| Query ID | Natural Language |",
            "| --- | --- |",
        ]
    )
    for query in list(payload.get("update_queries") or [])[:20]:
        query_payload = dict(query.get("query") or {})
        lines.append("| {0} | {1} |".format(query.get("id", ""), query_payload.get("natural_language", "")))
    lines.append("")
    return "\n".join(lines)


def _render_benchmark_yaml(queries: Sequence[Mapping[str, Any]]) -> str:
    lines = ["queries:"]
    for query in queries:
        query_payload = dict(query.get("query") or {})
        lines.extend(
            [
                "  - id: {0}".format(query.get("id")),
                "    description: {0}".format(_yaml_scalar(query.get("description"))),
                "    top_k: {0}".format(int(query.get("top_k") or 3)),
                "    candidate_pool: {0}".format(int(query.get("candidate_pool") or 16)),
                "    apply_query_overrides: false",
                "    tags: [{0}]".format(", ".join(_slug(tag) for tag in list(query.get("tags") or []) if tag)),
                "    query:",
                "      natural_language: {0}".format(_yaml_scalar(query_payload.get("natural_language"))),
                "      actors: [{0}]".format(", ".join(query_payload.get("actors") or [])),
                "      positions: [{0}]".format(", ".join(query_payload.get("positions") or [])),
                "      behaviors: [{0}]".format(", ".join(query_payload.get("behaviors") or [])),
                "      risk_terms: [{0}]".format(", ".join(query_payload.get("risk_terms") or [])),
            ]
        )
        map_constraints = dict(query_payload.get("map_constraints") or {})
        if map_constraints:
            lines.append("      map_constraints:")
            for key, value in sorted(map_constraints.items()):
                lines.append("        {0}: {1}".format(key, "true" if value else "false"))
        thresholds = dict(query_payload.get("thresholds") or {})
        if thresholds:
            lines.append("      thresholds:")
            for key, value in sorted(thresholds.items()):
                lines.append("        {0}: {1}".format(key, value))
        lines.append("")
    return "\n".join(lines)


def _yaml_scalar(value: Any) -> str:
    text = str(value or "")
    if not text:
        return "''"
    if any(char in text for char in [":", "#", "'", '"']):
        return json.dumps(text)
    return text


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    chars = [char if char.isalnum() or char == "_" else "_" for char in text]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unspecified"
