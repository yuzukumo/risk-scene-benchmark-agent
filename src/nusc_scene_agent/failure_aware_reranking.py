from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

from nusc_scene_agent.artifact_manifest import build_artifact_entry, write_artifact_manifest
from nusc_scene_agent.benchmark_schema import apply_benchmark_spec, load_benchmark_config
from nusc_scene_agent.learned_retrieval import (
    DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
    rerank_candidates_with_learned_model,
)
from nusc_scene_agent.llm_query_planner import resolve_query
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate, ValidatedCase
from nusc_scene_agent.retrieval import retrieve_candidates
from nusc_scene_agent.validation import validate_candidate


FAILURE_AWARE_RERANKING_SCHEMA = "failure_aware_reranking_eval_v1"
DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT = Path("outputs/failure_aware_reranking_eval_v4")
DEFAULT_FAILURE_UPDATE_QUERIES = Path("outputs/model_in_the_loop_failure_mining_v2/failure_update_queries.yaml")


def run_failure_aware_reranking_eval(
    query_config: Path = DEFAULT_FAILURE_UPDATE_QUERIES,
    db_path: Path = Path("artifacts/index/v1.0-trainval.sqlite"),
    output_dir: Path = DEFAULT_FAILURE_AWARE_RERANKING_OUTPUT,
    *,
    learned_checkpoint: Path = DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
    candidate_pool: int = 48,
    top_k: int = 3,
    max_queries: int = 24,
) -> Dict[str, Any]:
    specs = load_benchmark_config(Path(query_config))[: int(max_queries)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = [
            _evaluate_query(
                conn=conn,
                spec=spec,
                learned_checkpoint=Path(learned_checkpoint),
                candidate_pool=int(candidate_pool),
                top_k=int(top_k),
            )
            for spec in specs
        ]
    finally:
        conn.close()

    overview = _build_overview(rows)
    payload = {
        "schema": FAILURE_AWARE_RERANKING_SCHEMA,
        "query_config": str(query_config),
        "db_path": str(db_path),
        "learned_checkpoint": str(learned_checkpoint),
        "candidate_pool": int(candidate_pool),
        "top_k": int(top_k),
        "overview": overview,
        "query_results": rows,
    }
    json_path = output_dir / "failure_aware_reranking_eval.json"
    csv_path = output_dir / "failure_aware_reranking_eval.csv"
    md_path = output_dir / "failure_aware_reranking_eval.md"
    payload["artifact_manifest"] = {"path": str(output_dir / "artifact_manifest.json")}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(rows, csv_path)
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    write_artifact_manifest(
        output_dir=output_dir,
        artifacts=[
            build_artifact_entry(json_path, "evaluation", "failure_aware_reranking_metrics", output_dir),
            build_artifact_entry(csv_path, "evaluation", "failure_aware_reranking_table", output_dir),
            build_artifact_entry(md_path, "summary", "failure_aware_reranking_report", output_dir),
        ],
        metadata={"schema": FAILURE_AWARE_RERANKING_SCHEMA},
    )
    return payload


def _evaluate_query(
    conn: sqlite3.Connection,
    spec: Any,
    learned_checkpoint: Path,
    candidate_pool: int,
    top_k: int,
) -> Dict[str, Any]:
    query = apply_benchmark_spec(resolve_query(spec.natural_language, mode="rule"), spec)
    candidates = retrieve_candidates(
        conn,
        query=query,
        top_k=max(candidate_pool, top_k),
        candidate_pool=max(candidate_pool, top_k),
    )
    rule_cases = _validate_ranking(conn, query, candidates[:top_k])
    learned_candidates = rerank_candidates_with_learned_model(query, candidates, checkpoint_path=learned_checkpoint)
    learned_cases = _validate_ranking(conn, query, learned_candidates[:top_k])
    rule_metrics = _ranking_metrics(rule_cases)
    learned_metrics = _ranking_metrics(learned_cases)
    return {
        "query_id": spec.id,
        "description": spec.description,
        "source_tags": list(spec.tags),
        "candidate_count": len(candidates),
        "rule": rule_metrics,
        "learned": learned_metrics,
        "top1_score_delta": round(
            float(learned_metrics["top1_validation_quality_score"])
            - float(rule_metrics["top1_validation_quality_score"]),
            4,
        ),
        "best_score_delta": round(
            float(learned_metrics["best_validation_quality_score"])
            - float(rule_metrics["best_validation_quality_score"]),
            4,
        ),
        "max_score_delta": round(
            float(learned_metrics["max_validation_quality_score"])
            - float(rule_metrics["max_validation_quality_score"]),
            4,
        ),
        "pass_at_1_delta": int(bool(learned_metrics["pass_at_1"])) - int(bool(rule_metrics["pass_at_1"])),
        "pass_at_k_delta": int(bool(learned_metrics["pass_at_k"])) - int(bool(rule_metrics["pass_at_k"])),
    }


def _validate_ranking(
    conn: sqlite3.Connection,
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
) -> List[ValidatedCase]:
    return [
        validate_candidate(conn, query, candidate, include_map_geometries=False)
        for candidate in candidates
    ]


def _ranking_metrics(cases: Sequence[ValidatedCase]) -> Dict[str, Any]:
    top_case = cases[0] if cases else None
    scores = [float(case.validation_quality_score) for case in cases]
    passed = [case for case in cases if case.passed]
    best_case = max(
        cases,
        key=lambda case: (bool(case.passed), float(case.validation_quality_score)),
        default=None,
    )
    max_quality_case = max(cases, key=lambda case: float(case.validation_quality_score), default=None)
    return {
        "top1_scene": str(top_case.candidate.scene_name) if top_case else "",
        "top1_category": str(top_case.candidate.category_group) if top_case else "",
        "top1_distance_m": round(float(top_case.candidate.distance), 4) if top_case else None,
        "top1_validation_quality_score": round(float(top_case.validation_quality_score), 4) if top_case else 0.0,
        "best_validation_quality_score": round(float(best_case.validation_quality_score), 4) if best_case else 0.0,
        "max_validation_quality_score": round(float(max_quality_case.validation_quality_score), 4) if max_quality_case else 0.0,
        "mean_validation_quality_score": round(float(mean(scores)), 4) if scores else 0.0,
        # Backward-compatible aliases.
        "top1_validation_score": round(float(top_case.validation_quality_score), 4) if top_case else 0.0,
        "best_validation_score": round(float(best_case.validation_quality_score), 4) if best_case else 0.0,
        "mean_validation_score": round(float(mean(scores)), 4) if scores else 0.0,
        "validation_acceptance_at_1": bool(top_case.passed) if top_case else False,
        "validation_acceptance_at_k": bool(passed),
        "pass_at_1": bool(top_case.passed) if top_case else False,
        "pass_at_k": bool(passed),
        "passed_count": len(passed),
        "best_scene": str(best_case.candidate.scene_name) if best_case else "",
        "max_quality_scene": str(max_quality_case.candidate.scene_name) if max_quality_case else "",
    }


def _build_overview(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    query_count = len(rows)
    rule_pass_1 = sum(1 for row in rows if bool(row["rule"]["pass_at_1"]))
    learned_pass_1 = sum(1 for row in rows if bool(row["learned"]["pass_at_1"]))
    rule_pass_k = sum(1 for row in rows if bool(row["rule"]["pass_at_k"]))
    learned_pass_k = sum(1 for row in rows if bool(row["learned"]["pass_at_k"]))
    top1_deltas = [float(row.get("top1_score_delta") or 0.0) for row in rows]
    best_deltas = [float(row.get("best_score_delta") or 0.0) for row in rows]
    max_deltas = [float(row.get("max_score_delta") or 0.0) for row in rows]
    improved = sum(1 for value in top1_deltas if value > 1e-6)
    regressed = sum(1 for value in top1_deltas if value < -1e-6)
    mean_top1_delta = mean(top1_deltas) if top1_deltas else 0.0
    mean_best_delta = mean(best_deltas) if best_deltas else 0.0
    mean_max_delta = mean(max_deltas) if max_deltas else 0.0
    final_ranker_selected = (
        learned_pass_1 >= rule_pass_1
        and learned_pass_k >= rule_pass_k
        and mean_top1_delta >= 0.0
        and regressed <= improved
    )
    candidate_generator_selected = (
        learned_pass_k >= rule_pass_k
        and mean_best_delta >= 0.0
        and (learned_pass_k > rule_pass_k or mean_best_delta > 0.0)
    )
    if final_ranker_selected:
        selection_policy = "final_ranker"
    elif candidate_generator_selected:
        selection_policy = "validation_gated_candidate_generation"
    else:
        selection_policy = "rule_ranked_validation"
    return {
        "query_count": query_count,
        "rule_pass_at_1": rule_pass_1,
        "learned_pass_at_1": learned_pass_1,
        "rule_pass_at_k": rule_pass_k,
        "learned_pass_at_k": learned_pass_k,
        "mean_top1_score_delta": round(float(mean_top1_delta), 4),
        "mean_best_score_delta": round(float(mean_best_delta), 4),
        "mean_max_score_delta": round(float(mean_max_delta), 4),
        "improved_top1_score_count": improved,
        "regressed_top1_score_count": regressed,
        "final_ranker_selected": final_ranker_selected,
        "candidate_generator_selected": candidate_generator_selected,
        "selection_policy": selection_policy,
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "query_id",
        "candidate_count",
        "rule_pass_at_1",
        "learned_pass_at_1",
        "rule_pass_at_k",
        "learned_pass_at_k",
        "rule_top1_score",
        "learned_top1_score",
        "top1_score_delta",
        "rule_best_score",
        "learned_best_score",
        "best_score_delta",
        "rule_max_score",
        "learned_max_score",
        "max_score_delta",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_id": row.get("query_id", ""),
                    "candidate_count": row.get("candidate_count", 0),
                    "rule_pass_at_1": row.get("rule", {}).get("pass_at_1"),
                    "learned_pass_at_1": row.get("learned", {}).get("pass_at_1"),
                    "rule_pass_at_k": row.get("rule", {}).get("pass_at_k"),
                    "learned_pass_at_k": row.get("learned", {}).get("pass_at_k"),
                    "rule_top1_score": row.get("rule", {}).get("top1_validation_quality_score"),
                    "learned_top1_score": row.get("learned", {}).get("top1_validation_quality_score"),
                    "top1_score_delta": row.get("top1_score_delta"),
                    "rule_best_score": row.get("rule", {}).get("best_validation_quality_score"),
                    "learned_best_score": row.get("learned", {}).get("best_validation_quality_score"),
                    "best_score_delta": row.get("best_score_delta"),
                    "rule_max_score": row.get("rule", {}).get("max_validation_quality_score"),
                    "learned_max_score": row.get("learned", {}).get("max_validation_quality_score"),
                    "max_score_delta": row.get("max_score_delta"),
                }
            )


def _render_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines = [
        "# Failure-Aware Reranking Evaluation",
        "",
        f"- Queries: `{overview.get('query_count', 0)}`",
        f"- Rule validation-gated acceptance@1: `{overview.get('rule_pass_at_1', 0)}`",
        f"- Learned validation-gated acceptance@1: `{overview.get('learned_pass_at_1', 0)}`",
        f"- Rule validation-gated acceptance@K: `{overview.get('rule_pass_at_k', 0)}`",
        f"- Learned validation-gated acceptance@K: `{overview.get('learned_pass_at_k', 0)}`",
        f"- Mean top-1 score delta: `{overview.get('mean_top1_score_delta', 0.0)}`",
        "- Best quality score is acceptance-gated; maximum quality is reported separately as a diagnostic.",
        f"- Final-ranker selection: `{overview.get('final_ranker_selected', False)}`",
        f"- Candidate-generator selection: `{overview.get('candidate_generator_selected', False)}`",
        f"- Selection policy: `{overview.get('selection_policy', 'rule_ranked_validation')}`",
        "",
        "| Query | Rule@1 | Learned@1 | Rule Top-1 | Learned Top-1 | Top-1 Delta | Best Delta | Max Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("query_results", []):
        lines.append(
            "| `{0}` | `{1}` | `{2}` | `{3:.2f}` | `{4:.2f}` | `{5:.2f}` | `{6:.2f}` | `{7:.2f}` |".format(
                row.get("query_id", ""),
                row.get("rule", {}).get("pass_at_1"),
                row.get("learned", {}).get("pass_at_1"),
                float(row.get("rule", {}).get("top1_validation_quality_score") or 0.0),
                float(row.get("learned", {}).get("top1_validation_quality_score") or 0.0),
                float(row.get("top1_score_delta") or 0.0),
                float(row.get("best_score_delta") or 0.0),
                float(row.get("max_score_delta") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)
