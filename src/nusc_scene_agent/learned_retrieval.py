from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from nusc_scene_agent.benchmark_schema import apply_benchmark_spec, load_benchmark_config
from nusc_scene_agent.artifact_manifest import build_artifact_entry, collect_runtime_provenance, write_artifact_manifest
from nusc_scene_agent.llm_query_planner import resolve_query
from nusc_scene_agent.models import ParsedQuery, RetrievalCandidate
from nusc_scene_agent.retrieval import retrieve_candidates


DEFAULT_LEARNED_RETRIEVER_OUTPUT = Path("outputs/learned_retriever_v1")
DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT = Path("outputs/learned_retriever_trainval_v3")
DEFAULT_LEARNED_RETRIEVER_CHECKPOINT = DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT / "learned_retriever.pt"

CATEGORY_GROUPS = ["vehicle", "bus", "truck", "pedestrian", "bicycle", "motorcycle"]
POSITIONS = ["front", "rear", "left", "right"]
BEHAVIORS = ["crossing", "cut_in", "oncoming", "stopped_lead"]
RISK_TERMS = ["risky", "urgent"]


@dataclass(frozen=True)
class LearnedRetrieverConfig:
    text_hash_dim: int = 256
    hidden_dim: int = 128
    embedding_dim: int = 64
    temperature: float = 0.07
    negatives_per_query: int = 12
    candidate_pool: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.25
    seed: int = 7
    device: str = ""


@dataclass(frozen=True)
class LearnedScore:
    rank: int
    candidate_key: str
    ann_token: str
    scene_name: str
    sample_idx: int
    category_group: str
    distance_m: float
    ttc_s: float
    base_retrieval_score: float
    learned_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _TrainingGroup:
    spec_id: str
    case_key: str
    query: ParsedQuery
    positive: RetrievalCandidate
    negatives: List[RetrievalCandidate]
    split: str = ""


@dataclass(frozen=True)
class _WeakScenarioDefinition:
    name: str
    natural_language: str
    category_groups: List[str]
    positions: List[str]
    behaviors: List[str]
    risk_terms: List[str]
    near_distance_m: float
    max_ttc_s: float
    positive_where: str
    positive_params: List[Any]
    negative_where: str
    negative_params: List[Any]


def train_learned_scene_retriever(
    benchmark_path: Path,
    db_path: Path,
    output_dir: Path = DEFAULT_LEARNED_RETRIEVER_OUTPUT,
    config: LearnedRetrieverConfig | None = None,
) -> Dict[str, Any]:
    config = config or LearnedRetrieverConfig()
    groups = _build_training_groups(
        benchmark_path=Path(benchmark_path),
        db_path=Path(db_path),
        config=config,
    )
    return _train_from_groups(
        groups=groups,
        output_dir=Path(output_dir),
        config=config,
        report_context={
            "training_source": "reference_benchmark",
            "benchmark_path": str(benchmark_path),
            "db_path": str(db_path),
        },
        split_key="scene_token",
    )


def train_weakly_supervised_scene_retriever(
    db_path: Path,
    output_dir: Path = DEFAULT_LARGE_LEARNED_RETRIEVER_OUTPUT,
    config: LearnedRetrieverConfig | None = None,
    max_groups_per_family: int = 1000,
) -> Dict[str, Any]:
    config = config or LearnedRetrieverConfig(epochs=20)
    groups, family_summary = _build_weak_training_groups(
        db_path=Path(db_path),
        config=config,
        max_groups_per_family=int(max_groups_per_family),
    )
    return _train_from_groups(
        groups=groups,
        output_dir=Path(output_dir),
        config=config,
        report_context={
            "training_source": "weak_supervision_from_trainval_index",
            "db_path": str(db_path),
            "max_groups_per_family": int(max_groups_per_family),
            "weak_family_summary": family_summary,
        },
        split_key="scene_token",
    )


def _train_from_groups(
    groups: Sequence[_TrainingGroup],
    output_dir: Path,
    config: LearnedRetrieverConfig,
    report_context: Mapping[str, Any],
    split_key: str,
) -> Dict[str, Any]:
    torch = _load_torch()
    _set_seeds(config.seed, torch)
    device = _resolve_device(config.device, torch)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(groups) < 2:
        raise ValueError("At least two training groups are required.")

    train_groups, validation_groups = _split_groups(groups, config.validation_fraction, config.seed, split_key=split_key)
    integrity = _training_split_integrity(train_groups, validation_groups)
    if split_key == "scene_token" and not integrity["scene_disjoint"]:
        raise ValueError("Training and validation candidates must have disjoint scenes.")
    if not train_groups or (config.validation_fraction > 0.0 and not validation_groups):
        raise ValueError("Scene splitting requires both training and validation groups with same-split negatives.")
    feature_dim = len(_pair_feature_vector(groups[0].query, groups[0].positive, config))
    model = _build_pairwise_scorer(torch, feature_dim, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    losses: List[float] = []
    for _ in range(int(config.epochs)):
        random.shuffle(train_groups)
        epoch_losses = []
        model.train()
        for group in train_groups:
            candidates = [group.positive] + group.negatives[: int(config.negatives_per_query)]
            if len(candidates) < 2:
                continue
            pair_tensor = torch.tensor(
                [_pair_feature_vector(group.query, candidate, config) for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
            logits = model(pair_tensor).reshape(1, -1) / max(float(config.temperature), 1e-4)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch_losses:
            losses.append(float(np.mean(epoch_losses)))

    train_metrics = _evaluate_groups(model, train_groups, config, torch, device)
    validation_metrics = _evaluate_groups(model, validation_groups, config, torch, device)

    checkpoint_path = output_dir / "learned_retriever.pt"
    torch.save(
        {
            "schema": "learned_scene_retriever_checkpoint_v1",
            "model_type": "query_scene_pairwise_mlp_v1",
            "config": asdict(config),
            "feature_dim": feature_dim,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    report = {
        "schema": "learned_scene_retriever_training_report_v1",
        **dict(report_context),
        "checkpoint_path": str(checkpoint_path),
        "group_count": len(groups),
        "train_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "validation_split_key": split_key,
        "split_integrity": integrity,
        "provenance": collect_runtime_provenance(),
        "label_semantics": "deterministic weak-rule consistency, not independently annotated semantic recall",
        "config": asdict(config),
        "model_type": "query_scene_pairwise_mlp_v1",
        "feature_dim": feature_dim,
        "final_train_loss": round(float(losses[-1]), 6) if losses else None,
        "loss_curve": [round(float(value), 6) for value in losses],
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "training_groups": [
            {
                "spec_id": group.spec_id,
                "case_key": group.case_key,
                "positive_scene": group.positive.scene_name,
                "positive_scene_token": group.positive.scene_token,
                "positive_ann_token": group.positive.ann_token,
                "negative_scene_tokens": [candidate.scene_token for candidate in group.negatives],
                "negative_ann_tokens": [candidate.ann_token for candidate in group.negatives],
                "split": "train" if group.spec_id in {item.spec_id for item in train_groups} else "val",
                "negative_count": len(group.negatives),
            }
            for group in groups
        ],
    }
    report_json = output_dir / "training_report.json"
    report_md = output_dir / "training_report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_md.write_text(_render_training_markdown(report), encoding="utf-8")
    artifacts = [build_artifact_entry(path, "result", "learned_retrieval", output_dir)
                 for path in [checkpoint_path, report_json, report_md]]
    artifacts.extend(build_artifact_entry(Path(str(report_context[key])).resolve(), "input", "learned_retrieval")
                     for key in ["db_path", "benchmark_path"] if report_context.get(key))
    write_artifact_manifest(output_dir, artifacts)
    return report


def score_candidates_with_learned_model(
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
    checkpoint_path: Path = DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
) -> List[LearnedScore]:
    if not candidates:
        return []
    torch = _load_torch()
    _, model, config, device = _load_checkpoint(Path(checkpoint_path), torch)
    model.eval()
    with torch.no_grad():
        pair_tensor = torch.tensor(
            [_pair_feature_vector(query, candidate, config) for candidate in candidates],
            dtype=torch.float32,
            device=device,
        )
        logits = model(pair_tensor).reshape(-1)
        values = logits.detach().cpu().numpy().tolist()

    scored: List[LearnedScore] = []
    for candidate, score in zip(candidates, values):
        scored.append(
            LearnedScore(
                rank=0,
                candidate_key="{0}:{1}".format(candidate.scene_token, candidate.ann_token),
                ann_token=candidate.ann_token,
                scene_name=candidate.scene_name,
                sample_idx=int(candidate.sample_idx),
                category_group=candidate.category_group,
                distance_m=round(float(candidate.distance), 4),
                ttc_s=round(float(candidate.ttc), 4) if math.isfinite(float(candidate.ttc)) else float("inf"),
                base_retrieval_score=round(float(candidate.retrieval_score), 6),
                learned_score=round(float(score), 6),
            )
        )

    scored.sort(key=lambda item: item.learned_score, reverse=True)
    return [
        LearnedScore(
            rank=rank,
            candidate_key=item.candidate_key,
            ann_token=item.ann_token,
            scene_name=item.scene_name,
            sample_idx=item.sample_idx,
            category_group=item.category_group,
            distance_m=item.distance_m,
            ttc_s=item.ttc_s,
            base_retrieval_score=item.base_retrieval_score,
            learned_score=item.learned_score,
        )
        for rank, item in enumerate(scored, start=1)
    ]


def rerank_candidates_with_learned_model(
    query: ParsedQuery,
    candidates: Sequence[RetrievalCandidate],
    checkpoint_path: Path = DEFAULT_LEARNED_RETRIEVER_CHECKPOINT,
) -> List[RetrievalCandidate]:
    from dataclasses import replace

    score_by_token = {
        score.ann_token: score
        for score in score_candidates_with_learned_model(query, candidates, checkpoint_path=checkpoint_path)
    }
    candidate_by_token = {candidate.ann_token: candidate for candidate in candidates}
    return [
        replace(candidate_by_token[score.ann_token], retrieval_score=float(score.learned_score))
        for score in sorted(score_by_token.values(), key=lambda item: item.rank)
    ]


def run_learned_retrieval_report(
    db_path: Path,
    query_text: str,
    checkpoint_path: Path,
    output_dir: Path,
    top_k: int = 20,
    candidate_pool: int = 64,
) -> Dict[str, Any]:
    query = resolve_query(query_text, mode="rule")
    conn = sqlite3.connect(str(db_path))
    try:
        candidates = retrieve_candidates(conn, query=query, top_k=top_k, candidate_pool=max(candidate_pool, top_k))
    finally:
        conn.close()
    scores = score_candidates_with_learned_model(query, candidates, checkpoint_path=checkpoint_path)[:top_k]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "learned_scene_retrieval_report_v1",
        "query": query.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "candidate_count": len(scores),
        "scores": [score.to_dict() for score in scores],
    }
    json_path = output_dir / "learned_retrieval_report.json"
    csv_path = output_dir / "learned_retrieval_scores.csv"
    md_path = output_dir / "learned_retrieval_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_score_csv(scores, csv_path)
    md_path.write_text(_render_retrieval_markdown(payload), encoding="utf-8")
    return {
        "schema": payload["schema"],
        "output_dir": str(output_dir),
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "candidate_count": len(scores),
    }


def _build_training_groups(
    benchmark_path: Path,
    db_path: Path,
    config: LearnedRetrieverConfig,
) -> List[_TrainingGroup]:
    specs = [spec for spec in load_benchmark_config(benchmark_path) if spec.expect_match is not False]
    conn = sqlite3.connect(str(db_path))
    try:
        positives = {spec.id: _load_positive_candidate(conn, spec.reference_case_keys[0])
                     for spec in specs if spec.reference_case_keys}
        rng = random.Random(config.seed)
        positive_scenes = sorted({candidate.scene_token for candidate in positives.values() if candidate is not None})
        rng.shuffle(positive_scenes)
        validation_count = (min(len(positive_scenes) - 1, max(1, round(len(positive_scenes) * config.validation_fraction)))
                            if len(positive_scenes) > 1 and config.validation_fraction > 0.0 else 0)
        validation_scenes = set(positive_scenes[:validation_count])
        other_scenes = sorted({str(row[0]) for row in conn.execute("SELECT DISTINCT scene_token FROM agents")} - set(positive_scenes))
        rng.shuffle(other_scenes)
        validation_scenes.update(other_scenes[:round(len(other_scenes) * config.validation_fraction)])
        groups: List[_TrainingGroup] = []
        for spec in specs:
            if not spec.reference_case_keys:
                continue
            parsed = resolve_query(spec.natural_language, mode="rule")
            query = apply_benchmark_spec(parsed, spec)
            positive = positives.get(spec.id)
            if positive is None:
                continue
            retrieved = retrieve_candidates(
                conn,
                query=query,
                top_k=int(config.candidate_pool),
                candidate_pool=int(config.candidate_pool),
            )
            negatives = [
                candidate
                for candidate in retrieved
                if candidate.ann_token != positive.ann_token
                and "{0}:{1}".format(candidate.sample_token, candidate.instance_token) != spec.reference_case_keys[0]
                and (candidate.scene_token in validation_scenes) == (positive.scene_token in validation_scenes)
            ][: int(config.negatives_per_query)]
            if not negatives:
                continue
            groups.append(
                _TrainingGroup(
                    spec_id=spec.id,
                    case_key=spec.reference_case_keys[0],
                    query=query,
                    positive=positive,
                    negatives=negatives,
                    split="val" if positive.scene_token in validation_scenes else "train",
                )
            )
    finally:
        conn.close()
    return groups


def _build_weak_training_groups(
    db_path: Path,
    config: LearnedRetrieverConfig,
    max_groups_per_family: int,
) -> tuple[List[_TrainingGroup], List[Dict[str, Any]]]:
    rng = random.Random(int(config.seed))
    definitions = _weak_scenario_definitions()
    groups: List[_TrainingGroup] = []
    family_summary: List[Dict[str, Any]] = []
    conn = sqlite3.connect(str(db_path))
    try:
        scene_tokens = sorted(str(row[0]) for row in conn.execute("SELECT DISTINCT scene_token FROM agents"))
        rng.shuffle(scene_tokens)
        validation_count = (
            min(len(scene_tokens) - 1, max(1, round(len(scene_tokens) * config.validation_fraction)))
            if config.validation_fraction > 0.0 and len(scene_tokens) > 1 else 0
        )
        validation_scenes = set(scene_tokens[:validation_count])
        for definition in definitions:
            positives = _load_candidates_for_where(
                conn,
                definition.positive_where,
                definition.positive_params,
                limit=int(max_groups_per_family),
                balanced_scenes=True,
            )
            negative_pool = _load_candidates_for_where(
                conn,
                definition.negative_where,
                definition.negative_params,
                limit=max(int(max_groups_per_family) * max(int(config.negatives_per_query), 1) * 2, 256),
                balanced_scenes=True,
            )
            negative_partitions = {
                split: [candidate for candidate in negative_pool
                        if (candidate.scene_token in validation_scenes) == (split == "val")]
                for split in ("train", "val")
            }
            query = _weak_definition_query(definition)
            family_groups = 0
            for index, positive in enumerate(positives):
                split = "val" if positive.scene_token in validation_scenes else "train"
                positive_key = "{0}:{1}".format(positive.sample_token, positive.instance_token)
                available_negatives = [
                    candidate
                    for candidate in negative_partitions[split]
                    if candidate.ann_token != positive.ann_token
                    and "{0}:{1}".format(candidate.sample_token, candidate.instance_token) != positive_key
                ]
                if len(available_negatives) < 1:
                    continue
                negative_count = min(int(config.negatives_per_query), len(available_negatives))
                negatives = rng.sample(available_negatives, negative_count)
                groups.append(
                    _TrainingGroup(
                        spec_id="weak_{0}_{1:05d}".format(definition.name, index),
                        case_key=positive_key,
                        query=query,
                        positive=positive,
                        negatives=negatives,
                        split=split,
                    )
                )
                family_groups += 1
            family_summary.append(
                {
                    "family": definition.name,
                    "positive_pool_count": len(positives),
                    "negative_pool_count": len(negative_pool),
                    "training_group_count": family_groups,
                }
            )
    finally:
        conn.close()

    rng.shuffle(groups)
    return groups, family_summary


def _weak_scenario_definitions() -> List[_WeakScenarioDefinition]:
    vehicle_like = ["vehicle", "bus", "truck"]
    vru_like = ["pedestrian", "bicycle", "motorcycle"]
    return [
        _WeakScenarioDefinition(
            name="vru_crossing_front",
            natural_language="vulnerable road user crossing close in front of ego",
            category_groups=vru_like,
            positions=["front"],
            behaviors=["crossing"],
            risk_terms=["risky", "urgent"],
            near_distance_m=25.0,
            max_ttc_s=4.0,
            positive_where=(
                "a.category_group IN ({vru}) AND a.x_ego BETWEEN -2.0 AND 25.0 "
                "AND ABS(a.y_ego) >= 2.0 AND a.distance <= 25.0 AND a.num_lidar_pts > 0"
            ).format(vru=_placeholders(len(vru_like))),
            positive_params=list(vru_like),
            negative_where=(
                "a.category_group IN ({vru}) AND a.x_ego BETWEEN 0.0 AND 35.0 "
                "AND ABS(a.y_ego) <= 1.0 AND a.distance <= 35.0"
            ).format(vru=_placeholders(len(vru_like))),
            negative_params=list(vru_like),
        ),
        _WeakScenarioDefinition(
            name="stopped_lead_vehicle",
            natural_language="stopped lead vehicle blocking ego lane ahead",
            category_groups=vehicle_like,
            positions=["front"],
            behaviors=["stopped_lead"],
            risk_terms=["risky"],
            near_distance_m=30.0,
            max_ttc_s=5.0,
            positive_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN 0.0 AND 30.0 "
                "AND ABS(a.y_ego) <= 2.5 AND a.speed <= 0.8 AND a.distance <= 30.0 "
                "AND a.num_lidar_pts > 0"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            positive_params=list(vehicle_like),
            negative_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN 0.0 AND 35.0 "
                "AND ABS(a.y_ego) <= 2.5 AND a.speed >= 2.0 AND a.distance <= 35.0"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            negative_params=list(vehicle_like),
        ),
        _WeakScenarioDefinition(
            name="lateral_cut_in",
            natural_language="vehicle cutting in laterally near the ego lane",
            category_groups=vehicle_like,
            positions=["front", "left", "right"],
            behaviors=["cut_in"],
            risk_terms=["risky"],
            near_distance_m=35.0,
            max_ttc_s=5.0,
            positive_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN -5.0 AND 35.0 "
                "AND ABS(a.y_ego) >= 2.0 AND a.distance <= 35.0 AND a.speed >= 0.5 "
                "AND a.num_lidar_pts > 0"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            positive_params=list(vehicle_like),
            negative_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN 0.0 AND 40.0 "
                "AND ABS(a.y_ego) <= 1.0 AND a.distance <= 40.0 AND a.speed >= 0.5"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            negative_params=list(vehicle_like),
        ),
        _WeakScenarioDefinition(
            name="oncoming_vehicle",
            natural_language="oncoming vehicle approaching ego in front",
            category_groups=vehicle_like,
            positions=["front"],
            behaviors=["oncoming"],
            risk_terms=["risky", "urgent"],
            near_distance_m=45.0,
            max_ttc_s=6.0,
            positive_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN 0.0 AND 45.0 "
                "AND ABS(ABS(a.heading_delta) - 3.141592653589793) <= 1.2 "
                "AND a.rel_vx <= -1.0 AND a.distance <= 45.0 AND a.num_lidar_pts > 0"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            positive_params=list(vehicle_like),
            negative_where=(
                "a.category_group IN ({vehicle}) AND a.x_ego BETWEEN 0.0 AND 45.0 "
                "AND ABS(ABS(a.heading_delta) - 3.141592653589793) >= 1.6 "
                "AND a.distance <= 45.0"
            ).format(vehicle=_placeholders(len(vehicle_like))),
            negative_params=list(vehicle_like),
        ),
    ]


def _weak_definition_query(definition: _WeakScenarioDefinition) -> ParsedQuery:
    return ParsedQuery(
        original_text=definition.natural_language,
        normalized_text=definition.natural_language,
        category_groups=list(definition.category_groups),
        positions=list(definition.positions),
        behaviors=list(definition.behaviors),
        near_distance_m=float(definition.near_distance_m),
        max_ttc_s=float(definition.max_ttc_s),
        risk_terms=list(definition.risk_terms),
        specific_keywords=["weak_supervision:{0}".format(definition.name)],
    )


def _load_candidates_for_where(
    conn: sqlite3.Connection,
    where_clause: str,
    params: Sequence[Any],
    limit: int,
    balanced_scenes: bool = False,
) -> List[RetrievalCandidate]:
    cursor = conn.execute(
        """
        SELECT
            a.ann_token,
            a.sample_token,
            a.scene_token,
            a.scene_name,
            a.sample_idx,
            a.instance_token,
            a.category_name,
            a.category_group,
            a.distance,
            a.ttc,
            a.x_ego,
            a.y_ego,
            a.speed,
            a.rel_vx,
            a.rel_vy,
            a.heading_delta,
            a.num_lidar_pts,
            a.num_radar_pts,
            s.location,
            s.scene_description,
            1.0 AS retrieval_score
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE {0}
        ORDER BY {1}
        LIMIT ?
        """.format(where_clause, (
            "ROW_NUMBER() OVER (PARTITION BY a.scene_token ORDER BY a.sample_idx, a.instance_token), a.scene_token"
            if balanced_scenes else "a.scene_token, a.sample_idx, a.instance_token"
        )),
        list(params) + [int(limit)],
    )
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description or []]
    return [RetrievalCandidate.from_record(dict(zip(columns, row))) for row in rows]


def _placeholders(count: int) -> str:
    return ", ".join(["?"] * int(count))


def _load_positive_candidate(conn: sqlite3.Connection, case_key: str) -> RetrievalCandidate | None:
    parts = str(case_key).split(":")
    if len(parts) != 2:
        return None
    sample_token, instance_token = parts
    cursor = conn.execute(
        """
        SELECT
            a.ann_token,
            a.sample_token,
            a.scene_token,
            a.scene_name,
            a.sample_idx,
            a.instance_token,
            a.category_name,
            a.category_group,
            a.distance,
            a.ttc,
            a.x_ego,
            a.y_ego,
            a.speed,
            a.rel_vx,
            a.rel_vy,
            a.heading_delta,
            a.num_lidar_pts,
            a.num_radar_pts,
            s.location,
            s.scene_description,
            1.0 AS retrieval_score
        FROM agents a
        JOIN samples s ON s.sample_token = a.sample_token
        WHERE a.sample_token = ? AND a.instance_token = ?
        LIMIT 1
        """,
        (sample_token, instance_token),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description or []]
    return RetrievalCandidate.from_record(dict(zip(columns, row)))


def _split_groups(
    groups: Sequence[_TrainingGroup],
    validation_fraction: float,
    seed: int,
    split_key: str,
) -> tuple[List[_TrainingGroup], List[_TrainingGroup]]:
    if any(group.split for group in groups):
        if any(group.split not in {"train", "val"} for group in groups):
            raise ValueError("Every group must have a valid predefined split.")
        return [group for group in groups if group.split == "train"], [group for group in groups if group.split == "val"]
    rng = random.Random(seed)
    split_values = sorted({_group_split_value(group, split_key) for group in groups})
    rng.shuffle(split_values)
    validation_count = (
        max(1, int(round(len(split_values) * float(validation_fraction))))
        if len(split_values) > 1 and float(validation_fraction) > 0.0
        else 0
    )
    validation_keys = set(split_values[:validation_count])
    train = [group for group in groups if _group_split_value(group, split_key) not in validation_keys]
    validation = [group for group in groups if _group_split_value(group, split_key) in validation_keys]
    if not train:
        train, validation = list(groups), []
    return train, validation


def _training_split_integrity(train: Sequence[_TrainingGroup], validation: Sequence[_TrainingGroup]) -> Dict[str, Any]:
    def values(groups: Sequence[_TrainingGroup], attribute: str) -> set[str]:
        return {str(getattr(candidate, attribute)) for group in groups
                for candidate in [group.positive, *group.negatives]}

    train_scenes, val_scenes = values(train, "scene_token"), values(validation, "scene_token")
    shared_annotations = values(train, "ann_token") & values(validation, "ann_token")
    return {
        "scene_disjoint": not bool(train_scenes & val_scenes),
        "annotation_disjoint": not bool(shared_annotations),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(val_scenes),
        "shared_scene_tokens": sorted(train_scenes & val_scenes),
        "shared_annotation_count": len(shared_annotations),
    }


def _group_split_value(group: _TrainingGroup, split_key: str) -> str:
    if split_key == "scene_token":
        return str(group.positive.scene_token)
    return str(group.case_key)


def _evaluate_groups(model: Any, groups: Sequence[_TrainingGroup], config: LearnedRetrieverConfig, torch: Any, device: str) -> Dict[str, Any]:
    if not groups:
        return {"group_count": 0, "recall_at_1": None, "mean_rank": None}
    ranks: List[int] = []
    model.eval()
    with torch.no_grad():
        for group in groups:
            candidates = [group.positive] + group.negatives[: int(config.negatives_per_query)]
            pair_tensor = torch.tensor(
                [_pair_feature_vector(group.query, candidate, config) for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
            logits = model(pair_tensor).reshape(-1)
            order = torch.argsort(logits, descending=True).detach().cpu().tolist()
            ranks.append(int(order.index(0)) + 1)
    return {
        "group_count": len(groups),
        "recall_at_1": round(sum(1 for rank in ranks if rank == 1) / max(len(ranks), 1), 4),
        "mean_rank": round(float(np.mean(ranks)), 4),
        "ranks": ranks,
    }


def _query_feature_vector(query: ParsedQuery, config: LearnedRetrieverConfig) -> List[float]:
    tokens = _tokens(query.normalized_text or query.original_text)
    tokens.extend("actor_" + item for item in query.category_groups)
    tokens.extend("position_" + item for item in query.positions)
    tokens.extend("behavior_" + item for item in query.behaviors)
    tokens.extend("risk_" + item for item in query.risk_terms)
    hashed = _hashed_bow(tokens, int(config.text_hash_dim))
    structured = (
        _multi_hot(query.category_groups, CATEGORY_GROUPS)
        + _multi_hot(query.positions, POSITIONS)
        + _multi_hot(query.behaviors, BEHAVIORS)
        + _multi_hot(query.risk_terms, RISK_TERMS)
        + [min(1.0, float(query.near_distance_m) / 50.0), min(1.0, float(query.max_ttc_s) / 10.0)]
    )
    return hashed + structured


def _candidate_feature_vector(candidate: RetrievalCandidate) -> List[float]:
    positions, behaviors, risk_terms = _candidate_structural_tags(candidate)
    continuous = [
        min(1.0, max(0.0, float(candidate.distance)) / 50.0),
        min(1.0, max(0.0, float(candidate.ttc)) / 10.0) if math.isfinite(float(candidate.ttc)) else 1.0,
        max(-1.0, min(1.0, float(candidate.x_ego) / 50.0)),
        max(-1.0, min(1.0, float(candidate.y_ego) / 20.0)),
        min(1.0, abs(float(candidate.speed)) / 20.0),
        max(-1.0, min(1.0, float(candidate.rel_vx) / 20.0)),
        max(-1.0, min(1.0, float(candidate.rel_vy) / 20.0)),
        math.cos(float(candidate.heading_delta)),
        math.sin(float(candidate.heading_delta)),
        min(1.0, max(0, int(candidate.num_lidar_pts)) / 20.0),
        min(1.0, max(0, int(candidate.num_radar_pts)) / 10.0),
    ]
    return (
        _multi_hot([candidate.category_group], CATEGORY_GROUPS)
        + _multi_hot(positions, POSITIONS)
        + _multi_hot(behaviors, BEHAVIORS)
        + _multi_hot(risk_terms, RISK_TERMS)
        + continuous
    )


def _candidate_structural_tags(candidate: RetrievalCandidate) -> tuple[List[str], List[str], List[str]]:
    positions = []
    if candidate.x_ego >= 0.0:
        positions.append("front")
    else:
        positions.append("rear")
    if candidate.y_ego >= 1.5:
        positions.append("left")
    if candidate.y_ego <= -1.5:
        positions.append("right")
    behaviors = []
    if abs(candidate.y_ego) >= 2.0 and 0.0 <= candidate.x_ego <= 25.0:
        behaviors.append("crossing")
    if abs(candidate.y_ego) >= 2.0 and candidate.category_group in {"vehicle", "bus", "truck"}:
        behaviors.append("cut_in")
    if candidate.x_ego >= 0.0 and abs(abs(candidate.heading_delta) - math.pi) <= 1.2:
        behaviors.append("oncoming")
    if candidate.x_ego >= 0.0 and abs(candidate.y_ego) <= 3.0 and candidate.speed <= 0.8:
        behaviors.append("stopped_lead")
    risk_terms = []
    if candidate.distance <= 15.0:
        risk_terms.append("risky")
    if math.isfinite(float(candidate.ttc)) and candidate.ttc <= 3.0:
        risk_terms.append("urgent")
    return positions, behaviors, risk_terms


def _pair_feature_vector(query: ParsedQuery, candidate: RetrievalCandidate, config: LearnedRetrieverConfig) -> List[float]:
    positions, behaviors, risk_terms = _candidate_structural_tags(candidate)
    actor_match = _overlap_score(query.category_groups, [candidate.category_group])
    position_match = _overlap_score(query.positions, positions)
    behavior_match = _overlap_score(query.behaviors, behaviors)
    risk_match = _overlap_score(query.risk_terms, risk_terms)
    distance_score = max(0.0, min(1.0, 1.0 - float(candidate.distance) / max(float(query.near_distance_m), 1.0)))
    if math.isfinite(float(candidate.ttc)):
        ttc_score = max(0.0, min(1.0, 1.0 - float(candidate.ttc) / max(float(query.max_ttc_s), 0.5)))
    else:
        ttc_score = 0.0
    same_lane_front = 1.0 if candidate.x_ego >= 0.0 and abs(float(candidate.y_ego)) <= 3.0 else 0.0
    lateral_front = 1.0 if candidate.x_ego >= -5.0 and abs(float(candidate.y_ego)) >= 2.0 else 0.0
    stopped_front = 1.0 if candidate.x_ego >= 0.0 and abs(float(candidate.y_ego)) <= 3.0 and candidate.speed <= 0.8 else 0.0
    oncoming_heading = max(0.0, min(1.0, 1.0 - abs(abs(float(candidate.heading_delta)) - math.pi) / 1.2))
    closing_score = max(0.0, min(1.0, -float(candidate.rel_vx) / 8.0))
    interaction = [
        actor_match,
        position_match,
        behavior_match,
        risk_match,
        distance_score,
        ttc_score,
        same_lane_front,
        lateral_front,
        stopped_front,
        oncoming_heading,
        closing_score,
        actor_match * distance_score,
        actor_match * behavior_match,
        position_match * distance_score,
        behavior_match * ttc_score,
    ]
    return _query_feature_vector(query, config) + _candidate_feature_vector(candidate) + interaction


def _overlap_score(expected: Sequence[str], observed: Sequence[str]) -> float:
    expected_set = {str(item) for item in expected if item}
    observed_set = {str(item) for item in observed if item}
    if not expected_set:
        return 1.0
    return len(expected_set.intersection(observed_set)) / max(len(expected_set), 1)


def _build_pairwise_scorer(torch: Any, feature_dim: int, config: LearnedRetrieverConfig) -> Any:
    class PairwiseScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, int(config.hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(int(config.hidden_dim)),
                torch.nn.Linear(int(config.hidden_dim), int(config.embedding_dim)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(config.embedding_dim), 1),
            )

        def forward(self, pair_features: Any) -> Any:
            return self.scorer(pair_features).squeeze(-1)

    return PairwiseScorer()


def _load_checkpoint(checkpoint_path: Path, torch: Any) -> tuple[Mapping[str, Any], Any, LearnedRetrieverConfig, str]:
    if not checkpoint_path.exists():
        raise FileNotFoundError("Missing learned retriever checkpoint: {0}".format(checkpoint_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = LearnedRetrieverConfig(**dict(checkpoint.get("config") or {}))
    device = _resolve_device(config.device, torch)
    if checkpoint.get("model_type") != "query_scene_pairwise_mlp_v1":
        raise ValueError("Unsupported learned retriever checkpoint type: {0}".format(checkpoint.get("model_type")))
    model = _build_pairwise_scorer(torch, int(checkpoint["feature_dim"]), config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model, config, device


def _tokens(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9_]+", str(text).lower()) if token]


def _hashed_bow(tokens: Sequence[str], dim: int) -> List[float]:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dim
        vector[index] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector.tolist()


def _multi_hot(values: Sequence[str], vocabulary: Sequence[str]) -> List[float]:
    value_set = {str(value) for value in values}
    return [1.0 if item in value_set else 0.0 for item in vocabulary]


def _set_seeds(seed: int, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: str, torch: Any) -> str:
    requested = str(device or "").strip()
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError('Learned retrieval requires PyTorch. Install it with `pip install -e ".[learned]"`.') from exc
    return torch


def _write_score_csv(scores: Sequence[LearnedScore], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "candidate_key",
                "ann_token",
                "scene_name",
                "sample_idx",
                "category_group",
                "distance_m",
                "ttc_s",
                "base_retrieval_score",
                "learned_score",
            ],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(score.to_dict())


def _render_training_markdown(report: Mapping[str, Any]) -> str:
    validation = dict(report.get("validation_metrics") or {})
    train = dict(report.get("train_metrics") or {})
    lines = [
        "# Learned Scene Retriever Training",
        "",
        "Schema: `{0}`".format(report.get("schema")),
        "",
        "- Model: `{0}`".format(report.get("model_type")),
        "- Training source: `{0}`".format(report.get("training_source")),
        "- Validation split key: `{0}`".format(report.get("validation_split_key")),
        "- Feature dimension: `{0}`".format(report.get("feature_dim")),
        "- Training groups: `{0}`".format(report.get("train_group_count")),
        "- Validation groups: `{0}`".format(report.get("validation_group_count")),
        "- Final train loss: `{0}`".format(report.get("final_train_loss")),
        "- Train weak-anchor consistency@1: `{0}`".format(train.get("recall_at_1")),
        "- Validation weak-anchor consistency@1: `{0}`".format(validation.get("recall_at_1")),
        "- Validation mean rank: `{0}`".format(validation.get("mean_rank")),
        "- Checkpoint: `{0}`".format(report.get("checkpoint_path")),
        "",
    ]
    family_summary = list(report.get("weak_family_summary") or [])
    if family_summary:
        lines.extend(
            [
                "## Weak Supervision Families",
                "",
                "| Family | Positives | Negatives | Groups |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in family_summary:
            lines.append(
                "| {family} | {positives} | {negatives} | {groups} |".format(
                    family=str(item.get("family") or ""),
                    positives=int(item.get("positive_pool_count") or 0),
                    negatives=int(item.get("negative_pool_count") or 0),
                    groups=int(item.get("training_group_count") or 0),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _render_retrieval_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Learned Scene Retrieval Report",
        "",
        "Schema: `{0}`".format(payload.get("schema")),
        "",
        "Candidate count: `{0}`".format(payload.get("candidate_count")),
        "",
        "| Rank | Scene | Category | Distance | TTC | Learned Score | Base Score |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for score in list(payload.get("scores") or [])[:20]:
        lines.append(
            "| {rank} | {scene} | {category} | {distance:.3f} | {ttc} | {learned:.3f} | {base:.3f} |".format(
                rank=int(score.get("rank") or 0),
                scene=str(score.get("scene_name") or ""),
                category=str(score.get("category_group") or ""),
                distance=float(score.get("distance_m") or 0.0),
                ttc=_format_float(score.get("ttc_s")),
                learned=float(score.get("learned_score") or 0.0),
                base=float(score.get("base_retrieval_score") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_float(value: Any) -> str:
    try:
        numeric = float(value)
    except Exception:  # noqa: BLE001
        return ""
    if not math.isfinite(numeric):
        return "inf"
    return "{0:.3f}".format(numeric)
