import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nusc_scene_agent.learned_retrieval import (
    LearnedRetrieverConfig,
    run_learned_retrieval_report,
    train_learned_scene_retriever,
    train_weakly_supervised_scene_retriever,
    _build_weak_training_groups,
    _split_groups,
    _training_split_integrity,
)


def _insert_agent(conn: sqlite3.Connection, **values: object) -> None:
    columns = [
        "ann_token",
        "sample_token",
        "scene_token",
        "scene_name",
        "sample_idx",
        "instance_token",
        "category_name",
        "category_group",
        "distance",
        "ttc",
        "x_ego",
        "y_ego",
        "speed",
        "rel_vx",
        "rel_vy",
        "heading_delta",
        "is_stationary",
        "is_front",
        "is_rear",
        "is_left",
        "is_right",
        "num_lidar_pts",
        "num_radar_pts",
    ]
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        "INSERT INTO agents ({0}) VALUES ({1})".format(", ".join(columns), placeholders),
        [values[column] for column in columns],
    )


def _write_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE samples (
                sample_token TEXT PRIMARY KEY,
                location TEXT,
                scene_description TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE agents (
                ann_token TEXT PRIMARY KEY,
                sample_token TEXT,
                scene_token TEXT,
                scene_name TEXT,
                sample_idx INTEGER,
                instance_token TEXT,
                category_name TEXT,
                category_group TEXT,
                distance REAL,
                ttc REAL,
                x_ego REAL,
                y_ego REAL,
                speed REAL,
                rel_vx REAL,
                rel_vy REAL,
                heading_delta REAL,
                is_stationary INTEGER,
                is_front INTEGER,
                is_rear INTEGER,
                is_left INTEGER,
                is_right INTEGER,
                num_lidar_pts INTEGER,
                num_radar_pts INTEGER
            )
            """
        )
        for token in [
            "sample_ped_pos",
            "sample_ped_neg",
            "sample_vehicle_pos",
            "sample_vehicle_neg",
        ]:
            conn.execute(
                "INSERT INTO samples VALUES (?, ?, ?)",
                (token, "singapore-onenorth", "unit-test scene"),
            )

        _insert_agent(
            conn,
            ann_token="ann_ped_pos",
            sample_token="sample_ped_pos",
            scene_token="scene_ped",
            scene_name="scene-ped",
            sample_idx=10,
            instance_token="inst_ped_pos",
            category_name="human.pedestrian.adult",
            category_group="pedestrian",
            distance=8.0,
            ttc=2.0,
            x_ego=7.0,
            y_ego=2.5,
            speed=1.2,
            rel_vx=-0.5,
            rel_vy=1.8,
            heading_delta=0.2,
            is_stationary=0,
            is_front=1,
            is_rear=0,
            is_left=1,
            is_right=0,
            num_lidar_pts=12,
            num_radar_pts=2,
        )
        _insert_agent(
            conn,
            ann_token="ann_ped_neg",
            sample_token="sample_ped_neg",
            scene_token="scene_ped",
            scene_name="scene-ped-neg",
            sample_idx=11,
            instance_token="inst_ped_neg",
            category_name="human.pedestrian.adult",
            category_group="pedestrian",
            distance=18.0,
            ttc=9.0,
            x_ego=17.0,
            y_ego=0.3,
            speed=0.3,
            rel_vx=0.0,
            rel_vy=0.0,
            heading_delta=0.0,
            is_stationary=0,
            is_front=1,
            is_rear=0,
            is_left=0,
            is_right=0,
            num_lidar_pts=5,
            num_radar_pts=0,
        )
        _insert_agent(
            conn,
            ann_token="ann_vehicle_pos",
            sample_token="sample_vehicle_pos",
            scene_token="scene_vehicle",
            scene_name="scene-vehicle",
            sample_idx=20,
            instance_token="inst_vehicle_pos",
            category_name="vehicle.car",
            category_group="vehicle",
            distance=6.0,
            ttc=3.0,
            x_ego=6.0,
            y_ego=0.1,
            speed=0.0,
            rel_vx=-1.0,
            rel_vy=0.0,
            heading_delta=0.0,
            is_stationary=1,
            is_front=1,
            is_rear=0,
            is_left=0,
            is_right=0,
            num_lidar_pts=20,
            num_radar_pts=4,
        )
        _insert_agent(
            conn,
            ann_token="ann_vehicle_neg",
            sample_token="sample_vehicle_neg",
            scene_token="scene_vehicle",
            scene_name="scene-vehicle-neg",
            sample_idx=21,
            instance_token="inst_vehicle_neg",
            category_name="vehicle.car",
            category_group="vehicle",
            distance=16.0,
            ttc=8.0,
            x_ego=16.0,
            y_ego=2.4,
            speed=8.0,
            rel_vx=2.0,
            rel_vy=0.0,
            heading_delta=0.0,
            is_stationary=1,
            is_front=1,
            is_rear=0,
            is_left=1,
            is_right=0,
            num_lidar_pts=8,
            num_radar_pts=1,
        )
        conn.commit()
    finally:
        conn.close()


def _write_fixture_benchmark(path: Path) -> None:
    path.write_text(
        """
queries:
  - id: pedestrian_crossing_anchor
    description: pedestrian crossing anchor
    query:
      natural_language: pedestrian crossing close in front of ego
      actors: [pedestrian]
      positions: [front]
      behaviors: [crossing]
      risk_terms: [risky, urgent]
      thresholds:
        near_distance_m: 20
        max_ttc_s: 5
    reference_case_keys:
      - sample_ped_pos:inst_ped_pos
  - id: stopped_lead_anchor
    description: stopped lead vehicle anchor
    query:
      natural_language: stopped lead vehicle ahead of ego
      actors: [vehicle]
      positions: [front]
      behaviors: [stopped_lead]
      risk_terms: [risky]
      thresholds:
        near_distance_m: 20
        max_ttc_s: 5
    reference_case_keys:
      - sample_vehicle_pos:inst_vehicle_pos
""".lstrip(),
        encoding="utf-8",
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for learned retrieval tests.")
class LearnedRetrievalTest(unittest.TestCase):
    def test_train_checkpoint_and_write_retrieval_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "index.sqlite"
            benchmark_path = root / "benchmark.yaml"
            output_dir = root / "learned"
            report_dir = root / "report"

            _write_fixture_db(db_path)
            _write_fixture_benchmark(benchmark_path)

            report = train_learned_scene_retriever(
                benchmark_path=benchmark_path,
                db_path=db_path,
                output_dir=output_dir,
                config=LearnedRetrieverConfig(
                    text_hash_dim=32,
                    hidden_dim=16,
                    embedding_dim=8,
                    negatives_per_query=1,
                    candidate_pool=4,
                    epochs=3,
                    validation_fraction=0.5,
                    device="cpu",
                ),
            )

            checkpoint = Path(report["checkpoint_path"])
            self.assertTrue(checkpoint.exists())
            self.assertEqual(report["group_count"], 2)
            self.assertEqual(report["validation_split_key"], "scene_token")
            self.assertTrue(report["split_integrity"]["scene_disjoint"])
            self.assertTrue(report["split_integrity"]["annotation_disjoint"])
            self.assertTrue((output_dir / "training_report.json").exists())

            result = run_learned_retrieval_report(
                db_path=db_path,
                query_text="pedestrian crossing close in front of ego",
                checkpoint_path=checkpoint,
                output_dir=report_dir,
                top_k=2,
                candidate_pool=4,
            )
            payload = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "learned_scene_retrieval_report_v1")
            self.assertEqual(payload["candidate_count"], 2)
            self.assertEqual([item["rank"] for item in payload["scores"]], [1, 2])
            self.assertTrue(Path(result["csv"]).exists())
            self.assertTrue(Path(result["markdown"]).exists())

    def test_train_weakly_supervised_checkpoint_from_index_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "index.sqlite"
            output_dir = root / "weak_learned"
            _write_fixture_db(db_path)
            with sqlite3.connect(db_path) as conn:
                # Each scene contains positive and negative examples of both families.
                conn.execute("UPDATE agents SET scene_token = 'scene_base'")
                for index in range(12):
                    conn.execute(
                        "INSERT INTO agents SELECT ann_token || ?, sample_token, ?, scene_name, sample_idx, "
                        "instance_token || ?, category_name, category_group, distance, ttc, x_ego, y_ego, "
                        "speed, rel_vx, rel_vy, heading_delta, is_stationary, is_front, is_rear, is_left, "
                        "is_right, num_lidar_pts, num_radar_pts FROM agents WHERE scene_token = 'scene_base'",
                        (f"_{index}", f"scene_{index}", f"_{index}"),
                    )

            report = train_weakly_supervised_scene_retriever(
                db_path=db_path,
                output_dir=output_dir,
                max_groups_per_family=20,
                config=LearnedRetrieverConfig(
                    text_hash_dim=32,
                    hidden_dim=16,
                    embedding_dim=8,
                    negatives_per_query=1,
                    epochs=2,
                    validation_fraction=0.5,
                    device="cpu",
                ),
            )

            self.assertGreaterEqual(report["group_count"], 2)
            self.assertEqual(report["training_source"], "weak_supervision_from_trainval_index")
            self.assertTrue(Path(report["checkpoint_path"]).exists())
            self.assertTrue((output_dir / "training_report.md").exists())
            self.assertTrue(report["split_integrity"]["scene_disjoint"])
            self.assertTrue(report["split_integrity"]["annotation_disjoint"])
            self.assertGreater(report["validation_group_count"], 0)
            groups, _ = _build_weak_training_groups(db_path, LearnedRetrieverConfig(negatives_per_query=2), 20)
            train, validation = _split_groups(groups, 0.25, 7, "scene_token")
            self.assertTrue(_training_split_integrity(train, validation)["scene_disjoint"])
            for partition in (train, validation):
                positive_scenes = {group.positive.scene_token for group in partition}
                negative_scenes = {candidate.scene_token for group in partition for candidate in group.negatives}
                other = validation if partition is train else train
                other_scenes = {candidate.scene_token for group in other for candidate in [group.positive, *group.negatives]}
                self.assertFalse((positive_scenes | negative_scenes) & other_scenes)


if __name__ == "__main__":
    unittest.main()
