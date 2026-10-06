import unittest

from nusc_scene_agent.world_model_benchmark import build_world_model_comparison


class ForecastUncertaintyTest(unittest.TestCase):
    def test_all_profile_metrics_use_the_same_case_intersection(self):
        shared = {"benchmark_group": "shared", "scene_token": "scene_a", "ade_m": 2,
                  "min_ade_at_1": 3, "min_fde_at_5": 4, "closest_approach_time_error_s": 5}
        result = build_world_model_comparison([
            {"profile_name": "baseline", "overview": {"case_count": 2, "mean_closest_approach_time_error_s": 99},
             "forecast_metrics": {"mean_min_ade_at_1": 99, "mean_min_fde_at_5": 99},
             "case_metrics": [shared, {**shared, "benchmark_group": "excluded", "min_ade_at_1": 195}]},
            {"profile_name": "candidate", "case_metrics": [shared]},
        ])
        for profile in result["profiles"]:
            self.assertEqual(profile["case_count"], 1)
            self.assertEqual(profile["mean_min_ade_at_1"], 3)
            self.assertEqual(profile["mean_min_fde_at_5"], 4)
            self.assertEqual(profile["mean_closest_approach_time_error_s"], 5)

    def test_duplicating_correlated_targets_does_not_create_independent_evidence(self):
        def comparison(copies):
            summaries = []
            for name, shift in [("baseline", 0), ("candidate", 1)]:
                rows = [{"benchmark_group": f"case_{scene}_{copy}", "scene_token": scene,
                         "ade_m": value + shift, "fde_m": value + shift}
                        for scene, value in [("scene_a", 1), ("scene_b", 9)] for copy in range(copies)]
                summaries.append({"profile_name": name, "case_metrics": rows})
            return build_world_model_comparison(summaries)
        original, duplicated = comparison(1), comparison(4)
        self.assertEqual(duplicated["overview"]["bootstrap_cluster_count"], 2)
        for field in ["mean", "ci95_low", "ci95_high"]:
            self.assertEqual(original["profiles"][0]["uncertainty"]["ade_m"][field],
                             duplicated["profiles"][0]["uncertainty"]["ade_m"][field])

    def test_missing_fde_is_not_converted_to_zero_in_paired_intervals(self):
        result = build_world_model_comparison([
            {"profile_name": "baseline", "case_metrics": [
                {"benchmark_group": "a", "scene_token": "scene_a", "fde_m": None},
                {"benchmark_group": "b", "scene_token": "scene_b", "fde_m": 10},
            ]},
            {"profile_name": "candidate", "case_metrics": [
                {"benchmark_group": "a", "scene_token": "scene_a", "fde_m": 5},
                {"benchmark_group": "b", "scene_token": "scene_b", "fde_m": 11},
            ]},
        ])
        delta = result["paired_profile_comparisons"][0]["deltas"]["fde_m"]
        self.assertEqual(delta["case_count"], 1)
        self.assertEqual(delta["profile_b_minus_profile_a"], 1)
        self.assertEqual(delta["ci95_low"], 1)
        self.assertEqual(delta["ci95_high"], 1)
