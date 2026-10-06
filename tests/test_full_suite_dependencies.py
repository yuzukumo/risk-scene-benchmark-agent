import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from nusc_scene_agent.experiment_config import _full_suite_failure_inputs, run_experiment_config


class FullSuiteDependenciesTest(unittest.TestCase):
    def test_failure_mining_cannot_silently_load_historical_outputs(self):
        with self.assertRaisesRegex(ValueError, "current evaluation"):
            _full_suite_failure_inputs({})
        self.assertEqual(_full_suite_failure_inputs({
            "risk_benchmark_suite": {"proxy_studies": {"world_model": {"output_dir": "new/forecast"}}},
            "contextvae_world_model_study": {"output_dir": "new/contextvae"},
        }), ["new/forecast", "new/contextvae"])

    def test_current_forecast_checkpoint_and_failure_queries_flow_through_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "suite.yaml"
            library = root / "cases.json"
            library.write_text(json.dumps([{"passed": True}]))
            stages = {name: False for name in ["case_library_generation", "nuplan_replay_sweep", "nuplan_closed_loop_sweep"]}
            stages["case_library_generation"] = True
            stages.update({name: True for name in ["risk_benchmark_suite", "learned_retriever_training",
                                                  "contextvae_world_model_study", "failure_mining",
                                                  "failure_aware_reranking", "result_registry"]})
            config_path.write_text(yaml.safe_dump({
                "experiment": {"type": "full_benchmark_suite", "output": str(root)},
                "full_benchmark_suite": {"stages": stages},
                "risk_benchmark_suite": {"output": str(root / "risk"), "result_path": str(root / "risk/result.json"),
                                         "case_library": str(library)},
                "failure_mining": {"output": str(root / "failures")},
            }))
            module = "nusc_scene_agent.experiment_config."
            with patch(module + "_run_case_library_generation_experiment", side_effect=AssertionError("LLM generation must be disabled")), patch(module + "_run_risk_benchmark_suite_experiment", return_value={
                "outputs": {"world_model": "current/forecast.json"},
                "proxy_studies": {"world_model": {"output_dir": "current/proxy"}},
            }), patch(module + "_run_learned_retriever_training_experiment", return_value={
                "output_dir": "current/retriever",
            }), patch(module + "_run_contextvae_experiment", return_value={
                "output_dir": "current/contextvae",
            }) as forecast, patch(module + "_run_failure_mining_experiment", return_value={
                "output_dir": "current/failures", "report_json": "current/failures/failure_mining_report.json",
            }) as mining, patch(module + "_run_failure_aware_reranking_experiment", return_value={
                "output_dir": "current/reranking",
            }) as reranking, patch(module + "write_result_registry", return_value={}) as registry:
                payload = run_experiment_config(config_path, reuse_case_library=True)
            self.assertEqual(forecast.call_args.args[0]["contextvae_world_model_study"]["benchmark"], "current/forecast.json")
            self.assertEqual(mining.call_args.args[0]["failure_mining"]["inputs"], ["current/proxy", "current/contextvae"])
            rerank_config = reranking.call_args.args[0]["failure_aware_reranking"]
            self.assertEqual(rerank_config["query_config"], "current/failures/failure_update_queries.yaml")
            self.assertEqual(rerank_config["learned_checkpoint"], "current/retriever/learned_retriever.pt")
            self.assertIn(Path("current/reranking/failure_aware_reranking_eval.json"), registry.call_args.kwargs["sources"])
            self.assertIn("failure_aware_reranking", payload["result"]["stages"])
            self.assertEqual(payload["runtime_overrides"]["reused_case_library"]["path"], str(library))
            resolved = yaml.safe_load((root / "resolved_config.yaml").read_text())
            self.assertFalse(resolved["full_benchmark_suite"]["stages"]["case_library_generation"])
