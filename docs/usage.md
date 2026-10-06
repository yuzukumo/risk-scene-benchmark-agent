# Usage

## Corrected Evaluation Protocols

The current main experiments use fixed selection without a fitted mode calibrator, three training seeds, and all held-out test clips. Select available GPUs explicitly. The CARLA graphics adapter in its YAML is independent of the PyTorch device index.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m nusc_scene_agent run-experiment-config --config configs/bench2drive_controlled_study.yaml
CUDA_VISIBLE_DEVICES=6 python -m nusc_scene_agent run-experiment-config --config configs/carla_fixed_evaluation.yaml
python -m nusc_scene_agent.risk_case_expansion --output outputs/risk_case_expansion_v2
python -m nusc_scene_agent run-contextvae-world-model-study --benchmark benchmarks/trainval_world_model_slices_v2.json --output outputs/contextvae_world_model_study_v3
CUDA_VISIBLE_DEVICES=6 python -m nusc_scene_agent run-experiment-config --config configs/forecast_validation_study.yaml
```

The six planner arms train for 24 epochs each with seeds 7, 17 and 27. Completed training requests can be resumed with the same command. A changed training configuration or manifest requires a new output directory. The fixed CARLA evaluator retains all attempts, including simulator errors; rerunning a completed protocol reads its recorded outcomes instead of selecting new attempts.

Evaluation caches verify checkpoint, manifest, evaluator source and result hashes. An invalid evaluation is regenerated while retaining the validation-selected checkpoint. For independent seed workers, use `python -m nusc_scene_agent.bench2drive_study --config configs/bench2drive_controlled_study.yaml --seeds 27 --defer-summary`; assign disjoint seeds to concurrent workers, then run the full command once to publish the aggregate.

After all experiments complete, publish the compact tracked results and figures:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/benchmark_planner_inference.py
python scripts/publish_benchmark_snapshot.py
```

Select an otherwise idle GPU for the optional model-cost measurement. It times synthetic on-device inputs at batch size 1 in FP32; it is not an end-to-end camera-to-control latency measurement.

The publisher rejects incomplete three-seed studies, invalid result manifests and forecast validation slices containing non-validation scenes. `python scripts/refresh_forecast_metrics.py --study outputs/forecast_validation_study_v1/contextvae` can recompute metrics from verified archived predictions without drawing new stochastic trajectories.

The validation-study configuration also declares the development benchmarks. Its scene audit removes overlapping scenes from the primary comparison, while retaining the complete official-validation result. To repeat just this audit on archived forecasts:

```bash
python -m nusc_scene_agent.forecast_validation --study outputs/forecast_validation_study_v1/contextvae --development benchmarks/trainval_perception_slices_v1.json outputs/risk_case_expansion_v2/perception.json --output outputs/forecast_validation_study_v1/scene_disjoint_audit
```

For a validated environment, install the direct dependencies in `requirements-validated.txt` and PyTorch 2.5.1 / torchvision 0.20.1 CUDA 12.1 wheels. See [evaluation_protocol.md](evaluation_protocol.md) for data budgets, metric semantics and known boundaries. The lower-level training and calibration commands below remain available for individual experiments; their results must not be mixed into the fixed-calibration ablation table.

This page lists commands for reproducing the local benchmark pipeline. Generated files are written under `outputs/`, `artifacts/`, or `external/`; these directories are excluded from version control.

## Environment

```bash
conda env create -f environment.yml
conda activate nuscenes
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate nuscenes
```

## Data Preparation

Check local archives:

```bash
python -m nusc_scene_agent inspect-archives --workspace .
```

Prepare `nuScenes v1.0-trainval` with map expansion:

```bash
python -m nusc_scene_agent prepare-data --workspace . --profile trainval-full
```

Build the trainval index:

```bash
python -m nusc_scene_agent build-index \
  --version v1.0-trainval \
  --dataroot data/sets/nuscenes \
  --db artifacts/index/v1.0-trainval.sqlite
```

Validate or stamp a structurally compatible index created by an earlier project version:

```bash
python -m nusc_scene_agent migrate-index \
  --db artifacts/index/v1.0-trainval.sqlite
```

## Local LLM

The benchmark suite uses an Ollama server for structured query planning and artifact review. Run `ollama serve` in a separate shell if the service is not already active.

```bash
ollama pull gemma4:latest
ollama serve
```

The default endpoint and model are also configurable:

```bash
export NUSC_SCENE_AGENT_OLLAMA_BASE_URL="http://127.0.0.1:11434"
export NUSC_SCENE_AGENT_OLLAMA_MODEL="gemma4:latest"
```

Record local model metadata when publishing benchmark artifacts:

```bash
python -m nusc_scene_agent inspect-ollama-model \
  --output outputs/ollama_model_metadata.json

export NUSC_SCENE_AGENT_OLLAMA_DIGEST="$(python -c 'import json; print(json.load(open("outputs/ollama_model_metadata.json"))["digest"])')"
```

The full-suite configuration requires this digest. The metadata file records the resolved model identifier and the tag used to load it.

## Full Benchmark Suite

```bash
python -m nusc_scene_agent run-full-benchmark-suite
```

To run all evaluation stages from the already validated case library without an Ollama service:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m nusc_scene_agent run-full-benchmark-suite --reuse-case-library
```

This explicit option records the library hash and writes `resolved_config.yaml` with the effective stage configuration. The LLM case-generation stage is skipped; the other configured stages still execute.

The supplied expansion and forecast-validation configurations use `reuse_validated: true`. When matching mined cases already exist, their artifact hashes, database identity, scene split and candidate limits are checked before re-exporting the current benchmarks. The report preserves the original mining provenance and records the export refresh separately. A fresh output directory performs mining normally; set this option to `false` to repeat validation. Reuse preserves the archived acceptance decisions and does not claim a new validation run.

The suite uses the shared scenario taxonomy in [configs/scenario_taxonomy.yaml](../configs/scenario_taxonomy.yaml) to keep dataset mining, replay evaluation, planner diagnostics, and semantic simulator demos aligned by risk family.

This executes the configured pipeline in [configs/full_benchmark_suite.yaml](../configs/full_benchmark_suite.yaml):

- case-library generation from `benchmarks/trainval_suite_v1.yaml`
- `nuScenes` scenario-mining, perception, BEV occupancy, and world-model benchmark generation
- `nuPlan` replay-regression sweep
- `nuPlan` closed-loop replay sweep
- scene-disjoint learned-retriever training and ContextVAE evaluation on the compatible forecast subset
- three-seed planner ablations and fixed live-camera CARLA attempts
- validated risk-case expansion with coverage and exclusion reports
- forecasting on independently mined official-validation scenes, excluded from development feedback
- model-in-the-loop failure mining
- failure-query reranking using the current mining output and retrained checkpoint
- result-registry export

## Stage-Level Commands

Run the `nuScenes` benchmark layers from an existing case library:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/risk_benchmark_suite.yaml
```

Compare retrieval score profiles on the same query suite:

```bash
python -m nusc_scene_agent benchmark-score-sweep \
  --config benchmarks/trainval_suite_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/retrieval_score_profile_sweep_v1
```

Compare validation-quality weighting profiles while holding retrieval and acceptance fixed:

```bash
python -m nusc_scene_agent benchmark-validation-score-sweep \
  --config benchmarks/trainval_suite_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --profile default --profile equal \
  --output outputs/validation_score_profile_sweep_v1
```

Run a behavior-threshold robustness sweep on the same trainval index:

```bash
python -m nusc_scene_agent benchmark-threshold-sweep \
  --config benchmarks/trainval_suite_v1.yaml \
  --db artifacts/index/v1.0-trainval.sqlite \
  --scale 0.85 --scale 1.0 --scale 1.15 \
  --output outputs/validation_threshold_sweep_v1
```

The validation-quality and threshold sweeps vary one deterministic component at a time. Their metrics measure anchor consistency and score sensitivity, not independent semantic recall.

Generate a fixed author-audit subset from the validated case library:

```bash
python -m nusc_scene_agent generate-human-audit-set \
  --case-library outputs/trainval_case_library_v1/case_library.json \
  --output audits/author_audit_v1 \
  --sample-size 100 \
  --seed 7
```

Review `audits/author_audit_v1/review_queue.md`, fill `human_audit_items.jsonl` or the CSV copy, then evaluate the completed labels:

```bash
python -m nusc_scene_agent evaluate-human-audit-set \
  --annotations audits/author_audit_v1/human_audit_items.jsonl \
  --output audits/author_audit_v1/evaluation
```

Train the weakly supervised query-scene reranker:

```bash
pip install -e ".[learned]"

python -m nusc_scene_agent train-large-learned-retriever \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/learned_retriever_trainval_v3 \
  --max-groups-per-family 1000 \
  --epochs 20 \
  --negatives-per-query 12
```

Evaluate failure-aware candidate generation:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/failure_aware_reranking.yaml
```

Run structural multimodal retrieval for a single query:

```bash
python -m nusc_scene_agent run-multimodal-retrieval \
  "pedestrian crossing close in front of ego with dense surrounding traffic context" \
  --db artifacts/index/v1.0-trainval.sqlite \
  --output outputs/multimodal_retrieval_v1 \
  --top-k 10 \
  --candidate-pool 64
```

Run the `ContextVAE` world-model baseline:

```bash
pip install -e ".[forecast]"
git clone https://github.com/xupei0610/ContextVAE.git external/ContextVAE

python -m nusc_scene_agent run-contextvae-world-model-study \
  --benchmark benchmarks/trainval_world_model_slices_v2.json \
  --dataroot data/sets/nuscenes \
  --version v1.0-trainval \
  --output outputs/contextvae_world_model_study_v3 \
  --repo external/ContextVAE \
  --checkpoint external/ContextVAE/models/nuscenes_res18 \
  --device cuda \
  --batch-size 4 \
  --mode-count 5 \
  --clustering-samples 2000
```

Run `nuPlan` replay and closed-loop sweeps:

```bash
python -m nusc_scene_agent inspect-nuplan --dataset-root data/nuplan/dataset

python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_replay_sweep_medium.yaml

python -m nusc_scene_agent run-experiment-config \
  --config configs/nuplan_closed_loop_sweep_medium.yaml
```

Inspect the CARLA runtime and mine semantic-gated visual closed-loop demos:

```bash
python -m nusc_scene_agent inspect-carla \
  --carla-root external/carla/latest

python -m nusc_scene_agent print-carla-launch-command \
  --carla-root external/carla/latest \
  --cuda-visible-devices 4

python -m nusc_scene_agent run-experiment-config \
  --config configs/carla_semantic_demo.yaml

python -m nusc_scene_agent audit-carla-vision-rollout \
  --report outputs/carla_semantic_demo_final/carla_semantic_demo_report.json \
  --output outputs/carla_semantic_demo_final/carla_semantic_demo_audit.json \
  --require-semantic-match
```

Train and evaluate the Bench2Drive vision planner. This stage builds a predecoded tensor cache and trains with `torchrun` distributed data parallelism.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[vision]"

python -m nusc_scene_agent inspect-bench2drive \
  --dataset-root data/bench2drive/Bench2Drive-Base

python -m nusc_scene_agent build-bench2drive-vision-manifest \
  --dataset-root data/bench2drive/Bench2Drive-Base \
  --output artifacts/bench2drive/vision_e2e_manifest.jsonl \
  --frame-stride 5 \
  --future-steps 5 \
  --future-frame-stride 5 \
  --cache-root data/bench2drive/cache/vision_e2e

python -m nusc_scene_agent build-bench2drive-vision-tensor-cache \
  --manifest artifacts/bench2drive/vision_e2e_manifest.jsonl \
  --output artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl \
  --cache-dir data/bench2drive/cache/vision_e2e_tensor_cache \
  --image-size 160 \
  --num-workers 8 \
  --chunk-rows 512

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m nusc_scene_agent train-bench2drive-vision-planner \
  --manifest artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl \
  --output outputs/bench2drive_vision_e2e_final \
  --epochs 24 \
  --batch-size 64 \
  --image-size 160 \
  --model-size research \
  --architecture trajectory_transformer \
  --camera-pooling transformer \
  --trajectory-modes 4 \
  --trajectory-selection expected \
  --trajectory-temperature 0.5 \
  --dropout 0.05 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --precision fp16 \
  --selection-metric risk_aware \
  --brake-loss-weight 0.25 \
  --brake-positive-weight 1.25 \
  --risk-sample-weight 1.25 \
  --lateral-loss-weight 2.0 \
  --turn-sample-weight 2.0 \
  --selected-waypoint-loss-weight 0.5 \
  --displacement-loss-weight 0.1 \
  --endpoint-loss-weight 0.025 \
  --path-length-loss-weight 0.025 \
  --mode-classification-weight 0.05

CUDA_VISIBLE_DEVICES=0 python -m nusc_scene_agent evaluate-bench2drive-vision-planner \
  --manifest artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl \
  --checkpoint outputs/bench2drive_vision_e2e_final/vision_e2e_planner_best.pt \
  --output outputs/bench2drive_vision_e2e_final/eval_val \
  --split val \
  --batch-size 128 \
  --image-size 160 \
  --num-workers 4 \
  --prefetch-factor 4

python -m nusc_scene_agent calibrate-bench2drive-trajectory-selection \
  --predictions outputs/bench2drive_vision_e2e_final/eval_val/predictions.jsonl \
  --evaluation-report outputs/bench2drive_vision_e2e_final/eval_val/evaluation_report.json \
  --checkpoint outputs/bench2drive_vision_e2e_final/vision_e2e_planner_best.pt \
  --output-checkpoint outputs/bench2drive_vision_e2e_final/vision_e2e_planner_calibrated.pt \
  --output-report outputs/bench2drive_vision_e2e_final/trajectory_selection_calibration_report.json

CUDA_VISIBLE_DEVICES=0 python -m nusc_scene_agent evaluate-bench2drive-vision-planner \
  --manifest artifacts/bench2drive/vision_e2e_manifest_tensor_160.jsonl \
  --checkpoint outputs/bench2drive_vision_e2e_final/vision_e2e_planner_calibrated.pt \
  --output outputs/bench2drive_vision_e2e_final/eval_test \
  --split test \
  --batch-size 128 \
  --image-size 160 \
  --num-workers 4 \
  --prefetch-factor 4

python -m nusc_scene_agent diagnose-bench2drive-vision-planner \
  --predictions outputs/bench2drive_vision_e2e_final/eval_test/predictions.jsonl \
  --evaluation-report outputs/bench2drive_vision_e2e_final/eval_test/evaluation_report.json \
  --output outputs/bench2drive_vision_e2e_final/diagnostics

CUDA_VISIBLE_DEVICES=0 python -m nusc_scene_agent run-experiment-config \
  --config configs/bench2drive_vision_closed_loop.yaml
```

Run failure mining from generated metric artifacts:

```bash
python -m nusc_scene_agent run-failure-mining \
  --input outputs/trainval_bev_occupancy_proxy_study_v1 \
  --input outputs/trainval_world_model_proxy_study_v2 \
  --input outputs/contextvae_world_model_study_v3 \
  --input outputs/contextvae_expanded_study_v1 \
  --input outputs/nuplan_replay_sweep_v1 \
  --input outputs/nuplan_closed_loop_sweep_v1 \
  --output outputs/model_in_the_loop_failure_mining_v2 \
  --max-queries 24
```

Export the result registry:

```bash
python -m nusc_scene_agent run-experiment-config \
  --config configs/result_registry.yaml
```

Run the LangGraph artifact review workflow:

```bash
pip install -e ".[agent]"

python -m nusc_scene_agent run-research-agent \
  --output outputs/research_agent_review_v1
```

Inspect local benchmark and dataset metadata:

```bash
python -m nusc_scene_agent inspect-dataset-backends \
  --output outputs/dataset_backends_inventory.json

python -m nusc_scene_agent export-benchmark-catalog \
  --output outputs/benchmark_catalog.json
```
