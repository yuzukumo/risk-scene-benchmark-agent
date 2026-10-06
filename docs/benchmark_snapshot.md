# Benchmark Snapshot

Results use the corrected evaluation protocols described in [evaluation_protocol.md](evaluation_protocol.md). Legacy 64-case Bench2Drive replay results are withdrawn: their reference route contained a backward splice and states were compared at different timestamps. Old confidence intervals do not repair those errors.

## Controlled Planner Study

Configuration: [bench2drive_controlled_study.yaml](../configs/bench2drive_controlled_study.yaml).

Six arms are evaluated with training seeds 7, 17 and 27: baseline, geometric trajectory supervision, navigation-only input, 4x4 spatial pooling, random half-data sampling and mined-prior half-data sampling. All use identical trajectory selection without fitted mode calibration. Test evaluation includes 4,334 samples from all 97 held-out clips.

The completed study publishes `outputs/bench2drive_controlled_study_v2/study_summary.json`, per-seed comparisons, training requests, input hashes and model checkpoints. Compact results and source hashes are retained in [results_snapshot.json](results_snapshot.json).

<!-- CONTROLLED_RESULTS_START -->
The controlled study is running. No multi-seed improvement claim is made before all configured arms finish.
<!-- CONTROLLED_RESULTS_END -->

## Learned Retrieval

<!-- RETRIEVAL_RESULTS_START -->
| Quantity | Corrected result |
| --- | ---: |
| Weak-rule groups | 4,000 |
| Train / validation groups | 3,209 / 791 |
| Train / validation candidate scenes | 678 / 169 |
| Shared candidate scenes | 0 |
| Shared annotations | 0 |
| Validation weak-rule consistency@1 | 1.000 |
| Failure-query acceptance@1, rule / learned | 23/24 / 23/24 |
| Failure-query acceptance@K, rule / learned | 23/24 / 24/24 |

Learned-minus-rule mean top-1 validation quality is -2.7454; mean best-candidate quality is -0.2167. The recorded selection policy is `rule_ranked_validation` (learned final ranker selected: false; learned candidate generator selected: false).

Sources: `outputs/learned_retriever_trainval_v3/training_report.json` and `outputs/failure_aware_reranking_eval_v4/failure_aware_reranking_eval.json`.
<!-- RETRIEVAL_RESULTS_END -->

Deterministic validation makes the final selection. The labels are rule-derived. These are neither independent human semantic labels nor an unbiased estimate of unseen natural-language query performance. The earlier scene-held-out result was invalid because training negatives contained validation scenes.

## Forecast Slices

The revised benchmark preserves the mined anchor and obtains future actor observations from the database beyond the original event window. It retains 19 of the original 24 anchors with at least 3 seconds of future. Five cover approximately 6 seconds. Excluded cases and actual horizons are stored in [trainval_world_model_slices_v2.json](../benchmarks/trainval_world_model_slices_v2.json).

ContextVAE preparation additionally requires continuous actor context and 12 future keyframes, and deduplicates identical actor-anchor pairs.

<!-- FORECAST_RESULTS_START -->
Official-validation mining retains 22 compatible actor-anchor pairs from 70 forecast cases after continuity checks. A scene-identity audit excludes 7 cases from 6 scenes already present in the declared development benchmarks. The primary comparison below uses the remaining 15 cases from 14 scenes, with zero development-scene overlap. Exclusion depends only on scene identity; the archived predictions are unchanged. This validation slice is excluded from the development failure-query feedback loop.

| Model | Common cases | ADE (m) | FDE (m) | Risk fidelity |
| --- | ---: | ---: | ---: | ---: |
| Constant velocity and heading | 15 | 1.820 | 4.362 | 0.690 |
| ContextVAE | 15 | 3.427 | 9.372 | 0.483 |
| Physics oracle, GT-selected upper bound | 15 | 1.160 | 3.004 | 0.705 |

ContextVAE minus constant-velocity ADE is +1.606 m (paired scene-bootstrap 95% interval [-0.842, +3.336] m; 14 scene clusters). Lower ADE is better. Physics oracle selects a motion model using future ground truth and is not deployable.

Target durations range from 5.55 to 6.05 seconds; 6 cases cover approximately 6 seconds. Full-horizon coverage means covering each case's available targets, not a uniform 6-second benchmark.

The complete 22-case official-validation slice is retained as a secondary result: CV ADE 1.719 m and ContextVAE ADE 3.946 m. It includes development-scene overlap.

The other slices are development diagnostics because they include official training scenes. They are not pooled with the primary validation slice:

| Development slice | Cases / scenes | Train / val cases | CV ADE (m) | ContextVAE ADE (m) |
| --- | ---: | ---: | ---: | ---: |
| Original | 9 / 7 | 5 / 4 | 4.253 | 4.625 |
| Expanded | 28 / 26 | 23 / 5 | 1.315 | 2.913 |

Sources and paired intervals are retained in [results_snapshot.json](results_snapshot.json). These are mined slices with substantial exclusions, not the official nuScenes prediction benchmark.
<!-- FORECAST_RESULTS_END -->

Controlled proxy perturbations on the 19 revised cases yield kinematic ADE 2.801 m and risk fidelity 0.750. Oracle and perturbed-ground-truth outputs validate metric sensitivity, not learned world-model capability. Sparse occupancy uses actor-center cells, not dense scene occupancy.

## Fixed CARLA Evaluation

Protocol: [carla_fixed_evaluation.yaml](../configs/carla_fixed_evaluation.yaml). Five predetermined scenario configurations are repeated with three seeds. All attempts are retained. Safety override, lane guard and traffic-light conditioning are disabled. The evaluated legacy planner remains identified by its checkpoint hash; these results are separate from the newly trained ablation arms.

<!-- CARLA_RESULTS_START -->
The fixed suite completed 15/15 attempts with no simulator errors: 12 passed, 3 recorded collisions, and 94.2% mean route completion. All three collisions occurred on `adjacent_50`. Safety intervention was disabled throughout. These results evaluate the retained legacy checkpoint under the local protocol, separately from the newly trained ablation arms and official Bench2Drive routes.
<!-- CARLA_RESULTS_END -->

The retained README pedestrian demonstration is separate: 272 frames, 27.1 seconds, 13 Traffic Manager vehicles, 9 controlled pedestrians, no recorded collision, and a 9.6% safety-override ratio. It was selected for qualitative presentation and used traffic-light conditioning. Its one retained success is not a driving success rate.

## Risk-Case Coverage

The original case library contains 33 unique entries, of which 30 pass deterministic validation. Its compact benchmark has 24 anchors and 48 canonical/paraphrase queries. Expanded coverage is generated by `risk_case_expansion.py` using scene-balanced candidates, distinct actors and temporal/map validation; rejected attempts are recorded.

<!-- COVERAGE_RESULTS_START -->
The expanded development library retains 71 cases from 61 scenes: 10 crossing, 30 stopped-lead, 1 cut-in and 30 oncoming cases. Of these, 70 have at least 3 seconds of future targets. Uneven family counts and rejected candidates remain in `outputs/risk_case_expansion_v2/expansion_report.json`.
<!-- COVERAGE_RESULTS_END -->

The separate official-validation mining run retains 74 cases from 60 scenes, including 4 cut-ins; 70 have sufficient forecast targets and 22 satisfy the additional ContextVAE continuity requirements. Its results are not fed into development query generation.

No independent human semantic audit has been completed. Larger rule-validated samples improve coverage but do not remove this limitation.

## nuPlan and Perception Infrastructure

The nuPlan replay studies retain 112 sampled windows from 576 inspected logs. The history-kinematic profile has replay ADE 0.916 m and replay-simulation ADE 1.027 m. These are short-window diagnostics with logged actors and handcrafted profiles, not official nuPlan results for a trained planner.

Perception and actor-center occupancy slices provide adapters, ground-truth alignment and metric checks. Their oracle or controlled-drop results are infrastructure checks; they are not evidence of a newly trained detector or dense BEV occupancy model.

## Reproducibility

The full-suite runner executes controlled planner training and fixed CARLA evaluation as explicit stages. Training requests record sample budgets, splits, seeds, model settings and input hashes. Per-attempt simulator errors remain visible. Artifact files are hashed after their final write; repaired historical manifests preserve their original provenance and identify the hash-only repair.

Direct dependency versions are in [requirements-validated.txt](../requirements-validated.txt). Unit tests run in CI without private datasets. Full training, forecast and simulator reproduction additionally require the documented local datasets, checkpoints and simulator installation.
