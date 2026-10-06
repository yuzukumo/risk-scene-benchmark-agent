# Evaluation Protocol

## Data and Selection

Bench2Drive data contains 1,000 archived clips and 44,940 cached frames: 35,629 training, 4,977 validation, and 4,334 test frames. Splits are archive-disjoint. The test set contains 97 clips. Configurations use the historical Base dataset present in this workspace; these are not the current official simulator evaluation routes.

The controlled study runs seeds 7, 17 and 27. Each full-data arm uses the same 24-epoch schedule, learning rate, local batch size, device count and checkpoint-selection criterion. The two half-data arms each select 17,814 training frames without replacement, leaving validation and test unchanged. Checkpoint selection uses validation metrics, never replay or test outcomes.

The risk-prior arm derives scenario-family counts from passed nuScenes cases and maps these to Bench2Drive families. Sampling uses a 50/50 mixture of uniform probability and the mapped prior corrected for family availability. This is a taxonomy-guided sampling experiment, not scene reconstruction or domain adaptation.

## Planner

The baseline uses single-timestamp six-camera images at 160x160, a CNN encoder, four spatial tokens per camera, route features and a four-layer Transformer with four trajectory modes. Ablations replace images with zeros throughout training and inference, increase pooling to 4x4, or add displacement, endpoint and path-length supervision. All retain the same expected-trajectory selection rule at temperature 0.5. A fitted mode calibrator is not used in the controlled comparison.

## Logged-Sensor Replay

Protocol `logged_sensor_replay_v2`:

- Only logged poses inside the evaluation window define the reference route.
- The initial ego pose is included at t=0. Predictions at t drive integration toward the observation timestamp t+dt.
- Source frame IDs define elapsed time at 10 Hz. Large observation gaps are integrated in bounded substeps.
- Predictions are transformed from the logged ego frame, which generated their image and navigation inputs.
- All 97 test clips are included, with up to 21 observations and a 10-second horizon per clip.
- Sensor images and navigation remain fixed to the recording. Other actors do not react. Completion and custom tracking scores are diagnostic measures.

Reports with different protocol versions or control settings cannot be paired. Within each seed, confidence intervals resample paired clips/cases. Across seeds, means and standard deviations are reported separately; repeated training runs are not treated as extra independent driving scenes.

## CARLA

The local fixed suite contains two dense-traffic routes, two adjacent-traffic routes, and one pedestrian-crossing configuration, each with seeds 7, 17 and 27. Vehicles use CARLA Traffic Manager. Pedestrians use explicitly configured crossing behavior. Safety override, lane guard, and traffic-light conditioning are disabled.

A local protocol pass requires no collision, at least 90% completion, and zero safety intervention. This does not measure all traffic-law infractions or equal an official Bench2Drive success. Every planned attempt remains in the denominator; simulator errors are identified separately. The qualitative README video uses different assistance settings and is never pooled with these attempts.

## Forecasts and Retrieval

Forecast generation preserves the mined anchor and retrieves the actor track outside the narrow event window: up to 2 seconds of history and 6 seconds of future, with a minimum 3-second future for the revised benchmark. Missing futures are excluded without moving the anchor. ContextVAE requires continuous actor observations and 12 future keyframes; duplicate actor-anchor pairs are removed before export. Baselines use the same compatible subset.

Forecast mining filters official nuScenes validation scene names before candidate retrieval. A subsequent audit removes all scenes present in the original and expanded development perception benchmarks, including actors that differ from those used during development. The primary comparison uses this disjoint subset; the complete official-validation slice remains a secondary result. Exclusion depends on scene identity, never model quality. Both validation outputs are excluded from the development failure-query feedback loop. Original and expanded slices containing official training scenes are labeled development diagnostics. All exclusions and actual target durations remain visible; complete prediction coverage of a case does not imply that every case has a full 6-second target.

Forecast uncertainty uses 5,000 paired bootstrap resamples of scene clusters. All actor-anchor cases from a resampled scene move together. The same scene weights are used for model comparisons, and missing metrics remain missing. These intervals describe uncertainty within the mined slice; they do not correct its selection bias or substitute for an official prediction benchmark.

Learned retrieval partitions scenes before sampling positive-negative groups. A candidate scene cannot appear in both training and validation. Its labels remain deterministic weak rules. Human semantic annotations are not available and no semantic recall claim is made.

## Reproducibility

Controlled-run requests record the manifest hash, model-source hash, training parameters, seed and sampling allocation. Completed runs are reused only under the same training request. Evaluation caches additionally require matching checkpoint and evaluator-source hashes, and verified prediction and report files. Truncated or stale evaluations are regenerated without selecting another checkpoint. Result reports retain runtime provenance. Artifact hashes are computed after the referenced result files are final; the manifest is not embedded recursively in its own source artifacts.

`run-full-benchmark-suite --reuse-case-library` explicitly reuses the existing validated case library when LLM generation is unavailable or unnecessary. The result records that library's hash and the resolved configuration. It does not claim that the LLM generation stage was reproduced.

`requirements-validated.txt` pins direct dependencies observed in the experiment environment. It is not a complete cross-platform lock file. Dataset versions, CUDA wheels, CARLA version and external checkpoints remain explicit prerequisites.
