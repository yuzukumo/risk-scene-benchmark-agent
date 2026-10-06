# Planner Development

The current planner is a lightweight, navigation-conditioned imitation-learning baseline. It consumes one timestamp of six 160x160 camera images, route targets, route commands and current ego speed. A CNN and Transformer predict four modes of five future waypoints, plus control and braking outputs. Live driving uses the predicted waypoints through a low-level controller. This is not a pretrained driving foundation model.

## What Is Implemented

The controlled study adds a navigation-only ablation, 4x4 spatial pooling, optional geometric trajectory supervision, and equal-budget random versus mined-prior sampling. All six arms use three training seeds, validation-selected checkpoints and the same trajectory selection. [The result snapshot](benchmark_snapshot.md) reports their measured effects. Additional losses or tokens are not assumed to improve driving before the comparison finishes.

The replay evaluator fixes reference-route construction, coordinate anchoring and timestamp alignment. Its images still come from the recording, so it cannot measure visual recovery from the planner's own deviations. The fixed CARLA suite supplies a separate live-camera check and retains failures. It currently evaluates the legacy checkpoint identified in its protocol; it does not establish simulator performance for the new ablation arms.

## Priority Experiments

| Priority | Evidence and limitation | Proposed change | Acceptance criterion |
| --- | --- | --- | --- |
| 1. Braking and temporal context | All three recorded CARLA collisions occur on `adjacent_50`. In seed 7, the last pre-contact observation has speed 4.28 m/s, brake probability 0.261 and applied brake 0.0, below the configured 0.82 threshold. A single image does not directly represent relative motion. | Add a short, ego-aligned history with cached visual features; add explicit hazard or stopping supervision. Select any braking threshold using validation data. | Better brake recall and earlier braking on held-out events, without excessive false stops; confirm on previously unused live routes. Do not tune on the three reported failures and call them an unbiased test. |
| 2. Visual representation | The encoder is trained from scratch on 1,000 clips, with small images and a small number of spatial tokens. Camera identity embeddings do not supply calibrated 3D geometry. | Compare a pretrained image backbone, higher input resolution and camera-geometry positional features as separate experiments. | Gains over both the visual baseline and navigation-only control across seeds, with latency and memory reported. Confirm that gains survive different towns and appearance conditions. |
| 3. Multimodal decisions | Expected-trajectory selection averages mode coordinates. Distinct turn or avoidance hypotheses can produce an unsuitable intermediate path. | Learn route- and obstacle-conditioned candidate ranking, and retain distinct trajectory hypotheses through selection. | Improve selected-path errors and collisions, not just oracle minADE. No future ground truth may enter deployed selection. |
| 4. Training distribution | Scenario-family priors change class balance, but they do not directly target the particular mechanism behind a failure. Training observations remain expert-driven. | Mine braking, queueing, interaction and recovery events; collect perturbed live rollouts with expert recovery targets. Compare random, family-prior and failure-driven selection at equal sample and compute budgets. | Improvements on a frozen set of unseen routes and event families, with all interventions and failures counted. |

Temporal features should use only observations at or before the current timestamp. History must stop at clip boundaries, preserve camera order, and align past ego coordinates. Repeating a current frame is a required control for any claimed temporal benefit. Increasing image resolution or history length also changes inference cost, so evaluate latency alongside accuracy.

Displacement, endpoint and path-length losses provide trajectory geometry supervision. They do not enforce steering limits, acceleration bounds, tire dynamics or collision avoidance. A dynamics-constrained extension needs an explicit vehicle model and feasibility metrics before it can support those claims.

## Evaluation Order

1. Freeze the new hypothesis, training budget, seeds and checkpoint-selection rule.
2. Run the existing archive-disjoint open-loop comparison and paired clip statistics.
3. Test the selected checkpoints in live CARLA with fixed navigation and assistance settings. Reserve new routes after examining the current failure set.
4. Expand to a published simulator protocol before comparing driving scores with other research systems.

Three seeds quantify training variation, while the 97 held-out clips quantify the available test coverage. The five local CARLA configurations are too few to establish broad driving reliability. Independent human labels are still required for semantic-retrieval accuracy; deterministic weak rules cannot supply that evidence.

## Design References

[SparseDrive](https://arxiv.org/abs/2405.19620) is a reference for sparse scene representation and motion-planning interaction. [NAVSIM](https://arxiv.org/abs/2406.15349) motivates careful distinction between non-reactive evaluation and simulator closed loop. [Bench2Drive](https://arxiv.org/abs/2406.03877) provides a published CARLA evaluation protocol. These are design and protocol references, not systems reproduced by this repository or a claim about the current state-of-the-art ranking.
