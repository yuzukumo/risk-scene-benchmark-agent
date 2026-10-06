# Autonomous Driving Risk-Scenario Benchmark

**Mining validated risk scenarios from real-world data to improve end-to-end planner training and evaluation.**

[English](README.md) | [简体中文](README.zh-CN.md)

---

## Abstract

This work studies whether validated risk scenarios mined from driving logs can improve data selection for training and systematically expose planner failures. We present an end-to-end framework spanning scenario retrieval, controlled planner training, and multi-backend evaluation (Bench2Drive, nuPlan, CARLA). The system uses local Ollama-based query planning with deterministic spatiotemporal validation, trains vision-based imitation planners under controlled ablations, and evaluates them through logged-sensor replay and live closed-loop simulation.

**Key contributions:**
- Archive-disjoint scenario mining with 1,000 weak-rule consistency and 100% failure-query coverage
- Controlled planner study showing dynamics regularization reduces closed-loop ADE by 13.7% (11.26 → 9.72 m)
- Fixed-protocol CARLA evaluation achieving 94.2% route completion across 15 independent attempts
- Reproducible evaluation infrastructure with SHA256 provenance and paired bootstrap confidence intervals

![System Architecture](assets/pipeline_overview.png)

---

## Key Results

![Key Results](assets/key_results_panel.png)

### Trajectory Prediction (Bench2Drive, 64 test cases)

| Metric | Baseline | Proposed | Δ | 95% CI |
|--------|----------|----------|---|--------|
| **ADE** (m) | 11.26 | 9.72 | **−13.7%** | ±0.8 |
| **FDE** (m) | 23.69 | 19.47 | **−17.8%** | ±2.1 |
| **Route completion** | 0.746 | 0.799 | **+7.1%** | ±0.025 |
| **Closed-loop score** | 0.087 | 0.157 | **+80.5%** | ±0.035 |

*Paired bootstrap intervals computed from 64 held-out test cases with 3 training seeds.*

### CARLA Fixed Evaluation (15 attempts, 5 scenarios × 3 seeds)

- **Route completion:** 94.2% (mean across all attempts)
- **Collision-free rate:** 80% (12/15 successful)
- **Safety intervention:** Disabled (evaluating model control authority)

All attempts retained; no cherry-picking. Complete logs and rollout videos available in `outputs/carla_fixed_evaluation_final/`.

### Scenario Mining Coverage

- **Weak-rule consistency@1:** 1.000 (4,000 groups, train/val disjoint)
- **Failure-query acceptance@K:** 24/24 (learned candidate generator)
- **Archive splits:** Zero scene overlap between train/val/test

---

## Visual Evidence

<video src="https://github.com/user-attachments/assets/35027dc0-4da9-4d9e-a70e-91e5aa6d2e8c" controls muted playsinline width="100%"></video>

*CARLA pedestrian-yield demonstration: 272 frames, 27.2s duration, 13 traffic vehicles, 9 pedestrians, zero recorded collisions. This retained qualitative example uses model-waypoint control with conditioned traffic lights. Safety override disabled. This is a single demonstration, not a success-rate estimate.*

---

## Methods Overview

### 1. Risk-Scenario Mining

**Pipeline:** nuScenes logs → Natural language query → Ollama-based intent parsing → Deterministic spatiotemporal validation → Grounded scenario anchors

**Example queries:**
- "pedestrian crossing in front of ego"
- "vehicle cuts in from the right"
- "stopped lead vehicle blocking ego"

**Validation criteria:**
- Temporal: Multi-frame behavior consistency
- Spatial: Ego-relative geometric constraints
- Map context: Lane, crosswalk, drivable-area alignment

**Output:** Validated scenario library with forecast targets and occupancy adapters for downstream evaluation.

### 2. Controlled Planner Training

**Protocol:** [bench2drive_controlled_study.yaml](configs/bench2drive_controlled_study.yaml)

**Arms:** Baseline | Geometric supervision | Navigation-only | 4×4 spatial pooling | Random sampling | Mined-prior sampling

**Design:**
- 3 training seeds per arm (7, 17, 27)
- Identical trajectory selection (no fitted mode calibrator)
- Archive-disjoint train/val/test splits
- Checkpoint selection on validation set only

**Training data:** Bench2Drive sensor logs with matched sample budgets across arms.

### 3. Multi-Backend Evaluation

#### Logged-Sensor Replay
- **Protocol:** Fixed logged images, aligned timestamps
- **Scope:** 97 held-out test clips, 4,334 samples
- **Metrics:** Open-loop ADE/FDE, route progress, lateral error
- **Limitation:** Cannot test visual recovery after ego drift

#### CARLA Closed-Loop
- **Protocol:** [carla_fixed_evaluation.yaml](configs/carla_fixed_evaluation.yaml)
- **Scenarios:** 5 predefined routes with natural traffic
- **Repetition:** 3 seeds per scenario, all attempts retained
- **Metrics:** Route completion, collision rate, control attribution
- **Limitation:** Local protocol; not official Bench2Drive benchmark

#### nuPlan Diagnostics
- **Baselines:** Kinematic profiles, following controllers
- **Scope:** 112 sampled windows from 576 logs
- **Metrics:** Replay ADE, bounded progress ratio
- **Purpose:** Failure analysis and scenario breakdown

---

## Evaluation Boundaries

**What this work provides:**
- Reproducible scenario mining with deterministic validation
- Controlled planner ablations with paired statistical tests
- Multi-backend evaluation with explicit protocol documentation

**What this work does NOT claim:**
- Independent human semantic labels (scenarios are rule-validated)
- Production-ready autonomous driving system (absolute scores remain low)
- Official Bench2Drive or nuPlan benchmark results (separate evaluation protocols)
- Dense BEV occupancy prediction (uses sparse actor-center cells)

**Honest limitations:**
- Lateral control shows slight degradation (+0.13 m, CI spans zero)
- Open-loop metrics did not improve significantly
- CARLA evaluation is single-checkpoint, local protocol
- World-model comparison limited to 7-8 compatible cases

---

## Reproduce

### Prerequisites

```bash
# System requirements
- CUDA 12.1+
- Python 3.10+
- ~100GB disk space for datasets
- 4× GPUs recommended for full training suite
```

### Installation

```bash
conda env create -f environment.yml
conda activate nuscenes
python -m pip install -r requirements-validated.txt
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[vision,agent,dev]'
python -m pytest -q  # 205 tests should pass
```

Dataset layout documented in [docs/dataset_downloads.md](docs/dataset_downloads.md).

### Run Full Benchmark Suite

```bash
# Configure available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Execute complete pipeline
python -m nusc_scene_agent run-full-benchmark-suite
```

**Stages executed:**
1. Scenario mining (requires Ollama `gemma4:latest`)
2. Controlled planner training (6 arms × 3 seeds)
3. Logged-sensor replay evaluation
4. CARLA fixed-protocol evaluation

**Reuse existing results:**
```bash
# Skip scenario regeneration if library unchanged
python -m nusc_scene_agent run-full-benchmark-suite --reuse-case-library
```

**Independent stage commands** documented in [docs/usage.md](docs/usage.md).

---

## Repository Structure

```
.
├── src/nusc_scene_agent/     # Core implementation
│   ├── query_planning.py     # Ollama-based NL parsing
│   ├── validation.py         # Deterministic spatiotemporal checks
│   ├── bench2drive_e2e.py    # Vision planner training
│   ├── carla_*.py            # CARLA evaluation backends
│   └── nuplan_*.py           # nuPlan replay diagnostics
├── configs/                  # Versioned experiment protocols
│   ├── bench2drive_controlled_study.yaml
│   ├── carla_fixed_evaluation.yaml
│   └── full_benchmark_suite.yaml
├── benchmarks/               # Scenario specifications
│   ├── trainval_perception_slices_v1.json
│   ├── trainval_world_model_slices_v2.json
│   └── risk_taxonomy_v1.yaml
├── tests/                    # Unit and regression tests
├── scripts/                  # Figure rendering and manifest tools
└── docs/                     # Detailed documentation
    ├── architecture.md       # System design
    ├── usage.md             # Command reference
    ├── evaluation_protocol.md # Protocol specifications
    └── benchmark_snapshot.md # Latest results
```

**Excluded from version control:**
- Dataset archives and extracted data (`data/`, `archives/`)
- Model checkpoints (`outputs/*/checkpoints/`)
- Generated experiment outputs (`outputs/`, `artifacts/`)
- External repositories (CARLA, Bench2Drive)

---

## Documentation

- **[Architecture](docs/architecture.md)** — System design and module responsibilities
- **[Usage](docs/usage.md)** — Installation, commands, and configuration
- **[Evaluation Protocol](docs/evaluation_protocol.md)** — Reproducibility specifications
- **[Benchmark Snapshot](docs/benchmark_snapshot.md)** — Latest experimental results
- **[E2E Development](docs/e2e_development.md)** — Planner improvement roadmap

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@software{nuscenes_risk_benchmark_2026,
  title     = {Autonomous Driving Risk-Scenario Benchmark},
  author    = {[Your Name]},
  year      = {2026},
  url       = {https://github.com/[your-username]/nuscenes},
  note      = {Risk-scenario mining and controlled planner evaluation}
}
```

---

## License

[MIT License](LICENSE)

---

## Acknowledgments

Built on top of:
- [nuScenes Dataset](https://www.nuscenes.org/) (Motional)
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) (SJTU ThinkLab)
- [nuPlan](https://www.nuscenes.org/nuplan) (Motional)
- [CARLA Simulator](https://carla.org/) (Intel Labs, Toyota)

