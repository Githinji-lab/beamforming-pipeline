# Beamforming ML Pipeline

Adaptive beam selection pipeline for 5G-like MIMO channels with a focus on **sum capacity** and **low-latency inference**.

## Project Objectives

This project targets:
- **Lightweight AI approach** for practical deployment
- **Real-time adaptive beam selection**
- **Clustering + RL** architecture
- Evaluation on **SINR, BER, throughput, latency**

## Current Status vs Objectives

- ✅ **Low-latency deployable models** (`.tflite`) are working
- ✅ **Teacher-informed codebook + RL selectors** are implemented
- ✅ **Benchmark harness** compares classical and learned methods
- ⚠️ **Sum-capacity parity vs MMSE/ZF** is still under active tuning

## Repository Structure

```text
beamforming-project-pipeline/
├── data/
├── results/
├── src/
│   ├── simulators.py
│   ├── agents.py
│   ├── dqn_beam_agent.py
│   ├── baselines.py
│   ├── state_encoder.py
│   ├── preprocessing.py
│   └── domain_randomization.py
├── pipeline/
│   ├── data_gen.py
│   ├── train.py
│   ├── train_improved.py
│   ├── train_dqn_beam.py
│   ├── baseline_comparison.py
│   ├── benchmark_optimized.py
│   ├── evaluate.py
│   └── plot_baselines.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd /home/anon/Downloads/Beamforming-project-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Method Summary

### 1) Teacher-Derived Codebook (capacity-focused)
- Build beams using **MMSE/SLNR teacher solutions**
- Keep top-capacity teacher beams (`teacher_top`)
- Cluster them into deployable codebook entries

### 2) RL Selectors
- **SAC path**: latency-constrained reward + imitation warm-start + distilled student
- **DQN path**: discrete beam-index selector (better action-space match)

### 3) Deployment
- Export lightweight inference models (`.keras`, `.tflite`)
- Benchmark with latency and capacity on held-out channels

## Reproducible Workflows

### Data generation
```bash
python pipeline/data_gen.py
```

### Add external dataset zip archives
```bash
python pipeline/add_dataset_zip.py \
    --zip /path/to/dataset_a.zip \
    --zip /path/to/dataset_b.zip
```

This extracts archives under `data/external/` and writes a registry to `data/dataset_registry.json`.

### Optimized SAC training
```bash
python pipeline/train_improved.py
```

### DQN beam training (recommended)
```bash
python pipeline/train_dqn_beam.py \
    --episodes 120 \
    --steps 50 \
    --batch-size 64 \
    --imitation-samples 500 \
    --imitation-epochs 6 \
    --codebook-strategy teacher_top \
    --codebook-keep-ratio 0.25 \
    --dataset-zips /path/to/dataset_a.zip,/path/to/dataset_b.zip
```

### Dataset-backed training (external channels)
```bash
python pipeline/train_dqn_beam.py \
    --episodes 10 \
    --steps 20 \
    --batch-size 32 \
    --imitation-samples 120 \
    --imitation-epochs 2 \
    --channel-source external \
    --external-registry data/dataset_registry.json \
    --external-max-samples 5000
```

Channel source options:
- `simulator`: synthetic channels only (default)
- `external`: channels from `data/dataset_registry.json`
- `mixed`: blend simulator and external channels using `--external-mix-ratio`

### Baseline/optimized benchmark
```bash
python pipeline/benchmark_optimized.py
```

Custom iterations and JSON export:
```bash
python pipeline/benchmark_optimized.py --iterations 200 --json-out results/benchmark_custom.json
```

Benchmark on external dataset channels:
```bash
python pipeline/benchmark_optimized.py \
    --iterations 200 \
    --channel-source external \
    --external-registry data/dataset_registry.json \
    --external-max-samples 5000 \
    --json-out results/benchmark_external.json
```

Benchmark now reports (per method):
- Capacity mean/std
- Latency mean/P95
- SINR mean (dB)
- BER mean

### Quick smoke test
```bash
python pipeline/train_dqn_beam.py \
    --episodes 4 \
    --steps 10 \
    --batch-size 16 \
    --imitation-samples 60 \
    --imitation-epochs 1 \
    --codebook-strategy teacher_top \
    --codebook-keep-ratio 0.25
```

### Teacher-top keep-ratio sweep (capacity tuning)
```bash
python pipeline/sweep_dqn_keep_ratio.py \
    --ratios 0.10,0.15,0.20,0.30 \
    --episodes 240 \
    --steps 60 \
    --batch-size 128 \
    --imitation-samples 1200 \
    --imitation-epochs 8 \
    --benchmark-iterations 200 \
    --dataset-zips /path/to/dataset_a.zip,/path/to/dataset_b.zip \
    --channel-source mixed \
    --external-registry data/dataset_registry.json \
    --external-mix-ratio 0.5
```

### Defense-ready clean results package
Run fixed protocol first (locked seeds/iterations/source, 3-run stats + CI, ablations):
```bash
python pipeline/run_defense_protocol.py \
    --iterations 120 \
    --seeds 11,22,33 \
    --channel-source external \
    --dqn-rerank-topk 3 \
    --out-dir results/protocol
```

Optional teacher-top ratio ablation:
```bash
python pipeline/run_defense_protocol.py \
    --iterations 120 \
    --seeds 11,22,33 \
    --channel-source external \
    --dqn-rerank-topk 3 \
    --run-teacher-ratio-ablation \
    --teacher-ratios 0.20,0.30,0.35 \
    --out-dir results/protocol
```

Then build the clean defense package:
```bash
python pipeline/prepare_defense_results.py
```

This creates a clean folder `results/defense/` with:
- `primary_benchmark_protocol.json` (locked protocol settings)
- `headline_table.csv` (3-run mean/std/95% CI)
- `ablation_table.csv` (topk, dataset on/off, teacher_top ratio)
- `objective_summary.json` (objective % completion and selected top-k)
- `objective_scores.csv`
- `objective_progress.png`
- `topk_tradeoff.png`
- `selected_method_comparison.png`
- `results_claims_clean.txt`

## Main Artifacts (in `results/`)

### SAC artifacts
- `improved_sac_actor_model.keras`
- `improved_sac_actor_model.h5`
- `improved_sac_student.keras`
- `improved_sac_student_int8.tflite`
- `improved_sac_rewards.npy`
- `improved_sac_latency_ms.npy`
- `beam_codebook.pkl`

### DQN artifacts
- `dqn_beam_model.keras`
- `dqn_beam_model_int8.tflite`
- `dqn_beam_artifacts.pkl`
- `dqn_beam_rewards.npy`
- `dqn_beam_latency_p95_ms.npy`
- `dqn_beam_training.png`

## Reporting Template (for submission)

Use this table format after each benchmark run:

| Method | Capacity Mean | Capacity Std | Latency Mean (ms) | Latency P95 (ms) |
|---|---:|---:|---:|---:|
| MMSE |  |  |  |  |
| ZF |  |  |  |  |
| SAC Teacher |  |  |  |  |
| SAC Student TFLite |  |  |  |  |
| DQN |  |  |  |  |
| DQN TFLite |  |  |  |  |

## Interpretation Guidance

- If capacity is still below MMSE, report efficiency as:
    - **“ML achieves X% of MMSE capacity with Y% lower inference latency.”**
- For fairness, compare all methods on:
    - same channel seeds,
    - same evaluation iterations,
    - same hardware/runtime context.

## Next Capacity Uplift Levers

1. Sweep `--codebook-keep-ratio` (`0.10, 0.15, 0.20, 0.30`)
2. Increase teacher sample volume for codebook generation
3. Longer DQN training with curriculum scenarios
4. Top-k beam reranking at inference time