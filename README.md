# Beamforming ML Pipeline

This project implements a machine learning pipeline for optimizing beamforming in 5G wireless communication systems using reinforcement learning (SAC - Soft Actor-Critic).

## Project Structure

```
beamforming-project-pipeline/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── data/                   # Generated training data
├── results/                # Training results and models
├── src/                    # Source code
│   ├── simulators.py       # Beamforming simulators (V4, V5)
│   ├── agents.py           # RL agents (SAC, etc.)
│   ├── utils.py            # Utilities (ReplayBuffer, etc.)
│   └── preprocessing.py    # Data preprocessing functions
└── pipeline/               # Pipeline scripts
    ├── data_gen.py         # Data generation
    ├── train.py            # Training script
    ├── evaluate.py         # Evaluation script
    └── visualize.py        # Visualization tools
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script to see available commands:
```bash
python main.py
```

## Usage

### Generate Training Data
```bash
python pipeline/data_gen.py
```

### Train SAC Agent
```bash
python pipeline/train.py
```

### Train Optimized SAC (Latency-Constrained + Imitation + Distillation)
```bash
python pipeline/train_improved.py
```

By default this uses a refined teacher-derived codebook (`teacher_top`):
- generate MMSE/SLNR teacher beams,
- keep only top-capacity beams,
- cluster with KMeans into deployable beam entries.

This run now includes:
- constrained reward with latency budget penalty,
- imitation warm-start from MMSE/SLNR teacher beams,
- lightweight student distillation and quantized TFLite export.

### Evaluate Trained Agent
```bash
python pipeline/evaluate.py
```

### Benchmark Capacity and Latency (Traditional vs Optimized RL)
```bash
python pipeline/benchmark_optimized.py
```

### Train Discrete DQN Beam Selector (SAC-compatible hybrid path)
```bash
python pipeline/train_dqn_beam.py
```

Quick smoke run:
```bash
python pipeline/train_dqn_beam.py --episodes 10 --steps 20 --batch-size 32 --imitation-samples 120 --imitation-epochs 2
```

Optional baseline ablation (random codebook):
```bash
python pipeline/train_dqn_beam.py --codebook-strategy random
```

Capacity-focused refined codebook tuning:
```bash
python pipeline/train_dqn_beam.py --codebook-strategy teacher_top --codebook-keep-ratio 0.25
```

### Visualize Results
```bash
python pipeline/visualize.py
```

## Optimized Artifacts

After running `pipeline/train_improved.py`, these artifacts are generated in `results/`:
- `improved_sac_actor_model.keras`
- `improved_sac_actor_model.h5`
- `improved_sac_student.keras`
- `improved_sac_student_int8.tflite`
- `improved_sac_rewards.npy`
- `improved_sac_latency_ms.npy`
- `beam_codebook.pkl`

After running `pipeline/train_dqn_beam.py`, these artifacts are generated in `results/`:
- `dqn_beam_model.keras`
- `dqn_beam_model_int8.tflite`
- `dqn_beam_artifacts.pkl`
- `dqn_beam_rewards.npy`
- `dqn_beam_latency_p95_ms.npy`
- `dqn_beam_training.png`

## Key Components

- **BeamformingSimulatorV4**: Realistic 5G channel simulation with 3GPP path loss models
- **SACAgent**: Soft Actor-Critic agent for beamforming optimization
- **Preprocessing**: Channel state and action preprocessing for RL
- **Evaluation**: Performance comparison against traditional methods (MMSE)

## Features

- Realistic 3GPP-inspired channel modeling
- Reinforcement learning for adaptive beamforming
- Reward shaping for better convergence
- Comprehensive evaluation and visualization
- Modular, organized codebase

## Results

The SAC agent learns to optimize beamforming weights to maximize sum capacity in multi-user MIMO systems, outperforming traditional methods in dynamic environments.