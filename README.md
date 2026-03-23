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

### Evaluate Trained Agent
```bash
python pipeline/evaluate.py
```

### Visualize Results
```bash
python pipeline/visualize.py
```

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