# Realtime 3D Beamforming Demo

This module runs an interactive 3D Plotly simulation using your existing trained DQN beam model and artifacts.

## What it uses from your project

- `results/dqn_beam_model.keras`
- `results/dqn_beam_artifacts.pkl`
  - `state_encoder`
  - `state_scaler`
  - `codebook`
  - optional `phase1_augmenter`

The preprocessing flow mirrors your benchmark pipeline:

1. `state_encoder.encode(H, snr)`
2. optional `phase1_augmenter.transform(...)`
3. `state_scaler.transform(...)`
4. model inference -> beam index selection

## Run

```bash
python realtime_demo/main.py --compare-baseline
```

Recommended (benchmark-consistent, low-latency):

```bash
python realtime_demo/main.py \
  --inference-backend tflite \
  --channel-mode benchmark \
  --topk 1 \
  --compare-baseline \
  --enable-keyboard-controls
```

Adaptive multi-user mobility mode:

```bash
python realtime_demo/main.py \
  --inference-backend tflite \
  --channel-mode trajectory \
  --num-users 4 \
  --compare-baseline \
  --enable-keyboard-controls
```

Default CLI now uses defense-consistent settings:

- `--inference-backend tflite`
- `--channel-mode benchmark`
- `--topk 1`
- `--rerank-mode q_only`

Output HTML (default):

- `results/defense/realtime_beamforming_demo.html`

## Keyboard controls (optional)

Enable keyboard controls in the exported HTML:

```bash
python realtime_demo/main.py --compare-baseline --enable-keyboard-controls
```

Inside the opened HTML:

- `Right Arrow` / `D`: next frame
- `Left Arrow` / `A`: previous frame
- `Up Arrow` / `W`: speed up playback
- `Down Arrow` / `S`: slow down playback
- `Space`: play/pause
- `Home`: first frame
- `End`: last frame

## Useful options

```bash
python realtime_demo/main.py \
  --steps 220 \
  --inference-backend tflite \
  --channel-mode benchmark \
  --topk 2 \
  --rerank-mode hybrid \
  --hybrid-q-weight 0.65 \
  --compare-baseline \
  --enable-keyboard-controls \
  --output-html results/defense/realtime_demo_hybrid.html
```

## Streamlit deployment

Run local Streamlit app:

```bash
streamlit run realtime_demo/streamlit_app.py
```

The sidebar lets you choose backend, channel mode, reranking, and trajectory settings.
It also includes a **Moving users** slider for multi-user mobility in real time.
The dashboard includes a **Per-user SINR over time** chart and a **Worst-user SINR** metric for fairness.
For the defense profile, keep:

- backend = `tflite`
- channel mode = `benchmark`
- `topk=1`, rerank=`q_only`
