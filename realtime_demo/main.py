from __future__ import annotations

import argparse
import os
import time

import numpy as np

from model_loader import (
    estimate_signal_metrics,
    load_realtime_bundle,
    select_beam_index_from_qvals,
)
from simulation import (
    SimulationConfig,
    channel_from_user_position,
    channel_from_user_positions,
    generate_user_trajectory,
    generate_multi_user_trajectories,
    user_angles_from_position,
)
from visualization import (
    build_realtime_figure,
    save_realtime_html,
    derive_beam_direction,
    enable_keyboard_controls_in_html,
)


def _nearest_angle_beam_index(codebook, user_pos: np.ndarray) -> int:
    user_az, _, _ = user_angles_from_position(user_pos)
    user_vec = np.array([np.cos(user_az), np.sin(user_az), 0.0], dtype=np.float64)

    best_idx = 0
    best_score = -np.inf
    for idx in range(codebook.get_num_beams()):
        bvec = derive_beam_direction(codebook.get_beam(idx))
        score = float(np.dot(user_vec, bvec))
        if score > best_score:
            best_score = score
            best_idx = idx
    return int(best_idx)


def _nearest_angle_beam_index_from_positions(codebook, user_positions: np.ndarray) -> int:
    user_positions = np.asarray(user_positions, dtype=np.float32)
    if user_positions.ndim == 1:
        return _nearest_angle_beam_index(codebook, user_positions)
    centroid = np.mean(user_positions, axis=0)
    return _nearest_angle_beam_index(codebook, centroid)


def run_demo(args: argparse.Namespace) -> str:
    project_root = os.path.abspath(args.project_root)
    bundle = load_realtime_bundle(
        project_root=project_root,
        model_path=args.model_path,
        artifacts_path=args.artifacts_path,
        tflite_model_path=args.tflite_model_path,
        inference_backend=args.inference_backend,
    )

    trajectory = generate_multi_user_trajectories(
        SimulationConfig(
            steps=args.steps,
            radius=args.radius,
            angular_speed=args.angular_speed,
            vertical_center=args.vertical_center,
            vertical_amplitude=args.vertical_amplitude,
            radial_wobble=args.radial_wobble,
            random_seed=args.seed,
        ),
        num_users=args.num_users,
    )

    selected_indices = []
    baseline_indices = [] if args.compare_baseline else None
    latency_ms = []
    sinr_db = []
    strength = []

    prev_H = None
    snr = float(args.snr_db if args.snr_db is not None else bundle.default_snr_db)

    for step_idx, user_positions_t in enumerate(trajectory):
        if args.channel_mode == "benchmark":
            H = bundle.simulator.generate_channel_matrix_v4()
        else:
            H = channel_from_user_positions(
                user_positions=user_positions_t,
                simulator=bundle.simulator,
                random_seed=args.seed + step_idx,
            )
        state = bundle.preprocess_state(H=H, snr_db=snr, prev_H=prev_H)

        t0 = time.perf_counter()
        qvals = bundle.predict_qvalues(state)
        predicted_idx = select_beam_index_from_qvals(
            qvals=qvals,
            H=H,
            codebook=bundle.codebook,
            simulator=bundle.simulator,
            topk=args.topk,
            rerank_mode=args.rerank_mode,
            hybrid_q_weight=args.hybrid_q_weight,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        W_selected = bundle.codebook.get_beam(predicted_idx)
        sinr_value, strength_value = estimate_signal_metrics(bundle.simulator, H, W_selected)

        selected_indices.append(int(predicted_idx))
        latency_ms.append(float(dt_ms))
        sinr_db.append(float(sinr_value))
        strength.append(float(strength_value))

        if args.compare_baseline:
            baseline_indices.append(_nearest_angle_beam_index_from_positions(bundle.codebook, user_positions_t))

        prev_H = H

    fig = build_realtime_figure(
        positions=trajectory,
        codebook=bundle.codebook,
        selected_indices=selected_indices,
        latency_ms=latency_ms,
        sinr_db=sinr_db,
        strength=strength,
        baseline_indices=baseline_indices,
        title=args.title,
    )

    output_path = os.path.abspath(args.output_html)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_realtime_html(fig, output_path)
    if args.enable_keyboard_controls:
        enable_keyboard_controls_in_html(output_path=output_path, total_frames=len(trajectory))

    print("=" * 70)
    print("Realtime ML Beamforming Demo")
    print("=" * 70)
    print(f"Output HTML        : {output_path}")
    print(f"Frames             : {len(trajectory)}")
    print(f"Beam count         : {bundle.codebook.get_num_beams()}")
    print(f"Inference backend  : {bundle.inference_backend}")
    print(f"Channel mode       : {args.channel_mode}")
    print(f"Mean latency (ms)  : {np.mean(latency_ms):.4f}")
    print(f"P95 latency (ms)   : {np.percentile(latency_ms, 95):.4f}")
    print(f"Mean SINR (dB)     : {np.mean(sinr_db):.3f}")
    print(f"Mean strength      : {np.mean(strength):.5f}")
    print(f"Baseline compared  : {bool(args.compare_baseline)}")
    print("=" * 70)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D real-time beamforming simulation using trained project model")
    parser.add_argument("--project-root", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tflite-model-path", type=str, default=None)
    parser.add_argument("--artifacts-path", type=str, default=None)
    parser.add_argument("--inference-backend", type=str, default="keras", choices=["keras", "tflite"])
    parser.add_argument("--channel-mode", type=str, default="trajectory", choices=["benchmark", "trajectory"])
    parser.add_argument("--output-html", type=str, default="results/defense/realtime_beamforming_demo.html")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--snr-db", type=float, default=None)

    parser.add_argument("--radius", type=float, default=120.0)
    parser.add_argument("--angular-speed", type=float, default=0.075)
    parser.add_argument("--vertical-center", type=float, default=1.7)
    parser.add_argument("--vertical-amplitude", type=float, default=1.2)
    parser.add_argument("--radial-wobble", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-users", type=int, default=1)

    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--rerank-mode",
        type=str,
        default="hybrid",
        choices=["capacity", "hybrid", "q_only"],
    )
    parser.add_argument("--hybrid-q-weight", type=float, default=0.65)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--enable-keyboard-controls", action="store_true")
    parser.add_argument("--title", type=str, default="3D Real-Time ML Beamforming Simulation")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    run_demo(parsed_args)