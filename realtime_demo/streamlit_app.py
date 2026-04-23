from __future__ import annotations

import os
import time
import tempfile
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model_loader import (
    compute_user_sinr_db,
    estimate_signal_metrics,
    load_realtime_bundle,
    select_beam_index_from_qvals,
)
from simulation import (
    SimulationConfig,
    channel_from_user_positions,
    generate_multi_user_trajectories,
    user_angles_from_position,
)
from visualization import (
    build_realtime_figure,
    derive_beam_direction,
    enable_keyboard_controls_in_html,
    save_realtime_html,
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


def run_simulation(args: SimpleNamespace):
    project_root = os.path.abspath(args.project_root)
    bundle = load_realtime_bundle(
        project_root=project_root,
        model_path=args.model_path,
        tflite_model_path=args.tflite_model_path,
        artifacts_path=args.artifacts_path,
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
    sinr_db_per_user = []
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
        user_sinr_db = compute_user_sinr_db(bundle.simulator, H, W_selected)

        selected_indices.append(int(predicted_idx))
        latency_ms.append(float(dt_ms))
        sinr_db.append(float(sinr_value))
        sinr_db_per_user.append(np.asarray(user_sinr_db, dtype=np.float64))
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

    sinr_db_per_user_arr = np.stack(sinr_db_per_user, axis=0)
    effective_users = int(min(args.num_users, sinr_db_per_user_arr.shape[1]))

    metrics = {
        "backend": bundle.inference_backend,
        "channel_mode": args.channel_mode,
        "frames": len(trajectory),
        "num_users": int(args.num_users),
        "effective_users": effective_users,
        "beam_count": bundle.codebook.get_num_beams(),
        "lat_mean_ms": float(np.mean(latency_ms)),
        "lat_p95_ms": float(np.percentile(latency_ms, 95)),
        "sinr_mean_db": float(np.mean(sinr_db)),
        "sinr_worst_user_db": float(np.min(np.mean(sinr_db_per_user_arr[:, :effective_users], axis=0))),
        "strength_mean": float(np.mean(strength)),
        "baseline": bool(args.compare_baseline),
    }

    return fig, metrics, sinr_db_per_user_arr


def main() -> None:
    st.set_page_config(page_title="5G Beamforming Realtime Demo", layout="wide")
    st.title("5G Beamforming — Realtime ML Simulation (Defense)")

    project_root_default = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    with st.sidebar:
        st.header("Configuration")
        project_root = st.text_input("Project root", value=project_root_default)
        inference_backend = st.selectbox("Inference backend", ["keras", "tflite"], index=0)
        channel_mode = st.selectbox("Channel mode", ["trajectory", "benchmark"], index=0)
        steps = st.slider("Steps", min_value=30, max_value=400, value=180, step=10)
        compare_baseline = st.checkbox("Compare nearest-angle baseline", value=True)
        enable_keyboard_export = st.checkbox("Enable keyboard controls in exported HTML", value=True)

        st.subheader("Reranking")
        topk = st.slider("Top-k", min_value=1, max_value=5, value=2)
        rerank_mode = st.selectbox("Rerank mode", ["hybrid", "capacity", "q_only"], index=0)
        hybrid_q_weight = st.slider("Hybrid q-weight", min_value=0.0, max_value=1.0, value=0.65, step=0.05)

        st.subheader("Trajectory (used when channel_mode=trajectory)")
        radius = st.slider("Radius", min_value=30.0, max_value=250.0, value=120.0, step=5.0)
        angular_speed = st.slider("Angular speed", min_value=0.01, max_value=0.25, value=0.075, step=0.005)
        vertical_center = st.slider("Vertical center", min_value=0.5, max_value=5.0, value=1.7, step=0.1)
        vertical_amplitude = st.slider("Vertical amplitude", min_value=0.0, max_value=4.0, value=1.2, step=0.1)
        radial_wobble = st.slider("Radial wobble", min_value=0.0, max_value=60.0, value=18.0, step=1.0)
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
        num_users = st.slider("Moving users", min_value=1, max_value=8, value=1, step=1)

        run_clicked = st.button("Run simulation", type="primary")

    st.caption("Default view restores the original 3D demo setup; switch to `tflite + benchmark + topk=1 + q_only` only when you want the low-latency benchmark profile.")

    if not run_clicked and "last_result" not in st.session_state:
        st.info("Click **Run simulation** in the sidebar to start.")
        return

    if run_clicked:
        args = SimpleNamespace(
            project_root=project_root,
            model_path=None,
            tflite_model_path=None,
            artifacts_path=None,
            inference_backend=inference_backend,
            channel_mode=channel_mode,
            steps=int(steps),
            snr_db=None,
            radius=float(radius),
            angular_speed=float(angular_speed),
            vertical_center=float(vertical_center),
            vertical_amplitude=float(vertical_amplitude),
            radial_wobble=float(radial_wobble),
            seed=int(seed),
            num_users=int(num_users),
            topk=int(topk),
            rerank_mode=rerank_mode,
            hybrid_q_weight=float(hybrid_q_weight),
            compare_baseline=bool(compare_baseline),
            title="3D Real-Time ML Beamforming Simulation",
            enable_keyboard_controls=bool(enable_keyboard_export),
        )

        with st.spinner("Running simulation..."):
            fig, metrics, sinr_db_per_user_arr = run_simulation(args)

        st.session_state.last_result = {
            "fig": fig,
            "metrics": metrics,
            "sinr_db_per_user": sinr_db_per_user_arr,
            "enable_keyboard_export": enable_keyboard_export,
        }

    result = st.session_state.last_result
    fig = result["fig"]
    metrics = result["metrics"]
    sinr_db_per_user_arr = result["sinr_db_per_user"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean latency (ms)", f"{metrics['lat_mean_ms']:.4f}")
    col2.metric("P95 latency (ms)", f"{metrics['lat_p95_ms']:.4f}")
    col3.metric("Mean SINR (dB)", f"{metrics['sinr_mean_db']:.2f}")
    col4.metric("Worst-user SINR (dB)", f"{metrics['sinr_worst_user_db']:.2f}")
    col5.metric("Frames", str(metrics["frames"]))

    st.write(
        f"Backend: `{metrics['backend']}` | Channel mode: `{metrics['channel_mode']}` | "
        f"Users: `{metrics['num_users']}` (effective `{metrics['effective_users']}`) | "
        f"Beams: `{metrics['beam_count']}` | Baseline: `{metrics['baseline']}`"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-user SINR over time")
    sinr_fig = go.Figure()
    effective_users = int(metrics["effective_users"])
    time_steps = np.arange(sinr_db_per_user_arr.shape[0], dtype=np.int32)
    for user_idx in range(effective_users):
        sinr_fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=sinr_db_per_user_arr[:, user_idx],
                mode="lines",
                name=f"User {user_idx + 1}",
            )
        )
    sinr_fig.update_layout(
        xaxis_title="Timestep",
        yaxis_title="SINR (dB)",
        legend_title="Users",
        height=340,
    )
    st.plotly_chart(sinr_fig, use_container_width=True)

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name

    save_realtime_html(fig, tmp_path)
    if result.get("enable_keyboard_export", False):
        enable_keyboard_controls_in_html(tmp_path, total_frames=metrics["frames"])

    with open(tmp_path, "rb") as f:
        html_bytes = f.read()

    os.remove(tmp_path)

    st.download_button(
        label="Download interactive HTML",
        data=html_bytes,
        file_name="realtime_beamforming_demo_streamlit_export.html",
        mime="text/html",
    )


if __name__ == "__main__":
    main()