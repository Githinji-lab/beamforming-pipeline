from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go


USER_COLORS = ["crimson", "royalblue", "darkorange", "purple"]
ML_BEAM_COLORS = ["#ff8c00", "#ffb347", "#ffcc80", "#ffd699"]
BASELINE_BEAM_COLORS = ["#1b9e77", "#5abf9a", "#8dd3b7", "#bfe7da"]


def _normalize_positions(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim == 2 and positions.shape[1] == 3:
        return positions[:, None, :]
    if positions.ndim == 3 and positions.shape[2] == 3:
        return positions
    raise ValueError("positions must have shape (steps, 3) or (steps, num_users, 3)")


def _dominant_tx_vector(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W)
    if W.ndim == 1:
        vector = W
    else:
        vector = np.mean(W, axis=1)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.ones(vector.shape[0], dtype=np.complex128) / np.sqrt(vector.shape[0])
    return vector / norm


def derive_beam_direction(W: np.ndarray) -> np.ndarray:
    w = _dominant_tx_vector(W)
    if len(w) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    phase_step = np.angle(np.vdot(w[:-1], w[1:]))
    sin_az = np.clip(phase_step / np.pi, -1.0, 1.0)
    az = float(np.arcsin(sin_az))

    phase_var = float(np.var(np.unwrap(np.angle(w))))
    el = float(np.clip(0.15 + 0.25 * np.tanh(phase_var), -0.4, 0.6))

    vec = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )
    return vec / (np.linalg.norm(vec) + 1e-12)


def derive_beam_directions(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W)
    if W.ndim == 1:
        return np.array([derive_beam_direction(W)], dtype=np.float64)
    return np.array([derive_beam_direction(W[:, idx]) for idx in range(W.shape[1])], dtype=np.float64)


def _build_beam_cloud_trace(beam_dirs: np.ndarray, scale: float) -> go.Scatter3d:
    x, y, z = [0.0], [0.0], [0.0]
    for direction in beam_dirs:
        endpoint = scale * direction
        x.extend([endpoint[0], None])
        y.extend([endpoint[1], None])
        z.extend([endpoint[2], None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="rgba(70,130,180,0.25)", width=2),
        name="All Beams",
        hoverinfo="skip",
    )


def _beam_traces(directions: np.ndarray, scale: float, colors: Sequence[str], name_prefix: str, width: int) -> List[go.Scatter3d]:
    traces: List[go.Scatter3d] = []
    for idx, direction in enumerate(directions):
        endpoint = scale * direction
        traces.append(
            go.Scatter3d(
                x=[0.0, endpoint[0]],
                y=[0.0, endpoint[1]],
                z=[0.0, endpoint[2]],
                mode="lines",
                line=dict(color=colors[idx % len(colors)], width=width),
                name=f"{name_prefix} {idx + 1}",
            )
        )
    return traces


def _user_trace(positions_frame: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=[float(pos[0]) for pos in positions_frame],
        y=[float(pos[1]) for pos in positions_frame],
        z=[float(pos[2]) for pos in positions_frame],
        mode="markers+text",
        text=[f"U{idx + 1}" for idx in range(len(positions_frame))],
        textposition="top center",
        marker=dict(size=7, color=USER_COLORS[: len(positions_frame)]),
        name="Users",
    )


def _user_paths_trace(all_positions: np.ndarray) -> go.Scatter3d:
    x, y, z = [], [], []
    for user_idx in range(all_positions.shape[1]):
        for position in all_positions[:, user_idx, :]:
            x.append(float(position[0]))
            y.append(float(position[1]))
            z.append(float(position[2]))
        x.append(None)
        y.append(None)
        z.append(None)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="rgba(220,20,60,0.18)", width=2, dash="dot"),
        name="User Paths",
        hoverinfo="skip",
    )


def _base_station_trace() -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers",
        marker=dict(size=10, color="black", symbol="square"),
        name="Base Station",
    )


def build_realtime_figure(
    positions: np.ndarray,
    codebook,
    selected_indices: Sequence[int],
    latency_ms: Sequence[float],
    sinr_db: Sequence[float],
    strength: Sequence[float],
    baseline_indices: Optional[Sequence[int]] = None,
    title: str = "3D Real-Time ML Beamforming Simulation",
) -> go.Figure:
    all_positions = _normalize_positions(positions)
    num_users = all_positions.shape[1]

    num_beams = codebook.get_num_beams()
    beam_dirs = np.array([derive_beam_direction(codebook.get_beam(i)) for i in range(num_beams)], dtype=np.float64)
    radial_extent = float(np.max(np.linalg.norm(all_positions.reshape(-1, 3), axis=1))) * 1.2
    beam_scale = max(45.0, radial_extent * 0.55)

    ml_initial_dirs = derive_beam_directions(codebook.get_beam(int(selected_indices[0])))[:num_users]
    traces: List[go.BaseTraceType] = [
        _base_station_trace(),
        _user_paths_trace(all_positions),
        _user_trace(all_positions[0]),
        _build_beam_cloud_trace(beam_dirs, beam_scale),
        *_beam_traces(ml_initial_dirs, beam_scale, ML_BEAM_COLORS, "ML Beam", 7),
    ]

    if baseline_indices is not None:
        baseline_initial_dirs = derive_beam_directions(codebook.get_beam(int(baseline_indices[0])))[:num_users]
        traces.extend(_beam_traces(baseline_initial_dirs, beam_scale, BASELINE_BEAM_COLORS, "Baseline Beam", 4))

    frames: List[go.Frame] = []
    for t in range(all_positions.shape[0]):
        ml_dirs = derive_beam_directions(codebook.get_beam(int(selected_indices[t])))[:num_users]
        frame_data: List[go.BaseTraceType] = [
            _base_station_trace(),
            _user_paths_trace(all_positions),
            _user_trace(all_positions[t]),
            _build_beam_cloud_trace(beam_dirs, beam_scale),
            *_beam_traces(ml_dirs, beam_scale, ML_BEAM_COLORS, "ML Beam", 7),
        ]

        if baseline_indices is not None:
            baseline_dirs = derive_beam_directions(codebook.get_beam(int(baseline_indices[t])))[:num_users]
            frame_data.extend(_beam_traces(baseline_dirs, beam_scale, BASELINE_BEAM_COLORS, "Baseline Beam", 4))

        metrics_text = (
            f"step={t} | users={num_users} | beam={int(selected_indices[t])}"
            f" | latency={latency_ms[t]:.3f} ms | SINR={sinr_db[t]:.2f} dB | strength={strength[t]:.4f}"
        )
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(t),
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=metrics_text,
                            x=0.01,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=13, color="black"),
                            bgcolor="rgba(255,255,255,0.8)",
                        )
                    ]
                ),
            )
        )

    sliders = [
        {
            "steps": [
                {
                    "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": str(t),
                    "method": "animate",
                }
                for t in range(all_positions.shape[0])
            ],
            "currentvalue": {"prefix": "Timestep: "},
        }
    ]

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=1100,
        height=760,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 55, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "x": 0.0,
                "y": 1.08,
            }
        ],
        sliders=sliders,
        annotations=[
            dict(
                text=(
                    f"step=0 | users={num_users} | beam={int(selected_indices[0])}"
                    f" | latency={float(latency_ms[0]):.3f} ms"
                    f" | SINR={float(sinr_db[0]):.2f} dB | strength={float(strength[0]):.4f}"
                ),
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
            )
        ],
    )
    return fig


def save_realtime_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def enable_keyboard_controls_in_html(output_path: str, total_frames: int) -> None:
    if total_frames <= 0:
        return

    keyboard_script = f"""
<script>
(function() {{
  const totalFrames = {int(total_frames)};
  let frameIndex = 0;
  let isPlaying = false;
  let playTimer = null;
  let frameDelayMs = 60;
  const minDelayMs = 20;
  const maxDelayMs = 300;

  function getPlotDiv() {{
    return document.querySelector('.plotly-graph-div');
  }}

  function gotoFrame(nextIndex) {{
    const gd = getPlotDiv();
    if (!gd) return;
    frameIndex = Math.max(0, Math.min(totalFrames - 1, nextIndex));
    Plotly.animate(
      gd,
      [String(frameIndex)],
      {{ frame: {{ duration: 0, redraw: true }}, mode: 'immediate', transition: {{ duration: 0 }} }}
    );
  }}

  function stopPlay() {{
    isPlaying = false;
    if (playTimer) {{
      clearInterval(playTimer);
      playTimer = null;
    }}
  }}

  function clampFrameDelay() {{
    frameDelayMs = Math.max(minDelayMs, Math.min(maxDelayMs, frameDelayMs));
  }}

  function startPlay() {{
    stopPlay();
    clampFrameDelay();
    isPlaying = true;
    playTimer = setInterval(function() {{
      if (frameIndex >= totalFrames - 1) {{
        stopPlay();
        return;
      }}
      gotoFrame(frameIndex + 1);
    }}, frameDelayMs);
  }}

  function restartIfPlaying() {{
    if (isPlaying) {{
      startPlay();
    }}
  }}

  document.addEventListener('keydown', function(event) {{
    if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
    if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') {{
      event.preventDefault();
      stopPlay();
      gotoFrame(frameIndex + 1);
      return;
    }}
    if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') {{
      event.preventDefault();
      stopPlay();
      gotoFrame(frameIndex - 1);
      return;
    }}
    if (event.key === 'ArrowUp' || event.key === 'w' || event.key === 'W') {{
      event.preventDefault();
      frameDelayMs = Math.max(minDelayMs, Math.round(frameDelayMs * 0.8));
      restartIfPlaying();
      return;
    }}
    if (event.key === 'ArrowDown' || event.key === 's' || event.key === 'S') {{
      event.preventDefault();
      frameDelayMs = Math.min(maxDelayMs, Math.round(frameDelayMs * 1.25));
      restartIfPlaying();
      return;
    }}
    if (event.key === 'Home') {{
      event.preventDefault();
      stopPlay();
      gotoFrame(0);
      return;
    }}
    if (event.key === 'End') {{
      event.preventDefault();
      stopPlay();
      gotoFrame(totalFrames - 1);
      return;
    }}
    if (event.key === ' ' || event.code === 'Space') {{
      event.preventDefault();
      if (isPlaying) {{
        stopPlay();
      }} else {{
        startPlay();
      }}
      return;
    }}
  }});
}})();
</script>
"""

    with open(output_path, "r", encoding="utf-8") as f:
        html = f.read()

    if "</body>" in html:
        html = html.replace("</body>", keyboard_script + "\n</body>")
    else:
        html += keyboard_script

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

from typing import List, Optional, Sequence

import numpy as np
import plotly.graph_objects as go


def _dominant_tx_vector(W: np.ndarray) -> np.ndarray:
    vector = np.mean(W, axis=1)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.ones(W.shape[0], dtype=np.complex128) / np.sqrt(W.shape[0])
    return vector / norm


def derive_beam_direction(W: np.ndarray) -> np.ndarray:
    w = _dominant_tx_vector(W)
    if len(w) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    phase_step = np.angle(np.vdot(w[:-1], w[1:]))
    sin_az = np.clip(phase_step / np.pi, -1.0, 1.0)
    az = float(np.arcsin(sin_az))

    phase_var = float(np.var(np.unwrap(np.angle(w))))
    el = float(np.clip(0.15 + 0.25 * np.tanh(phase_var), -0.4, 0.6))

    vec = np.array(
        [
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ],
        dtype=np.float64,
    )
    return vec / (np.linalg.norm(vec) + 1e-12)


def _build_beam_cloud_trace(beam_dirs: np.ndarray, scale: float) -> go.Scatter3d:
    x, y, z = [0.0], [0.0], [0.0]
    for direction in beam_dirs:
        endpoint = scale * direction
        x.extend([endpoint[0], None])
        y.extend([endpoint[1], None])
        z.extend([endpoint[2], None])

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(color="rgba(70,130,180,0.25)", width=2),
        name="All Beams",
        hoverinfo="skip",
    )


def _selected_beam_trace(direction: np.ndarray, scale: float, color: str, name: str, width: int) -> go.Scatter3d:
    endpoint = scale * direction
    return go.Scatter3d(
        x=[0.0, endpoint[0]],
        y=[0.0, endpoint[1]],
        z=[0.0, endpoint[2]],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
    )


def _user_trace(position: np.ndarray) -> go.Scatter3d:
    position = np.asarray(position)
    if position.ndim == 1:
        position = position.reshape(1, 3)

    x_vals = position[:, 0].astype(float).tolist()
    y_vals = position[:, 1].astype(float).tolist()
    z_vals = position[:, 2].astype(float).tolist()

    return go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="markers",
        marker=dict(size=7, color="crimson", symbol="circle", opacity=0.9),
        name="Users",
    )


def _base_station_trace() -> go.Scatter3d:
    return go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers",
        marker=dict(size=10, color="black", symbol="square"),
        name="Base Station",
    )


def build_realtime_figure(
    positions: np.ndarray,
    codebook,
    selected_indices: Sequence[int],
    latency_ms: Sequence[float],
    sinr_db: Sequence[float],
    strength: Sequence[float],
    baseline_indices: Optional[Sequence[int]] = None,
    title: str = "3D Real-Time ML Beamforming Simulation",
) -> go.Figure:
    positions = np.asarray(positions)
    if positions.ndim == 2:
        positions = positions[:, np.newaxis, :]
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (T, 3) or (T, U, 3)")

    num_beams = codebook.get_num_beams()
    beam_dirs = np.array([derive_beam_direction(codebook.get_beam(i)) for i in range(num_beams)], dtype=np.float64)
    radial_extent = float(np.max(np.linalg.norm(positions.reshape(-1, 3), axis=1))) * 1.2
    beam_scale = max(45.0, radial_extent * 0.55)

    initial_idx = int(selected_indices[0])
    selected_trace = _selected_beam_trace(beam_dirs[initial_idx], beam_scale, "orange", "ML Selected Beam", 8)
    baseline_trace = None
    if baseline_indices is not None:
        initial_baseline_idx = int(baseline_indices[0])
        baseline_trace = _selected_beam_trace(
            beam_dirs[initial_baseline_idx],
            beam_scale,
            "green",
            "Nearest-Angle Beam",
            5,
        )

    traces = [
        _base_station_trace(),
        _user_trace(positions[0]),
        _build_beam_cloud_trace(beam_dirs, beam_scale),
        selected_trace,
    ]
    if baseline_trace is not None:
        traces.append(baseline_trace)

    frames: List[go.Frame] = []
    for t in range(positions.shape[0]):
        ml_idx = int(selected_indices[t])
        ml_endpoint = beam_dirs[ml_idx] * beam_scale

        frame_data = [
            _base_station_trace(),
            _user_trace(positions[t]),
            _build_beam_cloud_trace(beam_dirs, beam_scale),
            go.Scatter3d(
                x=[0.0, ml_endpoint[0]],
                y=[0.0, ml_endpoint[1]],
                z=[0.0, ml_endpoint[2]],
                mode="lines",
                line=dict(color="orange", width=8),
                name="ML Selected Beam",
            ),
        ]

        if baseline_indices is not None:
            base_idx = int(baseline_indices[t])
            base_endpoint = beam_dirs[base_idx] * beam_scale
            frame_data.append(
                go.Scatter3d(
                    x=[0.0, base_endpoint[0]],
                    y=[0.0, base_endpoint[1]],
                    z=[0.0, base_endpoint[2]],
                    mode="lines",
                    line=dict(color="green", width=5),
                    name="Nearest-Angle Beam",
                )
            )

        metrics_text = (
            f"step={t} | beam={ml_idx} | latency={latency_ms[t]:.3f} ms"
            f" | SINR={sinr_db[t]:.2f} dB | strength={strength[t]:.4f}"
        )

        frames.append(
            go.Frame(
                data=frame_data,
                name=str(t),
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=metrics_text,
                            x=0.01,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            font=dict(size=13, color="black"),
                            bgcolor="rgba(255,255,255,0.8)",
                        )
                    ]
                ),
            )
        )

    sliders = [
        {
            "steps": [
                {
                    "args": [[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": str(t),
                    "method": "animate",
                }
                for t in range(len(positions))
            ],
            "currentvalue": {"prefix": "Timestep: "},
        }
    ]

    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=1100,
        height=760,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 55, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
                "x": 0.0,
                "y": 1.08,
            }
        ],
        sliders=sliders,
        annotations=[
            dict(
                text=(
                    f"step=0 | beam={int(selected_indices[0])} | latency={float(latency_ms[0]):.3f} ms"
                    f" | SINR={float(sinr_db[0]):.2f} dB | strength={float(strength[0]):.4f}"
                ),
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
            )
        ],
    )
    return fig


def save_realtime_html(fig: go.Figure, output_path: str) -> None:
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def enable_keyboard_controls_in_html(output_path: str, total_frames: int) -> None:
        if total_frames <= 0:
                return

        keyboard_script = f"""
<script>
(function() {{
    const totalFrames = {int(total_frames)};
    let frameIndex = 0;
    let isPlaying = false;
    let playTimer = null;
    let frameDelayMs = 60;
    const minDelayMs = 20;
    const maxDelayMs = 300;

    function getPlotDiv() {{
        return document.querySelector('.plotly-graph-div');
    }}

    function gotoFrame(nextIndex) {{
        const gd = getPlotDiv();
        if (!gd) return;
        frameIndex = Math.max(0, Math.min(totalFrames - 1, nextIndex));
        Plotly.animate(
            gd,
            [String(frameIndex)],
            {{ frame: {{ duration: 0, redraw: true }}, mode: 'immediate', transition: {{ duration: 0 }} }}
        );
    }}

    function stopPlay() {{
        isPlaying = false;
        if (playTimer) {{
            clearInterval(playTimer);
            playTimer = null;
        }}
    }}

    function clampFrameDelay() {{
        frameDelayMs = Math.max(minDelayMs, Math.min(maxDelayMs, frameDelayMs));
    }}

    function startPlay() {{
        stopPlay();
        clampFrameDelay();
        isPlaying = true;
        playTimer = setInterval(function() {{
            if (frameIndex >= totalFrames - 1) {{
                stopPlay();
                return;
            }}
            gotoFrame(frameIndex + 1);
        }}, frameDelayMs);
    }}

    function restartIfPlaying() {{
        if (isPlaying) {{
            startPlay();
        }}
    }}

    document.addEventListener('keydown', function(event) {{
        if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
        if (event.key === 'ArrowRight' || event.key === 'd' || event.key === 'D') {{
            event.preventDefault();
            stopPlay();
            gotoFrame(frameIndex + 1);
            return;
        }}
        if (event.key === 'ArrowLeft' || event.key === 'a' || event.key === 'A') {{
            event.preventDefault();
            stopPlay();
            gotoFrame(frameIndex - 1);
            return;
        }}
        if (event.key === 'ArrowUp' || event.key === 'w' || event.key === 'W') {{
            event.preventDefault();
            frameDelayMs = Math.max(minDelayMs, Math.round(frameDelayMs * 0.8));
            restartIfPlaying();
            return;
        }}
        if (event.key === 'ArrowDown' || event.key === 's' || event.key === 'S') {{
            event.preventDefault();
            frameDelayMs = Math.min(maxDelayMs, Math.round(frameDelayMs * 1.25));
            restartIfPlaying();
            return;
        }}
        if (event.key === 'Home') {{
            event.preventDefault();
            stopPlay();
            gotoFrame(0);
            return;
        }}
        if (event.key === 'End') {{
            event.preventDefault();
            stopPlay();
            gotoFrame(totalFrames - 1);
            return;
        }}
        if (event.key === ' ' || event.code === 'Space') {{
            event.preventDefault();
            if (isPlaying) {{
                stopPlay();
            }} else {{
                startPlay();
            }}
            return;
        }}
    }});
}})();
</script>
"""

        with open(output_path, "r", encoding="utf-8") as f:
                html = f.read()

        if "</body>" in html:
                html = html.replace("</body>", keyboard_script + "\n</body>")
        else:
                html += keyboard_script

        with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)