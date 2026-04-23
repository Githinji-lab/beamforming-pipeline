from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SimulationConfig:
    steps: int = 160
    radius: float = 120.0
    angular_speed: float = 0.075
    vertical_center: float = 1.7
    vertical_amplitude: float = 1.2
    radial_wobble: float = 18.0
    random_seed: int = 42


def generate_user_trajectory(config: SimulationConfig) -> np.ndarray:
    return generate_multi_user_trajectories(config=config, num_users=1)[:, 0, :]


def generate_multi_user_trajectories(config: SimulationConfig, num_users: int) -> np.ndarray:
    num_users = int(max(1, num_users))
    rng = np.random.RandomState(config.random_seed)
    t = np.arange(config.steps, dtype=np.float64)

    all_positions = []
    for user_idx in range(num_users):
        phase = (2.0 * np.pi * user_idx) / max(1, num_users)
        theta = config.angular_speed * t + phase

        user_radius = config.radius + 10.0 * np.cos(phase)
        r = user_radius + config.radial_wobble * np.sin(0.15 * t + 0.5 + 0.4 * user_idx)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        z = (
            config.vertical_center
            + config.vertical_amplitude * np.sin(0.11 * t + 1.0 + 0.3 * user_idx)
            + 0.10 * rng.randn(config.steps)
        )
        z = np.clip(z, 0.8, 4.0)

        all_positions.append(np.stack([x, y, z], axis=1))

    stacked = np.stack(all_positions, axis=1)
    return stacked.astype(np.float32)


def user_angles_from_position(user_pos: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(user_pos[0]), float(user_pos[1]), float(user_pos[2])
    dist = float(np.sqrt(x * x + y * y + z * z) + 1e-9)
    az = float(np.arctan2(y, x))
    el = float(np.arcsin(np.clip(z / dist, -1.0, 1.0)))
    return az, el, dist


def _ula_response(n_tx: int, azimuth_rad: float, d_over_lambda: float = 0.5) -> np.ndarray:
    n = np.arange(n_tx, dtype=np.float64)
    phase = 2.0 * np.pi * d_over_lambda * np.sin(azimuth_rad) * n
    a = np.exp(1j * phase)
    return a / (np.linalg.norm(a) + 1e-12)


def channel_from_user_position(
    user_pos: np.ndarray,
    simulator,
    nlos_paths: int = 4,
    los_k_factor: float = 6.0,
    random_seed: int = 42,
) -> np.ndarray:
    return channel_from_user_positions(
        user_positions=np.asarray(user_pos, dtype=np.float32).reshape(1, 3),
        simulator=simulator,
        nlos_paths=nlos_paths,
        los_k_factor=los_k_factor,
        random_seed=random_seed,
    )


def _single_user_channel(
    user_pos: np.ndarray,
    simulator,
    nlos_paths: int,
    los_k_factor: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    az, _, dist = user_angles_from_position(user_pos)
    path_gain = 1.0 / max(dist, 1.0)

    az_user = az + rng.uniform(-0.08, 0.08)
    a_los = _ula_response(simulator.N_tx, az_user)

    a_nlos = np.zeros(simulator.N_tx, dtype=np.complex128)
    for _ in range(nlos_paths):
        az_n = az_user + rng.uniform(-0.8, 0.8)
        gain = (rng.randn() + 1j * rng.randn()) / np.sqrt(2.0 * max(1, nlos_paths))
        a_nlos += gain * _ula_response(simulator.N_tx, az_n)

    h = np.sqrt(los_k_factor / (1.0 + los_k_factor)) * a_los + np.sqrt(1.0 / (1.0 + los_k_factor)) * a_nlos
    h = h / (np.linalg.norm(h) + 1e-12)
    return path_gain * h


def channel_from_user_positions(
    user_positions: np.ndarray,
    simulator,
    nlos_paths: int = 4,
    los_k_factor: float = 6.0,
    random_seed: int = 42,
) -> np.ndarray:
    user_positions = np.asarray(user_positions, dtype=np.float32)
    if user_positions.ndim != 2 or user_positions.shape[1] != 3:
        raise ValueError("user_positions must have shape (num_users, 3)")

    num_input_users = int(user_positions.shape[0])
    target_k = int(simulator.K)
    H = np.zeros((target_k, simulator.N_tx), dtype=np.complex128)

    centroid = np.mean(user_positions, axis=0)
    for k in range(target_k):
        if k < num_input_users:
            pos = user_positions[k]
        else:
            jitter = np.array([2.0 * (k + 1), -1.5 * (k + 1), 0.0], dtype=np.float32)
            pos = centroid + jitter

        seed_offset = int(abs(pos[0] * 10) + abs(pos[1] * 10) + 97 * k)
        rng = np.random.RandomState((random_seed + seed_offset) % (2**32 - 1))
        H[k, :] = _single_user_channel(
            user_pos=pos,
            simulator=simulator,
            nlos_paths=nlos_paths,
            los_k_factor=los_k_factor,
            rng=rng,
        )

    return H