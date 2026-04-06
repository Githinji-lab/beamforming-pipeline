import json
import os

import numpy as np
from scipy.io import loadmat


CHANNEL_KEYS = ("Hvirtual", "Harray", "H", "channel", "channels")


def _normalize_channel_matrix(matrix):
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    return matrix / row_norms


def _adapt_channel_shape(matrix, target_k, target_n_tx):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D channel matrix, got shape={matrix.shape}")

    matrix = matrix.astype(np.complex64, copy=False)
    k_in, n_tx_in = matrix.shape
    out = np.zeros((target_k, target_n_tx), dtype=np.complex64)
    min_k = min(k_in, target_k)
    min_n_tx = min(n_tx_in, target_n_tx)
    out[:min_k, :min_n_tx] = matrix[:min_k, :min_n_tx]
    return _normalize_channel_matrix(out)


def _extract_channels_from_array(array, target_k, target_n_tx, max_samples_per_file=None):
    arr = np.asarray(array)
    channels = []

    if arr.ndim == 3:
        count = arr.shape[0]
        if max_samples_per_file is not None and count > max_samples_per_file:
            indices = np.linspace(0, count - 1, num=max_samples_per_file, dtype=int)
        else:
            indices = np.arange(count)

        for idx in indices:
            channels.append(_adapt_channel_shape(arr[idx], target_k=target_k, target_n_tx=target_n_tx))
    elif arr.ndim == 2:
        channels.append(_adapt_channel_shape(arr, target_k=target_k, target_n_tx=target_n_tx))

    return channels


def _load_from_mat(file_path, target_k, target_n_tx, max_samples_per_file=None):
    loaded = loadmat(file_path)
    channels = []
    for key in CHANNEL_KEYS:
        if key in loaded:
            channels.extend(
                _extract_channels_from_array(
                    loaded[key],
                    target_k=target_k,
                    target_n_tx=target_n_tx,
                    max_samples_per_file=max_samples_per_file,
                )
            )
            if channels:
                break
    return channels


def _load_from_npz(file_path, target_k, target_n_tx, max_samples_per_file=None):
    loaded = np.load(file_path, allow_pickle=True)
    channels = []
    for key in CHANNEL_KEYS:
        if key in loaded:
            channels.extend(
                _extract_channels_from_array(
                    loaded[key],
                    target_k=target_k,
                    target_n_tx=target_n_tx,
                    max_samples_per_file=max_samples_per_file,
                )
            )
            if channels:
                break
    return channels


def load_channels_from_registry(
    registry_path,
    target_k,
    target_n_tx,
    max_total_samples=20000,
    max_samples_per_file=4000,
):
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")

    with open(registry_path, "r") as f:
        registry = json.load(f)

    dataset_files = registry.get("dataset_files", [])
    all_channels = []

    for file_path in dataset_files:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".mat":
                channels = _load_from_mat(
                    file_path,
                    target_k=target_k,
                    target_n_tx=target_n_tx,
                    max_samples_per_file=max_samples_per_file,
                )
            elif ext == ".npz":
                channels = _load_from_npz(
                    file_path,
                    target_k=target_k,
                    target_n_tx=target_n_tx,
                    max_samples_per_file=max_samples_per_file,
                )
            else:
                channels = []

            all_channels.extend(channels)
            if len(all_channels) >= max_total_samples:
                break
        except Exception:
            continue

    if len(all_channels) == 0:
        raise ValueError(
            "No channel matrices found in registry dataset files. "
            "Expected keys like Hvirtual/Harray in .mat or .npz files."
        )

    if len(all_channels) > max_total_samples:
        all_channels = all_channels[:max_total_samples]

    return np.array(all_channels, dtype=np.complex64)


class ExternalChannelSampler:
    def __init__(self, channels, seed=42):
        self.channels = np.asarray(channels)
        if self.channels.ndim != 3:
            raise ValueError(f"Expected channels shape [N, K, N_tx], got {self.channels.shape}")
        self.rng = np.random.default_rng(seed)

    def sample(self):
        idx = int(self.rng.integers(0, len(self.channels)))
        return self.channels[idx]
