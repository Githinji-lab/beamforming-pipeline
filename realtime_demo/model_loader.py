from __future__ import annotations

import os
import sys
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def _ensure_src_on_path(project_root: str) -> str:
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    return src_path


@dataclass
class RealtimeBeamModel:
    project_root: str
    model: Optional[tf.keras.Model]
    tflite_interpreter: Optional[tf.lite.Interpreter]
    inference_backend: str
    codebook: object
    state_encoder: object
    state_scaler: object
    phase1_augmenter: Optional[object]
    simulator: object
    default_snr_db: float

    def preprocess_state(self, H: np.ndarray, snr_db: float, prev_H: Optional[np.ndarray] = None) -> np.ndarray:
        base_state = self.state_encoder.encode(H, snr_db)
        if self.phase1_augmenter is not None:
            base_state = self.phase1_augmenter.transform(
                base_state=base_state,
                H=H,
                snr=snr_db,
                prev_H=prev_H,
            )
        state = self.state_scaler.transform(np.asarray(base_state, dtype=np.float32).reshape(1, -1))
        return state.astype(np.float32)

    def predict_qvalues(self, state: np.ndarray) -> np.ndarray:
        if self.inference_backend == "tflite":
            if self.tflite_interpreter is None:
                raise RuntimeError("TFLite backend selected but interpreter is not loaded.")
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            self.tflite_interpreter.set_tensor(input_details[0]["index"], state.astype(np.float32))
            self.tflite_interpreter.invoke()
            qvals = self.tflite_interpreter.get_tensor(output_details[0]["index"])[0]
            return np.asarray(qvals, dtype=np.float32)

        if self.model is None:
            raise RuntimeError("Keras backend selected but model is not loaded.")
        qvals = self.model(state, training=False).numpy()[0]
        return np.asarray(qvals, dtype=np.float32)


def _proxy_capacity_score(simulator, H: np.ndarray, W: np.ndarray) -> float:
    capacities = []
    K = H.shape[0]
    for k in range(K):
        sig = simulator.P_tx_linear * np.abs(H[k, :] @ W[:, k]) ** 2
        intf = sum(
            simulator.P_tx_linear * np.abs(H[k, :] @ W[:, j]) ** 2
            for j in range(K)
            if j != k
        )
        capacities.append(np.log2(1.0 + sig / (intf + simulator.noise_power_linear + 1e-10)))
    return float(np.sum(capacities))


def select_beam_index_from_qvals(
    qvals: np.ndarray,
    H: np.ndarray,
    codebook,
    simulator,
    topk: int = 1,
    rerank_mode: str = "capacity",
    hybrid_q_weight: float = 0.5,
) -> int:
    if topk <= 1:
        return int(np.argmax(qvals))

    k = min(int(topk), len(qvals))
    candidate_indices = np.argpartition(qvals, -k)[-k:]

    if rerank_mode == "q_only":
        return int(candidate_indices[np.argmax(qvals[candidate_indices])])

    candidate_qvals = np.asarray(qvals[candidate_indices], dtype=np.float64)
    proxy_scores = np.asarray(
        [_proxy_capacity_score(simulator, H, codebook.get_beam(int(idx))) for idx in candidate_indices],
        dtype=np.float64,
    )

    if rerank_mode == "hybrid":
        q_min, q_max = float(candidate_qvals.min()), float(candidate_qvals.max())
        s_min, s_max = float(proxy_scores.min()), float(proxy_scores.max())
        q_norm = (candidate_qvals - q_min) / max(q_max - q_min, 1e-9)
        s_norm = (proxy_scores - s_min) / max(s_max - s_min, 1e-9)
        blended = float(hybrid_q_weight) * q_norm + (1.0 - float(hybrid_q_weight)) * s_norm
        return int(candidate_indices[int(np.argmax(blended))])

    return int(candidate_indices[int(np.argmax(proxy_scores))])


def estimate_signal_metrics(simulator, H: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
    sinrs = compute_user_sinr_linear(simulator, H, W)
    sinr_db = float(10.0 * np.log10(np.mean(sinrs) + 1e-10))
    strength = float(np.mean(np.abs(H @ W) ** 2))
    return sinr_db, strength


def compute_user_sinr_linear(simulator, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    path_loss_linear = 10 ** (-simulator.calculate_path_loss_3gpp() / 10)
    sinrs = []
    K = H.shape[0]
    for k in range(K):
        sig = simulator.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, k]) ** 2
        intf = sum(
            simulator.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, j]) ** 2
            for j in range(K)
            if j != k
        )
        sinrs.append(float(sig / (intf + simulator.noise_power_linear + 1e-10)))
    return np.asarray(sinrs, dtype=np.float64)


def compute_user_sinr_db(simulator, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    sinr_linear = compute_user_sinr_linear(simulator, H, W)
    return 10.0 * np.log10(sinr_linear + 1e-10)


def load_realtime_bundle(
    project_root: str,
    model_path: Optional[str] = None,
    artifacts_path: Optional[str] = None,
    tflite_model_path: Optional[str] = None,
    inference_backend: str = "tflite",
) -> RealtimeBeamModel:
    _ensure_src_on_path(project_root)

    from simulators import BeamformingSimulatorV4
    import dqn_beam_agent

    _ = dqn_beam_agent

    results_dir = os.path.join(project_root, "results")
    resolved_model_path = model_path or os.path.join(results_dir, "dqn_beam_model.keras")
    resolved_tflite_path = tflite_model_path or os.path.join(results_dir, "dqn_beam_model_int8.tflite")
    resolved_artifacts_path = artifacts_path or os.path.join(results_dir, "dqn_beam_artifacts.pkl")

    inference_backend = str(inference_backend).lower().strip()
    if inference_backend not in {"keras", "tflite"}:
        raise ValueError("inference_backend must be either 'keras' or 'tflite'")

    if not os.path.exists(resolved_artifacts_path):
        raise FileNotFoundError(f"Artifacts not found: {resolved_artifacts_path}")

    model = None
    interpreter = None

    if inference_backend == "tflite":
        if not os.path.exists(resolved_tflite_path):
            raise FileNotFoundError(f"TFLite model not found: {resolved_tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=resolved_tflite_path)
        interpreter.allocate_tensors()
    else:
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"Model not found: {resolved_model_path}")
        
        # APPLY FIX HERE: Add compile=False
        model = tf.keras.models.load_model(resolved_model_path, compile=False)

    with open(resolved_artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    default_snr_db = float(simulator.snr_db_list[len(simulator.snr_db_list) // 2])

    return RealtimeBeamModel(
        project_root=project_root,
        model=model,
        tflite_interpreter=interpreter,
        inference_backend=inference_backend,
        codebook=artifacts["codebook"],
        state_encoder=artifacts["state_encoder"],
        state_scaler=artifacts["state_scaler"],
        phase1_augmenter=artifacts.get("phase1_augmenter", None),
        simulator=simulator,
        default_snr_db=default_snr_db,
    )
