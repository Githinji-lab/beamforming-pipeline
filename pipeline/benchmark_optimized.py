import os
import sys
import time
import pickle
import json
import argparse
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from preprocessing import calculate_mmse_weights_adjusted
from baselines import calculate_zf_weights_adjusted
from external_dataset import load_channels_from_registry, ExternalChannelSampler
import dqn_beam_agent  # registers custom Keras layers used by DQN models


def _run_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def _compute_sinr_ber(simulator, H, W):
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
        sinr = float(sig / (intf + simulator.noise_power_linear + 1e-10))
        sinrs.append(sinr)

    sinrs = np.array(sinrs, dtype=np.float64)
    avg_sinr_lin = float(np.mean(sinrs))
    min_sinr_lin = float(np.min(sinrs))
    avg_sinr_db = float(10.0 * np.log10(avg_sinr_lin + 1e-10))
    min_sinr_db = float(10.0 * np.log10(min_sinr_lin + 1e-10))

    # QPSK-like approximation
    ber_users = 0.5 * np.exp(-sinrs)
    avg_ber = float(np.mean(ber_users))
    return avg_sinr_db, min_sinr_db, avg_ber


def _proxy_capacity_score(simulator, H, W):
    capacities = []
    K = H.shape[0]
    for k in range(K):
        sig = simulator.P_tx_linear * np.abs(H[k, :] @ W[:, k]) ** 2
        intf = sum(
            simulator.P_tx_linear * np.abs(H[k, :] @ W[:, j]) ** 2
            for j in range(K)
            if j != k
        )
        capacities.append(np.log2(1 + sig / (intf + simulator.noise_power_linear + 1e-10)))
    return float(np.sum(capacities))


def _rerank_beam_idx_from_qvals(
    simulator,
    H,
    qvals,
    codebook,
    topk,
    rerank_mode='capacity',
    hybrid_q_weight=0.5,
):
    if topk <= 1:
        return int(np.argmax(qvals))

    k = min(int(topk), len(qvals))
    candidate_indices = np.argpartition(qvals, -k)[-k:]

    if rerank_mode == 'q_only':
        return int(candidate_indices[np.argmax(qvals[candidate_indices])])

    candidate_qvals = np.asarray(qvals[candidate_indices], dtype=np.float64)
    proxy_scores = np.asarray(
        [
            _proxy_capacity_score(simulator, H, codebook.get_beam(int(idx)))
            for idx in candidate_indices
        ],
        dtype=np.float64,
    )

    if rerank_mode == 'hybrid':
        q_min, q_max = float(candidate_qvals.min()), float(candidate_qvals.max())
        s_min, s_max = float(proxy_scores.min()), float(proxy_scores.max())
        q_norm = (candidate_qvals - q_min) / max(q_max - q_min, 1e-9)
        s_norm = (proxy_scores - s_min) / max(s_max - s_min, 1e-9)
        blended = hybrid_q_weight * q_norm + (1.0 - hybrid_q_weight) * s_norm
        return int(candidate_indices[int(np.argmax(blended))])

    best_idx = int(candidate_indices[0])
    best_score = -np.inf
    for idx, score in zip(candidate_indices, proxy_scores):
        if score > best_score:
            best_score = float(score)
            best_idx = int(idx)
    return best_idx


def _summarize_stats(stats):
    summary = {}
    for method, d in stats.items():
        if len(d['capacity']) == 0:
            continue
        cap = np.array(d['capacity'], dtype=np.float64)
        lat = np.array(d['latency_ms'], dtype=np.float64)
        sinr = np.array(d['sinr_db'], dtype=np.float64)
        ber = np.array(d['ber'], dtype=np.float64)
        summary[method] = {
            'cap_mean': float(cap.mean()),
            'cap_std': float(cap.std()),
            'lat_mean_ms': float(lat.mean()),
            'lat_p95_ms': float(np.percentile(lat, 95)),
            'sinr_mean_db': float(sinr.mean()),
            'sinr_p05_db': float(np.percentile(sinr, 5)),
            'ber_mean': float(ber.mean()),
            'ber_p95': float(np.percentile(ber, 95)),
        }
    return summary


def benchmark(
    num_iterations=200,
    save_json_path=None,
    channel_source='simulator',
    external_registry_path='data/dataset_registry.json',
    external_max_samples=20000,
    external_mix_ratio=0.5,
    dqn_rerank_topk=1,
    dqn_rerank_mode='capacity',
    dqn_hybrid_q_weight=0.5,
    seed=None,
):
    if seed is not None:
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))

    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(project_root, 'results')

    codebook_path = os.path.join(results_dir, 'beam_codebook.pkl')
    actor_path = os.path.join(results_dir, 'improved_sac_actor_model.keras')
    student_tflite_path = os.path.join(results_dir, 'improved_sac_student_int8.tflite')
    dqn_model_path = os.path.join(results_dir, 'dqn_beam_model.keras')
    dqn_tflite_path = os.path.join(results_dir, 'dqn_beam_model_int8.tflite')
    dqn_artifacts_path = os.path.join(results_dir, 'dqn_beam_artifacts.pkl')

    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"Missing {codebook_path}. Run train_improved first.")

    with open(codebook_path, 'rb') as f:
        artifacts = pickle.load(f)
    codebook = artifacts['codebook']
    state_encoder = artifacts['state_encoder']

    actor_model = tf.keras.models.load_model(actor_path) if os.path.exists(actor_path) else None
    dqn_model = tf.keras.models.load_model(dqn_model_path) if os.path.exists(dqn_model_path) else None

    interpreter = None
    if os.path.exists(student_tflite_path):
        interpreter = tf.lite.Interpreter(model_path=student_tflite_path)
        interpreter.allocate_tensors()

    dqn_interpreter = None
    if os.path.exists(dqn_tflite_path):
        dqn_interpreter = tf.lite.Interpreter(model_path=dqn_tflite_path)
        dqn_interpreter.allocate_tensors()

    dqn_artifacts = None
    if os.path.exists(dqn_artifacts_path):
        with open(dqn_artifacts_path, 'rb') as f:
            dqn_artifacts = pickle.load(f)

    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    external_sampler = None

    if channel_source in ('external', 'mixed'):
        channels = load_channels_from_registry(
            registry_path=external_registry_path,
            target_k=simulator.K,
            target_n_tx=simulator.N_tx,
            max_total_samples=external_max_samples,
        )
        external_sampler = ExternalChannelSampler(channels, seed=seed if seed is not None else 42)
        print(f"Loaded external channels for benchmark: {channels.shape[0]}")

    def sample_channel():
        if channel_source == 'external':
            return external_sampler.sample()
        if channel_source == 'mixed':
            if np.random.rand() < external_mix_ratio:
                return external_sampler.sample()
            return simulator.generate_channel_matrix_v4()
        return simulator.generate_channel_matrix_v4()

    stats = {
        'mmse': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
        'zf': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
        'rl_teacher': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
        'rl_student_tflite': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
        'dqn_beam': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
        'dqn_beam_tflite': {'capacity': [], 'latency_ms': [], 'sinr_db': [], 'ber': []},
    }

    prev_h_for_dqn = None

    for i in range(num_iterations):
        H = sample_channel()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]

        t0 = time.perf_counter()
        W_mmse = calculate_mmse_weights_adjusted(H, simulator)
        stats['mmse']['latency_ms'].append((time.perf_counter() - t0) * 1000)
        stats['mmse']['capacity'].append(simulator.calculate_sum_capacity(H, W_mmse))
        sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_mmse)
        stats['mmse']['sinr_db'].append(sinr_db)
        stats['mmse']['ber'].append(ber)

        t0 = time.perf_counter()
        W_zf = calculate_zf_weights_adjusted(H, simulator)
        stats['zf']['latency_ms'].append((time.perf_counter() - t0) * 1000)
        stats['zf']['capacity'].append(simulator.calculate_sum_capacity(H, W_zf))
        sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_zf)
        stats['zf']['sinr_db'].append(sinr_db)
        stats['zf']['ber'].append(ber)

        encoded = state_encoder.encode(H, snr).reshape(1, -1)

        if actor_model is not None:
            t0 = time.perf_counter()
            mean, _ = actor_model(encoded, training=False)
            logits = tf.tanh(mean).numpy()[0]
            beam_idx = int(np.argmax(logits))
            W_rl = codebook.get_beam(beam_idx)
            stats['rl_teacher']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['rl_teacher']['capacity'].append(simulator.calculate_sum_capacity(H, W_rl))
            sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_rl)
            stats['rl_teacher']['sinr_db'].append(sinr_db)
            stats['rl_teacher']['ber'].append(ber)

        if interpreter is not None:
            t0 = time.perf_counter()
            logits = _run_tflite(interpreter, encoded)[0]
            beam_idx = int(np.argmax(logits))
            W_st = codebook.get_beam(beam_idx)
            stats['rl_student_tflite']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['rl_student_tflite']['capacity'].append(simulator.calculate_sum_capacity(H, W_st))
            sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_st)
            stats['rl_student_tflite']['sinr_db'].append(sinr_db)
            stats['rl_student_tflite']['ber'].append(ber)

        if dqn_model is not None and dqn_artifacts is not None:
            dqn_encoder = dqn_artifacts['state_encoder']
            dqn_scaler = dqn_artifacts['state_scaler']
            dqn_codebook = dqn_artifacts['codebook']
            dqn_phase1 = dqn_artifacts.get('phase1_augmenter', None)
            dqn_prev_h = None
            if i > 0:
                # Approximate temporal feature for inference-time consistency.
                dqn_prev_h = prev_h_for_dqn

            dqn_base_state = dqn_encoder.encode(H, snr)
            if dqn_phase1 is not None:
                dqn_base_state = dqn_phase1.transform(
                    base_state=dqn_base_state,
                    H=H,
                    snr=snr,
                    prev_H=dqn_prev_h,
                )
            dqn_state = dqn_scaler.transform(np.array(dqn_base_state, dtype=np.float32).reshape(1, -1))

            t0 = time.perf_counter()
            qvals = dqn_model(dqn_state, training=False).numpy()[0]
            beam_idx = _rerank_beam_idx_from_qvals(
                simulator=simulator,
                H=H,
                qvals=qvals,
                codebook=dqn_codebook,
                topk=dqn_rerank_topk,
                rerank_mode=dqn_rerank_mode,
                hybrid_q_weight=dqn_hybrid_q_weight,
            )
            W_dqn = dqn_codebook.get_beam(beam_idx)
            stats['dqn_beam']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['dqn_beam']['capacity'].append(simulator.calculate_sum_capacity(H, W_dqn))
            sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_dqn)
            stats['dqn_beam']['sinr_db'].append(sinr_db)
            stats['dqn_beam']['ber'].append(ber)

            if dqn_interpreter is not None:
                t0 = time.perf_counter()
                qvals_tflite = _run_tflite(dqn_interpreter, dqn_state)[0]
                beam_idx_tflite = _rerank_beam_idx_from_qvals(
                    simulator=simulator,
                    H=H,
                    qvals=qvals_tflite,
                    codebook=dqn_codebook,
                    topk=dqn_rerank_topk,
                    rerank_mode=dqn_rerank_mode,
                    hybrid_q_weight=dqn_hybrid_q_weight,
                )
                W_dqn_tfl = dqn_codebook.get_beam(beam_idx_tflite)
                stats['dqn_beam_tflite']['latency_ms'].append((time.perf_counter() - t0) * 1000)
                stats['dqn_beam_tflite']['capacity'].append(simulator.calculate_sum_capacity(H, W_dqn_tfl))
                sinr_db, _, ber = _compute_sinr_ber(simulator, H, W_dqn_tfl)
                stats['dqn_beam_tflite']['sinr_db'].append(sinr_db)
                stats['dqn_beam_tflite']['ber'].append(ber)

            prev_h_for_dqn = H
        else:
            prev_h_for_dqn = H

    summary = _summarize_stats(stats)

    print("=" * 72)
    print("BENCHMARK: CAPACITY, LATENCY, SINR, BER")
    print("=" * 72)
    for method, d in summary.items():
        if d is None:
            continue
        print(
            f"{method:18s} | "
            f"cap_mean={d['cap_mean']:7.3f} | cap_std={d['cap_std']:6.3f} | "
            f"lat_mean={d['lat_mean_ms']:7.3f} ms | lat_p95={d['lat_p95_ms']:7.3f} ms | "
            f"sinr_mean={d['sinr_mean_db']:6.2f} dB | ber_mean={d['ber_mean']:.3e}"
        )

    if save_json_path is None:
        save_json_path = os.path.join(results_dir, 'benchmark_optimized_summary.json')
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    with open(save_json_path, 'w') as f:
        json.dump(
            {
                'num_iterations': num_iterations,
                'protocol': {
                    'channel_source': channel_source,
                    'dqn_rerank_topk': int(dqn_rerank_topk),
                    'dqn_rerank_mode': dqn_rerank_mode,
                    'dqn_hybrid_q_weight': float(dqn_hybrid_q_weight),
                    'seed': seed,
                },
                'summary': summary,
            },
            f,
            indent=2,
        )

    return stats, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--json-out', type=str, default=None)
    p.add_argument('--channel-source', type=str, default='simulator', choices=['simulator', 'external', 'mixed'])
    p.add_argument('--external-registry', type=str, default='data/dataset_registry.json')
    p.add_argument('--external-max-samples', type=int, default=20000)
    p.add_argument('--external-mix-ratio', type=float, default=0.5)
    p.add_argument('--dqn-rerank-topk', type=int, default=1)
    p.add_argument('--dqn-rerank-mode', type=str, default='capacity', choices=['capacity', 'hybrid', 'q_only'])
    p.add_argument('--dqn-hybrid-q-weight', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark(
        num_iterations=args.iterations,
        save_json_path=args.json_out,
        channel_source=args.channel_source,
        external_registry_path=args.external_registry,
        external_max_samples=args.external_max_samples,
        external_mix_ratio=args.external_mix_ratio,
        dqn_rerank_topk=args.dqn_rerank_topk,
        dqn_rerank_mode=args.dqn_rerank_mode,
        dqn_hybrid_q_weight=args.dqn_hybrid_q_weight,
        seed=args.seed,
    )
