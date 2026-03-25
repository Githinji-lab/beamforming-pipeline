import os
import sys
import time
import pickle
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from preprocessing import calculate_mmse_weights_adjusted
from baselines import calculate_zf_weights_adjusted


def _run_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def benchmark(num_iterations=200):
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

    stats = {
        'mmse': {'capacity': [], 'latency_ms': []},
        'zf': {'capacity': [], 'latency_ms': []},
        'rl_teacher': {'capacity': [], 'latency_ms': []},
        'rl_student_tflite': {'capacity': [], 'latency_ms': []},
        'dqn_beam': {'capacity': [], 'latency_ms': []},
        'dqn_beam_tflite': {'capacity': [], 'latency_ms': []},
    }

    for i in range(num_iterations):
        H = simulator.generate_channel_matrix_v4()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]

        t0 = time.perf_counter()
        W_mmse = calculate_mmse_weights_adjusted(H, simulator)
        stats['mmse']['latency_ms'].append((time.perf_counter() - t0) * 1000)
        stats['mmse']['capacity'].append(simulator.calculate_sum_capacity(H, W_mmse))

        t0 = time.perf_counter()
        W_zf = calculate_zf_weights_adjusted(H, simulator)
        stats['zf']['latency_ms'].append((time.perf_counter() - t0) * 1000)
        stats['zf']['capacity'].append(simulator.calculate_sum_capacity(H, W_zf))

        encoded = state_encoder.encode(H, snr).reshape(1, -1)

        if actor_model is not None:
            t0 = time.perf_counter()
            mean, _ = actor_model(encoded, training=False)
            logits = tf.tanh(mean).numpy()[0]
            beam_idx = int(np.argmax(logits))
            W_rl = codebook.get_beam(beam_idx)
            stats['rl_teacher']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['rl_teacher']['capacity'].append(simulator.calculate_sum_capacity(H, W_rl))

        if interpreter is not None:
            t0 = time.perf_counter()
            logits = _run_tflite(interpreter, encoded)[0]
            beam_idx = int(np.argmax(logits))
            W_st = codebook.get_beam(beam_idx)
            stats['rl_student_tflite']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['rl_student_tflite']['capacity'].append(simulator.calculate_sum_capacity(H, W_st))

        if dqn_model is not None and dqn_artifacts is not None:
            dqn_encoder = dqn_artifacts['state_encoder']
            dqn_scaler = dqn_artifacts['state_scaler']
            dqn_codebook = dqn_artifacts['codebook']
            dqn_state = dqn_scaler.transform(dqn_encoder.encode(H, snr).reshape(1, -1))

            t0 = time.perf_counter()
            qvals = dqn_model(dqn_state, training=False).numpy()[0]
            beam_idx = int(np.argmax(qvals))
            W_dqn = dqn_codebook.get_beam(beam_idx)
            stats['dqn_beam']['latency_ms'].append((time.perf_counter() - t0) * 1000)
            stats['dqn_beam']['capacity'].append(simulator.calculate_sum_capacity(H, W_dqn))

            if dqn_interpreter is not None:
                t0 = time.perf_counter()
                qvals_tflite = _run_tflite(dqn_interpreter, dqn_state)[0]
                beam_idx_tflite = int(np.argmax(qvals_tflite))
                W_dqn_tfl = dqn_codebook.get_beam(beam_idx_tflite)
                stats['dqn_beam_tflite']['latency_ms'].append((time.perf_counter() - t0) * 1000)
                stats['dqn_beam_tflite']['capacity'].append(simulator.calculate_sum_capacity(H, W_dqn_tfl))

    print("=" * 72)
    print("BENCHMARK: CAPACITY AND LATENCY")
    print("=" * 72)
    for method, d in stats.items():
        if not d['capacity']:
            continue
        cap = np.array(d['capacity'])
        lat = np.array(d['latency_ms'])
        print(
            f"{method:18s} | "
            f"cap_mean={cap.mean():7.3f} | cap_std={cap.std():6.3f} | "
            f"lat_mean={lat.mean():7.3f} ms | lat_p95={np.percentile(lat,95):7.3f} ms"
        )


if __name__ == '__main__':
    benchmark()
