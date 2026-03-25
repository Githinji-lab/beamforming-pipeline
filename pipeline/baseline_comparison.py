import numpy as np
import pickle
import os
import sys
import time

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
results_dir = os.path.join(project_root, 'results')

from simulators import BeamformingSimulatorV4
from preprocessing import calculate_mmse_weights_adjusted
from baselines import (calculate_zf_weights_adjusted, 
                       calculate_mrt_weights_adjusted,
                       calculate_slnr_weights_adjusted,
                       calculate_greedy_codebook_beam,
                       calculate_multi_objective_reward)
from state_encoder import BeamCodebook
import tensorflow as tf

# Constants
TARGET_N_TX = 8
TARGET_K = 4
NUM_BEAMS = 32

def compare_all_baselines(num_eval_iterations=500, save_results=True):
    """Comprehensive comparison of all baseline methods."""
    
    print("="*60)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("="*60)
    
    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)
    
    # Initialize results dictionary
    results = {
        'mmse': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
        'zf': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
        'mrt': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
        'slnr': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
        'greedy_codebook': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
        'random': {'capacities': [], 'sinrs': [], 'latencies': [], 'times': []},
    }
    
    # Generate codebook for greedy selection
    print("\nGenerating beam codebook...")
    codebook = BeamCodebook(N_tx=TARGET_N_TX, K=TARGET_K, num_beams=NUM_BEAMS)
    codebook.generate_codebook(simulator, num_samples=1000)
    
    actor_model_path = os.path.join(results_dir, 'sac_actor_model.h5')
    if os.path.exists(actor_model_path):
        print(f"Detected trained SAC actor model at {actor_model_path}.")
    else:
        print("No SAC actor model found in results; running classical baselines only.")
    
    print(f"\nRunning evaluation for {num_eval_iterations} iterations...")
    
    for i in range(num_eval_iterations):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{num_eval_iterations}")
        
        # Generate channel
        H = simulator.generate_channel_matrix_v4()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]
        
        # --- MMSE ---
        t_start = time.time()
        W_mmse = calculate_mmse_weights_adjusted(H, simulator)
        t_mmse = time.time() - t_start
        cap_mmse = simulator.calculate_sum_capacity(H, W_mmse)
        results['mmse']['capacities'].append(cap_mmse)
        results['mmse']['times'].append(t_mmse)
        
        # --- ZF ---
        t_start = time.time()
        W_zf = calculate_zf_weights_adjusted(H, simulator)
        t_zf = time.time() - t_start
        cap_zf = simulator.calculate_sum_capacity(H, W_zf)
        results['zf']['capacities'].append(cap_zf)
        results['zf']['times'].append(t_zf)
        
        # --- MRT ---
        t_start = time.time()
        W_mrt = calculate_mrt_weights_adjusted(H, simulator)
        t_mrt = time.time() - t_start
        cap_mrt = simulator.calculate_sum_capacity(H, W_mrt)
        results['mrt']['capacities'].append(cap_mrt)
        results['mrt']['times'].append(t_mrt)
        
        # --- SLNR ---
        t_start = time.time()
        W_slnr = calculate_slnr_weights_adjusted(H, simulator)
        t_slnr = time.time() - t_start
        cap_slnr = simulator.calculate_sum_capacity(H, W_slnr)
        results['slnr']['capacities'].append(cap_slnr)
        results['slnr']['times'].append(t_slnr)
        
        # --- Greedy Codebook ---
        t_start = time.time()
        W_greedy = calculate_greedy_codebook_beam(H, codebook, simulator)
        t_greedy = time.time() - t_start
        cap_greedy = simulator.calculate_sum_capacity(H, W_greedy)
        results['greedy_codebook']['capacities'].append(cap_greedy)
        results['greedy_codebook']['times'].append(t_greedy)
        
        # --- Random Beam ---
        t_start = time.time()
        W_random = (np.random.randn(TARGET_N_TX, TARGET_K) + 
                   1j * np.random.randn(TARGET_N_TX, TARGET_K)) / np.sqrt(2)
        power_per_user = simulator.P_tx_linear / TARGET_K
        for k in range(TARGET_K):
            norm = np.linalg.norm(W_random[:, k])
            if norm > 1e-9:
                W_random[:, k] = (W_random[:, k] / norm) * np.sqrt(power_per_user)
        t_random = time.time() - t_start
        cap_random = simulator.calculate_sum_capacity(H, W_random)
        results['random']['capacities'].append(cap_random)
        results['random']['times'].append(t_random)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for method in results.keys():
        caps = results[method]['capacities']
        times = results[method]['times']

        if len(caps) == 0:
            continue
        
        mean_cap = np.mean(caps)
        std_cap = np.std(caps)
        mean_time = np.mean(times) * 1000  # Convert to ms
        
        print(f"\n{method.upper():20s}")
        print(f"  Avg Capacity: {mean_cap:7.3f} ± {std_cap:.3f} bps/Hz")
        print(f"  Avg Latency:  {mean_time:7.3f} ms")
        print(f"  Min/Max Cap:  {np.min(caps):7.3f} / {np.max(caps):7.3f} bps/Hz")
    
    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, 'baseline_comparison.pkl')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    compare_all_baselines(num_eval_iterations=500)
