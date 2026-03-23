import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import time

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from agents import SACAgent
from preprocessing import preprocess_channel, reconstruct_complex_weights, calculate_mmse_weights_adjusted

# Constants
TARGET_N_TX = 8
TARGET_K = 4

def calculate_sinr(H, W, noise_power, P_tx_linear):
    """Calculate SINR for each user."""
    sinr_values = []
    K = H.shape[0]
    for k in range(K):
        signal_power = P_tx_linear * np.abs(H[k, :] @ W[:, k])**2
        interference_power = sum(P_tx_linear * np.abs(H[k, :] @ W[:, j])**2 for j in range(K) if j != k)
        sinr = signal_power / (interference_power + noise_power)
        sinr_values.append(sinr)
    return np.array(sinr_values)

def calculate_ber(sinr_values):
    """Estimate BER from SINR using QPSK modulation approximation."""
    ber_values = []
    for sinr in sinr_values:
        # QPSK BER approximation
        ber = 0.5 * np.exp(-sinr)
        ber_values.append(ber)
    return np.array(ber_values)

def evaluate_agent_performance(agent, num_eval_iterations=50, snr_list=None):
    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)

    if snr_list is None:
        snr_list = simulator.snr_db_list

    results_by_snr = {snr: {'sac': [], 'mmse': []} for snr in snr_list}
    metrics_by_snr = {snr: {'sac_sinr': [], 'mmse_sinr': [], 'sac_ber': [], 'mmse_ber': [], 
                             'sac_latency': [], 'mmse_latency': []} for snr in snr_list}

    print(f"Starting evaluation for {num_eval_iterations} iterations across {len(snr_list)} SNR levels...")

    for snr in snr_list:
        print(f"\n  Evaluating at SNR={snr} dB...")
        for i in range(num_eval_iterations):
            if (i + 1) % 25 == 0:
                print(f"    Progress: {i + 1}/{num_eval_iterations} samples")
            
            H = simulator.generate_channel_matrix_v4()
            state_eval_raw = preprocess_channel(H, snr, TARGET_N_TX, TARGET_K)
            state_eval_scaled = agent.scaler_X.transform(state_eval_raw.reshape(1, -1))[0]
            
            # SAC Agent
            start_time = time.time()
            action_eval_scaled = agent.choose_action(state_eval_scaled, evaluate=True)
            action_eval_W = reconstruct_complex_weights(action_eval_scaled, TARGET_N_TX, TARGET_K, simulator, agent.scaler_y)
            sac_latency = time.time() - start_time
            cap_sac = simulator.calculate_sum_capacity(H, action_eval_W)
            sinr_sac = calculate_sinr(H, action_eval_W, simulator.noise_power_linear, simulator.P_tx_linear)
            ber_sac = calculate_ber(sinr_sac)
            
            results_by_snr[snr]['sac'].append(cap_sac)
            metrics_by_snr[snr]['sac_sinr'].append(np.mean(sinr_sac))
            metrics_by_snr[snr]['sac_ber'].append(np.mean(ber_sac))
            metrics_by_snr[snr]['sac_latency'].append(sac_latency)
            
            # MMSE
            start_time = time.time()
            W_mmse = calculate_mmse_weights_adjusted(H, simulator)
            mmse_latency = time.time() - start_time
            cap_mmse = simulator.calculate_sum_capacity(H, W_mmse)
            sinr_mmse = calculate_sinr(H, W_mmse, simulator.noise_power_linear, simulator.P_tx_linear)
            ber_mmse = calculate_ber(sinr_mmse)
            
            results_by_snr[snr]['mmse'].append(cap_mmse)
            metrics_by_snr[snr]['mmse_sinr'].append(np.mean(sinr_mmse))
            metrics_by_snr[snr]['mmse_ber'].append(np.mean(ber_mmse))
            metrics_by_snr[snr]['mmse_latency'].append(mmse_latency)
    
    # Summarize results
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    sac_throughputs = []
    mmse_throughputs = []
    snrs = []
    
    for snr in snr_list:
        avg_cap_sac = np.mean(results_by_snr[snr]['sac'])
        avg_cap_mmse = np.mean(results_by_snr[snr]['mmse'])
        avg_sinr_sac = np.mean(metrics_by_snr[snr]['sac_sinr'])
        avg_sinr_mmse = np.mean(metrics_by_snr[snr]['mmse_sinr'])
        avg_ber_sac = np.mean(metrics_by_snr[snr]['sac_ber'])
        avg_ber_mmse = np.mean(metrics_by_snr[snr]['mmse_ber'])
        avg_latency_sac = np.mean(metrics_by_snr[snr]['sac_latency'])
        avg_latency_mmse = np.mean(metrics_by_snr[snr]['mmse_latency'])
        
        print(f"\nSNR = {snr} dB:")
        print(f"  Throughput: SAC {avg_cap_sac:.2f} | MMSE {avg_cap_mmse:.2f} bps/Hz")
        print(f"  SINR:       SAC {10*np.log10(avg_sinr_sac):.2f} | MMSE {10*np.log10(avg_sinr_mmse):.2f} dB")
        print(f"  BER:        SAC {avg_ber_sac:.2e} | MMSE {avg_ber_mmse:.2e}")
        print(f"  Latency:    SAC {avg_latency_sac*1000:.3f} | MMSE {avg_latency_mmse*1000:.3f} ms")
        
        sac_throughputs.append(avg_cap_sac)
        mmse_throughputs.append(avg_cap_mmse)
        snrs.append(snr)
    
    return snrs, sac_throughputs, mmse_throughputs, results_by_snr, metrics_by_snr

def plot_comprehensive_metrics(snrs, sac_throughputs, mmse_throughputs, results_by_snr, metrics_by_snr):
    """Plot comprehensive evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Throughput vs SNR
    axes[0, 0].plot(snrs, sac_throughputs, 'o-', label='SAC Agent', linewidth=2, markersize=8)
    axes[0, 0].plot(snrs, mmse_throughputs, 's--', label='MMSE', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('SNR (dB)', fontsize=11)
    axes[0, 0].set_ylabel('Throughput (bps/Hz)', fontsize=11)
    axes[0, 0].set_title('Throughput vs SNR', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SINR vs SNR
    sac_sinrs = [10*np.log10(np.mean(metrics_by_snr[snr]['sac_sinr'])) for snr in snrs]
    mmse_sinrs = [10*np.log10(np.mean(metrics_by_snr[snr]['mmse_sinr'])) for snr in snrs]
    axes[0, 1].plot(snrs, sac_sinrs, 'o-', label='SAC Agent', linewidth=2, markersize=8)
    axes[0, 1].plot(snrs, mmse_sinrs, 's--', label='MMSE', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('SNR (dB)', fontsize=11)
    axes[0, 1].set_ylabel('SINR (dB)', fontsize=11)
    axes[0, 1].set_title('SINR vs SNR', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # BER vs SNR
    sac_bers = [np.mean(metrics_by_snr[snr]['sac_ber']) for snr in snrs]
    mmse_bers = [np.mean(metrics_by_snr[snr]['mmse_ber']) for snr in snrs]
    axes[1, 0].semilogy(snrs, sac_bers, 'o-', label='SAC Agent', linewidth=2, markersize=8)
    axes[1, 0].semilogy(snrs, mmse_bers, 's--', label='MMSE', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('SNR (dB)', fontsize=11)
    axes[1, 0].set_ylabel('BER', fontsize=11)
    axes[1, 0].set_title('BER vs SNR (QPSK)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    # Latency vs SNR
    sac_latencies = [np.mean(metrics_by_snr[snr]['sac_latency']) * 1000 for snr in snrs]
    mmse_latencies = [np.mean(metrics_by_snr[snr]['mmse_latency']) * 1000 for snr in snrs]
    axes[1, 1].plot(snrs, sac_latencies, 'o-', label='SAC Agent', linewidth=2, markersize=8)
    axes[1, 1].plot(snrs, mmse_latencies, 's--', label='MMSE', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('SNR (dB)', fontsize=11)
    axes[1, 1].set_ylabel('Latency (ms)', fontsize=11)
    axes[1, 1].set_title('Latency vs SNR', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    print("\nComprehensive metrics plot saved to results/comprehensive_evaluation.png")
    plt.close()

if __name__ == "__main__":
    # Load trained agent
    state_dim = (TARGET_K * TARGET_N_TX * 2) + 1
    action_dim = TARGET_N_TX * TARGET_K * 2

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(np.random.randn(100, state_dim))
    scaler_y.fit(np.random.randn(100, action_dim))

    replay_buffer = []  # Dummy

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        scaler_X=scaler_X,
        scaler_y=scaler_y
    )

    # Load weights
    agent.actor_model.load_weights('results/sac_actor_model.h5')

    snrs, sac_throughputs, mmse_throughputs, results_by_snr, metrics_by_snr = evaluate_agent_performance(agent, num_eval_iterations=50)
    plot_comprehensive_metrics(snrs, sac_throughputs, mmse_throughputs, results_by_snr, metrics_by_snr)