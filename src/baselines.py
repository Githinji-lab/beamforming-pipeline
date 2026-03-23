import numpy as np
from scipy import linalg

def calculate_zf_weights_adjusted(H, simulator_instance):
    """Zero Forcing beamforming."""
    K = H.shape[0]
    if K == 1:
        W = H.conj().T / (H @ H.conj().T + 1e-6)
    else:
        W = H.conj().T @ np.linalg.inv(H @ H.conj().T + 1e-6 * np.eye(K))
    
    power_per_user = simulator_instance.P_tx_linear / K
    W_normalized = np.zeros_like(W, dtype=complex)
    for k_idx in range(K):
        col_norm = np.linalg.norm(W[:, k_idx])
        if col_norm > 1e-9:
            W_normalized[:, k_idx] = W[:, k_idx] / col_norm * np.sqrt(power_per_user)
        else:
            W_normalized[:, k_idx] = np.zeros_like(W[:, k_idx])
    return W_normalized


def calculate_mrt_weights_adjusted(H, simulator_instance):
    """Maximum Ratio Transmission beamforming."""
    K = H.shape[0]
    W = H.conj().T
    
    power_per_user = simulator_instance.P_tx_linear / K
    W_normalized = np.zeros_like(W, dtype=complex)
    for k_idx in range(K):
        col_norm = np.linalg.norm(W[:, k_idx])
        if col_norm > 1e-9:
            W_normalized[:, k_idx] = W[:, k_idx] / col_norm * np.sqrt(power_per_user)
        else:
            W_normalized[:, k_idx] = np.zeros_like(W[:, k_idx])
    return W_normalized


def calculate_slnr_weights_adjusted(H, simulator_instance):
    """Signal-to-Leakage-and-Noise-Ratio beamforming."""
    K = H.shape[0]
    N_tx = H.shape[1]
    W = np.zeros((N_tx, K), dtype=complex)
    
    noise_factor = simulator_instance.noise_power_linear / simulator_instance.P_tx_linear
    
    for k in range(K):
        H_k = np.delete(H, k, axis=0)
        R_nl = noise_factor * np.eye(N_tx)
        if H_k.size > 0:
            R_nl += H_k.conj().T @ H_k
        
        h_k = H[k, :].reshape(-1, 1)
        R_s = h_k @ h_k.conj().T
        
        try:
            eigenvalues, eigenvectors = linalg.eig(R_s, R_nl)
            max_idx = np.argmax(np.real(eigenvalues))
            w_k = eigenvectors[:, max_idx]
        except:
            w_k = h_k.conj().flatten() / (np.linalg.norm(h_k) + 1e-9)
        
        W[:, k] = w_k / (np.linalg.norm(w_k) + 1e-9)
    
    # Normalize power
    power_per_user = simulator_instance.P_tx_linear / K
    for k in range(K):
        norm = np.linalg.norm(W[:, k])
        if norm > 1e-9:
            W[:, k] /= norm * np.sqrt(power_per_user)
    
    return W


def calculate_greedy_codebook_beam(H, codebook, simulator_instance):
    """Greedy selection of best beam from codebook."""
    best_capacity = -np.inf
    best_beam_idx = 0
    
    for beam_idx in range(codebook.get_num_beams()):
        W = codebook.get_beam(beam_idx)
        cap = simulator_instance.calculate_sum_capacity(H, W)
        if cap > best_capacity:
            best_capacity = cap
            best_beam_idx = beam_idx
    
    return codebook.get_beam(best_beam_idx)


def calculate_multi_objective_reward(H, W, simulator_instance, 
                                     alpha=0.6, beta=0.3, gamma=0.1,
                                     target_snr=15.0):
    """Multi-objective reward: throughput - latency_penalty - ber_penalty."""
    
    # Calculate capacity (throughput objective)
    path_loss_db = simulator_instance.calculate_path_loss_3gpp()
    path_loss_linear = 10**(-path_loss_db / 10)
    
    capacities = []
    sinrs = []
    
    K = H.shape[0]
    for k in range(K):
        sig = simulator_instance.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, k])**2
        intf = sum(simulator_instance.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, j])**2 
                  for j in range(K) if j != k)
        sinr = sig / (intf + simulator_instance.noise_power_linear + 1e-10)
        sinrs.append(sinr)
        capacities.append(np.log2(1 + sinr))
    
    throughput = np.sum(capacities)
    avg_sinr = np.mean(sinrs)
    
    # Estimate BER (simplified: assume QPSK)
    min_sinr = np.min(sinrs)
    ber = 0.5 * np.exp(-min_sinr)
    ber_penalty = -np.log10(ber + 1e-10)  # Higher BER = lower penalty magnitude
    
    # Latency penalty (W change magnitude - prefer stable beams)
    W_norm = np.linalg.norm(W) + 1e-9
    latency_penalty = W_norm
    
    # Multi-objective reward
    reward = (alpha * throughput 
             - beta * latency_penalty 
             - gamma * ber_penalty)
    
    return reward, {
        'throughput': throughput,
        'avg_sinr': avg_sinr,
        'min_sinr': min_sinr,
        'ber': ber,
        'latency_norm': latency_penalty
    }
