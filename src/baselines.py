import numpy as np
from scipy import linalg
import time
from preprocessing import calculate_mmse_weights_adjusted

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
        R_nl = noise_factor * np.eye(N_tx, dtype=complex)
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
            W[:, k] = (W[:, k] / norm) * np.sqrt(power_per_user)
    
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


def _flatten_beam_complex(W):
    return np.concatenate([np.real(W.flatten()), np.imag(W.flatten())])


def nearest_codebook_index_from_beam(W, codebook):
    if codebook.codebook is None:
        raise ValueError("Codebook not generated. Call generate_codebook() first.")
    target = _flatten_beam_complex(W)
    distances = np.linalg.norm(codebook.codebook - target.reshape(1, -1), axis=1)
    return int(np.argmin(distances))


def select_teacher_beam_index(H, simulator_instance, codebook):
    W_mmse = calculate_mmse_weights_adjusted(H, simulator_instance)
    cap_mmse = simulator_instance.calculate_sum_capacity(H, W_mmse)

    W_slnr = calculate_slnr_weights_adjusted(H, simulator_instance)
    cap_slnr = simulator_instance.calculate_sum_capacity(H, W_slnr)

    W_teacher = W_mmse if cap_mmse >= cap_slnr else W_slnr
    return nearest_codebook_index_from_beam(W_teacher, codebook)


def calculate_multi_objective_reward(H, W, simulator_instance,
                                     alpha=0.6, beta=0.2, gamma=0.1,
                                     target_snr=15.0,
                                     inference_latency_ms=0.0,
                                     latency_budget_ms=1.0,
                                     latency_budget_weight=0.8):
    """Multi-objective reward with constrained latency penalty.

    Reward = alpha*throughput - beta*stability_penalty - gamma*ber_penalty - budget_latency_penalty
    """
    
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
    
    # Stability penalty (prefer smoother/less extreme beams)
    W_norm = np.linalg.norm(W) + 1e-9
    stability_penalty = W_norm

    # Constrained latency penalty: only penalize over-budget inference
    over_budget_ms = max(0.0, float(inference_latency_ms) - float(latency_budget_ms))
    budget_latency_penalty = latency_budget_weight * over_budget_ms
    
    # Multi-objective reward
    reward = (alpha * throughput 
             - beta * stability_penalty
             - gamma * ber_penalty
             - budget_latency_penalty)
    
    return reward, {
        'throughput': throughput,
        'avg_sinr': avg_sinr,
        'min_sinr': min_sinr,
        'ber': ber,
        'stability_penalty': stability_penalty,
        'inference_latency_ms': float(inference_latency_ms),
        'latency_budget_ms': float(latency_budget_ms),
        'over_budget_ms': over_budget_ms,
        'budget_latency_penalty': budget_latency_penalty
    }
