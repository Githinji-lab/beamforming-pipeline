import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Preprocessing Functions ---
def preprocess_channel(H_original, snr_value, target_N_TX, target_K):
    K_orig, N_TX_orig = H_original.shape
    H_adjusted = np.zeros((target_K, target_N_TX), dtype=complex)
    min_K, min_N = min(K_orig, target_K), min(N_TX_orig, target_N_TX)
    H_adjusted[:min_K, :min_N] = H_original[:min_K, :min_N]

    H_flat = H_adjusted.flatten()
    H_processed = np.concatenate((np.real(H_flat), np.imag(H_flat)))
    state_vector = np.append(H_processed, snr_value)
    return state_vector

def preprocess_weights(W_original, target_N_TX, target_K):
    N_TX_original, K_original = W_original.shape
    W_adjusted = np.zeros((target_N_TX, target_K), dtype=complex)
    min_N_TX, min_K = min(N_TX_original, target_N_TX), min(K_original, target_K)
    W_adjusted[:min_N_TX, :min_K] = W_original[:min_N_TX, :min_K]
    W_flat = np.concatenate((np.real(W_adjusted.flatten()), np.imag(W_adjusted.flatten())))
    return W_flat

def reconstruct_complex_weights(action_vector_scaled, N_tx, K, simulator_instance, scaler_y):
    action_vector_raw = scaler_y.inverse_transform(action_vector_scaled.reshape(1, -1))[0]
    split_idx = N_tx * K
    W_real = action_vector_raw[:split_idx].reshape(N_tx, K)
    W_imag = action_vector_raw[split_idx:].reshape(N_tx, K)
    W_complex = W_real + 1j * W_imag

    power_per_user = simulator_instance.P_tx_linear / K
    W_scaled_final = np.zeros_like(W_complex, dtype=complex)
    for k_idx in range(K):
        col_norm = np.linalg.norm(W_complex[:, k_idx])
        if col_norm > 1e-9:
            W_scaled_final[:, k_idx] = W_complex[:, k_idx] / col_norm * np.sqrt(power_per_user)
        else:
            W_scaled_final[:, k_idx] = np.zeros_like(W_complex[:, k_idx])
    return W_scaled_final

# --- Traditional Beamforming Functions ---
def calculate_mmse_weights_adjusted(H, simulator_instance):
    K = H.shape[0]
    reg_factor = simulator_instance.noise_power_linear / simulator_instance.P_tx_linear
    if K == 1:
        W = H.conj().T / (H @ H.conj().T + reg_factor)
    else:
        I = np.eye(K)
        W = H.conj().T @ np.linalg.inv(H @ H.conj().T + reg_factor * I)
    power_per_user = simulator_instance.P_tx_linear / K
    W_normalized = np.zeros_like(W, dtype=complex)
    for k_idx in range(K):
        col_norm = np.linalg.norm(W[:, k_idx])
        if col_norm > 1e-9:
            W_normalized[:, k_idx] = W[:, k_idx] / col_norm * np.sqrt(power_per_user)
        else:
            W_normalized[:, k_idx] = np.zeros_like(W[:, k_idx])
    return W_normalized