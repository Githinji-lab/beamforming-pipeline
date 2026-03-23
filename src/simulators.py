import numpy as np
from scipy import linalg

class BeamformingSimulatorV4:
    def __init__(self, N_tx=8, K=4, fc=3.5e9, distance=100, scenario='UMa_LoS',
                 P_tx_dbm=30, N0_dbm_hz=-174, B_hz=10e6, snr_db_list=None):
        self.N_tx = N_tx
        self.K = K
        self.fc = fc
        self.c = 3e8
        self.lambda_ = self.c / self.fc
        self.d = self.lambda_ / 2
        self.distance = distance
        self.scenario = scenario

        if snr_db_list is None:
            self.snr_db_list = np.array([0, 5, 10, 15, 20])
        else:
            self.snr_db_list = snr_db_list

        self.P_tx_linear = 10**((P_tx_dbm - 30) / 10)
        self.N0_linear = 10**((N0_dbm_hz - 30) / 10)
        self.noise_power_linear = self.N0_linear * B_hz
        self.K_rician = self._get_rician_k_factor(scenario)

    def _get_rician_k_factor(self, scenario):
        if scenario == 'UMa_LoS': return 10**(7/10)
        elif scenario == 'UMa_NLoS': return 10**(0/10)
        elif scenario == 'RMa_LoS': return 10**(10/10)
        else: return 10**(5/10)

    def calculate_path_loss_3gpp(self):
        f_c_ghz = self.fc / 1e9
        if self.scenario == 'UMa_LoS': pl = 28.0 + 22 * np.log10(self.distance) + 20 * np.log10(f_c_ghz)
        else: pl = 32.4 + 20 * np.log10(self.distance) + 20 * np.log10(f_c_ghz) + 9.5 * np.log10(f_c_ghz)
        return pl + np.random.normal(0, 8)

    def generate_rician_channel_v4(self, return_details=False):
        los_angle = np.random.uniform(-np.pi/3, np.pi/3)
        a_los = np.exp(1j * 2 * np.pi * self.d * np.arange(self.N_tx) * np.sin(los_angle))
        num_nlos = 6
        angles_nlos = np.random.uniform(-np.pi/2, np.pi/2, num_nlos)
        a_nlos = np.sum([(np.random.randn() + 1j*np.random.randn())/np.sqrt(2) * np.exp(1j * 2 * np.pi * self.d * np.arange(self.N_tx) * np.sin(np.random.uniform(-np.pi/2, np.pi/2))) for _ in range(num_nlos)], axis=0)
        h = np.sqrt(self.K_rician/(1 + self.K_rician)) * a_los + np.sqrt(1/(1 + self.K_rician)) * a_nlos
        h_normalized_fading = h / np.linalg.norm(h)

        if return_details:
            path_loss_db_inst = self.calculate_path_loss_3gpp()
            path_loss_linear_inst = 10**(-path_loss_db_inst / 10)
            h_full_channel = h_normalized_fading * np.sqrt(path_loss_linear_inst)
            return h_full_channel, path_loss_linear_inst, np.linalg.norm(h_normalized_fading)
        else:
            return h_normalized_fading

    def generate_channel_matrix_v4(self, return_details=False):
        H = np.zeros((self.K, self.N_tx), dtype=complex)
        for k in range(self.K):
            H[k, :] = self.generate_rician_channel_v4()
        return H

    def calculate_sum_capacity(self, H, W, debug_print=False):
        capacities = []
        path_loss_linear = 10**(-self.calculate_path_loss_3gpp() / 10)
        K = H.shape[0]
        for k in range(K):
            sig = self.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, k])**2
            intf = sum(self.P_tx_linear * path_loss_linear * np.abs(H[k, :] @ W[:, j])**2 for j in range(self.K) if j != k)
            capacities.append(np.log2(1 + sig / (intf + self.noise_power_linear)))
        return np.sum(capacities)

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

# --- ReplayBuffer ---