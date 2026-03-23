import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ChannelStateEncoder:
    """Compress high-dim channel state to low-dim representation via PCA."""
    
    def __init__(self, target_dim=8, random_state=42):
        self.target_dim = target_dim
        self.pca = PCA(n_components=target_dim, random_state=random_state)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, H_samples, snr_samples):
        """Fit PCA on channel magnitude + phase features."""
        # Extract features: magnitude + phase of each element
        mags = np.abs(H_samples).reshape(H_samples.shape[0], -1)  # (N, K*N_tx)
        phases = np.angle(H_samples).reshape(H_samples.shape[0], -1)  # (N, K*N_tx)
        
        # Combine magnitude and phase
        features = np.concatenate([mags, phases], axis=1)
        
        # Add SNR as feature
        features = np.concatenate([features, snr_samples.reshape(-1, 1)], axis=1)
        
        # Fit scaler and PCA
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)
        self.pca.fit(features_scaled)
        self.fitted = True
        
        return self
    
    def encode(self, H, snr):
        """Encode channel matrix to low-dim vector."""
        if not self.fitted:
            raise ValueError("StateEncoder not fitted. Call fit() first.")
        
        mags = np.abs(H).reshape(-1)  # (K*N_tx,)
        phases = np.angle(H).reshape(-1)  # (K*N_tx,)
        features = np.concatenate([mags, phases, [snr]])
        features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
        encoded = self.pca.transform(features_scaled.reshape(1, -1))[0]
        
        return encoded
    
    def get_encoded_dim(self):
        return self.target_dim


class BeamCodebook:
    """Quantize beamforming weights into discrete beam patterns."""
    
    def __init__(self, N_tx=8, K=4, num_beams=32, random_state=42):
        self.N_tx = N_tx
        self.K = K
        self.num_beams = num_beams
        self.rng = np.random.RandomState(random_state)
        self.codebook = None
        self.simulator = None
    
    def generate_codebook(self, simulator, num_samples=5000):
        """Generate codebook via k-means clustering of random beams."""
        from sklearn.cluster import KMeans
        
        self.simulator = simulator
        
        # Generate random channel matrices and corresponding optimal beams
        beam_samples = []
        
        print(f"Generating {num_samples} beam samples for codebook...")
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples}")
            
            H = simulator.generate_channel_matrix_v4()
            
            # Generate random beamforming weights
            W_random = (np.random.randn(self.N_tx, self.K) + 
                       1j * np.random.randn(self.N_tx, self.K)) / np.sqrt(2)
            
            # Normalize per user
            for k in range(self.K):
                norm = np.linalg.norm(W_random[:, k])
                if norm > 1e-9:
                    W_random[:, k] /= norm
            
            # Flatten to feature vector
            W_flat = np.concatenate([np.real(W_random.flatten()), 
                                    np.imag(W_random.flatten())])
            beam_samples.append(W_flat)
        
        beam_samples = np.array(beam_samples)
        
        # K-means clustering
        print(f"Clustering {num_samples} beams into {self.num_beams} codebook entries...")
        kmeans = KMeans(n_clusters=self.num_beams, random_state=42, n_init=10)
        kmeans.fit(beam_samples)
        
        # Store codebook
        self.codebook = kmeans.cluster_centers_
        print(f"Codebook generated with {self.num_beams} beam patterns.")
        
        return self
    
    def get_beam(self, action_idx, N_tx=None, K=None):
        """Retrieve beam pattern from codebook."""
        if self.codebook is None:
            raise ValueError("Codebook not generated. Call generate_codebook() first.")
        
        action_idx = int(np.clip(action_idx, 0, self.num_beams - 1))
        W_flat = self.codebook[action_idx]
        
        N_tx = N_tx or self.N_tx
        K = K or self.K
        
        split_idx = N_tx * K
        W_real = W_flat[:split_idx].reshape(N_tx, K)
        W_imag = W_flat[split_idx:].reshape(N_tx, K)
        W = W_real + 1j * W_imag
        
        # Normalize per user
        for k in range(K):
            norm = np.linalg.norm(W[:, k])
            if norm > 1e-9:
                W[:, k] /= norm
        
        return W
    
    def get_num_beams(self):
        return self.num_beams


class CNNStateEncoder(keras.Model):
    """CNN-based channel encoder for spatial feature extraction."""
    
    def __init__(self, target_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.target_dim = target_dim
        
        # Input: (K, N_tx, 2) - magnitude and phase channels
        self.conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')
        self.conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.pool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(target_dim, activation='linear')
    
    def call(self, channel_input):
        """
        Args:
            channel_input: (batch, K, N_tx, 2)
        Returns:
            encoded: (batch, target_dim)
        """
        x = self.conv1(channel_input)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
