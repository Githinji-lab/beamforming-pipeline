import numpy as np
import pickle
import os
import sys

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from preprocessing import calculate_mmse_weights_adjusted

def generate_training_data(num_samples=10000, save_path='data/training_data.pkl'):
    simulator = BeamformingSimulatorV4()

    data = {
        'H': [],
        'W_mmse': [],
        'capacity': [],
        'snr': []
    }

    print(f"Generating {num_samples} training samples...")

    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

        H = simulator.generate_channel_matrix_v4()
        snr = np.random.choice(simulator.snr_db_list)

        # Generate MMSE weights as target
        W_mmse = calculate_mmse_weights_adjusted(H, simulator)
        capacity = simulator.calculate_sum_capacity(H, W_mmse)

        data['H'].append(H)
        data['W_mmse'].append(W_mmse)
        data['capacity'].append(capacity)
        data['snr'].append(snr)

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    # Save data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {save_path}")
    print(f"Dataset shape: H: {data['H'].shape}, Capacity: {data['capacity'].shape}")

if __name__ == "__main__":
    generate_training_data()