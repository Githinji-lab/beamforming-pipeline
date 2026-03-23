import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Add src to path (for potential imports)
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

def plot_training_rewards(rewards_path='results/sac_rewards.npy'):
    rewards = np.load(rewards_path)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='SAC Agent')
    plt.title('Training Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (bps/Hz)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results/training_rewards.png')
    plt.show()

def plot_evaluation_comparison(ml_cap, mmse_cap, save_path='results/evaluation_comparison.png'):
    methods = ['SAC Agent', 'MMSE']
    capacities = [ml_cap, mmse_cap]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, capacities, color=['blue', 'orange'])
    plt.ylabel('Average Sum Capacity (bps/Hz)')
    plt.title('Beamforming Performance Comparison')
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, cap in zip(bars, capacities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{cap:.2f}', ha='center', va='bottom')

    plt.savefig(save_path)
    plt.show()

def visualize_channel_distribution(data_path='data/training_data.pkl'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    H_real = np.real(data['H']).flatten()
    H_imag = np.imag(data['H']).flatten()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(H_real, bins=50, alpha=0.7, label='Real')
    plt.title('Channel Real Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(H_imag, bins=50, alpha=0.7, label='Imaginary', color='orange')
    plt.title('Channel Imaginary Part Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/channel_distribution.png')
    plt.show()

def plot_capacity_vs_snr(data_path='data/training_data.pkl'):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    snr_values = np.unique(data['snr'])
    avg_capacities = []

    for snr in snr_values:
        mask = data['snr'] == snr
        avg_cap = np.mean(data['capacity'][mask])
        avg_capacities.append(avg_cap)

    plt.figure(figsize=(8, 6))
    plt.plot(snr_values, avg_capacities, 'o-', linewidth=2, markersize=8)
    plt.title('Average Capacity vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Sum Capacity (bps/Hz)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/capacity_vs_snr.png')
    plt.show()

if __name__ == "__main__":
    # Example usage - uncomment as needed
    # plot_training_rewards()
    # visualize_channel_distribution()
    # plot_capacity_vs_snr()
    print("Visualization functions defined. Call them as needed.")