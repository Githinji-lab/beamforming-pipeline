import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
results_dir = os.path.join(project_root, 'results')
data_dir = os.path.join(project_root, 'data')

def plot_baseline_comparison(results_path=None, save_path=None):
    """Plot comprehensive baseline comparison."""

    if results_path is None:
        results_path = os.path.join(results_dir, 'baseline_comparison.pkl')
    if save_path is None:
        save_path = os.path.join(results_dir, 'baseline_comparison.png')
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    methods = list(results.keys())
    capacities_mean = [np.mean(results[m]['capacities']) for m in methods]
    capacities_std = [np.std(results[m]['capacities']) for m in methods]
    times_mean = [np.mean(results[m]['times']) * 1000 for m in methods]  # ms
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Capacity comparison
    axes[0].bar(range(len(methods)), capacities_mean, 
                yerr=capacities_std, capsize=5, alpha=0.7)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylabel('Avg Sum Capacity (bps/Hz)')
    axes[0].set_title('Capacity Comparison (with std dev)')
    axes[0].grid(True, alpha=0.3)
    
    # Latency comparison
    axes[1].bar(range(len(methods)), times_mean, alpha=0.7, color='orange')
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylabel('Avg Inference Time (ms)')
    axes[1].set_title('Latency Comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Baseline comparison plot saved to {save_path}")
    plt.show()


def plot_capacity_vs_snr(data_path=None, save_path=None):
    """Plot capacity vs SNR for different methods."""

    if data_path is None:
        data_path = os.path.join(data_dir, 'training_data.pkl')
    if save_path is None:
        save_path = os.path.join(results_dir, 'capacity_vs_snr.png')
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    snr_values = sorted(np.unique(data['snr']))
    caps_mmse = []
    
    for snr in snr_values:
        mask = data['snr'] == snr
        avg_cap = np.mean(data['capacity'][mask]) if 'capacity' in data else 0
        caps_mmse.append(avg_cap)
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, caps_mmse, 'o-', linewidth=2, markersize=8, label='MMSE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Sum Capacity (bps/Hz)')
    plt.title('Capacity vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Capacity vs SNR plot saved to {save_path}")
    plt.show()


def create_comparison_summary_table(results_path=None, output_path=None):
    """Create text summary table of baseline comparison."""

    if results_path is None:
        results_path = os.path.join(results_dir, 'baseline_comparison.pkl')
    if output_path is None:
        output_path = os.path.join(results_dir, 'baseline_summary.txt')
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Method':<20} {'Avg Capacity':<15} {'Std Dev':<12} {'Latency (ms)':<15}\n")
        f.write("-"*80 + "\n")
        
        for method in results.keys():
            caps = results[method]['capacities']
            times = results[method]['times']
            
            mean_cap = np.mean(caps)
            std_cap = np.std(caps)
            mean_time = np.mean(times) * 1000
            
            f.write(f"{method:<20} {mean_cap:<15.4f} {std_cap:<12.4f} {mean_time:<15.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("- Avg Capacity: Higher is better (bps/Hz)\n")
        f.write("- Std Dev: Lower is better (more stable)\n")
        f.write("- Latency: Lower is better (faster inference)\n")
        f.write("="*80 + "\n")
    
    print(f"Summary table saved to {output_path}")


if __name__ == "__main__":
    print("Generating comparison plots...\n")
    plot_baseline_comparison()
    plot_capacity_vs_snr()
    create_comparison_summary_table()
