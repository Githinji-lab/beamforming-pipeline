import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

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

def evaluate_agent_performance(agent, num_eval_iterations=200):
    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)

    caps_ml_agent = []
    caps_mmse = []

    snr_for_eval = simulator.snr_db_list[len(simulator.snr_db_list) // 2]

    print(f"Starting evaluation for {num_eval_iterations} iterations...")

    for i in range(num_eval_iterations):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{num_eval_iterations} iterations")

        H = simulator.generate_channel_matrix_v4()
        state_eval_raw = preprocess_channel(H, snr_for_eval, TARGET_N_TX, TARGET_K)
        state_eval_scaled = agent.scaler_X.transform(state_eval_raw.reshape(1, -1))[0]

        # ML Agent
        action_eval_scaled = agent.choose_action(state_eval_scaled, evaluate=True)
        action_eval_W = reconstruct_complex_weights(action_eval_scaled, TARGET_N_TX, TARGET_K, simulator, agent.scaler_y)
        cap_ml = simulator.calculate_sum_capacity(H, action_eval_W)
        caps_ml_agent.append(cap_ml)

        # MMSE
        W_mmse = calculate_mmse_weights_adjusted(H, simulator)
        cap_mmse = simulator.calculate_sum_capacity(H, W_mmse)
        caps_mmse.append(cap_mmse)

    avg_cap_ml = np.mean(caps_ml_agent)
    avg_cap_mmse = np.mean(caps_mmse)

    print(f"Average Capacity - ML Agent: {avg_cap_ml:.2f} bps/Hz")
    print(f"Average Capacity - MMSE: {avg_cap_mmse:.2f} bps/Hz")

    # Plot comparison
    methods = ['ML Agent', 'MMSE']
    capacities = [avg_cap_ml, avg_cap_mmse]

    plt.bar(methods, capacities)
    plt.ylabel('Average Sum Capacity (bps/Hz)')
    plt.title('Beamforming Performance Comparison')
    plt.savefig('results/evaluation_comparison.png')
    plt.show()

    return avg_cap_ml, avg_cap_mmse

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

    evaluate_agent_performance(agent)