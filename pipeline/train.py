import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys

# Add src to path
# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from agents import SACAgent, patch_sac_agent
from utils import ReplayBuffer
from preprocessing import preprocess_channel, reconstruct_complex_weights

# Constants
TARGET_N_TX = 8
TARGET_K = 4

# Training hyperparameters
hyperparams = {
    'gamma': 0.99,
    'tau': 0.005,
    'actor_lr': 0.0003,
    'critic_lr': 0.0003,
    'alpha_lr': 0.0003,
    'reward_scale': 1.0,
    'target_entropy': -float(TARGET_N_TX * TARGET_K * 2),
    'batch_size': 64,
    'num_episodes': 100,
    'max_timesteps_per_episode': 50
}

def train_sac_agent():
    state_dim = (TARGET_K * TARGET_N_TX * 2) + 1
    action_dim = TARGET_N_TX * TARGET_K * 2

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(np.random.randn(100, state_dim))
    scaler_y.fit(np.random.randn(100, action_dim))

    replay_buffer = ReplayBuffer()

    # Initialize SAC Agent
    sac_agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        alpha_lr=hyperparams['alpha_lr'],
        reward_scale=hyperparams['reward_scale'],
        target_entropy=hyperparams['target_entropy']
    )

    # Patch if necessary
    patch_sac_agent(sac_agent)

    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)
    rewards_history = []

    for episode in range(hyperparams['num_episodes']):
        H_current = simulator.generate_channel_matrix_v4()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]
        current_state_raw = preprocess_channel(H_current, snr, TARGET_N_TX, TARGET_K)
        current_state_scaled = sac_agent.scaler_X.transform(current_state_raw.reshape(1, -1))[0]
        episode_reward = 0

        for t in range(hyperparams['max_timesteps_per_episode']):
            action_scaled = sac_agent.choose_action(current_state_scaled, evaluate=False)
            action_W = reconstruct_complex_weights(action_scaled, TARGET_N_TX, TARGET_K, simulator, sac_agent.scaler_y)
            reward = simulator.calculate_sum_capacity(H_current, action_W)
            episode_reward += reward

            H_next = simulator.generate_channel_matrix_v4()
            next_state_raw = preprocess_channel(H_next, snr, TARGET_N_TX, TARGET_K)
            next_state_scaled = sac_agent.scaler_X.transform(next_state_raw.reshape(1, -1))[0]
            done = (t == hyperparams['max_timesteps_per_episode'] - 1)

            sac_agent.replay_buffer.add(current_state_scaled, action_scaled, reward, next_state_scaled, done)

            if len(sac_agent.replay_buffer) > hyperparams['batch_size']:
                sac_agent.learn(hyperparams['batch_size'])
                sac_agent.update_target_networks()

            current_state_scaled = next_state_scaled
            H_current = H_next

        avg_reward = episode_reward / hyperparams['max_timesteps_per_episode']
        rewards_history.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

    return sac_agent, rewards_history

if __name__ == "__main__":
    trained_agent, rewards = train_sac_agent()

    # Save the trained agent
    trained_agent.actor_model.save('results/sac_actor_model.h5')
    np.save('results/sac_rewards.npy', rewards)

    # Plot rewards
    plt.plot(rewards)
    plt.title("SAC Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.savefig('results/sac_training_plot.png')
    plt.show()