import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
from tensorflow import keras

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from agents import SACAgent, patch_sac_agent
from utils import ReplayBuffer
from preprocessing import preprocess_channel, reconstruct_complex_weights
from state_encoder import ChannelStateEncoder, BeamCodebook
from baselines import calculate_multi_objective_reward
from domain_randomization import create_augmented_training_data

# Constants
TARGET_N_TX = 8
TARGET_K = 4
ENCODED_STATE_DIM = 8
NUM_BEAMS = 32

# Training hyperparameters
hyperparams = {
    'gamma': 0.99,
    'tau': 0.005,
    'actor_lr': 0.0003,
    'critic_lr': 0.0003,
    'alpha_lr': 0.0003,
    'reward_scale': 1.0,
    'target_entropy': -float(ENCODED_STATE_DIM),
    'batch_size': 64,
    'num_episodes': 100,
    'max_timesteps_per_episode': 50
}

def train_improved_sac_agent(use_domain_randomization=True, 
                             use_multi_objective_reward=True):
    """Train SAC agent with improved state encoding and discrete actions."""
    
    print("="*60)
    print("IMPROVED SAC TRAINING")
    print("="*60)
    print(f"State encoding: {ENCODED_STATE_DIM}-dim (PCA compressed)")
    print(f"Action space: Discrete {NUM_BEAMS} beams from codebook")
    print(f"Multi-objective reward: {use_multi_objective_reward}")
    print(f"Domain randomization: {use_domain_randomization}")
    print("="*60 + "\n")
    
    # Initialize simulator and codebook
    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)
    
    # Generate beam codebook
    print("Generating beam codebook...")
    codebook = BeamCodebook(N_tx=TARGET_N_TX, K=TARGET_K, num_beams=NUM_BEAMS)
    codebook.generate_codebook(simulator, num_samples=1000)
    
    # Create state encoder
    print("Fitting state encoder (PCA)...")
    state_encoder = ChannelStateEncoder(target_dim=ENCODED_STATE_DIM)
    
    # Generate sample data to fit encoder
    H_samples = np.array([simulator.generate_channel_matrix_v4() for _ in range(500)])
    snr_samples = np.random.choice(simulator.snr_db_list, 500)
    state_encoder.fit(H_samples, snr_samples)
    
    # Initialize scalers for action reconstruction
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X.fit(np.random.randn(100, ENCODED_STATE_DIM))
    scaler_y.fit(np.random.randn(100, NUM_BEAMS))
    
    replay_buffer = ReplayBuffer()
    
    # Initialize SAC Agent (action_dim = NUM_BEAMS for discrete selection)
    sac_agent = SACAgent(
        state_dim=ENCODED_STATE_DIM,
        action_dim=NUM_BEAMS,
        replay_buffer=replay_buffer,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        **hyperparams
    )
    
    patch_sac_agent(sac_agent)
    
    rewards_history = []
    
    print("Starting training loop...\n")
    
    for episode in range(hyperparams['num_episodes']):
        H_current = simulator.generate_channel_matrix_v4()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]
        
        # Encode state
        current_state_encoded = state_encoder.encode(H_current, snr)
        current_state_scaled = scaler_X.transform(current_state_encoded.reshape(1, -1))[0]
        
        episode_reward = 0
        
        for t in range(hyperparams['max_timesteps_per_episode']):
            # Choose action (beam index)
            action_logits = sac_agent.choose_action(current_state_scaled, evaluate=False)
            action_beam_idx = np.argmax(action_logits)  # Select beam with highest logit
            
            # Get beam from codebook
            W = codebook.get_beam(action_beam_idx)
            
            # Calculate multi-objective reward if enabled
            if use_multi_objective_reward:
                reward, metrics = calculate_multi_objective_reward(
                    H_current, W, simulator,
                    alpha=0.6, beta=0.3, gamma=0.1
                )
            else:
                reward = simulator.calculate_sum_capacity(H_current, W)
            
            episode_reward += reward
            
            # Generate next state
            H_next = simulator.generate_channel_matrix_v4()
            next_state_encoded = state_encoder.encode(H_next, snr)
            next_state_scaled = scaler_X.transform(next_state_encoded.reshape(1, -1))[0]
            
            done = (t == hyperparams['max_timesteps_per_episode'] - 1)
            
            # Store in replay buffer
            sac_agent.replay_buffer.add(
                current_state_scaled, 
                action_logits,  # Store action logits
                reward, 
                next_state_scaled, 
                done
            )
            
            # Learn from batch
            if len(sac_agent.replay_buffer) > hyperparams['batch_size']:
                sac_agent.learn(hyperparams['batch_size'])
                sac_agent.update_target_networks()
            
            current_state_scaled = next_state_scaled
            H_current = H_next
        
        avg_reward = episode_reward / hyperparams['max_timesteps_per_episode']
        rewards_history.append(avg_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:3d}, Avg Reward: {avg_reward:.3f}")
    
    print("\nTraining complete!")
    
    # Save models
    sac_agent.actor_model.save('../results/improved_sac_actor_model.h5')
    np.save('../results/improved_sac_rewards.npy', rewards_history)
    
    # Save codebook
    with open('../results/beam_codebook.pkl', 'wb') as f:
        import pickle
        pickle.dump({
            'codebook': codebook,
            'state_encoder': state_encoder,
            'num_beams': NUM_BEAMS,
            'encoded_state_dim': ENCODED_STATE_DIM
        }, f)
    
    print("Models saved to results/")
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, linewidth=2)
    plt.title('Improved SAC Training Rewards (PCA State + Discrete Actions)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/improved_sac_training.png', dpi=150)
    plt.show()
    
    return sac_agent, rewards_history, codebook, state_encoder


if __name__ == "__main__":
    agent, rewards, codebook, encoder = train_improved_sac_agent(
        use_domain_randomization=True,
        use_multi_objective_reward=True
    )
