import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
from tensorflow import keras
import time
import pickle

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from agents import SACAgent, patch_sac_agent
from utils import ReplayBuffer
from preprocessing import preprocess_channel, reconstruct_complex_weights
from state_encoder import ChannelStateEncoder, BeamCodebook
from baselines import calculate_multi_objective_reward, select_teacher_beam_index
from domain_randomization import DomainRandomizer

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
    'latency_budget_ms': 1.0,
    'latency_budget_weight': 0.8,
    'imitation_samples': 300,
    'imitation_epochs': 5,
    'distill_epochs': 5,
    'distill_samples': 300,
    'codebook_keep_ratio': 0.35,
    'batch_size': 64,
    'num_episodes': 100,
    'max_timesteps_per_episode': 50
}

AGENT_HYPERPARAM_KEYS = {
    'gamma', 'tau', 'actor_lr', 'critic_lr', 'alpha_lr', 'reward_scale', 'target_entropy'
}

def train_improved_sac_agent(use_domain_randomization=True,
                             use_multi_objective_reward=True,
                             codebook_strategy="teacher_top"):
    """Train SAC agent with improved state encoding and discrete actions."""
    
    print("="*60)
    print("IMPROVED SAC TRAINING")
    print("="*60)
    print(f"State encoding: {ENCODED_STATE_DIM}-dim (PCA compressed)")
    print(f"Action space: Discrete {NUM_BEAMS} beams from codebook")
    print(f"Codebook strategy: {codebook_strategy}")
    print(f"Codebook keep ratio: {hyperparams['codebook_keep_ratio']}")
    print(f"Multi-objective reward: {use_multi_objective_reward}")
    print(f"Domain randomization: {use_domain_randomization}")
    print(f"Latency budget: {hyperparams['latency_budget_ms']} ms")
    print("="*60 + "\n")
    
    # Initialize simulator and codebook
    simulator = BeamformingSimulatorV4(N_tx=TARGET_N_TX, K=TARGET_K)
    domain_randomizer = DomainRandomizer(simulator) if use_domain_randomization else None
    
    # Generate beam codebook
    print("Generating beam codebook...")
    codebook = BeamCodebook(N_tx=TARGET_N_TX, K=TARGET_K, num_beams=NUM_BEAMS)
    codebook.generate_codebook(
        simulator,
        num_samples=1200,
        strategy=codebook_strategy,
        teacher_keep_ratio=hyperparams['codebook_keep_ratio']
    )
    
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
    agent_hyperparams = {k: v for k, v in hyperparams.items() if k in AGENT_HYPERPARAM_KEYS}
    
    # Initialize SAC Agent (action_dim = NUM_BEAMS for discrete selection)
    sac_agent = SACAgent(
        state_dim=ENCODED_STATE_DIM,
        action_dim=NUM_BEAMS,
        replay_buffer=replay_buffer,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        **agent_hyperparams
    )
    
    patch_sac_agent(sac_agent)

    def build_teacher_targets(num_samples):
        state_batch = []
        target_batch = []
        for _ in range(num_samples):
            H = simulator.generate_channel_matrix_v4()
            snr_local = np.random.choice(simulator.snr_db_list)
            encoded = state_encoder.encode(H, snr_local)

            teacher_idx = select_teacher_beam_index(H, simulator, codebook)
            teacher_target = -np.ones(NUM_BEAMS, dtype=np.float32)
            teacher_target[teacher_idx] = 1.0

            state_batch.append(encoded)
            target_batch.append(teacher_target)
        return np.array(state_batch, dtype=np.float32), np.array(target_batch, dtype=np.float32)

    print("Running imitation warm-start...")
    imitation_states, imitation_targets = build_teacher_targets(hyperparams['imitation_samples'])
    imitation_states = scaler_X.fit_transform(imitation_states)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    dataset = tf.data.Dataset.from_tensor_slices((imitation_states, imitation_targets)).shuffle(512).batch(32)
    for epoch in range(hyperparams['imitation_epochs']):
        epoch_loss = 0.0
        batch_count = 0
        for states_b, targets_b in dataset:
            with tf.GradientTape() as tape:
                mean_pred, _ = sac_agent.actor_model(states_b, training=True)
                mean_pred = tf.tanh(mean_pred)
                loss = tf.reduce_mean(tf.square(mean_pred - targets_b))
            grads = tape.gradient(loss, sac_agent.actor_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, sac_agent.actor_model.trainable_variables))
            epoch_loss += float(loss)
            batch_count += 1
        print(f"Imitation epoch {epoch+1}/{hyperparams['imitation_epochs']} - loss: {epoch_loss/max(batch_count,1):.4f}")
    
    rewards_history = []
    latency_history_ms = []
    
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
            infer_start = time.perf_counter()
            action_logits = sac_agent.choose_action(current_state_scaled, evaluate=False)
            inference_latency_ms = (time.perf_counter() - infer_start) * 1000.0
            action_beam_idx = np.argmax(action_logits)  # Select beam with highest logit
            
            # Get beam from codebook
            W = codebook.get_beam(action_beam_idx)
            
            # Calculate multi-objective reward if enabled
            if use_multi_objective_reward:
                reward, metrics = calculate_multi_objective_reward(
                    H_current, W, simulator,
                    alpha=0.6, beta=0.2, gamma=0.1,
                    inference_latency_ms=inference_latency_ms,
                    latency_budget_ms=hyperparams['latency_budget_ms'],
                    latency_budget_weight=hyperparams['latency_budget_weight']
                )
            else:
                reward = simulator.calculate_sum_capacity(H_current, W)
            
            episode_reward += reward
            latency_history_ms.append(inference_latency_ms)
            
            # Generate next state
            if domain_randomizer is not None and np.random.rand() < 0.3:
                H_next = domain_randomizer.generate_randomized_channel()
            else:
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
            p95_latency = np.percentile(latency_history_ms[-hyperparams['max_timesteps_per_episode']:], 95) if latency_history_ms else 0.0
            print(f"Episode {episode + 1:3d}, Avg Reward: {avg_reward:.3f}, P95 latency: {p95_latency:.3f} ms")
    
    print("\nTraining complete!")
    
    # Save models
    results_dir = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    sac_agent.actor_model.save(os.path.join(results_dir, 'improved_sac_actor_model.h5'))
    sac_agent.actor_model.save(os.path.join(results_dir, 'improved_sac_actor_model.keras'))
    np.save(os.path.join(results_dir, 'improved_sac_rewards.npy'), rewards_history)
    np.save(os.path.join(results_dir, 'improved_sac_latency_ms.npy'), np.array(latency_history_ms, dtype=np.float32))

    def build_student_model(input_dim, output_dim):
        inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(output_dim, activation='tanh')(x)
        return keras.Model(inputs, outputs, name='distilled_actor_student')

    print("Running policy distillation to lightweight student model...")
    distill_states = []
    for _ in range(hyperparams['distill_samples']):
        H = simulator.generate_channel_matrix_v4()
        snr_local = np.random.choice(simulator.snr_db_list)
        distill_states.append(state_encoder.encode(H, snr_local))
    distill_states = scaler_X.transform(np.array(distill_states, dtype=np.float32))

    teacher_mean, _ = sac_agent.actor_model(distill_states, training=False)
    teacher_targets = tf.tanh(teacher_mean).numpy()

    student_model = build_student_model(ENCODED_STATE_DIM, NUM_BEAMS)
    student_model.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse')
    student_model.fit(distill_states, teacher_targets, epochs=hyperparams['distill_epochs'], batch_size=32, verbose=0)

    student_model.save(os.path.join(results_dir, 'improved_sac_student.keras'))

    converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(results_dir, 'improved_sac_student_int8.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # Save codebook
    with open(os.path.join(results_dir, 'beam_codebook.pkl'), 'wb') as f:
        pickle.dump({
            'codebook': codebook,
            'state_encoder': state_encoder,
            'num_beams': NUM_BEAMS,
            'encoded_state_dim': ENCODED_STATE_DIM,
            'hyperparams': hyperparams
        }, f)
    
    print("Models saved to results/")
    
    # Plot rewards (robust for short runs/headless mode)
    rewards_arr = np.array(rewards_history, dtype=float)
    rewards_arr = np.nan_to_num(rewards_arr, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(rewards_arr) + 1)

    if len(rewards_arr) == 1:
        ax.scatter(x, rewards_arr, s=80, color='tab:blue', label='Avg Reward')
    else:
        ax.plot(x, rewards_arr, linewidth=2, marker='o', markersize=3, label='Avg Reward')

    ax.set_title('Improved SAC Training Rewards (PCA State + Discrete Actions)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if len(rewards_arr) == 1:
        y = rewards_arr[0]
        delta = max(abs(y) * 0.1, 1e-3)
        ax.set_ylim(y - delta, y + delta)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'improved_sac_training.png'), dpi=150)
    plt.close(fig)
    
    return sac_agent, rewards_history, codebook, state_encoder


if __name__ == "__main__":
    agent, rewards, codebook, encoder = train_improved_sac_agent(
        use_domain_randomization=True,
        use_multi_objective_reward=True,
        codebook_strategy="teacher_top"
    )
