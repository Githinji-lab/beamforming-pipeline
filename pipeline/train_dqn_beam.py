import os
import sys
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4
from state_encoder import ChannelStateEncoder, BeamCodebook
from baselines import calculate_multi_objective_reward, select_teacher_beam_index
from domain_randomization import DomainRandomizer
from dqn_beam_agent import DQNBeamAgent


def train_dqn_beam(
    num_episodes=120,
    max_steps=50,
    batch_size=64,
    num_beams=24,
    encoded_dim=8,
    imitation_samples=500,
    imitation_epochs=6,
    latency_budget_ms=1.0,
    latency_budget_weight=0.8,
    use_domain_randomization=True,
    codebook_strategy="teacher_top",
    codebook_keep_ratio=0.35,
):
    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    randomizer = DomainRandomizer(simulator) if use_domain_randomization else None

    print("=" * 60)
    print("DQN BEAM TRAINING (DISCRETE ACTIONS)")
    print("=" * 60)
    print(f"num_beams={num_beams}, encoded_dim={encoded_dim}")
    print(f"codebook_strategy={codebook_strategy}")
    print(f"codebook_keep_ratio={codebook_keep_ratio}")
    print(f"latency budget={latency_budget_ms} ms")

    codebook = BeamCodebook(N_tx=8, K=4, num_beams=num_beams)
    codebook.generate_codebook(
        simulator,
        num_samples=1500,
        strategy=codebook_strategy,
        teacher_keep_ratio=codebook_keep_ratio,
    )

    encoder = ChannelStateEncoder(target_dim=encoded_dim)
    H_fit = np.array([simulator.generate_channel_matrix_v4() for _ in range(600)])
    snr_fit = np.random.choice(simulator.snr_db_list, 600)
    encoder.fit(H_fit, snr_fit)

    scaler = StandardScaler()
    fit_states = np.array([encoder.encode(H_fit[i], snr_fit[i]) for i in range(len(H_fit))], dtype=np.float32)
    scaler.fit(fit_states)

    agent = DQNBeamAgent(
        state_dim=encoded_dim,
        num_actions=num_beams,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_tau=0.01,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.996,
    )

    print("Running imitation warm-start for DQN...")
    im_states = []
    im_actions = []
    for _ in range(imitation_samples):
        H = simulator.generate_channel_matrix_v4()
        snr = np.random.choice(simulator.snr_db_list)
        im_states.append(encoder.encode(H, snr))
        im_actions.append(select_teacher_beam_index(H, simulator, codebook))
    im_states = scaler.transform(np.array(im_states, dtype=np.float32))
    agent.pretrain_imitation(im_states, np.array(im_actions, dtype=np.int32), epochs=imitation_epochs, batch_size=32)

    rewards_history = []
    p95_latency_history = []

    for ep in range(num_episodes):
        H = simulator.generate_channel_matrix_v4()
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]
        state = scaler.transform(encoder.encode(H, snr).reshape(1, -1))[0]

        ep_reward = 0.0
        step_latencies = []

        for t in range(max_steps):
            t0 = time.perf_counter()
            action_idx = agent.act(state, evaluate=False)
            infer_latency_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(infer_latency_ms)

            W = codebook.get_beam(action_idx)
            reward, _ = calculate_multi_objective_reward(
                H,
                W,
                simulator,
                alpha=0.7,
                beta=0.15,
                gamma=0.1,
                inference_latency_ms=infer_latency_ms,
                latency_budget_ms=latency_budget_ms,
                latency_budget_weight=latency_budget_weight,
            )

            if randomizer is not None and np.random.rand() < 0.3:
                H_next = randomizer.generate_randomized_channel()
            else:
                H_next = simulator.generate_channel_matrix_v4()

            next_state = scaler.transform(encoder.encode(H_next, snr).reshape(1, -1))[0]
            done = (t == max_steps - 1)

            agent.replay_buffer.add(state, action_idx, reward, next_state, float(done))
            agent.train_on_batch(batch_size=batch_size)

            state = next_state
            H = H_next
            ep_reward += reward

        avg_ep_reward = ep_reward / max_steps
        rewards_history.append(avg_ep_reward)
        p95_latency = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
        p95_latency_history.append(p95_latency)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d}: avg_reward={avg_ep_reward:.3f}, p95_latency={p95_latency:.3f} ms, eps={agent.epsilon:.3f}")

    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    dqn_path = os.path.join(results_dir, "dqn_beam_model.keras")
    agent.q_net.save(dqn_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(agent.q_net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dqn_tflite = converter.convert()
    with open(os.path.join(results_dir, "dqn_beam_model_int8.tflite"), "wb") as f:
        f.write(dqn_tflite)

    with open(os.path.join(results_dir, "dqn_beam_artifacts.pkl"), "wb") as f:
        pickle.dump(
            {
                "codebook": codebook,
                "state_encoder": encoder,
                "state_scaler": scaler,
                "num_beams": num_beams,
                "encoded_dim": encoded_dim,
            },
            f,
        )

    np.save(os.path.join(results_dir, "dqn_beam_rewards.npy"), np.array(rewards_history, dtype=np.float32))
    np.save(os.path.join(results_dir, "dqn_beam_latency_p95_ms.npy"), np.array(p95_latency_history, dtype=np.float32))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, len(rewards_history) + 1)
    ax.plot(x, rewards_history, label="Avg Reward", linewidth=2)
    ax.plot(x, p95_latency_history, label="P95 Latency (ms)", linewidth=1.5)
    ax.set_title("DQN Beam Training Progress")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "dqn_beam_training.png"), dpi=150)
    plt.close(fig)

    print("DQN training complete. Artifacts saved in results/.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-beams", type=int, default=24)
    p.add_argument("--imitation-samples", type=int, default=500)
    p.add_argument("--imitation-epochs", type=int, default=6)
    p.add_argument("--latency-budget-ms", type=float, default=1.0)
    p.add_argument("--codebook-strategy", type=str, default="teacher_top", choices=["teacher_top", "teacher", "random"])
    p.add_argument("--codebook-keep-ratio", type=float, default=0.35)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn_beam(
        num_episodes=args.episodes,
        max_steps=args.steps,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        imitation_samples=args.imitation_samples,
        imitation_epochs=args.imitation_epochs,
        latency_budget_ms=args.latency_budget_ms,
        codebook_strategy=args.codebook_strategy,
        codebook_keep_ratio=args.codebook_keep_ratio,
    )
