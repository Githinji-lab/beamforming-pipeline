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
from baselines import (
    calculate_multi_objective_reward,
    calculate_constrained_quality_reward,
    select_teacher_beam_index,
)
from domain_randomization import DomainRandomizer
from dqn_beam_agent import DQNBeamAgent
from dataset_ingestion import ingest_dataset_zips
from external_dataset import load_channels_from_registry, ExternalChannelSampler
from phase1_state import Phase1StateAugmenter


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
    dataset_zip_paths=None,
    channel_source="simulator",
    external_registry_path="data/dataset_registry.json",
    external_max_samples=20000,
    external_mix_ratio=0.5,
    phase1_enable=False,
    phase1_num_clusters=0,
    reward_mode="legacy",
    learning_rate=3e-4,
    gamma_rl=0.99,
    target_update_tau=0.01,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.996,
    replay_capacity=50000,
    dueling_dqn=False,
    prioritized_replay=False,
    priority_alpha=0.6,
    priority_beta_start=0.4,
    priority_beta_increment=1e-4,
    priority_eps=1e-6,
    reward_alpha=0.7,
    reward_beta=0.15,
    reward_gamma=0.1,
    constrained_cap_weight=0.8,
    constrained_sinr_weight=0.25,
    constrained_ber_weight=0.12,
    constrained_latency_weight=1.0,
):
    if dataset_zip_paths:
        ingest_info = ingest_dataset_zips(
            zip_paths=dataset_zip_paths,
            output_root="data/external",
            manifest_path="data/dataset_registry.json",
        )
        print(
            f"Ingested {len(ingest_info['archives'])} zip archive(s), "
            f"discovered {len(ingest_info['dataset_files'])} dataset file(s)."
        )

    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    randomizer = DomainRandomizer(simulator) if use_domain_randomization else None
    external_sampler = None

    if channel_source in ("external", "mixed"):
        channels = load_channels_from_registry(
            registry_path=external_registry_path,
            target_k=simulator.K,
            target_n_tx=simulator.N_tx,
            max_total_samples=external_max_samples,
        )
        external_sampler = ExternalChannelSampler(channels)
        print(f"Loaded external channels: {channels.shape[0]}")

    def sample_channel(domain_randomized=False):
        if channel_source == "external":
            return external_sampler.sample()
        if channel_source == "mixed":
            if np.random.rand() < external_mix_ratio:
                return external_sampler.sample()
            if domain_randomized and randomizer is not None and np.random.rand() < 0.3:
                return randomizer.generate_randomized_channel()
            return simulator.generate_channel_matrix_v4()
        if domain_randomized and randomizer is not None and np.random.rand() < 0.3:
            return randomizer.generate_randomized_channel()
        return simulator.generate_channel_matrix_v4()

    print("=" * 60)
    print("DQN BEAM TRAINING (DISCRETE ACTIONS)")
    print("=" * 60)
    print(f"num_beams={num_beams}, encoded_dim={encoded_dim}")
    print(f"codebook_strategy={codebook_strategy}")
    print(f"codebook_keep_ratio={codebook_keep_ratio}")
    print(f"channel_source={channel_source}")
    print(f"latency budget={latency_budget_ms} ms")
    print(f"reward_mode={reward_mode}")
    print(f"dueling_dqn={dueling_dqn}, prioritized_replay={prioritized_replay}")

    codebook = BeamCodebook(N_tx=8, K=4, num_beams=num_beams)
    codebook.generate_codebook(
        simulator,
        num_samples=1500,
        strategy=codebook_strategy,
        teacher_keep_ratio=codebook_keep_ratio,
    )

    encoder = ChannelStateEncoder(target_dim=encoded_dim)
    H_fit = np.array([sample_channel(domain_randomized=False) for _ in range(600)])
    snr_fit = np.random.choice(simulator.snr_db_list, 600)
    encoder.fit(H_fit, snr_fit)

    phase1_augmenter = Phase1StateAugmenter(
        enabled=phase1_enable,
        num_clusters=phase1_num_clusters,
    )
    phase1_augmenter.fit(H_fit)

    scaler = StandardScaler()
    fit_states = np.array(
        [
            phase1_augmenter.transform(
                base_state=encoder.encode(H_fit[i], snr_fit[i]),
                H=H_fit[i],
                snr=snr_fit[i],
                prev_H=None,
            )
            for i in range(len(H_fit))
        ],
        dtype=np.float32,
    )
    scaler.fit(fit_states)

    agent = DQNBeamAgent(
        state_dim=fit_states.shape[1],
        num_actions=num_beams,
        learning_rate=learning_rate,
        gamma=gamma_rl,
        target_update_tau=target_update_tau,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        replay_capacity=replay_capacity,
        dueling=dueling_dqn,
        prioritized_replay=prioritized_replay,
        priority_alpha=priority_alpha,
        priority_beta_start=priority_beta_start,
        priority_beta_increment=priority_beta_increment,
        priority_eps=priority_eps,
    )

    print("Running imitation warm-start for DQN...")
    im_states = []
    im_actions = []
    im_prev_h = None
    for _ in range(imitation_samples):
        H = sample_channel(domain_randomized=False)
        snr = np.random.choice(simulator.snr_db_list)
        im_states.append(
            phase1_augmenter.transform(
                base_state=encoder.encode(H, snr),
                H=H,
                snr=snr,
                prev_H=im_prev_h,
            )
        )
        im_actions.append(select_teacher_beam_index(H, simulator, codebook))
        im_prev_h = H
    im_states = scaler.transform(np.array(im_states, dtype=np.float32))
    agent.pretrain_imitation(im_states, np.array(im_actions, dtype=np.int32), epochs=imitation_epochs, batch_size=32)

    rewards_history = []
    p95_latency_history = []

    for ep in range(num_episodes):
        H = sample_channel(domain_randomized=False)
        snr = simulator.snr_db_list[len(simulator.snr_db_list) // 2]
        prev_h = None
        state_vec = phase1_augmenter.transform(
            base_state=encoder.encode(H, snr),
            H=H,
            snr=snr,
            prev_H=prev_h,
        )
        state = scaler.transform(state_vec.reshape(1, -1))[0]

        ep_reward = 0.0
        step_latencies = []

        for t in range(max_steps):
            t0 = time.perf_counter()
            action_idx = agent.act(state, evaluate=False)
            infer_latency_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies.append(infer_latency_ms)

            W = codebook.get_beam(action_idx)
            if reward_mode == "constrained":
                reward, _ = calculate_constrained_quality_reward(
                    H,
                    W,
                    simulator,
                    cap_weight=constrained_cap_weight,
                    sinr_weight=constrained_sinr_weight,
                    ber_weight=constrained_ber_weight,
                    latency_violation_weight=constrained_latency_weight,
                    latency_budget_ms=latency_budget_ms,
                    inference_latency_ms=infer_latency_ms,
                )
            else:
                reward, _ = calculate_multi_objective_reward(
                    H,
                    W,
                    simulator,
                    alpha=reward_alpha,
                    beta=reward_beta,
                    gamma=reward_gamma,
                    inference_latency_ms=infer_latency_ms,
                    latency_budget_ms=latency_budget_ms,
                    latency_budget_weight=latency_budget_weight,
                )

            H_next = sample_channel(domain_randomized=True)

            next_state_vec = phase1_augmenter.transform(
                base_state=encoder.encode(H_next, snr),
                H=H_next,
                snr=snr,
                prev_H=H,
            )
            next_state = scaler.transform(next_state_vec.reshape(1, -1))[0]
            done = (t == max_steps - 1)

            agent.replay_buffer.add(state, action_idx, reward, next_state, float(done))
            agent.train_on_batch(batch_size=batch_size)

            state = next_state
            prev_h = H
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
                "phase1_augmenter": phase1_augmenter,
                "phase1_enable": bool(phase1_enable),
                "phase1_num_clusters": int(phase1_num_clusters),
                "reward_mode": reward_mode,
                "learning_rate": learning_rate,
                "gamma_rl": gamma_rl,
                "target_update_tau": target_update_tau,
                "epsilon_start": epsilon_start,
                "epsilon_min": epsilon_min,
                "epsilon_decay": epsilon_decay,
                "replay_capacity": replay_capacity,
                "dueling_dqn": bool(dueling_dqn),
                "prioritized_replay": bool(prioritized_replay),
                "priority_alpha": float(priority_alpha),
                "priority_beta_start": float(priority_beta_start),
                "priority_beta_increment": float(priority_beta_increment),
                "priority_eps": float(priority_eps),
                "reward_alpha": reward_alpha,
                "reward_beta": reward_beta,
                "reward_gamma": reward_gamma,
                "constrained_cap_weight": constrained_cap_weight,
                "constrained_sinr_weight": constrained_sinr_weight,
                "constrained_ber_weight": constrained_ber_weight,
                "constrained_latency_weight": constrained_latency_weight,
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
    p.add_argument(
        "--dataset-zips",
        type=str,
        default="",
        help="Comma-separated zip archive paths to ingest before training.",
    )
    p.add_argument("--channel-source", type=str, default="simulator", choices=["simulator", "external", "mixed"])
    p.add_argument("--external-registry", type=str, default="data/dataset_registry.json")
    p.add_argument("--external-max-samples", type=int, default=20000)
    p.add_argument("--external-mix-ratio", type=float, default=0.5)
    p.add_argument("--phase1-enable", action="store_true")
    p.add_argument("--phase1-num-clusters", type=int, default=0)
    p.add_argument("--reward-mode", type=str, default="legacy", choices=["legacy", "constrained"])
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma-rl", type=float, default=0.99)
    p.add_argument("--target-update-tau", type=float, default=0.01)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.996)
    p.add_argument("--replay-capacity", type=int, default=50000)
    p.add_argument("--dueling-dqn", action="store_true")
    p.add_argument("--prioritized-replay", action="store_true")
    p.add_argument("--priority-alpha", type=float, default=0.6)
    p.add_argument("--priority-beta-start", type=float, default=0.4)
    p.add_argument("--priority-beta-increment", type=float, default=1e-4)
    p.add_argument("--priority-eps", type=float, default=1e-6)
    p.add_argument("--reward-alpha", type=float, default=0.7)
    p.add_argument("--reward-beta", type=float, default=0.15)
    p.add_argument("--reward-gamma", type=float, default=0.1)
    p.add_argument("--constrained-cap-weight", type=float, default=0.8)
    p.add_argument("--constrained-sinr-weight", type=float, default=0.25)
    p.add_argument("--constrained-ber-weight", type=float, default=0.12)
    p.add_argument("--constrained-latency-weight", type=float, default=1.0)
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
        dataset_zip_paths=[z.strip() for z in args.dataset_zips.split(",") if z.strip()],
        channel_source=args.channel_source,
        external_registry_path=args.external_registry,
        external_max_samples=args.external_max_samples,
        external_mix_ratio=args.external_mix_ratio,
        phase1_enable=args.phase1_enable,
        phase1_num_clusters=args.phase1_num_clusters,
        reward_mode=args.reward_mode,
        learning_rate=args.learning_rate,
        gamma_rl=args.gamma_rl,
        target_update_tau=args.target_update_tau,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        replay_capacity=args.replay_capacity,
        dueling_dqn=args.dueling_dqn,
        prioritized_replay=args.prioritized_replay,
        priority_alpha=args.priority_alpha,
        priority_beta_start=args.priority_beta_start,
        priority_beta_increment=args.priority_beta_increment,
        priority_eps=args.priority_eps,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        constrained_cap_weight=args.constrained_cap_weight,
        constrained_sinr_weight=args.constrained_sinr_weight,
        constrained_ber_weight=args.constrained_ber_weight,
        constrained_latency_weight=args.constrained_latency_weight,
    )
