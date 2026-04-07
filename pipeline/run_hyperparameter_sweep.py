import argparse
import csv
import itertools
import json
import os
from copy import deepcopy

from benchmark_optimized import benchmark
from train_dqn_beam import train_dqn_beam


DEFAULT_GRID = {
    "learning_rate": [1e-4, 3e-4],
    "epsilon_decay": [0.994, 0.996],
    "codebook_keep_ratio": [0.25, 0.35],
    "phase1_num_clusters": [2, 4],
    "dqn_rerank_topk": [1, 2],
}


def _parse_list(text, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _iter_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _score(summary, latency_budget_ms):
    d = summary["dqn_beam_tflite"]
    feasible = float(d["lat_mean_ms"] <= latency_budget_ms)
    capacity = float(d["cap_mean"])
    sinr = float(d["sinr_mean_db"])
    ber = float(d["ber_mean"])
    lat = float(d["lat_mean_ms"])
    lat_p95 = float(d["lat_p95_ms"])

    # Feasible runs rank ahead of infeasible ones.
    score = (
        feasible * 1000.0
        + capacity * 10.0
        + sinr
        - 50.0 * ber
        - 0.5 * lat
        - 0.25 * lat_p95
    )
    return score, feasible


def _write_csv(rows, out_csv):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Run DQN/Phase-1 hyperparameter sweep and rank configurations.")
    p.add_argument("--out-dir", type=str, default="results/hparam_sweep")
    p.add_argument("--train-episodes", type=int, default=12)
    p.add_argument("--train-steps", type=int, default=20)
    p.add_argument("--bench-iterations", type=int, default=30)
    p.add_argument("--channel-source", type=str, default="simulator", choices=["simulator", "external", "mixed"])
    p.add_argument("--external-registry", type=str, default="data/dataset_registry.json")
    p.add_argument("--external-max-samples", type=int, default=5000)
    p.add_argument("--external-mix-ratio", type=float, default=0.5)
    p.add_argument("--latency-budget-ms", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase1-enable", action="store_true")
    p.add_argument("--reward-mode", type=str, default="constrained", choices=["legacy", "constrained"])
    p.add_argument("--learning-rates", type=str, default="1e-4,3e-4")
    p.add_argument("--epsilon-decays", type=str, default="0.994,0.996")
    p.add_argument("--codebook-keep-ratios", type=str, default="0.25,0.35")
    p.add_argument("--phase1-clusters", type=str, default="2,4")
    p.add_argument("--topk-values", type=str, default="1,2")
    p.add_argument("--dqn-rerank-mode", type=str, default="capacity", choices=["capacity", "hybrid", "q_only"])
    p.add_argument("--dqn-hybrid-q-weight", type=float, default=0.5)
    p.add_argument("--reward-alpha", type=float, default=0.7)
    p.add_argument("--reward-beta", type=float, default=0.15)
    p.add_argument("--reward-gamma", type=float, default=0.1)
    p.add_argument("--constrained-cap-weight", type=float, default=0.8)
    p.add_argument("--constrained-sinr-weight", type=float, default=0.25)
    p.add_argument("--constrained-ber-weight", type=float, default=0.12)
    p.add_argument("--constrained-latency-weight", type=float, default=1.0)
    p.add_argument("--dueling-dqn", action="store_true")
    p.add_argument("--prioritized-replay", action="store_true")
    p.add_argument("--priority-alpha", type=float, default=0.6)
    p.add_argument("--priority-beta-start", type=float, default=0.4)
    p.add_argument("--priority-beta-increment", type=float, default=1e-4)
    p.add_argument("--priority-eps", type=float, default=1e-6)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grid = deepcopy(DEFAULT_GRID)
    grid["learning_rate"] = _parse_list(args.learning_rates, float)
    grid["epsilon_decay"] = _parse_list(args.epsilon_decays, float)
    grid["codebook_keep_ratio"] = _parse_list(args.codebook_keep_ratios, float)
    grid["phase1_num_clusters"] = _parse_list(args.phase1_clusters, int)
    grid["dqn_rerank_topk"] = _parse_list(args.topk_values, int)

    rows = []
    run_index = 0

    for combo in _iter_grid(grid):
        run_index += 1
        run_name = (
            f"run{run_index:03d}_lr{combo['learning_rate']}_eps{combo['epsilon_decay']}"
            f"_keep{combo['codebook_keep_ratio']}_cl{combo['phase1_num_clusters']}_topk{combo['dqn_rerank_topk']}"
        ).replace(".", "p")
        run_dir = os.path.join(args.out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        train_dqn_beam(
            num_episodes=args.train_episodes,
            max_steps=args.train_steps,
            batch_size=32,
            num_beams=24,
            imitation_samples=180,
            imitation_epochs=2,
            latency_budget_ms=args.latency_budget_ms,
            codebook_strategy="teacher_top",
            codebook_keep_ratio=float(combo["codebook_keep_ratio"]),
            channel_source=args.channel_source,
            external_registry_path=args.external_registry,
            external_max_samples=args.external_max_samples,
            external_mix_ratio=args.external_mix_ratio,
            phase1_enable=bool(args.phase1_enable),
            phase1_num_clusters=int(combo["phase1_num_clusters"]) if args.phase1_enable else 0,
            reward_mode=args.reward_mode,
            learning_rate=float(combo["learning_rate"]),
            epsilon_decay=float(combo["epsilon_decay"]),
            dueling_dqn=bool(args.dueling_dqn),
            prioritized_replay=bool(args.prioritized_replay),
            priority_alpha=float(args.priority_alpha),
            priority_beta_start=float(args.priority_beta_start),
            priority_beta_increment=float(args.priority_beta_increment),
            priority_eps=float(args.priority_eps),
            reward_alpha=args.reward_alpha,
            reward_beta=args.reward_beta,
            reward_gamma=args.reward_gamma,
            constrained_cap_weight=args.constrained_cap_weight,
            constrained_sinr_weight=args.constrained_sinr_weight,
            constrained_ber_weight=args.constrained_ber_weight,
            constrained_latency_weight=args.constrained_latency_weight,
        )

        json_out = os.path.join(run_dir, "benchmark_summary.json")
        _, summary = benchmark(
            num_iterations=args.bench_iterations,
            save_json_path=json_out,
            channel_source=args.channel_source,
            external_registry_path=args.external_registry,
            external_max_samples=args.external_max_samples,
            external_mix_ratio=args.external_mix_ratio,
            dqn_rerank_topk=int(combo["dqn_rerank_topk"]),
            dqn_rerank_mode=args.dqn_rerank_mode,
            dqn_hybrid_q_weight=args.dqn_hybrid_q_weight,
            seed=args.seed,
        )

        score, feasible = _score(summary, args.latency_budget_ms)
        d = summary["dqn_beam_tflite"]
        row = {
            "run_name": run_name,
            "phase1_enable": bool(args.phase1_enable),
            "reward_mode": args.reward_mode,
            "learning_rate": combo["learning_rate"],
            "epsilon_decay": combo["epsilon_decay"],
            "codebook_keep_ratio": combo["codebook_keep_ratio"],
            "phase1_num_clusters": combo["phase1_num_clusters"] if args.phase1_enable else 0,
            "dqn_rerank_topk": combo["dqn_rerank_topk"],
            "dqn_rerank_mode": args.dqn_rerank_mode,
            "dqn_hybrid_q_weight": args.dqn_hybrid_q_weight,
            "dueling_dqn": bool(args.dueling_dqn),
            "prioritized_replay": bool(args.prioritized_replay),
            "priority_alpha": args.priority_alpha,
            "priority_beta_start": args.priority_beta_start,
            "priority_beta_increment": args.priority_beta_increment,
            "cap_mean": d["cap_mean"],
            "lat_mean_ms": d["lat_mean_ms"],
            "lat_p95_ms": d["lat_p95_ms"],
            "sinr_mean_db": d["sinr_mean_db"],
            "ber_mean": d["ber_mean"],
            "latency_feasible": int(feasible),
            "score": score,
            "benchmark_json": json_out,
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["latency_feasible"], r["score"]), reverse=True)
    _write_csv(rows, os.path.join(args.out_dir, "sweep_ranked.csv"))

    with open(os.path.join(args.out_dir, "best_config.json"), "w") as f:
        json.dump(rows[0] if rows else {}, f, indent=2)

    with open(os.path.join(args.out_dir, "sweep_grid.json"), "w") as f:
        json.dump(
            {
                "grid": grid,
                "phase1_enable": args.phase1_enable,
                "reward_mode": args.reward_mode,
                "dqn_rerank_mode": args.dqn_rerank_mode,
                "dqn_hybrid_q_weight": float(args.dqn_hybrid_q_weight),
                "dueling_dqn": bool(args.dueling_dqn),
                "prioritized_replay": bool(args.prioritized_replay),
                "priority_alpha": float(args.priority_alpha),
                "priority_beta_start": float(args.priority_beta_start),
                "priority_beta_increment": float(args.priority_beta_increment),
                "priority_eps": float(args.priority_eps),
            },
            f,
            indent=2,
        )

    print(f"Saved hyperparameter sweep outputs to: {args.out_dir}")
    print("Files: sweep_ranked.csv, best_config.json, sweep_grid.json, per-run benchmark_summary.json")


if __name__ == "__main__":
    main()
