import argparse
import csv
import json
import os

from train_dqn_beam import train_dqn_beam
from benchmark_optimized import benchmark
from external_dataset import load_channels_from_registry
from simulators import BeamformingSimulatorV4


def _extract_method(summary, method="dqn_beam_tflite"):
    d = summary.get(method, {})
    return {
        "method": method,
        "cap_mean": d.get("cap_mean", float("nan")),
        "lat_mean_ms": d.get("lat_mean_ms", float("nan")),
        "lat_p95_ms": d.get("lat_p95_ms", float("nan")),
        "sinr_mean_db": d.get("sinr_mean_db", float("nan")),
        "ber_mean": d.get("ber_mean", float("nan"),),
    }


def _write_rows(rows, out_csv):
    fieldnames = [
        "variant",
        "method",
        "cap_mean",
        "lat_mean_ms",
        "lat_p95_ms",
        "sinr_mean_db",
        "ber_mean",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="Run baseline vs Phase-1 DQN ablation.")
    p.add_argument("--out-dir", type=str, default="results/phase1")
    p.add_argument("--train-episodes", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=30)
    p.add_argument("--bench-iterations", type=int, default=40)
    p.add_argument("--channel-source", type=str, default="external", choices=["simulator", "external", "mixed"])
    p.add_argument("--external-registry", type=str, default="data/dataset_registry.json")
    p.add_argument("--external-max-samples", type=int, default=5000)
    p.add_argument("--external-mix-ratio", type=float, default=0.5)
    p.add_argument("--phase1-num-clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-channel-fallback",
        action="store_true",
        help="If external dataset channels cannot be loaded, fallback to simulator mode instead of failing.",
    )
    return p.parse_args()


def _resolve_channel_source(args):
    source = args.channel_source
    if source not in ("external", "mixed"):
        return source

    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    try:
        channels = load_channels_from_registry(
            registry_path=args.external_registry,
            target_k=simulator.K,
            target_n_tx=simulator.N_tx,
            max_total_samples=max(32, int(args.external_max_samples)),
        )
        print(f"Preflight check: loaded {channels.shape[0]} external channels from registry.")
        return source
    except Exception as exc:
        if args.allow_channel_fallback:
            print(
                "Preflight warning: external channel loading failed; "
                "falling back to channel-source=simulator.\n"
                f"Reason: {exc}"
            )
            return "simulator"
        raise RuntimeError(
            "External channel preflight failed. Your registry does not currently expose channel matrices "
            "with supported keys (Hvirtual/Harray/H/channel/channels) in .mat/.npz files. "
            "Fix registry contents or rerun with --allow-channel-fallback.\n"
            f"Original error: {exc}"
        ) from exc


def _train_and_benchmark(variant_name, train_kwargs, bench_kwargs, out_dir):
    train_dqn_beam(**train_kwargs)
    json_out = os.path.join(out_dir, f"benchmark_{variant_name}.json")
    _, summary = benchmark(save_json_path=json_out, **bench_kwargs)
    return summary


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    resolved_channel_source = _resolve_channel_source(args)

    common_train = {
        "num_episodes": args.train_episodes,
        "max_steps": args.train_steps,
        "batch_size": 32,
        "num_beams": 24,
        "imitation_samples": 180,
        "imitation_epochs": 2,
        "channel_source": resolved_channel_source,
        "external_registry_path": args.external_registry,
        "external_max_samples": args.external_max_samples,
        "external_mix_ratio": args.external_mix_ratio,
    }

    common_bench = {
        "num_iterations": args.bench_iterations,
        "channel_source": resolved_channel_source,
        "external_registry_path": args.external_registry,
        "external_max_samples": args.external_max_samples,
        "external_mix_ratio": args.external_mix_ratio,
        "dqn_rerank_topk": 3,
        "seed": args.seed,
    }

    baseline_summary = _train_and_benchmark(
        variant_name="baseline",
        train_kwargs={
            **common_train,
            "phase1_enable": False,
            "phase1_num_clusters": 0,
            "reward_mode": "legacy",
        },
        bench_kwargs=common_bench,
        out_dir=args.out_dir,
    )

    phase1_summary = _train_and_benchmark(
        variant_name="phase1",
        train_kwargs={
            **common_train,
            "phase1_enable": True,
            "phase1_num_clusters": args.phase1_num_clusters,
            "reward_mode": "constrained",
        },
        bench_kwargs=common_bench,
        out_dir=args.out_dir,
    )

    rows = [
        {"variant": "baseline", **_extract_method(baseline_summary)},
        {"variant": "phase1", **_extract_method(phase1_summary)},
    ]

    csv_out = os.path.join(args.out_dir, "phase1_ablation_table.csv")
    _write_rows(rows, csv_out)

    with open(os.path.join(args.out_dir, "phase1_ablation_summary.json"), "w") as f:
        json.dump({"baseline": baseline_summary, "phase1": phase1_summary}, f, indent=2)

    print(f"Saved Phase-1 ablation outputs to: {args.out_dir}")
    print("Files: benchmark_baseline.json, benchmark_phase1.json, phase1_ablation_table.csv, phase1_ablation_summary.json")


if __name__ == "__main__":
    main()
