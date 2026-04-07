import argparse
import csv
import json
import os
import sys
from copy import deepcopy

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, src_path)

from benchmark_optimized import benchmark
from train_dqn_beam import train_dqn_beam


HEADLINE_METHODS = [
    "mmse",
    "zf",
    "rl_student_tflite",
    "dqn_beam",
    "dqn_beam_tflite",
]

METRICS = ["cap_mean", "lat_mean_ms", "lat_p95_ms", "sinr_mean_db", "ber_mean"]


def _aggregate_run_summaries(run_summaries):
    aggregated = {}
    n = len(run_summaries)
    for method in run_summaries[0].keys():
        aggregated[method] = {}
        for metric in METRICS:
            values = np.array([run[method][metric] for run in run_summaries], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(max(len(values), 1)))
            aggregated[method][metric] = {
                "mean": mean,
                "std": std,
                "ci95": ci95,
                "runs": [float(v) for v in values.tolist()],
            }
    aggregated["_meta"] = {"num_runs": n}
    return aggregated


def _write_headline_table(aggregated, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "cap_mean",
                "cap_std",
                "cap_ci95",
                "lat_mean_ms",
                "lat_std_ms",
                "lat_ci95_ms",
                "lat_p95_ms_mean",
                "sinr_mean_db",
                "sinr_ci95_db",
                "ber_mean",
                "ber_ci95",
            ]
        )
        for method in HEADLINE_METHODS:
            if method not in aggregated:
                continue
            row = aggregated[method]
            writer.writerow(
                [
                    method,
                    row["cap_mean"]["mean"],
                    row["cap_mean"]["std"],
                    row["cap_mean"]["ci95"],
                    row["lat_mean_ms"]["mean"],
                    row["lat_mean_ms"]["std"],
                    row["lat_mean_ms"]["ci95"],
                    row["lat_p95_ms"]["mean"],
                    row["sinr_mean_db"]["mean"],
                    row["sinr_mean_db"]["ci95"],
                    row["ber_mean"]["mean"],
                    row["ber_mean"]["ci95"],
                ]
            )


def _parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _run_repeated_benchmark(protocol, seeds, out_dir, tag):
    run_summaries = []
    for seed in seeds:
        json_out = os.path.join(out_dir, f"benchmark_{tag}_seed{seed}.json")
        _, summary = benchmark(
            num_iterations=protocol["iterations"],
            save_json_path=json_out,
            channel_source=protocol["channel_source"],
            external_registry_path=protocol["external_registry"],
            external_max_samples=protocol["external_max_samples"],
            external_mix_ratio=protocol["external_mix_ratio"],
            dqn_rerank_topk=protocol["dqn_rerank_topk"],
            dqn_rerank_mode=protocol["dqn_rerank_mode"],
            dqn_hybrid_q_weight=protocol["dqn_hybrid_q_weight"],
            seed=seed,
        )
        run_summaries.append(summary)
    return _aggregate_run_summaries(run_summaries)


def _run_topk_dataset_ablation(base_protocol, seeds, out_dir, skip_external=False):
    rows = []
    for channel_source, dataset_on in [("simulator", 0), ("external", 1)]:
        if skip_external and channel_source == "external":
            print(f"[ablation] Skipping external-dataset variant (--skip-external-ablation set).")
            continue
        for topk in [1, 2, 3]:
            protocol = deepcopy(base_protocol)
            protocol["channel_source"] = channel_source
            protocol["dqn_rerank_topk"] = topk
            tag = f"ablation_dataset{dataset_on}_topk{topk}"
            agg = _run_repeated_benchmark(protocol, seeds, out_dir, tag)
            d = agg["dqn_beam_tflite"]
            rows.append(
                {
                    "ablation": "topk_dataset",
                    "dataset_on": dataset_on,
                    "channel_source": channel_source,
                    "topk": topk,
                    "teacher_top_ratio": "NA",
                    "cap_mean": d["cap_mean"]["mean"],
                    "cap_std": d["cap_mean"]["std"],
                    "cap_ci95": d["cap_mean"]["ci95"],
                    "lat_mean_ms": d["lat_mean_ms"]["mean"],
                    "lat_ci95_ms": d["lat_mean_ms"]["ci95"],
                    "lat_p95_ms": d["lat_p95_ms"]["mean"],
                    "sinr_mean_db": d["sinr_mean_db"]["mean"],
                    "ber_mean": d["ber_mean"]["mean"],
                }
            )
    return rows


def _run_teacher_ratio_ablation(base_protocol, seeds, out_dir, ratios, train_cfg):
    rows = []
    for ratio in ratios:
        train_dqn_beam(
            num_episodes=train_cfg["episodes"],
            max_steps=train_cfg["steps"],
            batch_size=train_cfg["batch_size"],
            num_beams=train_cfg["num_beams"],
            imitation_samples=train_cfg["imitation_samples"],
            imitation_epochs=train_cfg["imitation_epochs"],
            codebook_strategy="teacher_top",
            codebook_keep_ratio=ratio,
            channel_source=train_cfg["channel_source"],
            external_registry_path=train_cfg["external_registry"],
            external_max_samples=train_cfg["external_max_samples"],
            external_mix_ratio=train_cfg["external_mix_ratio"],
        )

        protocol = deepcopy(base_protocol)
        protocol["channel_source"] = "external"
        protocol["dqn_rerank_topk"] = train_cfg["topk_for_eval"]
        tag = f"ablation_teacher_ratio_{ratio:.2f}".replace(".", "p")
        agg = _run_repeated_benchmark(protocol, seeds, out_dir, tag)
        d = agg["dqn_beam_tflite"]
        rows.append(
            {
                "ablation": "teacher_top_ratio",
                "dataset_on": 1,
                "channel_source": "external",
                "topk": train_cfg["topk_for_eval"],
                "teacher_top_ratio": ratio,
                "cap_mean": d["cap_mean"]["mean"],
                "cap_std": d["cap_mean"]["std"],
                "cap_ci95": d["cap_mean"]["ci95"],
                "lat_mean_ms": d["lat_mean_ms"]["mean"],
                "lat_ci95_ms": d["lat_mean_ms"]["ci95"],
                "lat_p95_ms": d["lat_p95_ms"]["mean"],
                "sinr_mean_db": d["sinr_mean_db"]["mean"],
                "ber_mean": d["ber_mean"]["mean"],
            }
        )
    return rows


def _write_ablation_csv(rows, out_csv):
    if not rows:
        return
    keys = [
        "ablation",
        "dataset_on",
        "channel_source",
        "topk",
        "teacher_top_ratio",
        "cap_mean",
        "cap_std",
        "cap_ci95",
        "lat_mean_ms",
        "lat_ci95_ms",
        "lat_p95_ms",
        "sinr_mean_db",
        "ber_mean",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed defense protocol with 3-run stats and ablations.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="results/protocol")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--channel-source", type=str, default="external", choices=["simulator", "external", "mixed"])
    parser.add_argument("--external-registry", type=str, default="data/dataset_registry.json")
    parser.add_argument("--external-max-samples", type=int, default=5000)
    parser.add_argument("--external-mix-ratio", type=float, default=0.5)
    parser.add_argument("--dqn-rerank-topk", type=int, default=3)
    parser.add_argument("--dqn-rerank-mode", type=str, default="capacity", choices=["capacity", "hybrid", "q_only"])
    parser.add_argument("--dqn-hybrid-q-weight", type=float, default=0.5)

    parser.add_argument("--skip-external-ablation", action="store_true",
                        help="Skip the external-dataset variant in the topk ablation (use when external registry is unavailable).")
    parser.add_argument("--run-teacher-ratio-ablation", action="store_true")
    parser.add_argument("--teacher-ratios", type=str, default="0.20,0.30,0.35")
    parser.add_argument("--ablation-train-episodes", type=int, default=20)
    parser.add_argument("--ablation-train-steps", type=int, default=30)
    parser.add_argument("--ablation-train-batch", type=int, default=32)
    parser.add_argument("--ablation-train-beams", type=int, default=24)
    parser.add_argument("--ablation-imitation-samples", type=int, default=180)
    parser.add_argument("--ablation-imitation-epochs", type=int, default=2)
    parser.add_argument("--ablation-eval-topk", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    protocol = {
        "iterations": int(args.iterations),
        "channel_source": args.channel_source,
        "external_registry": args.external_registry,
        "external_max_samples": int(args.external_max_samples),
        "external_mix_ratio": float(args.external_mix_ratio),
        "dqn_rerank_topk": int(args.dqn_rerank_topk),
        "dqn_rerank_mode": args.dqn_rerank_mode,
        "dqn_hybrid_q_weight": float(args.dqn_hybrid_q_weight),
    }

    headline_agg = _run_repeated_benchmark(protocol, seeds, args.out_dir, "headline")
    with open(os.path.join(args.out_dir, "headline_aggregate.json"), "w") as f:
        json.dump(
            {
                "protocol": protocol,
                "seeds": seeds,
                "aggregate": headline_agg,
            },
            f,
            indent=2,
        )
    _write_headline_table(headline_agg, os.path.join(args.out_dir, "headline_table.csv"))

    ablation_rows = _run_topk_dataset_ablation(protocol, seeds, args.out_dir,
                                                skip_external=args.skip_external_ablation)

    if args.run_teacher_ratio_ablation:
        teacher_ratios = _parse_float_list(args.teacher_ratios)
        train_cfg = {
            "episodes": int(args.ablation_train_episodes),
            "steps": int(args.ablation_train_steps),
            "batch_size": int(args.ablation_train_batch),
            "num_beams": int(args.ablation_train_beams),
            "imitation_samples": int(args.ablation_imitation_samples),
            "imitation_epochs": int(args.ablation_imitation_epochs),
            "channel_source": "mixed",
            "external_registry": args.external_registry,
            "external_max_samples": int(args.external_max_samples),
            "external_mix_ratio": float(args.external_mix_ratio),
            "topk_for_eval": int(args.ablation_eval_topk),
        }
        ablation_rows.extend(_run_teacher_ratio_ablation(protocol, seeds, args.out_dir, teacher_ratios, train_cfg))

    _write_ablation_csv(ablation_rows, os.path.join(args.out_dir, "ablation_table.csv"))

    claims_path = os.path.join(args.out_dir, "results_claims.txt")
    dqn_t = headline_agg.get("dqn_beam_tflite", {})
    cap = dqn_t.get("cap_mean", {}).get("mean", float("nan"))
    lat = dqn_t.get("lat_mean_ms", {}).get("mean", float("nan"))
    with open(claims_path, "w") as f:
        f.write("Results Claims (Defense)\n")
        f.write("========================\n")
        f.write(f"Latency success: mean latency is {lat:.3f} ms (<1 ms target).\n")
        f.write(f"Capacity parity: still in progress (current dqn_tflite cap_mean={cap:.3f}).\n")

    print(f"Saved protocol outputs to: {args.out_dir}")
    print("Files: headline_aggregate.json, headline_table.csv, ablation_table.csv, results_claims.txt")


if __name__ == "__main__":
    main()
