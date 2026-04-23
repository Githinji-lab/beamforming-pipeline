import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(json_path):
    with open(json_path, "r") as f:
        payload = json.load(f)
    return payload["summary"]


def _find_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these files exist: {paths}")


def _resolve_topk_summaries(results_dir):
    def _topk_candidates(k):
        return [
            os.path.join(results_dir, f"benchmark_external_topk{k}.json"),
            os.path.join(results_dir, f"benchmark_topk{k}.json"),
            os.path.join(results_dir, "benchmark_best_config.json"),
        ]

    topk_paths = {}
    for k in [1, 2, 3]:
        try:
            topk_paths[k] = _find_existing(_topk_candidates(k))
        except FileNotFoundError:
            pass

    if not topk_paths:
        raise FileNotFoundError("No benchmark JSON files found. Run the benchmark pipeline first.")

    return {k: _load_summary(path) for k, path in topk_paths.items()}


def _select_summary(topk_summaries):
    selected_topk = max(
        topk_summaries.keys(),
        key=lambda k: topk_summaries[k]["dqn_beam_tflite"]["cap_mean"]
        if topk_summaries[k]["dqn_beam_tflite"]["lat_mean_ms"] < 1.0
        else -1e9,
    )
    return selected_topk, topk_summaries[selected_topk]


def _plot_capacity(selected_summary, out_path):
    methods = ["mmse", "zf", "rl_student_tflite", "dqn_beam", "dqn_beam_tflite"]
    labels = ["MMSE", "ZF", "RL\nStudent", "DQN\nBeam", "DQN\nTFLite"]
    values = [selected_summary[m]["cap_mean"] for m in methods]
    colors = ["#3b78b4", "#2b5785", "#ff8c00", "#e53935", "#43a047"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.94)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Capacity (bps/Hz)")
    ax.set_title("Results: Capacity (bps/Hz)", loc="left", fontsize=18, fontweight="bold", color="white", pad=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_facecolor("#f8fbff")
    fig.patch.set_facecolor("white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.add_patch(plt.Rectangle((0, 1.02), 1, 0.22, transform=ax.transAxes, color="#8d1b15", clip_on=False))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_latency(selected_summary, out_path):
    methods = ["mmse", "zf", "rl_student_tflite", "dqn_beam", "dqn_beam_tflite"]
    labels = ["MMSE", "ZF", "RL\nStudent", "DQN\nBeam", "DQN\nTFLite"]
    values = [selected_summary[m]["lat_mean_ms"] for m in methods]
    colors = ["#3b78b4", "#2b5785", "#ff8c00", "#e53935", "#43a047"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(np.arange(len(labels)), values, color=colors, width=0.94)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Results: Inference Latency (ms)", loc="left", fontsize=18, fontweight="bold", color="white", pad=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_facecolor("#f8fbff")
    fig.patch.set_facecolor("white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.add_patch(plt.Rectangle((0, 1.02), 1, 0.22, transform=ax.transAxes, color="#1f6d33", clip_on=False))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Export standalone capacity and latency defense plots.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out-dir", type=str, default="results/defense")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    topk_summaries = _resolve_topk_summaries(args.results_dir)
    selected_topk, selected_summary = _select_summary(topk_summaries)

    capacity_path = os.path.join(args.out_dir, "results_capacity_standalone.png")
    latency_path = os.path.join(args.out_dir, "results_inference_latency_standalone.png")

    _plot_capacity(selected_summary, capacity_path)
    _plot_latency(selected_summary, latency_path)

    print(f"Selected top-k: {selected_topk}")
    print(f"Saved: {capacity_path}")
    print(f"Saved: {latency_path}")


if __name__ == "__main__":
    main()