import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METHOD_ORDER = [
    "mmse",
    "zf",
    "rl_teacher",
    "rl_student_tflite",
    "dqn_beam",
    "dqn_beam_tflite",
]

METRICS = [
    ("cap_mean", "Capacity Mean", "bps/Hz", True),
    ("lat_mean_ms", "Latency Mean", "ms", False),
    ("sinr_mean_db", "SINR Mean", "dB", True),
    ("ber_mean", "BER Mean", "", False),
]


def _load_summary(path):
    with open(path, "r") as f:
        payload = json.load(f)
    return payload["summary"]


def _format_method_name(name):
    return name.replace("_", " ").upper()


def plot_before_after_all_methods(before_summary, after_summary, output_path):
    methods = [m for m in DEFAULT_METHOD_ORDER if m in before_summary and m in after_summary]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    x = np.arange(len(methods))
    width = 0.38

    for ax, (metric_key, metric_title, unit, higher_better) in zip(axes, METRICS):
        before_vals = [before_summary[m][metric_key] for m in methods]
        after_vals = [after_summary[m][metric_key] for m in methods]

        ax.bar(x - width / 2, before_vals, width, label="Before", alpha=0.85)
        ax.bar(x + width / 2, after_vals, width, label="After", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([_format_method_name(m) for m in methods], rotation=25, ha="right")
        ax.set_title(f"{metric_title} ({'Higher better' if higher_better else 'Lower better'})")
        ylabel = metric_title if unit == "" else f"{metric_title} ({unit})"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("External Dataset Impact: Before vs After Retraining", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_delta_focus(before_summary, after_summary, output_path):
    focus_methods = ["dqn_beam", "dqn_beam_tflite"]
    available_methods = [m for m in focus_methods if m in before_summary and m in after_summary]

    metric_keys = [m[0] for m in METRICS]
    metric_labels = [m[1] for m in METRICS]

    deltas = []
    for method in available_methods:
        method_deltas = []
        for metric_key in metric_keys:
            method_deltas.append(after_summary[method][metric_key] - before_summary[method][metric_key])
        deltas.append(method_deltas)
    deltas = np.array(deltas, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(deltas, cmap="coolwarm", aspect="auto")

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(available_methods)))
    ax.set_yticklabels([_format_method_name(m) for m in available_methods])
    ax.set_title("Delta Heatmap (After - Before) for DQN Methods")

    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            ax.text(j, i, f"{deltas[i, j]:+.4f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Delta Value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize benchmark improvements after adding external datasets.")
    parser.add_argument(
        "--before-json",
        type=str,
        default="results/benchmark_external_before_retrain.json",
    )
    parser.add_argument(
        "--after-json",
        type=str,
        default="results/benchmark_external_after_retrain.json",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="results/external_improvement",
        help="Prefix for output png files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    before_summary = _load_summary(args.before_json)
    after_summary = _load_summary(args.after_json)

    output_dir = os.path.dirname(args.out_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_methods_path = f"{args.out_prefix}_all_methods.png"
    delta_focus_path = f"{args.out_prefix}_dqn_delta_heatmap.png"

    plot_before_after_all_methods(before_summary, after_summary, all_methods_path)
    plot_delta_focus(before_summary, after_summary, delta_focus_path)

    print("Generated visualizations:")
    print(f"- {all_methods_path}")
    print(f"- {delta_focus_path}")


if __name__ == "__main__":
    main()
