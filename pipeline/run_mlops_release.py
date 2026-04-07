import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, src_path)

from benchmark_optimized import benchmark
from train_dqn_beam import train_dqn_beam


def _utc_now_compact():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_method_summary(benchmark_json_path, method_key):
    payload = _read_json(benchmark_json_path)
    summary = payload.get("summary", {})
    if method_key not in summary:
        raise KeyError(
            f"Method '{method_key}' not found in {benchmark_json_path}. "
            f"Available: {list(summary.keys())}"
        )
    return summary[method_key]


def _evaluate_gate(baseline, candidate, args):
    checks = {}
    checks["capacity_non_decrease"] = (candidate["cap_mean"] - baseline["cap_mean"]) >= args.min_capacity_gain
    checks["latency_budget"] = candidate["lat_mean_ms"] <= args.latency_budget_ms
    checks["latency_p95_budget"] = candidate["lat_p95_ms"] <= args.latency_p95_budget_ms
    checks["ber_regression_limit"] = (candidate["ber_mean"] - baseline["ber_mean"]) <= args.max_ber_regression
    checks["sinr_drop_limit"] = (baseline["sinr_mean_db"] - candidate["sinr_mean_db"]) <= args.max_sinr_drop_db

    deltas = {
        "cap_mean": candidate["cap_mean"] - baseline["cap_mean"],
        "lat_mean_ms": candidate["lat_mean_ms"] - baseline["lat_mean_ms"],
        "lat_p95_ms": candidate["lat_p95_ms"] - baseline["lat_p95_ms"],
        "sinr_mean_db": candidate["sinr_mean_db"] - baseline["sinr_mean_db"],
        "ber_mean": candidate["ber_mean"] - baseline["ber_mean"],
    }

    passed = all(checks.values())
    return passed, checks, deltas


def _copy_artifacts(artifact_paths, dest_dir):
    copied = []
    missing = []
    for path in artifact_paths:
        src = Path(path)
        if not src.exists():
            missing.append(str(src))
            continue
        dest = Path(dest_dir) / src.name
        shutil.copy2(src, dest)
        copied.append(
            {
                "source": str(src),
                "dest": str(dest),
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
            }
        )
    return copied, missing


def _load_config_overrides(args):
    if not args.config:
        return {}
    return _read_json(args.config)


def _resolve_artifact_paths(args, config):
    if args.artifact_paths:
        return [x.strip() for x in args.artifact_paths.split(",") if x.strip()]
    return config.get("artifact_paths", [])


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end MLOps release runner: train -> benchmark -> gate -> optional promote.")
    p.add_argument("--config", type=str, default="pipeline/mlops_release_config.json")

    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--model-name", type=str, default="dqn_beam_phase2")
    p.add_argument("--results-root", type=str, default="results/mlops")

    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")

    p.add_argument("--candidate-benchmark-json", type=str, default="")
    p.add_argument("--baseline-json", type=str, default="")
    p.add_argument("--candidate-method", type=str, default="")
    p.add_argument("--baseline-method", type=str, default="")

    p.add_argument("--channel-source", type=str, default="simulator", choices=["simulator", "external", "mixed"])
    p.add_argument("--external-registry", type=str, default="data/dataset_registry.json")
    p.add_argument("--external-max-samples", type=int, default=20000)
    p.add_argument("--external-mix-ratio", type=float, default=0.5)

    p.add_argument("--train-episodes", type=int, default=80)
    p.add_argument("--train-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--imitation-samples", type=int, default=500)
    p.add_argument("--imitation-epochs", type=int, default=6)

    p.add_argument("--phase1-enable", action="store_true")
    p.add_argument("--phase1-num-clusters", type=int, default=2)
    p.add_argument("--reward-mode", type=str, default="constrained", choices=["legacy", "constrained"])

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--epsilon-decay", type=float, default=0.994)
    p.add_argument("--codebook-keep-ratio", type=float, default=0.35)
    p.add_argument("--dueling-dqn", action="store_true")
    p.add_argument("--prioritized-replay", action="store_true")

    p.add_argument("--benchmark-iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dqn-rerank-topk", type=int, default=2)
    p.add_argument("--dqn-rerank-mode", type=str, default="capacity", choices=["capacity", "hybrid", "q_only"])
    p.add_argument("--dqn-hybrid-q-weight", type=float, default=0.5)

    p.add_argument("--latency-budget-ms", type=float, default=1.0)
    p.add_argument("--latency-p95-budget-ms", type=float, default=2.0)
    p.add_argument("--min-capacity-gain", type=float, default=0.0)
    p.add_argument("--max-ber-regression", type=float, default=0.002)
    p.add_argument("--max-sinr-drop-db", type=float, default=0.4)

    p.add_argument("--artifact-paths", type=str, default="")
    p.add_argument("--promote-on-pass", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = _load_config_overrides(args)

    run_id = args.run_id or _utc_now_compact()
    results_root = Path(args.results_root)
    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_json = args.baseline_json or config.get("baseline_json", "results/benchmark_best_config.json")
    candidate_method = args.candidate_method or config.get("candidate_method", "dqn_beam_tflite")
    baseline_method = args.baseline_method or config.get("baseline_method", "dqn_beam_tflite")

    if args.candidate_benchmark_json:
        candidate_json = args.candidate_benchmark_json
    else:
        candidate_json = str(run_dir / "candidate_benchmark.json")

    train_started = False
    if not args.skip_train and not args.skip_benchmark and not args.candidate_benchmark_json:
        train_started = True
        train_dqn_beam(
            num_episodes=args.train_episodes,
            max_steps=args.train_steps,
            batch_size=args.batch_size,
            imitation_samples=args.imitation_samples,
            imitation_epochs=args.imitation_epochs,
            codebook_keep_ratio=args.codebook_keep_ratio,
            channel_source=args.channel_source,
            external_registry_path=args.external_registry,
            external_max_samples=args.external_max_samples,
            external_mix_ratio=args.external_mix_ratio,
            phase1_enable=args.phase1_enable,
            phase1_num_clusters=args.phase1_num_clusters,
            reward_mode=args.reward_mode,
            learning_rate=args.learning_rate,
            epsilon_decay=args.epsilon_decay,
            dueling_dqn=args.dueling_dqn,
            prioritized_replay=args.prioritized_replay,
        )

    if not args.skip_benchmark and not args.candidate_benchmark_json:
        benchmark(
            num_iterations=args.benchmark_iterations,
            save_json_path=candidate_json,
            channel_source=args.channel_source,
            external_registry_path=args.external_registry,
            external_max_samples=args.external_max_samples,
            external_mix_ratio=args.external_mix_ratio,
            dqn_rerank_topk=args.dqn_rerank_topk,
            dqn_rerank_mode=args.dqn_rerank_mode,
            dqn_hybrid_q_weight=args.dqn_hybrid_q_weight,
            seed=args.seed,
        )

    if not Path(candidate_json).exists():
        raise FileNotFoundError(
            f"Candidate benchmark not found: {candidate_json}. "
            f"Provide --candidate-benchmark-json or run without --skip-benchmark."
        )
    if not Path(baseline_json).exists():
        raise FileNotFoundError(f"Baseline benchmark not found: {baseline_json}")

    baseline = _extract_method_summary(baseline_json, baseline_method)
    candidate = _extract_method_summary(candidate_json, candidate_method)
    gate_passed, checks, deltas = _evaluate_gate(baseline, candidate, args)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": _utc_now_compact(),
        "model_name": args.model_name,
        "inputs": {
            "baseline_json": baseline_json,
            "candidate_json": candidate_json,
            "baseline_method": baseline_method,
            "candidate_method": candidate_method,
        },
        "train": {
            "started": train_started,
            "skip_train": bool(args.skip_train),
            "phase1_enable": bool(args.phase1_enable),
            "phase1_num_clusters": int(args.phase1_num_clusters),
            "reward_mode": args.reward_mode,
            "learning_rate": args.learning_rate,
            "epsilon_decay": args.epsilon_decay,
            "codebook_keep_ratio": args.codebook_keep_ratio,
            "dueling_dqn": bool(args.dueling_dqn),
            "prioritized_replay": bool(args.prioritized_replay),
        },
        "benchmark": {
            "skip_benchmark": bool(args.skip_benchmark),
            "iterations": int(args.benchmark_iterations),
            "channel_source": args.channel_source,
            "dqn_rerank_topk": int(args.dqn_rerank_topk),
            "dqn_rerank_mode": args.dqn_rerank_mode,
            "dqn_hybrid_q_weight": float(args.dqn_hybrid_q_weight),
            "seed": int(args.seed),
        },
        "gate": {
            "passed": bool(gate_passed),
            "checks": checks,
            "thresholds": {
                "latency_budget_ms": args.latency_budget_ms,
                "latency_p95_budget_ms": args.latency_p95_budget_ms,
                "min_capacity_gain": args.min_capacity_gain,
                "max_ber_regression": args.max_ber_regression,
                "max_sinr_drop_db": args.max_sinr_drop_db,
            },
            "deltas": deltas,
            "baseline": baseline,
            "candidate": candidate,
        },
    }

    _write_json(str(run_dir / "run_manifest.json"), manifest)

    promotion_record = None
    if gate_passed and args.promote_on_pass:
        registry_root = results_root / "registry" / args.model_name
        version = run_id
        version_dir = registry_root / version
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = _resolve_artifact_paths(args, config)
        copied, missing = _copy_artifacts(artifact_paths, version_dir)

        shutil.copy2(candidate_json, version_dir / "candidate_benchmark.json")
        shutil.copy2(run_dir / "run_manifest.json", version_dir / "run_manifest.json")

        promotion_record = {
            "model_name": args.model_name,
            "version": version,
            "run_id": run_id,
            "promoted_at_utc": _utc_now_compact(),
            "candidate_json": candidate_json,
            "copied_artifacts": copied,
            "missing_artifacts": missing,
            "gate_passed": True,
            "key_metrics": {
                "cap_mean": candidate.get("cap_mean"),
                "lat_mean_ms": candidate.get("lat_mean_ms"),
                "lat_p95_ms": candidate.get("lat_p95_ms"),
                "sinr_mean_db": candidate.get("sinr_mean_db"),
                "ber_mean": candidate.get("ber_mean"),
            },
        }

        index_path = registry_root / "index.json"
        if index_path.exists():
            index_data = _read_json(str(index_path))
        else:
            index_data = {"model_name": args.model_name, "versions": [], "latest": None}

        index_data["versions"].append(promotion_record)
        index_data["latest"] = {
            "version": version,
            "run_id": run_id,
            "path": str(version_dir),
        }
        _write_json(str(index_path), index_data)

    report = {
        "run_id": run_id,
        "gate_passed": gate_passed,
        "candidate_json": candidate_json,
        "baseline_json": baseline_json,
        "manifest_path": str(run_dir / "run_manifest.json"),
        "promotion": promotion_record,
    }
    _write_json(str(run_dir / "release_report.json"), report)

    print("=" * 72)
    print("MLOPS RELEASE REPORT")
    print("=" * 72)
    print(f"run_id: {run_id}")
    print(f"candidate_json: {candidate_json}")
    print(f"baseline_json: {baseline_json}")
    print(f"gate_passed: {gate_passed}")
    for key, value in checks.items():
        print(f"  - {key}: {value}")
    print(f"manifest: {run_dir / 'run_manifest.json'}")
    print(f"report: {run_dir / 'release_report.json'}")
    if promotion_record is not None:
        print(f"promoted_version: {promotion_record['version']}")
        print(f"registry_path: {results_root / 'registry' / args.model_name / promotion_record['version']}")


if __name__ == "__main__":
    main()
