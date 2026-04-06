import os
import json
import argparse
from datetime import datetime

from train_dqn_beam import train_dqn_beam
from benchmark_optimized import benchmark


def parse_ratios(text):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def run_sweep(
    ratios,
    episodes,
    steps,
    batch_size,
    imitation_samples,
    imitation_epochs,
    benchmark_iterations,
    num_beams,
    dataset_zips,
    channel_source,
    external_registry,
    external_max_samples,
    external_mix_ratio,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(project_root, 'results')
    sweep_dir = os.path.join(results_dir, 'sweeps')
    os.makedirs(sweep_dir, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    aggregate_path = os.path.join(sweep_dir, f'dqn_keep_ratio_sweep_{run_id}.json')

    all_results = {
        'run_id': run_id,
        'config': {
            'episodes': episodes,
            'steps': steps,
            'batch_size': batch_size,
            'imitation_samples': imitation_samples,
            'imitation_epochs': imitation_epochs,
            'benchmark_iterations': benchmark_iterations,
            'num_beams': num_beams,
            'ratios': ratios,
        },
        'results': [],
    }

    for ratio in ratios:
        print('=' * 80)
        print(f'SWEEP ratio={ratio:.3f}')
        print('=' * 80)

        train_dqn_beam(
            num_episodes=episodes,
            max_steps=steps,
            batch_size=batch_size,
            num_beams=num_beams,
            imitation_samples=imitation_samples,
            imitation_epochs=imitation_epochs,
            codebook_strategy='teacher_top',
            codebook_keep_ratio=ratio,
            dataset_zip_paths=dataset_zips,
            channel_source=channel_source,
            external_registry_path=external_registry,
            external_max_samples=external_max_samples,
            external_mix_ratio=external_mix_ratio,
        )

        bench_json = os.path.join(sweep_dir, f'benchmark_ratio_{ratio:.3f}_{run_id}.json')
        _, summary = benchmark(
            num_iterations=benchmark_iterations,
            save_json_path=bench_json,
            channel_source=channel_source,
            external_registry_path=external_registry,
            external_max_samples=external_max_samples,
            external_mix_ratio=external_mix_ratio,
        )

        result_entry = {
            'keep_ratio': ratio,
            'benchmark_json': bench_json,
            'summary': summary,
        }
        all_results['results'].append(result_entry)

        if 'dqn_beam_tflite' in summary:
            s = summary['dqn_beam_tflite']
            print(
                f"dqn_beam_tflite | cap_mean={s['cap_mean']:.3f} | "
                f"lat_p95={s['lat_p95_ms']:.3f} ms | sinr_mean={s['sinr_mean_db']:.2f} dB"
            )

    with open(aggregate_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    ranked = []
    for r in all_results['results']:
        s = r['summary'].get('dqn_beam_tflite', None)
        if s is not None:
            ranked.append((r['keep_ratio'], s['cap_mean'], s['lat_p95_ms'], s['sinr_mean_db']))

    ranked.sort(key=lambda x: (-x[1], x[2]))

    print('\n' + '=' * 80)
    print('SWEEP RANKING (dqn_beam_tflite)')
    print('=' * 80)
    for ratio, cap, lat95, sinr in ranked:
        print(f'ratio={ratio:.3f} | cap_mean={cap:.3f} | lat_p95={lat95:.3f} ms | sinr_mean={sinr:.2f} dB')

    print(f'\nSaved aggregate sweep results to: {aggregate_path}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ratios', type=str, default='0.10,0.15,0.20,0.30')
    p.add_argument('--episodes', type=int, default=240)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--imitation-samples', type=int, default=1200)
    p.add_argument('--imitation-epochs', type=int, default=8)
    p.add_argument('--benchmark-iterations', type=int, default=200)
    p.add_argument('--num-beams', type=int, default=24)
    p.add_argument('--dataset-zips', type=str, default='')
    p.add_argument('--channel-source', type=str, default='simulator', choices=['simulator', 'external', 'mixed'])
    p.add_argument('--external-registry', type=str, default='data/dataset_registry.json')
    p.add_argument('--external-max-samples', type=int, default=20000)
    p.add_argument('--external-mix-ratio', type=float, default=0.5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_sweep(
        ratios=parse_ratios(args.ratios),
        episodes=args.episodes,
        steps=args.steps,
        batch_size=args.batch_size,
        imitation_samples=args.imitation_samples,
        imitation_epochs=args.imitation_epochs,
        benchmark_iterations=args.benchmark_iterations,
        num_beams=args.num_beams,
        dataset_zips=[z.strip() for z in args.dataset_zips.split(',') if z.strip()],
        channel_source=args.channel_source,
        external_registry=args.external_registry,
        external_max_samples=args.external_max_samples,
        external_mix_ratio=args.external_mix_ratio,
    )
