#!/usr/bin/env python3
"""Collect a 10-scene Sharpa pilot or replay its saved commands without Expert."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_learning.data.sharpa_demo import collect_episode, replay_episode


def pilot_cases():
    base = dict(x=.27, y=0., yaw_deg=0., size=[.12]*3, seed=0)
    result = [dict(base, name='canonical')]
    for axis in ('x', 'y'):
        for delta in (-.01, -.005, .005, .01):
            result.append(dict(base, name=f'{axis}{delta:+.3f}', **{axis:base[axis]+delta}))
    result.append(dict(base, name='cube11cm', size=[.11]*3))
    return result


def collect_job(job):
    path, case = job
    return collect_episode(path, case)


def replay_job(job):
    path, render_dir = job
    return replay_episode(path, render_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('--output', type=Path, required=True)
    collect.add_argument('--count', type=int, default=10, help='First N of the fixed, distinct pilot conditions (1..10)')
    collect.add_argument('--resume', action='store_true')
    collect.add_argument('--workers', type=int, default=3)
    replay = sub.add_parser('replay')
    replay.add_argument('--input', type=Path, required=True, help='Episode .npz or directory')
    replay.add_argument('--report', type=Path, required=True)
    replay.add_argument('--render-dir', type=Path)
    replay.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.mode == 'collect':
        if not 1 <= args.count <= 10:
            parser.error('--count must be in 1..10')
        selected = pilot_cases()[:args.count]
        args.output.mkdir(parents=True, exist_ok=True)
        jobs = []
        for i, case in enumerate(selected, 1):
            path = args.output / f'episode_{i:04d}_{case["name"]}.npz'
            if path.exists():
                if not args.resume:
                    parser.error(f'{path} exists; use another output directory or --resume')
                import numpy as np
                from humanoid_learning.data.sharpa_demo import environment_hash
                with np.load(path, allow_pickle=False) as archive:
                    meta = json.loads(str(archive['metadata']))
                if meta['case'] != case or meta['environment_sha256'] != environment_hash():
                    parser.error(f'{path} is incompatible with the requested case/current environment')
                print(f'KEEP {path}', flush=True)
            else:
                jobs.append((path, case))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(collect_job, jobs):
                print(json.dumps(result), flush=True)
        # The manifest covers resumed episodes as well as new ones.
        import numpy as np
        index = []
        for i, case in enumerate(selected, 1):
            path = args.output / f'episode_{i:04d}_{case["name"]}.npz'
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['metadata']))
            index.append(dict(path=path.name, case=case, **meta['result']))
        (args.output / 'index.json').write_text(json.dumps(index, indent=2)+'\n')
        print(f'COLLECTED {len(index)}; physical success {sum(r["physical_success"] for r in index)}/{len(index)}')
        if not all(r['physical_success'] for r in index):
            raise SystemExit(1)
    else:
        paths = sorted(args.input.glob('episode_*.npz')) if args.input.is_dir() else [args.input]
        if not paths or any(not p.is_file() for p in paths):
            parser.error('No episode files found')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = []
            for result in pool.map(replay_job, [(p, args.render_dir) for p in paths]):
                results.append(result)
                print(json.dumps(result), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2)+'\n')
        print(f'REPLAY PASS {sum(r["passed"] for r in results)}/{len(results)}')
        if not all(r['passed'] for r in results):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
