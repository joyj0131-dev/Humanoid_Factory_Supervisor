"""Parallel residual PPO on the real Line 2 recovery, with portfolio-grade logs.

Workers (CPU processes, <=4 on a 16 GB machine) each run the full controller and
collect policy steps; the learner updates one actor-critic. Every run directory
keeps manifest.json (all parameters + git commit), train.csv, episodes.jsonl,
eval.jsonl (fixed conditions, zero-residual baseline first) and every checkpoint.
Make figures/videos afterwards with scripts/recovery_rl_report.py.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_learning.reinforcement.parallel import RolloutWorkers
from humanoid_learning.reinforcement.ppo import PPOConfig, train_parallel
from humanoid_learning.reinforcement.recovery_env import ResidualConfig

# Fixed evaluation conditions (dropped-block start offset mm, block mass scale).
EVAL_CONDITIONS = [
    dict(name='nominal', part_offset_mm=[0, 0], part_mass_scale=1.),
    dict(name='x+15', part_offset_mm=[15, 0], part_mass_scale=1.),
    dict(name='x-15', part_offset_mm=[-15, 0], part_mass_scale=1.),
    dict(name='y+15', part_offset_mm=[0, 15], part_mass_scale=1.),
    dict(name='y-15', part_offset_mm=[0, -15], part_mass_scale=1.),
    dict(name='mass-10%', part_offset_mm=[0, 0], part_mass_scale=.9),
    dict(name='mass+10%', part_offset_mm=[0, 0], part_mass_scale=1.1),
    dict(name='diag+10', part_offset_mm=[10, 10], part_mass_scale=1.),
    # Added for v3 (16 conditions: one condition = 6 percentage points).
    dict(name='x+8', part_offset_mm=[8, 0], part_mass_scale=1.),
    dict(name='x-8', part_offset_mm=[-8, 0], part_mass_scale=1.),
    dict(name='y+8', part_offset_mm=[0, 8], part_mass_scale=1.),
    dict(name='y-8', part_offset_mm=[0, -8], part_mass_scale=1.),
    dict(name='mass-5%', part_offset_mm=[0, 0], part_mass_scale=.95),
    dict(name='mass+5%', part_offset_mm=[0, 0], part_mass_scale=1.05),
    dict(name='diag-10', part_offset_mm=[-10, -10], part_mass_scale=1.),
    dict(name='anti+10', part_offset_mm=[10, -10], part_mass_scale=1.),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--rollout', type=int, default=64, help='policy steps per worker per update')
    p.add_argument('--reward-version', choices=['v1', 'v3'], default='v3')
    p.add_argument('--active-stages', default='HOLD,DOWN,RELEASE', help='v3: stages the residual acts in')
    p.add_argument('--skip-near-table-gap-m', type=float, default=.062, help='v3 training: skip blocks this close to the table (0: off)')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--init-log-std', type=float, default=-2.3)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--part-offset-mm', type=float, default=15.)
    p.add_argument('--mass-jitter', type=float, default=.1)
    p.add_argument('--eval-every', type=int, default=50, help='updates between fixed-condition evaluations (0: off)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--min-free-mb', type=int, default=1500, help='pause while system MemAvailable is below this')
    p.add_argument('--max-resets', type=int, default=4, help='env rebuilds per worker process before it is replaced (memory)')
    p.add_argument('--eval-limit', type=int, default=0, help='smoke tests only: evaluate the first N conditions')
    p.add_argument('--resume', help='checkpoint to continue the same run directory from')
    args = p.parse_args()
    if not 1 <= args.workers <= 6:
        p.error('1-6 workers (each needs ~1.5-2 GB of RAM)')
    v3 = args.reward_version == 'v3'
    env_cfg = ResidualConfig(part_offset_mm=args.part_offset_mm, mass_jitter=args.mass_jitter,
                             reward_version=args.reward_version,
                             active_stages=tuple(args.active_stages.split(',')) if v3 else ('HOLD', 'DOWN', 'RELEASE'),
                             skip_inactive=v3, skip_near_table_gap_m=args.skip_near_table_gap_m if v3 else 0.)
    ppo = PPOConfig(total_steps=args.updates*args.workers*args.rollout, rollout_steps=args.rollout,
                    epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    seed=args.seed, init_log_std=args.init_log_std)
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except OSError:
        commit = None
    chosen = EVAL_CONDITIONS[:args.eval_limit] if args.eval_limit else EVAL_CONDITIONS
    conditions = [{k: v for k, v in c.items() if k != 'name'} for c in chosen]
    workers = RolloutWorkers(asdict(env_cfg), args.workers, base_seed=10000*(args.seed+1), max_resets=args.max_resets)
    try:
        obs_dim = 704  # 2 frames x 352 (checked against the env on the first rollout)
        train_parallel(workers, ppo, args.output, env_config=asdict(env_cfg), obs_dim=obs_dim,
                       n_workers=args.workers, updates=args.updates, eval_conditions=conditions,
                       eval_every=args.eval_every, resume=args.resume, min_free_mb=args.min_free_mb,
                       extra_manifest=dict(git_commit=commit, command=sys.argv, max_resets=args.max_resets,
                                           eval_condition_names=[c['name'] for c in chosen],
                                           reward=('v3: v1 terms + grip -0.002*|L-R|/(L+R) per held tick, place 0..10 '
                                                   'at set-down and again after retract (position within 6 cm, yaw within '
                                                   '0.15 rad), disturb -0.01 per tick of hand-block contact while '
                                                   'retracting. v1: +100 verified restart / -25 failure, first-time stage '
                                                   'milestones 2-5, 5*distance gain, -0.001/tick, -0.01/forbidden-contact '
                                                   'tick, -0.002*|a|^2 per active tick, -0.01*|da|^2') if v3 else
                                                  ('v1: +100 verified restart / -25 failure, first-time stage '
                                                   'milestones 2-5, 5*distance gain, -0.001/tick, -0.01/forbidden-contact '
                                                   'tick, -0.002*|a|^2 per active tick, -0.01*|da|^2')))
    finally:
        workers.close()


if __name__ == '__main__':
    main()
