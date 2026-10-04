"""Portfolio report for a parallel residual-PPO run (read-only on the run's logs).

usage: recovery_rl_report.py RUN_DIR [--videos] [--condition NAME ...]

Writes RUN_DIR/report/: learning_curves.png, reward_terms.png, eval_matrix.png,
params.png, summary.json and, with --videos, side-by-side evaluation episodes
(zero-residual baseline vs the final checkpoint, same fixed condition) recorded
by scripts/recovery_evidence (video, phase screenshots, contact sheet).
Training-episode success includes exploration noise; only eval.jsonl rows are
deterministic fixed-condition evaluations.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--videos', action='store_true')
    p.add_argument('--condition', action='append', help='eval condition name(s) to film (default: nominal)')
    p.add_argument('--checkpoint', help='checkpoint to film (default: the final one)')
    args = p.parse_args()
    run = Path(args.run)
    out = run/'report'
    out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest = json.loads((run/'manifest.json').read_text())
    updates = list(csv.DictReader((run/'train.csv').open())) if (run/'train.csv').exists() else []
    episodes = read_jsonl(run/'episodes.jsonl')
    evals = read_jsonl(run/'eval.jsonl')
    names = manifest.get('eval_condition_names') or [str(c) for c in manifest['eval_conditions']]

    # Learning curves.
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    if episodes:
        ok = np.array([bool(e.get('success')) for e in episodes], float)
        k = min(20, len(ok))
        roll = np.convolve(ok, np.ones(k)/k, mode='valid')*100
        ax[0, 0].plot(np.arange(k, len(ok)+1), roll, label=f'rolling {k} episodes')
        ax[0, 0].plot(np.arange(1, len(ok)+1), np.cumsum(ok)/np.arange(1, len(ok)+1)*100, alpha=.6, label='cumulative')
        ax[0, 0].legend()
        ret = [e['episode_return'] for e in episodes]
        ax[0, 1].plot(np.arange(1, len(ret)+1), ret, '.', alpha=.5)
        if len(ret) >= 10:
            ax[0, 1].plot(np.arange(10, len(ret)+1), np.convolve(ret, np.ones(10)/10, mode='valid'), label='rolling 10')
            ax[0, 1].legend()
    ax[0, 0].set(title='Training episodes: success (%) — includes exploration noise', xlabel='episode', ylim=(-5, 105))
    ax[0, 1].set(title='Training episode return', xlabel='episode')
    if evals:
        base = [e for e in evals if e['label'].startswith('zero')]
        pol = [e for e in evals if not e['label'].startswith('zero')]
        if base:
            ax[1, 0].axhline(100*base[0]['success_rate'], color='gray', ls='--',
                             label=f"scripted controller (zero residual) {base[0]['successes']}/{base[0]['n']}")
        if pol:
            ax[1, 0].plot([e['steps'] for e in pol], [100*e['success_rate'] for e in pol], 'o-', label='PPO policy (deterministic)')
        if 'reachable_n' in evals[0]:
            rate = lambda e: 100*e['reachable_successes']/max(e['reachable_n'], 1)
            if base:
                ax[1, 0].axhline(rate(base[0]), color='gray', ls=':',
                                 label=f"scripted, hand-fixable conditions {base[0]['reachable_successes']}/{base[0]['reachable_n']}")
            if pol:
                ax[1, 0].plot([e['steps'] for e in pol], [rate(e) for e in pol], 's--', label='PPO, hand-fixable conditions')
        ax[1, 0].legend()
    ax[1, 0].set(title=f'Fixed-condition evaluation ({len(names)} conditions)', xlabel='training policy steps',
                 ylabel='success (%)', ylim=(-5, 105))
    if updates:
        s = [int(r['steps']) for r in updates]
        ax[1, 1].plot(s, [float(r['action_std']) for r in updates], label='action std (normalized)')
        ax[1, 1].plot(s, [float(r['approx_kl']) for r in updates], label='approx KL')
        ax[1, 1].legend()
    ax[1, 1].set(title='Policy exploration and update size', xlabel='training policy steps')
    for a in ax.flat:
        a.grid(alpha=.25)
    fig.suptitle(f"{run.name}: residual PPO on the Line 2 floor->table recovery")
    fig.tight_layout()
    fig.savefig(out/'learning_curves.png', dpi=130)
    plt.close(fig)

    # Reward terms over training episodes.
    if episodes and episodes[0].get('reward_terms'):
        keys = list(episodes[0]['reward_terms'])
        fig, a = plt.subplots(figsize=(11, 4))
        for key in keys:
            v = [e['reward_terms'][key] for e in episodes]
            k = min(10, len(v))
            a.plot(np.arange(k, len(v)+1), np.convolve(v, np.ones(k)/k, mode='valid'), label=key)
        a.set(title='Reward terms per training episode (rolling mean)', xlabel='episode')
        a.legend(ncol=4, fontsize=8)
        a.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out/'reward_terms.png', dpi=130)
        plt.close(fig)

    # Evaluation matrix (checkpoint x condition).
    if evals:
        mat = np.array([[1. if r['success'] else 0. for r in e['results']] for e in evals])
        fig, a = plt.subplots(figsize=(1.1*len(names)+3, .45*len(evals)+1.6))
        a.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        a.set_xticks(range(len(names)), names, rotation=30, ha='right')
        a.set_yticks(range(len(evals)), [f"{e['label']} ({e['successes']}/{e['n']})" for e in evals])
        for j, r in enumerate(evals[0]['results']):
            if r.get('near_table'):
                a.add_patch(plt.Rectangle((j-.5, -.5), 1, len(evals), fill=False, hatch='//', ec='k', lw=0))
        for i, e in enumerate(evals):
            for j, r in enumerate(e['results']):
                a.text(j, i, 'OK' if r['success'] else (r['failure'] or '')[:10], ha='center', va='center', fontsize=6)
        a.set_title('Fixed-condition evaluations (green = verified recovery and line restart; hatched = block near the table)')
        fig.tight_layout()
        fig.savefig(out/'eval_matrix.png', dpi=130)
        plt.close(fig)

    # Efficiency metrics (v4): mean over successful evaluation episodes.
    rows_m = [e for e in evals if e.get('metrics_mean_successful')]
    if rows_m:
        keys = [('flow_seconds', 'crouch->verify time (s)'), ('place_error_m', 'place error (m)'),
                ('place_yaw_rad', 'place yaw error (rad)'), ('max_collision_force_n', 'max torso/pelvis contact (N)'),
                ('collision_impulse_ns', 'torso/pelvis contact impulse (N s)'),
                ('max_hand_object_force_n', 'max hand-block force (N)'), ('max_roll_rad', 'max body roll (rad)'),
                ('mean_time_scale', 'mean pace (x scripted)')]
        keys = [k for k in keys if k[0] in rows_m[0]['metrics_mean_successful']]
        fig, axes = plt.subplots(2, (len(keys)+1)//2, figsize=(3.2*((len(keys)+1)//2), 6.5))
        for a, (k, lab) in zip(axes.flat, keys):
            vals = [e['metrics_mean_successful'].get(k, np.nan) for e in rows_m]
            a.bar(range(len(rows_m)), vals, color=['#9e9e9e']+['#2e7d32']*(len(rows_m)-1))
            a.set_xticks(range(len(rows_m)), [e['label'].replace('zero_residual_baseline', 'scripted').replace('ppo_update_', 'u')
                                              for e in rows_m], rotation=30, fontsize=7)
            a.set_title(lab, fontsize=9)
            for i, v in enumerate(vals):
                a.text(i, v, f'{v:.3g}', ha='center', va='bottom', fontsize=7)
        for a in list(axes.flat)[len(keys):]:
            a.axis('off')
        fig.suptitle('Efficiency on successful fixed-condition evaluations (lower is better except pace)')
        fig.tight_layout()
        fig.savefig(out/'metrics.png', dpi=130)
        plt.close(fig)

    # Parameters table.
    ppo, env = manifest['ppo_config'], manifest['env_config']
    rows = [('algorithm', 'PPO (clipped), residual on the scripted controller'),
            ('action', '%d-D: both hands xyz offset (+-%.0f mm) + squeeze (+-%.0f mm)%s'
             % (manifest['action_dim'], env['position_scale_m']*1000, env['squeeze_scale_m']*1000,
                ' + pace of scripted stages' if manifest['action_dim'] == 5 else '')),
            ('observation', f"{manifest['obs_dim']}-D (robot state, contacts, stage, last actions; 2 frames)"),
            ('policy step', f"{env['action_repeat']} physics ticks = {env['action_repeat']*0.01:.2f} s"),
            ('randomization', f"block start +-{env['part_offset_mm']:.0f} mm, mass +-{100*env['mass_jitter']:.0f} %"),
            ('workers x rollout', f"{manifest['n_workers']} x {ppo['rollout_steps']} = {manifest['samples_per_update']} steps/update"),
            ('learning rate', ppo['learning_rate']), ('epochs / batch', f"{ppo['epochs']} / {ppo['batch_size']}"),
            ('gamma / lambda', f"{ppo['gamma']} / {ppo['gae_lambda']}"), ('clip / target KL', f"{ppo['clip']} / {ppo['target_kl']}"),
            ('initial log std', ppo['init_log_std']), ('entropy coef', ppo['entropy_coef']),
            ('policy window', ', '.join(env.get('policy_stages') or env.get('active_stages') or []) + (' (other stages run through)' if env.get('skip_inactive') else '')),
            ('hand offset stages', ', '.join(env.get('active_stages') or [])),
            ('pace action', f"1 +- {env.get('speed_range')}" if env.get('reward_version') == 'v4' else 'none'),
            ('excluded in training', f"blocks < {1000*env.get('skip_near_table_gap_m', 0):.0f} mm from the table face"
             if env.get('skip_near_table_gap_m') else 'none'),
            ('reward', manifest.get('reward', '')), ('git commit', (manifest.get('git_commit') or '')[:12])]
    import textwrap
    cells = [[k, '\n'.join(textwrap.wrap(str(v), 95))] for k, v in rows]
    fig, a = plt.subplots(figsize=(12, .34*sum(c[1].count('\n')+1 for c in cells)+.6))
    a.axis('off')
    table = a.table(cellText=cells, colWidths=[.18, .82], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_height(.034*(cells[r][1].count('\n')+1)*14/len(cells))
        cell.get_text().set_ha('left')
    fig.tight_layout()
    fig.savefig(out/'params.png', dpi=130)
    plt.close(fig)

    summary = dict(run=run.name, updates=len(updates), policy_steps=int(updates[-1]['steps']) if updates else 0,
                   training_episodes=len(episodes), training_successes=sum(bool(e.get('success')) for e in episodes),
                   wall_hours=float(updates[-1]['wall_seconds'])/3600 if updates else 0.,
                   evaluations=[dict(label=e['label'], steps=e['steps'], success_rate=e['success_rate'],
                                     successes=e['successes'], n=e['n']) for e in evals],
                   note='training success includes exploration noise; evaluations are deterministic on fixed conditions')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.videos:
        cks = sorted((run/'checkpoints').glob('step_*.pt'))
        ck = Path(args.checkpoint) if args.checkpoint else (cks[-1] if cks else None)
        conds = dict(zip(names, manifest['eval_conditions']))
        env_vars = dict(os.environ, MUJOCO_GL='glfw', DISPLAY=os.environ.get('DISPLAY', ':0'),
                        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for name in args.condition or ['nominal']:
            cond = json.dumps(conds[name])
            jobs = [('baseline', ['--zero', '--config-from', str(run/'manifest.json')])]
            if ck is not None:
                jobs.append((ck.stem, ['--checkpoint', str(ck)]))
            for tag, extra in jobs:
                dest = out/'videos'/f'{name}_{tag}'
                if dest.exists():
                    continue
                dest.parent.mkdir(exist_ok=True)
                subprocess.run([sys.executable, str(ROOT/'scripts/recovery_evidence'), 'record', '--output', str(dest),
                                '--backend', 'glfw', '--condition', cond, '--title', f'condition {name}', *extra],
                               env=env_vars, check=False)


if __name__ == '__main__':
    main()
