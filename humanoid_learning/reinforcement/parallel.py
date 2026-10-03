"""Parallel on-policy rollouts for the Line 2 residual env (CPU processes).

Each worker owns one RecoveryResidualEnv and, per update, receives the current
actor-critic weights, collects a fixed number of policy steps with them (resetting
its own env at episode ends, so a slow reset never stalls the other workers) and
returns the transitions. Evaluation runs whole episodes with the deterministic
policy on fixed conditions. The fork start method shares the imported code; each
worker keeps a single env, so memory stays ~1.5 GB per worker.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np


def _worker(conn, env_config, worker_id, base_seed, start_episode=0, max_resets=8):
    import torch
    torch.set_num_threads(1)
    from humanoid_learning.reinforcement.ppo import ActorCritic
    from humanoid_learning.reinforcement.recovery_env import RecoveryResidualEnv, ResidualConfig
    env = RecoveryResidualEnv(ResidualConfig(**env_config))
    episode = start_episode
    resets = 0
    obs = None
    policy = None

    def fresh(seed, options=None):
        # A seeded condition whose scripted prefix fails before LOWER is not a
        # residual-policy sample: skip it (counted) and draw the next seed.
        nonlocal episode, resets
        skipped = []
        while True:
            try:
                resets += 1
                return env.reset(seed=seed, options=options)[0], skipped
            except RuntimeError as exc:
                skipped.append(dict(seed=seed, error=str(exc)))
                if options is not None:
                    raise
                episode += 1
                seed = base_seed + 1000*episode + worker_id

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == 'close':
                break
            if cmd == 'rollout':
                weights, steps = arg
                skipped = []
                if obs is None:
                    obs, skipped = fresh(base_seed + 1000*episode + worker_id)
                policy = ActorCritic(len(obs), 4)
                policy.load_state_dict(weights)
                policy.eval()
                buf = dict(obs=[], raw=[], logp=[], value=[], next_value=[], reward=[], term=[], done=[])
                finished = []
                for _ in range(steps):
                    action, raw, lp, v = policy.act(obs)
                    nxt, reward, term, trunc, info = env.step(action)
                    with torch.no_grad():
                        nv = policy.value(torch.as_tensor(nxt)).item()
                    for key, val in zip(buf, (obs, raw, lp, v, nv, reward, term, term or trunc)):
                        buf[key].append(val)
                    obs = nxt
                    if term or trunc:
                        finished.append(dict(info, worker=worker_id, seed=base_seed + 1000*episode + worker_id,
                                             terminated=term, truncated=trunc))
                        episode += 1
                        obs, more = fresh(base_seed + 1000*episode + worker_id)
                        skipped += more
                # The rollout may end mid-episode: the last transition bootstraps
                # (done marks only the GAE boundary between workers' sequences).
                buf['done'][-1] = True
                # Each env rebuild leaks ~45 MB (measured); retire the process
                # after a few and let the parent fork a fresh one.
                retire = resets >= max_resets
                conn.send(('ok', dict({k: np.asarray(v) for k, v in buf.items()}, finished=finished, skipped=skipped,
                                      retire=retire, next_episode=episode + 1)))
                if retire:
                    break
            elif cmd == 'evaluate':
                weights, condition, seed = arg
                o, _ = fresh(seed, options=dict(condition=condition))
                if weights is not None:
                    policy = ActorCritic(len(o), 4)
                    policy.load_state_dict(weights)
                    policy.eval()
                # The scripted stages before the policy window can already end
                # the episode (e.g. a block near the table): same for any policy.
                term, trunc, info = env.done, False, env._info()
                while not (term or trunc):
                    a = np.zeros(4) if weights is None else policy.act(o, deterministic=True)[0]
                    o, _, term, trunc, info = env.step(a)
                obs = None  # the interrupted training episode restarts with a new seed
                episode += 1
                retire = resets >= max_resets
                conn.send(('ok', dict(info, condition_request=condition, terminated=term, truncated=trunc,
                                      retire=retire, next_episode=episode)))
                if retire:
                    break
    except Exception:
        conn.send(('error', traceback.format_exc()))
    finally:
        env.close()
        conn.close()


class RolloutWorkers:
    def __init__(self, env_config, n, base_seed=0, max_resets=8):
        self.ctx = mp.get_context('fork')
        self.env_config, self.base_seed, self.max_resets = env_config, base_seed, max_resets
        self.conns, self.procs = [None]*n, [None]*n
        self.respawns = 0
        for i in range(n):
            self._spawn(i, 0)

    def _spawn(self, i, start_episode):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=_worker, args=(child, self.env_config, i, self.base_seed, start_episode,
                                                      self.max_resets), daemon=True)
        proc.start()
        child.close()
        self.conns[i], self.procs[i] = parent, proc

    def _gather(self, conns):
        out = []
        for conn in conns:
            status, payload = conn.recv()
            if status != 'ok':
                raise RuntimeError(f'rollout worker failed:\n{payload}')
            out.append(payload)
            if payload.get('retire'):
                i = self.conns.index(conn)
                self.procs[i].join(timeout=60)
                conn.close()
                self._spawn(i, payload['next_episode'])
                self.respawns += 1
        return out

    def rollout(self, weights, steps):
        for conn in self.conns:
            conn.send(('rollout', (weights, steps)))
        return self._gather(self.conns)

    def evaluate(self, weights, conditions, seed=0):
        """Whole deterministic episodes, one per condition, spread over workers."""
        results = [None]*len(conditions)
        pending = list(enumerate(conditions))
        while pending:
            batch, pending = pending[:len(self.conns)], pending[len(self.conns):]
            for (i, cond), conn in zip(batch, self.conns):
                conn.send(('evaluate', (weights, cond, seed)))
            for (i, _), res in zip(batch, self._gather(self.conns[:len(batch)])):
                results[i] = res
        return results

    def close(self):
        for conn in self.conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
