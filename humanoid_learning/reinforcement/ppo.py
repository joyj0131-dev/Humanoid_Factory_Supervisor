"""Small on-policy PPO implementation with timeout-aware GAE.

Latent Gaussian actions are squashed by tanh. Ratios use latent log-probability:
the identical tanh Jacobian cancels between old and new policies. The entropy
bonus is explicitly latent Gaussian entropy, not bounded-action entropy.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import csv
import json
import time
import numpy as np
import torch
from torch import nn

@dataclass
class PPOConfig:
    total_steps: int = 4096
    rollout_steps: int = 256
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-5
    gamma: float = .9995
    gae_lambda: float = .95
    clip: float = .2
    entropy_coef: float = .001
    target_kl: float = .03
    seed: int = 0
    init_log_std: float = -3.

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, init_log_std=-3.):
        super().__init__()
        def trunk(out):
            return nn.Sequential(nn.Linear(obs_dim,128),nn.LayerNorm(128),nn.Tanh(),
                                 nn.Linear(128,128),nn.Tanh(),nn.Linear(128,out))
        self.actor,self.critic=trunk(action_dim),trunk(1)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        self.log_std=nn.Parameter(torch.full((action_dim,),float(init_log_std)))

    def distribution(self, obs):
        obs=obs.clamp(-20,20)
        return torch.distributions.Normal(self.actor(obs),self.log_std.clamp(-5,0).exp())

    def value(self, obs):
        return self.critic(obs.clamp(-20,20)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x=torch.as_tensor(obs,dtype=torch.float32)
        dist=self.distribution(x)
        raw=dist.mean if deterministic else dist.sample()
        return raw.tanh().numpy(),raw.numpy(),dist.log_prob(raw).sum(-1).item(),self.value(x).item()

def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Bootstrap time limits, but never propagate GAE through an env reset."""
    adv=np.zeros(len(rewards),dtype=np.float32);tail=0.
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*(1-float(terminated[t]))*next_values[t]-values[t]
        tail=delta+gamma*lam*(1-float(done[t]))*tail
        adv[t]=tail
    return adv,adv+np.asarray(values)

def source_fingerprint():
    root=Path(__file__).resolve().parents[2]
    h=hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        h.update(str(path.relative_to(root)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def load_policy(path, obs_dim, action_dim, task, env_config):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    for key,expected in [('obs_dim',obs_dim),('action_dim',action_dim),('task',task)]:
        if ck[key]!=expected:raise ValueError(f'checkpoint {key} mismatch')
    if ck['source_sha256']!=source_fingerprint():
        raise ValueError('source fingerprint changed: re-evaluate/retrain as a new experiment')
    # Full-episode evaluation changes reset timing only, not the policy contract.
    # Run modes (where the episode starts, run-through filming, training-only
    # skipping) change sampling, not the policy's inputs or actions.
    left,right=dict(ck['env_config']),dict(env_config)
    for cfg in (left,right):
        for key in ('start_at_lower','skip_inactive','skip_near_table_gap_m'):cfg.pop(key,None)
        for key,val in cfg.items():
            if isinstance(val,list):cfg[key]=tuple(val)
    if left!=right:raise ValueError('checkpoint environment configuration mismatch')
    p=ActorCritic(obs_dim,action_dim);p.load_state_dict(ck['policy']);p.eval()
    return p,ck

def train(env, config, output, *, task, env_config):
    if min(config.total_steps,config.rollout_steps,config.epochs,config.batch_size)<1:
        raise ValueError('training budgets must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.random.seed(config.seed);torch.manual_seed(config.seed);torch.set_num_threads(1)
    obs,reset_info=env.reset(seed=config.seed)
    policy=ActorCritic(len(obs),env.action_space.shape[0])
    optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
    manifest=dict(task=task,env_config=env_config,ppo_config=asdict(config),
                  obs_dim=len(obs),action_dim=env.action_space.shape[0],
                  source_sha256=source_fingerprint(),reset_info=json.loads(json.dumps(reset_info,default=str)))
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    steps=episodes=0;started=time.monotonic()
    with (output/'train.csv').open('w',newline='') as log,(output/'episodes.jsonl').open('w') as ep:
        writer=csv.DictWriter(log,fieldnames=['steps','episodes','reward_mean','loss','approx_kl','wall_seconds'])
        writer.writeheader()
        while steps<config.total_steps:
            observations=[];raws=[];logps=[];values=[];next_values=[];rewards=[];terms=[];dones=[]
            count=min(config.rollout_steps,config.total_steps-steps)
            for _ in range(count):
                action,raw,lp,v=policy.act(obs)
                nxt,reward,term,trunc,info=env.step(action)
                with torch.no_grad():nv=policy.value(torch.as_tensor(nxt)).item()
                observations.append(obs);raws.append(raw);logps.append(lp);values.append(v)
                next_values.append(nv);rewards.append(reward);terms.append(term);dones.append(term or trunc)
                obs=nxt;steps+=1
                if term or trunc:
                    episodes+=1
                    ep.write(json.dumps(dict(info,terminated=term,truncated=trunc),default=str)+'\n');ep.flush()
                    if steps<config.total_steps:
                        obs,_=env.reset(seed=config.seed+episodes)
            adv,ret=advantages(rewards,values,next_values,terms,dones,config.gamma,config.gae_lambda)
            adv=(adv-adv.mean())/(adv.std()+1e-8)
            x=torch.as_tensor(np.asarray(observations));u=torch.as_tensor(np.asarray(raws))
            old=torch.tensor(logps);a=torch.tensor(adv);target=torch.tensor(ret,dtype=torch.float32)
            for _ in range(config.epochs):
                for ids in np.array_split(np.random.permutation(count),max(1,int(np.ceil(count/config.batch_size)))):
                    dist=policy.distribution(x[ids]);lp=dist.log_prob(u[ids]).sum(-1)
                    ratio=(lp-old[ids]).exp()
                    actor=-torch.minimum(ratio*a[ids],ratio.clamp(1-config.clip,1+config.clip)*a[ids]).mean()
                    value=.5*(policy.value(x[ids])-target[ids]).square().mean()
                    loss=actor+.5*value-config.entropy_coef*dist.entropy().sum(-1).mean()
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                with torch.no_grad():
                    logratio=policy.distribution(x).log_prob(u).sum(-1)-old
                    kl=float((logratio.exp()-1-logratio).mean())
                if kl>config.target_kl:break
            row=dict(steps=steps,episodes=episodes,reward_mean=float(np.mean(rewards)),loss=float(loss.detach()),
                     approx_kl=kl,wall_seconds=time.monotonic()-started)
            writer.writerow(row);log.flush();print(json.dumps(row),flush=True)
            checkpoint=dict(manifest,policy=policy.state_dict(),optimizer=optimizer.state_dict(),steps=steps)
            # Not an exact environment-resume checkpoint: no simulator/controller state stored.
            temp=output/'checkpoint.tmp';torch.save(checkpoint,temp);temp.replace(output/'policy.pt')
    return policy


def _update(policy, optimizer, config, x, u, old, a, target):
    """PPO epochs on one batch; returns loss pieces for the log."""
    count=len(x);stats=dict(policy_loss=0.,value_loss=0.,entropy=0.,clip_frac=0.);n=0;kl=0.
    for _ in range(config.epochs):
        for ids in np.array_split(np.random.permutation(count),max(1,int(np.ceil(count/config.batch_size)))):
            dist=policy.distribution(x[ids]);lp=dist.log_prob(u[ids]).sum(-1)
            ratio=(lp-old[ids]).exp()
            actor=-torch.minimum(ratio*a[ids],ratio.clamp(1-config.clip,1+config.clip)*a[ids]).mean()
            value=.5*(policy.value(x[ids])-target[ids]).square().mean()
            entropy=dist.entropy().sum(-1).mean()
            loss=actor+.5*value-config.entropy_coef*entropy
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
            stats['policy_loss']+=float(actor);stats['value_loss']+=float(value);stats['entropy']+=float(entropy)
            stats['clip_frac']+=float(((ratio-1).abs()>config.clip).float().mean());n+=1
        with torch.no_grad():
            logratio=policy.distribution(x).log_prob(u).sum(-1)-old
            kl=float((logratio.exp()-1-logratio).mean())
        if kl>config.target_kl:break
    return {k:v/max(n,1) for k,v in stats.items()}|dict(approx_kl=kl)


def _available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    return float('inf')


def wait_for_memory(min_free_mb, log=print):
    """Pause (never kill) while the machine is short of memory: the run shares
    a desktop with other work, and an OOM kill took the editor down once."""
    paused=0.
    while _available_mb() < min_free_mb:
        if paused == 0.:
            log(json.dumps(dict(paused_for_memory_mb=round(_available_mb()))))
        time.sleep(30);paused+=30
    return paused


def train_parallel(workers, config, output, *, env_config, obs_dim, n_workers, updates, eval_conditions,
                   eval_every=0, checkpoint_every=1, resume=None, extra_manifest=None, min_free_mb=1500, action_dim=4):
    """Collect config.rollout_steps per worker per update in parallel, then PPO.

    Writes manifest.json (all parameters), train.csv (per update), episodes.jsonl
    (every finished training episode with its sampled condition and reward terms),
    eval.jsonl (deterministic fixed-condition evaluations, zero-residual baseline
    first) and checkpoints/step_*.pt (never overwritten)."""
    output=Path(output);ck_dir=output/'checkpoints'
    np.random.seed(config.seed);torch.manual_seed(config.seed);torch.set_num_threads(1)
    policy=ActorCritic(obs_dim,action_dim,config.init_log_std)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate)
    steps=episodes=successes=start_update=0
    if resume:
        ck=torch.load(resume,map_location='cpu',weights_only=True)
        policy.load_state_dict(ck['policy']);optimizer.load_state_dict(ck['optimizer'])
        steps,episodes,successes,start_update=ck['steps'],ck['episodes'],ck['successes'],ck['update']
    else:
        output.mkdir(parents=True,exist_ok=False);ck_dir.mkdir()
        manifest=dict(task='recovery',env_config=env_config,ppo_config=asdict(config),obs_dim=obs_dim,action_dim=action_dim,
                      n_workers=n_workers,updates=updates,samples_per_update=n_workers*config.rollout_steps,
                      eval_every=eval_every,eval_conditions=eval_conditions,source_sha256=source_fingerprint(),
                      created_at=time.time(),**(extra_manifest or {}))
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    base=dict(task='recovery',env_config=env_config,ppo_config=asdict(config),obs_dim=obs_dim,action_dim=action_dim,
              source_sha256=source_fingerprint())
    fields=['update','steps','episodes','update_episodes','update_successes','success_rate_update','success_rate_cum',
            'reward_mean','episode_return_mean','policy_loss','value_loss','entropy','clip_frac','approx_kl',
            'action_std','skipped_resets','memory_pause_s','available_mb','wall_seconds']
    new_log=not (output/'train.csv').exists()
    started=time.monotonic()

    def evaluate(update,label,weights):
        wait_for_memory(min_free_mb,lambda m:print(m,flush=True))
        t0=time.monotonic()
        res=workers.evaluate(weights,eval_conditions,seed=config.seed)
        ok=[bool(r['success']) for r in res]
        reach=[bool(r['success']) for r in res if not r.get('near_table')]
        row=dict(update=update,steps=steps,label=label,success_rate=float(np.mean(ok)),successes=int(sum(ok)),n=len(ok),
                 reachable_successes=int(sum(reach)),reachable_n=len(reach),
                 wall_seconds=time.monotonic()-t0,results=[dict(condition=r.get('condition_request'),success=r['success'],
                 state=r['recovery_state'],failure=r['failure'],stage=r['floor_stage'],return_=r['episode_return'],
                 physics_steps=r['physics_steps'],reward_terms=r.get('reward_terms'),table_gap_m=r.get('table_gap_m'),
                 near_table=r.get('near_table'),ended_before_policy=r.get('ended_before_policy'),
                 metrics=r.get('metrics')) for r in res])
        good=[r['metrics'] for r in res if r['success'] and r.get('metrics')]
        if good:
            row['metrics_mean_successful']={k:float(np.mean([g[k] for g in good if g.get(k) is not None]))
                                            for k in good[0] if any(g.get(k) is not None for g in good)}
        with (output/'eval.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
        print(json.dumps(dict(eval=label,update=update,success_rate=row['success_rate'],successes=row['successes'],n=row['n'],
                              reachable=f"{row['reachable_successes']}/{row['reachable_n']}")),flush=True)

    if not resume and eval_every:
        evaluate(0,'zero_residual_baseline',None)
    with (output/'train.csv').open('a',newline='') as log,(output/'episodes.jsonl').open('a') as ep:
        writer=csv.DictWriter(log,fieldnames=fields)
        if new_log:writer.writeheader()
        for update in range(start_update+1,updates+1):
            paused=wait_for_memory(min_free_mb,lambda m:print(m,flush=True))
            batches=workers.rollout({k:v.clone() for k,v in policy.state_dict().items()},config.rollout_steps)
            obs=[];raws=[];logps=[];advs=[];rets=[];rewards=[];finished=[];skipped=0
            for b in batches:
                adv,ret=advantages(b['reward'],b['value'],b['next_value'],b['term'],b['done'],config.gamma,config.gae_lambda)
                obs.append(b['obs']);raws.append(b['raw']);logps.append(b['logp']);advs.append(adv);rets.append(ret)
                rewards.append(b['reward']);finished+=b['finished'];skipped+=len(b['skipped'])
                with (output/'skipped.jsonl').open('a') as sk:
                    for item in b['skipped']:sk.write(json.dumps(dict(item,update=update))+'\n')
            adv=np.concatenate(advs);adv=(adv-adv.mean())/(adv.std()+1e-8)
            x=torch.as_tensor(np.concatenate(obs),dtype=torch.float32);u=torch.as_tensor(np.concatenate(raws),dtype=torch.float32)
            old=torch.as_tensor(np.concatenate(logps),dtype=torch.float32)
            a=torch.as_tensor(adv,dtype=torch.float32);target=torch.as_tensor(np.concatenate(rets),dtype=torch.float32)
            stats=_update(policy,optimizer,config,x,u,old,a,target)
            steps+=len(x);episodes+=len(finished);wins=sum(bool(e.get('success')) for e in finished);successes+=wins
            for e in finished:ep.write(json.dumps(dict(e,update=update),default=str)+'\n')
            ep.flush()
            row=dict(update=update,steps=steps,episodes=episodes,update_episodes=len(finished),update_successes=wins,
                     success_rate_update=(wins/len(finished)) if finished else '',
                     success_rate_cum=(successes/episodes) if episodes else '',
                     reward_mean=float(np.concatenate(rewards).mean()),
                     episode_return_mean=float(np.mean([e['episode_return'] for e in finished])) if finished else '',
                     action_std=float(policy.log_std.clamp(-5,0).exp().mean()),skipped_resets=skipped,
                     memory_pause_s=paused,available_mb=round(_available_mb()),
                     wall_seconds=time.monotonic()-started,**stats)
            writer.writerow(row);log.flush();print(json.dumps(row),flush=True)
            checkpoint=dict(base,policy=policy.state_dict(),optimizer=optimizer.state_dict(),steps=steps,update=update,
                            episodes=episodes,successes=successes)
            if update%checkpoint_every==0 or update==updates:
                torch.save(checkpoint,ck_dir/f'step_{steps:08d}.pt')
            tmp=output/'checkpoint.tmp';torch.save(checkpoint,tmp);tmp.replace(output/'policy.pt')
            if eval_every and (update%eval_every==0 or update==updates):
                evaluate(update,f'ppo_update_{update}',{k:v.clone() for k,v in policy.state_dict().items()})
    return policy
