"""Train/evaluate controller-assisted Line 2 residual PPO, or smoke-test reach."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from humanoid_learning.reinforcement.ppo import PPOConfig,train,load_policy
from humanoid_learning.reinforcement.recovery_env import RecoveryResidualEnv,ResidualConfig

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--task',choices=['recovery','reach'],default='recovery')
    p.add_argument('--output',required=True,help='new directory (training) or JSON file (evaluation)')
    p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--rollout',type=int,default=256)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--checkpoint')
    p.add_argument('--zero',action='store_true',help='evaluate the existing controller with zero correction')
    p.add_argument('--full-episode',action='store_true',help='include fault/walk prefix in policy observations')
    p.add_argument('--action-repeat',type=int,default=20)
    p.add_argument('--mass-jitter',type=float,default=0.)
    args=p.parse_args()
    if args.mode=='evaluate' and (bool(args.checkpoint)==bool(args.zero)):
        p.error('evaluation requires exactly one of --checkpoint or --zero')
    cfg=ResidualConfig(start_at_lower=not args.full_episode,action_repeat=args.action_repeat,mass_jitter=args.mass_jitter)
    if args.task=='recovery':
        env=RecoveryResidualEnv(cfg);contract=asdict(cfg)
    else:
        from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
        from humanoid_learning.envs.task_config import EnvConfig
        env=BimanualReachEnv(EnvConfig.from_yaml(Path(__file__).resolve().parents[1]/'configs/environment.yaml'))
        contract={'foundation':'unchanged'}
    try:
        if args.mode=='train':
            train(env,PPOConfig(total_steps=args.steps,rollout_steps=args.rollout,seed=args.seed),args.output,
                  task=args.task,env_config=contract)
        else:
            path=Path(args.output)
            if path.exists():raise FileExistsError(path)
            records=[]
            for episode in range(args.episodes):
                obs,info=env.reset(seed=args.seed+episode)
                policy=None
                if args.checkpoint:policy,_=load_policy(args.checkpoint,len(obs),env.action_space.shape[0],args.task,contract)
                total=0.
                while True:
                    action=np.zeros(env.action_space.shape) if policy is None else policy.act(obs,True)[0]
                    obs,reward,term,trunc,info=env.step(action);total+=reward
                    if term or trunc:break
                records.append(dict(info,seed=args.seed+episode,return_=total,terminated=term,truncated=trunc))
                print(json.dumps(records[-1],default=str),flush=True)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(task=args.task,config=contract,baseline=args.zero,episodes=records),indent=2,default=str))
    finally:env.close()

if __name__=='__main__':main()
