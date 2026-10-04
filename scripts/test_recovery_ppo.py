"""Fast numerical contracts; --physics additionally checks real residual authority."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import tempfile
import numpy as np
import torch
from humanoid_learning.reinforcement.ppo import ActorCritic,advantages,source_fingerprint,load_policy
from humanoid_learning.reinforcement.recovery_env import RecoveryResidualEnv,ResidualConfig

def test_timeout_bootstrap():
    # Truncation bootstraps but must not leak advantages from the reset episode.
    a,r=advantages([1.,2.],[0.,0.],[10.,20.],[False,True],[True,True],.9,.95)
    np.testing.assert_allclose(a,[10.,2.]);np.testing.assert_allclose(r,a)

def test_rollout_bootstrap():
    a,_=advantages([0.,1.],[2.,3.],[3.,4.],[False,False],[False,False],1.,1.)
    np.testing.assert_allclose(a,[3.,2.])

def test_policy_bounds_and_update():
    torch.manual_seed(0);p=ActorCritic(10,4)
    action,raw,lp,v=p.act(np.ones(10,dtype=np.float32))
    assert np.isfinite([lp,v]).all() and np.max(np.abs(action))<1
    np.testing.assert_array_equal(p.act(np.ones(10,dtype=np.float32),True)[0],np.zeros(4))
    old=p.actor[-1].weight.detach().clone();opt=torch.optim.Adam(p.parameters(),lr=1e-3)
    loss=-p.distribution(torch.ones(2,10)).log_prob(torch.ones(2,4)).mean()
    opt.zero_grad();loss.backward();opt.step();assert not torch.equal(old,p.actor[-1].weight)

def test_checkpoint_roundtrip_and_contract():
    p=ActorCritic(10,4);cfg=asdict(ResidualConfig())
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'p.pt'
        torch.save(dict(obs_dim=10,action_dim=4,task='recovery',env_config=cfg,
                        source_sha256=source_fingerprint(),policy=p.state_dict()),path)
        other,_=load_policy(path,10,4,'recovery',dict(cfg,start_at_lower=False))
        np.testing.assert_array_equal(p.act(np.ones(10,dtype=np.float32),True)[0],other.act(np.ones(10,dtype=np.float32),True)[0])
        try:load_policy(path,10,4,'recovery',dict(cfg,action_repeat=4))
        except ValueError:pass
        else:raise AssertionError('action period mismatch accepted')

def test_real_residual():
    # v1 window (from LOWER) to check authority through the whole-body reach too.
    e=RecoveryResidualEnv(ResidualConfig(reward_version='v1',skip_inactive=False,skip_near_table_gap_m=0.))
    try:
        obs,info=e.reset(seed=0)
        assert e.observation_space.contains(obs) and info['floor_stage']=='LOWER'
        f=e.recovery.floor_pickup
        palms=f._commanded_palms();rot=[f._carry_fk.site_xmat[s].reshape(3,3).copy() for s in f.posture.palms]
        f.rl_hand_residual_m=None;a=f._servo_arms(palms,rot)
        f.rl_hand_residual_m=np.zeros(4);b=f._servo_arms(palms,rot)
        np.testing.assert_array_equal(a,b)
        f.rl_hand_residual_m=np.array([.003,0,0,0]);c=f._servo_arms(palms,rot)
        assert np.max(np.abs(c-b))>1e-6,'nonzero residual has no actuator authority'
        obs,reward,t,tr,info=e.step(np.array([.1,0,0,.1]))
        assert np.isfinite(obs).all() and np.isfinite(reward) and info['applied_residual_calls']>1
        assert not info['success'],'short smoke must not claim recovery'
        for invalid in ([0,0], [np.nan,0,0,0], [2,0,0,0]):
            try:e.step(invalid)
            except ValueError:pass
            else:raise AssertionError('invalid action accepted')
        print('physics contract:',obs.shape,info,flush=True)
    finally:e.close()

def test_v3_window_and_terms():
    e=RecoveryResidualEnv(ResidualConfig())
    try:
        obs,info=e.reset(seed=0,options=dict(condition=dict(part_offset_mm=[0,0],part_mass_scale=1.)))
        assert info['floor_stage'] in e.config.active_stages and not info['near_table'],info
        for _ in range(3):
            obs,reward,t,tr,info=e.step(np.array([.2,0,0,.1]))
            assert np.isclose(sum(e.last_terms.values()),reward) and np.isfinite(obs).all()
        assert info['residual_ticks']>0 and not info['success']
        print('v3 contract:',info['floor_stage'],info['reward_terms'],flush=True)
    finally:e.close()

def test_v4_pace_and_metrics():
    e=RecoveryResidualEnv(ResidualConfig(reward_version='v4',start_stage='CROUCH',skip_near_table_gap_m=0.))
    try:
        obs,info=e.reset(seed=0,options=dict(condition=dict(part_offset_mm=[0,0],part_mass_scale=1.)))
        assert e.action_space.shape==(5,) and info['floor_stage']=='CROUCH',info['floor_stage']
        f=e.recovery.floor_pickup;t0=f.tick
        for _ in range(3):
            obs,reward,t,tr,info=e.step(np.array([0,0,0,0,1.]))
            assert np.isclose(sum(e.last_terms.values()),reward)
        assert f.rl_time_scale>1. and f.tick-t0>60,(f.rl_time_scale,f.tick-t0)  # faster than 60 scripted ticks
        assert info['metrics']['flow_seconds']>0 and info['metrics']['mean_time_scale']>1.
        print('v4 contract:',f.tick-t0,info['metrics'],flush=True)
    finally:e.close()

if __name__=='__main__':
    tests=[test_timeout_bootstrap,test_rollout_bootstrap,test_policy_bounds_and_update,test_checkpoint_roundtrip_and_contract]
    if '--physics' in sys.argv:tests+=[test_real_residual,test_v3_window_and_terms,test_v4_pace_and_metrics]
    for test in tests:test();print('PASS',test.__name__,flush=True)
