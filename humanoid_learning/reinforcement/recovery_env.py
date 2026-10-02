"""Residual RL around the real Line 2 controller, not learned whole-body control.

Four actions: heading-frame hand xyz offsets and symmetric inward squeeze.
Reset physically runs the scripted prefix (no checkpoint state injection).
Full evaluation disables prefix skipping; both modes use the same physics.
"""
from dataclasses import dataclass
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from humanoid_learning.envs.factory_config import FactoryConfig
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig

SCHEMA = 'line2_hand_residual_v2'
STAGES = ('NONE','CROUCH','LOWER','CLOSE','ALIGN','PICK_CLEAR','RISE','HOLD',
          'DOWN','RELEASE','RETRACT','STAND','READY_TO_VERIFY','BACKOFF','ABORT')
STATES = ('WAIT_FAULT','PREPARE_HANDS','AISLE_APPROACH','TURN','WALK','SETTLE',
          'FLOOR_PICKUP','PLACE','VERIFY','RESTART_OBSERVE','RECOVERED','FAILED')
ACTIVE = {'LOWER','CLOSE','ALIGN','PICK_CLEAR','RISE','HOLD','DOWN'}
MILESTONES = {'CLOSE':2.,'ALIGN':2.,'PICK_CLEAR':3.,'RISE':5.,'DOWN':5.,
              'RELEASE':5.,'RETRACT':3.,'READY_TO_VERIFY':3.}

@dataclass
class ResidualConfig:
    action_repeat: int = 20
    position_scale_m: float = .015
    squeeze_scale_m: float = .004
    smoothing: float = .1
    history: int = 2
    start_at_lower: bool = True
    max_physics_steps: int = 30000
    restart_observe_steps: int = 1500
    mass_jitter: float = 0.
    # Domain randomization per episode (sampled from the reset seed): the
    # dropped block's start offset (uniform per axis, mm). The controller is
    # validated on 'smooth' motion; keep the training scene identical.
    part_offset_mm: float = 0.
    motion_profile: str = 'smooth'

    def __post_init__(self):
        if min(self.action_repeat,self.history,self.max_physics_steps)<1:
            raise ValueError('positive repeat/history/budget required')
        if not 0<self.smoothing<=1 or not 0<=self.mass_jitter<.5 or self.part_offset_mm<0:
            raise ValueError('invalid smoothing or mass jitter')
        if min(self.position_scale_m,self.squeeze_scale_m)<=0:
            raise ValueError('positive residual scales required')

class RecoveryResidualEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None):
        self.config = config or ResidualConfig()
        self.action_space = spaces.Box(-1.,1.,(4,),np.float32)
        self.factory = self.recovery = None
        self.observation_space = None  # determined from compiled model at reset
        self.frames = deque(maxlen=self.config.history)
        self.scale = np.array([self.config.position_scale_m]*3+[self.config.squeeze_scale_m])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()  # controllers change plant tuning; start with a fresh model
        mass = .1*(1+self.np_random.uniform(-self.config.mass_jitter,self.config.mass_jitter))
        offset = self.np_random.uniform(-self.config.part_offset_mm,self.config.part_offset_mm,2)
        forced = (options or {}).get('condition')
        if forced:  # fixed evaluation conditions override the seeded sample
            mass = .1*float(forced.get('part_mass_scale',1.))
            offset = np.asarray(forced.get('part_offset_mm',[0.,0.]),dtype=float)
        self.condition = dict(part_mass_kg=round(float(mass),5),part_offset_mm=np.round(offset,2).tolist())
        self.factory = FactoryEnv(FactoryConfig(part_mass=mass))
        self.factory.reset(seed=seed,options={'fault_workcell':1,'scenario':'arm_drop'})
        if np.any(offset):
            from humanoid_learning.envs import factory_config as fcfg
            body = self.factory.model.body(fcfg.part_body_name(1)).id
            self.factory._set_part(1,self.factory.data.xpos[body]+np.r_[offset/1000.,0.])
        self.recovery = FactoryRecovery(self.factory,RecoveryConfig(motion_profile=self.config.motion_profile,
            floor_pad_grip=True,floor_frog_stance=True,floor_table_place=True,
            restart_observe_steps=self.config.restart_observe_steps,max_steps=self.config.max_physics_steps))
        self.previous_action = np.zeros(4)
        self.filtered = np.zeros(4)
        self.seen = set()
        self.done = False
        self.episode_return = 0.
        self.terms = dict(time=0.,contact=0.,magnitude=0.,smoothness=0.,progress=0.,distance=0.,outcome=0.)
        self.policy_steps = self.residual_ticks = 0
        if self.config.start_at_lower:
            while self._stage()!='LOWER':
                self.recovery.step()
                if self.recovery.state in self.recovery.TERMINAL or self.recovery.total_steps>=self.config.max_physics_steps:
                    raise RuntimeError(f'prefix failed: {self.recovery.state}: {self.recovery.failure}')
        self.prefix_steps = self.recovery.total_steps
        self.previous_distance = self._distance()
        first = self._frame()
        self.frames.clear()
        self.frames.extend(first.copy() for _ in range(self.config.history))
        obs = self._obs()
        self.observation_space = spaces.Box(-np.inf,np.inf,obs.shape,np.float32)
        return obs,self._info()

    def _stage(self):
        return getattr(getattr(self.recovery,'floor_pickup',None),'stage','NONE')

    def _distance(self):
        f=self.factory
        target=np.r_[f.poses[1].canonical_part_xy,f._part_rest_pos(1)[2]]
        return float(np.linalg.norm(f.part_position(1)-target))

    def _frame(self):
        f,r=self.factory,self.recovery
        floor=getattr(r,'floor_pickup',None)
        loads=getattr(floor,'normal_loads',{})
        squeeze=getattr(floor,'squeeze',{})
        _,groups=r.grasp._group_contact_forces()
        feet=r._last_info.get('floor_ground_foot_loads_n') or [0.,0.]
        frame=np.r_[f.data.qpos,.1*f.data.qvel,f.data.ctrl,f.poses[1].canonical_part_xy,
            [float(r.state==s) for s in STATES],[float(self._stage()==s) for s in STAGES],
            r.ticks/30000,getattr(floor,'tick',0)/10000,getattr(floor,'stage_ticks',0)/10000,
            f.task_manager.recovery_progress(f),[loads.get(s,0.)/10 for s in ('left','right')],
            [squeeze.get(s,0.) for s in ('left','right')],getattr(floor,'block_bias',np.zeros(3)),
            groups/10,np.asarray(feet)/400,self.filtered/self.scale,self.previous_action]
        if not np.isfinite(frame).all():
            raise FloatingPointError('nonfinite policy observation')
        return frame.astype(np.float32)

    def _obs(self):
        return np.concatenate(self.frames).astype(np.float32)

    def _info(self):
        r=self.recovery
        return dict(schema=SCHEMA,recovery_state=r.state,floor_stage=self._stage(),
            failure=r.failure,success=self._success(),physics_steps=r.total_steps,
            prefix_steps=self.prefix_steps,target_distance_m=self._distance(),
            episode_return=self.episode_return,policy_steps=self.policy_steps,residual_ticks=self.residual_ticks,
            condition=getattr(self,'condition',None),reward_terms=dict(getattr(self,'terms',{})),
            applied_residual_calls=getattr(getattr(r,'floor_pickup',None),'rl_applied_calls',0),
            max_penetration_m=r.max_object_hand_penetration_m,
            forbidden_contact_ticks=r.forbidden_contact_ticks,restart_report=getattr(r,'restart_report',None))

    def _success(self):
        r,f=self.recovery,self.factory
        verified=(r.state=='RECOVERED' and f.task_manager.recovered_step is not None
                  and f.task_manager.state.name=='RUNNING' and f.belt.line_running[1])
        if self.config.restart_observe_steps:
            report=getattr(r,'restart_report',{})
            verified=verified and report.get('arm_travel_rad',0.)>.01 and not report.get('arm_faulted',True)
        return bool(verified)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required')
        action=np.asarray(action,dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1.000001):
            raise ValueError('expected finite bounded action (4,)')
        action=np.clip(action,-1,1)
        terms=dict(time=0.,contact=0.,magnitude=0.,smoothness=0.,progress=0.,distance=0.,outcome=0.)
        for _ in range(self.config.action_repeat):
            floor=getattr(self.recovery,'floor_pickup',None)
            active=floor is not None and self._stage() in ACTIVE
            target=action*self.scale if active else np.zeros(4)
            self.filtered+=self.config.smoothing*(target-self.filtered)
            if floor is not None:
                floor.rl_hand_residual_m=self.filtered.copy() if active else np.zeros(4)
            if active:
                self.residual_ticks+=1
                terms['magnitude']-=.002*float(np.square(action).mean())
            before=self.recovery.forbidden_contact_ticks
            self.recovery.step()
            terms['time']-=.001
            terms['contact']-=.01*(self.recovery.forbidden_contact_ticks-before)
            stage=self._stage()
            if stage in MILESTONES and stage not in self.seen:
                terms['progress']+=MILESTONES[stage]
                self.seen.add(stage)
            if self.recovery.state in self.recovery.TERMINAL or self.recovery.total_steps>=self.config.max_physics_steps:
                break
        distance=self._distance()
        terms['distance']=5*(self.previous_distance-distance)
        self.previous_distance=distance
        timeout=(self.recovery.total_steps>=self.config.max_physics_steps
                 and (self.recovery.failure is None or self.recovery.failure.endswith('_TIMEOUT'))
                 and self.recovery.state!='RECOVERED')
        terminated=self.recovery.state in self.recovery.TERMINAL and not timeout
        truncated=timeout
        if terminated:
            terms['outcome']=100. if self._success() else -25.
        terms['smoothness']=-.01*float(np.square(action-self.previous_action).mean())
        reward=sum(terms.values())
        for k,v in terms.items():self.terms[k]+=v
        self.previous_action=action.copy()
        self.policy_steps+=1
        self.episode_return+=reward
        self.done=terminated or truncated
        self.frames.append(self._frame())
        info=self._info();info['truncated']=truncated
        return self._obs(),float(reward),terminated,truncated,info

    def close(self):
        if self.recovery is not None:
            self.recovery.close();self.recovery=None
        if self.factory is not None:
            self.factory.close();self.factory=None
