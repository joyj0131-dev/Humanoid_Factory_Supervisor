"""Complete Sharpa commands and independent physical replay, schema version 1.

States are reference measurements only: replay never assigns recorded qpos/qvel.
The grasp POLICY (SharpaBimanualGraspExpert) is imported only inside
collect_episode(), never by replay -- a replay reproduces the trajectory from
the stored commands alone. PhysicalMonitor does import two read-only contact
census helpers that happen to live on SharpaContactLift; they observe forces
and never drive the robot.

Hash scope, stated so the guarantee is not overread: environment_sha256 covers
humanoid_learning/envs/*.py and model_sha256 covers the compiled model, so a
change to either is reported as an explicit incompatibility. The measurement
code above is NOT hashed -- editing it surfaces as a telemetry mismatch during
replay (a loud failure, not a silent pass), not as a version error.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc, task_config as tc
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_command import SharpaGraspCommand
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]


def environment_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'envs').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256(model.names)
    for name in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'geom_type',
                 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref',
                 'geom_solimp', 'geom_contype', 'geom_conaffinity', 'mesh_vert',
                 'mesh_face', 'jnt_axis', 'jnt_range', 'dof_damping', 'dof_armature',
                 'actuator_trnid', 'actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm'):
        digest.update(np.asarray(getattr(model, name)).tobytes())
    return digest.hexdigest()


def targets(env):
    return np.concatenate([env._waist_target, env._arm_target, env._group_synergy])


class PhysicalMonitor:
    """Measure support/clearance without consulting the Expert's success flag."""
    def __init__(self, env):
        self.env = env
        self.object_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, tc.OBJECT_GEOM)
        self.table_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, tc.TABLE_GEOM)
        self.bodies = SharpaContactLift.hand_wrist_body_ids(env)
        self.streak = 0
        self.max_penetration = 0.0
        self.max_hold_penetration = 0.0
        self.clearance = 0.0

    def sample(self):
        env = self.env
        forces = SharpaContactLift.measure_support_forces(env, self.bodies)
        supported = all(np.linalg.norm(f) > .5 for f in forces.values()) and np.dot(
            forces['left'][:2], forces['right'][:2]) < 0
        clearance = float(mujoco.mj_geomDistance(env.model, env.data, self.object_geom,
                                               self.table_geom, 1.0, None))
        penetration = table_force = 0.0
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            if env._object_body_id in (env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]):
                penetration = max(penetration, -float(c.dist))
            if {c.geom1, c.geom2} == {self.object_geom, self.table_geom}:
                f = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, f)
                table_force = max(table_force, float(np.linalg.norm(f[:3])))
        good = supported and clearance >= .05 and table_force == 0.0
        self.streak = self.streak + 1 if good else 0
        self.max_penetration = max(self.max_penetration, penetration)
        if good:
            self.max_hold_penetration = max(self.max_hold_penetration, penetration)
        self.clearance = clearance
        # Same column order is recorded explicitly in metadata.
        return np.array([clearance, float(supported), table_force, penetration,
                         *forces['left'], *forces['right'],
                         *[env._group_contact_force(s, g)[1] for s in sc.SIDES for g in sc.GROUPS]])

    def summary(self):
        # Deliberately the FINAL streak, not the longest one: the object has to
        # still be held when the episode ends, so a mid-episode hold followed by
        # a drop cannot be reported as a successful demonstration.
        seconds = self.streak * self.env.config.frame_skip * self.env.model.opt.timestep
        return dict(physical_success=bool(seconds >= 5.0), final_clearance_m=self.clearance,
                    continuous_air_hold_seconds=seconds,
                    max_object_penetration_m=self.max_penetration,
                    max_air_hold_penetration_m=self.max_hold_penetration)


def collect_episode(path, case, max_steps=8000):
    from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
        SharpaBimanualGraspExpert, BimanualGraspState, BimanualGraspConfig)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    cfg = GraspEnvConfig(arm_gravity_compensation=True, max_episode_steps=max_steps,
        object_pos=(case['x'], case['y'], 0.), object_half_extents=tuple(x / 2 for x in case['size']),
        object_yaw_rad=np.deg2rad(case.get('yaw_deg', 0.)))
    env = SharpaGraspEnv(cfg)
    try:
        obs, _ = env.reset(seed=case['seed'])
        expert = SharpaBimanualGraspExpert(env)
        monitor = PhysicalMonitor(env)
        aux_ids = np.concatenate([env._preshape_act_ids[s] for s in sc.SIDES])
        metadata = dict(schema_version=SCHEMA_VERSION, case=case, config=asdict(cfg),
            expert_config=asdict(BimanualGraspConfig()), mujoco_version=mujoco.__version__,
            environment_sha256=environment_hash(), model_sha256=model_hash(env.model),
            dt=env.config.frame_skip * env.model.opt.timestep,
            actuator_names=[env.model.actuator(i).name for i in range(env.model.nu)],
            preshape_actuator_names=[env.model.actuator(int(i)).name for i in aux_ids],
            telemetry_columns=['clearance_m', 'supported', 'table_force_n', 'penetration_m',
                'left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz',
                *[f'{s}_{g}_force_n' for s in sc.SIDES for g in sc.GROUPS]],
            learner_ready=False, quality_review_required=True,
            observation_alignment='observations[t] precedes commands[t]; T commands, T+1 states')
        rows = {k: [] for k in ('actions', 'preshape', 'noslip', 'states', 'telemetry')}
        reference = dict(observations=[obs.copy()], qpos=[env.data.qpos.copy()],
                         qvel=[env.data.qvel.copy()], ctrl=[env.data.ctrl.copy()],
                         targets=[targets(env)], times=[float(env.data.time)])
        for _ in range(cfg.max_episode_steps):
            before_targets = targets(env)
            before_ctrl = env.data.ctrl.copy()
            action = expert.step()
            # A command may change auxiliary controls/solver settings, never
            # secretly advance physics or change the replay's integrator state.
            np.testing.assert_array_equal(env.data.qpos, reference['qpos'][-1])
            np.testing.assert_array_equal(env.data.qvel, reference['qvel'][-1])
            np.testing.assert_array_equal(targets(env), before_targets)
            assert env.data.time == reference['times'][-1]
            non_aux = np.setdiff1d(np.arange(env.model.nu), aux_ids)
            np.testing.assert_array_equal(env.data.ctrl[non_aux], before_ctrl[non_aux])
            command = env.capture_command(action)
            obs, _, terminated, truncated, _ = env.step_command(command)
            for key, value in [('actions', command.action), ('preshape', command.preshape_targets),
                               ('noslip', command.noslip_iterations), ('states', expert.state.name),
                               ('telemetry', monitor.sample())]:
                rows[key].append(value)
            for key, value in [('observations', obs.copy()), ('qpos', env.data.qpos.copy()),
                               ('qvel', env.data.qvel.copy()), ('ctrl', env.data.ctrl.copy()),
                               ('targets', targets(env)), ('times', float(env.data.time))]:
                reference[key].append(value)
            if terminated or truncated or expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
        metadata['result'] = dict(monitor.summary(), expert_state=expert.state.name,
                                  failure_reason=str(expert.failure_reason), steps=len(rows['actions']))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of user demonstrations.
        with path.open('xb') as f:
            np.savez_compressed(f, metadata=np.array(json.dumps(metadata)),
                                **{k: np.asarray(v) for k, v in (rows | reference).items()})
        return dict(path=str(path), **metadata['result'])
    finally:
        env.close()


def replay_episode(path, render_dir=None):
    """Replay commands only. No Expert construction, IK solve or state injection."""
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if meta['schema_version'] != SCHEMA_VERSION or meta['mujoco_version'] != mujoco.__version__:
        raise ValueError('incompatible recording schema/MuJoCo version')
    if meta['environment_sha256'] != environment_hash():
        raise ValueError('environment code differs from recording; regenerate or explicitly migrate the episode')
    env = SharpaGraspEnv(GraspEnvConfig(**meta['config']))
    try:
        env.reset(seed=meta['case']['seed'])
        if model_hash(env.model) != meta['model_sha256']:
            raise ValueError('compiled model differs from recording')
        aux_ids = np.concatenate([env._preshape_act_ids[s] for s in sc.SIDES])
        if [env.model.actuator(int(i)).name for i in aux_ids] != meta['preshape_actuator_names']:
            raise ValueError('preshape actuator ordering changed')
        n = len(data['actions'])
        if n != meta['result']['steps']:
            raise ValueError('recorded step count does not match the stored commands')
        for name in ('qpos', 'qvel', 'ctrl', 'targets', 'times', 'observations'):
            if len(data[name]) != n + 1:
                raise ValueError(f'invalid {name} alignment')
        for name in ('preshape', 'noslip', 'states', 'telemetry'):
            if len(data[name]) != n:
                raise ValueError(f'invalid {name} alignment')
        monitor = PhysicalMonitor(env)
        errors = {k: 0. for k in ('qpos', 'qvel', 'ctrl', 'targets', 'observation', 'telemetry', 'time')}

        def compare(index, obs):
            values = dict(qpos=env.data.qpos, qvel=env.data.qvel, ctrl=env.data.ctrl,
                          targets=targets(env), observations=obs, times=env.data.time)
            for name, value in values.items():
                key = {'observations': 'observation', 'times': 'time'}.get(name, name)
                difference = float(np.max(np.abs(np.asarray(value) - data[name][index])))
                if not np.isfinite(difference):
                    raise ValueError(f'nonfinite {name} at step {index}')
                errors[key] = max(errors[key], difference)
        compare(0, env._get_obs())
        if max(errors.values()) > 1e-9:
            raise ValueError('reset state differs from recording')
        for t in range(n):
            command = SharpaGraspCommand(data['actions'][t], data['preshape'][t], int(data['noslip'][t]))
            obs, _, term, trunc, _ = env.step_command(command)
            compare(t + 1, obs)
            telemetry = monitor.sample()
            errors['telemetry'] = max(errors['telemetry'], float(np.max(np.abs(telemetry-data['telemetry'][t]))))
            if (term or trunc) and t != n-1:
                raise RuntimeError(f'replay terminated early at {t}')
        matched = max(errors.values()) <= 1e-6
        result = dict(path=str(path), replay_matches=bool(matched), max_errors=errors, **monitor.summary())
        result['passed'] = bool(matched and result['physical_success'] and meta['result']['physical_success'])
        if render_dir is not None:
            from PIL import Image
            render_dir = Path(render_dir)
            render_dir.mkdir(parents=True, exist_ok=True)
            with mujoco.Renderer(env.model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [.27, 0, .86]
                camera.distance, camera.azimuth, camera.elevation = 1.1, 180, -15
                renderer.update_scene(env.data, camera)
                Image.fromarray(renderer.render()).save(render_dir / (Path(path).stem + '.png'))
        return result
    finally:
        env.close()
