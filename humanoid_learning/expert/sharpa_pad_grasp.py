"""Measured elastomer-face targets for an extended-finger bilateral clamp.

The vendored DP frame has its length along +X and its rubber face on +Y
(verified against both the rubber and hard-shell mesh vertices below).
No material, mass, collision flag or live joint position is changed here.
"""
import mujoco
import numpy as np
from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import _wahba_R


def upper_side_patch(rotation, half_size, local_point):
    """Upper side-face strip even when a dropped box rests on local X/Y."""
    patch = np.clip(local_point, -.7*half_size, .7*half_size)
    up_axis = int(np.argmax(np.abs(rotation[2])))
    sign = np.sign(rotation[2, up_axis])
    patch[up_axis] = sign*np.clip(sign*local_point[up_axis],
                                  .3*half_size[up_axis], .7*half_size[up_axis])
    return patch, up_axis


class SharpaPadGrasp:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.geoms, self.bodies, self.points = {}, {}, {}
        self.original_open = {s: [v.copy() for v in env._group_open[s]] for s in sc.SIDES}
        self.original_close = {s: [v.copy() for v in env._group_close[s]] for s in sc.SIDES}
        self.pad_open = {s: [v.copy() for v in env._group_open[s]] for s in sc.SIDES}
        self.pad_close = {s: [v.copy() for v in env._group_close[s]] for s in sc.SIDES}
        for side in sc.SIDES:
            for group in range(1, 4):
                index = 0
                for finger in sc.GROUP_FINGERS[sc.GROUPS[group]]:
                    for suffix in sc.curl_suffixes(finger):
                        j = m.joint(sc.sharpa_joint(side, finger, suffix))
                        # Articulate at the knuckle; keep the two distal
                        # joints extended so the rubber face is not a hook.
                        close = {'MCP_FE': .60, 'PIP': .10, 'DIP': .08}[suffix]
                        self.pad_open[side][group][index] = np.clip(0., *j.range)
                        self.pad_close[side][group][index] = np.clip(close, *j.range)
                        index += 1
            for finger in ('index', 'middle', 'ring', 'pinky'):
                gid = m.geom(f'{side}_{side}_{finger}_elastomer').id
                shell = m.geom(f'{side}_{side}_{finger}_DP').id
                vertices, hard = self._vertices(gid), self._vertices(shell)
                if vertices[:, 1].mean() <= hard[:, 1].mean() + .003:
                    raise ValueError('unverified Sharpa pad-face convention')
                front = vertices[vertices[:, 1] >= np.quantile(vertices[:, 1], .8)]
                self.geoms[side, finger] = gid
                self.bodies[side, finger] = int(m.geom_bodyid[gid])
                self.points[side, finger] = front.mean(axis=0)
        self.geom_side = {gid: side for (side, _), gid in self.geoms.items()}

    def _vertices(self, gid):
        m = self.env.model
        mid = m.geom_dataid[gid]
        verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[gid])
        return verts @ R.reshape(3, 3).T + m.geom_pos[gid]

    def point(self, side, finger, data=None):
        d = self.env.data if data is None else data
        b = self.bodies[side, finger]
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.points[side, finger]

    def normal(self, side, finger, data=None):
        d = self.env.data if data is None else data
        return d.xmat[self.bodies[side, finger]].reshape(3, 3)[:, 1].copy()

    def reach_targets(self, expert, gap=.120):
        m = self.env.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.env.data.qpos
        for side in sc.SIDES:
            for group in range(1, 4):
                d.qpos[self.env._group_qpos_adr[side][group]] = (
                    .2*self.pad_open[side][group]+.8*self.pad_close[side][group])
            for finger in sc.FINGERS:
                for suffix in sc.preshape_suffixes(finger):
                    j = m.joint(sc.sharpa_joint(side, finger, suffix))
                    d.qpos[j.qposadr[0]] = (0. if finger == 'thumb' else sc.PRESHAPE_TARGETS[finger][suffix])
        mujoco.mj_forward(m, d)
        obj = m.geom(self.env.object_geom_name).id
        object_R = d.geom_xmat[obj].reshape(3, 3)
        goals, rotations = [], []
        for side in sc.SIDES:
            palm = self.env._left_palm_site if side == 'left' else self.env._right_palm_site
            palm_R = d.site_xmat[palm].reshape(3, 3)
            b = self.bodies[side, 'middle']
            local_n = palm_R.T @ self.normal(side, 'middle', d)
            local_long = palm_R.T @ d.xmat[b].reshape(3, 3)[:, 0]
            inward = expert._inward_direction(side)
            # An inclined, extended finger keeps the bulky wrists outside the
            # block. Do not fold the distal joints into a hook to make space.
            tilt = np.deg2rad(20.)
            up = np.array([0., 0., 1.])
            normal = np.cos(tilt)*inward + np.sin(tilt)*up
            longitudinal = np.sin(tilt)*inward - np.cos(tilt)*up
            R = _wahba_R([local_n, local_long], [normal, longitudinal], [1., 1.])
            half_width = np.abs(object_R.T @ inward) @ m.geom_size[obj]
            pad_goal = d.geom_xpos[obj] - inward*(half_width+gap)
            # Grip the upper part of the side face: the shorter pinky's middle
            # shell must clear the top edge before the longer rubber pads load.
            pad_goal[2] += .5*(np.abs(object_R[2]) @ m.geom_size[obj])
            local_point = palm_R.T @ (self.point(side, 'middle', d)-d.site_xpos[palm])
            goals.append(pad_goal - R@local_point)
            rotations.append(R)
        return np.asarray(goals), rotations

    def apply_shape(self, progress):
        """Ramp the existing group-command mapping; do not write live qpos."""
        for side in sc.SIDES:
            for group in range(1, 4):
                self.env._group_open[side][group][:] = ((1-progress)*self.original_open[side][group]
                                                       + progress*self.pad_open[side][group])
                self.env._group_close[side][group][:] = ((1-progress)*self.original_close[side][group]
                                                        + progress*self.pad_close[side][group])

    def forces(self):
        m, d = self.env.model, self.env.data
        result = {s: np.zeros(3) for s in sc.SIDES}
        for i, c in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.env._object_body_id not in (b1, b2):
                continue
            other = c.geom2 if b1 == self.env._object_body_id else c.geom1
            if other not in self.geom_side:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f)
            world = c.frame.reshape(3, 3).T @ f[:3]
            result[self.geom_side[other]] += -world if b1 == self.env._object_body_id else world
        return result
