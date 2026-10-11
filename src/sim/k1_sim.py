import math

import mujoco
import numpy as np

BASE_BODY = "pelvis"
FOOT_GEOM_PREFIXES = ("left_ankle_roll_link_collision_", "right_ankle_roll_link_collision_")

ARMATURE_QC080 = 0.01936542
ARMATURE_QC060 = 0.00564892
QC080_MIN_TORQUE = 60.0

FOOT_FRICTION = 0.6
FOOT_CLEARANCE = 0.001

FALL_HEIGHT = 0.45
FALL_GRAVITY_Z = -0.5


class K1Sim:

    def __init__(self, scene_xml, joint_names, default_pos, stiffness, damping, control_dt):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        if self.base_id < 0:
            raise ValueError(f"MJCF has no body '{BASE_BODY}'.")

        joint_ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MJCF has no joint '{name}'.")
            joint_ids.append(joint_id)
        self.qpos_adr = m.jnt_qposadr[joint_ids]
        self.qvel_adr = m.jnt_dofadr[joint_ids]

        actuator_by_joint = {}
        for actuator_id in range(m.nu):
            actuator_by_joint[int(m.actuator_trnid[actuator_id, 0])] = actuator_id
        self.actuator_ids = np.array([actuator_by_joint[j] for j in joint_ids])
        self.torque_limit = m.actuator_ctrlrange[self.actuator_ids, 1].copy()

        self.default_pos = np.asarray(default_pos, dtype=np.float64)
        self.stiffness = np.asarray(stiffness, dtype=np.float64)

        m.dof_damping[self.qvel_adr] = np.asarray(damping, dtype=np.float64)
        m.dof_armature[self.qvel_adr] = np.where(
            self.torque_limit > QC080_MIN_TORQUE, ARMATURE_QC080, ARMATURE_QC060)

        foot_geom_ids = []
        for geom_id in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is not None and name.startswith(FOOT_GEOM_PREFIXES):
                foot_geom_ids.append(geom_id)
        self.foot_geom_ids = np.array(foot_geom_ids)
        m.geom_priority[self.foot_geom_ids] = 1
        m.geom_condim[self.foot_geom_ids] = 3
        self.set_foot_friction(FOOT_FRICTION)

        self.decimation = int(round(control_dt / m.opt.timestep))
        if not math.isclose(self.decimation * m.opt.timestep, control_dt, rel_tol=1e-6):
            raise ValueError("Control period is not an integer multiple of the sim timestep.")

    def set_foot_friction(self, friction):
        self.model.geom_friction[self.foot_geom_ids, 0] = friction

    def reset(self, yaw=0.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = (0.0, 0.0, 1.0)
        d.qpos[3:7] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        d.qpos[self.qpos_adr] = self.default_pos
        mujoco.mj_forward(m, d)

        lowest = np.min(d.geom_xpos[self.foot_geom_ids, 2] - m.geom_size[self.foot_geom_ids, 0])
        d.qpos[2] -= lowest - FOOT_CLEARANCE
        mujoco.mj_forward(m, d)

    def step(self, target_pos):
       
        d = self.data
        for _ in range(self.decimation):
            error = target_pos - d.qpos[self.qpos_adr]
            torque = np.clip(self.stiffness * error, -self.torque_limit, self.torque_limit)
            d.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, d)

    def push(self, velocity_xy):
        self.data.qvel[0:2] += velocity_xy

    def _base_rotation(self):
        return self.data.xmat[self.base_id].reshape(3, 3)

    def sensors(self):
        d = self.data
        rotation = self._base_rotation()
        return {
            "gravity": rotation.T @ np.array([0.0, 0.0, -1.0]),
            # The free joint angular velocity is expressed in the body frame (same frame as a gyro).
            "ang_vel": d.qvel[3:6].copy(),
            "joint_pos_rel": d.qpos[self.qpos_adr] - self.default_pos,
            "joint_vel": d.qvel[self.qvel_adr].copy(),
        }

    def pose(self):
        rotation = self._base_rotation()
        position = self.data.xpos[self.base_id]
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        return np.array([position[0], position[1], position[2], yaw])

    def fallen(self):
        rotation = self._base_rotation()
        gravity_z = (rotation.T @ np.array([0.0, 0.0, -1.0]))[2]
        return self.data.xpos[self.base_id, 2] < FALL_HEIGHT or gravity_z > FALL_GRAVITY_Z
