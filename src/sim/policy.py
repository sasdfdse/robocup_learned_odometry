import numpy as np
import onnxruntime as ort
import yaml

SENSOR_KEY = {
    "base_ang_vel": "ang_vel",
    "projected_gravity": "gravity",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel",
}
RESERVED_KEYS = ("scale_first", "use_gym_history")


class ObservationTerm:

    def __init__(self, name, cfg, scale_first):
        self.name = name
        self.history_length = int(cfg["history_length"])
        self.scale = None
        if cfg.get("scale") is not None:
            self.scale = np.asarray(cfg["scale"], dtype=np.float32)
        self.clip = cfg.get("clip")
        self.scale_first = scale_first
        self.buffer = []

    def _process(self, obs):
        obs = np.asarray(obs, dtype=np.float32).copy()
        if self.scale_first:
            if self.scale is not None:
                obs *= self.scale
            if self.clip is not None:
                obs = np.clip(obs, self.clip[0], self.clip[1])
        else:
            if self.clip is not None:
                obs = np.clip(obs, self.clip[0], self.clip[1])
            if self.scale is not None:
                obs *= self.scale
        return obs

    def reset(self, obs):
        processed = self._process(obs)
        self.buffer = [processed.copy() for _ in range(self.history_length)]

    def add(self, obs):
        self.buffer.append(self._process(obs))
        while len(self.buffer) > self.history_length:
            self.buffer.pop(0)

    def flat(self):
        return np.concatenate(self.buffer)


class WalkPolicy:

    def __init__(self, onnx_path, params_path):
        with open(params_path) as f:
            params = yaml.safe_load(f)

        self.joint_names = list(params["policy_joints"])
        self.step_dt = float(params["step_dt"])

        props = params["joint_properties"]
        self.default_pos = np.array([props[n]["default_position"] for n in self.joint_names])
        self.stiffness = np.array([props[n]["stiffness"] for n in self.joint_names])
        self.damping = np.array([props[n]["damping"] for n in self.joint_names])

        action_cfg = params["actions"]["joint_pos"]
        self.action_scale = np.asarray(action_cfg["scale"], dtype=np.float32)
        self.action_offset = np.asarray(action_cfg["offset"], dtype=np.float32)
        self.action_clip = action_cfg.get("clip")

        self.command_ranges = params["commands"]["base_velocity"]["ranges"]

        obs_cfg = params["observations"]
        scale_first = bool(obs_cfg.get("scale_first", False))
        self.terms = []
        for name, cfg in obs_cfg.items():
            if name in RESERVED_KEYS:
                continue
            if name not in SENSOR_KEY and name not in ("velocity_commands", "last_action"):
                raise ValueError(f"Unsupported observation term: {name}")
            self.terms.append(ObservationTerm(name, cfg, scale_first))

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)

    def _raw_obs(self, term_name, sensors, command):
        if term_name == "velocity_commands":
            return command
        if term_name == "last_action":
            return self.last_action
        return sensors[SENSOR_KEY[term_name]]

    def reset(self, sensors, command):
        self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
        for term in self.terms:
            term.reset(self._raw_obs(term.name, sensors, command))

    def act(self, sensors, command):

        for term in self.terms:
            term.add(self._raw_obs(term.name, sensors, command))
        obs = np.concatenate([term.flat() for term in self.terms])[None, :]

        raw_action = self.session.run(None, {self.input_name: obs})[0][0].astype(np.float32)
        target = raw_action * self.action_scale + self.action_offset
        if self.action_clip is not None:
            target = np.clip(target, self.action_clip[0], self.action_clip[1])

        self.last_action = raw_action
        return raw_action, target
