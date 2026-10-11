from collections import deque

import numpy as np
import torch

from model import DTHETA_PRIOR, OdomModel, gyro_dtheta


class OdomEstimator:
    
    def __init__(self, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.device = torch.device(device)
        self.hist_len = int(checkpoint["hist_len"])
        self.model = OdomModel(int(checkpoint["num_prop"]), self.hist_len).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.label_mean = torch.tensor(checkpoint["label_mean"], device=self.device)
        self.label_std = torch.tensor(checkpoint["label_std"], device=self.device)
        self.num_joints = len(checkpoint["joint_names"])
        self.ang_vel_scale = float(checkpoint["ang_vel_scale"])
        self.joint_vel_scale = float(checkpoint["joint_vel_scale"])
        self.step_dt = float(checkpoint["step_dt"])
        # Checkpoints trained before the gyro prior have no key and predict dtheta directly.
        self.use_gyro_prior = checkpoint.get("dtheta_prior") == DTHETA_PRIOR
        self.history = deque(maxlen=self.hist_len)

    def _scale(self, feature):
        scaled = np.asarray(feature, dtype=np.float32).copy()
        scaled[3:6] *= self.ang_vel_scale
        scaled[6 + self.num_joints:6 + 2 * self.num_joints] *= self.joint_vel_scale
        return scaled

    def reset(self, feature):
        scaled = self._scale(feature)
        self.history.clear()
        for _ in range(self.hist_len):
            self.history.append(scaled)

    def step(self, feature):
        self.history.append(self._scale(feature))
        window = torch.from_numpy(np.stack(self.history)).to(self.device)[None]
        with torch.no_grad():
            out = self.model(window)[0] * self.label_std + self.label_mean
            if self.use_gyro_prior:
                out[3] += gyro_dtheta(window, self.ang_vel_scale, self.step_dt)[0]
        return out.cpu().numpy()
