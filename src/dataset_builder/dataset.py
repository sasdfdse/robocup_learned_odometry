import json
from pathlib import Path

import numpy as np
import torch

from constants import ANG_VEL_SCALE, DATASET_DIR, JOINT_VEL_SCALE
from model import se2_relative


def latest_run():
    runs = sorted(p for p in DATASET_DIR.iterdir() if (p / "data.npz").is_file())
    if not runs:
        raise FileNotFoundError(f"No dataset in {DATASET_DIR}. Run make collect first.")
    return runs[-1]


def load_run(run_dir):
    run_dir = Path(run_dir)
    with np.load(run_dir / "data.npz") as f:
        data = {key: f[key] for key in f.files}
    meta = json.loads((run_dir / "meta.json").read_text())
    return data, meta


def scale_features(feats, num_joints):

    scaled = feats.astype(np.float32).copy()
    scaled[:, 3:6] *= ANG_VEL_SCALE
    scaled[:, 6 + num_joints:6 + 2 * num_joints] *= JOINT_VEL_SCALE
    return scaled


def step_labels(pose, ep_start, ep_len):
    pose_t = torch.from_numpy(pose).double()
    labels = np.full((len(pose), 4), np.nan, dtype=np.float32)
    for start, length in zip(ep_start, ep_len):
        prev = pose_t[start:start + length - 1]
        curr = pose_t[start + 1:start + length]
        dx, dy, dtheta = se2_relative(prev[:, 0], prev[:, 1], prev[:, 3], curr[:, 0], curr[:, 1], curr[:, 3])
        labels[start + 1:start + length] = torch.stack((dx, dy, curr[:, 2], dtheta), dim=-1).numpy()
    return labels


def window_ends(ep_start, ep_len, hist_len, episodes):

    first = max(1, hist_len - 1)
    ends = [np.arange(ep_start[e] + first, ep_start[e] + ep_len[e]) for e in episodes]
    return np.concatenate(ends)


def gather_windows(feats_t, ends_t, hist_len):

    offsets = torch.arange(-hist_len + 1, 1, device=feats_t.device)
    return feats_t[ends_t[:, None] + offsets]
