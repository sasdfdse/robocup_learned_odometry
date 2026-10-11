import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import DATASET_DIR, HIST_LEN, POLICY_ONNX, POLICY_PARAMS, SCENE_XML  # noqa: E402
from sim import K1Sim, WalkPolicy  # noqa: E402

SETTLE_STEPS = 50              # steps standing with a zero command after reset
COMMAND_HOLD_S = (1.5, 4.0)    # range of time each command is held
STAND_PROB = 0.15              # probability of a zero (stand) command
AXIS_ZERO_PROB = 0.3           # per-axis zero probability (keeps pure forward/lateral/turn segments)
EDGE_PROB = 0.4                # probability that a command is drawn near the range limits
EDGE_LOW = 0.7                 # edge commands use |value| in [EDGE_LOW, 1] x the range limit
FRICTION_RANGE = (0.4, 1.0)
PUSH_INTERVAL_S = (3.0, 6.0)
PUSH_MAX_VEL = 0.3
FALL_TRIM_STEPS = 50           # steps dropped right before a fall

_sim = None
_policy = None


def _init_worker():
    global _sim, _policy
    _policy = WalkPolicy(POLICY_ONNX, POLICY_PARAMS)
    _sim = K1Sim(SCENE_XML, _policy.joint_names, _policy.default_pos,
                 _policy.stiffness, _policy.damping, _policy.step_dt)


def sample_axis(rng, low, high, edge):
    """Uniform in [low, high], or near one of the limits for edge commands.

    Uniform sampling alone leaves fast turns (and fast turns while walking fast)
    rare, which biases the learned dtheta toward zero at high yaw rates.
    """
    if not edge:
        return rng.uniform(low, high)
    limit = high
    if rng.random() < 0.5:
        limit = low
    return limit * rng.uniform(EDGE_LOW, 1.0)


def sample_command(rng, ranges):
    if rng.random() < STAND_PROB:
        return np.zeros(3, dtype=np.float32)
    edge = rng.random() < EDGE_PROB
    command = np.array([
        sample_axis(rng, *ranges["lin_vel_x"], edge),
        sample_axis(rng, *ranges["lin_vel_y"], edge),
        sample_axis(rng, *ranges["ang_vel_z"], edge),
    ], dtype=np.float32)
    for axis in range(3):
        if rng.random() < AXIS_ZERO_PROB:
            command[axis] = 0.0
    return command


def hold_steps(rng, interval_s, step_dt):
    return max(1, int(round(rng.uniform(*interval_s) / step_dt)))


def features(sensors, last_action):
    return np.concatenate([
        sensors["gravity"], sensors["ang_vel"],
        sensors["joint_pos_rel"], sensors["joint_vel"], last_action,
    ]).astype(np.float32)


def run_episode(task):
    episode_seed, num_steps, use_push = task
    rng = np.random.default_rng(episode_seed)
    sim, policy = _sim, _policy
    step_dt = policy.step_dt

    friction = rng.uniform(*FRICTION_RANGE)
    sim.set_foot_friction(friction)
    sim.reset(yaw=rng.uniform(-math.pi, math.pi))

    command = np.zeros(3, dtype=np.float32)
    policy.reset(sim.sensors(), command)

    feats = [features(sim.sensors(), policy.last_action)]
    poses = [sim.pose()]
    commands = [command.copy()]

    next_command_step = SETTLE_STEPS
    next_push_step = SETTLE_STEPS + hold_steps(rng, PUSH_INTERVAL_S, step_dt)
    fell = False

    for step in range(1, num_steps + 1):
        if step >= next_command_step:
            command = sample_command(rng, policy.command_ranges)
            next_command_step = step + hold_steps(rng, COMMAND_HOLD_S, step_dt)
        if use_push and step >= next_push_step:
            sim.push(rng.uniform(-PUSH_MAX_VEL, PUSH_MAX_VEL, size=2))
            next_push_step = step + hold_steps(rng, PUSH_INTERVAL_S, step_dt)

        raw_action, target = policy.act(sim.sensors(), command)
        sim.step(target)
        if sim.fallen():
            fell = True
            break

        feats.append(features(sim.sensors(), raw_action))
        poses.append(sim.pose())
        commands.append(command.copy())

    if fell:
        keep = len(feats) - FALL_TRIM_STEPS
        feats, poses, commands = feats[:keep], poses[:keep], commands[:keep]

    if len(feats) < HIST_LEN + 1:
        return {"seed": episode_seed, "fell": fell, "friction": friction, "length": 0}

    return {
        "seed": episode_seed,
        "fell": fell,
        "friction": friction,
        "length": len(feats),
        "feats": np.stack(feats),
        "pose": np.stack(poses).astype(np.float64),
        "cmd": np.stack(commands),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode-s", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-push", action="store_true", help="disable push disturbances")
    parser.add_argument("--name", type=str, default=None, help="output folder name (default: date_time)")
    args = parser.parse_args()

    probe = WalkPolicy(POLICY_ONNX, POLICY_PARAMS)
    num_steps = int(round(args.episode_s / probe.step_dt))
    tasks = [(args.seed * 100000 + i, num_steps, not args.no_push) for i in range(args.episodes)]

    name = args.name
    if name is None:
        name = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DATASET_DIR / name
    out_dir.mkdir(parents=True, exist_ok=False)

    print(f"[collect] {args.episodes} episodes x {args.episode_s:.0f}s, workers={args.workers} -> {out_dir}")
    started = time.time()
    results = []
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for done, result in enumerate(pool.imap_unordered(run_episode, tasks), start=1):
            results.append(result)
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  ({time.time() - started:.0f}s)")

    results.sort(key=lambda r: r["seed"])
    kept = [r for r in results if r["length"] > 0]
    if not kept:
        raise RuntimeError("No episodes left.")

    lengths = np.array([r["length"] for r in kept])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    np.savez_compressed(
        out_dir / "data.npz",
        feats=np.concatenate([r["feats"] for r in kept]),
        pose=np.concatenate([r["pose"] for r in kept]),
        cmd=np.concatenate([r["cmd"] for r in kept]),
        ep_start=starts,
        ep_len=lengths,
        ep_friction=np.array([r["friction"] for r in kept]),
        ep_fell=np.array([r["fell"] for r in kept]),
    )

    num_joints = len(probe.joint_names)
    meta = {
        "joint_names": probe.joint_names,
        "step_dt": probe.step_dt,
        "feature_layout": {
            "gravity": [0, 3],
            "ang_vel": [3, 6],
            "joint_pos_rel": [6, 6 + num_joints],
            "joint_vel": [6 + num_joints, 6 + 2 * num_joints],
            "last_action": [6 + 2 * num_joints, 6 + 3 * num_joints],
        },
        "episodes_requested": args.episodes,
        "episodes_kept": len(kept),
        "episodes_fell": int(sum(r["fell"] for r in results)),
        "steps": int(lengths.sum()),
        "episode_s": args.episode_s,
        "seed": args.seed,
        "push": not args.no_push,
        "friction_range": list(FRICTION_RANGE),
        "edge_prob": EDGE_PROB,
        "edge_low": EDGE_LOW,
        "policy_onnx": str(POLICY_ONNX),
        "policy_params": str(POLICY_PARAMS),
        "scene_xml": str(SCENE_XML),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    hours = lengths.sum() * probe.step_dt / 3600.0
    print(f"[collect] kept {len(kept)}/{len(results)} episodes, fell {meta['episodes_fell']}, "
          f"{lengths.sum()} steps ({hours:.2f} h), {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
