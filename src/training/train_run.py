import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import ANG_VEL_SCALE, HIST_LEN, JOINT_VEL_SCALE, MODELS_DIR, RESULTS_DIR  # noqa: E402
from dataset_builder.dataset import (  # noqa: E402
    gather_windows, latest_run, load_run, scale_features, step_labels, window_ends)
from model import DTHETA_PRIOR, OdomModel, gyro_dtheta, se2_compose  # noqa: E402

LABEL_NAMES = ("dx", "dy", "z", "dtheta")
PLOT_EPISODES = 4


def predict(model, feats_t, ends, label_mean, label_std, batch_size, step_dt):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(ends), batch_size):
            batch_ends = ends[i:i + batch_size]
            window = gather_windows(feats_t, batch_ends, HIST_LEN)
            out = model(window) * label_std + label_mean
            out[:, 3] += gyro_dtheta(window, ANG_VEL_SCALE, step_dt)
            outputs.append(out)
    return torch.cat(outputs)


def integrate(start_pose, steps):
    x = torch.tensor([start_pose[0]], dtype=torch.float64)
    y = torch.tensor([start_pose[1]], dtype=torch.float64)
    theta = torch.tensor([start_pose[2]], dtype=torch.float64)
    traj = [(start_pose[0], start_pose[1], start_pose[2])]
    steps = torch.as_tensor(steps, dtype=torch.float64)
    for dx, dy, dtheta in steps:
        x, y, theta = se2_compose(x, y, theta, dx[None], dy[None], dtheta[None])
        traj.append((x.item(), y.item(), theta.item()))
    return np.array(traj)


def evaluate_trajectories(model, feats_t, data, episodes, label_mean, label_std, step_dt, batch_size, device):
    pose = data["pose"]
    cmd = data["cmd"]
    rows = []
    plots = []
    for e in episodes:
        start, length = int(data["ep_start"][e]), int(data["ep_len"][e])
        ends_np = window_ends(data["ep_start"], data["ep_len"], HIST_LEN, [e])
        ends = torch.from_numpy(ends_np).to(device)
        pred = predict(model, feats_t, ends, label_mean, label_std, batch_size, step_dt).cpu().numpy()

        first = ends_np[0]
        gt = pose[first - 1:start + length][:, [0, 1, 3]]
        learned = integrate(gt[0], pred[:, [0, 1, 3]])
        command = integrate(gt[0], cmd[ends_np] * step_dt)

        distance = np.sum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))
        if distance < 1.0:
            continue
        learned_err = np.linalg.norm(learned[-1, :2] - gt[-1, :2])
        command_err = np.linalg.norm(command[-1, :2] - gt[-1, :2])
        rows.append((distance, learned_err, command_err))
        if len(plots) < PLOT_EPISODES:
            plots.append((e, gt, learned, command))

    if not rows:
        return {"eval_episodes": 0}, plots

    rows = np.array(rows)
    metrics = {
        "eval_episodes": int(len(rows)),
        "mean_distance_m": float(rows[:, 0].mean()),
        "learned_final_err_m": float(rows[:, 1].mean()),
        "command_final_err_m": float(rows[:, 2].mean()),
        "learned_drift_pct": float((rows[:, 1] / rows[:, 0]).mean() * 100.0),
        "command_drift_pct": float((rows[:, 2] / rows[:, 0]).mean() * 100.0),
    }
    return metrics, plots


def save_plot(plots, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 5), squeeze=False)
    for ax, (episode, gt, learned, command) in zip(axes[0], plots):
        ax.plot(gt[:, 0], gt[:, 1], "k-", linewidth=2, label="GT")
        ax.plot(learned[:, 0], learned[:, 1], "r--", label="learned")
        ax.plot(command[:, 0], command[:, 1], "b:", label="command")
        ax.plot(gt[0, 0], gt[0, 1], "go")
        ax.set_title(f"episode {episode}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=str, default=None, help="dataset folder (default: latest)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="std of Gaussian noise added to scaled gravity / joint_pos / joint_vel inputs")
    parser.add_argument("--gyro-noise", type=float, default=0.01,
                        help="std of Gaussian noise added to the gyro, in rad/s before scaling. Kept near real IMU "
                             "noise: a large value attenuates the predicted dtheta toward zero.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        device = torch.device("cpu")

    if args.data is None:
        run_dir = latest_run()
    else:
        run_dir = Path(args.data)
    data, meta = load_run(run_dir)
    num_joints = len(meta["joint_names"])
    step_dt = float(meta["step_dt"])
    print(f"[train] data={run_dir}  steps={len(data['feats'])}  episodes={len(data['ep_len'])}  device={device}")

    feats = scale_features(data["feats"], num_joints)
    labels = step_labels(data["pose"], data["ep_start"], data["ep_len"])
    sensor_dims = 6 + 2 * num_joints  # everything before last_action
    noise_std = torch.full((sensor_dims,), args.noise)
    noise_std[3:6] = args.gyro_noise * ANG_VEL_SCALE

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(data["ep_len"]))
    num_val = max(1, int(round(len(order) * args.val_frac)))
    val_episodes = np.sort(order[:num_val])
    train_episodes = np.sort(order[num_val:])

    feats_t = torch.from_numpy(feats).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    noise_std = noise_std.to(device)
    train_ends = torch.from_numpy(window_ends(data["ep_start"], data["ep_len"], HIST_LEN, train_episodes)).to(device)
    val_ends = torch.from_numpy(window_ends(data["ep_start"], data["ep_len"], HIST_LEN, val_episodes)).to(device)

    # dtheta is learned as a residual over the gyro prior, so the stats are taken on the residual.
    train_targets = labels_t[train_ends].clone()
    prior_windows = torch.stack((feats_t[train_ends - 1], feats_t[train_ends]), dim=1)
    train_targets[:, 3] -= gyro_dtheta(prior_windows, ANG_VEL_SCALE, step_dt)
    label_mean = train_targets.mean(dim=0)
    label_std = train_targets.std(dim=0).clamp_min(1e-6)

    model = OdomModel(feats.shape[1], HIST_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = RESULTS_DIR / f"odom_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    checkpoint_path = out_dir / "odom.pt"

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = train_ends[torch.randperm(len(train_ends), device=device)]
        train_loss = 0.0
        num_batches = 0
        for i in range(0, len(perm), args.batch):
            batch_ends = perm[i:i + args.batch]
            x = gather_windows(feats_t, batch_ends, HIST_LEN)
            if torch.any(noise_std > 0.0):
                x = x.clone()
                x[:, :, :sensor_dims] += noise_std * torch.randn_like(x[:, :, :sensor_dims])
            target = labels_t[batch_ends].clone()
            target[:, 3] -= gyro_dtheta(x, ANG_VEL_SCALE, step_dt)
            y = (target - label_mean) / label_std
            loss = torch.mean((model(x) - y) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        scheduler.step()
        train_loss /= num_batches

        pred = predict(model, feats_t, val_ends, label_mean, label_std, args.batch * 4, step_dt)
        err = pred - labels_t[val_ends]
        val_loss = torch.mean((err / label_std) ** 2).item()
        rmse = torch.sqrt(torch.mean(err ** 2, dim=0)).cpu().numpy()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        **{f"rmse_{n}": float(v) for n, v in zip(LABEL_NAMES, rmse)}})
        print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"rmse dx {rmse[0] * 1000:.2f}mm dy {rmse[1] * 1000:.2f}mm "
              f"z {rmse[2] * 1000:.1f}mm dtheta {np.degrees(rmse[3]):.3f}deg")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "num_prop": feats.shape[1],
                "hist_len": HIST_LEN,
                "label_names": list(LABEL_NAMES),
                "label_mean": label_mean.cpu().tolist(),
                "label_std": label_std.cpu().tolist(),
                "joint_names": meta["joint_names"],
                "feature_layout": meta["feature_layout"],
                "ang_vel_scale": ANG_VEL_SCALE,
                "joint_vel_scale": JOINT_VEL_SCALE,
                "step_dt": step_dt,
                "dtheta_prior": DTHETA_PRIOR,
                "dataset": str(run_dir),
                "epoch": epoch,
            }, checkpoint_path)

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    metrics, plots = evaluate_trajectories(
        model, feats_t, data, val_episodes, label_mean, label_std, step_dt, args.batch * 4, device)
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["best_val_loss"] = best_val
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if plots:
        save_plot(plots, out_dir / "trajectories.png")

    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(checkpoint_path, MODELS_DIR / "odom.pt")

    print(f"[train] best epoch {checkpoint['epoch']}  val {best_val:.4f}")
    if metrics["eval_episodes"] > 0:
        print(f"[eval] {metrics['eval_episodes']} episodes, mean distance {metrics['mean_distance_m']:.1f} m")
        print(f"       learned  final err {metrics['learned_final_err_m']:.3f} m  drift {metrics['learned_drift_pct']:.1f}%")
        print(f"       command  final err {metrics['command_final_err_m']:.3f} m  drift {metrics['command_drift_pct']:.1f}%")
    else:
        print("[eval] No validation episode moved more than 1 m; skipping trajectory evaluation.")
    print(f"[train] -> {out_dir}, models/odom.pt")


if __name__ == "__main__":
    main()
