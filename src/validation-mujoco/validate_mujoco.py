import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import MODELS_DIR, POLICY_ONNX, POLICY_PARAMS, SCENE_XML  # noqa: E402
from dataset_builder.collect_run import features  # noqa: E402
from inference import OdomEstimator  # noqa: E402
from sim import K1Sim, WalkPolicy  # noqa: E402
from x11_window import resize_window  # noqa: E402

KEY_RELEASE_GRACE_S = 0.05   
PLOT_EVERY_STEPS = 10       
PLOT_MARGIN_M = 0.5
PLOT_MIN_HALF_M = 1.0
MAX_LAG_S = 0.1
VIEWER_SIZE = (800, 800)

RAMP_RATE = 1.0
DECAY_RATE = 2.0
MAX_TELEOP_DT = 0.1

AXIS_KEYS = {
    "w": (0, 1.0),
    "s": (0, -1.0),
    "a": (1, 1.0),
    "d": (1, -1.0),
    "q": (2, 1.0),
    "e": (2, -1.0),
}

KEY_HELP = """Keys are read from the trajectory plot window. Hold to accelerate, release to slow down.
  W / S : forward / backward      A / D : left / right      Q / E : turn left / turn right
  Space : stop      R : reset trajectories      Esc : quit
Plot: GT (black), learned odometry (red), integrated velocity command (blue). The robot is reset automatically when it falls."""

def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(pose, dx, dy, dtheta):
    x, y, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (x + c * dx - s * dy, y + s * dx + c * dy, wrap_angle(yaw + dtheta))


class Session:
    """GUI-independent simulation + policy + odometry estimation + trajectory accumulation."""

    def __init__(self, checkpoint_path):
        self.policy = WalkPolicy(POLICY_ONNX, POLICY_PARAMS)
        self.sim = K1Sim(SCENE_XML, self.policy.joint_names, self.policy.default_pos,
                         self.policy.stiffness, self.policy.damping, self.policy.step_dt)
        self.estimator = OdomEstimator(checkpoint_path)
        self.step_dt = self.policy.step_dt
        self.falls = 0
        self.reset_robot()

    def _gt(self):
        pose = self.sim.pose()
        return (pose[0], pose[1], pose[3])

    def reset_robot(self):
        self.sim.reset(yaw=0.0)
        self.policy.reset(self.sim.sensors(), np.zeros(3, dtype=np.float32))
        self.estimator.reset(features(self.sim.sensors(), self.policy.last_action))
        self.reset_trajectories()

    def reset_trajectories(self):
        start = self._gt()
        self.gt = [start]
        self.learned = [start]
        self.command = [start]
        self.distance = 0.0

    def step(self, command):
        raw_action, target = self.policy.act(self.sim.sensors(), command)
        self.sim.step(target)
        if self.sim.fallen():
            self.falls += 1
            self.reset_robot()
            return False

        dx, dy, _, dtheta = self.estimator.step(features(self.sim.sensors(), raw_action))
        self.learned.append(compose(self.learned[-1], dx, dy, dtheta))
        self.command.append(compose(
            self.command[-1],
            command[0] * self.step_dt, command[1] * self.step_dt, command[2] * self.step_dt))

        gt = self._gt()
        self.distance += math.hypot(gt[0] - self.gt[-1][0], gt[1] - self.gt[-1][1])
        self.gt.append(gt)
        return True

    def errors(self):
        gt = self.gt[-1]
        learned_err = math.hypot(self.learned[-1][0] - gt[0], self.learned[-1][1] - gt[1])
        command_err = math.hypot(self.command[-1][0] - gt[0], self.command[-1][1] - gt[1])
        return learned_err, command_err


class KeyboardTeleop:

    def __init__(self, figure, positive_max, negative_max):
        self.positive_max = np.asarray(positive_max, dtype=np.float32)
        self.negative_max = np.asarray(negative_max, dtype=np.float32)
        self.axes = np.zeros(3, dtype=np.float32)
        self.updated_at = time.monotonic()
        self.pressed = {}
        self.released_at = {}
        self.reset_requested = False
        self.quit_requested = False
        figure.canvas.mpl_connect("key_press_event", self._on_press)
        figure.canvas.mpl_connect("key_release_event", self._on_release)

    @staticmethod
    def _key(event):
        if event.key is None:
            return None
        return event.key.lower().replace("shift+", "")

    def _on_press(self, event):
        key = self._key(event)
        if key is None:
            return
        if key == " ":
            self.pressed.clear()
            self.released_at.clear()
            self.axes[:] = 0.0
        elif key == "r":
            self.reset_requested = True
        elif key == "escape":
            self.quit_requested = True
        else:
            self.pressed[key] = True
            self.released_at.pop(key, None)

    def _on_release(self, event):
        key = self._key(event)
        if key is not None and key in self.pressed:
            self.released_at[key] = time.monotonic()

    def _held(self, key, now):
        if not self.pressed.get(key, False):
            return False
        released = self.released_at.get(key)
        if released is None:
            return True
        if now - released < KEY_RELEASE_GRACE_S:
            return True
        self.pressed[key] = False
        return False

    def command(self):
        now = time.monotonic()
        dt = min(now - self.updated_at, MAX_TELEOP_DT)
        self.updated_at = now

        directions = np.zeros(3, dtype=np.float32)
        for key, (axis, direction) in AXIS_KEYS.items():
            if self._held(key, now):
                directions[axis] += direction

        for axis in range(3):
            if directions[axis] != 0.0:
                self.axes[axis] = np.clip(self.axes[axis] + directions[axis] * RAMP_RATE * dt, -1.0, 1.0)
            elif abs(self.axes[axis]) <= DECAY_RATE * dt:
                self.axes[axis] = 0.0
            else:
                self.axes[axis] -= np.sign(self.axes[axis]) * DECAY_RATE * dt

        return np.where(self.axes >= 0.0, self.axes * self.positive_max, self.axes * self.negative_max)


class TrajectoryPlot:
    def __init__(self, plt):
        self.plt = plt
        self.figure, self.ax = plt.subplots(figsize=(7, 7))
        self.figure.canvas.manager.set_window_title("learned odometry (keyboard focus here)")

        ax = self.ax
        (self.gt_line,) = ax.plot([], [], "k-", linewidth=2, label="GT")
        (self.learned_line,) = ax.plot([], [], "r--", linewidth=1.5, label="learned")
        (self.command_line,) = ax.plot([], [], "b:", linewidth=1.5, label="command")
        (self.gt_head,) = ax.plot([], [], "k-", linewidth=3)
        (self.learned_head,) = ax.plot([], [], "r-", linewidth=3)
        (self.command_head,) = ax.plot([], [], "b-", linewidth=3)
        self.status = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        ax.set_title("hold W/S A/D Q/E to accelerate, release to slow down\n"
                     "Space stop  R reset  Esc quit", fontsize=9)
        plt.show(block=False)

    def is_open(self):
        return self.plt.fignum_exists(self.figure.number)

    @staticmethod
    def _heading(line, pose, length=0.25):
        x, y, yaw = pose
        line.set_data([x, x + length * math.cos(yaw)], [y, y + length * math.sin(yaw)])

    def update(self, session, command):
        gt = np.asarray(session.gt)
        learned = np.asarray(session.learned)
        commanded = np.asarray(session.command)

        self.gt_line.set_data(gt[:, 0], gt[:, 1])
        self.learned_line.set_data(learned[:, 0], learned[:, 1])
        self.command_line.set_data(commanded[:, 0], commanded[:, 1])
        self._heading(self.gt_head, session.gt[-1])
        self._heading(self.learned_head, session.learned[-1])
        self._heading(self.command_head, session.command[-1])

        points = np.concatenate([gt[:, :2], learned[:, :2], commanded[:, :2]])
        low = points.min(axis=0)
        high = points.max(axis=0)
        center = (low + high) / 2.0
        half = max(PLOT_MIN_HALF_M, float(np.max(high - low)) / 2.0 + PLOT_MARGIN_M)
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)

        learned_err, command_err = session.errors()
        distance = max(session.distance, 1e-6)
        self.status.set_text(
            f"cmd      vx {command[0]:+.2f}  vy {command[1]:+.2f}  wz {command[2]:+.2f}\n"
            f"distance {session.distance:6.2f} m   falls {session.falls}\n"
            f"learned  err {learned_err:5.2f} m  ({learned_err / distance * 100:5.1f}%)\n"
            f"command  err {command_err:5.2f} m  ({command_err / distance * 100:5.1f}%)")
        self.figure.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=KEY_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "odom.pt"))
    parser.add_argument("--backward", type=float, default=0.4,
                        help="Maximum backward speed for S (m/s).")
    args = parser.parse_args()

    checkpoint_path = Path(args.model)
    if not checkpoint_path.is_file():
        sys.exit(f"{checkpoint_path} not found. Run make train first.")

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = []
    plt.ion()

    torch.set_num_threads(1)
    session = Session(checkpoint_path)
    plot = TrajectoryPlot(plt)
    ranges = session.policy.command_ranges
    positive_max = [ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]]
    negative_max = [args.backward, -ranges["lin_vel_y"][0], -ranges["ang_vel_z"][0]]
    teleop = KeyboardTeleop(plot.figure, positive_max, negative_max)

    print(KEY_HELP)
    print(f"Max speed: forward {positive_max[0]:.2f}  backward {negative_max[0]:.2f}  "
          f"left {positive_max[1]:.2f}  right {negative_max[1]:.2f} m/s,  "
          f"turn left {positive_max[2]:.2f}  turn right {negative_max[2]:.2f} rad/s")
    # Raising KeyboardInterrupt inside viewer.sync() can leave the GLFW render thread
    # half torn down and hang the process. The first Ctrl+C only asks the loop to stop
    # (same path as Esc); a second one falls back to the default handler and kills it.
    def request_stop(signum, frame):
        print("\nStopping (Ctrl+C) ...")
        teleop.quit_requested = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    signal.signal(signal.SIGINT, request_stop)
    try:
        run(session, plot, teleop)
    finally:
        plt.close("all")


def run(session, plot, teleop):
    import mujoco
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
            session.sim.model, session.sim.data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = session.sim.base_id
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20.0
        viewer.cam.azimuth = 135.0
        if not resize_window("MuJoCo", *VIEWER_SIZE):
            print("Could not resize the MuJoCo viewer window (X11 only).")

        next_time = time.perf_counter()
        step = 0
        command = np.zeros(3, dtype=np.float32)
        while viewer.is_running() and plot.is_open() and not teleop.quit_requested:
            plot.figure.canvas.flush_events()
            if teleop.reset_requested:
                session.reset_trajectories()
                teleop.reset_requested = False

            command = teleop.command()
            with viewer.lock():
                session.step(command)
            viewer.sync()

            step += 1
            if step % PLOT_EVERY_STEPS == 0:
                plot.update(session, command)

            next_time += session.step_dt
            delay = next_time - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            elif -delay > MAX_LAG_S:
                next_time = time.perf_counter()


if __name__ == "__main__":
    main()
