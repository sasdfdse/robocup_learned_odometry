"""Learned odometry ROS node: ai_sapiens policy observation -> ikcoordinate.

Subscribes to /policy_input/raw_observation (ai_sapiens sim2real with
debug_publish_enabled:=true, published once per policy step), takes the newest
frame of each term as the odometry input, and publishes the per-step body-frame
displacement as IkCoordMsg (int32 x = forward, y = left) for robocup_localization.

Requires a sourced ROS 2 environment with humanoid_interfaces.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation-mujoco"))

from constants import MODELS_DIR, POLICY_PARAMS  # noqa: E402
from inference import OdomEstimator  # noqa: E402

# Odometry input order (collect_run.features) -> ai_sapiens observation term.
FEATURE_TERMS = ("projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action")
RESERVED_KEYS = ("scale_first", "use_gym_history")

# Same unit as k1_mujoco_bridge (mm x 0.1), so robocup_localization's odom scales stay valid.
DEFAULT_UNITS_PER_METER = 100.0
# A longer gap means the walk policy stopped (mode switch, damping); the history is refilled.
DEFAULT_RESET_GAP_S = 0.2


class RawObservationLayout:
    """Where the newest frame of each term sits in the flat policy observation.

    ai_sapiens concatenates terms in sim2real.yaml order, each term as its history
    oldest -> newest, every frame clipped then scaled.
    """

    def __init__(self, params_path):
        with open(params_path) as f:
            params = yaml.safe_load(f)
        num_joints = len(params["policy_joints"])
        term_sizes = {
            "base_ang_vel": 3,
            "projected_gravity": 3,
            "velocity_commands": 3,
            "joint_pos_rel": num_joints,
            "joint_vel_rel": num_joints,
            "last_action": num_joints,
        }

        self.slices = {}
        self.scales = {}
        offset = 0
        for name, cfg in params["observations"].items():
            if name in RESERVED_KEYS:
                continue
            if name not in term_sizes:
                raise ValueError(f"Unsupported observation term: {name}")
            if cfg.get("clip") is not None and name in FEATURE_TERMS:
                raise ValueError(f"Clipped term '{name}' cannot be unscaled exactly.")
            size = term_sizes[name]
            history = int(cfg["history_length"])
            newest = offset + (history - 1) * size
            self.slices[name] = slice(newest, newest + size)
            scale = np.ones(size, dtype=np.float32)
            if cfg.get("scale") is not None:
                scale = np.asarray(cfg["scale"], dtype=np.float32)
            self.scales[name] = scale
            offset += size * history
        self.size = offset

        for name in FEATURE_TERMS:
            if name not in self.slices:
                raise ValueError(f"Policy observation has no '{name}' term.")

    def features(self, observation):
        """Unscaled odometry input (same layout as collect_run.features), or None on a size mismatch."""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.size != self.size:
            return None
        return np.concatenate([observation[self.slices[name]] / self.scales[name] for name in FEATURE_TERMS])


class ResidualQuantizer:
    """Rounds per-step displacement to IkCoordMsg int32 units without losing slow motion.

    Rounding each step alone drops anything under half a unit per step (at cm and
    50 Hz, every speed below 0.25 m/s), so the rounding remainder is carried over.
    """

    def __init__(self, units_per_meter):
        self.units_per_meter = units_per_meter
        self.remainder = np.zeros(2)

    def reset(self):
        self.remainder[:] = 0.0

    def __call__(self, dx_m, dy_m):
        value = np.array([dx_m, dy_m]) * self.units_per_meter + self.remainder
        rounded = np.round(value)
        self.remainder = value - rounded
        return int(rounded[0]), int(rounded[1])


def make_node(args):
    from humanoid_interfaces.msg import IkCoordMsg
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray

    class LearnedOdometryNode(Node):
        def __init__(self):
            super().__init__("learned_odometry")
            self.layout = RawObservationLayout(args.policy_params)
            self.estimator = OdomEstimator(args.model)
            self.quantizer = ResidualQuantizer(args.units_per_meter)
            self.reset_gap_s = args.reset_gap
            self.last_received = None
            self.size_warned = False

            self.publisher = self.create_publisher(IkCoordMsg, args.output_topic, 10)
            self.create_subscription(Float64MultiArray, args.input_topic, self.on_observation, 10)
            self.get_logger().info(
                f"{args.input_topic} ({self.layout.size}) -> {args.output_topic}, "
                f"model {args.model}, {args.units_per_meter:g} units/m")

        def on_observation(self, message):
            feature = self.layout.features(message.data)
            if feature is None:
                # Another policy (mimic etc.) is active; its observation has a different layout.
                if not self.size_warned:
                    self.get_logger().warn(
                        f"Ignoring observation of size {len(message.data)} (walk policy expects {self.layout.size}).")
                    self.size_warned = True
                self.last_received = None
                return
            self.size_warned = False

            now = time.monotonic()
            stale = self.last_received is None or now - self.last_received > self.reset_gap_s
            self.last_received = now
            if stale:
                self.estimator.reset(feature)
                self.quantizer.reset()
                self.get_logger().info("Odometry history reset.")
                return

            dx, dy, _, _ = self.estimator.step(feature)
            out = IkCoordMsg()
            out.x, out.y = self.quantizer(float(dx), float(dy))
            self.publisher.publish(out)

    return LearnedOdometryNode()


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "odom.pt"))
    parser.add_argument("--policy-params", type=str, default=str(POLICY_PARAMS),
                        help="sim2real.yaml of the walk policy whose observation is subscribed")
    parser.add_argument("--input-topic", type=str, default="/policy_input/raw_observation")
    parser.add_argument("--output-topic", type=str, default="ikcoordinate")
    parser.add_argument("--units-per-meter", type=float, default=DEFAULT_UNITS_PER_METER,
                        help="IkCoordMsg unit (default 100 = cm, same as k1_mujoco_bridge)")
    parser.add_argument("--reset-gap", type=float, default=DEFAULT_RESET_GAP_S,
                        help="refill the history after an observation gap longer than this (s)")
    args, _ = parser.parse_known_args(argv)
    return args


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException

    args = parse_args(sys.argv[1:])
    if not Path(args.model).is_file():
        sys.exit(f"{args.model} not found. Run make train first.")

    torch.set_num_threads(1)
    rclpy.init()
    node = make_node(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
