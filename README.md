# robocup_learned_odometry

Learned odometry that regresses the SE(2) increment of one control step from a history of proprioceptive sensors (IMU + joint states + previous action).
The goal is to replace the `ikcoordinate` input of `robocup_localization` with this estimate instead of the IK command values.

The model architecture is taken from the learned odometry in [humanoid-soccer](../humanoid-soccer) (Booster Robotics, Apache-2.0).
The original targets Booster T1 with its own kick policy and ships no weights, so this package collects data with K1 + the ai_sapiens walking policy and retrains it.
It does not depend on other simulation packages (robocup_mujoco_sim, etc.).

## Model

| Item | Value |
| --- | --- |
| Input | `(B, 50, 75)` — 1 s of history at 50 Hz |
| One step | `projected_gravity(3) + ang_vel(3)×0.2 + joint_pos_rel(23) + joint_vel(23)×0.05 + last_action(23)` |
| Output | `[dx, dy, z, dtheta]` — dx, dy, dtheta are increments in the previous step's body frame; z is the absolute pelvis height |
| Architecture | Linear → Conv1d ×3 → 64 latent → MLP 256 → 128 → 4 (ELU) |
| Loss | MSE on labels normalized by training-set mean/std (normalization values are stored in the checkpoint) |
| dtheta | Learns only the residual on top of trapezoidal gyro integration `0.5 × (ω_z[t-1] + ω_z[t]) × dt` (`src/model/gyro_prior.py`) |
| Input noise | gravity / joint: σ=0.01 after scaling; gyro: σ=0.01 rad/s (`--noise`, `--gyro-noise`) |

Joint order and input scaling match the policy observation in ai_sapiens `walk_default/params/sim2real.yaml`.
`last_action` is the raw policy output (before scale/offset is applied).

## Data collection

`src/sim/` reads the ai_sapiens assets directly and makes K1 walk in MuJoCo.

- Model: `ai_sapiens_description/mujoco/k1/scene.xml`
- Policy: `walk_default/exported/policy.onnx`; observation history / scaling / action handling follow the ai_sapiens sim2real code
- PD gains, default pose: `sim2real.yaml` (deployed values). The D term is applied as joint damping
- Armature, foot friction 0.6: cyclo_mjlab `k1_rev1.py` (armature is 0 in the MJCF)
- IMU: pelvis origin (angular velocity in the body frame)

Episode randomization: initial yaw, velocity commands (every 1.5–4 s, 15% standing, 30% zero per axis), foot friction 0.4–1.0, horizontal push ±0.3 m/s every 3–6 s.
40% of commands draw each axis magnitude from 70–100% of the range limit (`EDGE_PROB`, `EDGE_LOW`).
With uniform sampling alone, fast turning (+ fast forward) segments made up only about 2% of the data, and dtheta was underestimated in those segments.

## Usage

Run from the package root.

| Command | Description |
| --- | --- |
| `make install` | Install dependencies |
| `make validate-mujoco` | Drive K1 in MuJoCo with the keyboard and display GT / learned odometry / command-integrated trajectories in real time (`models/odom.pt`) |
| `make odom-node` | Inference node: `/policy_input/raw_observation` → `ikcoordinate` (see "Inference node" below) |
| `make collect` | Collect 600 episodes × 20 s (about 3.3 hours of data, 70 s on 20 cores) → `dataset/<date_time>/` |
| `make collect-smoke` | Smoke test with 8 episodes × 10 s |
| `make train` | Train on the latest dataset + evaluate validation trajectories → `results/odom_<date_time>/`, `models/odom.pt` |

## Sources

| File | Contents |
| --- | --- |
| `src/model/odom.py` | `humanoid-soccer/model/odom.py` (same architecture) |
| `src/model/se2.py` | `se2_relative`, `se2_compose` from `humanoid-soccer/runner/odom.py` |
| `src/sim/k1_sim.py` | K1 MuJoCo, PD control, sensors/GT |
| `src/sim/policy.py` | ai_sapiens walking policy runner |
| `src/dataset_builder/` | Collection / labeling, window generation |
| `src/training/train_run.py` | Training + trajectory evaluation |
| `src/validation-mujoco/inference/odom_estimator.py` | Loads a checkpoint and estimates step by step (history, scaling, denormalization) |
| `src/validation-mujoco/validate_mujoco.py` | Keyboard teleop + real-time trajectory validation |

## Inference node

`src/inference-ros/odom_node.py` — a plain Python rclpy script.

```bash
source /opt/ros/jazzy/setup.bash && source <workspace containing humanoid_interfaces>/install/setup.bash
# Launch ai_sapiens sim2real with debug_publish_enabled:=true (publishes once per policy step)
make odom-node
```
