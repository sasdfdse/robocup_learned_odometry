from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
WS_SRC = PKG_ROOT.parent

AI_SAPIENS = WS_SRC / "ai_sapiens"
SCENE_XML = AI_SAPIENS / "ai_sapiens_description" / "mujoco" / "k1" / "scene.xml"
POLICY_DIR = AI_SAPIENS / "ai_sapiens_sim2real" / "assets" / "k1" / "locomotion" / "velocity" / "walk_default"
POLICY_ONNX = POLICY_DIR / "exported" / "policy.onnx"
POLICY_PARAMS = POLICY_DIR / "params" / "sim2real.yaml"

DATASET_DIR = PKG_ROOT / "dataset"
RESULTS_DIR = PKG_ROOT / "results"
MODELS_DIR = PKG_ROOT / "models"

# Odometry input history length (1 s at 50 Hz). StateHistoryEncoder only accepts 10/20/50.
HIST_LEN = 50

# Odometry input scales. Kept equal to the walking policy observation scales (sim2real.yaml).
ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
