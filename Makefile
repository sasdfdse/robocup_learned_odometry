PY := PYTHONPATH=src python3

.PHONY: help install validate-mujoco odom-node collect collect-smoke train clean

help:
	@echo "targets:"
	@echo "  install        의존성 설치 (pip -r requirements.txt)"
	@echo "  validate-mujoco  MuJoCo 에서 K1 키보드 조종 + 학습 odometry 궤적 실시간 표시 (models/odom.pt)"
	@echo "  odom-node      추론 노드: /policy_input/raw_observation -> ikcoordinate (ROS 2 + humanoid_interfaces source 필요)"
	@echo "  collect        K1 + ai_sapiens 보행 정책으로 데이터 수집 -> dataset/<날짜_시각>/"
	@echo "  collect-smoke  짧은 수집 (8 에피소드 x 10초) 동작 확인용"
	@echo "  train          최신 데이터셋으로 학습 + 궤적 평가 -> results/odom_<날짜_시각>/, models/odom.pt"
	@echo "  clean          __pycache__ 정리"

install:
	python3 -m pip install -r requirements.txt

validate-mujoco:
	$(PY) src/validation-mujoco/validate_mujoco.py

odom-node:
	$(PY) src/inference-ros/odom_node.py

collect:
	$(PY) src/dataset_builder/collect_run.py

collect-smoke:
	$(PY) src/dataset_builder/collect_run.py --episodes 8 --episode-s 10 --workers 4 --name smoke

train:
	$(PY) src/training/train_run.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
