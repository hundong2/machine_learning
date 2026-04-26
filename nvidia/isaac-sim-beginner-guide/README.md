# 🤖 NVIDIA Isaac Sim 초보자 완전 정복 가이드

> Isaac Sim을 처음 시작하는 분들을 위한 한국어 학습 리포지토리입니다.
> 기초 개념 → 수학/물리 → 예제 코드 → 강화학습까지 단계별로 학습할 수 있습니다.

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0+-76B900?style=flat&logo=nvidia)](https://docs.isaacsim.omniverse.nvidia.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📚 학습 경로

```
Phase 1: 기초 지식 (이론)
    └─ 수학 prerequisite (선형대수, 회전 표현)
    └─ 물리 prerequisite (강체 동역학, 관성)
    └─ USD (Universal Scene Description) 이해

Phase 2: Isaac Sim 기본 (실습)
    └─ 01_basics        → 씬 구성, 프리미티브 객체
    └─ 02_physics       → 중력, 충돌, 조인트
    └─ 03_robotics      → 로봇 로딩, 관절 제어, IK
    └─ 04_sensors       → 카메라, Lidar, IMU
    └─ 05_rl            → 강화학습 환경 구축
```

## 🎯 이 리포지토리의 목표

1. **완전 초보자도** Isaac Sim을 시작할 수 있도록
2. **필수 수학/물리 지식**을 로봇공학 맥락에서 설명
3. **바로 실행 가능한 예제 코드** 제공
4. **공식 문서의 어려운 부분**을 한국어로 친절히 풀어서 설명

## 🚀 빠른 시작

### 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | RTX 3070 (8GB VRAM) | RTX 4080 이상 (16GB+) |
| RAM | 32GB | 64GB |
| Storage | 50GB SSD | 100GB NVMe SSD |
| OS | Ubuntu 22.04 / Windows 10+ | Ubuntu 22.04 |
| Driver | NVIDIA 535+ | 최신 버전 |

### 설치

```bash
# 1. pip을 통한 설치 (가장 쉬운 방법)
pip install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com

# 2. 또는 Omniverse Launcher에서 Isaac Sim 설치
# https://www.nvidia.com/en-us/omniverse/download/

# 3. 첫 실행 검증
python -c "from isaacsim import SimulationApp; print('Isaac Sim Ready!')"
```

### 이 리포지토리 클론

```bash
git clone https://github.com/<your-username>/isaac-sim-beginner-guide.git
cd isaac-sim-beginner-guide
```

## 📖 학습 순서

### 🧮 Phase 0: 사전 지식
먼저 [`math_prerequisites/`](math_prerequisites/) 폴더를 학습하세요.
- [01. 선형대수 기초](math_prerequisites/01_linear_algebra.md)
- [02. 회전과 변환](math_prerequisites/02_rotations.md)
- [03. 강체 동역학](math_prerequisites/03_rigid_body_dynamics.md)
- [04. 로봇 운동학](math_prerequisites/04_kinematics.md)

### 🎨 Phase 1: 기본기
[`examples/01_basics/`](examples/01_basics/) 폴더에서 시작합니다.
- `01_hello_world.py` - 첫 번째 시뮬레이션
- `02_add_primitives.py` - 기본 도형 추가
- `03_camera_and_lighting.py` - 카메라와 조명

### ⚛️ Phase 2: 물리 시뮬레이션
[`examples/02_physics/`](examples/02_physics/)
- `01_gravity_collision.py` - 중력과 충돌
- `02_rigid_bodies.py` - 강체 동역학
- `03_joints_and_articulations.py` - 조인트와 관절

### 🦾 Phase 3: 로봇 제어
[`examples/03_robotics/`](examples/03_robotics/)
- `01_load_robot.py` - 로봇 URDF/USD 로딩
- `02_joint_control.py` - 관절 제어 (PD 제어)
- `03_inverse_kinematics.py` - 역기구학

### 📡 Phase 4: 센서
[`examples/04_sensors/`](examples/04_sensors/)
- `01_camera.py` - RGB/Depth 카메라
- `02_lidar.py` - LiDAR 센서
- `03_imu.py` - IMU 센서

### 🧠 Phase 5: 강화학습
[`examples/05_reinforcement_learning/`](examples/05_reinforcement_learning/)
- `01_gym_wrapper.py` - Gymnasium 호환 래퍼
- `02_cartpole_ppo.py` - CartPole PPO 학습
- `03_custom_task.py` - 커스텀 태스크 만들기

## 🔗 참고 자료

- [Isaac Sim 공식 문서](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab (학습 프레임워크)](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim GitHub](https://github.com/isaac-sim/IsaacSim)
- [USD 공식 문서](https://openusd.org/release/index.html)

## 📝 기여하기

버그 리포트, 예제 추가, 문서 개선은 언제든 환영합니다!
[CONTRIBUTING.md](CONTRIBUTING.md)를 참고해 주세요.

## 📄 라이선스

MIT License - 자유롭게 사용하되 출처를 밝혀주세요.
