# 🦾 Phase 3: 로봇 제어

미리 만들어진 로봇을 로딩하고 제어합니다.

## 📋 예제 목록

### 01_load_robot.py
**목표**: Franka Panda 로봇 팔 로딩
```bash
python 01_load_robot.py
```

### 02_joint_control.py
**목표**: 관절 제어 (Position / Velocity / Effort)
```bash
python 02_joint_control.py
```

**배우는 내용**:
- PD Controller 동작
- Sinusoidal trajectory tracking
- Step response 분석

### 03_inverse_kinematics.py
**목표**: 엔드이펙터가 원 궤적을 그리도록 IK 제어
```bash
python 03_inverse_kinematics.py
```

**배우는 내용**:
- Lula Kinematics Solver
- RMPFlow Controller (장애물 회피까지)

## 🤖 Isaac Sim 내장 로봇 목록

| 카테고리 | 로봇 |
|---------|------|
| 산업용 팔 | Franka Panda, UR3/UR5/UR10, Kuka IIWA |
| 이동 로봇 | Jetbot, Carter, Jackal, TurtleBot |
| 휴머노이드 | Humanoid (RL용), H1, Unitree G1 |
| 4족 | Anymal, Unitree A1/Go1/Go2, Spot |
| 드론 | Crazyflie (실험적) |

## 📁 에셋 경로 구조

```
omniverse://localhost/Isaac/
├── Robots/
│   ├── Franka/
│   │   ├── franka.usd          # 표준 Franka
│   │   └── franka_instanceable.usd  # RL 대량 학습용
│   ├── UR10/
│   ├── Jetbot/
│   └── ...
├── Environments/
│   ├── Simple_Warehouse/
│   ├── Grid_Room/
│   └── ...
└── Props/
    ├── YCB/                    # 그리퍼 훈련용 물체들
    └── Shapes/
```

## 💡 제어 모드 선택 가이드

| 상황 | 추천 모드 |
|------|----------|
| 정확한 자세 유지 | Position Control |
| 일정 속도 주행 (이동 로봇) | Velocity Control |
| 힘 제어 / 컴플라이언트 제어 | Effort Control |
| 궤적 추종 | Position + 충분한 damping |

## ➡️ 다음 단계

[Phase 4: 센서](../04_sensors/)
