# ⚛️ 03. 강체 동역학 (Rigid Body Dynamics)

Isaac Sim의 물리 엔진(PhysX)이 계산하는 내용을 이해해봅시다.

## 1. 강체(Rigid Body)란?

**형태가 변하지 않는 물체**. 로봇의 각 링크, 장애물, 공 등이 모두 강체입니다.

강체의 상태는 다음으로 완전히 기술됩니다:
- **위치** (position): 3차원 벡터 `[x, y, z]`
- **방향** (orientation): 쿼터니언 `[w, x, y, z]`
- **선속도** (linear velocity): `[vx, vy, vz]`
- **각속도** (angular velocity): `[ωx, ωy, ωz]`

```
상태 벡터: [position, orientation, linear_vel, angular_vel]
         = 3 + 4 + 3 + 3 = 13차원
```

## 2. 뉴턴의 운동 법칙

### 2.1 선형 운동 (Newton's 2nd Law)

```
F = m·a
```

- `F`: 힘 (Force, 뉴턴 N)
- `m`: 질량 (mass, kg)
- `a`: 가속도 (acceleration, m/s²)

```python
# 예: 질량 2kg 물체에 10N의 힘이 작용할 때
mass = 2.0  # kg
force = np.array([10, 0, 0])  # N
acceleration = force / mass  # [5, 0, 0] m/s²

# 1초 후 속도 변화 (초기 속도 0이라 가정)
dt = 1.0
velocity = acceleration * dt  # [5, 0, 0] m/s
```

### 2.2 회전 운동 (Euler's Equations)

```
τ = I·α
```

- `τ`: 토크 (Torque, Nm)
- `I`: 관성 텐서 (Inertia tensor, 3x3 행렬)
- `α`: 각가속도 (angular acceleration, rad/s²)

## 3. 관성 텐서 (Inertia Tensor)

**물체가 회전에 얼마나 저항하는지**를 나타내는 3x3 행렬.

```
I = | Ixx  Ixy  Ixz |
    | Ixy  Iyy  Iyz |
    | Ixz  Iyz  Izz |
```

### 주요 도형의 관성

**정육면체 (한 변 길이 a, 질량 m)**:
```
I = (m·a²/6) · 단위행렬
```

**구 (반지름 r, 질량 m)**:
```
I = (2·m·r²/5) · 단위행렬
```

**원통 (반지름 r, 높이 h, 질량 m, 축은 Z)**:
```
Izz = m·r²/2
Ixx = Iyy = m·(3r² + h²)/12
```

```python
def cuboid_inertia(mass, size):
    """직육면체의 관성 텐서 계산"""
    w, h, d = size  # width, height, depth
    Ixx = mass * (h**2 + d**2) / 12
    Iyy = mass * (w**2 + d**2) / 12
    Izz = mass * (w**2 + h**2) / 12
    return np.diag([Ixx, Iyy, Izz])
```

### Isaac Sim에서 자동 계산
Isaac Sim은 기본 도형(cube, sphere, cylinder)의 관성을 **자동 계산**합니다.
URDF나 USD 파일에서는 직접 지정할 수도 있습니다.

## 4. 시뮬레이션의 수치 적분

시뮬레이터는 **미분 방정식**을 작은 시간 간격 `dt`로 적분하여 다음 상태를 예측합니다.

### Euler Method (간단)
```
x(t+dt) = x(t) + v(t)·dt
v(t+dt) = v(t) + a(t)·dt
```

### Isaac Sim의 PhysX
PhysX는 **Semi-implicit Euler**와 **RK4(Runge-Kutta)** 같은 더 정확한 방법을 사용합니다.

### 타임스텝
```python
# Isaac Sim 기본값: 60Hz (dt = 1/60 s)
from isaacsim.core.api import World

world = World(
    physics_dt=1.0/60.0,    # 물리 시뮬레이션 주기
    rendering_dt=1.0/30.0,  # 렌더링 주기 (보통 물리의 절반)
)
```

### ⚠️ 타임스텝의 trade-off

| dt 값 | 장점 | 단점 |
|-------|------|------|
| 작음 (1/240s) | 정확, 안정적 | 느림 |
| 큼 (1/30s) | 빠름 | 물리 불안정, 관통 발생 가능 |

## 5. 충돌 (Collision)

### 충돌 감지 (Collision Detection)
물체들이 서로 겹치는지 확인하는 과정. Broadphase와 Narrowphase로 나뉩니다.

### 충돌 응답 (Collision Response)
충돌이 감지되면 물체의 속도를 어떻게 바꿀지 계산.

**탄성 충돌 (복원 계수 e)**:
```
v' = -e · v
```
- `e = 1`: 완전 탄성 충돌 (에너지 보존)
- `e = 0`: 완전 비탄성 충돌 (붙어버림)

### Isaac Sim에서 설정
```python
from isaacsim.core.api.materials import PhysicsMaterial

# 물리 재질 정의
material = PhysicsMaterial(
    prim_path="/World/PhysicsMaterial",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.8,  # 탄성 계수
)
```

## 6. 마찰 (Friction)

```
F_friction = μ · N
```

- `μ`: 마찰 계수
- `N`: 수직항력

- **정적 마찰** (static): 정지한 물체가 움직이기 시작하기 전
- **동적 마찰** (kinetic): 이미 움직이는 물체에 작용

## 7. 실습: 자유 낙하 시뮬레이션 검증

```python
import numpy as np

# 이론: h = (1/2)·g·t²
g = 9.81  # m/s²
h = 10    # 초기 높이 (m)

# 바닥에 도달하는 시간 (이론)
t_theory = np.sqrt(2 * h / g)
print(f"이론 낙하 시간: {t_theory:.3f}s")

# Isaac Sim 시뮬레이션에서 실제 시간을 측정해서 비교
# examples/02_physics/01_gravity_collision.py 참고
```

## 8. 조인트와 관절 (Joints)

로봇 팔은 강체들이 **조인트로 연결**된 구조입니다.

| 조인트 타입 | 자유도 | 예시 |
|------------|-------|------|
| Revolute (회전) | 1 | 로봇 팔꿈치 |
| Prismatic (직선) | 1 | 리프트 |
| Spherical (구) | 3 | 어깨 |
| Fixed (고정) | 0 | 용접부 |

```python
# URDF에서의 조인트 정의 예시
"""
<joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
</joint>
"""
```

## ✅ 체크리스트

- [ ] 강체의 13차원 상태 벡터를 설명할 수 있다
- [ ] F=ma와 τ=Iα의 차이를 안다
- [ ] 관성 텐서가 왜 행렬인지 이해한다
- [ ] 시뮬레이션의 dt가 작을수록 정확하지만 느린 이유를 안다
- [ ] 조인트 4가지 타입을 구분할 수 있다

➡️ 다음: [04. 로봇 운동학](04_kinematics.md)
