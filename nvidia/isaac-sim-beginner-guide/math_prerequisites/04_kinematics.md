# 🦾 04. 로봇 운동학 (Robot Kinematics)

로봇의 **관절 각도**와 **엔드이펙터(end-effector, 그리퍼) 위치**의 관계를 다룹니다.

## 1. 순기구학 (Forward Kinematics, FK)

**관절 각도 → 엔드이펙터 위치/자세**

```
관절값 q = [θ₁, θ₂, θ₃, ...] → 엔드이펙터 위치 T_ee
```

각 관절마다 변환 행렬을 곱해서 얻습니다:

```
T_ee = T₁(θ₁) · T₂(θ₂) · T₃(θ₃) · ... · T_n(θ_n)
```

### 간단한 2-DOF 평면 로봇 예제

두 링크로 된 평면 로봇 (길이 L₁, L₂):

```
       ___L₂___ (x, y)
      /
     /θ₂
    ● ← 관절2
    |
    |L₁
    |
    ●───── θ₁
   (0,0)
```

**순기구학 공식**:
```
x = L₁·cos(θ₁) + L₂·cos(θ₁+θ₂)
y = L₁·sin(θ₁) + L₂·sin(θ₁+θ₂)
```

```python
import numpy as np

def forward_kinematics_2dof(theta1, theta2, L1=1.0, L2=1.0):
    """2-DOF 평면 로봇의 순기구학"""
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

# 관절 각도 → 엔드이펙터 위치
pos = forward_kinematics_2dof(np.pi/4, np.pi/4)
print(pos)  # [0.707, 1.707]
```

## 2. 역기구학 (Inverse Kinematics, IK)

**엔드이펙터 위치 → 관절 각도**

이게 훨씬 어렵습니다. 이유:
- **여러 해가 존재** (elbow-up, elbow-down)
- **해가 없을 수도** (목표가 로봇 도달 범위 밖)
- **비선형 방정식** 이라 해석해가 일반적으로 어려움

### 2-DOF의 해석적 해 (특별한 경우)

```python
def inverse_kinematics_2dof(x, y, L1=1.0, L2=1.0):
    """2-DOF 로봇의 역기구학 (해석적 해)"""
    # 엔드이펙터까지 거리
    r = np.sqrt(x**2 + y**2)
    
    # 도달 가능성 체크
    if r > L1 + L2 or r < abs(L1 - L2):
        raise ValueError("목표가 로봇 도달 범위 밖입니다")
    
    # 코사인 법칙으로 θ₂ 계산
    cos_theta2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2 = np.arccos(cos_theta2)  # elbow-up 해
    
    # θ₁ 계산
    theta1 = np.arctan2(y, x) - np.arctan2(
        L2 * np.sin(theta2),
        L1 + L2 * np.cos(theta2)
    )
    
    return theta1, theta2
```

### 일반적인 6-DOF 로봇: 수치적 해법

일반 산업용 로봇(UR5, Franka Panda 등)은 **반복 최적화**로 해를 찾습니다.

**Jacobian-based IK (자코비안 기반)**:
```
Δq = J⁺ · Δx
```

- `J`: 자코비안 행렬 (순기구학의 미분)
- `J⁺`: 의사역행렬(pseudo-inverse)
- `Δx`: 목표 위치와 현재 위치 차이
- `Δq`: 관절 각도 변화량

```python
def jacobian_ik(target_pos, initial_q, fk_func, jacobian_func, 
                max_iter=100, threshold=1e-4, lr=0.1):
    """자코비안 기반 IK"""
    q = initial_q.copy()
    
    for i in range(max_iter):
        current_pos = fk_func(q)
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < threshold:
            return q, True  # 수렴 성공
        
        J = jacobian_func(q)
        J_pinv = np.linalg.pinv(J)
        q = q + lr * J_pinv @ error
    
    return q, False  # 수렴 실패
```

### Isaac Sim에서의 IK
Isaac Sim은 IK를 직접 구현할 필요 없이 **Lula Kinematics**를 제공합니다:

```python
# Isaac Sim의 IK 사용 예 (Franka Panda)
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver, ArticulationKinematicsSolver
)
from isaacsim.robot.manipulators.examples.franka import Franka

# 로봇 로딩
franka = Franka(prim_path="/World/Franka")

# Kinematics solver 설정
kinematics_solver = LulaKinematicsSolver(
    robot_description_path="franka_description.yaml",
    urdf_path="franka.urdf"
)
articulation_kinematics = ArticulationKinematicsSolver(
    franka, kinematics_solver, "panda_hand"
)

# 목표 위치로 IK 계산
target_position = np.array([0.5, 0.0, 0.5])
target_orientation = np.array([0, 1, 0, 0])  # 쿼터니언

action, success = articulation_kinematics.compute_inverse_kinematics(
    target_position, target_orientation
)
```

## 3. DH 파라미터 (Denavit-Hartenberg)

로봇의 기하학적 구조를 **4개의 파라미터**로 표현:

| 파라미터 | 의미 |
|---------|------|
| `a` (link length) | 링크의 길이 |
| `α` (link twist) | 링크의 꼬임 |
| `d` (link offset) | 조인트 오프셋 |
| `θ` (joint angle) | 조인트 각도 |

```
 i번째 관절의 변환 행렬:
T_i = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

현대적인 로봇 설명에서는 **URDF**나 **USD**를 더 많이 쓰지만, 이론적 이해를 위해 알아두면 좋습니다.

## 4. 속도 운동학 (Velocity Kinematics)

관절 속도 `q̇`와 엔드이펙터 속도 `ẋ`의 관계:

```
ẋ = J(q) · q̇
```

여기서 `J`는 **자코비안 행렬**:
```
J = | ∂x/∂θ₁  ∂x/∂θ₂  ...  ∂x/∂θ_n |
    | ∂y/∂θ₁  ∂y/∂θ₂  ...  ∂y/∂θ_n |
    | ∂z/∂θ₁  ∂z/∂θ₂  ...  ∂z/∂θ_n |
    | ... (회전 부분도 추가)          |
```

### 2-DOF 로봇의 자코비안

```python
def jacobian_2dof(theta1, theta2, L1=1.0, L2=1.0):
    """2-DOF 로봇의 자코비안"""
    J = np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1+theta2), -L2*np.sin(theta1+theta2)],
        [ L1*np.cos(theta1) + L2*np.cos(theta1+theta2),  L2*np.cos(theta1+theta2)]
    ])
    return J
```

## 5. 특이점 (Singularity)

`det(J) = 0`이 되는 자세. 이때 로봇은:
- 특정 방향으로 움직이지 못함
- IK 해가 무한대로 발산 가능

예시: 2-DOF 로봇에서 팔이 완전히 펴졌을 때(`θ₂ = 0`).

```python
def is_singular(J, threshold=1e-3):
    """특이점 근처인지 확인"""
    # SVD의 최소 특이값이 0에 가까우면 특이점
    _, singular_values, _ = np.linalg.svd(J)
    return singular_values.min() < threshold
```

## 6. 동역학 (Dynamics) — 힘/토크까지 고려

순기구학/역기구학이 **위치**만 다뤘다면, 동역학은 **힘과 토크**까지 다룹니다.

**로봇의 운동방정식**:
```
τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)
```

- `M(q)`: 질량 행렬
- `C(q,q̇)`: 코리올리/원심력
- `G(q)`: 중력 항
- `τ`: 관절 토크

Isaac Sim의 PhysX가 이를 자동으로 계산해줍니다.

## ✅ 체크리스트

- [ ] FK와 IK의 차이를 설명할 수 있다
- [ ] IK가 왜 여러 해를 가질 수 있는지 이해한다
- [ ] 자코비안의 역할을 안다
- [ ] 특이점이 무엇인지 설명할 수 있다
- [ ] URDF가 무엇인지 안다

## 🎓 더 공부하고 싶다면

- 교재: **"Modern Robotics"** by Kevin Lynch (무료 강의: https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
- 교재: **"Robot Dynamics and Control"** by Spong
- Coursera: "Robotics Specialization" by UPenn

➡️ 이론 학습 완료! 이제 실습으로 넘어갑니다: [examples/01_basics/](../examples/01_basics/)
