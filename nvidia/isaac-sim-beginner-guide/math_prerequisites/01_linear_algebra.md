# 📐 01. 선형대수 기초 (로봇공학 관점)

로봇 시뮬레이션을 이해하려면 **벡터와 행렬**을 다룰 줄 알아야 합니다.
여기서는 Isaac Sim에서 실제로 쓰이는 개념만 정리합니다.

## 1. 벡터 (Vector)

로봇의 **위치**, **속도**, **힘**은 모두 3D 벡터로 표현합니다.

```python
import numpy as np

# 위치 벡터 (x, y, z) — 단위: meter
position = np.array([1.0, 0.5, 0.3])

# 속도 벡터 (vx, vy, vz) — 단위: m/s
velocity = np.array([0.1, 0.0, 0.0])

# Isaac Sim의 좌표계: +Z가 위쪽, 오른손 좌표계
```

### 주요 연산

| 연산 | 의미 | NumPy |
|------|------|-------|
| 내적 (dot) | 두 벡터의 유사도, 투영 | `np.dot(a, b)` |
| 외적 (cross) | 회전축, 토크 계산 | `np.cross(a, b)` |
| 노름 (norm) | 벡터의 크기 (거리) | `np.linalg.norm(v)` |
| 정규화 | 단위 벡터로 변환 | `v / np.linalg.norm(v)` |

```python
# 두 점 사이의 거리 계산 (로봇이 목표까지 얼마나 떨어졌는가)
robot_pos = np.array([0, 0, 0])
target_pos = np.array([3, 4, 0])
distance = np.linalg.norm(target_pos - robot_pos)  # 5.0 (3-4-5 삼각형)
```

## 2. 행렬 (Matrix)

**변환(Transformation)** 을 표현하는 핵심 도구입니다.

### 2.1 회전 행렬 (Rotation Matrix) — 3×3

Z축 기준 θ만큼 회전:

```
R_z(θ) = | cos(θ)  -sin(θ)  0 |
         | sin(θ)   cos(θ)  0 |
         |   0       0      1 |
```

```python
def rotation_z(theta):
    """Z축 회전 행렬"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# 점 (1, 0, 0)을 Z축 기준 90도 회전 → (0, 1, 0)
point = np.array([1, 0, 0])
rotated = rotation_z(np.pi/2) @ point
print(rotated)  # [0, 1, 0]
```

### 2.2 동차 변환 행렬 (Homogeneous Transformation) — 4×4

**회전 + 이동**을 한 번에 표현. 로봇 팔의 각 링크 자세를 이걸로 표현합니다.

```
T = | R  t |  ← R: 3×3 회전, t: 3×1 이동
    | 0  1 |
```

```python
def make_transform(R, t):
    """4x4 동차 변환 행렬 생성"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# 로봇 베이스에서 엔드이펙터까지의 변환을 체인으로 계산
T_base_to_link1 = make_transform(rotation_z(0.5), [0, 0, 0.1])
T_link1_to_link2 = make_transform(rotation_z(0.3), [0.2, 0, 0])
T_base_to_link2 = T_base_to_link1 @ T_link1_to_link2  # 행렬 곱
```

## 3. 왜 중요한가? (Isaac Sim에서의 쓰임)

| 개념 | Isaac Sim에서의 사용처 |
|------|---------------------|
| 3D 벡터 | `prim.set_world_pose()`의 position 인자 |
| 쿼터니언 | 회전을 표현 (다음 문서에서 자세히) |
| 변환 행렬 | 로봇 관절의 순기구학(Forward Kinematics) |
| 내적/외적 | 힘, 토크, 각속도 계산 |

## 4. 실습 코드

```python
# examples/01_basics에서 실행 가능
import numpy as np

# 로봇이 원을 그리며 움직이도록 위치 계산
radius = 1.0
num_points = 36
circle_points = []
for i in range(num_points):
    theta = 2 * np.pi * i / num_points
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 0.5
    circle_points.append([x, y, z])

# 각 점에서의 속도 벡터 (접선 방향)
velocities = []
for i in range(num_points):
    theta = 2 * np.pi * i / num_points
    vx = -radius * np.sin(theta)  # 접선 속도의 x 성분
    vy = radius * np.cos(theta)   # 접선 속도의 y 성분
    velocities.append([vx, vy, 0])
```

## ✅ 체크리스트

- [ ] 벡터의 내적/외적을 설명할 수 있다
- [ ] 회전 행렬 3×3을 이해한다
- [ ] 동차 변환 행렬 4×4의 용도를 안다
- [ ] `A @ B`와 `A * B`의 차이를 안다 (`@`는 행렬곱, `*`는 원소별 곱)

➡️ 다음: [02. 회전과 변환](02_rotations.md)
