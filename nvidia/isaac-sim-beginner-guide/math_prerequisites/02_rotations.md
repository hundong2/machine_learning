# 🔄 02. 회전 표현 (Rotation Representations)

회전을 표현하는 방법은 여러 가지입니다. Isaac Sim에서는 주로 **쿼터니언(Quaternion)** 을 사용합니다.

## 1. 오일러 각 (Euler Angles) — Roll, Pitch, Yaw

가장 직관적이지만 **짐벌락(Gimbal Lock)** 문제가 있습니다.

```
- Roll  (φ): X축 기준 회전 (비행기가 옆으로 기울어짐)
- Pitch (θ): Y축 기준 회전 (비행기가 위아래로 끄덕임)
- Yaw   (ψ): Z축 기준 회전 (비행기가 좌우로 방향 전환)
```

```python
import numpy as np

# 오일러 각 (라디안)
roll, pitch, yaw = 0.1, 0.2, 0.3

# 오일러 각 → 회전 행렬 (ZYX 순서, 로봇공학 표준)
def euler_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    
    return R_z @ R_y @ R_x
```

### ⚠️ 짐벌락 문제
두 회전축이 일치하면 자유도가 하나 사라집니다. 이 때문에 3D 시뮬레이션에서는 쿼터니언을 선호합니다.

## 2. 쿼터니언 (Quaternion)

**4개의 숫자 [w, x, y, z]** 로 회전을 표현. 짐벌락이 없고, 보간이 부드럽습니다.

```
q = w + xi + yj + zk
```

- `w`: 실수부 (scalar) — `cos(θ/2)`
- `(x, y, z)`: 허수부 (vector) — `sin(θ/2) * axis`

**단위 쿼터니언**은 `w² + x² + y² + z² = 1`을 만족해야 합니다.

```python
def axis_angle_to_quaternion(axis, angle):
    """축-각(axis-angle) 표현을 쿼터니언으로 변환"""
    axis = axis / np.linalg.norm(axis)  # 정규화
    half = angle / 2
    w = np.cos(half)
    x, y, z = np.sin(half) * axis
    return np.array([w, x, y, z])

# Z축 기준 90도 회전
q = axis_angle_to_quaternion(np.array([0, 0, 1]), np.pi/2)
print(q)  # [0.707, 0, 0, 0.707]
```

### Isaac Sim에서의 쿼터니언 사용

```python
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# 쿼터니언 순서 주의: Isaac Sim은 (w, x, y, z) 순서
cube = DynamicCuboid(
    prim_path="/World/Cube",
    position=np.array([0, 0, 1]),
    orientation=np.array([0.707, 0, 0, 0.707])  # Z축 90도 회전
)
```

### 쿼터니언 연산

```python
def quaternion_multiply(q1, q2):
    """두 쿼터니언의 곱 (회전 합성)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    """쿼터니언 역(켤레). 역회전을 의미"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])
```

## 3. 회전 표현 비교

| 표현 | 파라미터 수 | 장점 | 단점 |
|------|------------|------|------|
| 오일러 각 | 3 | 직관적 | 짐벌락 |
| 회전 행렬 | 9 | 연산 직관적 | 저장 비효율 |
| **쿼터니언** | **4** | **짐벌락 없음, 부드러운 보간** | **직관적이지 않음** |
| 축-각 | 4 | 직관적 | 보간 어려움 |

## 4. SLERP (구면 선형 보간)

두 회전 사이를 부드럽게 이어주는 기법. 로봇 모션 플래닝에 필수.

```python
def slerp(q1, q2, t):
    """
    두 쿼터니언 사이의 구면 보간
    t: 0 ~ 1 (0이면 q1, 1이면 q2)
    """
    dot = np.dot(q1, q2)
    
    # 최단 경로로 회전하기 위해 부호 조정
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # 두 쿼터니언이 거의 같으면 선형 보간
    if dot > 0.9995:
        return q1 + t * (q2 - q1)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return s1 * q1 + s2 * q2
```

## 5. scipy를 사용한 간편한 변환

`scipy.spatial.transform.Rotation`을 사용하면 매우 편리합니다.

```python
from scipy.spatial.transform import Rotation as R

# 오일러 각에서 생성
r = R.from_euler('xyz', [30, 45, 60], degrees=True)

# 다양한 형식으로 변환
print(r.as_quat())        # [x, y, z, w] — scipy 순서 주의!
print(r.as_matrix())      # 3x3 회전 행렬
print(r.as_euler('xyz'))  # 오일러 각

# ⚠️ scipy는 (x,y,z,w), Isaac Sim은 (w,x,y,z) 순서
# 변환 시 주의!
scipy_quat = r.as_quat()  # [x, y, z, w]
isaac_quat = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])
```

## ✅ 체크리스트

- [ ] Roll, Pitch, Yaw가 무엇인지 안다
- [ ] 짐벌락 문제를 설명할 수 있다
- [ ] 쿼터니언 4개 숫자의 의미를 안다
- [ ] Isaac Sim의 쿼터니언 순서가 (w,x,y,z)임을 기억한다
- [ ] SLERP의 용도를 안다

➡️ 다음: [03. 강체 동역학](03_rigid_body_dynamics.md)
