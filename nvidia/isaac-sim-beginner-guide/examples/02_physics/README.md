# ⚛️ Phase 2: 물리 시뮬레이션

PhysX 엔진을 이용한 물리 시뮬레이션을 배웁니다.

## 📋 예제 목록

### 01_gravity_collision.py
**목표**: 자유낙하와 반발 계수의 영향 관찰
```bash
python 01_gravity_collision.py
```

**배우는 내용**:
- 복원 계수(restitution) 비교
- 이론 낙하 시간과 시뮬레이션 결과 비교
- Physics Material 설정

### 02_rigid_bodies.py
**목표**: 힘, 토크, 속도 적용
```bash
python 02_rigid_bodies.py
```

**배우는 내용**:
- `set_linear_velocity` / `set_angular_velocity`
- 포물선 운동
- F = ma, τ = Iα 확인

### 03_joints_and_articulations.py
**목표**: 조인트로 2-링크 로봇 만들기
```bash
python 03_joints_and_articulations.py
```

**배우는 내용**:
- Revolute Joint (회전 조인트)
- Articulation Root API
- PD Controller의 stiffness / damping

## 💡 핵심 개념

### physics_dt의 선택
```python
# 기본값 (60Hz) — 대부분 OK
physics_dt=1.0/60.0

# 높은 정확도 (240Hz) — 충돌 많거나 빠른 움직임
physics_dt=1.0/240.0

# 낮은 정확도 (30Hz) — 매우 느린 시뮬레이션 (권장X)
physics_dt=1.0/30.0
```

### 물리 안정성 팁
- 물체가 관통하면 → `physics_dt`를 줄이세요
- 시뮬레이션이 폭발적으로 튀면 → 질량 비율 확인 (1000:1 이상이면 불안정)
- 조인트가 삐걱거리면 → stiffness와 damping 튜닝

## ➡️ 다음 단계

[Phase 3: 로봇 제어](../03_robotics/)
