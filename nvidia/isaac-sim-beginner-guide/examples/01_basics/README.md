# 📘 Phase 1: Isaac Sim 기본기

Isaac Sim의 가장 기본적인 사용법을 익힙니다.

## 📋 예제 목록

### 01_hello_world.py
**목표**: Isaac Sim을 처음 실행하고 빈 씬에 바닥을 추가합니다.
```bash
python 01_hello_world.py
```

**배우는 내용**:
- `SimulationApp`을 **반드시 먼저** 생성해야 하는 이유
- `World`와 `physics_dt`의 의미
- 시뮬레이션 루프 구조

### 02_add_primitives.py
**목표**: 큐브, 구, 실린더 같은 기본 도형을 추가합니다.
```bash
python 02_add_primitives.py
```

**배우는 내용**:
- `VisualCuboid` vs `DynamicCuboid` (시각 vs 물리)
- 위치, 회전(쿼터니언), 크기, 색상 설정
- Prim Path 계층 구조 (USD 기본)

### 03_camera_and_lighting.py
**목표**: 카메라와 다양한 조명을 추가합니다.
```bash
python 03_camera_and_lighting.py
```

**배우는 내용**:
- 조명의 종류 (Distant, Dome, Sphere 등)
- 카메라 위치와 방향 설정
- 이미지 캡처 및 저장

## 🎯 완료 후 이해해야 할 것

- [ ] `SimulationApp` import 순서가 왜 중요한가?
- [ ] Isaac Sim의 좌표계 (Z-up, 오른손 좌표계)
- [ ] 쿼터니언의 (w, x, y, z) 순서
- [ ] 시뮬레이션의 dt와 렌더링 dt의 차이

## ⚠️ 자주 하는 실수

1. **import 순서**: `from isaacsim import SimulationApp`를 먼저 하고 `simulation_app = SimulationApp(...)`를 호출한 **뒤에** 다른 isaacsim 모듈을 import해야 합니다.

2. **좌표계 혼동**: Isaac Sim은 Z축이 위쪽입니다 (+Z = up). Y축이 위인 다른 엔진과 다릅니다.

3. **쿼터니언 순서**: Isaac Sim은 `(w, x, y, z)`, scipy는 `(x, y, z, w)`입니다.

## ➡️ 다음 단계

[Phase 2: 물리 시뮬레이션](../02_physics/) 으로 이동하세요.
