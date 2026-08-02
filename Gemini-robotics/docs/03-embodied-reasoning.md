# 03. Embodied reasoning과 로봇 좌표계

## 1. Embodied reasoning이란

텍스트 상식만이 아니라 물리 공간에서 행동하기 위해 필요한 추론입니다.

- object와 part의 위치
- 앞/뒤/안/위 같은 공간 관계
- 무엇을 잡을 수 있는지에 대한 affordance
- 어떤 경로가 충돌을 피하는지
- 행동이 진행 중인지 완료됐는지
- 로봇의 payload·gripper·workspace로 가능한지
- 모호할 때 사람에게 무엇을 물어야 하는지

## 2. Gemini의 정규화 좌표

공식 spatial guide의 point는 `[y, x]` 순서이며 각 값은 0~1000입니다.

```text
[0, 0]       = 왼쪽 위
[500, 500]   = 이미지 중앙 근처
[1000, 1000] = 오른쪽 아래 경계
```

이미지 폭 `W`, 높이 `H`일 때:

```text
pixel_x = x_norm / 1000 × (W - 1)
pixel_y = y_norm / 1000 × (H - 1)
```

순서를 `[x, y]`로 잘못 읽는 것이 가장 흔한 오류입니다.

## 3. Bounding box

공식 예시는 다음 형식을 사용합니다.

```json
{"label":"blue block","y":200,"x":300,"y2":420,"x2":520}
```

- `(x, y)`: 왼쪽 위
- `(x2, y2)`: 오른쪽 아래
- 모든 값: 0~1000 정규화

검증:

- `0 ≤ x < x2 ≤ 1000`
- `0 ≤ y < y2 ≤ 1000`
- 최소 면적
- 이미지 경계 clamp가 아니라 잘못된 결과 거부를 기본으로 함

## 4. 2D에서 실제 로봇 좌표로

2D point만으로 일반적인 3D 위치는 결정할 수 없습니다. 가능한 방법은 다음과 같습니다.

### 평면 작업대

모든 물체가 같은 테이블 평면에 있다고 가정하면 homography로 픽셀을 테이블 `(X, Y)`에 매핑할 수 있습니다.

```text
[u, v, 1] --H--> [X', Y', w]
X = X'/w, Y = Y'/w
```

`labs/geometry.py`의 `PlanarCalibration`이 이 계산을 수행합니다.

### 깊이 카메라

depth `Z`와 camera intrinsic을 사용합니다.

```text
X_cam = (u - cx) / fx × Z
Y_cam = (v - cy) / fy × Z
Z_cam = Z
```

이후 camera extrinsic으로 robot base frame에 변환합니다.

### 다중 시점

두 카메라의 대응점과 calibration으로 triangulation합니다. Gemini의 multi-view correspondence는 대응 후보를 찾는 데 도움을 줄 수 있지만 최종 기하 검증이 필요합니다.

## 5. 좌표 frame

이름 없는 `(x, y, z)`는 위험합니다.

```json
{
  "frame_id": "table",
  "units": "meter",
  "x": 0.31,
  "y": -0.12,
  "z": 0.08,
  "observed_at": "2026-08-02T10:00:00Z"
}
```

최소한 frame, 단위, timestamp, calibration version을 함께 전달합니다.

## 6. 오차 전파

모델 point 오차, 카메라 calibration 오차, depth noise, robot kinematic 오차가 합쳐집니다. 따라서 한 점을 절대 진실로 사용하지 않습니다.

- 여러 번 질의해 consensus 계산
- box 중심과 local detector 결합
- uncertainty radius 추가
- 접근 단계마다 다시 관찰
- 마지막 grasp는 force/tactile 센서로 검증

## 완료 조건

- `[y, x]`와 pixel `(x, y)` 차이를 설명한다.
- 정규화 point·box를 검증하고 변환한다.
- 2D point가 곧 3D robot pose가 아님을 설명한다.
- calibration version과 frame을 데이터 계약에 포함한다.

