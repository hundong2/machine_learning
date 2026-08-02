# 04. Pointing, bounding box, trajectory

## 공식 기능 요약 번역

Gemini Robotics ER 2는 이미지의 물체를 가리키고, 영상에서 추적하고, bounding box로 탐지하며, 이동 trajectory의 waypoint를 제안할 수 있습니다. 출력 좌표는 일반적으로 0~1000 범위의 `[y, x]`입니다.

## 1. Pointing

가장 단순하고 강한 실습은 자연어 대상의 한 점을 찾는 것입니다.

```text
"파란 블록 하나의 중심점을 반환하라.
JSON: [{\"point\":[y,x],\"label\":string}]
좌표는 0~1000, y가 먼저다."
```

좋은 prompt는 다음을 명시합니다.

- 대상의 수와 선택 기준
- 좌표 순서와 범위
- 보이지 않을 때 빈 배열 반환
- JSON 외 텍스트 금지
- 중복 대상 처리

## 2. Point를 로봇 행동에 연결하기 전 검사

```text
model text
→ JSON 추출
→ schema 검사
→ 좌표 범위 검사
→ pixel 변환
→ calibration 적용
→ workspace 검사
→ 사람 승인
→ hover pose
→ 재관찰
→ 낮은 속도로 접근
```

모델 응답에서 `eval()`을 절대 사용하지 않습니다. JSON parser와 명시적 schema를 사용합니다.

## 3. Bounding box

point는 작은 물체 중심을 표현하기 쉽지만 크기와 범위를 모릅니다. box를 사용하면 다음을 계산할 수 있습니다.

- 중심점
- 면적
- aspect ratio
- 이미지 경계와의 접촉
- 여러 질의의 IoU consensus

전문 평가:

- IoU@0.5, IoU@0.75
- center distance normalized by image diagonal
- class/label accuracy
- no-object false positive
- calibration 후 robot-space error

## 4. 영상 tracking

공식 guide는 영상 파일과 “대상이 나타나는 frame마다 point” 요청을 보여줍니다. 그러나 자연어 출력만으로 frame index 계약이 불분명할 수 있습니다.

권장 schema:

```json
{
  "tracks": [
    {"time_s": 0.0, "point": [300, 420], "visible": true},
    {"time_s": 0.5, "point": null, "visible": false}
  ]
}
```

실시간 제어에는 ER 2의 저주기 semantic tracking과 OpenCV/전용 tracker의 고주기 local tracking을 결합하는 편이 낫습니다.

## 5. Trajectory

모델이 2D waypoint를 생성할 수 있어도 실제 robot trajectory는 아닙니다.

모델 trajectory가 빠뜨릴 수 있는 것:

- 3D obstacle와 높이
- gripper orientation
- joint limit와 self-collision
- velocity, acceleration, jerk
- payload와 force
- dynamic obstacle

따라서 모델 waypoint는 motion planner의 **goal hint**로 사용합니다. MoveIt 2, 충돌 검사, IK, time parameterization 같은 결정론적 계층이 최종 trajectory를 만듭니다.

## 6. Consensus

공식 best practice는 정밀 작업에서 여러 번 질의하고 결과를 평균하는 방법을 제시합니다.

단순 평균의 조건:

- 같은 물체를 가리키는지 label/box로 먼저 matching
- outlier 제거
- 분산이 threshold보다 클 때 행동 거부
- 평균으로 latency와 비용이 증가한다는 기록

```text
points = [(y1,x1), (y2,x2), ...]
median_point = coordinate-wise median
spread = max distance from median
if spread > threshold: ASK_HUMAN
```

## 실습

- `examples/01_offline_spatial_grounding.py`: sample JSON 파싱과 좌표 변환
- `examples/02_api_pointing.py`: 실제 ER 2 호출, parse, pixel 표시
- `notebooks/01_coordinate_grounding.ipynb`: 수식과 시각화

## 완료 조건

- malformed JSON과 범위 밖 좌표를 거부한다.
- no-object를 행동 없음으로 처리한다.
- 모델 trajectory를 motor command로 직접 보내지 않는다.
- pixel 오차와 robot-space 오차를 따로 측정한다.

