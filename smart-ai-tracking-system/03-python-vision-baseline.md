# 03. Python 비전 기준선 만들기

## 목표

Python 구현은 최종 제품이 아니라 C++ 구현이 맞는지 판단하는 **실행 가능한 명세**입니다. 같은 입력에 대해 전처리, 모델 출력, 후처리, 최종 탐지 결과를 저장해야 합니다.

## 1단계: 고정 입력 준비

웹캠 실시간 영상만으로 개발하면 매번 입력이 달라 회귀를 찾기 어렵습니다.

1. 20~30초짜리 MP4를 준비합니다.
2. 한 사람, 여러 사람, 부분 가림, 빠른 이동, 대상 유실을 포함합니다.
3. 영상의 SHA-256, 해상도, FPS, 코덱을 기록합니다.
4. 추가로 작은 정지 이미지 20~50개를 `golden/` 세트로 만듭니다.

개인정보가 포함된 영상은 공개 저장소에 올리지 않습니다. 공개 가능한 자체 촬영 영상은 등장 인물의 동의를 받고, 아니면 라이선스가 명확한 공개 영상을 사용합니다.

## 2단계: 사전학습 모델로 수직 경로 연결

먼저 작은 사전학습 모델로 `person` 탐지만 구현합니다. 최신 Ultralytics 예시에서는 다음과 같은 패턴을 사용합니다.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.predict(
    source="samples/tracking.mp4",
    classes=[0],
    conf=0.35,
    imgsz=416,
    stream=True,
    verbose=False,
)

for frame_id, result in enumerate(results):
    # boxes.xyxy, boxes.conf, boxes.cls를 프로젝트 공통 형식으로 변환
    pass
```

모델 이름은 예시일 뿐입니다. YOLOv8을 계속 사용해도 되지만 다음을 고정합니다.

- 모델 파일 checksum
- `ultralytics`, `torch`, CUDA/CPU 버전
- 입력 크기와 letterbox 정책
- confidence와 IoU threshold
- class map
- NMS가 모델 내부인지 외부인지

모델 버전을 최신으로 바꾸는 것은 단순 업그레이드가 아니라 모델 계약 변경입니다. 별도 브랜치에서 parity와 성능을 다시 검증합니다.

## 3단계: 공통 탐지 형식 만들기

Python과 C++가 다음 의미 형식을 공유하게 합니다.

```json
{
  "frame_id": 42,
  "capture_ts_ns": 123456789,
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.9123,
      "xyxy": [421.4, 103.2, 812.6, 698.1]
    }
  ]
}
```

좌표는 원본 이미지 픽셀 기준의 `x1,y1,x2,y2`로 통일합니다. JSON에는 소수점 자리수를 제한하되, 비교 전 내부 값은 float로 유지합니다.

## 4단계: 전처리를 독립 함수로 분리

전처리는 최소한 다음 정보를 함께 반환합니다.

```python
@dataclass(frozen=True)
class PreprocessMeta:
    original_hw: tuple[int, int]
    network_hw: tuple[int, int]
    scale: float
    pad_left: int
    pad_top: int
```

명시할 내용:

- OpenCV BGR → 모델 RGB 변환 여부
- uint8 → float32
- `[0,255]` → `[0,1]` 정규화
- HWC → CHW
- batch 차원 추가
- stretch인지 aspect ratio 보존 letterbox인지
- padding 값과 홀수 padding 분배

이 정보가 없으면 C++ 결과가 “거의 맞지만 박스가 조금 밀리는” 문제가 생깁니다.

## 5단계: 평가 지표

### 모델 지표

- precision, recall
- mAP@0.5
- mAP@0.5:0.95
- 조도·거리·가림 slice별 recall

### 제품 지표

- `person`이 없는 프레임의 오탐률
- 대상 선택이 바뀐 횟수
- 탐지 유실 구간 길이
- 프레임당 Python 종단 지연

모델의 mAP가 높아도 카메라 흔들림 때문에 target switching이 많으면 추적 제품은 실패합니다.

## 6단계: 파인튜닝 결정

다음 조건 중 하나가 확인될 때만 커스텀 학습으로 이동합니다.

- 목표 환경 slice에서 recall이 요구 기준보다 낮다.
- 탐지할 물체가 사전학습 클래스에 없다.
- 작은 물체나 특수 시점 때문에 반복되는 오류가 있다.

학습 전에 오류 50개를 분류하고, 데이터로 해결할 문제인지 threshold·추적·카메라 배치로 해결할 문제인지 나눕니다.

## 실습

1. 고정 영상으로 기준 결과 JSONL을 생성합니다.
2. 10개 프레임의 전처리 tensor를 `.npy`로 저장합니다.
3. letterbox 전·후 이미지를 나란히 저장해 좌표 복원을 눈으로 확인합니다.
4. 신뢰도 threshold를 0.2~0.7로 바꾸며 precision/recall 변화를 기록합니다.
5. 가장 나쁜 20개 프레임에 오류 유형 태그를 붙입니다.

## 완료 조건

- 같은 환경에서 두 번 실행한 결과가 허용 오차 내 동일하다.
- 전처리 메타데이터만으로 원본 좌표를 복원할 수 있다.
- 골든 이미지, 전처리 tensor, 최종 탐지 JSON이 있다.
- 파인튜닝 필요 여부를 오류 분석으로 설명할 수 있다.

## 면접 포인트

> “사전학습 모델로 먼저 end-to-end 기준선을 만들고, 조도와 가림 slice에서 recall을 측정했습니다. 커스텀 학습은 전체 mAP가 아니라 실제 실패 slice를 개선할 때만 수행했습니다.”

