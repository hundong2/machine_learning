# 04. ONNX 계약과 출력 동등성

## 목표

ONNX 파일을 “변환된 모델”이 아니라 Python 생산자와 C++ 소비자 사이의 버전이 있는 API로 다룹니다.

## 1단계: 가장 단순한 모델부터 내보내기

초기 기준은 다음과 같습니다.

- batch 1
- 고정 입력 크기
- FP32
- CPU 실행
- 외부 전처리

Ultralytics의 현재 내보내기 예시는 다음 패턴입니다.

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.export(
    format="onnx",
    imgsz=416,
    dynamic=False,
    simplify=True,
    opset=None,
)
```

PyTorch 모델을 직접 내보낸다면 PyTorch 2.6+의 `torch.export` 기반 ONNX exporter와 `dynamo=True` 경로를 우선 검토합니다. dynamic input은 최신 공식 문서의 `dynamic_shapes`를 사용합니다.

고정 FP32 모델의 parity를 먼저 통과한 뒤 dynamic shape, FP16, INT8, 내장 NMS를 각각 독립 실험으로 추가합니다.

## 2단계: 출력 형식을 실제로 검사

모델 세대와 export 옵션에 따라 출력은 다릅니다.

- end-to-end detection: 예를 들어 `(batch, max_detections, 6)`와 `[x1,y1,x2,y2,score,class]`
- raw detection head: 예를 들어 `(batch, 4+classes, candidates)`이고 confidence 계산과 NMS가 외부 책임
- 일부 모델: 여러 output tensor 또는 다른 축 순서

따라서 C++ 코드에서 shape를 추측하지 않습니다. 모델을 로드할 때 이름, dtype, rank, 각 차원을 출력하고 계약과 다르면 즉시 실패시킵니다.

## 3단계: 계약 파일

예시:

```yaml
contract_version: 1
model:
  family: yolo
  artifact: person-tracker-fp32.onnx
  sha256: "REPLACE_ME"
input:
  name: images
  dtype: float32
  layout: NCHW
  shape: [1, 3, 416, 416]
  color: RGB
  range: [0.0, 1.0]
  resize: letterbox
  pad_value: 114
output:
  mode: raw
  names: [output0]
  coordinates: cxcywh_network_pixels
postprocess:
  confidence_threshold: 0.35
  iou_threshold: 0.50
  nms: external_class_aware
classes:
  0: person
```

`output.mode`와 좌표 단위는 실제 모델을 검사해 작성합니다. 예시를 복사해 맞다고 가정하면 안 됩니다.

## 4단계: 그래프 검증

```python
import onnx
import onnxruntime as ort

model = onnx.load("models/person-tracker-fp32.onnx")
onnx.checker.check_model(model)

session = ort.InferenceSession(
    "models/person-tracker-fp32.onnx",
    providers=["CPUExecutionProvider"],
)

for item in session.get_inputs():
    print(item.name, item.type, item.shape)
for item in session.get_outputs():
    print(item.name, item.type, item.shape)
```

ONNX checker 통과는 원본 모델과 결과가 같다는 뜻이 아닙니다. 그래프 유효성 다음에 수치·의미 parity를 별도로 시험합니다.

## 5단계: 골든 parity 시험

50개 이상의 고정 입력으로 다음을 비교합니다.

### 전처리 parity

- tensor shape와 dtype
- 최대 절대 오차
- 평균 절대 오차
- 채널별 최소·최대
- 특정 픽셀의 RGB 값

### 출력 tensor parity

- output 이름과 shape
- finite 값 여부
- 최대·평균 오차
- 분포 요약

### 의미 parity

- class 일치율
- confidence 차이
- Hungarian 또는 greedy IoU matching
- 매칭 박스 IoU
- 누락·추가 탐지 수

FP32 CPU 기준 초기 목표 예시:

- class 일치율 ≥ 99%
- 매칭 박스 IoU ≥ 0.99
- confidence 절대 차이 ≤ 1e-4~1e-3

정확한 허용 오차는 모델과 연산자에 따라 조정하고 근거를 남깁니다.

## 6단계: 버전과 무결성

모델 파일 옆에 다음을 둡니다.

```text
person-tracker-fp32.onnx
person-tracker-fp32.onnx.sha256
person-tracker-fp32.contract.yaml
person-tracker-fp32.model-card.md
```

애플리케이션 시작 시 checksum과 계약 버전을 검사합니다. 파일 이름만 같고 내용이 바뀌는 “조용한 모델 교체”를 금지합니다.

## 7단계: 최적화 실험

한 번에 하나만 바꿉니다.

1. ORT graph optimization
2. dynamic shape
3. input resolution
4. FP16
5. INT8 calibration
6. 실행 provider
7. TensorRT 또는 전용 NPU

각 실험은 정확도 parity, p50/p95 지연, 메모리, 모델 크기를 동일 입력으로 비교합니다. TensorRT는 NVIDIA GPU/Jetson 경로이며 Raspberry Pi CPU의 기본 해법이 아닙니다.

## 완료 조건

- 모델 checksum과 계약 파일이 있다.
- 입력·출력 메타데이터를 자동 검사한다.
- Python 원본과 Python ORT parity가 통과한다.
- C++에서 동일 골든 입력을 검사할 준비가 되어 있다.
- 모델 변경 시 실패하는 계약 테스트가 있다.

## 흔한 실패

| 증상 | 우선 확인 |
| --- | --- |
| 결과가 전혀 다름 | RGB/BGR, 0~1 정규화, NCHW |
| 박스가 일정하게 밀림 | letterbox padding과 좌표 복원 |
| 탐지가 너무 많음 | raw output에 NMS 누락 |
| 클래스 점수가 이상함 | objectness 유무와 축 순서 |
| 개발 PC만 성공 | opset, provider, unsupported op |
| 새 모델에서 crash | output rank/shape 하드코딩 |

