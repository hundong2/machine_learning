<!-- curriculum: cycle=3; level=production-engineering; source_index=6/18; source=02-05.YOLO.md; part=1/3 -->

# YOLOv1 운영: letterbox ABI부터 결정적 NMS까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-05 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 6/18 · `02-05.YOLO.md` |
| Part | 1/3 · YOLOv1 dense grid 운영 계약 |
| 이전 소스 | `02-04.Fast_R_CNN.md` |
| 다음 소스 | `02-05.YOLO.md` Part 2/3 · SSD/FPN 다중 스케일 운영 |

## 오늘의 운영 질문

YOLOv1은 한 번의 network forward로 grid별 box와 class score를 만든다. 그러나 모델 파일 하나만 배포해서는 같은 검출 결과를 보장할 수 없다.

- 원본 이미지를 정사각 canvas로 바꾼 정확한 resize와 padding 수치는 어디에 남기는가?
- `NCHW` head를 grid tensor로 바꿀 때 box와 class channel 순서를 어떻게 고정하는가?
- Python, C++, C#이 box의 끝점, clip 시점, score 계산을 같은 의미로 구현하는가?
- candidate가 많은 장면에서 NMS latency와 메모리를 어떻게 제한하는가?
- 같은 score가 나온 두 box의 순서가 runtime마다 뒤집히지 않게 할 수 있는가?
- shape는 정상인데 box가 일정하게 밀리는 장애를 어떤 metric으로 찾는가?

1회차는 YOLOv1의 grid 책임과 손실 직관을, 2회차는 target assignment·loss·AP 구현을 다뤘다. 오늘은 이를 반복하지 않고 **전처리 계보, dense head ABI, 결정적 후처리, 크로스런타임 golden, 운영 telemetry**로 확장한다.

원본 [02-05.YOLO.md](../05.ImageClassification/02-05.YOLO.md)는 YOLOv1, SSD/FPN, Focal Loss/RetinaNet의 세 주제를 포함한다. 방대한 소스를 건너뛰지 않기 위해 이번 회차도 세 Part로 나눈다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

- YOLOv1과 후대 anchor 기반 YOLO의 출력 계약을 혼동하지 않는다.
- 실제 반올림된 resize 크기를 이용해 letterbox 정방향·역방향 좌표를 계산한다.
- raw head의 layout, channel order, activation, score 의미를 manifest에 고정한다.
- 후보 생성, stable top-k, class별 NMS, clipping 순서를 재현한다.
- tensor shape와 후보 수에 따른 activation·NMS 비용을 산정한다.
- NumPy oracle과 실행 가능한 PyTorch serving core를 교차 검증한다.
- C++17과 C#에서 같은 letterbox box를 원본 좌표로 복원한다.
- ONNX release gate, canary, drift·latency monitoring, rollback을 설계한다.

## 선수 지식과 기호

- `cxcywh`, `xyxy`, 반열린 box 구간
- IoU, confidence threshold, class별 NMS
- `NCHW`·`NHWC`, `float32`·`float16`, `int64`
- resize, letterbox, affine transform
- p50·p95·p99 latency, dynamic batching
- ONNX graph와 execution provider

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $H_0,W_0$ | 원본 이미지 높이와 너비 |
| $H_t,W_t$ | model canvas 높이와 너비 |
| $H_r,W_r$ | 실제 정수 resize 높이와 너비 |
| $S$ | YOLOv1 grid 한 축의 cell 수 |
| $B$ | cell당 box predictor 수 |
| $C$ | class 수 |
| $K$ | pre-NMS로 유지할 최대 후보 수 |
| $b=(x_1,y_1,x_2,y_2)$ | 반열린 corner box |

## 1. 직관: 검출 결과는 모델과 decoder의 공동 산출물이다

분류 모델은 class logit의 순서를 잘못 읽으면 바로 label이 틀어진다. 검출 모델은 더 많은 계약이 동시에 맞아야 한다. 같은 raw tensor라도 다음 중 하나만 달라지면 완전히 다른 box가 된다.

1. 원본 좌표인지 canvas 좌표인지
2. `xyxy`인지 `cxcywh`인지
3. 좌표가 pixel인지 $[0,1]$ 정규화 값인지
4. channel-first인지 channel-last인지
5. class score가 conditional probability인지 독립 logit인지
6. padding을 언제 제거하는지
7. NMS가 class별인지 class-agnostic인지

따라서 배포 단위는 `model.onnx` 하나가 아니다. 최소한 `model.onnx`, `manifest.json`, class map, 전처리·후처리 구현, golden fixture, calibration report를 하나의 immutable bundle로 묶어야 한다.

## 2. 원본에서 유지할 것과 교정할 것

원본의 YOLOv1 부분에서 유지할 핵심은 다음과 같다.

- 이미지를 $S\times S$ grid로 보고 객체 중심이 들어간 cell에 책임을 준다.
- 각 cell은 $B$개 box와 한 벌의 $C$개 conditional class score를 예측한다.
- box confidence는 객체 존재와 localization 품질을 함께 반영하도록 학습한다.
- proposal stage 없이 dense output을 한 번에 만든다.

운영 계약으로 옮길 때는 다음을 바로잡아야 한다.

### 2.1 YOLOv1은 일반적인 최신 YOLO decoder가 아니다

YOLOv1의 cell class score는 box마다 한 벌이 아니라 cell마다 한 벌이다. 또한 원 논문의 $w,h$ parameterization과 SSE 손실을 최신 YOLO 계열의 anchor, objectness, BCE, distribution regression에 그대로 적용할 수 없다.

| 항목 | YOLOv1 | 후대 구현에서 흔한 선택 |
| --- | --- | --- |
| class score | cell마다 $C$개 | anchor 또는 point마다 $C$개 |
| box prior | 명시적 anchor 없음 | anchor 기반 또는 anchor-free point |
| 출력식 | $S\times S\times(5B+C)$ | 여러 feature level과 구현별 channel schema |
| loss | SSE 중심 | BCE, IoU 계열, distribution loss 등 |
| decoder | cell offset + image-relative 크기 | stride·anchor·distance 등 구현별 상이 |

`family=yolo`만으로 decoder를 고르면 안 된다. `architecture=yolov1`, `decoder_schema=yolov1-cell-v1`처럼 정확한 revision을 기록한다.

### 2.2 원본의 속도 수치는 현재 SLA가 아니다

원본의 FPS는 당시 논문과 hardware·입력 크기·batch 조건의 결과다. 운영에서는 decode, image copy, resize, queue wait, NMS까지 포함한 end-to-end p95와 p99를 측정해야 한다. model forward FPS만으로 실시간 안전성을 주장할 수 없다.

### 2.3 confidence는 보정된 확률이 아니다

YOLOv1의 box confidence는 다음 의미로 학습된다.

$$
q=P(\mathrm{object})\operatorname{IoU}(b, b^{*})
$$

class $c$의 ranking score는 보통 다음처럼 만든다.

$$
s_c=q\,P(c\mid\mathrm{object})
$$

이 곱은 ranking에는 유용하지만 $0.8$이 실제 정답률 80%라는 보장은 없다. threshold는 class·camera slice별 precision-recall과 calibration을 보고 정한다.

### 2.4 NMS는 network graph 밖의 사소한 코드가 아니다

NMS의 IoU convention, class grouping, 동점 순서, pre/post top-k가 결과를 바꾼다. Python과 edge runtime이 서로 다른 NMS를 쓰면 model output이 같아도 최종 detection이 달라진다. 후처리도 release ABI와 회귀 테스트의 대상이다.

### 2.5 “한 번 본다”는 memory copy가 한 번이라는 뜻이 아니다

단일 network forward라도 decode, color conversion, CPU-to-device copy, output download가 여러 번 일어날 수 있다. zero-copy 가능 여부와 execution provider별 fallback op를 profile로 확인해야 한다.

## 3. release bundle과 manifest

### 3.1 bundle 구조

```text
yolov1-release-2026-09-05/
  model.onnx
  manifest.json
  classes.txt
  golden-input.npy
  golden-raw-output.npy
  golden-detections.json
  calibration.json
  checksums.sha256
```

위 구조는 **설명용 manifest 예시**다. 실제 배포에서는 artifact registry가 각 파일의 hash와 접근 권한을 관리해야 한다.

### 3.2 manifest 예시

```json
{
  "schema": "dense-detector-serving/v3",
  "architecture": "yolov1",
  "decoder_schema": "yolov1-cell-v1",
  "input_name": "images",
  "input_layout": "NCHW",
  "input_dtype": "float32",
  "input_color": "RGB",
  "canvas_hw": [448, 448],
  "resize_rounding": "round-half-away-from-zero",
  "padding_split": "floor-left-top",
  "padding_value": [114, 114, 114],
  "grid_size": 7,
  "predictors_per_cell": 2,
  "class_count": 20,
  "raw_layout": "NCHW",
  "channel_order": "boxes[B,tx,ty,w,h,q]-then-class[C]",
  "box_space": "canvas-normalized-cxcywh",
  "score_rule": "confidence-times-conditional-class",
  "score_threshold": 0.05,
  "pre_nms_topk": 300,
  "nms_mode": "per-class",
  "nms_iou_threshold": 0.5,
  "post_nms_topk": 100
}
```

`round`라는 단어만으로는 부족하다. 일부 언어는 정확히 `.5`인 값을 짝수로 보내고, 일부 구현은 0에서 먼 쪽으로 보낸다. 크로스런타임 구현은 rounding mode까지 맞추거나 manifest에 실제 $H_r,W_r,p_x,p_y$를 요청별로 저장해야 한다.

## 4. letterbox 좌표 계보

### 4.1 이상적 scale과 실제 scale

aspect ratio를 보존해 canvas 안에 넣는 이상적 scale은 다음과 같다.

$$
r=\min\left(\frac{W_t}{W_0},\frac{H_t}{H_0}\right)
$$

그러나 image resize API는 정수 크기를 요구한다. 정해진 반올림 함수 $R$을 사용한다.

$$
W_r=R(rW_0),\qquad H_r=R(rH_0)
$$

실제 좌표 변환에는 이상적 $r$을 재사용하지 않고 다음 scale을 쓴다.

$$
s_x=\frac{W_r}{W_0},\qquad s_y=\frac{H_r}{H_0}
$$

$s_x$와 $s_y$는 정수 반올림 때문에 미세하게 다를 수 있다. 작은 차이도 왕복 golden에서는 드러난다.

### 4.2 padding split

남는 폭과 높이를 다음처럼 나눈다.

$$
P_w=W_t-W_r,\qquad P_h=H_t-H_r
$$

$$
p_l=\left\lfloor\frac{P_w}{2}\right\rfloor,\qquad
p_t=\left\lfloor\frac{P_h}{2}\right\rfloor
$$

$$
p_r=P_w-p_l,\qquad p_b=P_h-p_t
$$

홀수 padding에서는 오른쪽 또는 아래가 한 pixel 더 크다. 양쪽이 같다고 가정하면 역변환 box가 1 pixel 밀릴 수 있다.

### 4.3 정방향과 역방향 box

원본 pixel box를 canvas pixel box로 보내면 다음과 같다.

$$
b_c=(s_xx_1+p_l,\ s_yy_1+p_t,\ s_xx_2+p_l,\ s_yy_2+p_t)
$$

역변환은 다음과 같다.

$$
b_0=\left(
\frac{x_{c1}-p_l}{s_x},
\frac{y_{c1}-p_t}{s_y},
\frac{x_{c2}-p_l}{s_x},
\frac{y_{c2}-p_t}{s_y}
\right)
$$

마지막에 원본 범위 $[0,W_0]\times[0,H_0]$로 clip한다. canvas에서 먼저 무조건 clip하면 padding 영역으로 나간 예측의 크기가 바뀌어 NMS 결과도 달라질 수 있다. 이 문서의 계약은 **decode → score filter → 원본 역변환 → 원본 clip → 퇴화 box 제거 → NMS** 순서다.

### 4.4 수작업 검산

원본이 $H_0\times W_0=300\times500$, canvas가 $448\times448$이라고 하자.

$$
r=\min(448/500,448/300)=0.896
$$

$$
(H_r,W_r)=(269,448)
$$

따라서 실제 scale과 padding은 다음과 같다.

$$
s_x=448/500=0.896,\qquad s_y=269/300
$$

$$
(p_l,p_t,p_r,p_b)=(0,89,0,90)
$$

원본 box $(50,30,450,270)$은 canvas에서 약 $(44.8,115.9,403.2,331.1)$이 되고, 위 역변환을 적용하면 원래 box로 돌아온다.

## 5. NumPy 독립 oracle

다음은 **실행 가능한 검증용 예제**다. framework decoder를 호출하지 않고 letterbox 왕복, IoU, 결정적 class별 NMS를 검사한다.

```python
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Letterbox:
    original_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    pad_ltrb: tuple[int, int, int, int]

    @property
    def scale_xy(self):
        h0, w0 = self.original_hw
        hr, wr = self.resized_hw
        return wr / w0, hr / h0


def round_half_away_nonnegative(x):
    return int(np.floor(float(x) + 0.5))


def make_letterbox(original_hw, canvas_hw):
    h0, w0 = original_hw
    ht, wt = canvas_hw
    ratio = min(wt / w0, ht / h0)
    wr = round_half_away_nonnegative(ratio * w0)
    hr = round_half_away_nonnegative(ratio * h0)
    pad_w, pad_h = wt - wr, ht - hr
    left, top = pad_w // 2, pad_h // 2
    return Letterbox(
        original_hw,
        canvas_hw,
        (hr, wr),
        (left, top, pad_w - left, pad_h - top),
    )


def to_canvas(boxes, meta):
    boxes = np.asarray(boxes, dtype=np.float64)
    sx, sy = meta.scale_xy
    left, top, _, _ = meta.pad_ltrb
    scale = np.array([sx, sy, sx, sy])
    offset = np.array([left, top, left, top])
    return boxes * scale + offset


def to_original(boxes, meta):
    boxes = np.asarray(boxes, dtype=np.float64)
    sx, sy = meta.scale_xy
    left, top, _, _ = meta.pad_ltrb
    scale = np.array([sx, sy, sx, sy])
    offset = np.array([left, top, left, top])
    return (boxes - offset) / scale


def pair_iou(a, b):
    left_top = np.maximum(a[:2], b[:2])
    right_bottom = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0.0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0.0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0.0))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def stable_nms(boxes, scores, class_ids, threshold):
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    kept = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        # 1순위 score 내림차순, 2순위 원래 candidate index 오름차순.
        order = indices[np.lexsort((indices, -scores[indices]))]
        while order.size:
            current = int(order[0])
            kept.append(current)
            order = np.array(
                [i for i in order[1:] if pair_iou(boxes[current], boxes[i]) <= threshold],
                dtype=np.int64,
            )
    return np.array(sorted(kept, key=lambda i: (-scores[i], i)), dtype=np.int64)


meta = make_letterbox((300, 500), (448, 448))
assert meta.resized_hw == (269, 448)
assert meta.pad_ltrb == (0, 89, 0, 90)

original = np.array([[50.0, 30.0, 450.0, 270.0]])
canvas = to_canvas(original, meta)
np.testing.assert_allclose(to_original(canvas, meta), original, atol=1e-12)

boxes = np.array([
    [10.0, 10.0, 30.0, 30.0],
    [11.0, 11.0, 31.0, 31.0],
    [11.0, 11.0, 31.0, 31.0],
])
scores = np.array([0.9, 0.8, 0.8])
classes = np.array([0, 0, 1])
np.testing.assert_array_equal(stable_nms(boxes, scores, classes, 0.5), [0, 2])
print("numpy golden passed:", meta, canvas[0].round(6).tolist())
```

동일 score의 마지막 두 box는 class가 다르므로 둘 다 경쟁하지 않는다. class 0의 첫 두 box는 크게 겹쳐 index 0만 남는다.

## 6. YOLOv1 raw tensor의 실행 계약

### 6.1 shape와 channel slice

convolution head의 raw output을 다음과 같이 둔다.

$$
R\in\mathbb{R}^{N\times(5B+C)\times S\times S}
$$

논리적 grid view는 다음과 같다.

$$
G=\operatorname{permute}(R,0,2,3,1)
\in\mathbb{R}^{N\times S\times S\times(5B+C)}
$$

이 문서의 channel 순서는 predictor별 `[tx,ty,w,h,q]`를 $B$번 둔 뒤 cell class $C$개를 둔다. 다른 저장소는 모든 좌표, 모든 confidence, class 순으로 묶을 수 있으므로 shape만 보고 추론하지 않는다.

### 6.2 작은 batch의 shape 추적

$N=2$, $S=7$, $B=2$, $C=3$인 예를 보자.

| 단계 | shape | layout·dtype | 의미 |
| --- | --- | --- | --- |
| 원본 이미지 | ragged | encoded bytes | 서로 다른 $H_0,W_0$ |
| letterbox canvas | `[2,3,448,448]` | `NCHW float32` | RGB, 정규화 완료 |
| backbone feature | `[2,256,7,7]` | `NCHW float16` | 예시 feature |
| raw head | `[2,13,7,7]` | `NCHW float16` | $13=5\cdot2+3$ |
| logical grid | `[2,7,7,13]` | `NHWC float32` | decoder 누산은 FP32 |
| box fields | `[2,7,7,2,5]` | `float32` | box와 confidence |
| class fields | `[2,7,7,3]` | `float32` | cell당 한 벌 |
| dense class candidate | `[2,7,7,2,3]` | `float32` | confidence와 class의 곱 |

`permute` 없이 `reshape`만 하면 element 수는 맞아도 channel과 spatial 위치가 섞인다. 이 오류는 shape assertion으로 잡히지 않으므로 impulse fixture가 필요하다.

### 6.3 cell에서 canvas box로

cell row $i$, column $j$의 offset을 $(t_x,t_y)$라 하자.

$$
c_x=\frac{j+t_x}{S},\qquad c_y=\frac{i+t_y}{S}
$$

$w,h$는 canvas 전체에 대한 정규화 크기로 해석한다.

$$
b_{xyxy}=\left(c_x-\frac{w}{2},\ c_y-\frac{h}{2},\
c_x+\frac{w}{2},\ c_y+\frac{h}{2}\right)
$$

이 값을 $(W_t,H_t,W_t,H_t)$와 곱해 canvas pixel 좌표로 만든 뒤 letterbox를 역변환한다. $t_x,t_y,w,h$에 어떤 activation을 쓸지는 checkpoint와 학습 구현에 종속된다. 임의로 sigmoid를 추가하면 기존 checkpoint를 깨뜨린다.

## 7. 실행 가능한 PyTorch serving core

다음 코드는 **실행 가능한 축소 검증 예제**다. 원 논문의 전체 backbone을 재현하는 목적이 아니라 layout, decode, finite backward, 결정적 후보 순서를 검증한다.

```python
import torch
from torch import nn


class TinyYoloV1Head(nn.Module):
    def __init__(self, channels=8, grid=2, boxes=2, classes=3):
        super().__init__()
        self.grid = grid
        self.boxes = boxes
        self.classes = classes
        self.projection = nn.Conv2d(channels, 5 * boxes + classes, 1)

    def forward(self, feature):
        if feature.shape[-2:] != (self.grid, self.grid):
            raise ValueError("feature grid does not match manifest")
        return self.projection(feature)


def decode_yolov1(raw, boxes, classes):
    if raw.ndim != 4 or raw.shape[1] != 5 * boxes + classes:
        raise ValueError("raw output violates channel contract")
    grid_h, grid_w = raw.shape[-2:]
    if grid_h != grid_w:
        raise ValueError("this YOLOv1 contract requires a square grid")

    logical = raw.permute(0, 2, 3, 1).float().contiguous()
    box_raw = logical[..., : 5 * boxes].reshape(
        raw.shape[0], grid_h, grid_w, boxes, 5
    )
    class_raw = logical[..., 5 * boxes :]

    offsets = box_raw[..., :2].sigmoid()
    sizes = box_raw[..., 2:4].sigmoid()
    confidence = box_raw[..., 4].sigmoid()
    class_probability = class_raw.softmax(dim=-1)

    rows, columns = torch.meshgrid(
        torch.arange(grid_h, device=raw.device),
        torch.arange(grid_w, device=raw.device),
        indexing="ij",
    )
    cell = torch.stack((columns, rows), dim=-1).view(1, grid_h, grid_w, 1, 2)
    center = (cell + offsets) / grid_h
    xy_min = center - sizes / 2.0
    xy_max = center + sizes / 2.0
    boxes_xyxy = torch.cat((xy_min, xy_max), dim=-1)
    scores = confidence.unsqueeze(-1) * class_probability.unsqueeze(-2)
    return boxes_xyxy, scores


torch.manual_seed(20260905)
head = TinyYoloV1Head()
feature = torch.randn(2, 8, 2, 2, requires_grad=True)
raw = head(feature)
decoded_boxes, scores = decode_yolov1(raw, boxes=2, classes=3)

assert raw.shape == (2, 13, 2, 2)
assert decoded_boxes.shape == (2, 2, 2, 2, 4)
assert scores.shape == (2, 2, 2, 2, 3)
assert torch.isfinite(decoded_boxes).all()
assert torch.isfinite(scores).all()

loss = decoded_boxes.square().mean() + scores.square().mean()
loss.backward()
assert torch.isfinite(feature.grad).all()

# 같은 score이면 flatten index가 작은 후보를 먼저 둔다.
flat_scores = torch.tensor([0.8, 0.9, 0.9, 0.7])
indices = torch.arange(flat_scores.numel())
order = sorted(indices.tolist(), key=lambda i: (-float(flat_scores[i]), i))
assert order == [1, 2, 0, 3]
print("pytorch serving core passed:", tuple(raw.shape), float(loss.detach()))
```

이 예제는 수치 범위를 단순화하려고 모든 box field에 sigmoid를 사용한다. 이는 **교육용 serving fixture의 ABI**이며 원 논문 checkpoint의 parameterization을 재현한다는 뜻이 아니다.

## 8. 후보 생성과 결정적 NMS

### 8.1 연산 순서

후처리의 권장 순서를 명시적으로 고정한다.

1. raw tensor layout과 channel 수 검증
2. FP32에서 activation과 box decode
3. confidence와 class score 결합
4. `NaN`·`inf` 후보 제거 및 counter 증가
5. score threshold 적용
6. stable pre-NMS top-k
7. canvas에서 원본 좌표로 역변환
8. 원본 경계 clip과 퇴화 box 제거
9. class별 stable NMS
10. stable post-NMS top-k
11. 요청별 원래 순서로 batch 해제

### 8.2 stable ordering

candidate $i$의 정렬 key를 다음처럼 둔다.

$$
k_i=(-s_i,\ c_i,\ i)
$$

score 내림차순, class id 오름차순, 원래 flat index 오름차순이다. FP16 score를 그대로 비교하면 runtime별 반올림으로 동점 집합이 달라질 수 있으므로 decoder score를 FP32로 만들고 허용 오차를 golden에 기록한다.

### 8.3 class별 NMS

같은 class의 남은 후보 집합을 $A$라 하자. 가장 높은 우선순위 후보 $m$을 선택한 뒤 다음 집합으로 갱신한다.

$$
A\leftarrow\{i\in A\setminus\{m\}:\operatorname{IoU}(b_m,b_i)\le\tau\}
$$

서로 다른 class의 box는 이 계약에서 억제하지 않는다. class-agnostic NMS를 쓰고 싶다면 별도 decoder revision으로 배포하고 정확도 차이를 canary한다.

## 9. C++17 크로스런타임 golden

다음은 **실행 가능한 C++17 예제**다. 외부 library 없이 수작업 예제의 letterbox 역변환을 검증한다.

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

struct Letterbox {
    int original_h;
    int original_w;
    int resized_h;
    int resized_w;
    int pad_left;
    int pad_top;
};

std::array<double, 4> ToOriginal(
    const std::array<double, 4>& box,
    const Letterbox& meta) {
    const double sx = static_cast<double>(meta.resized_w) / meta.original_w;
    const double sy = static_cast<double>(meta.resized_h) / meta.original_h;
    return {
        (box[0] - meta.pad_left) / sx,
        (box[1] - meta.pad_top) / sy,
        (box[2] - meta.pad_left) / sx,
        (box[3] - meta.pad_top) / sy,
    };
}

int main() {
    const Letterbox meta{300, 500, 269, 448, 0, 89};
    const double sx = 448.0 / 500.0;
    const double sy = 269.0 / 300.0;
    const std::array<double, 4> canvas{
        50.0 * sx,
        30.0 * sy + 89.0,
        450.0 * sx,
        270.0 * sy + 89.0,
    };
    const auto original = ToOriginal(canvas, meta);
    const std::array<double, 4> expected{50.0, 30.0, 450.0, 270.0};
    for (std::size_t i = 0; i < original.size(); ++i) {
        assert(std::abs(original[i] - expected[i]) < 1e-12);
    }
    std::cout << std::fixed << std::setprecision(6)
              << original[0] << " " << original[1] << " "
              << original[2] << " " << original[3] << "\n";
}
```

## 10. C# 크로스런타임 golden

다음은 **실행 가능한 C# 예제**다. C++ 예제와 같은 값과 tolerance를 사용한다.

```csharp
using System;

public static class Program
{
    private static double[] ToOriginal(
        double[] box,
        int originalH,
        int originalW,
        int resizedH,
        int resizedW,
        int padLeft,
        int padTop)
    {
        double sx = (double)resizedW / originalW;
        double sy = (double)resizedH / originalH;
        return new double[] {
            (box[0] - padLeft) / sx,
            (box[1] - padTop) / sy,
            (box[2] - padLeft) / sx,
            (box[3] - padTop) / sy
        };
    }

    public static void Main()
    {
        double sx = 448.0 / 500.0;
        double sy = 269.0 / 300.0;
        double[] canvas = {
            50.0 * sx,
            30.0 * sy + 89.0,
            450.0 * sx,
            270.0 * sy + 89.0
        };
        double[] actual = ToOriginal(canvas, 300, 500, 269, 448, 0, 89);
        double[] expected = { 50.0, 30.0, 450.0, 270.0 };
        for (int i = 0; i < actual.Length; ++i)
        {
            if (Math.Abs(actual[i] - expected[i]) >= 1e-12)
            {
                throw new Exception("letterbox golden mismatch");
            }
        }
        Console.WriteLine(
            string.Format(
                System.Globalization.CultureInfo.InvariantCulture,
                "{0:F6} {1:F6} {2:F6} {3:F6}",
                actual[0], actual[1], actual[2], actual[3]));
    }
}
```

`CultureInfo.InvariantCulture`를 쓰지 않으면 locale에 따라 소수점이 쉼표로 출력되어 golden parser가 실패할 수 있다. 이 문제는 model math가 아니라 운영 I/O 계약의 문제다.

## 11. framework 간 layout·dtype 대응

| 경계 | PyTorch Python | ONNX | C++ runtime | C# runtime |
| --- | --- | --- | --- | --- |
| image | `Tensor[N,3,H,W]` | `float[N,3,H,W]` | contiguous `float*` | `DenseTensor<float>` |
| raw head | `NCHW` | manifest 고정 | dimension 검사 후 span | dimension 검사 후 tensor |
| logical grid | `permute(...).contiguous()` | graph 내부 또는 host | explicit index 계산 | explicit index 계산 |
| score math | FP32 | FP32 node 권장 | `float`, golden은 `double` oracle | `float`, golden은 `double` oracle |
| class id | `torch.int64` | `int64` | `int64_t` | `long` |
| count·offset | `torch.int64` | `int64` | `std::int64_t` | `long` |
| final box | `float32 xyxy` | `[R,4]` | `float[4]` | `float[4]` |

### 11.1 `NHWC` execution provider

일부 accelerator는 내부적으로 `NHWC`를 선호한다. graph optimizer가 transpose를 흡수하는 것은 허용할 수 있지만 외부 output ABI까지 자동으로 바뀌었다고 가정하면 안 된다. session load 직후 output rank와 각 dimension을 manifest와 비교한다.

### 11.2 FP16 경계

backbone과 head는 FP16이어도 다음은 FP32로 올리는 편이 안전하다.

- sigmoid와 softmax의 최종 score
- 작은 box의 corner 변환
- IoU의 교집합·합집합
- NMS threshold 비교

특히 거의 같은 box에서 FP16 union이 거칠게 반올림되면 IoU가 threshold 양쪽으로 흔들릴 수 있다.

## 12. 비용과 메모리 예산

### 12.1 raw output 크기

raw element 수는 다음과 같다.

$$
E_{raw}=NS^2(5B+C)
$$

dtype가 $d$ byte라면 activation byte는 다음과 같다.

$$
M_{raw}=dNS^2(5B+C)
$$

YOLOv1의 $S=7$, $B=2$, $C=20$, FP32, $N=32$라면 약 188 KiB로 작다. 그러나 후대 다중 scale detector에서는 모든 location·anchor·class score를 펼치는 순간 후보 tensor가 훨씬 커진다. Part 2에서 이 비용을 별도로 계산한다.

### 12.2 NMS 비용

단순 NMS의 최악 비교 횟수는 후보 수 $M$에 대해 $O(M^2)$이다. score threshold 뒤 stable top-k로 $K$개만 남기면 비교 상계는 대략 다음과 같다.

$$
\frac{K(K-1)}{2}
$$

$K=300$이면 class 하나의 최악 pair가 44,850개다. threshold를 지나치게 낮추거나 class마다 $K$를 독립 적용하면 tail latency가 커진다. 다음 값을 요청마다 기록한다.

- raw candidate 수
- threshold 통과 수
- pre-NMS top-k에서 잘린 수
- NMS 비교 수 또는 kernel 시간
- post-NMS detection 수

### 12.3 dynamic batch

서로 다른 원본 크기도 같은 canvas로 letterbox하면 tensor batch는 쉽다. 하지만 decode와 NMS 비용은 각 이미지의 candidate 분포에 따라 ragged하다. forward batch가 빨라도 한 이미지의 후보 폭증이 batch 전체 응답을 붙잡을 수 있다.

운영 정책은 다음 중 하나를 선택한다.

- 이미지별 pre-NMS $K$를 독립 적용한다.
- NMS를 batch item별 병렬 실행한다.
- deadline을 넘긴 item만 격리하고 나머지를 반환한다.
- admission control에서 최대 class 수와 candidate 수를 제한한다.

## 13. 테스트와 디버깅

### 13.1 단위 테스트 행렬

| 테스트 | 입력 | 기대 결과 |
| --- | --- | --- |
| landscape letterbox | `300 x 500` | 위아래 padding `89,90` |
| portrait letterbox | `500 x 300` | 좌우 padding 비대칭 가능 |
| odd resize | `.5` 경계 크기 | manifest rounding과 정확히 일치 |
| coordinate round trip | 원본 box 4개 | 최대 오차 tolerance 이하 |
| impulse channel | raw 한 원소만 큰 값 | 정확한 cell·field만 변함 |
| empty candidate | 모든 score가 threshold 아래 | 빈 detection, 정상 status |
| degenerate box | $x_2\le x_1$ | NMS 전 제거·counter 증가 |
| equal score | 같은 class·같은 score | flat index 순서로 결정 |
| cross-class overlap | 같은 box·다른 class | class별 NMS에서 둘 다 유지 |
| NaN score | 후보 하나가 `NaN` | 제거·metric 증가, 전체 오염 없음 |
| max candidate | 정확히 $K$개 | buffer overflow 없이 완료 |

### 13.2 통합 golden

golden fixture는 세 단계로 나눈다.

1. 전처리된 input tensor checksum
2. NMS 전 raw output과 decoded candidate
3. NMS 후 최종 detection

최종 detection만 비교하면 오차가 model에서 시작했는지 decoder에서 시작했는지 알기 어렵다. 반대로 raw output만 비교하면 후처리 회귀를 놓친다.

### 13.3 비교 tolerance

모든 값에 하나의 tolerance를 쓰지 않는다.

| 값 | 비교 방법 |
| --- | --- |
| integer shape·class·index | exact |
| resize 크기·padding | exact |
| FP32 raw output | provider별 `atol`, `rtol` |
| decoded box | pixel absolute error |
| score | absolute·relative error |
| NMS keep set | stable key 기준 exact |

NMS threshold에 매우 가까운 fixture는 두 종류로 관리한다. 일반 parity fixture에서는 경계에서 충분히 떨어진 값을 쓰고, 별도 boundary fixture에서 provider 차이를 의도적으로 측정한다.

## 14. 자주 만나는 장애와 진단 순서

### 14.1 모든 box가 세로로 일정하게 밀린다

가능성이 높은 원인은 top padding 누락 또는 홀수 padding split 불일치다.

1. 요청 metadata의 `resized_hw`와 `pad_ltrb`를 확인한다.
2. canvas에 원본 정답 box를 정방향 투영해 overlay한다.
3. model과 무관한 synthetic box를 역변환한다.
4. $r$과 실제 $s_y=H_r/H_0$ 중 무엇을 사용했는지 확인한다.

### 14.2 confidence는 비슷한데 class가 바뀐다

cell class channel 시작 offset이 $5B$인지 확인한다. box마다 class가 있다고 잘못 reshape하면 spatial pattern은 그럴듯해도 class가 뒤섞인다. class map hash와 background class 유무도 확인한다.

### 14.3 CPU와 GPU에서 NMS 결과가 가끔 한 개 다르다

동점 score, IoU threshold 근처, FP16 계산을 먼저 의심한다. NMS 입력을 FP32로 고정하고 stable secondary key를 적용한다. 그래도 경계 후보가 다르면 score와 IoU margin을 telemetry에 남긴다.

### 14.4 p99만 갑자기 증가한다

forward latency와 NMS latency를 분리한다. candidate count 분포, 특정 class 폭증, 입력 decode 실패 재시도, provider CPU fallback, dynamic batch queue wait를 함께 본다.

### 14.5 빈 장면에서 수백 개 box가 나온다

padding 색상 변화, normalization 누락, wrong color order, score activation 중복, threshold config drift를 확인한다. 빈 장면 false-positive rate를 독립 SLO로 둔다.

## 15. 수치 안정성

### 15.1 softmax

class logit $z_c$의 softmax는 최대값을 빼고 계산한다.

$$
p_c=\frac{\exp(z_c-m)}{\sum_k\exp(z_k-m)},\qquad m=\max_k z_k
$$

export graph 안에 softmax가 이미 있다면 host에서 다시 적용하지 않는다. double activation은 score ranking을 바꾼다.

### 15.2 IoU

교집합과 합집합은 다음과 같다.

$$
w_I=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B))
$$

$$
h_I=\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B))
$$

$$
\operatorname{IoU}(A,B)=
\begin{cases}
\dfrac{w_Ih_I}{|A|+|B|-w_Ih_I},& |A|+|B|-w_Ih_I>0\\
0,&\text{otherwise}
\end{cases}
$$

퇴화 box를 먼저 제거하면 분모 epsilon에 의존하는 정도를 줄일 수 있다. `NaN`을 0으로 조용히 바꾸지 말고 제거 수를 기록한다.

### 15.3 좌표 clip

clip 전후 면적 비를 기록한다.

$$
\rho_{clip}=\frac{|b_{after}|}{\max(|b_{before}|,\varepsilon)}
$$

$\rho_{clip}$이 작은 box가 급증하면 camera framing 변화일 수도 있고 decoder 좌표계 장애일 수도 있다.

## 16. ONNX와 배포 release gate

### 16.1 graph 경계 선택

두 가지 배포 방식을 비교한다.

| 방식 | 장점 | 위험 |
| --- | --- | --- |
| graph는 raw head까지만 | host decoder 교체·디버깅 쉬움 | 언어별 decoder drift |
| decode·NMS까지 graph 포함 | 단일 artifact 의미가 강함 | provider별 NMS 지원·동점 차이 |

어느 쪽이든 schema revision을 바꾸지 않고 경계를 이동하면 안 된다. 이 문서는 raw head를 graph output으로 하고 공용 decoder library를 versioning하는 방식을 택한다.

### 16.2 필수 release gate

1. artifact hash와 manifest signature 검증
2. input/output name, rank, dimension, dtype 검사
3. CPU reference raw-output parity
4. target provider raw-output parity
5. Python/C++/C# letterbox·decode golden
6. NMS keep set과 stable ordering exact 비교
7. representative set의 mAP·class별 recall 회귀
8. end-to-end p50·p95·p99와 peak memory 측정
9. empty·crowded·portrait·panorama slice 검증
10. canary와 자동 rollback rule 확인

현재 로컬 검증에서 `onnx` 또는 `onnxruntime`이 없다면 실제 export·provider parity는 **미검증**으로 남긴다. PyTorch 결과만으로 ONNX target runtime 동등성을 주장하지 않는다.

### 16.3 rollout

- shadow 단계에서 구·신 decoder의 raw input을 동일하게 복제한다.
- detection count, matched-box IoU, class disagreement, score delta를 비교한다.
- 작은 traffic canary에서 latency와 false-positive slice를 확인한다.
- budget을 넘으면 model과 decoder bundle을 함께 이전 revision으로 되돌린다.

## 17. 모니터링과 SLO

### 17.1 latency 분해

$$
T_{e2e}=T_{decode}+T_{resize}+T_{queue}+T_{copy}+T_{model}+T_{post}+T_{serialize}
$$

각 항의 p50·p95·p99를 따로 기록한다. GPU model time만 짧고 queue나 NMS가 긴 경우를 분리해야 한다.

### 17.2 데이터와 출력 metric

- 원본 aspect ratio histogram
- 실제 resize scale과 비대칭 padding 비율
- pixel mean·standard deviation·saturation
- raw confidence quantile
- class별 threshold 통과 후보 수
- NMS 전후 후보 수와 억제율
- clip 면적 비와 퇴화 box 비율
- 이미지당 detection 수
- empty-scene false-positive rate
- label 지연 구간의 class별 precision·recall·ECE

### 17.3 alert 예시

| 신호 | 경고 조건 예시 | 첫 조사 대상 |
| --- | --- | --- |
| `invalid_box_rate` | baseline의 5배 | decoder·NaN·clip 순서 |
| `candidate_p99` | pre-NMS $K$의 95% 초과 | threshold·입력 drift |
| `postprocess_p99` | latency budget 초과 | NMS·CPU fallback |
| `vertical_shift_px` | golden 0.25 px 초과 | top padding·실제 scale |
| `empty_fp_rate` | canary budget 초과 | padding 색·normalization |
| `class_disagreement` | shadow baseline 초과 | channel order·class map |

고정 숫자는 서비스마다 calibration한다. alert는 단일 요청이 아니라 충분한 sample과 지속 시간을 조건으로 둔다.

## 18. 실무 실패 사례: model은 같았지만 decoder가 달랐다

### 상황

Python reference와 edge C# 서비스가 같은 ONNX hash를 사용했다. raw tensor checksum도 tolerance 안이었다. 그런데 세로로 긴 camera frame에서 C# box가 아래쪽으로 약 1 pixel 이동했고 NMS 후 detection 수도 가끔 달랐다.

### 원인

- Python은 실제 $H_r/H_0$을 역변환 scale로 사용했다.
- C#은 반올림 전 이상적 단일 ratio $r$을 재사용했다.
- 홀수 vertical padding을 Python은 `(top,bottom)=(89,90)`, C#은 `(90,89)`로 나눴다.
- 동점 score의 NMS가 안정 정렬을 보장하지 않았다.

### 왜 늦게 발견했는가

square image fixture만 있었고 raw output만 비교했다. coordinate round trip과 NMS keep-set golden이 없었다.

### 수정

1. 요청 metadata에 실제 `resized_hw`와 `pad_ltrb`를 저장했다.
2. 모든 runtime이 실제 $s_x,s_y$로 역변환하도록 했다.
3. landscape·portrait·홀수 padding fixture를 추가했다.
4. score 다음에 flat index를 쓰는 stable ordering을 추가했다.
5. raw, decoded, final detection의 3단계 golden을 release gate로 승격했다.

### 재발 방지

model hash뿐 아니라 decoder schema와 preprocessor hash를 deployment identity에 포함했다. box shift와 NMS disagreement를 shadow metric으로 상시 관측했다.

## 19. 운영 체크리스트

### 데이터·전처리

- [ ] RGB/BGR, normalization, padding 값이 manifest와 같다.
- [ ] resize 반올림과 padding split 규칙이 모든 언어에서 같다.
- [ ] 요청마다 원본·resize·canvas 크기와 padding을 보존한다.
- [ ] EXIF orientation을 적용한 뒤의 크기를 좌표 기준으로 쓴다.

### 모델·decoder

- [ ] YOLOv1과 후대 YOLO decoder schema를 구분한다.
- [ ] raw layout, channel order, activation 소유자를 명시한다.
- [ ] class map hash와 class 수가 output channel과 맞는다.
- [ ] FP32 score·IoU와 stable tie-break를 사용한다.
- [ ] class별 또는 class-agnostic NMS 선택을 versioning한다.

### 성능·안정성

- [ ] image별 pre/post top-k와 최대 detection을 제한한다.
- [ ] crowded scene과 empty scene의 p99를 따로 잰다.
- [ ] provider CPU fallback과 host-device copy를 profile한다.
- [ ] `NaN`, 퇴화 box, clip 비율을 metric으로 남긴다.

### 릴리스·운영

- [ ] model·manifest·decoder·class map을 하나의 bundle로 배포한다.
- [ ] raw·decoded·final 3단계 golden을 통과한다.
- [ ] target C++·C# runtime parity를 실제 provider에서 확인한다.
- [ ] shadow, canary, rollback threshold가 준비되어 있다.
- [ ] model과 decoder를 함께 되돌릴 수 있다.

## 20. 연습문제

### 문제 1

원본이 $720\times1280$, canvas가 $640\times640$이다. 이상적 ratio, 실제 resize 크기, padding을 구하라. 반올림은 nonnegative half-away-from-zero, 남는 홀수 pixel은 오른쪽·아래에 더 둔다.

### 문제 2

$N=4$, $S=7$, $B=2$, $C=20$인 FP32 raw output의 element 수와 byte를 구하라.

### 문제 3

같은 좌표와 같은 score를 가진 class 2 box와 class 7 box가 있다. class별 NMS와 class-agnostic NMS는 각각 어떻게 처리할 수 있는가?

### 문제 4

Python과 C++의 raw output은 일치하지만 최종 detection이 다르다. 최소한 어느 세 중간 결과를 비교해야 하는가?

### 문제 5

왜 canvas에서 box를 먼저 clip한 뒤 원본으로 역변환하는 방식이 이 문서의 계약과 다를 수 있는가?

## 21. 해답

### 해답 1

$$
r=\min(640/1280,640/720)=0.5
$$

resize는 $360\times640$이고 horizontal padding은 없다. vertical padding은 총 280 pixel이므로 top과 bottom이 각각 140 pixel이다.

### 해답 2

$$
E=4\cdot7^2\cdot(5\cdot2+20)=5{,}880
$$

FP32는 element당 4 byte이므로 23,520 byte다.

### 해답 3

class별 NMS에서는 서로 경쟁하지 않아 둘 다 남을 수 있다. class-agnostic NMS에서는 stable priority가 높은 하나가 다른 하나를 억제할 수 있다. 어느 의미가 맞는지는 decoder revision에 포함해야 한다.

### 해답 4

전처리 input checksum, NMS 전 decoded candidate, NMS 후 keep set과 최종 detection을 비교한다. raw output이 같다면 좌표 역변환, score filter, ordering, NMS convention을 좁혀 볼 수 있다.

### 해답 5

canvas clip은 padding 영역으로 나간 정도를 먼저 잘라낸다. 그 결과 box 크기와 IoU가 바뀔 수 있다. 이 문서는 원본 역변환 후 원본 경계에서 clip하므로 NMS 입력과 결과가 달라질 수 있다.

## 22. 핵심 요약

1. YOLOv1 decoder는 최신 YOLO 계열과 교환할 수 있는 일반 규격이 아니다.
2. letterbox는 이상적 ratio가 아니라 실제 정수 resize의 $s_x,s_y$와 비대칭 padding으로 역변환한다.
3. raw layout, channel order, activation, score, NMS를 model과 함께 versioning한다.
4. `permute`와 `reshape` 혼동은 shape 검사만으로 잡히지 않아 impulse golden이 필요하다.
5. FP32 score·IoU와 stable secondary key가 크로스런타임 NMS 재현성을 높인다.
6. candidate top-k는 정확도 설정이면서 tail latency와 memory를 지키는 admission control이다.
7. raw output, decoded candidate, final detection을 나누어 비교해야 장애 지점을 찾을 수 있다.
8. model, preprocessor, decoder, class map, golden fixture를 하나의 immutable release bundle로 배포한다.

## 23. 다음 학습 예고

다음 Part 2/3에서는 같은 원본의 SSD와 FPN 구간으로 이동한다. 여러 feature level의 stride·anchor schema, level별 tensor layout, feature pyramid memory, small-object routing, C++·C# decoder parity, ONNX multi-output graph, level별 candidate 폭증과 latency 격리를 실무 운영 계약으로 만든다.
