<!-- curriculum: cycle=3; level=production-engineering; source_index=5/18; source=02-04.Fast_R_CNN.md; part=1/1 -->

# Fast R-CNN 운영: proposal ABI부터 크로스런타임 검출까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-04 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 5/18 · `02-04.Fast_R_CNN.md` |
| Part | 1/1 |
| 이전 소스 | `02-03.SPP.md` |
| 다음 소스 | `02-05.YOLO.md` |

## 오늘의 운영 질문

Fast R-CNN은 한 이미지의 convolution feature를 여러 proposal이 공유한다. 이 아이디어를 서비스로 옮기면 모델 정확도 외에 더 까다로운 질문이 생긴다.

- resize와 letterbox 뒤 proposal 좌표를 누가 변환하는가?
- 이미지마다 다른 proposal 수를 batch로 묶되 요청 경계를 어떻게 보존하는가?
- `RoIPool`과 `RoIAlign`의 좌표 의미를 Python, C++, C#에서 어떻게 고정하는가?
- proposal 폭증이 GPU 메모리와 tail latency를 무너뜨리지 않게 어떻게 제한하는가?
- backbone과 head를 분리 배포할 때 오래된 feature와 새 head의 혼합을 어떻게 막는가?
- 출력 shape는 정상인데 box가 한쪽으로 밀리는 장애를 어떤 telemetry로 잡는가?

1회차는 RoI Pooling의 직관과 수학을, 2회차는 target assignment·두 손실·gradient routing을 구현했다. 오늘은 이를 반복하지 않고 Fast R-CNN을 **버전이 있는 proposal ABI, ragged batching, 크로스런타임 release bundle, 관측 가능한 검출 서비스**로 확장한다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

- 원본의 Fast R-CNN과 Faster R-CNN 내용을 운영 책임 경계로 구분한다.
- 원본·리사이즈·패딩·feature 좌표계를 affine transform으로 추적한다.
- proposal을 검증하고 이미지별 개수를 보존하는 ragged batch를 만든다.
- `RoIAlign` sampling과 `aligned` 옵션을 release ABI에 고정한다.
- tensor shape, activation byte, proposal별 FLOP와 latency를 계산한다.
- NumPy oracle과 실행 가능한 PyTorch serving core를 교차 검증한다.
- C++17·C#에서 동일한 좌표 변환과 clipping golden을 재현한다.
- ONNX 분할 방식, canary, 모니터링, rollback 절차를 설계한다.

## 선수 지식

- `xyxy` box와 반열린 구간
- affine transform, bilinear interpolation, softmax
- `NCHW`·`NHWC`, `float32`·`float16`
- p50·p95·p99 latency와 dynamic batching
- PyTorch inference mode와 `torchvision.ops.roi_align`
- ONNX graph, execution provider, model manifest

## 1. 직관: proposal은 좌표 네 개가 아니라 조회 요청이다

Fast R-CNN의 backbone feature를 큰 지도라고 하자. proposal은 지도에서 읽을 창문을 지정하는 조회 요청이다. 데이터베이스 조회에 schema와 tenant ID가 필요하듯 proposal에도 다음 정보가 필요하다.

1. 어느 이미지의 box인가?
2. 어느 좌표계의 숫자인가?
3. 끝점은 포함인가, 제외인가?
4. resize·crop·padding 중 어떤 변환을 이미 거쳤는가?
5. 유효하지 않은 box는 거부하는가, 고치는가?
6. 결과를 어느 요청에 되돌려 줄 것인가?

`[17, 23, 91, 140]`만 저장하면 이 질문에 답할 수 없다. 같은 네 수가 원본 픽셀, letterbox canvas, 정규화 좌표, feature 좌표에서 전혀 다른 영역을 뜻하기 때문이다. 운영 시스템에서 proposal은 tensor 한 행이 아니라 **좌표 의미와 계보를 포함한 ABI record**다.

## 2. 원본에서 유지할 것과 교정할 것

원본 `02-04.Fast_R_CNN.md`에서 유지할 핵심은 다음과 같다.

- 이미지 전체의 convolution feature를 한 번 계산해 proposal들이 공유한다.
- 제각각인 region을 고정 공간 크기로 바꾸어 FC head에 연결한다.
- class score와 box refinement를 한 모델에서 함께 예측한다.
- anchor와 RPN은 proposal 생성 병목을 줄이는 Faster R-CNN의 핵심 확장이다.

운영 문서로 읽을 때는 다음을 교정해야 한다.

### 2.1 파일 후반의 RPN은 Fast R-CNN 자체가 아니다

Fast R-CNN은 외부 proposal을 입력받는다. RPN, anchor 생성, RPN NMS까지 포함하면 Faster R-CNN 계열이다. 두 시스템은 장애 책임도 다르다.

| 증상 | Fast R-CNN 경계에서 볼 것 | Faster R-CNN 추가 경계 |
| --- | --- | --- |
| 정답을 전혀 덮지 못함 | 입력 proposal recall | anchor·RPN score·RPN NMS |
| class는 맞고 box만 나쁨 | head delta·좌표 decode | proposal 품질도 함께 점검 |
| latency 폭증 | proposal 수·RoI op·head | RPN pre/post-NMS 수도 점검 |

### 2.2 RoI 양자화 공식은 구현마다 하나가 아니다

원본은 단순한 `floor`·`ceil` 식으로 RoI Pooling을 설명한다. 실제 연산은 `RoIPool`인지 `RoIAlign`인지, 좌표 scale과 끝점, `aligned`, `sampling_ratio`에 따라 달라진다. 비슷한 출력 shape는 동등성을 보장하지 않는다.

### 2.3 Fast R-CNN은 proposal 생성까지 end-to-end가 아니다

backbone부터 class·box head까지 gradient가 흐르는 것은 맞다. 그러나 Selective Search 같은 외부 proposal 생성기는 검출 loss로 학습되지 않는다. 원본의 “모든 과정 통합”은 검출 network 내부에 한정해 읽어야 한다.

### 2.4 “연산량 90% 절약”은 서비스 SLA가 아니다

feature 공유는 proposal별 backbone 재실행을 없앤다. 하지만 decode, resize, proposal I/O, RoI op, FC head, NMS, queue wait가 남는다. 실제 개선 폭은 proposal 분포와 hardware에 따라 측정해야 한다.

### 2.5 `7 x 7`은 의미가 아니라 설정값이다

고정 출력은 head 입력을 고정하지만 작은 box의 정보를 복원하지 않는다. 출력 grid를 키우면 RoI activation과 head 비용도 커진다. 정확도·latency·메모리 ablation으로 선택해야 한다.

## 3. proposal ABI를 먼저 정의하기

### 3.1 요청 record

서비스 경계에서는 각 이미지에 다음 metadata를 붙인다.

```json
{
  "request_id": "req-7f2",
  "image_id": "cam17-000184",
  "original_hw": [480, 640],
  "canvas_hw": [640, 640],
  "resize_scale_xy": [1.0, 1.0],
  "pad_xy": [0.0, 80.0],
  "box_format": "xyxy-half-open",
  "box_space": "original-pixel",
  "proposal_revision": "selective-search/v4",
  "proposal_count": 317
}
```

`request_id`는 batch를 풀 때 필요하고 `image_id`는 lineage와 재현에 필요하다. 두 값은 역할이 다르다.

### 3.2 box 불변식

원본 크기가 $H_0\times W_0$이고 box를 $b=(x_1,y_1,x_2,y_2)$로 두자. 반열린 좌표계의 유효 조건은 다음과 같다.

$$
0\le x_1<x_2\le W_0
$$

$$
0\le y_1<y_2\le H_0
$$

`NaN`, `inf`, 음의 면적, 범위를 크게 벗어난 좌표는 조용히 clipping하지 않고 거부한다. 경계에서 생긴 작은 부동소수점 오차만 명시된 tolerance 안에서 clipping한다.

### 3.3 affine 좌표 변환

원본에서 canvas로의 resize scale을 $(s_x,s_y)$, padding을 $(p_x,p_y)$라 하자.

$$
\begin{bmatrix}
x_c\\
y_c\\
1
\end{bmatrix}
=
\begin{bmatrix}
s_x&0&p_x\\
0&s_y&p_y\\
0&0&1
\end{bmatrix}
\begin{bmatrix}
x_0\\
y_0\\
1
\end{bmatrix}
$$

따라서 box의 두 corner는 다음처럼 이동한다.

$$
b_c=(s_xx_1+p_x,\ s_yy_1+p_y,\ s_xx_2+p_x,\ s_yy_2+p_y)
$$

backbone의 유효 stride가 $(S_x,S_y)$이고 feature 좌표를 직접 만든다면 다음과 같다.

$$
b_f=\left(\frac{x_{c1}}{S_x},\frac{y_{c1}}{S_y},
\frac{x_{c2}}{S_x},\frac{y_{c2}}{S_y}\right)
$$

그러나 `torchvision.ops.roi_align`에 canvas 좌표를 넣고 `spatial_scale=1/S`를 전달한다면 애플리케이션이 다시 나누면 안 된다. **좌표 변환과 `spatial_scale` 중 하나만 scale을 소유**해야 한다.

### 3.4 manifest

```json
{
  "schema": "fast-rcnn-serving/v3",
  "input_layout": "NCHW",
  "input_dtype": "float32",
  "canvas_hw": [640, 640],
  "box_format": "xyxy-half-open",
  "box_space": "canvas-pixel",
  "roi_op": "roi_align",
  "roi_output_hw": [7, 7],
  "spatial_scale": 0.0625,
  "sampling_ratio": 2,
  "aligned": true,
  "class_count_including_background": 21,
  "box_head": "class-specific",
  "max_proposals_per_image": 512,
  "score_threshold": 0.05,
  "nms_iou_threshold": 0.5
}
```

manifest가 다른 backbone·head·preprocessor를 조합하면 load를 거부한다. 같은 shape만으로 호환성을 판단하지 않는다.

## 4. ragged proposal batch

이미지 $i$의 proposal 수를 $R_i$라 하면 batch 전체 수는 다음과 같다.

$$
R=\sum_{i=0}^{N-1}R_i
$$

RoI tensor는 `(R,5)`이고 각 행은 `[batch_index,x1,y1,x2,y2]`다. 요청별 경계는 prefix sum으로 보존한다.

$$
o_0=0,\qquad o_{i+1}=o_i+R_i
$$

이미지 $i$의 출력 slice는 `[o_i:o_{i+1})`다. `batch_index`만 믿고 출력 순서를 복원하지 말고 입력 순서를 보존하는지 contract test로 확인한다.

### 4.1 tensor shape 추적

`N=3`, proposal 수 `(4,1,3)`, class 수 $K=5$인 작은 batch를 보자.

| 단계 | shape | dtype | 의미 |
| --- | --- | --- | --- |
| canvas image | `[3,3,224,224]` | `float32` | NCHW |
| stride-16 feature | `[3,256,14,14]` | `float32` | backbone 출력 |
| packed RoI | `[8,5]` | `float32` | batch index + canvas `xyxy` |
| prefix offset | `[4]` | `int64` | `[0,4,5,8]` |
| `RoIAlign` | `[8,256,7,7]` | `float32` | RoI별 feature |
| flatten | `[8,12544]` | `float32` | $256\cdot7\cdot7$ |
| hidden | `[8,1024]` | `float32` | shared head |
| class logits | `[8,5]` | `float32` | background 포함 |
| box delta | `[8,4,4]` | `float32` | 객체 class 4개 |

### 4.2 zero-proposal 이미지

`R_i=0`은 오류가 아닐 수 있다. 빈 장면이나 proposal filter 결과일 수 있다. 서비스는 다음 중 하나를 명시해야 한다.

- 빈 detection list를 정상 반환한다.
- 최소 fallback proposal을 만든다.
- upstream proposal 장애로 분류해 요청을 실패시킨다.

정책 없이 `torch.cat([])`에 맡기면 batch 전체가 실패한다.

### 4.3 admission control

한 이미지의 proposal 수가 $R_{max}$를 넘으면 score 기반 stable top-k를 적용하거나 요청을 격리한다. 동점은 원래 proposal index로 결정해 재실행 결과를 고정한다. 모든 proposal을 받은 뒤 GPU OOM이 나게 두는 것은 admission control이 아니다.

## 5. RoIAlign을 샘플링 연산으로 이해하기

RoIAlign은 정수 bin으로 먼저 양자화하지 않는다. feature map $F$의 실수 좌표 $(y,x)$에서 bilinear interpolation을 사용한다.

주변 네 점의 index를 $y_0=\lfloor y\rfloor$, $y_1=y_0+1$, $x_0=\lfloor x\rfloor$, $x_1=x_0+1$이라 하자.

$$
\delta_y=y-y_0,\qquad \delta_x=x-x_0
$$

$$
\begin{aligned}
v(y,x)={}&(1-\delta_y)(1-\delta_x)F_{y_0x_0}\\
&+(1-\delta_y)\delta_xF_{y_0x_1}\\
&+\delta_y(1-\delta_x)F_{y_1x_0}\\
&+\delta_y\delta_xF_{y_1x_1}
\end{aligned}
$$

출력 bin마다 `sampling_ratio=q`이면 대개 $q\times q$ sample을 평균한다. `q=-1` 같은 adaptive 설정은 box 크기에 따라 sample 수가 달라져 latency 분산도 달라진다.

`aligned=true`는 좌표에 반 픽셀 보정을 적용하는 구현 의미론이다. 이를 `false`로 바꾸면 출력 shape는 같고 값만 달라진다. 따라서 다음 네 필드를 독립적으로 versioning한다.

- operator 종류
- `spatial_scale`
- `sampling_ratio`
- `aligned`

## 6. NumPy 수작업 검증

다음은 **실행 가능한 코드**다. letterbox box 변환, clipping, 역변환을 독립적인 NumPy oracle로 확인한다.

```python
import numpy as np


def transform_boxes(boxes, scale_xy, pad_xy):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [R,4]")
    if not np.isfinite(boxes).all():
        raise ValueError("non-finite box")
    sx, sy = scale_xy
    px, py = pad_xy
    out = boxes.copy()
    out[:, [0, 2]] = out[:, [0, 2]] * sx + px
    out[:, [1, 3]] = out[:, [1, 3]] * sy + py
    return out


def clip_and_validate(boxes, height, width):
    out = boxes.copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, width)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, height)
    if np.any(out[:, 2] <= out[:, 0]) or np.any(out[:, 3] <= out[:, 1]):
        raise ValueError("degenerate box")
    return out


original = np.array([
    [10.0, 20.0, 110.0, 220.0],
    [600.0, 430.0, 650.0, 490.0],
])
canvas = transform_boxes(original, (1.0, 1.0), (0.0, 80.0))
canvas = clip_and_validate(canvas, height=640, width=640)
expected = np.array([
    [10.0, 100.0, 110.0, 300.0],
    [600.0, 510.0, 640.0, 570.0],
])
np.testing.assert_allclose(canvas, expected)

recovered = transform_boxes(canvas, (1.0, 1.0), (0.0, -80.0))
np.testing.assert_allclose(recovered[0], original[0])
print(canvas.tolist())
print("NumPy coordinate golden passed")
```

두 번째 box는 원본 $W_0=640$을 벗어난다. 이 예제는 canvas에서 clip하지만 production ingress에서는 원본 범위 초과량을 metric으로 남긴다. 큰 초과는 upstream 좌표계 오류일 가능성이 높다.

## 7. 실행 가능한 PyTorch serving core

다음 코드는 학습 loop가 아니라 **실행 가능한 추론 경계 예제**다. 2회차의 loss 구현 대신 입력 검증, stable truncation, ragged packing, `RoIAlign`, request별 unpack을 검증한다.

```python
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.ops import roi_align


@dataclass(frozen=True)
class RoiContract:
    canvas_h: int = 16
    canvas_w: int = 16
    spatial_scale: float = 0.25
    output_h: int = 2
    output_w: int = 2
    sampling_ratio: int = 2
    aligned: bool = True
    max_proposals: int = 3


def validate_and_limit(boxes, scores, contract):
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be [R,4]")
    if scores.shape != (boxes.shape[0],):
        raise ValueError("scores must be [R]")
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError("non-finite input")
    boxes = boxes.clone()
    boxes[:, [0, 2]].clamp_(0.0, float(contract.canvas_w))
    boxes[:, [1, 3]].clamp_(0.0, float(contract.canvas_h))
    if ((boxes[:, 2:] - boxes[:, :2]) <= 0).any():
        raise ValueError("degenerate box")

    # Python의 stable sort는 동점에서 원래 index를 보존한다.
    order = torch.argsort(scores, descending=True, stable=True)
    order = order[: contract.max_proposals]
    return boxes[order], scores[order], order


def pack_rois(boxes_per_image):
    rows = []
    offsets = [0]
    for batch_index, boxes in enumerate(boxes_per_image):
        if boxes.numel():
            index = torch.full(
                (boxes.shape[0], 1), float(batch_index), dtype=boxes.dtype
            )
            rows.append(torch.cat((index, boxes), dim=1))
        offsets.append(offsets[-1] + boxes.shape[0])
    rois = torch.cat(rows) if rows else torch.empty((0, 5), dtype=torch.float32)
    return rois, torch.tensor(offsets, dtype=torch.int64)


class ServingHead(nn.Module):
    def __init__(self, channels, classes, contract):
        super().__init__()
        self.contract = contract
        features = channels * contract.output_h * contract.output_w
        self.classifier = nn.Linear(features, classes)

    def forward(self, feature, rois):
        if rois.shape[0] == 0:
            return feature.new_empty((0, self.classifier.out_features))
        pooled = roi_align(
            feature,
            rois,
            output_size=(self.contract.output_h, self.contract.output_w),
            spatial_scale=self.contract.spatial_scale,
            sampling_ratio=self.contract.sampling_ratio,
            aligned=self.contract.aligned,
        )
        return self.classifier(pooled.flatten(1))


torch.manual_seed(904)
contract = RoiContract()
raw_boxes = [
    torch.tensor([
        [0.0, 0.0, 8.0, 8.0],
        [4.0, 4.0, 16.0, 16.0],
        [1.0, 1.0, 15.0, 15.0],
        [2.0, 2.0, 10.0, 10.0],
    ]),
    torch.tensor([[0.0, 0.0, 16.0, 16.0]]),
]
raw_scores = [torch.tensor([0.8, 0.9, 0.9, 0.2]), torch.tensor([0.7])]

limited = []
orders = []
for boxes, scores in zip(raw_boxes, raw_scores):
    kept_boxes, _, kept_order = validate_and_limit(boxes, scores, contract)
    limited.append(kept_boxes)
    orders.append(kept_order)

rois, offsets = pack_rois(limited)
assert offsets.tolist() == [0, 3, 4]
assert orders[0].tolist() == [1, 2, 0]  # 0.9 동점에서 index 1이 먼저다.

feature = torch.arange(2 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 2, 4, 4)
model = ServingHead(channels=2, classes=3, contract=contract).eval()
with torch.inference_mode():
    logits = model(feature, rois)
    again = model(feature, rois)
torch.testing.assert_close(logits, again, rtol=0.0, atol=0.0)
assert logits.shape == (4, 3)
per_request = [logits[offsets[i]:offsets[i + 1]] for i in range(2)]
assert [part.shape[0] for part in per_request] == [3, 1]

changed = RoiContract(aligned=False)
other = ServingHead(channels=2, classes=3, contract=changed).eval()
other.load_state_dict(model.state_dict())
with torch.inference_mode():
    shifted = other(feature, rois)
assert not torch.allclose(logits, shifted)

try:
    validate_and_limit([[1.0, 1.0, 1.0, 2.0]], [0.4], contract)
except ValueError as error:
    assert "degenerate" in str(error)
else:
    raise AssertionError("degenerate box was accepted")

print("rois", tuple(rois.shape), "offsets", offsets.tolist())
print("logits", tuple(logits.shape), "aligned_max_delta", float((logits - shifted).abs().max()))
print("PyTorch serving contract passed")
```

이 예제는 `aligned` 하나만 바꿔도 weight와 output shape가 같지만 값이 달라짐을 증명한다. manifest 비교 없이 모델 파일 hash만 검사하면 이런 semantic drift를 놓친다.

## 8. 성능과 메모리 예산

### 8.1 RoI activation

RoI 출력이 `[R,C,P_h,P_w]`이고 원소 byte가 $b$이면 activation byte는 다음과 같다.

$$
M_{roi}=RCP_hP_wb
$$

$R=512$, $C=256$, $P_h=P_w=7$, FP32라면 다음과 같다.

$$
M_{roi}=512\cdot256\cdot7\cdot7\cdot4=25{,}690{,}112\text{ bytes}
$$

약 24.5 MiB이며, FC intermediate와 allocator workspace는 별도다. 동시 batch의 총 proposal 수가 메모리의 핵심 축이다.

### 8.2 첫 FC 비용

첫 input이 $D=CP_hP_w$, output이 $U$인 Linear의 곱셈-덧셈은 대략 다음과 같다.

$$
\operatorname{FLOPs}_{fc}\approx2RDU
$$

$C=256$, $P_h=P_w=7$, $U=1024$, $R=512$이면 약 13.15 GFLOPs다. backbone을 공유해도 proposal 수가 두 배면 head 비용도 거의 두 배가 된다.

### 8.3 queue와 batch 정책

request batch size $N$만으로 bucket을 만들면 안 된다. 비용 key에 총 proposal 수를 포함한다.

```text
(canvas_h, canvas_w, dtype, provider, total_roi_bucket)
```

예를 들어 `total_roi_bucket`을 `(1-64, 65-256, 257-512)`로 나누면 작은 요청이 proposal 2,000개 요청 뒤에서 기다리는 head-of-line blocking을 줄일 수 있다.

### 8.4 측정 항목

- decode·preprocess·backbone·RoI op·head·postprocess별 latency
- 이미지당·batch당 proposal 수 histogram
- queue wait와 compute latency 분리
- provider별 peak allocated·reserved byte
- admission reject·truncate·fallback 횟수
- `RoIAlign` adaptive sampling을 쓴다면 sample 수 추정 histogram

## 9. C++17 portable golden

다음은 **외부 ML library 없이 실행 가능한 C++17 코드**다. ONNX Runtime wrapper 앞단의 letterbox transform, clipping, ragged offset을 검증한다.

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Box = std::array<float, 4>;

Box transform(Box b, float sx, float sy, float px, float py) {
    for (float value : b) {
        if (!std::isfinite(value)) throw std::invalid_argument("non-finite box");
    }
    return {b[0] * sx + px, b[1] * sy + py,
            b[2] * sx + px, b[3] * sy + py};
}

Box clip_and_validate(Box b, float height, float width) {
    b[0] = std::clamp(b[0], 0.0F, width);
    b[2] = std::clamp(b[2], 0.0F, width);
    b[1] = std::clamp(b[1], 0.0F, height);
    b[3] = std::clamp(b[3], 0.0F, height);
    if (!(b[0] < b[2] && b[1] < b[3])) {
        throw std::invalid_argument("degenerate box");
    }
    return b;
}

std::vector<int> prefix_offsets(const std::vector<int>& counts) {
    std::vector<int> out{0};
    for (int count : counts) {
        if (count < 0) throw std::invalid_argument("negative count");
        out.push_back(out.back() + count);
    }
    return out;
}

int main() {
    const Box first = clip_and_validate(
        transform({10, 20, 110, 220}, 1, 1, 0, 80), 640, 640);
    const Box second = clip_and_validate(
        transform({600, 430, 650, 490}, 1, 1, 0, 80), 640, 640);
    assert((first == Box{10, 100, 110, 300}));
    assert((second == Box{600, 510, 640, 570}));
    const auto offsets = prefix_offsets({3, 0, 1});
    assert((offsets == std::vector<int>{0, 3, 3, 4}));

    std::cout << std::fixed << std::setprecision(1)
              << first[0] << ' ' << first[1] << ' '
              << second[2] << ' ' << second[3] << "\n";
    std::cout << "offsets 0 3 3 4\nC++ proposal ABI passed\n";
}
```

실제 C++ runtime에서는 flat box buffer를 `{R,4}`로 해석하는지, `int64` offset이 API 경계에서 잘리지 않는지 추가로 검사한다.

## 10. C# portable golden

다음은 C++와 같은 fixture를 사용하는 **외부 ML library 없는 실행 가능한 C# 코드**다.

```csharp
using System;
using System.Collections.Generic;

public static class ProposalAbi
{
    static float[] Transform(float[] b, float sx, float sy, float px, float py)
    {
        foreach (float value in b)
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw new ArgumentException("non-finite box");
        return new[] { b[0] * sx + px, b[1] * sy + py,
                       b[2] * sx + px, b[3] * sy + py };
    }

    static float[] ClipAndValidate(float[] b, float height, float width)
    {
        b[0] = Math.Clamp(b[0], 0.0F, width);
        b[2] = Math.Clamp(b[2], 0.0F, width);
        b[1] = Math.Clamp(b[1], 0.0F, height);
        b[3] = Math.Clamp(b[3], 0.0F, height);
        if (!(b[0] < b[2] && b[1] < b[3]))
            throw new ArgumentException("degenerate box");
        return b;
    }

    static int[] PrefixOffsets(int[] counts)
    {
        var result = new List<int> { 0 };
        foreach (int count in counts)
        {
            if (count < 0) throw new ArgumentException("negative count");
            result.Add(result[result.Count - 1] + count);
        }
        return result.ToArray();
    }

    static void AssertArray<T>(T[] actual, T[] expected)
    {
        if (actual.Length != expected.Length) throw new Exception("length mismatch");
        for (int i = 0; i < actual.Length; i++)
            if (!actual[i].Equals(expected[i])) throw new Exception("value mismatch");
    }

    public static void Main()
    {
        var first = ClipAndValidate(
            Transform(new[] { 10F, 20F, 110F, 220F }, 1, 1, 0, 80), 640, 640);
        var second = ClipAndValidate(
            Transform(new[] { 600F, 430F, 650F, 490F }, 1, 1, 0, 80), 640, 640);
        AssertArray(first, new[] { 10F, 100F, 110F, 300F });
        AssertArray(second, new[] { 600F, 510F, 640F, 570F });
        AssertArray(PrefixOffsets(new[] { 3, 0, 1 }), new[] { 0, 3, 3, 4 });
        Console.WriteLine($"{first[0]:F1} {first[1]:F1} {second[2]:F1} {second[3]:F1}");
        Console.WriteLine("offsets 0 3 3 4\nC# proposal ABI passed");
    }
}
```

C#의 `float[]`에는 shape나 좌표계가 없다. `DenseTensor<float>`를 만들기 전에 model metadata와 manifest를 대조하고 `R=0`, `R=1`, 여러 이미지가 섞인 fixture를 모두 통과시킨다.

## 11. 프레임워크 간 shape·layout·dtype 대응

| 경계 | Python·PyTorch | C++ | C# | release 조건 |
| --- | --- | --- | --- | --- |
| 이미지 | NCHW tensor | 보통 flat buffer + dims | `DenseTensor<float>` | layout·색 순서 parity |
| proposal | `[R,4]` 또는 list | `float[R*4]` | `float[]` + `{R,4}` | `xyxy`·좌표계 동일 |
| batch index | RoI 첫 열 float 의미 | cast·concat 주의 | float buffer 주의 | 정수 값·범위 검사 |
| offsets | `int64 [N+1]` | `int64_t` 권장 | `long[]` 권장 | zero-proposal 포함 |
| RoI 출력 | `[R,C,Ph,Pw]` | graph metadata | flat + dims | NCHW checksum |
| class logit | `[R,K]` | row-major 확인 | flat + dims | class map hash |
| box delta | `[R,K-1,4]` | reshape 순서 | reshape 순서 | class slice golden |

### 11.1 dtype 경계

- 좌표 transform과 box decode는 최소 FP32로 수행한다.
- FP16 feature를 쓰더라도 NMS·softmax·`exp`가 provider에서 어느 dtype으로 계산되는지 확인한다.
- class ID와 offsets는 부동소수점으로 장기 보관하지 않는다.
- INT8 RoI feature는 per-tensor·per-channel scale과 zero point가 graph revision에 묶여야 한다.

### 11.2 class map

`class 0=background`인 head와 객체 class만 출력하는 postprocessor를 혼합하면 모든 label이 한 칸 이동한다. 다음을 bundle에 넣는다.

```json
{
  "class_map_revision": "voc20-bg0/v2",
  "logit_0": "background",
  "box_delta_axis": "foreground-class-only",
  "box_delta_index": "class_id-1"
}
```

## 12. 테스트 전략

### 12.1 좌표 단위 테스트

- identity resize에서 box가 변하지 않는다.
- letterbox padding이 $x$와 $y$ 중 올바른 축에만 더해진다.
- transform 뒤 inverse가 clipping되지 않은 box를 복원한다.
- `NaN`, `inf`, 음의 면적, zero-area box를 거부한다.
- 오른쪽·아래 끝점이 image 크기와 같아도 반열린 좌표계에서 유효하다.

### 12.2 ragged batch property test

임의의 nonnegative counts에 대해 다음을 확인한다.

$$
o_0=0
$$

$$
o_{i+1}\ge o_i
$$

$$
o_N=R
$$

pack 뒤 unpack한 box와 request ID 순서가 원본과 같아야 한다.

### 12.3 operator parity test

작은 정수 feature와 fractional RoI fixture를 저장하고 다음 matrix를 비교한다.

| 축 | 값 |
| --- | --- |
| provider | CPU, CUDA, target accelerator |
| dtype | FP32, 허용 시 FP16 |
| `aligned` | release 값만 허용, 반대 값은 negative test |
| proposal 수 | 0, 1, 여러 개 |
| batch index | 0만, 여러 이미지 혼합 |

### 12.4 metamorphic test

이미지와 proposal을 같은 비율로 scale하고 대응 feature가 이상적으로 scale-equivalent라면 결과가 허용 오차 안에 있어야 한다. 현실 backbone의 padding 때문에 완전 불변은 아니므로 golden tolerance를 empirical하게 정한다.

### 12.5 load·rollout test

- 잘못된 preprocessor revision이면 model load 실패
- 잘못된 class map이면 model load 실패
- 최대 proposal 경계 `R_max-1`, `R_max`, `R_max+1`
- canary에서 Python과 target runtime detection count·score·box parity
- rollback 뒤 이전 bundle hash와 metric 회복

## 13. 디버깅 플레이북

### 증상 1: 모든 box가 오른쪽 아래로 일정하게 밀린다

resize scale보다 padding offset 누락을 먼저 본다. 원본·canvas·feature 공간의 golden box를 한 요청에서 함께 log한다.

### 증상 2: batch size 1은 맞고 2부터 결과가 섞인다

RoI 첫 열의 batch index, prefix offset, request ID unpack 순서를 확인한다. 동일한 proposal 수의 테스트만 있으면 interleave 버그를 숨길 수 있다.

### 증상 3: 작은 객체에서만 Python·C++ 차이가 크다

`RoIPool` 대 `RoIAlign`, `aligned`, `sampling_ratio`, coordinate endpoint를 비교한다. 작은 box는 반 픽셀 차이에 더 민감하다.

### 증상 4: 평균 latency는 정상인데 p99가 급증한다

이미지 수가 아니라 총 proposal 수와 RoI 면적별로 latency를 나눈다. adaptive sampling과 outlier request가 원인일 수 있다.

### 증상 5: detection은 나오지만 class가 전부 한 칸씩 틀린다

background 포함 여부와 class map revision을 확인한다. 문자열 label 파일만 새것으로 바뀐 경우도 있다.

### 증상 6: box 폭이 `inf`가 된다

box delta의 $d_w,d_h$ clamp가 빠졌거나 잘못된 normalization std를 사용했다. `exp` 전에 finite와 상한을 검사한다.

### 증상 7: 재배포 후 결과 개수만 줄었다

score threshold, top-k, NMS IoU, class-aware 여부를 graph 밖 postprocess config까지 비교한다.

### 증상 8: GPU OOM 뒤 같은 pod가 계속 느리다

큰 proposal 요청을 격리하고 allocator reserved memory와 retry 폭주를 확인한다. 무조건 재시도하면 장애를 증폭한다.

## 14. 수치 안정성

### 14.1 안정적인 softmax

logit $z_k$는 최댓값을 빼고 exponentiation한다.

$$
p_k=\frac{\exp(z_k-z_{max})}{\sum_j\exp(z_j-z_{max})}
$$

### 14.2 box decode clamp

proposal width $p_w$와 예측 $d_w$에서 폭을 복원하면 다음과 같다.

$$
\widehat{w}=p_w\exp(d_w)
$$

따라서 $d_w$를 manifest의 $d_{max}$로 clamp한다. clamp 값은 image 크기와 학습 implementation에 맞춰 정하고 runtime마다 동일하게 둔다.

### 14.3 NMS 동점

같은 score의 box 순서가 provider마다 달라지면 NMS 결과도 달라질 수 있다. score 다음에 stable proposal ID를 tie-break key로 사용하거나 동점 비결정성을 허용 오차 정책에 명시한다.

### 14.4 `NaN` 정책

`NaN` score를 threshold 비교에 맡기지 않는다. request를 실패시키거나 해당 detection을 제거하고 counter를 증가시키는 정책을 고정한다. 조용히 0으로 바꾸면 model corruption을 숨길 수 있다.

## 15. ONNX와 배포 설계

### 15.1 세 가지 graph 경계

| 방식 | graph 입력·출력 | 장점 | 위험 |
| --- | --- | --- | --- |
| 전체 graph | image + proposals → detections | 계약이 한곳에 있음 | dynamic RoI·NMS 지원 차이 |
| backbone/head 분리 | image → feature, feature + RoI → logits·delta | 캐시·독립 확장 | revision 혼합·큰 feature 전송 |
| head only | feature + RoI → logits·delta | edge feature 재사용 | feature ABI 결합이 매우 강함 |

분리 graph를 쓰면 feature tensor에 다음 lineage를 붙인다.

```text
(backbone_revision, preprocess_revision, layout, dtype, shape, request_id)
```

오래된 cache feature를 새 head에 넣지 않도록 compatibility matrix를 load 시 검사한다.

### 15.2 export 성공은 release 성공이 아니다

release gate는 다음 순서다.

1. source framework eager golden
2. exported graph structural inspection
3. ONNX checker
4. CPU execution provider parity
5. production execution provider parity
6. zero·one·max proposal 경계
7. latency·peak memory benchmark
8. shadow traffic과 canary

이 환경에는 `onnx`와 `onnxruntime` 설치 여부가 보장되지 않으므로 오늘 코드는 실제 ONNX provider parity를 주장하지 않는다. 대신 NumPy·PyTorch·C++·C# portable ingress 계약을 실행 검증한다.

### 15.3 bundle 구성

```text
release/
  backbone.onnx
  roi_head.onnx
  preprocessor.json
  proposal_abi.json
  class_map.json
  postprocess.json
  golden_inputs/
  golden_outputs/
  compatibility.json
  checksums.sha256
```

파일 하나만 rollback하지 않고 bundle revision 전체를 rollback한다.

## 16. 관측성과 SLO

### 16.1 입력 지표

- `proposal_count_per_image`
- invalid·clipped·truncated proposal count
- box width·height·aspect ratio histogram
- 원본·canvas 해상도와 resize scale
- proposal generator revision

### 16.2 모델 지표

- class entropy와 background ratio
- score threshold 전후 detection count
- box delta $d_x,d_y,d_w,d_h$ quantile
- NMS 전후 count와 suppression ratio
- RoI feature norm·finite failure count

### 16.3 시스템 지표

- stage별 p50·p95·p99
- queue wait, batch size, total RoI
- provider fallback과 kernel failure
- GPU peak allocated·reserved memory
- timeout·admission reject·retry count

### 16.4 drift 해석

background ratio 상승만으로 model drift라 단정하지 않는다. proposal generator revision, camera crop, box size 분포가 먼저 바뀌었을 수 있다. 입력 lineage와 결과 지표를 같은 revision label로 연결한다.

## 17. 실무 실패 사례

### 사례 A: 이미지만 letterbox하고 box는 원본 좌표를 유지했다

shape와 dtype은 정상이라 runtime 오류가 없었다. 모든 RoI가 실제 객체보다 위쪽을 읽었고 padding이 큰 영상에서만 정확도가 급락했다. 해결은 affine golden과 좌표 space metadata였다.

### 사례 B: head만 hot swap했다

새 head는 `aligned=true`로 학습되었지만 서비스 graph는 `false`였다. model 파일은 정상 load되었고 output shape도 같았다. operator semantic을 bundle hash에 포함해 재발을 막았다.

### 사례 C: request batch만 제한했다

batch 4 제한은 있었지만 한 이미지가 proposal 8,000개를 보냈다. RoI activation과 FC가 OOM을 냈다. 이미지당·batch당 proposal budget을 둘 다 추가했다.

### 사례 D: zero-proposal을 예외로 처리했다

빈 장면이 하나 섞일 때 `cat` 경로에서 batch 전체가 실패했다. 빈 output을 정상 값으로 정의하고 offset fixture에 `0` count를 넣었다.

### 사례 E: background class를 제거하며 box index를 그대로 뒀다

class score는 그럴듯했지만 다른 class의 delta를 적용했다. class map과 `box_delta_index`를 manifest에서 함께 versioning했다.

### 사례 F: 평균 latency만 보았다

p50은 개선되었지만 adaptive sampling과 proposal outlier 때문에 p99가 악화했다. total RoI bucket과 queue wait를 분리해 병목을 찾았다.

### 사례 G: Fast R-CNN 장애에 RPN metric을 찾았다

실제 서비스는 외부 proposal을 입력받았는데 문서 이름이 Faster R-CNN으로 잘못 등록되어 있었다. architecture boundary와 owner를 runbook 첫 줄에 명시했다.

## 18. 운영 체크리스트

### 좌표·ABI

- [ ] `xyxy` 끝점 규칙과 coordinate space가 명시되어 있다.
- [ ] resize·crop·padding transform과 inverse가 golden으로 검증된다.
- [ ] `RoIPool`·`RoIAlign`, scale, sampling, aligned가 versioning된다.
- [ ] class map과 box delta index 규칙이 bundle에 있다.

### batch·입력

- [ ] zero-proposal image를 정상 처리한다.
- [ ] request ID와 prefix offset으로 결과를 복원한다.
- [ ] 이미지당·batch당 proposal 상한이 있다.
- [ ] stable top-k와 동점 정책이 있다.

### 성능·수치

- [ ] total RoI별 latency와 peak memory를 측정한다.
- [ ] box delta `exp` 전에 clamp한다.
- [ ] non-finite feature·logit·box 정책이 있다.
- [ ] dtype·provider별 golden tolerance를 기록한다.

### 배포·운영

- [ ] production provider에서 graph parity를 확인한다.
- [ ] backbone·head·preprocessor compatibility를 load 시 검사한다.
- [ ] shadow·canary·rollback이 bundle revision 단위다.
- [ ] proposal·feature·detection metric이 같은 lineage label을 쓴다.

## 19. 연습문제

### 문제 1

원본 box `(20,30,120,230)`을 $s_x=0.5$, $s_y=0.5$, padding `(0,40)`인 canvas로 옮겨라.

### 문제 2

proposal 수가 `(2,0,5)`일 때 prefix offset을 구하고 두 번째 이미지의 output slice를 쓰라.

### 문제 3

`R=256`, `C=128`, RoI 출력 `7 x 7`, FP16일 때 RoI activation byte를 구하라.

### 문제 4

Python과 C++의 RoI output shape는 같지만 작은 객체 값만 다른 경우 가장 먼저 비교할 네 설정은 무엇인가?

### 문제 5

왜 image batch size 상한만으로 OOM을 막을 수 없는가?

### 문제 6

backbone과 head를 분리 배포할 때 feature record에 필요한 lineage 네 가지 이상을 쓰라.

### 문제 7

zero-proposal 이미지가 정상 입력이라면 class logits와 offsets의 shape는 어떻게 되어야 하는가?

### 문제 8

Fast R-CNN의 proposal recall이 낮을 때 class head만 재학습하는 것이 왜 해결책이 아닐 수 있는가?

## 20. 해답

### 해답 1

corner마다 scale과 padding을 적용한다.

$$
(10,15,60,115)+(0,40)=(10,55,60,155)
$$

### 해답 2

offset은 `(0,2,2,7)`이다. 두 번째 이미지의 slice는 `[2:2)`이므로 비어 있다.

### 해답 3

$$
256\cdot128\cdot7\cdot7\cdot2=3{,}211{,}264\text{ bytes}
$$

약 3.06 MiB다. FC intermediate와 workspace는 포함하지 않는다.

### 해답 4

operator 종류, `spatial_scale`, `sampling_ratio`, `aligned`를 비교한다. coordinate endpoint와 box space도 이어서 확인한다.

### 해답 5

RoI activation과 head 계산량은 이미지 수뿐 아니라 총 proposal 수 $R$에 비례한다. 이미지 한 장도 수천 proposal을 가질 수 있다.

### 해답 6

`backbone_revision`, `preprocess_revision`, layout, dtype, shape, request ID 중 최소 네 가지가 필요하다. 실제로는 모두 저장하는 편이 안전하다.

### 해답 7

전체 batch의 $R=0$이면 logits는 `[0,K]`, offsets는 이미지 수가 $N$일 때 `[N+1]`이며 모든 값이 0이어야 한다.

### 해답 8

class head는 받은 proposal만 분류·보정한다. 정답 객체를 덮는 proposal 자체가 없으면 head가 그 객체를 복구할 수 없다. proposal generator와 recall을 먼저 진단한다.

## 21. 핵심 요약

1. proposal은 좌표 네 개가 아니라 image identity·좌표계·변환 계보를 가진 ABI다.
2. Fast R-CNN과 RPN 기반 Faster R-CNN의 책임 경계를 운영 문서에서 분리한다.
3. resize·letterbox·feature scale은 affine transform으로 추적하고 scale 소유권을 하나로 둔다.
4. ragged proposal은 prefix offset과 request ID로 pack·unpack한다.
5. `RoIAlign`의 scale·sampling·aligned는 output shape에 드러나지 않는 핵심 semantic이다.
6. 메모리와 head 비용은 이미지 수보다 총 proposal 수에 직접 좌우된다.
7. NumPy·PyTorch·C++·C# golden은 좌표와 request 경계를 함께 검증해야 한다.
8. ONNX export 성공과 production provider parity는 별도 release gate다.
9. bundle은 backbone·head뿐 아니라 preprocessor·proposal·class map·postprocess revision을 포함한다.
10. drift와 장애는 proposal lineage, RoI 수, box 분포, stage latency를 함께 보아야 해석할 수 있다.

## 22. 다음 학습 예고

다음은 3회차 실무 엔지니어 6/18 `02-05.YOLO.md`다. 원본의 YOLOv1·SSD/FPN·Focal Loss/RetinaNet 범위를 최대 3개 Part로 나눌지 먼저 판단한다. 첫 Part에서는 dense prediction의 grid·anchor ABI, resize·letterbox 역변환, decoder·NMS의 크로스런타임 동등성, throughput·메모리·관측성을 운영 시스템으로 연결한다.
