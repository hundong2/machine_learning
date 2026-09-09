<!-- curriculum: cycle=3; level=production-engineering; source_index=7/18; source=02-06.FCOS_DETR.md; part=1/2 -->

# FCOS 운영: point ABI부터 패딩 마스크와 후보 예산까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-09 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 7/18 · `02-06.FCOS_DETR.md` |
| Part | 1/2 · FCOS serving·운영 계약 |
| 이전 소스 | 6/18 · `02-05.YOLO.md` Part 3/3 · RetinaNet score 운영 |
| 다음 소스 | 7/18 · `02-06.FCOS_DETR.md` Part 2/2 · DETR serving·운영 |

## 오늘의 운영 질문

오프라인 AP가 같은 FCOS 모델을 새 runtime에 배포했더니 영상 가장자리의 가짜 박스가 늘고 p99 지연도 두 배가 되었다. 가중치가 같다면 원인은 어디에 있을까?

```text
raw image -> resize/pad -> FPN -> point grid -> class/LTRB/center logits
          -> valid-point mask -> distance decode -> score fusion
          -> level budget -> inverse geometry -> class NMS -> response
```

FCOS의 결과는 가중치만의 함수가 아니다. point offset, stride, 회귀 단위, padding mask, score 결합, 후보 제한, 좌표 역변환이 모두 같은 릴리스 계약에 속한다. 오늘은 이 경계를 재현 가능한 운영 단위로 묶는다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. FCOS의 point grid를 버전이 있는 ABI로 정의한다.
2. `LTRB`가 pixel 단위인지 feature-cell 단위인지 구분해 decode한다.
3. resize·padding 계보를 추적하고 padding point를 후보에서 제거한다.
4. class와 center-ness logit을 수치적으로 안정하게 결합한다.
5. level별 tensor shape, flatten 순서, candidate 예산을 계산한다.
6. Python, C++17, C# decoder의 golden 결과를 맞춘다.
7. ONNX export와 runtime별 layout·dtype 차이를 release gate로 검사한다.
8. 후보 폭주, 경계 박스, stride 불일치 장애를 metric으로 진단한다.

## 선수 지식과 기호

- FPN의 stride와 다중 해상도 feature
- `XYXY` 박스, IoU, class별 NMS
- sigmoid, Focal Loss, binary cross entropy
- FP16, INT8, ONNX Runtime의 기본 개념
- p50·p95·p99 latency와 canary release

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $K$ | foreground class 수 |
| $L$ | FPN level 수 |
| $H_l,W_l$ | level $l$의 feature 높이와 너비 |
| $s_l$ | 입력 tensor 좌표 기준 stride |
| $Q_l=H_lW_l$ | level $l$의 point 수 |
| $p_{lij}=(x,y)$ | level $l$, 행 $j$, 열 $i$의 reference point |
| $d=(l,t,r,b)$ | point에서 박스 네 변까지의 거리 |
| $z^c,z^{ctr}$ | class logit과 center-ness logit |
| $R_l$ | level $l$에서 예산 적용 뒤 남은 후보 수 |
| $(a_x,a_y)$ | 실제 정수 resize 비율 |
| $(p_x,p_y)$ | 왼쪽·위 padding 크기 |

좌표는 연속 `XYXY=(x_0,y_0,x_1,y_1)`, image와 feature는 기본적으로 `NCHW`, point는 `(x,y)`, feature index는 `(j,i)` 순서를 사용한다. 박스의 오른쪽·아래 경계를 inclusive pixel index로 다루지 않는다.

## 1. 원본과 앞선 회차에서 이번에 확장하는 것

원본 [02-06.FCOS_DETR.md](../05.ImageClassification/02-06.FCOS_DETR.md)는 anchor-free 거리 회귀와 center-ness의 직관을 제공한다. 1회차는 point·`LTRB`·level range를 유도했고, 2회차는 target assigner부터 loss와 NMS까지 구현했다. 이번 글은 학습 산출물을 실제 서비스에 연결한다.

| 과거 초점 | 오늘 추가하는 운영 계약 |
| --- | --- |
| feature 위치에서 `LTRB` 회귀 | point offset·stride·회귀 단위를 manifest에 고정 |
| positive target과 center sampling | 실제 content rectangle과 valid-point mask를 추론에도 적용 |
| class·box·center-ness loss | raw output 이름, layout, dtype, activation 책임을 고정 |
| score 결합 뒤 NMS | level별 threshold·top-k와 candidate funnel을 계측 |
| 단일 runtime 정확성 | Python·C++·C#·ONNX의 golden tensor와 좌표를 비교 |
| 모델 checkpoint 저장 | 전처리·모델·decoder·NMS·calibrator를 immutable bundle로 배포 |

원문의 표현도 운영 관점에서 다음처럼 구분해야 한다.

1. “각 픽셀”은 보통 입력 원본 pixel이 아니라 FPN feature의 spatial location이다.
2. 예측 거리가 음수인지 보고 background target을 정하지 않는다. target assignment는 고정 point와 GT로 수행하며, 배포에서는 head의 거리 parameterization을 그대로 재현한다.
3. anchor shape 열거는 사라지지만 stride, point offset, FPN range 같은 기하 계약은 남는다.
4. center-ness는 보통 별도 BCE로 학습하고 추론 score를 정렬할 때 class probability와 결합한다.
5. 고전적 FCOS는 NMS를 사용한다. NMS 없는 집합 예측은 Part 2의 DETR와 구별해야 한다.
6. self-attention만으로 중복이 사라진다는 원본의 DETR 설명도 불완전하다. 일대일 매칭 목적함수가 핵심이며 이는 다음 글에서 다룬다.

## 2. 직관: FCOS 배포는 보이지 않는 격자를 복원하는 일이다

FCOS ONNX graph가 `cls_p3`, `box_p3`, `ctr_p3`만 반환한다고 하자. 출력에는 point 좌표가 들어 있지 않을 수 있다. 소비자가 feature shape와 stride로 point를 다시 만든다.

point를 `(i+0.5)s` 대신 `is`로 만들면 모든 박스가 반 stride만큼 왼쪽 위로 이동한다. P3에서는 4 pixel, P7에서는 64 pixel의 오차다. 회귀값이 feature-cell 단위인데 pixel 단위로 읽으면 P7 상자는 128배 작아진다. padding 영역을 mask하지 않으면 검은 테두리에서 높은 logit을 낸 point가 실제 영상 밖 박스로 변환된다.

가중치 checksum이 같아도 decoder가 다르면 서로 다른 모델을 서비스하는 셈이다.

## 3. point ABI와 좌표 계보

### 3.1 point 생성식

cell-center offset을 $o=0.5$로 두면 point는 다음과 같다.

$$
x=s_l(i+o)
$$

$$
y=s_l(j+o)
$$

$$
0\le i<W_l,\qquad 0\le j<H_l
$$

point tensor는 level 안에서 행 우선으로 펼친다.

$$
P_l\in\mathbb{R}^{Q_l\times2},\qquad q=jW_l+i
$$

모든 level을 잇는 순서는 `P3,P4,P5,P6,P7`처럼 manifest에 명시한다.

$$
P=\operatorname{concat}(P_3,P_4,P_5,P_6,P_7)
$$

### 3.2 실제 stride를 검증한다

nominal stride가 8이더라도 동적 입력과 `ceil_mode`, asymmetric padding을 섞으면 feature shape만으로 정확히 8배 관계가 아닐 수 있다. 모델이 보장하는 stride를 사용하되 다음 조건을 각 허용 input profile에서 검사한다.

$$
H_l=\left\lceil\frac{H_{in}}{s_l}\right\rceil,\qquad
W_l=\left\lceil\frac{W_{in}}{s_l}\right\rceil
$$

이 식이 모델과 다르면 point를 출력 shape에 억지로 맞추지 말고 backbone의 padding 규칙부터 확인한다.

### 3.3 resize와 padding의 실제 비율

원본 크기를 $(H_0,W_0)$, 정수 resize 결과를 $(H_r,W_r)$라 하자. 요청한 실수 scale이 아니라 실제 정수 결과에서 축별 비율을 계산한다.

$$
a_x=\frac{W_r}{W_0},\qquad a_y=\frac{H_r}{H_0}
$$

왼쪽과 위에 각각 $p_x,p_y$만큼 padding했다면 원본 좌표로의 역변환은 다음과 같다.

$$
x^{orig}=\frac{x^{pad}-p_x}{a_x}
$$

$$
y^{orig}=\frac{y^{pad}-p_y}{a_y}
$$

요청 scale 하나를 두 축에 공통으로 쓰면 정수 반올림 때문에 가장자리에서 drift가 생길 수 있다.

### 3.4 valid-point mask

padding된 입력 좌표에서 실제 content rectangle은 다음과 같다.

$$
\mathcal{C}=[p_x,p_x+W_r)\times[p_y,p_y+H_r)
$$

point가 이 반열린 영역 안에 있을 때만 유효하다.

$$
m_q=\mathbb{1}[p_x\le x_q<p_x+W_r]\mathbb{1}[p_y\le y_q<p_y+H_r]
$$

score threshold 전에 invalid point를 `-inf` logit으로 바꾸거나 boolean mask로 제거한다. decode 뒤 clip만 하는 방식은 padding 후보가 영상 테두리의 정상 박스로 접혀 살아남을 수 있다.

## 4. raw output과 distance decode 계약

### 4.1 level별 output shape

anchor가 한 위치당 하나인 FCOS의 raw output은 다음과 같다.

$$
Z_l^{cls}\in\mathbb{R}^{N\times K\times H_l\times W_l}
$$

$$
Z_l^{box}\in\mathbb{R}^{N\times4\times H_l\times W_l}
$$

$$
Z_l^{ctr}\in\mathbb{R}^{N\times1\times H_l\times W_l}
$$

세 tensor는 모두 `[N,C,H,W] -> [N,H,W,C] -> [N,Q_l,C]`로 바꾼다. class만 column-major로 펼쳐도 shape는 맞지만 point와 내용이 어긋난다. 비정사각 $H_l\ne W_l$의 index-coded tensor가 필요한 이유다.

### 4.2 distance parameterization

runtime이 받는 box output $u$가 실제 distance $d$로 바뀌는 규칙은 구현마다 다를 수 있다.

| 계약 이름 | 변환 | 주의점 |
| --- | --- | --- |
| `pixel_relu` | $d=\max(u,0)$ | 이미 입력 pixel 단위 |
| `cell_relu` | $d=s_l\max(u,0)$ | decode 때 stride 필요 |
| `cell_exp` | $d=s_l\exp(u)$ | 큰 logit overflow 방지 필요 |
| `scaled_relu` | $d=s_l\max(\gamma_lu,0)$ | 학습된 level scale $\gamma_l$ 포함 |

이 글의 golden 예제는 `cell_relu`를 쓴다. manifest에는 `regression_unit`, `activation`, `level_scale`, `stride`를 각각 저장한다. 단순히 출력 이름을 `boxes`라고 쓰면 부족하다.

### 4.3 `LTRB` decode

point $p=(x,y)$와 pixel distance $d=(l,t,r,b)$에서 padded-input 박스를 얻는다.

$$
x_0=x-l,\qquad y_0=y-t
$$

$$
x_1=x+r,\qquad y_1=y+b
$$

거리의 channel 순서를 `LTRB` 대신 `TLBR`로 읽는 오류는 정사각 대칭 golden box에서 발견되지 않는다. 네 거리가 모두 다른 비대칭 값을 사용한다.

## 5. score fusion과 후보 예산

### 5.1 stable sigmoid

logit $z$의 sigmoid는 다음과 같다.

$$
\sigma(z)=\frac{1}{1+\exp(-z)}
$$

overflow를 피하는 분기형 계산은 다음과 같다.

$$
\sigma(z)=
\begin{cases}
\dfrac{1}{1+\exp(-z)},&z\ge0\\
\dfrac{\exp(z)}{1+\exp(z)},&z<0
\end{cases}
$$

### 5.2 class와 center-ness 결합

class $c$의 probability와 center-ness probability를 다음처럼 둔다.

$$
p_c=\sigma(z^c),\qquad p_{ctr}=\sigma(z^{ctr})
$$

이 글의 ranking score 계약은 원래 FCOS 구현에서 흔히 쓰는 다음 식이다.

$$
s_c=\sqrt{p_cp_{ctr}}
$$

제곱근은 $p_cp_{ctr}$의 순서를 바꾸지 않지만 같은 숫자 threshold를 쓰면 통과 집합은 달라진다.

$$
s_c\ge\tau\quad\Longleftrightarrow\quad p_cp_{ctr}\ge\tau^2
$$

한 runtime이 곱에 threshold `0.05`를 쓰고 다른 runtime이 제곱근에 `0.05`를 쓰면 서로 다른 운영점이다.

### 5.3 level별 후보 funnel

threshold를 통과한 level $l$의 point-class 후보 수를 $C_l$이라 하자.

$$
C_l=\sum_{q=1}^{Q_l}\sum_{c=1}^{K}
\mathbb{1}[m_q=1]\mathbb{1}[s_{lqc}\ge\tau_{lc}]
$$

level별 pre-NMS cap $B_l$을 적용하면 다음과 같다.

$$
R_l=\min(C_l,B_l)
$$

$$
R=\sum_{l=1}^{L}R_l
$$

threshold만 믿지 말고 cap을 둔다. score drift가 생겨도 후처리 비용의 상한을 유지할 수 있다. 단, cap이 작은 객체 recall을 자르지 않는지 size bucket별로 검증한다.

### 5.4 메모리와 NMS 비용

후보 하나에 FP32 box 4개, score 1개와 int32 class·level·point index 3개를 저장하면 최소 32 bytes다.

$$
M_{cand}\ge32R\ \text{bytes}
$$

단순 pairwise NMS의 최악 비교량은 $O(R^2)$다. 그래서 `valid mask -> threshold -> level top-k -> decode -> class NMS` 순서가 중요하다. 모든 위치를 먼저 decode하고 host로 복사하면 GPU 계산보다 전송과 정렬이 더 비쌀 수 있다.

## 6. 수작업·NumPy golden 검증

다음 코드는 **실행 가능한 독립 예제**다. point, cell 단위 거리, padding mask, score 결합, 원본 좌표 역변환을 한 번에 검증한다.

```python
import math
import numpy as np


def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def make_points(height, width, stride, offset=0.5):
    ys = (np.arange(height, dtype=np.float64) + offset) * stride
    xs = (np.arange(width, dtype=np.float64) + offset) * stride
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def decode_cell_ltrb(points, raw_ltrb, stride):
    distance = np.maximum(raw_ltrb, 0.0) * stride
    return np.column_stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ]
    )


def valid_points(points, content_xyxy):
    x0, y0, x1, y1 = content_xyxy
    return (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )


stride = 8.0
points = make_points(height=2, width=3, stride=stride)
expected_points = np.array(
    [[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]],
    dtype=np.float64,
)
np.testing.assert_array_equal(points, expected_points)

# q=4의 point (12,12), 서로 다른 LTRB=(1, 0.5, 2, 1.5) cells.
raw_box = np.zeros((6, 4), dtype=np.float64)
raw_box[4] = [1.0, 0.5, 2.0, 1.5]
boxes = decode_cell_ltrb(points, raw_box, stride)
np.testing.assert_allclose(boxes[4], [4.0, 8.0, 28.0, 24.0])

# 실제 content는 padded 좌표에서 [4,4,20,16)이다.
valid = valid_points(points, [4.0, 4.0, 20.0, 16.0])
np.testing.assert_array_equal(valid, [True, True, False, True, True, False])

class_logit = np.array([-20.0, 0.0, 20.0])
ctr_logit = np.array([20.0, 0.0, -20.0])
score = np.sqrt(stable_sigmoid(class_logit) * stable_sigmoid(ctr_logit))
assert np.isfinite(score).all()
np.testing.assert_allclose(score[1], 0.5, atol=1e-12)

# 원본 8x12 -> 실제 resize 16x24, 왼쪽 4·위 8 padding.
box_pad = np.array([4.0, 8.0, 28.0, 24.0])
ax, ay = 24.0 / 12.0, 16.0 / 8.0
box_orig = box_pad.copy()
box_orig[[0, 2]] = (box_orig[[0, 2]] - 4.0) / ax
box_orig[[1, 3]] = (box_orig[[1, 3]] - 8.0) / ay
np.testing.assert_allclose(box_orig, [0.0, 0.0, 12.0, 8.0])

print("points:", points.tolist())
print("valid:", valid.tolist())
print("decoded padded:", boxes[4].tolist())
print("decoded original:", box_orig.tolist())
print("score middle:", f"{score[1]:.6f}")
```

예상 출력은 다음과 같다.

```text
points: [[4.0, 4.0], [12.0, 4.0], [20.0, 4.0], [4.0, 12.0], [12.0, 12.0], [20.0, 12.0]]
valid: [True, True, False, True, True, False]
decoded padded: [4.0, 8.0, 28.0, 24.0]
decoded original: [0.0, 0.0, 12.0, 8.0]
score middle: 0.500000
```

## 7. PyTorch mini head와 serving path

다음 코드는 **실행 가능한 학습·추론 검증 예제**다. production 모델의 축소판이며 output layout, finite backward, masking, level별 cap을 검사한다.

```python
import torch
from torch import nn


torch.manual_seed(20260909)


class MiniFCOSHead(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.cls = nn.Conv2d(channels, classes, 3, padding=1)
        self.box = nn.Conv2d(channels, 4, 3, padding=1)
        self.ctr = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        return self.cls(h), torch.relu(self.box(h)), self.ctr(h)


def flatten_nchw(x):
    return x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])


def points_torch(height, width, stride, device):
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def decode(points, raw_ltrb, stride):
    distance = raw_ltrb.float() * float(stride)
    return torch.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        dim=-1,
    )


head = MiniFCOSHead(channels=8, classes=3)
features = {
    "p3": torch.randn(2, 8, 5, 7, requires_grad=True),
    "p4": torch.randn(2, 8, 3, 4, requires_grad=True),
}
strides = {"p3": 8, "p4": 16}

all_losses = []
all_candidates = []
for level in ("p3", "p4"):
    cls_raw, box_raw, ctr_raw = head(features[level])
    cls = flatten_nchw(cls_raw)
    box = flatten_nchw(box_raw)
    ctr = flatten_nchw(ctr_raw)
    height, width = cls_raw.shape[-2:]
    point = points_torch(height, width, strides[level], cls.device)

    assert cls.shape == (2, height * width, 3)
    assert box.shape == (2, height * width, 4)
    assert ctr.shape == (2, height * width, 1)
    assert point.shape == (height * width, 2)

    # 설명용 surrogate loss지만 실제 backward가 가능하다.
    loss = cls.float().square().mean() + box.float().mean() + ctr.float().square().mean()
    all_losses.append(loss)

    score = torch.sqrt(torch.sigmoid(cls[0].float()) * torch.sigmoid(ctr[0].float()))
    valid = (point[:, 0] < 40.0) & (point[:, 1] < 32.0)
    score = score.masked_fill(~valid[:, None], -torch.inf)
    flat_score = score.reshape(-1)
    keep = torch.nonzero(flat_score >= 0.45, as_tuple=False).squeeze(1)
    cap = min(5, keep.numel())
    if cap:
        order = torch.argsort(flat_score[keep], descending=True, stable=True)[:cap]
        keep = keep[order]
    all_candidates.append(int(keep.numel()))

    decoded = decode(point, box[0], strides[level])
    assert decoded.shape == (height * width, 4)
    assert torch.isfinite(decoded).all()

total_loss = torch.stack(all_losses).sum()
total_loss.backward()

assert torch.isfinite(total_loss)
assert all(torch.isfinite(x.grad).all() for x in features.values())
print("p3 shapes:", tuple(flatten_nchw(head(features["p3"])[0]).shape))
print("p4 shapes:", tuple(flatten_nchw(head(features["p4"])[0]).shape))
print("candidates per level:", all_candidates)
print("finite backward: true")
```

이 예제에서 `box`는 `cell_relu`다. 실제 모델이 이미 pixel distance를 출력한다면 `decode`에서 stride를 곱하면 안 된다. 예제 loss는 shape와 autograd를 검증하기 위한 설명용 surrogate이며 production FCOS의 Focal·IoU/GIoU·center-ness loss를 대신하지 않는다.

## 8. C++17 decoder golden

다음 코드는 외부 라이브러리 없이 컴파일 가능한 **실행 가능한 C++17 예제**다.

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

double StableSigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double e = std::exp(z);
    return e / (1.0 + e);
}

std::array<double, 4> DecodeCellLtrb(
    double x,
    double y,
    const std::array<double, 4>& raw,
    double stride) {
    const double l = std::max(raw[0], 0.0) * stride;
    const double t = std::max(raw[1], 0.0) * stride;
    const double r = std::max(raw[2], 0.0) * stride;
    const double b = std::max(raw[3], 0.0) * stride;
    return {x - l, y - t, x + r, y + b};
}

int main() {
    const auto box = DecodeCellLtrb(12.0, 12.0, {1.0, 0.5, 2.0, 1.5}, 8.0);
    const double score = std::sqrt(StableSigmoid(0.0) * StableSigmoid(0.0));
    assert(std::abs(box[0] - 4.0) < 1e-12);
    assert(std::abs(box[1] - 8.0) < 1e-12);
    assert(std::abs(box[2] - 28.0) < 1e-12);
    assert(std::abs(box[3] - 24.0) < 1e-12);
    assert(std::abs(score - 0.5) < 1e-12);
    std::cout << std::fixed << std::setprecision(6)
              << box[0] << " " << box[1] << " "
              << box[2] << " " << box[3] << " " << score << "\n";
}
```

예상 출력은 `4.000000 8.000000 28.000000 24.000000 0.500000`이다.

## 9. C# decoder golden

다음 코드는 `System`만 사용하는 **실행 가능한 C# 예제**다.

```csharp
using System;
using System.Globalization;

public static class FcosGolden
{
    static double StableSigmoid(double z)
    {
        if (z >= 0.0) return 1.0 / (1.0 + Math.Exp(-z));
        double e = Math.Exp(z);
        return e / (1.0 + e);
    }

    static double[] DecodeCellLtrb(
        double x, double y, double[] raw, double stride)
    {
        double l = Math.Max(raw[0], 0.0) * stride;
        double t = Math.Max(raw[1], 0.0) * stride;
        double r = Math.Max(raw[2], 0.0) * stride;
        double b = Math.Max(raw[3], 0.0) * stride;
        return new[] { x - l, y - t, x + r, y + b };
    }

    static void Near(double actual, double expected)
    {
        if (Math.Abs(actual - expected) > 1e-12)
            throw new Exception("golden mismatch");
    }

    public static void Main()
    {
        double[] box = DecodeCellLtrb(
            12.0, 12.0, new[] { 1.0, 0.5, 2.0, 1.5 }, 8.0);
        double score = Math.Sqrt(StableSigmoid(0.0) * StableSigmoid(0.0));
        Near(box[0], 4.0);
        Near(box[1], 8.0);
        Near(box[2], 28.0);
        Near(box[3], 24.0);
        Near(score, 0.5);
        Console.WriteLine(
            string.Format(
                CultureInfo.InvariantCulture,
                "{0:F6} {1:F6} {2:F6} {3:F6} {4:F6}",
                box[0], box[1], box[2], box[3], score));
    }
}
```

예상 출력은 C++과 동일하다. `InvariantCulture`를 쓰지 않으면 일부 locale에서 소수점이 쉼표로 출력되어 golden text 비교가 실패할 수 있다.

## 10. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch 학습 | ONNX tensor | C++ 소비자 | C# 소비자 |
| --- | --- | --- | --- | --- |
| image | `float32/16` `NCHW` | `images` `NCHW` | contiguous buffer | `DenseTensor<float>` 등 |
| class raw | `[N,K,H,W]` logit | `cls_p3` 등 | shape를 먼저 검증 | output name으로 조회 |
| box raw | `[N,4,H,W]` | `box_p3` 등 | `LTRB`, unit 확인 | `float`에서 decode |
| center raw | `[N,1,H,W]` logit | `ctr_p3` 등 | sigmoid 책임 확인 | sigmoid 중복 금지 |
| point | Python 생성 또는 constant | 보통 graph 밖 | double golden, float runtime | 동일 offset·stride |
| score | FP32 sigmoid·sqrt | graph 안/밖 중 하나 | stable sigmoid | `Math.Exp` 분기 |
| output box | padded `XYXY` FP32 | decoder 선택에 따라 다름 | actual scale로 역변환 | 원본 크기로 clip |
| indices | `int64`가 흔함 | `int64` | `int64_t` | `long` |

다음 항목은 bundle manifest에서 최소한 고정한다.

```json
{
  "schema_version": 1,
  "input_layout": "NCHW",
  "input_color": "RGB",
  "levels": ["p3", "p4", "p5", "p6", "p7"],
  "strides": [8, 16, 32, 64, 128],
  "point_offset": 0.5,
  "box_order": "LTRB",
  "regression_unit": "feature_cell",
  "regression_activation": "relu",
  "score_fusion": "sqrt(sigmoid(cls)*sigmoid(ctr))",
  "coordinate_frame": "padded_input_xyxy_half_open"
}
```

JSON은 설명용 manifest 예시다. 실제 bundle에는 모델 SHA, preprocessing revision, class map, threshold table, per-level cap, NMS IoU, runtime/provider version도 넣는다.

## 11. 테스트와 디버깅 전략

### 11.1 계약 테스트

배포 전 다음 테스트를 자동화한다.

1. `point_offset`: 첫 point가 모든 runtime에서 `(s/2,s/2)`인지 검사한다.
2. `flatten_order`: 비정사각 `2x3` index-coded tensor가 `y,x` 행 우선인지 검사한다.
3. `distance_unit`: cell 값 1이 stride만큼 이동하는지 검사한다.
4. `box_order`: 비대칭 `1,0.5,2,1.5`가 정확한 네 변으로 가는지 검사한다.
5. `padding_mask`: content 오른쪽 경계 위 point가 invalid인지 검사한다.
6. `score_fusion`: 두 logit 0의 score가 0.5인지 검사한다.
7. `top_k`: 동점 score에서 `(level,point,class)` 순으로 결정적인지 검사한다.
8. `inverse_geometry`: 실제 정수 resize 비율로 원본 박스가 복원되는지 검사한다.
9. `empty_result`: 후보 0개일 때 shape `[0,4]`, `[0]`, `[0]`을 반환하는지 검사한다.
10. `runtime_parity`: 고정 input에서 box 최대 오차, score 최대 오차, 최종 index를 비교한다.

### 11.2 증상에서 원인으로 좁히기

| 증상 | 우선 확인할 계약 | 구분용 실험 |
| --- | --- | --- |
| level이 높을수록 박스 크기 오류 증가 | stride·회귀 단위 | 모든 level에 raw distance 1 주입 |
| 모든 박스가 일정 거리 이동 | point offset·padding origin | raw distance 0 golden |
| 좌우와 상하가 뒤바뀜 | `LTRB` order·`x,y` order | 네 값이 다른 비대칭 box |
| 영상 테두리 가짜 박스 증가 | valid-point mask | padding logit만 크게 만든 fixture |
| 후보 수와 p99가 동시 급증 | sigmoid 중복·sqrt·threshold | 단계별 funnel count 비교 |
| Python만 정상 | output name·layout·dtype | raw tensor checksum부터 비교 |
| 작은 객체 recall만 하락 | P3 cap·threshold·resize | level·size bucket recall 비교 |

최종 box만 비교하면 문제 지점을 찾기 어렵다. `raw tensor -> valid count -> threshold count -> top-k count -> NMS count`를 같은 request trace ID로 남긴다.

## 12. 성능·메모리·수치 안정성

### 12.1 decode를 늦춘다

class가 $K$개여도 box는 point당 하나다. 모든 point-class 조합에 box를 복제하지 않는다. 먼저 score 후보의 `(point,class)` index를 뽑고 선택된 point의 box만 decode한다.

### 12.2 level별 연산을 유지한다

모든 level을 큰 tensor로 concat하면 peak memory와 동기화 비용이 커질 수 있다. level마다 mask·threshold·top-k를 적용한 뒤 작은 후보만 합친다. level cap은 latency 격리 장치이자 recall trade-off이므로 configuration revision으로 관리한다.

### 12.3 FP16 경계

- sigmoid와 `sqrt(p_c p_{ctr})`는 FP32로 승격한다.
- `cell_exp`라면 exp 입력을 학습 계약에 맞는 범위로 clamp한다.
- 큰 좌표와 distance 산술은 FP32로 수행한다.
- threshold 근처 표본은 FP16·FP32·INT8 admission flip rate를 측정한다.
- NMS IoU의 면적 계산에서 음수 폭·높이를 0으로 clamp한다.

### 12.4 batch와 동적 shape

서로 다른 원본 크기를 같은 padded batch로 묶으면 image마다 content rectangle이 다르다. point grid는 공유해도 valid mask는 `[N,Q]`여야 한다. 동적 shape별로 point를 매 요청 재생성하지 말고 `(H_l,W_l,s_l,o,device,dtype)` key로 cache할 수 있다. cache hit rate와 memory upper bound를 함께 감시한다.

## 13. 실무 실패 사례: padding 후보가 NMS를 점령하다

### 상황

학습 서비스는 batch padding 영역을 target에서 ignore했다. 새 C++ 추론기는 단일 고정 크기 입력을 위해 오른쪽과 아래를 크게 padding했지만 valid-point mask를 구현하지 않았다.

### 관측

- 전체 AP는 대표 검증 세트에서 0.3만 감소했다.
- 세로로 긴 실제 트래픽에서 border false positive가 4배 증가했다.
- P3 threshold 통과 후보가 1,800개에서 14,000개로 늘었다.
- GPU inference p95는 비슷했지만 CPU NMS p99가 2.4배 증가했다.
- clip 뒤 많은 padding box가 오른쪽·아래 테두리에 같은 좌표로 겹쳤다.

### 잘못된 대응

전역 score threshold만 올리면 후보 수는 줄지만 작은 객체 recall도 함께 떨어진다. 모델을 재학습해도 decoder가 padding point를 허용하면 원인이 남는다.

### 근본 원인과 수정

전처리기는 content rectangle을 계산했지만 decoder ABI로 전달하지 않았다. 다음을 하나의 수정으로 배포한다.

1. image별 `content_xyxy`를 decoder input metadata에 포함한다.
2. sigmoid와 threshold 전에 invalid point를 제거한다.
3. `invalid_point_candidate_count`를 반드시 0으로 assert한다.
4. 세로·가로 극단 aspect ratio를 canary fixture에 추가한다.
5. model만이 아니라 preprocess·decoder·threshold bundle 전체를 rollback 단위로 지정한다.

## 14. ONNX·배포·모니터링 계약

### 14.1 export 선택

두 가지 배포 형태가 가능하다.

- raw export: graph가 class, box, center logits만 반환하고 소비자가 point·decode·NMS를 담당한다.
- fused export: graph가 decode와 일부 top-k까지 포함하고 runtime 밖에서 NMS만 수행한다.

raw export는 디버깅과 runtime 교체가 쉽지만 소비자 중복 구현 위험이 크다. fused export는 계약 수를 줄이지만 동적 shape·custom NMS·provider 호환성이 어려울 수 있다. 어느 쪽이든 activation을 graph와 consumer 양쪽에서 중복 수행하지 않는다.

### 14.2 release gate

release candidate는 다음 순서를 통과해야 한다.

```text
artifact hash -> manifest schema -> output names/shapes
              -> synthetic golden -> real golden batch
              -> provider parity -> load/soak test
              -> shadow traffic -> canary -> gradual rollout
```

권장 비교 항목은 다음과 같다.

- raw logit 최대·평균 절대 오차
- decoded padded box와 original box 최대 오차
- threshold admission 일치율
- level별 후보 수와 최종 detection 수
- class별 NMS keep index 일치율
- batch size·input profile별 peak memory와 p50·p95·p99

### 14.3 운영 metric

| 단계 | metric | 경보 예시 |
| --- | --- | --- |
| preprocess | resize/pad 비율, aspect bucket | 새 비율 bucket 급증 |
| raw head | level별 logit quantile, NaN/Inf | finite 위반 즉시 차단 |
| valid mask | valid point ratio | 학습 기준 band 이탈 |
| threshold | level·class별 후보 수 | p99가 예산 초과 |
| top-k | cap hit rate | P3 cap hit 지속 상승 |
| NMS | input/output 수, 소요 시간 | p99 SLO 초과 |
| response | size bucket별 detection 수 | small-object 급락 |
| quality | precision/recall, calibration proxy | 승인 baseline 이탈 |

분포 metric에는 model revision, decoder revision, threshold revision, runtime provider, input profile label을 붙인다. label cardinality가 무한히 늘지 않도록 request ID나 원본 크기 자체를 metric label로 쓰지는 않는다.

### 14.4 rollback 단위

다음 artifact는 서로 독립적으로 바꾸지 않는다.

```text
preprocess + model + class map + point/stride schema
+ regression transform + score fusion + threshold/cap
+ NMS + runtime/provider + golden fingerprints
```

bundle ID로 atomic하게 승격하고 이전 bundle을 즉시 재활성화할 수 있어야 한다. 모델 파일만 rollback하면 새 decoder와 옛 output 계약이 섞일 수 있다.

## 15. 운영 체크리스트

### 모델·기하

- [ ] level 이름과 순서가 학습·export·serving에서 같다.
- [ ] stride와 point offset이 manifest와 golden test로 고정됐다.
- [ ] box channel 순서와 regression unit·activation이 명시됐다.
- [ ] resize의 실제 $a_x,a_y$와 padding origin을 요청별로 보존한다.
- [ ] image별 valid-point mask를 threshold 전에 적용한다.

### 수치·후처리

- [ ] raw output이 logit인지 probability인지 중복 activation 없이 확인했다.
- [ ] score fusion이 product인지 square root인지 threshold와 함께 versioning됐다.
- [ ] sigmoid·score·decode·IoU가 필요한 구간에서 FP32다.
- [ ] level별 threshold·top-k와 tie-break가 결정적이다.
- [ ] empty result와 모든 후보 cap hit를 테스트했다.

### 릴리스·운영

- [ ] Python·C++·C# golden output이 허용 오차 안에서 같다.
- [ ] ONNX provider별 raw·decoded·final parity를 확인했다.
- [ ] extreme aspect ratio와 dynamic batch를 load test했다.
- [ ] candidate funnel과 NMS p99를 canary에서 관찰한다.
- [ ] immutable bundle과 atomic rollback이 준비됐다.

## 16. 연습문제

### 문제 1

stride 16, offset 0.5인 feature에서 $(i,j)=(2,3)$의 point를 구하라. consumer가 offset 0을 썼을 때 이동량도 구하라.

### 문제 2

point가 $(40,56)$이고 raw `LTRB`가 `(1,2,3,4)`다. `cell_relu` 계약에서 padded-input box를 구하라. 같은 값을 `pixel_relu`로 잘못 읽은 box도 구하라.

### 문제 3

class logit과 center logit이 모두 0이다. product score와 square-root product score를 각각 구하라. threshold가 0.4이면 두 계약의 admission 결과는 같은가?

### 문제 4

원본 크기 $W_0=101$, 요청 scale 2.5, 실제 정수 resize 폭 $W_r=252$, 왼쪽 padding $p_x=10$이다. padded 좌표 $x=136$을 원본으로 복원하라. 요청 scale 2.5를 그대로 쓴 결과와 비교하라.

### 문제 5

P3, P4, P5의 threshold 통과 후보가 각각 9000, 800, 120개이고 cap은 1000, 500, 200개다. 최종 pre-NMS 후보 수와 P3 cap hit ratio $R_3/C_3$를 구하라.

### 문제 6

padding 영역의 높은 score를 decode한 뒤 원본 크기로 clip하면 왜 valid mask를 대체하지 못하는지 설명하라.

## 17. 해답

### 해답 1

$$
x=16(2+0.5)=40,\qquad y=16(3+0.5)=56
$$

offset 0 consumer는 $(32,48)$을 만들므로 두 축 모두 `-8` pixel 이동한다.

### 해답 2

`cell_relu` distance는 `(16,32,48,64)` pixel이다.

$$
(x_0,y_0,x_1,y_1)=(-8,24,88,120)
$$

`pixel_relu`로 잘못 읽으면 `(39,54,43,60)`이 된다. level stride가 커질수록 오류가 커진다.

### 해답 3

각 sigmoid는 0.5다.

$$
p_cp_{ctr}=0.25
$$

$$
\sqrt{p_cp_{ctr}}=0.5
$$

threshold 0.4에서 product는 탈락하고 square-root product는 통과한다. score 식과 threshold를 한 계약으로 배포해야 한다.

### 해답 4

실제 비율은 $a_x=252/101$이다.

$$
x^{orig}=\frac{136-10}{252/101}=50.5
$$

요청 scale 2.5를 쓰면 $126/2.5=50.4$다. 작은 차이도 반복 resize나 정수 좌표 변환과 결합하면 경계 IoU를 흔들 수 있다.

### 해답 5

$$
R=\min(9000,1000)+\min(800,500)+\min(120,200)=1620
$$

$$
\frac{R_3}{C_3}=\frac{1000}{9000}\approx0.1111
$$

P3 후보의 약 88.9%가 cap에서 잘리므로 작은 객체 recall을 별도로 점검해야 한다.

### 해답 6

clip은 좌표만 영상 경계 안으로 접는다. padding point의 class score와 후보 자격은 제거하지 않는다. 서로 다른 padding 박스가 같은 경계 박스로 겹쳐 NMS 비용과 false positive를 만들 수 있으므로 point 단계에서 mask해야 한다.

## 18. 핵심 요약

1. FCOS의 anchor-free는 무계약이 아니라 point·stride·distance 계약으로의 전환이다.
2. `point_offset`, level order, flatten order는 model artifact와 함께 versioning한다.
3. 회귀값의 단위와 activation을 모르면 올바른 `LTRB` decode가 불가능하다.
4. 실제 정수 resize 비율과 padding origin을 보존해야 원본 좌표를 정확히 복원한다.
5. padding point는 threshold 전에 제거해야 하며 decode 후 clip으로 대체할 수 없다.
6. product와 square-root product는 순위는 같아도 같은 threshold의 통과 집합은 다르다.
7. level별 threshold·top-k는 품질 파라미터이면서 latency·memory circuit breaker다.
8. Python·C++·C#·ONNX parity는 raw tensor, 후보 funnel, 최종 box의 세 단계에서 검사한다.
9. model, preprocess, decoder, threshold, NMS는 하나의 immutable rollback bundle이다.

## 다음 학습 예고

다음 Part는 같은 원본의 DETR를 실무 엔지니어 관점에서 다룬다. object query와 고정 cardinality output, Hungarian matching과 no-object class, padding attention mask, `CXCYWH` 좌표 ABI, 보조 decoder output, NMS 없는 serving의 중복 탐지 감시, ONNX·C++·C# parity를 하나의 운영 계약으로 연결한다.
