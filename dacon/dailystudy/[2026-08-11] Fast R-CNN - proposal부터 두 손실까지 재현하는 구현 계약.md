<!-- curriculum: cycle=2; level=implementation; source_index=5/18; source=02-04.Fast_R_CNN.md; part=1/1 -->

# Fast R-CNN: proposal부터 두 손실까지 재현하는 구현 계약

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-11 |
| 회차·수준 | 2회차 · 구현 (`implementation`) |
| 현재 소스 | 5/18 · `02-04.Fast_R_CNN.md` |
| Part | 1/1 |
| 이전 소스 | `02-03.SPP.md` |
| 다음 소스 | `02-05.YOLO.md` |

## 오늘의 질문

Fast R-CNN의 구조를 그림으로 설명하는 것과 실제로 학습시키는 것 사이에는 큰 간격이 있다. 구현자는 다음 질문에 답해야 한다.

- 이미지마다 개수가 다른 proposal을 한 tensor로 어떻게 묶는가?
- proposal과 정답 상자를 어떤 IoU 규칙으로 positive, negative, ignore에 배정하는가?
- background가 분류에는 참여하지만 box 회귀에는 참여하지 않게 만드는 방법은 무엇인가?
- 클래스별 box delta 중 정답 클래스의 네 값만 어떻게 선택하는가?
- 좌표 encode와 decode가 역함수인지, gradient가 의도한 slice에만 흐르는지 어떻게 증명하는가?
- 회귀 손실을 껐을 때와 켰을 때 정말 box 품질이 달라지는가?

오늘은 Fast R-CNN을 **proposal 계약, RoI 연산, 분류 손실, 클래스별 box 회귀 손실, 재현 가능한 평가**로 분해한다. 1회차 문서의 역사와 기본 RoI Pooling 직관을 반복하지 않고, 독립 수치 oracle과 실행 가능한 미니 학습 파이프라인으로 구현 공백을 채운다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

- proposal과 정답의 IoU 행렬에서 label과 matched target을 만든다.
- positive·negative 비율을 유지하면서 ignore proposal을 제외해 sampling한다.
- box transform의 encode와 decode를 구현하고 round trip을 검증한다.
- `torchvision.ops.roi_pool`을 포함한 미니 Fast R-CNN을 학습·평가한다.
- 분류와 회귀 gradient가 흐르는 parameter 및 RoI를 단위 테스트한다.
- 고정 seed로 ablation을 재실행하고 결과가 완전히 같은지 확인한다.
- Python, C++17, C#의 좌표·shape·layout·dtype 계약을 맞춘다.
- 성능, 메모리, 수치 안정성, ONNX 배포 실패를 진단한다.

## 선수 지식

- `NCHW` tensor와 합성곱 stride
- `(x1, y1, x2, y2)` 상자와 IoU
- softmax cross entropy
- Smooth L1 손실
- max pooling의 argmax와 역전파
- PyTorch의 `nn.Module`, optimizer, train·eval mode
- 1회차의 RoI Pooling 양자화와 클래스별 box 회귀

## 1. 원본과 1회차를 구현 관점에서 다시 읽기

원본 `02-04.Fast_R_CNN.md`에는 두 모델이 함께 들어 있다.

- 앞부분의 공유 특징 맵, 외부 proposal, RoI Pooling, 두 갈래 head는 Fast R-CNN이다.
- 후반의 anchor, RPN head, proposal 생성은 Faster R-CNN이다.

오늘 구현은 Fast R-CNN의 경계를 지키므로 proposal은 입력으로 받는다. RPN을 붙이지 않는다. 이 구분은 이름의 문제가 아니라 데이터와 손실의 책임 범위를 고정하는 문제다.

원본의 다음 내용도 구현 전에 교정해야 한다.

1. SPP-net의 합성곱 미세 조정이 어려웠던 이유는 max pooling이 미분 불가능해서가 아니다. 학습 절차와 메모리·계산 제약, multi-stage pipeline이 핵심이었다.
2. `floor(h/P)` 크기로 bin을 반복하면 나머지 행·열이 누락될 수 있다. 실제 RoI kernel의 양자화 규칙을 golden test로 고정해야 한다.
3. Fast R-CNN은 검출 network 내부에서 backbone까지 gradient가 흐르지만 외부 Selective Search까지 미분하는 완전한 시스템은 아니다.
4. 원문의 RPN loss는 유용한 확장 자료지만 Fast R-CNN head의 loss와 섞어 구현하면 안 된다.

1회차 문서는 RoI Pooling의 기본 경계, 손계산, 클래스별 box head, 배포 계약을 다뤘다. 이번에는 다음 새 내용을 추가한다.

- IoU 기반 target assignment와 ignore 구간
- 재현 가능한 balanced sampling
- encode·decode round trip
- background 회귀 차단과 클래스별 gradient routing
- 완전한 train·eval loop와 회귀 ablation
- 다중 언어 golden contract

## 2. 학습 표본의 데이터 계약

### 2.1 입력과 출력 기호

이미지 batch와 backbone feature를 다음처럼 둔다.

$$
I \in \mathbb{R}^{N \times C_{in} \times H \times W}
$$

$$
F = f_{\theta}(I) \in \mathbb{R}^{N \times C \times H_f \times W_f}
$$

batch 전체 proposal 수를 $R$이라 하자. RoI tensor는 다음 shape을 사용한다.

$$
B \in \mathbb{R}^{R \times 5}
$$

각 행은 다음 순서다.

```text
[batch_index, x1, y1, x2, y2]
```

`batch_index`는 정수 의미를 가지지만 `torchvision` API에서는 좌표와 같은 floating tensor 안에 들어간다. 나머지 네 좌표는 **입력 이미지 좌표계**다. backbone stride가 $S$이면 RoI 연산에 다음 scale을 전달한다.

$$
\operatorname{spatial\_scale}=\frac{1}{S}
$$

### 2.2 Tensor shape 추적

오늘의 실행 예제는 `N=4`, 입력 `16 x 16`, stride `4`, 이미지당 proposal 4개를 사용한다.

| 단계 | shape | dtype | 의미 |
| --- | --- | --- | --- |
| 이미지 | `[4, 1, 16, 16]` | `float32` | NCHW |
| backbone feature | `[4, 8, 4, 4]` | `float32` | stride 4 |
| RoI | `[16, 5]` | `float32` | batch index와 이미지 좌표 |
| RoI Pooling | `[16, 8, 2, 2]` | `float32` | proposal별 고정 격자 |
| flatten | `[16, 32]` | `float32` | head 입력 |
| hidden | `[16, 32]` | `float32` | 공유 FC 표현 |
| class logits | `[16, 3]` | `float32` | background와 객체 2종 |
| box deltas | `[16, 2, 4]` | `float32` | 객체 클래스별 네 delta |

분류 label은 다음 규칙을 사용한다.

```text
-1: ignore
 0: background
 1: foreground class 1
 2: foreground class 2
```

box delta의 class 축에는 background가 없다. 따라서 label $u>0$의 회귀 slice는 index $u-1$이다.

## 3. IoU와 target assignment

### 3.1 IoU 행렬

proposal $b_i$와 정답 $g_j$의 IoU를 다음처럼 정의한다.

$$
\operatorname{IoU}(b_i,g_j)
=
\frac{|b_i \cap g_j|}{|b_i|+|g_j|-|b_i \cap g_j|}
$$

$R$개 proposal과 $M$개 정답이 있으면 IoU 행렬 shape은 다음과 같다.

$$
Q \in \mathbb{R}^{R \times M}
$$

각 proposal은 가장 큰 IoU를 주는 정답에 잠정 배정한다.

$$
j_i^{*}=\operatorname*{argmax}_{j}Q_{ij}
$$

$$
q_i^{*}=Q_{i,j_i^{*}}
$$

### 3.2 Positive, negative, ignore

오늘 예제의 threshold는 다음과 같다.

$$
y_i=
\begin{cases}
c_{j_i^{*}}, & q_i^{*}\ge 0.5 \\
-1, & 0.3\le q_i^{*}<0.5 \\
0, & q_i^{*}<0.3
\end{cases}
$$

중간 IoU를 ignore하는 이유는 경계가 애매한 proposal을 억지로 background로 학습시키지 않기 위해서다. threshold는 데이터셋·구현별 설정값이며 보편 상수가 아니다.

실제 detector는 각 정답에 최소 한 개 positive를 보장하는 규칙, crowd·ignore annotation, 이미지 경계 clipping 같은 정책을 더할 수 있다. 그 정책은 checkpoint와 함께 versioning해야 한다.

### 3.3 수작업 검증

proposal과 정답을 다음처럼 두자.

$$
b=(0,0,10,10)
$$

$$
g=(1,1,9,9)
$$

반열린 좌표계에서 proposal 면적은 $100$, 정답 면적은 $64$, 교집합은 $64$다.

$$
\operatorname{IoU}(b,g)=\frac{64}{100+64-64}=0.64
$$

따라서 positive threshold가 $0.5$이면 이 proposal은 positive다.

## 4. Box transform의 encode와 decode

proposal 중심과 크기를 $(p_x,p_y,p_w,p_h)$, matched 정답을 $(g_x,g_y,g_w,g_h)$라 하자. 학습 target은 다음과 같다.

$$
t_x=\frac{g_x-p_x}{p_w}
$$

$$
t_y=\frac{g_y-p_y}{p_h}
$$

$$
t_w=\log\left(\frac{g_w}{p_w}\right)
$$

$$
t_h=\log\left(\frac{g_h}{p_h}\right)
$$

예측 delta $d=(d_x,d_y,d_w,d_h)$를 proposal에 적용하는 decode는 다음과 같다.

$$
\widehat{g}_x=p_x+d_xp_w
$$

$$
\widehat{g}_y=p_y+d_yp_h
$$

$$
\widehat{g}_w=p_w\exp(d_w)
$$

$$
\widehat{g}_h=p_h\exp(d_h)
$$

corner 좌표는 다음처럼 복원한다.

$$
\widehat{x}_1=\widehat{g}_x-\frac{\widehat{g}_w}{2},\qquad
\widehat{x}_2=\widehat{g}_x+\frac{\widehat{g}_w}{2}
$$

$$
\widehat{y}_1=\widehat{g}_y-\frac{\widehat{g}_h}{2},\qquad
\widehat{y}_2=\widehat{g}_y+\frac{\widehat{g}_h}{2}
$$

정확한 구현이면 다음 round trip이 성립해야 한다.

$$
\operatorname{decode}(\operatorname{encode}(g,b),b)\approx g
$$

`exp` overflow를 막기 위해 배포 구현은 $d_w,d_h$의 상한을 둔다. 오늘 Python 구현은 다음 값을 사용한다.

$$
d_{max}=\log\left(\frac{1000}{16}\right)
$$

이는 수치 안전장치이지 학습 target을 임의로 바꾸라는 뜻은 아니다.

## 5. 분류와 클래스별 회귀 손실

### 5.1 분류 손실

분류 logits을 $s\in\mathbb{R}^{R_s\times(K+1)}$라 하자. sampling 뒤 남은 RoI 수가 $R_s$다. 분류 손실은 background를 포함한다.

$$
L_{cls}=\frac{1}{R_s}\sum_{i=1}^{R_s}-\log p_{i,y_i}
$$

여기서 $p=\operatorname{softmax}(s)$이고 $y_i\in\{0,1,\ldots,K\}$다. ignore label은 sampling 전에 제거한다.

### 5.2 회귀 손실

box head 출력은 다음 shape이다.

$$
D\in\mathbb{R}^{R_s\times K\times4}
$$

positive RoI만 회귀에 참여한다.

$$
L_{box}
=
\frac{1}{R_s}
\sum_{i:y_i>0}
\sum_{m\in\{x,y,w,h\}}
\operatorname{smooth}_{L1}(D_{i,y_i-1,m}-T_{i,m})
$$

오늘은 beta가 1인 Smooth L1을 사용한다.

$$
\operatorname{smooth}_{L1}(z)=
\begin{cases}
\frac{1}{2}z^2, & |z|<1 \\
|z|-\frac{1}{2}, & |z|\ge 1
\end{cases}
$$

전체 손실은 다음과 같다.

$$
L=L_{cls}+\lambda L_{box}
$$

background는 $L_{cls}$에는 참여하지만 $L_{box}$에는 참여하지 않는다. 또한 positive label이 2라면 box tensor의 class index 1만 gradient를 받아야 한다.

### 5.3 Positive가 없는 batch

positive가 하나도 없을 때 상수 `torch.tensor(0.0)`을 새로 만들면 box head와의 graph 연결이 끊어진다. 다음처럼 graph에 연결된 0을 만든다.

```python
regression_loss = box_deltas.sum() * 0.0
```

이 값은 0이지만 backward가 안전하고 device·dtype도 box tensor와 같다.

## 6. 독립 NumPy oracle

다음 코드는 **실행 가능**하다. IoU, encode, decode를 framework detection operator 없이 검증한다.

```python
import numpy as np


def iou_numpy(a, b):
    top_left = np.maximum(a[:2], b[:2])
    bottom_right = np.minimum(a[2:], b[2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    inter = wh[0] * wh[1]
    area_a = np.prod(a[2:] - a[:2])
    area_b = np.prod(b[2:] - b[:2])
    return inter / (area_a + area_b - inter)


def encode_numpy(gt, proposal):
    pw, ph = proposal[2:] - proposal[:2]
    px, py = proposal[:2] + 0.5 * np.array([pw, ph])
    gw, gh = gt[2:] - gt[:2]
    gx, gy = gt[:2] + 0.5 * np.array([gw, gh])
    return np.array([
        (gx - px) / pw,
        (gy - py) / ph,
        np.log(gw / pw),
        np.log(gh / ph),
    ])


def decode_numpy(delta, proposal):
    pw, ph = proposal[2:] - proposal[:2]
    px, py = proposal[:2] + 0.5 * np.array([pw, ph])
    gx = px + delta[0] * pw
    gy = py + delta[1] * ph
    gw = pw * np.exp(delta[2])
    gh = ph * np.exp(delta[3])
    return np.array([
        gx - 0.5 * gw,
        gy - 0.5 * gh,
        gx + 0.5 * gw,
        gy + 0.5 * gh,
    ])


proposal = np.array([0.0, 0.0, 10.0, 10.0])
gt = np.array([1.0, 1.0, 9.0, 9.0])
assert abs(iou_numpy(proposal, gt) - 0.64) < 1e-12
delta = encode_numpy(gt, proposal)
np.testing.assert_allclose(
    delta,
    [0.0, 0.0, np.log(0.8), np.log(0.8)],
    rtol=0,
    atol=1e-12,
)
np.testing.assert_allclose(decode_numpy(delta, proposal), gt, rtol=0, atol=1e-12)
print("IoU=0.64, round trip: PASS")
```

이 oracle은 반열린 corner 좌표를 사용한다. inclusive 좌표로 면적에 `+1`을 넣는 C++ 구현과 섞으면 IoU와 target이 달라진다.

## 7. 실행 가능한 PyTorch 핵심 구현

다음 코드는 **실행 가능**하며 오늘의 통합 학습 예제에서 그대로 사용한다.

```python
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_pool


def box_iou(boxes1, boxes2):
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-12)


def encode_boxes(gt, proposals):
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gx = gt[:, 0] + 0.5 * gw
    gy = gt[:, 1] + 0.5 * gh
    invalid = (pw <= 0).any() or (ph <= 0).any() or (gw <= 0).any() or (gh <= 0).any()
    if bool(invalid):
        raise ValueError("boxes must have positive width and height")
    return torch.stack(
        ((gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)),
        dim=1,
    )


def decode_boxes(deltas, proposals):
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]
    px = proposals[:, 0] + 0.5 * pw
    py = proposals[:, 1] + 0.5 * ph
    dx, dy, dw, dh = deltas.unbind(dim=1)
    maximum = math.log(1000.0 / 16.0)
    dw = dw.clamp(max=maximum)
    dh = dh.clamp(max=maximum)
    gx = px + dx * pw
    gy = py + dy * ph
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack(
        (gx - 0.5 * gw, gy - 0.5 * gh, gx + 0.5 * gw, gy + 0.5 * gh),
        dim=1,
    )


def assign_targets(
    proposals,
    gt_boxes,
    gt_labels,
    positive_iou=0.5,
    negative_iou=0.3,
):
    ious = box_iou(proposals, gt_boxes)
    best_iou, matched = ious.max(dim=1)
    labels = gt_labels[matched].clone()
    labels[best_iou < positive_iou] = -1
    labels[best_iou < negative_iou] = 0
    targets = encode_boxes(gt_boxes[matched], proposals)
    return labels, targets, best_iou


def sample_rois(labels, batch_size, positive_fraction, generator):
    positive = torch.where(labels > 0)[0]
    negative = torch.where(labels == 0)[0]
    wanted_pos = min(int(batch_size * positive_fraction), positive.numel())
    wanted_neg = min(batch_size - wanted_pos, negative.numel())
    positive_order = torch.randperm(positive.numel(), generator=generator)
    negative_order = torch.randperm(negative.numel(), generator=generator)
    positive = positive[positive_order[:wanted_pos]]
    negative = negative[negative_order[:wanted_neg]]
    return torch.cat((positive, negative))


def fast_rcnn_loss(class_logits, box_deltas, labels, regression_targets):
    classification = F.cross_entropy(class_logits, labels)
    positive = torch.where(labels > 0)[0]
    if positive.numel() == 0:
        regression = box_deltas.sum() * 0.0
    else:
        class_index = labels[positive] - 1
        selected = box_deltas[positive, class_index]
        regression = F.smooth_l1_loss(
            selected,
            regression_targets[positive],
            beta=1.0,
            reduction="sum",
        ) / labels.numel()
    return classification, regression


class MiniFastRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(nn.Linear(8 * 2 * 2, 32), nn.ReLU())
        self.classifier = nn.Linear(32, num_classes + 1)
        self.box_regressor = nn.Linear(32, num_classes * 4)
        self.num_classes = num_classes

    def forward(self, images, rois):
        feature = self.backbone(images)
        pooled = roi_pool(
            feature,
            rois,
            output_size=(2, 2),
            spatial_scale=0.25,
        )
        hidden = self.fc(pooled.flatten(start_dim=1))
        logits = self.classifier(hidden)
        deltas = self.box_regressor(hidden).reshape(-1, self.num_classes, 4)
        return logits, deltas, feature, pooled
```

### 7.1 왜 RoI tensor에 batch index가 필요한가

이미지마다 proposal 수가 다르다. proposal을 batch 전체에서 concatenate하면 각 행이 어느 feature map을 읽을지 잃는다. 첫 열의 `batch_index`가 이 관계를 복구한다.

```python
rois = []
for batch_index, boxes in enumerate(proposals_per_image):
    column = torch.full((len(boxes), 1), float(batch_index))
    rois.append(torch.cat((column, boxes), dim=1))
rois = torch.cat(rois, dim=0)
```

이미지 순서를 shuffle했는데 proposal의 batch index를 다시 만들지 않으면, shape은 맞아도 다른 이미지의 feature를 읽는 치명적인 silent bug가 생긴다.

### 7.2 Sampling의 재현성

global RNG에 기대지 않고 전용 generator를 전달한다.

```python
generator = torch.Generator().manual_seed(20260811)
selected = sample_rois(
    labels,
    batch_size=128,
    positive_fraction=0.25,
    generator=generator,
)
```

동일 seed를 매 epoch 재사용하면 매번 같은 proposal만 뽑을 수 있다. 실전에서는 run seed에서 epoch·worker·image index를 조합해 서로 다르면서 재현 가능한 seed를 만든다.

## 8. Gradient routing 단위 테스트

다음 코드는 **실행 가능**하다. label 2인 positive는 class index 1의 box delta만 갱신하고 background는 어느 box delta도 갱신하지 않는지 확인한다.

```python
import numpy as np
import torch


proposals = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [20.0, 20.0, 30.0, 30.0],
])
gt = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
labels, targets, ious = assign_targets(proposals, gt, torch.tensor([2]))

assert labels.tolist() == [2, 0]
np.testing.assert_allclose(ious.numpy(), [0.64, 0.0], rtol=0, atol=1e-7)
torch.testing.assert_close(decode_boxes(targets[:1], proposals[:1]), gt)

logits = torch.zeros(2, 3, requires_grad=True)
deltas = torch.zeros(2, 2, 4, requires_grad=True)
cls_loss, box_loss = fast_rcnn_loss(logits, deltas, labels, targets)
(cls_loss + box_loss).backward()

assert deltas.grad is not None
assert torch.count_nonzero(deltas.grad[1]) == 0
assert torch.count_nonzero(deltas.grad[0, 0]) == 0
assert torch.count_nonzero(deltas.grad[0, 1]) > 0
print("target, round trip, gradient routing: PASS")
```

이 테스트는 단순히 loss가 finite인지 확인하는 것보다 강하다. 잘못된 class offset, background 회귀, 모든 클래스 동시 회귀를 직접 잡는다.

## 9. 완전한 학습·평가와 ablation

### 9.1 실험 질문

동일한 합성 데이터와 초기화에서 box loss weight만 바꾼다.

| 실험 | $\lambda$ | 확인 질문 |
| --- | ---: | --- |
| 분류만 | 0 | class accuracy는 높아도 box가 개선되지 않는가? |
| 분류와 회귀 | 1 | class accuracy를 유지하며 positive box IoU가 높아지는가? |

이미지는 두 종류의 줄무늬 사각형을 포함한다. 각 이미지에는 positive proposal 2개와 멀리 떨어진 negative proposal 2개가 있다. 첫 8장을 학습, 뒤 4장을 평가에 사용한다.

### 9.2 실행 가능한 dataset·train·eval 코드

다음 코드는 **실행 가능**하다. 바로 앞의 `MiniFastRCNN`, target 함수, loss 함수가 같은 파일에 정의되어 있다고 가정한다.

```python
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_dataset():
    images = []
    all_proposals = []
    all_labels = []
    all_targets = []
    for i in range(12):
        cls = 1 + i % 2
        x1 = 2 + (i * 3) % 5
        y1 = 3 + (i * 2) % 4
        gt = torch.tensor([[x1, y1, x1 + 6, y1 + 6]], dtype=torch.float32)
        image = torch.zeros(1, 16, 16)
        if cls == 1:
            image[0, y1:y1 + 6, x1:x1 + 6] = 1.0
            image[0, y1:y1 + 6:2, x1:x1 + 6] = 1.5
        else:
            image[0, y1:y1 + 6, x1:x1 + 6] = 2.0
            image[0, y1:y1 + 6, x1:x1 + 6:2] = 2.5

        proposals = torch.tensor([
            [x1 - 1, y1, x1 + 6, y1 + 7],
            [x1, y1 - 1, x1 + 7, y1 + 6],
            [0, 0, 4, 4],
            [11, 11, 15, 15],
        ], dtype=torch.float32)
        labels, targets, _ = assign_targets(
            proposals,
            gt,
            torch.tensor([cls]),
            positive_iou=0.5,
            negative_iou=0.3,
        )
        assert (labels >= 0).all()
        images.append(image)
        all_proposals.append(proposals)
        all_labels.append(labels)
        all_targets.append(targets)
    return torch.stack(images), all_proposals, all_labels, all_targets


def flatten_batch(images, proposals, labels, targets):
    rois = []
    for batch_index, boxes in enumerate(proposals):
        column = torch.full((len(boxes), 1), float(batch_index))
        rois.append(torch.cat((column, boxes), dim=1))
    return images, torch.cat(rois), torch.cat(labels), torch.cat(targets)


def train_and_evaluate(box_loss_weight):
    seed_everything(20260811)
    images, proposals, labels, targets = make_dataset()
    train = flatten_batch(images[:8], proposals[:8], labels[:8], targets[:8])
    test = flatten_batch(images[8:], proposals[8:], labels[8:], targets[8:])

    model = MiniFastRCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(80):
        model.train()
        logits, deltas, _, _ = model(train[0], train[1])
        cls_loss, reg_loss = fast_rcnn_loss(logits, deltas, train[2], train[3])
        loss = cls_loss + box_loss_weight * reg_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, deltas, _, pooled = model(test[0], test[1])
        accuracy = (logits.argmax(1) == test[2]).float().mean().item()
        positive = torch.where(test[2] > 0)[0]
        chosen = deltas[positive, test[2][positive] - 1]
        decoded = decode_boxes(chosen, test[1][positive, 1:])
        gt = decode_boxes(test[3][positive], test[1][positive, 1:])
        mean_iou = box_iou(decoded, gt).diag().mean().item()

    return {
        "accuracy": round(accuracy, 6),
        "mean_iou": round(mean_iou, 6),
        "pooled_shape": tuple(pooled.shape),
    }


without_regression = train_and_evaluate(0.0)
with_regression = train_and_evaluate(1.0)
repeated = train_and_evaluate(1.0)
assert with_regression == repeated
print("box_loss_weight=0:", without_regression)
print("box_loss_weight=1:", with_regression)
print("reproducibility: exact match")
```

오늘 로컬 CPU 실행 결과는 다음과 같다.

```text
box_loss_weight=0: {'accuracy': 1.0, 'mean_iou': 0.0, 'pooled_shape': (16, 8, 2, 2)}
box_loss_weight=1: {'accuracy': 1.0, 'mean_iou': 0.82943, 'pooled_shape': (16, 8, 2, 2)}
reproducibility: exact match
```

### 9.3 결과 해석

분류만 학습한 모델도 작은 합성 데이터의 class는 맞혔다. 그러나 사용하지 않은 box head의 임의 delta를 decode하므로 예측 box가 무너져 mean IoU가 0이 되었다. 회귀 손실을 켜면 class accuracy를 유지하면서 mean IoU가 약 `0.82943`으로 높아졌다.

이 결과는 Fast R-CNN의 성능 benchmark가 아니다. 다음 한 가지만 격리하는 executable ablation이다.

> 분류 정확도만으로는 localization head가 학습되었는지 알 수 없다.

두 번째 실행의 결과 dictionary가 첫 실행과 정확히 같으므로 오늘 CPU 환경에서 seed·deterministic 설정이 유효함도 확인했다.

### 9.4 실전 평가에서 더 필요한 것

- 여러 IoU threshold의 AP
- class별 precision·recall
- proposal recall 상한
- decode 뒤 clipping과 class별 NMS
- 작은·중간·큰 객체별 성능
- 고정 validation split과 dataset hash
- seed 여러 개의 평균과 편차

Fast R-CNN head는 주어진 proposal 밖의 객체를 복구할 수 없다. proposal recall이 낮으면 head ablation을 해석하기 전에 입력 상한부터 측정해야 한다.

## 10. C++17 golden contract

다음 코드는 **실행 가능**하며 반열린 좌표계의 IoU와 encode·decode를 검증한다. deep learning runtime을 재구현하지 않고 언어 경계에서 가장 자주 깨지는 box 수학만 고정한다.

```cpp
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

using Box = std::array<double, 4>;

double area(const Box& b) {
    return std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
}

double iou(const Box& a, const Box& b) {
    Box intersection{
        std::max(a[0], b[0]),
        std::max(a[1], b[1]),
        std::min(a[2], b[2]),
        std::min(a[3], b[3]),
    };
    const double overlap = area(intersection);
    return overlap / (area(a) + area(b) - overlap);
}

Box encode(const Box& gt, const Box& p) {
    const double pw = p[2] - p[0];
    const double ph = p[3] - p[1];
    const double px = p[0] + 0.5 * pw;
    const double py = p[1] + 0.5 * ph;
    const double gw = gt[2] - gt[0];
    const double gh = gt[3] - gt[1];
    const double gx = gt[0] + 0.5 * gw;
    const double gy = gt[1] + 0.5 * gh;
    return {(gx - px) / pw, (gy - py) / ph, std::log(gw / pw), std::log(gh / ph)};
}

Box decode(const Box& d, const Box& p) {
    const double pw = p[2] - p[0];
    const double ph = p[3] - p[1];
    const double px = p[0] + 0.5 * pw;
    const double py = p[1] + 0.5 * ph;
    const double gx = px + d[0] * pw;
    const double gy = py + d[1] * ph;
    const double gw = pw * std::exp(d[2]);
    const double gh = ph * std::exp(d[3]);
    return {gx - 0.5 * gw, gy - 0.5 * gh, gx + 0.5 * gw, gy + 0.5 * gh};
}

int main() {
    const Box proposal{0.0, 0.0, 10.0, 10.0};
    const Box gt{1.0, 1.0, 9.0, 9.0};
    const Box delta = encode(gt, proposal);
    const Box recovered = decode(delta, proposal);
    assert(std::abs(iou(proposal, gt) - 0.64) < 1e-12);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(recovered[i] - gt[i]) < 1e-12);
    }
    std::cout << std::fixed << std::setprecision(6)
              << "iou=" << iou(proposal, gt)
              << " tw=" << delta[2]
              << " recovered=" << recovered[0] << "," << recovered[2] << "\n";
}
```

예상 출력은 다음과 같다.

```text
iou=0.640000 tw=-0.223144 recovered=1.000000,9.000000
```

## 11. C# golden contract

다음 코드는 **실행 가능**하며 C++과 같은 계약을 검증한다.

```csharp
using System;

public static class FastRcnnContract
{
    static double Area(double[] b)
    {
        return Math.Max(0.0, b[2] - b[0]) * Math.Max(0.0, b[3] - b[1]);
    }

    static double IoU(double[] a, double[] b)
    {
        var intersection = new[] {
            Math.Max(a[0], b[0]),
            Math.Max(a[1], b[1]),
            Math.Min(a[2], b[2]),
            Math.Min(a[3], b[3]),
        };
        double overlap = Area(intersection);
        return overlap / (Area(a) + Area(b) - overlap);
    }

    static double[] Encode(double[] gt, double[] p)
    {
        double pw = p[2] - p[0], ph = p[3] - p[1];
        double px = p[0] + 0.5 * pw, py = p[1] + 0.5 * ph;
        double gw = gt[2] - gt[0], gh = gt[3] - gt[1];
        double gx = gt[0] + 0.5 * gw, gy = gt[1] + 0.5 * gh;
        return new[] {
            (gx - px) / pw,
            (gy - py) / ph,
            Math.Log(gw / pw),
            Math.Log(gh / ph),
        };
    }

    static double[] Decode(double[] d, double[] p)
    {
        double pw = p[2] - p[0], ph = p[3] - p[1];
        double px = p[0] + 0.5 * pw, py = p[1] + 0.5 * ph;
        double gx = px + d[0] * pw, gy = py + d[1] * ph;
        double gw = pw * Math.Exp(d[2]), gh = ph * Math.Exp(d[3]);
        return new[] {
            gx - 0.5 * gw,
            gy - 0.5 * gh,
            gx + 0.5 * gw,
            gy + 0.5 * gh,
        };
    }

    public static void Main()
    {
        var proposal = new[] { 0.0, 0.0, 10.0, 10.0 };
        var gt = new[] { 1.0, 1.0, 9.0, 9.0 };
        var delta = Encode(gt, proposal);
        var recovered = Decode(delta, proposal);
        if (Math.Abs(IoU(proposal, gt) - 0.64) >= 1e-12)
            throw new Exception("IoU mismatch");
        for (int i = 0; i < 4; ++i)
            if (Math.Abs(recovered[i] - gt[i]) >= 1e-12)
                throw new Exception("round trip mismatch");
        Console.WriteLine(
            "iou={0:F6} tw={1:F6} recovered={2:F6},{3:F6}",
            IoU(proposal, gt), delta[2], recovered[0], recovered[2]
        );
    }
}
```

예상 출력은 C++과 같다.

## 12. 프레임워크 간 shape·layout·dtype 대응

### 12.1 오늘의 실행 검증 상태

- NumPy IoU·encode·decode 독립 oracle: 실행 PASS
- PyTorch target·sampling·gradient routing test: 실행 PASS
- PyTorch train·eval 회귀 ablation과 동일 seed 재실행: 실행 PASS
- C++17: Apple clang으로 compile·실행 PASS
- C#: `csc`와 Mono로 compile·실행 PASS
- `pytest`: 로컬 환경에 없어 dependency-free direct assertion으로 대체 PASS
- `onnx`, `onnxruntime`: 로컬 환경에 없어 export·runtime은 미검증이며 아래 절차만 설명

| 경계 | PyTorch Python | C++ runtime | C# runtime | 고정할 계약 |
| --- | --- | --- | --- | --- |
| 이미지 | `[N,C,H,W]` | backend에 따라 NCHW·NHWC | backend에 따라 NCHW·NHWC | layout 이름과 transpose |
| RoI | `[R,5]` `float32` | 보통 `[R,5]` | 보통 `[R,5]` | batch index, 좌표 순서 |
| 좌표 | 이미지 scale의 `xyxy` | 같은 scale의 `xyxy` | 같은 scale의 `xyxy` | 반열린 끝점 |
| pooled | `[R,C,P_h,P_w]` | operator 계약 확인 | operator 계약 확인 | RoI Pooling과 RoI Align 구분 |
| logits | `[R,K+1]` | flat buffer 가능 | multidimensional tensor 가능 | background index 0 |
| deltas | `[R,K,4]` | `[R,K*4]`일 수 있음 | `[R,K*4]`일 수 있음 | class-major, `xywh` 순서 |
| label | `int64` | `int32`가 흔함 | `Int32`가 흔함 | cast와 ignore `-1` |
| 계산값 | `float32` | `float32` | `Single` | golden tolerance |

### 12.2 RoI Pooling과 RoI Align은 교환 가능하지 않다

두 연산은 출력 shape이 같을 수 있지만 값과 gradient는 다르다. RoI Pooling은 좌표와 bin을 양자화하고 max를 취한다. RoI Align은 실수 좌표에서 보간 sampling한다. 학습은 `roi_pool`, 배포는 `roi_align`으로 바꾸고 shape test만 통과시키면 모델 의미가 달라진다.

### 12.3 `xyxy`의 끝점 규칙

오늘 box math는 반열린 좌표를 사용하므로 폭은 `x2 - x1`이다. 고전 코드의 inclusive 좌표는 `x2 - x1 + 1`을 쓸 수 있다. 둘 중 무엇이 절대적으로 옳은 것이 아니라 데이터 변환, IoU, encode, clipping, NMS 전체가 한 규칙을 써야 한다.

## 13. 테스트 전략

### 13.1 단위 테스트 목록

- 동일 box의 IoU가 1인지 확인한다.
- 분리된 box의 IoU가 0인지 확인한다.
- 알려진 box 쌍의 IoU가 0.64인지 확인한다.
- `decode(encode(gt, proposal), proposal)`이 정답을 복원하는지 확인한다.
- 폭 또는 높이가 0인 box를 encode 전에 거부한다.
- threshold 바로 아래·위에서 `0`, `-1`, positive label이 맞는지 확인한다.
- 같은 generator seed의 sample index가 같은지 확인한다.
- ignore proposal이 sampler 출력에 없는지 확인한다.
- background의 box gradient가 0인지 확인한다.
- 정답 class가 아닌 box slice의 gradient가 0인지 확인한다.
- positive가 없는 batch에서도 backward가 성공하는지 확인한다.
- RoI batch index를 바꾸면 다른 feature를 읽는지 확인한다.
- 두 입력 해상도에서 pooled shape이 동일한지 확인한다.

### 13.2 Property test

양의 폭·높이를 가진 random proposal과 그 안팎의 random target을 생성해 다음 성질을 반복한다.

$$
\operatorname{decode}(\operatorname{encode}(g,b),b)\approx g
$$

$$
0\le\operatorname{IoU}(a,b)\le1
$$

$$
\operatorname{IoU}(a,b)=\operatorname{IoU}(b,a)
$$

좌표 크기가 매우 크거나 매우 작은 경우에는 `float32` 허용 오차를 명시한다. exact equality를 요구하면 올바른 구현도 연산 순서 차이로 실패할 수 있다.

### 13.3 Dataset contract test

학습 전에 이미지별로 다음 통계를 저장한다.

```text
proposal_count
positive_count
negative_count
ignore_count
max_iou_per_gt
box_width_min/max
box_height_min/max
```

`max_iou_per_gt`가 threshold보다 낮은 정답은 현재 proposal 집합으로 학습되지 않을 수 있다. 이 문제는 head optimizer를 바꿔도 해결되지 않는다.

## 14. 디버깅 플레이북

### 증상 1: 분류 loss만 내려가고 box loss는 0이다

`labels > 0`의 개수를 로그로 남긴다. label offset을 잘못 정해 모든 객체가 background가 되었거나, positive IoU threshold가 지나치게 높을 수 있다.

### 증상 2: 모든 class의 box weight가 같이 변한다

`box_deltas[positive, labels[positive] - 1]`처럼 advanced indexing했는지 확인한다. `[positive]`만 선택하면 모든 class delta가 회귀 손실에 들어갈 수 있다.

### 증상 3: RoI feature가 전부 거의 같다

proposal 좌표가 이미지 기준인데 `spatial_scale=1.0`을 넣었거나, 이미 축소한 좌표에 다시 `1/S`를 적용했을 수 있다. 고정 feature map에 좌표가 다른 두 RoI를 넣어 golden 값을 비교한다.

### 증상 4: batch 크기를 2 이상으로 늘리면 정확도가 무너진다

RoI 첫 열의 `batch_index`를 확인한다. 이미지별 proposal을 concatenate한 뒤 index를 모두 0으로 둔 오류가 흔하다.

### 증상 5: box가 폭발해 `inf`가 된다

decode 전 $d_w,d_h$의 quantile과 최대값을 본다. target normalization, learning rate, invalid proposal 폭·높이, `exp` clamp를 점검한다.

### 증상 6: 재실행 결과가 달라진다

Python, NumPy, PyTorch seed뿐 아니라 sampler generator, DataLoader worker seed, augmentation RNG, backend deterministic 설정, 데이터 split hash를 기록한다.

### 증상 7: training IoU는 오르지만 validation IoU는 0이다

evaluation에서 predicted class에 해당하는 delta를 골랐는지 확인한다. 학습 때는 정답 class delta를 선택하지만 추론 때는 예측 class delta를 선택한다.

## 15. 성능과 메모리

### 15.1 RoI activation 메모리

pooled tensor의 원소 수는 다음과 같다.

$$
E=RCP_hP_w
$$

`float32` 메모리는 대략 다음과 같다.

$$
M_{bytes}=4RCP_hP_w
$$

예를 들어 `R=512`, `C=256`, `P_h=P_w=7`이면 forward pooled tensor만 다음 크기다.

$$
4\times512\times256\times7\times7=25{,}690{,}112\ \text{bytes}
$$

이는 약 24.5 MiB다. backward를 위한 activation과 FC 중간값은 별도다. 이미지당 proposal 수의 변동이 OOM의 직접 원인이 될 수 있다.

### 15.2 계산 최적화 순서

1. invalid·낮은 점수 proposal을 RoI 연산 전에 제거한다.
2. 이미지당 sample 수를 고정하거나 상한을 둔다.
3. 작은 tensor를 반복 호출하지 말고 batch 전체 RoI를 묶는다.
4. profiler로 backbone, RoI operator, FC head, data copy 시간을 분리한다.
5. mixed precision은 operator 지원과 box math의 안정성을 확인한 뒤 적용한다.

### 15.3 Sampling bias와 throughput

positive fraction을 높이면 localization gradient는 늘지만 실제 background 분포를 덜 본다. batch size를 늘리면 통계는 안정될 수 있지만 RoI activation과 FC 비용이 선형 증가한다. accuracy뿐 아니라 images/s, RoIs/s, peak memory를 함께 기록한다.

## 16. 수치 안정성

- IoU 분모는 `clamp(min=epsilon)`으로 0 나눗셈을 막되 invalid box를 조용히 정상으로 취급하지 않는다.
- box 폭과 높이는 encode 전에 양수인지 검사한다.
- $d_w,d_h$ decode 전 상한을 둬 `exp` overflow를 막는다.
- logits에는 softmax를 먼저 적용하지 않고 `cross_entropy`에 그대로 전달한다.
- mixed precision에서 IoU threshold 근처의 label이 바뀔 수 있으므로 assignment를 `float32`로 유지하는 선택을 검토한다.
- `NaN` 좌표는 comparison에서 예측하기 어려운 경로로 빠지므로 입력 단계에서 `isfinite`로 거부한다.

## 17. 실무 실패 사례

### 사례 A: Resize 뒤 proposal 좌표를 갱신하지 않았다

이미지는 `800 x 600`에서 `400 x 300`으로 줄였지만 proposal은 원본 좌표로 남았다. RoI kernel은 범위 밖을 clamp했고 에러 없이 모서리 feature만 반복해서 반환했다. 해결책은 resize transform이 image와 모든 box를 함께 변환하도록 하나의 함수로 묶는 것이다.

### 사례 B: Ignore를 background로 바꿨다

중간 IoU proposal을 모두 0으로 채우자 background accuracy는 높아졌지만 객체 경계 근처 feature가 background로 강하게 학습되었다. label histogram과 IoU 구간별 loss를 모니터링해야 한다.

### 사례 C: Class label과 delta index가 한 칸 어긋났다

분류 label은 background가 0이라 객체가 1부터 시작하지만 box tensor는 객체 class 0부터 시작한다. `labels`를 그대로 box class index로 써 마지막 class에서 out-of-range가 나거나 엉뚱한 slice를 학습했다. 해결은 `labels - 1` 계약 테스트다.

### 사례 D: 학습과 배포가 다른 RoI operator를 사용했다

학습 graph에는 RoI Pooling, ONNX runtime에는 RoI Align을 연결했다. shape은 같고 export도 성공했지만 score와 box가 달라졌다. 고정 feature·RoI의 pooled tensor 자체를 runtime 간 비교해야 한다.

### 사례 E: Proposal recall을 head 문제로 오해했다

정답을 IoU 0.5 이상으로 덮는 proposal이 없는데 optimizer와 loss weight만 반복 변경했다. head 입력 이전에 recall upper bound를 측정해야 한다.

### 사례 F: 원문의 RPN 코드를 Fast R-CNN 실험에 포함했다

학습 대상과 latency 경계가 바뀌었는데 실험 이름은 그대로 유지했다. 이 결과는 Fast R-CNN head ablation과 비교할 수 없다. proposal source를 metadata에 명시해야 한다.

## 18. 배포 관점

### 18.1 모델 입력 계약

배포 artifact와 함께 다음 metadata를 저장한다.

```text
schema_version=fast-rcnn-head-v2
image_layout=NCHW
image_dtype=float32
roi_format=batch_index,x1,y1,x2,y2
roi_coordinate_space=resized_image
roi_end=exclusive
spatial_scale=0.25
pool=roi_pool
pool_output=2x2
background_label=0
box_layout=roi,class,dx,dy,dw,dh
num_foreground_classes=2
```

### 18.2 ONNX 검증 순서

1. export 전 eager output을 고정 fixture로 저장한다.
2. export model의 input·output 이름과 dynamic axis를 확인한다.
3. target runtime이 같은 RoI operator semantics를 지원하는지 확인한다.
4. feature, pooled tensor, logits, deltas를 단계별로 비교한다.
5. decode·clip·NMS까지 포함한 최종 box를 비교한다.
6. `R=0`, 이미지별 proposal 수가 다른 경우, 최대 proposal 수를 검사한다.

오늘 환경에서는 `onnx`와 `onnxruntime` 설치 여부를 별도 확인한 뒤에만 export를 검증한다. 패키지가 없으면 export 성공을 주장하지 않고 PyTorch eager, NumPy, C++17, C# golden으로 검증 범위를 명시한다.

### 18.3 운영 모니터링

- 이미지당 proposal 수의 p50·p95·p99
- positive score와 background score 분포
- decode 전 $d_w,d_h$의 p99와 최대값
- clipping 전후 box 면적 변화
- 유효하지 않은 box 비율
- class별 NMS 전후 box 수
- proposal recall proxy
- RoI operator latency와 peak memory
- 입력 schema version 불일치 횟수

label이 없는 운영 환경에서도 분포 이동과 좌표 pipeline 오류를 빠르게 감지할 수 있다.

## 19. 구현 체크리스트

### 좌표와 target

- [ ] `xyxy`와 끝점 규칙을 문서화했다.
- [ ] resize·crop·flip이 image와 box에 함께 적용된다.
- [ ] invalid·`NaN` box를 assignment 전에 거부한다.
- [ ] positive, negative, ignore threshold를 config에 저장한다.
- [ ] 정답별 proposal recall을 측정한다.

### Loss와 gradient

- [ ] background가 분류에 참여한다.
- [ ] background와 ignore가 회귀에 참여하지 않는다.
- [ ] positive의 정답 class delta만 선택한다.
- [ ] positive가 0개여도 backward가 성공한다.
- [ ] encode·decode round trip test가 통과한다.

### 학습과 재현성

- [ ] split, seed, dataset hash를 기록한다.
- [ ] sampler가 전용 generator를 사용한다.
- [ ] train과 eval metric을 분리한다.
- [ ] classification과 localization metric을 함께 본다.
- [ ] ablation은 한 변수만 바꾼다.

### 성능과 배포

- [ ] proposal 수와 pooled activation 메모리에 상한이 있다.
- [ ] 학습·배포 RoI operator가 같다.
- [ ] layout·dtype·class offset metadata가 있다.
- [ ] runtime 간 pooled tensor와 최종 box golden test가 있다.
- [ ] decode overflow와 invalid box metric을 모니터링한다.

## 20. 연습문제

### 문제 1

`R=256`, `C=128`, pooled size가 `7 x 7`일 때 `float32` pooled tensor의 크기를 byte와 MiB로 계산하라.

### 문제 2

label이 `[2, 0, -1, 1]`일 때 분류, 회귀, sampling에 각각 참여하는 index를 적어라.

### 문제 3

box delta tensor shape이 `[R, 3, 4]`이고 positive label이 3이라면 선택할 class index는 얼마인가?

### 문제 4

proposal `(0,0,10,10)`과 정답 `(1,1,9,9)`의 $t_x,t_y,t_w,t_h$를 구하라.

### 문제 5

이미지 좌표 proposal을 stride 16 feature에 적용할 때 `spatial_scale`은 얼마인가? 이미 feature 좌표로 변환했다면 얼마인가?

### 문제 6

분류 정확도는 100%지만 predicted box IoU가 0에 가깝다. 가장 먼저 분리해 볼 두 가지 항목은 무엇인가?

### 문제 7

학습은 RoI Pooling, 배포는 RoI Align인데 두 output shape이 같다. shape test가 충분하지 않은 이유를 설명하라.

### 문제 8

positive가 없는 batch에서 box loss를 `box_deltas.sum() * 0.0`으로 만드는 이유를 설명하라.

## 21. 해답

### 해답 1

원소 수는 다음과 같다.

$$
256\times128\times7\times7=1{,}605{,}632
$$

byte 수는 다음과 같다.

$$
1{,}605{,}632\times4=6{,}422{,}528
$$

MiB로는 약 $6.125$ MiB다.

### 해답 2

- 분류: index 0, 1, 3이 참여하고 ignore인 index 2는 제외한다.
- 회귀: positive인 index 0과 3만 참여한다.
- sampling 후보: index 0, 1, 3이며 index 2는 제외한다.

### 해답 3

background가 box class 축에 없으므로 `3 - 1 = 2`다.

### 해답 4

두 box의 중심은 모두 `(5,5)`이므로 $t_x=t_y=0$이다. proposal 크기는 10, 정답 크기는 8이다.

$$
t_w=t_h=\log(8/10)=\log(0.8)\approx-0.223144
$$

### 해답 5

이미지 좌표를 전달하면 `1/16=0.0625`다. 이미 feature 좌표로 변환했다면 `1.0`이다. 둘을 섞으면 이중 축소가 일어난다.

### 해답 6

box loss가 실제로 positive에 적용되는지와 decode할 때 올바른 class slice를 선택하는지 먼저 본다. 그다음 proposal recall과 target transform을 확인한다.

### 해답 7

RoI Pooling은 정수 양자화와 max, RoI Align은 실수 위치의 보간 sampling을 사용한다. shape이 같아도 feature 값과 gradient가 다르므로 학습된 head의 입력 분포가 바뀐다.

### 해답 8

box head와 graph 연결, device, dtype을 유지하는 0이기 때문이다. 새 CPU 상수 0을 만들면 device mismatch나 graph 단절을 만들 수 있다.

## 22. 핵심 요약

1. Fast R-CNN은 외부 proposal을 받는다. RPN은 Faster R-CNN의 구성 요소다.
2. 구현의 핵심은 `[batch_index,x1,y1,x2,y2]` RoI 계약과 `spatial_scale`이다.
3. proposal은 IoU로 positive, negative, ignore에 배정하고 sampling한다.
4. box encode와 decode는 round trip golden test로 검증해야 한다.
5. background는 분류에는 참여하지만 box 회귀에는 참여하지 않는다.
6. 클래스별 box head에서는 positive label의 `label - 1` slice만 학습한다.
7. class accuracy만으로 localization 학습을 판단할 수 없다.
8. 오늘 ablation은 회귀 손실을 켰을 때 mean IoU가 `0.0`에서 `0.82943`으로 개선됨을 확인했다.
9. 학습과 배포의 RoI operator, 끝점 규칙, layout, dtype이 모두 같아야 한다.
10. proposal recall, RoI 수, delta 분포를 운영 지표로 관찰해야 한다.

## 23. 다음 학습 예고

다음 소스는 `02-05.YOLO.md`다. 1회차에는 YOLOv1, SSD·FPN, Focal Loss·RetinaNet을 3 Part로 나눠 배웠다. 2회차 구현에서는 원본의 넓이를 유지하되 이전 예제를 반복하지 않고, grid target assignment와 decode·NMS를 포함한 작은 one-stage detector의 완전한 학습·평가 코드부터 시작한다. 원본 규모와 과거 3개 Part를 비교해 최대 3 Part 안에서 구현 진도를 명시한다.
