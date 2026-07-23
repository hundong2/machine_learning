<!-- curriculum: cycle=1; level=foundation; source_index=6/18; source=02-05.YOLO.md; part=3/3 -->

# Focal Loss와 RetinaNet: 쉬운 배경의 그라디언트를 줄이는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-07-24 |
| 회차·수준 | 1회차·기초 |
| 현재 소스 | 6/18 `02-05.YOLO.md` |
| Part | 3/3 |
| 이전 소스 | `02-05.YOLO.md` Part 2/3: SSD와 FPN |
| 다음 소스 | 7/18 `02-06.FCOS_DETR.md` |

이번 Part로 `02-05.YOLO.md`를 마친다. Part 1에서는 YOLOv1의 격자 책임과 손실을, Part 2에서는 SSD default box와 FPN을 다뤘다. 오늘은 수많은 배경 후보가 만드는 학습 불균형을 Focal Loss가 어떻게 완화하는지 배우고, 그 손실이 RetinaNet의 분류 head와 어떻게 결합되는지 확인한다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. 표본 수 불균형과 손실·그라디언트 불균형을 구분한다.
2. binary cross entropy에서 $p_t$, $\alpha_t$, modulating factor를 유도한다.
3. 쉬운 표본에서 Focal Loss의 손실과 그라디언트가 얼마나 빨리 감소하는지 계산한다.
4. logits에서 직접 계산하는 수치적으로 안정적인 PyTorch 구현을 작성한다.
5. RetinaNet의 FPN별 classification·box regression tensor shape를 추적한다.
6. 낮은 foreground prior를 반영하는 초기 bias를 계산하고 검증한다.
7. Python, C++, C# 사이의 layout·dtype·reduction 계약을 맞춘다.
8. 학습, export, mixed precision에서 자주 생기는 실패를 진단한다.

## 선수 지식과 기호

필요한 선수 지식은 sigmoid, binary cross entropy, 미분의 chain rule, convolution 출력 shape, FPN이다.

| 기호 | 뜻 |
| --- | --- |
| $z$ | 한 class에 대한 logit |
| $y$ | binary target, $y \in \{0, 1\}$ |
| $p$ | 양성 확률, $p=\sigma(z)$ |
| $p_t$ | 정답 class에 할당된 확률 |
| $\alpha$ | positive class weight, 보통 $0 < \alpha < 1$ |
| $\alpha_t$ | target에 따라 선택된 class-balancing weight |
| $\gamma$ | focusing parameter, $\gamma \ge 0$ |
| $N$ | batch 크기 |
| $K$ | foreground class 수 |
| $A$ | 한 spatial location의 anchor 수 |
| $H_l, W_l$ | FPN level $l$의 높이와 너비 |

이 글은 RetinaNet식 **독립 sigmoid 분류**를 다룬다. 각 anchor에서 $K+1$ class softmax를 쓰는 모델과는 target 및 출력 계약이 다르다.

## 1. 직관: 쉬운 문제 만 장이 어려운 문제 하나를 덮는다

한 이미지에서 100,000개 anchor를 평가하고 그중 10개만 foreground라고 하자. 학습 초기에 배경 예측이 꽤 정확해져 각 쉬운 negative의 손실이 작아지더라도, 수가 너무 많으면 합계는 여전히 크다.

예를 들어 정답 확률이 $p_t=0.9$인 쉬운 표본의 cross entropy는 다음과 같다.

$$
-\log(0.9) \approx 0.10536
$$

이런 표본이 100,000개면 단순 합은 약 10,536이다. 반면 $p_t=0.1$인 어려운 positive 10개의 합은 약 23.03이다. 이는 단순히 label 비율만의 문제가 아니다. optimizer가 실제로 받는 **손실과 그라디언트의 총량**이 쉬운 배경 쪽으로 쏠리는 문제다.

Focal Loss의 생각은 간단하다.

> 이미 잘 맞힌 표본의 기여는 빠르게 줄이고, 아직 틀리는 표본의 기여는 상대적으로 보존한다.

hard negative mining은 표본을 골라 버린다. Focal Loss는 모든 표본을 유지하되 연속적인 가중치로 기여도를 바꾼다. 따라서 threshold 경계에서 표본이 갑자기 포함되거나 제외되지 않는다.

## 2. 원본에서 바로잡을 점

원본의 방향은 좋지만 구현과 역사 설명에는 중요한 교정이 필요하다.

| 원본 내용 | 교정 |
| --- | --- |
| Focal Loss를 사실상 Kaiming He 한 사람의 발견으로 서술 | RetinaNet·Focal Loss 논문은 Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár의 공동 연구다. 한 명에게만 귀속하지 않는다. |
| RetinaNet이 모든 2-stage detector를 최초로 압도했다고 단정 | 논문의 핵심은 당시 대표적인 2-stage detector와 견줄 만한 정확도와 높은 속도를 보였다는 것이다. backbone, scale, test protocol이 다른 모든 모델에 대한 절대 명제는 아니다. |
| `alpha * loss`를 모든 target에 동일 적용 | 표준 binary 정의는 positive에 $\alpha$, negative에 $1-\alpha$를 적용한다. 즉 $\alpha_t=\alpha y+(1-\alpha)(1-y)$다. |
| `exp(-BCE)`가 언제나 $p_t$라고 설명 | class weight와 label smoothing이 없는 unreduced binary BCE에서만 그 등식이 직접 성립한다. weighted BCE를 넣은 뒤 `exp(-loss)`를 쓰면 $p_t$가 아니다. |
| 쉬운 배경을 수학적으로 소거한다고 표현 | 유한한 logit에서 정확히 0이 되는 것이 아니라 강하게 down-weight된다. underflow나 finite precision에서 0처럼 보일 수는 있다. |
| Focal Loss를 모든 불균형 문제의 일반 해법처럼 확대 | detection의 dense foreground-background 불균형에 맞춰 고안됐다. calibration, label noise, long-tail class imbalance에는 별도 검증과 다른 기법이 필요하다. |

또한 원본에는 RetinaNet의 매우 중요한 초기화가 빠져 있다. classification head의 bias를 낮은 foreground prior로 초기화하지 않으면 첫 step에서 모든 anchor가 $p \approx 0.5$를 내고, 수많은 negative가 큰 손실을 만들어 학습이 불안정해질 수 있다.

## 3. BCE를 정답 확률 하나로 쓰기

### 3.1 sigmoid와 binary cross entropy

logit $z$를 확률로 바꾸는 sigmoid는 다음과 같다.

$$
p = \sigma(z) = \frac{1}{1+\exp(-z)}
$$

binary cross entropy는 다음과 같다.

$$
\operatorname{BCE}(p,y)
=
-y\log(p)-(1-y)\log(1-p)
$$

정답 class의 확률 $p_t$를 target에 따라 정의한다.

$$
p_t
=
\begin{cases}
p & \text{if } y=1 \\
1-p & \text{if } y=0
\end{cases}
$$

이를 하나의 식으로 쓰면 다음과 같다.

$$
p_t = yp + (1-y)(1-p)
$$

따라서 BCE는 간단히 다음과 같이 쓸 수 있다.

$$
\operatorname{CE}(p_t)=-\log(p_t)
$$

### 3.2 class-balancing factor

positive와 negative의 정적 비율을 조절하는 $\alpha_t$는 다음과 같다.

$$
\alpha_t
=
\begin{cases}
\alpha & \text{if } y=1 \\
1-\alpha & \text{if } y=0
\end{cases}
$$

벡터 연산에 편한 형태는 다음과 같다.

$$
\alpha_t = \alpha y + (1-\alpha)(1-y)
$$

$\alpha=0.25$이면 positive 한 개의 계수는 $0.25$, negative 한 개의 계수는 $0.75$다. 얼핏 보면 negative를 더 크게 가중하는 것처럼 보인다. 그러나 $\alpha$는 positive 비율 자체를 뜻하지 않으며, $\gamma$와 함께 검증 데이터에서 선택된 hyperparameter다. 논문 설정을 맥락 없이 모든 데이터셋에 복사해서는 안 된다.

## 4. Focal Loss 유도

### 4.1 modulating factor

정답을 잘 맞히면 $p_t \to 1$이고, 틀리면 $p_t \to 0$이다. 그러므로 다음 값은 쉬운 표본에서 0에 가까워진다.

$$
1-p_t
$$

감쇠의 세기를 조절하기 위해 $\gamma$ 제곱을 적용한다.

$$
m(p_t)=(1-p_t)^\gamma
$$

이를 BCE에 곱하면 alpha-balanced Focal Loss가 된다.

$$
\operatorname{FL}(p_t)
=
-\alpha_t(1-p_t)^\gamma\log(p_t)
$$

$\gamma=0$이면 modulating factor가 1이므로 alpha-balanced BCE로 돌아간다.

$$
\operatorname{FL}(p_t;\gamma=0)
=
-\alpha_t\log(p_t)
$$

### 4.2 쉬운 표본과 어려운 표본의 수작업 비교

$\alpha_t$를 잠시 제외하고 $\gamma=2$라고 하자.

쉬운 표본 $p_t=0.9$에서는 다음과 같다.

$$
(1-0.9)^2 = 0.01
$$

$$
\operatorname{FL}(0.9)
=
0.01 \times 0.10536
\approx
0.0010536
$$

어려운 표본 $p_t=0.1$에서는 다음과 같다.

$$
(1-0.1)^2=0.81
$$

$$
\operatorname{FL}(0.1)
=
0.81 \times 2.30259
\approx
1.86510
$$

감쇠 비율은 각각 0.01과 0.81이다. Focal Loss는 어려운 표본의 손실도 줄이지만, 쉬운 표본을 훨씬 더 강하게 줄인다.

### 4.3 쉬운 표본 근처의 점근적 해석

$p_t=1-\varepsilon$이고 $0<\varepsilon\ll1$이라고 하자. Taylor 전개로 다음을 얻는다.

$$
-\log(1-\varepsilon)
\approx
\varepsilon
$$

따라서 BCE는 쉬운 표본에서 대략 $O(\varepsilon)$로 줄어든다.

$$
\operatorname{CE}(p_t)
\approx
\varepsilon
$$

Focal Loss는 modulating factor $\varepsilon^\gamma$를 더 곱하므로 다음과 같다.

$$
\operatorname{FL}(p_t)
\approx
\alpha_t\varepsilon^{\gamma+1}
$$

$\gamma=2$이면 쉬운 표본의 손실이 대략 1차가 아니라 3차로 감소한다. 이것이 “쉬운 배경의 꼬리를 눌러 준다”는 말의 수학적 의미다.

## 5. 그라디언트까지 봐야 하는 이유

optimizer가 직접 사용하는 것은 loss 값이 아니라 logit에 대한 gradient다.

### 5.1 BCE gradient

sigmoid와 BCE를 결합하면 잘 알려진 결과를 얻는다.

$$
\frac{\partial \operatorname{BCE}}{\partial z}
=
p-y
$$

positive target에서는 $y=1$이므로 gradient가 $p-1=-(1-p)$다. negative target에서는 $y=0$이므로 gradient가 $p$다.

### 5.2 Focal Loss gradient

먼저 정답 logit 방향을 통일하기 위해 다음을 정의한다.

$$
s=2y-1
$$

$$
q=sz
$$

그러면 $p_t=\sigma(q)$다. $q$에 대한 Focal Loss의 미분은 다음과 같다.

$$
\frac{\partial \operatorname{FL}}{\partial q}
=
\alpha_t
\left[
\gamma p_t(1-p_t)^\gamma\log(p_t)
-
(1-p_t)^{\gamma+1}
\right]
$$

$q=sz$이므로 다음과 같다.

$$
\frac{\partial \operatorname{FL}}{\partial z}
=
s\alpha_t
\left[
\gamma p_t(1-p_t)^\gamma\log(p_t)
-
(1-p_t)^{\gamma+1}
\right]
$$

쉬운 표본에서 $p_t=1-\varepsilon$를 대입하면 gradient 크기도 대략 다음 차수로 줄어든다.

$$
\left|
\frac{\partial \operatorname{FL}}{\partial z}
\right|
\approx
\alpha_t(\gamma+1)\varepsilon^{\gamma+1}
$$

따라서 $\gamma=2$이면 쉬운 표본의 gradient도 대략 3차로 줄어든다. 단순히 보고되는 loss 숫자만 작아지는 것이 아니다.

## 6. NumPy로 손실과 gradient를 수작업 검증하기

다음은 **실행 가능한 예제**다. stable BCE와 Focal Loss를 계산하고, analytic gradient를 central finite difference와 비교한다.

```python
import numpy as np


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x):
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def focal_from_logits(z, y, alpha=0.25, gamma=2.0):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bce = softplus(z) - y * z
    p = sigmoid(z)
    pt = y * p + (1.0 - y) * (1.0 - p)
    alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    return alpha_t * np.power(1.0 - pt, gamma) * bce


def focal_grad(z, y, alpha=0.25, gamma=2.0):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sign = 2.0 * y - 1.0
    pt = sigmoid(sign * z)
    alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    term = (
        gamma
        * pt
        * np.power(1.0 - pt, gamma)
        * np.log(pt)
        - np.power(1.0 - pt, gamma + 1.0)
    )
    return sign * alpha_t * term


z = np.array([2.197224577, -2.197224577, -2.197224577, 2.197224577])
y = np.array([1.0, 0.0, 1.0, 0.0])
loss = focal_from_logits(z, y)
grad = focal_grad(z, y)

eps = 1e-6
numeric = np.empty_like(z)
for i in range(z.size):
    plus = z.copy()
    minus = z.copy()
    plus[i] += eps
    minus[i] -= eps
    numeric[i] = (
        focal_from_logits(plus, y).sum()
        - focal_from_logits(minus, y).sum()
    ) / (2.0 * eps)

np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)
print("loss:", np.round(loss, 8))
print("analytic:", np.round(grad, 8))
print("numeric:", np.round(numeric, 8))
```

앞의 두 표본은 $p_t=0.9$인 쉬운 positive와 negative이고, 뒤의 두 표본은 $p_t=0.1$인 어려운 positive와 negative다. 같은 난이도라도 $\alpha_t$가 달라 loss와 gradient 크기가 달라진다.

## 7. logits 기반 PyTorch 구현

### 7.1 왜 확률보다 logits를 받는가

`sigmoid`를 먼저 계산한 뒤 `log(p)`를 취하면 큰 절댓값의 logit에서 $p$가 0 또는 1로 반올림될 수 있다. `binary_cross_entropy_with_logits`는 sigmoid와 log를 결합한 stable 식을 사용한다.

$$
\operatorname{BCEWithLogits}(z,y)
=
\max(z,0)-zy+\log(1+\exp(-|z|))
$$

### 7.2 실행 가능한 구현

다음은 **실행 가능한 예제**다. `targets`는 0 또는 1이어야 하며 `logits`와 shape가 같아야 한다.

```python
import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
    reduction="none",
):
    if logits.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: {tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only 0 or 1")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    prob = torch.sigmoid(logits)
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    modulating = (1.0 - pt).pow(gamma)

    if alpha is None:
        alpha_t = torch.ones_like(logits)
    else:
        alpha_t = (
            targets * alpha
            + (1.0 - targets) * (1.0 - alpha)
        )

    loss = alpha_t * modulating * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unknown reduction: {reduction}")


logits = torch.tensor(
    [2.197224577, -2.197224577, -2.197224577, 2.197224577],
    dtype=torch.float64,
    requires_grad=True,
)
targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
loss = sigmoid_focal_loss(logits, targets, reduction="sum")
loss.backward()

assert torch.isfinite(loss)
assert torch.isfinite(logits.grad).all()
assert logits.grad[0] < 0
assert logits.grad[1] > 0
assert abs(logits.grad[0]) < abs(logits.grad[2])
assert abs(logits.grad[1]) < abs(logits.grad[3])
print(f"loss={loss.item():.9f}")
print("grad=", logits.grad.tolist())
```

### 7.3 원본 구현과의 차이

원본 코드는 다음과 같은 형태다.

```python
# 설명용: 표준 alpha_t 구현이 아니므로 그대로 사용하지 않는다.
F_loss = alpha * (1 - pt) ** gamma * BCE_loss
```

이 식은 positive와 negative 모두에 `alpha`를 적용한다. 올바른 binary $\alpha_t$는 다음 코드처럼 target에 따라 달라야 한다.

```python
# 실행 가능한 핵심 식
alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
loss = alpha_t * (1.0 - pt).pow(gamma) * bce
```

## 8. RetinaNet의 구조와 tensor shape

RetinaNet은 크게 세 부분으로 구성된다.

1. ResNet 같은 backbone이 bottom-up feature를 만든다.
2. FPN이 $P_3$부터 $P_7$까지 multi-scale feature를 만든다.
3. 모든 level에 weight를 공유하는 classification subnet과 box regression subnet을 적용한다.

### 8.1 level별 head 출력

입력 feature가 NCHW layout이라고 하자.

$$
P_l \in \mathbb{R}^{N \times C \times H_l \times W_l}
$$

classification head의 raw 출력은 다음 shape다.

$$
Z_l^{cls}
\in
\mathbb{R}^{N \times (A K) \times H_l \times W_l}
$$

box regression head의 raw 출력은 다음 shape다.

$$
Z_l^{box}
\in
\mathbb{R}^{N \times (4 A) \times H_l \times W_l}
$$

classification 출력은 학습을 위해 다음 논리 shape로 바꾼다.

$$
(N,AK,H_l,W_l)
\rightarrow
(N,H_l,W_l,A,K)
\rightarrow
(N,H_lW_lA,K)
$$

box 출력도 같은 anchor 순서를 유지해야 한다.

$$
(N,4A,H_l,W_l)
\rightarrow
(N,H_l,W_l,A,4)
\rightarrow
(N,H_lW_lA,4)
$$

두 head의 flatten 순서가 다르면 class score와 box delta가 서로 다른 anchor를 가리키는 조용한 버그가 생긴다.

### 8.2 수치 예제

$N=2$, $A=9$, $K=80$, $P_3$의 공간 크기가 $100 \times 160$이라고 하자.

classification raw channel 수는 다음과 같다.

$$
AK=9 \times 80=720
$$

따라서 raw shape는 $(2,720,100,160)$이고, flatten 뒤 shape는 다음과 같다.

$$
(2,144000,80)
$$

box raw channel 수는 $4A=36$이며 flatten 뒤 shape는 $(2,144000,4)$다.

### 8.3 전체 anchor 수

level 집합을 $\mathcal{L}$이라고 하면 이미지 한 장의 전체 anchor 수는 다음과 같다.

$$
M=A\sum_{l \in \mathcal{L}}H_lW_l
$$

최종 결합 shape는 classification이 $(N,M,K)$, box regression이 $(N,M,4)$다.

## 9. PyTorch RetinaNet mini head

다음은 **실행 가능한 교육용 구현**이다. 실제 RetinaNet 전체가 아니라 FPN feature를 받는 두 subnet과 reshape 계약을 검증한다.

```python
import math
import torch
from torch import nn


class RetinaMiniHead(nn.Module):
    def __init__(self, channels, anchors, classes, prior=0.01):
        super().__init__()
        self.anchors = anchors
        self.classes = classes
        self.cls_tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, anchors * classes, 3, padding=1),
        )
        self.box_tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, anchors * 4, 3, padding=1),
        )

        bias = math.log(prior / (1.0 - prior))
        nn.init.constant_(self.cls_tower[-1].bias, bias)

    def flatten_cls(self, x):
        n, _, h, w = x.shape
        x = x.view(n, self.anchors, self.classes, h, w)
        return x.permute(0, 3, 4, 1, 2).reshape(
            n,
            h * w * self.anchors,
            self.classes,
        )

    def flatten_box(self, x):
        n, _, h, w = x.shape
        x = x.view(n, self.anchors, 4, h, w)
        return x.permute(0, 3, 4, 1, 2).reshape(
            n,
            h * w * self.anchors,
            4,
        )

    def forward(self, pyramid):
        cls_all = []
        box_all = []
        for feature in pyramid:
            cls_all.append(self.flatten_cls(self.cls_tower(feature)))
            box_all.append(self.flatten_box(self.box_tower(feature)))
        return torch.cat(cls_all, dim=1), torch.cat(box_all, dim=1)


torch.manual_seed(7)
head = RetinaMiniHead(channels=8, anchors=3, classes=5, prior=0.01)
pyramid = [
    torch.randn(2, 8, 16, 20),
    torch.randn(2, 8, 8, 10),
    torch.randn(2, 8, 4, 5),
]
cls_logits, box_delta = head(pyramid)
expected_anchors = 3 * (16 * 20 + 8 * 10 + 4 * 5)

assert cls_logits.shape == (2, expected_anchors, 5)
assert box_delta.shape == (2, expected_anchors, 4)
assert cls_logits.is_contiguous()
assert box_delta.is_contiguous()

targets = torch.zeros_like(cls_logits)
targets[0, 17, 2] = 1.0
classification_loss = sigmoid_focal_loss(
    cls_logits,
    targets,
    reduction="sum",
) / 1.0
regression_loss = box_delta[0, 17].abs().sum()
total = classification_loss + regression_loss
total.backward()

assert torch.isfinite(total)
assert head.cls_tower[-1].weight.grad is not None
print("classification:", tuple(cls_logits.shape))
print("box:", tuple(box_delta.shape))
print(f"total={total.item():.6f}")
```

마지막 나눗셈의 `1.0`은 positive anchor 수로 정규화한다는 뜻을 단순화한 것이다. 실제 구현에서는 분산 학습 worker 전체의 positive 수를 집계하고, 0개일 때의 정책을 명시해야 한다.

## 10. 초기 foreground prior bias

### 10.1 식 유도

sigmoid 출력의 초기 foreground 확률을 $\pi$로 만들고 싶다.

$$
\sigma(b)=\pi
$$

sigmoid의 역함수를 적용하면 다음을 얻는다.

$$
b=\log\left(\frac{\pi}{1-\pi}\right)
$$

$\pi=0.01$이면 다음과 같다.

$$
b=\log\left(\frac{0.01}{0.99}\right)
\approx
-4.59512
$$

weight가 0에 가깝다면 초기 class 확률도 약 0.01이 된다. 모든 anchor가 처음부터 $p=0.5$라고 주장하는 상황을 피하므로 negative loss 폭발을 완화한다.

### 10.2 prior와 alpha는 다른 손잡이다

- prior bias는 **초기 prediction distribution**을 정한다.
- $\alpha_t$는 학습 중 **positive와 negative loss의 상대 weight**를 정한다.
- $\gamma$는 confidence에 따라 **쉬운 표본을 얼마나 억제할지** 정한다.

세 값은 서로 대체 관계가 아니다. prior를 바꾸고도 alpha와 gamma를 그대로 둘 수 있지만, 최적점이 유지된다는 보장은 없다.

## 11. ignore target과 normalization

dense detector의 target은 흔히 세 상태를 가진다.

| 상태 | 의미 | classification loss |
| --- | --- | --- |
| positive | 특정 foreground class와 매칭 | 포함 |
| negative | background | 포함 |
| ignore | IoU 경계 또는 crowd 등 | 제외 |

ignore를 0으로 바꿔 넣으면 배경으로 학습된다. loss를 계산하기 전에 mask로 제외해야 한다.

$$
\mathcal{L}_{cls}
=
\frac{1}{\max(1,N_{pos})}
\sum_{i \in \mathcal{V}}
\sum_{k=1}^{K}
\operatorname{FL}(z_{ik},y_{ik})
$$

$\mathcal{V}$는 ignore가 아닌 유효 anchor 집합이다. 분모를 전체 anchor 수로 두면 positive 밀도에 따라 gradient scale이 달라질 수 있다. 반대로 positive 수가 0인 batch에서 분모를 그대로 쓰면 0으로 나눈다.

실무 정책은 다음 중 하나를 명확히 선택한다.

1. 분모를 $\max(1,N_{pos})$로 둔다.
2. 분산 학습에서는 all-reduce한 global positive 수를 사용한다.
3. positive가 없는 image도 negative classification을 학습할지 결정한다.

## 12. C++ scalar 구현

다음은 **C++17에서 실행 가능한 예제**다. framework tensor 연산 대신 한 표본의 stable sigmoid와 Focal Loss를 구현한다.

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

double sigmoid(double z) {
    if (z >= 0.0) {
        return 1.0 / (1.0 + std::exp(-z));
    }
    const double e = std::exp(z);
    return e / (1.0 + e);
}

double softplus(double z) {
    return std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
}

double focal(
    double z,
    int y,
    double alpha = 0.25,
    double gamma = 2.0
) {
    assert(y == 0 || y == 1);
    const double target = static_cast<double>(y);
    const double p = sigmoid(z);
    const double pt = target * p + (1.0 - target) * (1.0 - p);
    const double alpha_t =
        target * alpha + (1.0 - target) * (1.0 - alpha);
    const double bce = softplus(z) - target * z;
    return alpha_t * std::pow(1.0 - pt, gamma) * bce;
}

int main() {
    const double easy_positive = focal(2.197224577, 1);
    const double hard_positive = focal(-2.197224577, 1);
    const double easy_negative = focal(-2.197224577, 0);
    const double hard_negative = focal(2.197224577, 0);

    assert(easy_positive < hard_positive);
    assert(easy_negative < hard_negative);
    assert(std::isfinite(hard_negative));
    std::cout << std::fixed << std::setprecision(9)
              << easy_positive << " "
              << hard_positive << " "
              << easy_negative << " "
              << hard_negative << "\n";
}
```

`std::max`의 선언을 간접 include에 기대지 않고 `<algorithm>`에서 직접 가져온다. standard library 구현이 달라도 같은 source가 컴파일되어야 한다.

## 13. C# scalar 구현

다음은 **C#에서 실행 가능한 예제**다.

```csharp
using System;

public static class FocalExample
{
    static double Sigmoid(double z)
    {
        if (z >= 0.0)
            return 1.0 / (1.0 + Math.Exp(-z));
        double e = Math.Exp(z);
        return e / (1.0 + e);
    }

    static double Softplus(double z)
    {
        return Math.Max(z, 0.0) + Math.Log(1.0 + Math.Exp(-Math.Abs(z)));
    }

    static double Focal(
        double z,
        int y,
        double alpha = 0.25,
        double gamma = 2.0)
    {
        if (y != 0 && y != 1)
            throw new ArgumentException("y must be 0 or 1");
        double target = y;
        double p = Sigmoid(z);
        double pt = target * p + (1.0 - target) * (1.0 - p);
        double alphaT =
            target * alpha + (1.0 - target) * (1.0 - alpha);
        double bce = Softplus(z) - target * z;
        return alphaT * Math.Pow(1.0 - pt, gamma) * bce;
    }

    public static void Main()
    {
        double easyPositive = Focal(2.197224577, 1);
        double hardPositive = Focal(-2.197224577, 1);
        double easyNegative = Focal(-2.197224577, 0);
        double hardNegative = Focal(2.197224577, 0);

        if (!(easyPositive < hardPositive))
            throw new Exception("positive ordering failed");
        if (!(easyNegative < hardNegative))
            throw new Exception("negative ordering failed");
        Console.WriteLine(
            "{0:F9} {1:F9} {2:F9} {3:F9}",
            easyPositive,
            hardPositive,
            easyNegative,
            hardNegative);
    }
}
```

## 14. 프레임워크 간 shape·layout·dtype 대응

| 항목 | PyTorch Python | C++ runtime 예 | C# runtime 예 |
| --- | --- | --- | --- |
| raw image 기본 layout | NCHW | 명시적 stride를 가진 NCHW 또는 NHWC | 배열 계약에 따라 명시 |
| classification raw | `(N, A*K, H, W)` | 엔진 output metadata 확인 | 엔진 output metadata 확인 |
| classification 논리 view | `(N, H*W*A, K)` | index 식으로 view | index 식으로 view |
| target | 동일 shape의 0/1 float와 ignore mask | byte 또는 float target | `byte` 또는 `float` target |
| training 권장 dtype | FP32 또는 AMP | 대개 학습보다 추론 | 대개 학습보다 추론 |
| probability | sigmoid를 한 번만 적용 | logit이면 sigmoid 적용 | logit이면 sigmoid 적용 |
| reduction | `none`, `sum`, `mean` 명시 | 직접 누적·정규화 | 직접 누적·정규화 |

flatten index를 문서화하면 언어 간 검증이 쉬워진다. NHWAK 순서에서 anchor index는 다음과 같다.

$$
i=(((nH+h)W+w)A+a)
$$

class를 포함한 1차원 offset은 다음과 같다.

$$
j=iK+k
$$

C++와 C#에서 이 식으로 접근한 값이 PyTorch의 `(N,H,W,A,K)` view와 같아야 한다.

dtype 주의점은 다음과 같다.

- target을 정수 나눗셈이나 boolean 연산 뒤 그대로 두지 말고 logit dtype으로 변환한다.
- FP16에서 큰 음수 logit의 `sigmoid`와 `log`를 따로 계산하지 않는다.
- loss accumulation과 positive count normalization은 가능하면 FP32로 수행한다.
- export된 모델이 logits를 내는지 probability를 내는지 metadata로 고정한다.

## 15. 테스트 전략

### 15.1 값 테스트

다음 기준점을 고정한다.

- $p_t=0.9$, $\gamma=2$의 modulating factor는 0.01이다.
- $\gamma=0$이면 alpha-balanced BCE와 같다.
- $\alpha$가 `None`이면 class balancing 없이 동작한다.
- positive와 negative는 동일한 $p_t$라도 $\alpha_t$ 때문에 값이 다를 수 있다.

### 15.2 gradient 테스트

작은 float64 tensor에서 다음을 확인한다.

1. autograd와 finite difference가 일치한다.
2. 쉬운 표본의 gradient 절댓값이 어려운 표본보다 작다.
3. positive logit gradient는 logit을 키우는 방향이고 negative는 줄이는 방향이다.
4. 매우 큰 양수·음수 logit에서도 NaN과 Inf가 없다.

### 15.3 shape 테스트

- 각 level의 channel이 각각 $AK$, $4A$인지 확인한다.
- concat 뒤 anchor 수가 $A\sum_lH_lW_l$인지 확인한다.
- classification과 regression의 level·spatial·anchor 순서가 같은지 marker tensor로 확인한다.
- odd input shape에서도 FPN level과 anchor generator가 같은 공간 크기를 사용하는지 확인한다.

### 15.4 negative 테스트

다음 입력은 실패해야 한다.

- `logits.shape != targets.shape`
- target에 0과 1 이외의 값이 있으나 soft target을 지원한다고 선언하지 않은 경우
- $\gamma<0$
- 알 수 없는 reduction
- ignore label을 mask하지 않고 BCE에 전달한 경우

## 16. 디버깅 지도

| 증상 | 먼저 볼 것 | 가능한 원인 |
| --- | --- | --- |
| 첫 step loss가 매우 큼 | class bias와 초기 probability | prior bias 누락 |
| positive recall이 0에 가까움 | matcher positive 수, $\alpha$, bias | positive가 ignore되거나 과도한 down-weight |
| loss가 갑자기 NaN | logit 범위, AMP scaler | 확률에서 직접 `log`, FP16 underflow |
| class loss만 빠르게 0 | per-class positive count | target encoding 또는 anchor matching 오류 |
| box는 맞는데 class가 엉뚱함 | flatten marker test | cls·box anchor 순서 불일치 |
| single GPU와 DDP scale이 다름 | normalization 분모 | local positive count 사용 |
| export 뒤 score가 지나치게 작음 | sigmoid 적용 횟수 | graph와 client에서 sigmoid 중복 |
| 쉬운 negative가 여전히 지배 | loss histogram | $\gamma$, normalization, ignore mask 오류 |

평균 loss 하나만 보면 원인을 놓치기 쉽다. positive, negative, ignore 수와 easy·hard bucket별 loss·gradient histogram을 함께 기록한다.

## 17. 성능·메모리·수치 안정성

### 17.1 출력 메모리

classification logit 수는 다음과 같다.

$$
N K A \sum_l H_lW_l
$$

FP32 byte 수는 원소 수의 4배다. 예를 들어 $N=2$, $K=80$, $A=9$, 전체 spatial location 수가 22,000이면 logit만 약 31.68 million개, 즉 약 126.7 MB다. backward를 위한 activation까지 고려하면 더 커진다.

메모리를 줄이는 방법은 다음과 같다.

- level별 loss를 계산해 불필요한 대형 concat을 피한다.
- AMP를 쓰되 loss 핵심 연산과 reduction의 FP32 승격을 확인한다.
- inference에서는 threshold를 이른 단계에 적용하되 정확도와 top-k 계약을 검증한다.

### 17.2 `pow`와 극단 확률

$p_t$가 1로 반올림되면 $(1-p_t)^\gamma$는 0이다. 이는 쉬운 표본에 기대한 방향이지만, gradient 검증에서는 float64를 사용해야 analytic 차이를 볼 수 있다.

확률을 clamp하면 NaN을 막을 수 있지만 gradient를 인위적으로 자른다. 우선 stable BCE-with-logits를 사용하고, clamp가 정말 필요한지 측정한다.

### 17.3 gamma의 비용

`gamma=2`처럼 정수면 제곱으로 최적화할 수 있지만, hyperparameter를 실수로 허용하면 일반 `pow`가 필요하다. 대부분 전체 detector 비용에서 작지만, 매우 큰 dense tensor에서는 profiler로 확인한다.

### 17.4 성능 측정

다음 항목을 따로 측정한다.

- backbone·FPN·head forward latency
- focal loss forward·backward latency
- peak allocated memory
- image당 valid·positive anchor 수
- p50뿐 아니라 p95 latency
- export runtime의 decode·threshold·NMS 포함 end-to-end latency

## 18. 실무 실패 사례

### 사례 A: $\alpha$를 positive와 negative에 똑같이 적용

원본 예제처럼 전체 loss에 `alpha`를 곱하면 class balancing이 아니라 learning rate를 일정 비율로 줄인 것과 비슷해진다.

**방지:** target별 $\alpha_t$ 값의 단위 테스트를 만든다.

### 사례 B: weighted BCE에서 `exp(-BCE)`로 $p_t$ 복원

`pos_weight` 또는 sample weight가 포함된 BCE는 더 이상 단순한 $-\log(p_t)$가 아니다.

**방지:** $p_t$는 sigmoid probability와 target에서 직접 만든다.

### 사례 C: ignore anchor를 background로 변환

IoU 경계의 anchor나 crowd 영역이 negative로 학습되어 decision boundary가 오염된다.

**방지:** target encoding에서 positive, negative, ignore count를 기록하고 ignore mask를 loss 전에 적용한다.

### 사례 D: positive가 없는 batch에서 0으로 나눔

희소 데이터나 random crop 뒤 positive가 사라지면 loss가 NaN이 된다.

**방지:** 분모 정책을 `max(1, N_pos)`처럼 명시하고 해당 batch를 테스트한다.

### 사례 E: prior bias의 부호를 반대로 사용

$\log((1-\pi)/\pi)$를 넣으면 초기 foreground 확률이 0.99가 되어 수많은 false positive와 큰 loss를 만든다.

**방지:** 초기화 직후 `sigmoid(bias)`가 $\pi$인지 assert한다.

### 사례 F: Focal Loss가 label noise까지 해결한다고 기대

잘못 붙은 label은 모델 관점에서 계속 어려운 표본이다. Focal Loss가 오히려 noisy sample에 집중할 수 있다.

**방지:** annotation audit, robust loss, sample filtering을 별도로 검토한다.

### 사례 G: class imbalance와 foreground-background imbalance를 혼동

Focal Loss가 배경 anchor 문제를 줄여도 희귀 class와 빈번한 class 사이의 long tail이 자동으로 해결되지는 않는다.

**방지:** class별 AP·recall을 보고 reweighting, resampling, logit adjustment를 별도 비교한다.

## 19. 배포 관점

Focal Loss는 일반적으로 **학습 전용**이다. 추론 graph에는 classification logits와 후처리만 필요하다. 따라서 배포 계약은 다음을 분리해야 한다.

1. 모델 출력이 logits인지 sigmoid probability인지 명시한다.
2. output level 순서와 각 level의 $(H_l,W_l,A,K)$를 고정한다.
3. anchor generator의 scale·ratio·center convention을 학습과 동일하게 유지한다.
4. class별 sigmoid, threshold, pre-NMS top-k, NMS, max detections 순서를 명시한다.
5. resize·letterbox 좌표 변환과 inverse transform을 테스트한다.

### 19.1 ONNX

추론용 export에 loss를 포함하지 않는 편이 단순하다. head raw logits를 export하고 client 또는 graph의 후처리에서 sigmoid를 정확히 한 번 적용한다.

dynamic shape를 허용하면 FPN의 $H_l,W_l$과 anchor 생성도 같은 입력 shape에서 계산되어야 한다. 고정 anchor constant를 넣고 입력만 dynamic으로 만들면 개수가 어긋난다.

### 19.2 calibration

Focal Loss로 학습한 score는 BCE 학습 score와 calibration 특성이 다를 수 있다. threshold 0.5를 관성적으로 사용하지 않는다. 운영 데이터에서 precision-recall curve, expected calibration error, class별 threshold를 검토한다.

### 19.3 모니터링

운영에서는 다음을 관찰한다.

- image당 threshold 통과 후보 수
- class별 score quantile
- NMS 전후 detection 수
- 작은 객체와 큰 객체의 recall proxy
- 입력 해상도·aspect ratio 분포
- 특정 class의 score drift

후보 수가 갑자기 늘면 sigmoid 중복 제거, threshold 설정, input normalization, model version을 먼저 확인한다.

## 20. 체크리스트

### 수학과 구현

- [ ] $p_t$가 target에 따라 올바르게 선택되는가?
- [ ] $\alpha_t$가 positive에는 $\alpha$, negative에는 $1-\alpha$인가?
- [ ] $\gamma=0$에서 alpha-balanced BCE와 일치하는가?
- [ ] stable BCE-with-logits를 사용하는가?
- [ ] ignore anchor를 loss에서 제외하는가?
- [ ] reduction과 normalization 분모가 문서화되어 있는가?

### shape와 target

- [ ] classification channel이 $AK$인가?
- [ ] regression channel이 $4A$인가?
- [ ] 모든 level의 anchor 순서가 cls·box·target에서 같은가?
- [ ] concat 뒤 전체 anchor 수를 assert하는가?
- [ ] positive, negative, ignore 개수를 기록하는가?

### 초기화와 학습

- [ ] `sigmoid(class_bias)`가 설정한 prior와 같은가?
- [ ] DDP에서 positive count normalization이 일관적인가?
- [ ] easy·hard sample별 loss 또는 gradient 분포를 보는가?
- [ ] AMP에서 loss와 gradient가 finite인가?
- [ ] label noise를 별도 점검하는가?

### 배포

- [ ] export output이 logit인지 probability인지 명시했는가?
- [ ] sigmoid를 정확히 한 번 적용하는가?
- [ ] dynamic shape와 anchor 생성이 함께 변하는가?
- [ ] threshold와 calibration을 운영 데이터에서 정했는가?
- [ ] 학습·배포의 resize와 좌표 convention이 같은가?

## 21. 연습문제

### 문제 1

$y=1$, $p=0.8$, $\alpha=0.25$, $\gamma=2$일 때 $p_t$, $\alpha_t$, modulating factor와 Focal Loss를 계산하라.

### 문제 2

$y=0$, $p=0.2$일 때 문제 1과 같은 값을 계산하라. 두 표본의 $p_t$는 같은데 loss가 다른 이유를 설명하라.

### 문제 3

$\gamma=0$일 때 Focal Loss가 무엇이 되는지 유도하라.

### 문제 4

초기 foreground prior $\pi=0.05$를 만들 bias를 계산하라.

### 문제 5

$N=1$, $A=9$, $K=20$이고 FPN 공간 크기가 $80 \times 80$, $40 \times 40$, $20 \times 20$일 때 전체 anchor 수와 classification flatten shape를 구하라.

### 문제 6

positive anchor가 0개인 image에서 $\max(1,N_{pos})$ 정규화를 쓰는 이유와, 그래도 결정해야 할 정책 하나를 설명하라.

### 문제 7

원본 코드의 `alpha * focal_term * BCE`가 표준 $\alpha_t$를 구현하지 못하는 이유를 코드 한 줄로 바로잡아라.

### 문제 8

Focal Loss가 label noise가 심한 데이터에서 위험할 수 있는 이유를 설명하라.

## 22. 해답

### 해답 1

positive이므로 다음과 같다.

$$
p_t=0.8
$$

$$
\alpha_t=0.25
$$

$$
(1-p_t)^2=0.04
$$

따라서 다음을 얻는다.

$$
\operatorname{FL}
=
-0.25 \times 0.04 \times \log(0.8)
\approx
0.002231
$$

### 해답 2

negative이고 $p=0.2$이므로 정답인 negative의 확률은 $p_t=1-p=0.8$이다. $\alpha_t=1-\alpha=0.75$이고 modulating factor는 0.04다.

$$
\operatorname{FL}
=
-0.75 \times 0.04 \times \log(0.8)
\approx
0.006694
$$

난이도를 나타내는 $p_t$는 같지만 class-balancing factor가 각각 0.25와 0.75라서 loss가 3배 차이 난다.

### 해답 3

$(1-p_t)^0=1$이므로 다음과 같다.

$$
\operatorname{FL}
=
-\alpha_t\log(p_t)
$$

즉 alpha-balanced BCE다.

### 해답 4

$$
b
=
\log\left(\frac{0.05}{0.95}\right)
\approx
-2.94444
$$

### 해답 5

전체 spatial location 수는 다음과 같다.

$$
80^2+40^2+20^2
=
6400+1600+400
=
8400
$$

전체 anchor 수는 다음과 같다.

$$
M=9 \times 8400=75600
$$

classification flatten shape는 $(1,75600,20)$이다.

### 해답 6

0으로 나누어 NaN이 되는 것을 막기 위해서다. 다만 positive가 없는 image에서도 negative classification loss를 학습할지, 아니면 그 image의 classification loss를 0으로 둘지는 별도 정책으로 결정해야 한다.

### 해답 7

다음처럼 target에 따라 계수를 선택한다.

```python
alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
```

그 뒤 `alpha_t * focal_term * BCE`를 계산한다.

### 해답 8

잘못된 label은 모델이 계속 틀리는 hard example로 남는다. Focal Loss는 쉬운 표본을 줄이고 어려운 표본에 상대적으로 집중하므로, annotation noise의 영향까지 키울 수 있다.

## 핵심 요약

1. dense detector의 문제는 표본 수뿐 아니라 쉬운 negative가 만드는 손실·gradient 총량의 불균형이다.
2. $p_t$는 정답 class 확률이고, $\alpha_t$는 positive·negative에 다른 class-balancing weight를 준다.
3. Focal Loss는 $-\alpha_t(1-p_t)^\gamma\log(p_t)$다.
4. 쉬운 표본에서 손실과 gradient는 대략 $(1-p_t)^{\gamma+1}$ 차수로 빠르게 줄어든다.
5. 원본의 모든 표본에 동일한 `alpha`를 곱하는 코드는 표준 $\alpha_t$ 구현이 아니다.
6. logits와 stable BCE를 사용해야 극단 확률과 mixed precision에서 안전하다.
7. RetinaNet head는 level별 $(N,AK,H,W)$와 $(N,4A,H,W)$를 같은 anchor 순서로 flatten한다.
8. prior bias $b=\log(\pi/(1-\pi))$는 초기 foreground 확률을 낮춰 첫 step을 안정시킨다.
9. ignore mask, positive-count normalization, DDP 집계는 loss 수식만큼 중요하다.
10. Focal Loss는 label noise, long-tail class imbalance, calibration을 자동 해결하지 않는다.

## 다음 학습 예고

다음 문서는 1회차 기초 7/18 `02-06.FCOS_DETR.md`다. anchor 없이 위치별 거리로 box를 예측하는 FCOS와, object query와 bipartite matching으로 set prediction을 수행하는 DETR을 비교한다. dense anchor의 불균형을 손실로 완화한 오늘의 관점이, 다음에는 **anchor 자체를 없애거나 예측 집합의 정의를 바꾸는 설계**로 이어진다.
