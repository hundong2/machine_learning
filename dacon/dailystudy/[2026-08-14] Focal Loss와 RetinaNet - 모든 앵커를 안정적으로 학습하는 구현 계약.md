<!-- curriculum: cycle=2; level=implementation; source_index=6/18; source=02-05.YOLO.md; part=3/3 -->

# Focal Loss와 RetinaNet: 모든 앵커를 안정적으로 학습하는 구현 계약

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-14 |
| 회차·수준 | 2회차 · 구현 (`implementation`) |
| 현재 소스 | 6/18 · `02-05.YOLO.md` |
| Part | 3/3 · Focal Loss와 RetinaNet 학습 파이프라인 |
| 이전 소스 | `02-05.YOLO.md` Part 2/3 · SSD/FPN 구현 |
| 다음 소스 | 7/18 · `02-06.FCOS_DETR.md` Part 1/2 · FCOS 구현 |

## 오늘의 질문

SSD의 hard-negative mining은 어려운 배경 일부를 선택한다. RetinaNet의 Focal Loss는 모든 유효 anchor를 남기되 쉬운 표본의 기여를 연속적으로 줄인다. 오늘은 이 차이를 다음 실행 경로로 고정한다.

```text
P3...P7 -> shared classification/box subnet -> flatten
        -> positive/negative/ignore targets
        -> stable sigmoid Focal Loss + Smooth L1
        -> positive-count normalization -> backward
        -> score filter -> decode -> NMS -> deployment
```

1회차 문서는 $p_t$, $\alpha_t$, gradient와 RetinaNet shape를 기초부터 유도했다. 이번 글은 같은 설명을 반복하지 않고 다음 구현 경계를 검증한다.

- `ignore=-1`이 배경 0으로 조용히 바뀌지 않는가?
- positive가 없는 image에서도 negative classification gradient를 보존하는가?
- classification과 box subnet이 모든 FPN level에서 실제로 weight를 공유하는가?
- prior bias의 부호와 초기 확률이 정확한가?
- hard-negative mining과 Focal Loss를 같은 예산과 지표로 비교했는가?
- 분산 학습에서 worker별 positive 수 차이가 gradient scale을 바꾸지 않는가?

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. logits에서 수치적으로 안정한 sigmoid Focal Loss를 구현한다.
2. target별 $\alpha_t$와 `ignore` mask를 독립적으로 테스트한다.
3. RetinaNet 공유 subnet의 level별 tensor를 같은 anchor 순서로 펼친다.
4. classification과 regression을 global positive 수로 정규화한다.
5. prior bias, 빈 image, 극단 logit, mixed precision 실패를 재현한다.
6. hard-negative BCE와 Focal Loss ablation을 동일 데이터로 비교한다.
7. Python, C++17, C#에서 같은 scalar golden 값을 얻는다.
8. 학습 graph와 배포 graph의 책임을 분리한다.

## 선수 지식과 기호

- binary cross entropy와 sigmoid
- FPN의 `NCHW` feature와 anchor flatten 순서
- positive, negative, ignore anchor matching
- box delta와 Smooth L1 loss
- PyTorch optimizer, autograd, mixed precision

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $K$ | foreground class 수 |
| $A$ | 위치당 anchor 수 |
| $M$ | 모든 level의 총 anchor 수 |
| $z_{ik}$ | anchor $i$, class $k$의 logit |
| $y_{ik}$ | binary class target |
| $v_i$ | anchor 유효 여부, ignore이면 0 |
| $r_i$ | regression target을 가진 positive 여부 |
| $N_+$ | positive anchor 수 |
| $\pi$ | 초기 foreground prior |

RetinaNet식 분류는 background를 별도 class로 둔 softmax가 아니라 $K$개 독립 sigmoid를 사용한다. 논리 shape는 class가 `[N,M,K]`, box가 `[N,M,4]`다.

## 1. 원본과 1회차에서 이번에 확장하는 것

원본 [02-05.YOLO.md](../05.ImageClassification/02-05.YOLO.md)는 극단적 foreground-background 불균형과 Focal Loss의 방향을 소개한다. 실행 가능한 학습기로 옮기려면 다음을 고쳐야 한다.

| 원본 또는 흔한 축약 | 이번 구현 계약 |
| --- | --- |
| 모든 target에 `alpha * loss` 적용 | positive에는 $\alpha$, negative에는 $1-\alpha$인 $\alpha_t$를 적용한다. |
| `exp(-BCE)`로 언제나 $p_t$ 복원 | class weight나 label smoothing이 없는 unreduced BCE에서만 직접 성립한다. 여기서는 logit에서 sigmoid를 따로 계산한다. |
| 쉬운 배경을 수학적으로 소거 | 유한 logit에서는 0이 아니라 작아진다. dtype underflow와 수학적 0을 구분한다. |
| RetinaNet은 Focal Loss만 바꾼 SSD | FPN, 공유 subnet, 독립 sigmoid, anchor 설계, prior bias, 정규화까지 함께 계약한다. |
| `mean()` reduction 사용 | anchor 수가 바뀌면 scale이 달라진다. 유효 loss의 합을 global positive 수로 나눈다. |
| ignore를 target 0으로 치환 | ignore anchor는 classification과 regression 모두에서 mask로 제외한다. |
| positive가 0이면 loss도 0 | 정책에 따라 negative classification은 유지하고 분모만 1로 clamp한다. |
| Focal Loss가 모든 불균형 해결 | label noise, class long tail, calibration은 별도의 문제다. |

원문의 저자 귀속도 바로잡는다. RetinaNet·Focal Loss는 Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár의 공동 연구다. 또한 당시 대표적인 2-stage detector와 경쟁한 결과를 모든 backbone과 protocol에 대한 절대적 우월성으로 일반화하지 않는다.

## 2. logits에서 Focal Loss를 다시 조립하기

### 2.1 stable BCE

logit $z$에 대한 binary cross entropy는 확률을 먼저 반올림하지 않고 다음처럼 계산할 수 있다.

$$
\operatorname{BCE}(z,y)
=
\max(z,0)-zy+\log(1+\exp(-|z|))
$$

sigmoid 확률과 정답 class 확률은 다음과 같다.

$$
p=\sigma(z),\qquad
p_t=yp+(1-y)(1-p)
$$

target별 balancing weight는 다음과 같다.

$$
\alpha_t=\alpha y+(1-\alpha)(1-y)
$$

최종 Focal Loss는 다음과 같다.

$$
\operatorname{FL}(z,y)
=
\alpha_t(1-p_t)^\gamma\operatorname{BCE}(z,y)
$$

$\gamma=0$이면 alpha-balanced BCE다. $\alpha$를 `None`으로 두면 $\alpha_t=1$인 unbalanced Focal Loss가 된다.

### 2.2 수작업 golden 값

$p_t=0.9$, $\gamma=2$인 쉬운 positive를 생각하자. $\alpha=0.25$이면 다음과 같다.

$$
\operatorname{FL}
=0.25\times(1-0.9)^2\times[-\log(0.9)]
\approx0.000263401
$$

같은 난이도의 negative는 $\alpha_t=0.75$이므로 약 `0.000790204`다. 모든 표본에 `0.25`를 곱하는 원본 축약은 이 차이를 잃는다.

### 2.3 mask와 정규화

anchor state를 `positive=1`, `negative=0`, `ignore=-1`로 저장한다고 하자. classification 유효 mask는 다음과 같다.

$$
v_i=\mathbb{1}[s_i\ne-1]
$$

positive anchor에서는 해당 class만 1이고 나머지 $K-1$ class는 0이다. negative anchor에서는 $K$개가 모두 0이다. ignore anchor는 어떤 class loss에도 참여하지 않는다.

$$
L_{cls}
=
\frac{
\sum_i v_i\sum_{k=1}^{K}\operatorname{FL}(z_{ik},y_{ik})
}{\max(1,N_+)}
$$

regression은 positive만 계산한다.

$$
L_{box}
=
\frac{
\sum_i r_i\operatorname{SmoothL1}(d_i,t_i)
}{\max(1,N_+)}
$$

전체 손실은 $L=L_{cls}+\lambda_{box}L_{box}$다.

## 3. NumPy oracle: 값, gradient, mask를 독립 검증하기

다음은 **실행 가능한 독립 검증 예제**다. PyTorch 구현을 부르지 않고 stable loss와 central finite difference를 비교한다.

```python
import numpy as np


def softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def focal_logits(logits, targets, alpha=0.25, gamma=2.0):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    bce = softplus(logits) - targets * logits
    probability = sigmoid(logits)
    pt = targets * probability + (1.0 - targets) * (1.0 - probability)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return alpha_t * np.power(1.0 - pt, gamma) * bce


logits = np.array([2.197224577, -2.197224577, -2.197224577, 2.197224577])
targets = np.array([1.0, 0.0, 1.0, 0.0])
loss = focal_logits(logits, targets)
np.testing.assert_allclose(
    loss,
    [0.000263401, 0.000790204, 0.466273481, 1.398820444],
    rtol=1e-6,
    atol=1e-9,
)

valid = np.array([True, True, False, True])
masked_sum = loss[valid].sum()
assert np.isclose(masked_sum, loss[0] + loss[1] + loss[3])

epsilon = 1e-6
numeric_gradient = np.empty_like(logits)
for index in range(logits.size):
    plus = logits.copy()
    minus = logits.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    numeric_gradient[index] = (
        focal_logits(plus, targets).sum()
        - focal_logits(minus, targets).sum()
    ) / (2.0 * epsilon)

assert np.isfinite(numeric_gradient).all()
assert numeric_gradient[0] < 0.0
assert numeric_gradient[1] > 0.0
assert abs(numeric_gradient[0]) < abs(numeric_gradient[2])
assert abs(numeric_gradient[1]) < abs(numeric_gradient[3])
print("numpy focal oracle passed:", np.round(loss, 9).tolist())
```

## 4. RetinaNet subnet과 tensor shape

### 4.1 level별 raw shape

FPN level $l$의 입력은 다음과 같다.

$$
P_l\in\mathbb{R}^{N\times C\times H_l\times W_l}
$$

classification과 regression raw 출력은 다음과 같다.

$$
Z_l^{cls}\in\mathbb{R}^{N\times(AK)\times H_l\times W_l}
$$

$$
Z_l^{box}\in\mathbb{R}^{N\times(4A)\times H_l\times W_l}
$$

anchor-major channel을 명시적으로 복원한다.

```text
classification:
[N, A*K, H, W] -> [N, A, K, H, W]
-> [N, H, W, A, K] -> [N, H*W*A, K]

regression:
[N, A*4, H, W] -> [N, A, 4, H, W]
-> [N, H, W, A, 4] -> [N, H*W*A, 4]
```

`reshape` 결과가 맞더라도 channel interleave 규칙이 anchor generator와 다르면 class와 box가 다른 anchor를 가리킨다. 각 level의 첫 위치에 anchor ID를 심는 synthetic test가 필요하다.

### 4.2 공유와 비공유를 구분하기

RetinaNet은 같은 classification subnet instance를 $P_3$부터 $P_7$까지 반복 호출한다. level마다 새 module을 생성하지 않는다. classification과 box subnet끼리는 parameter를 공유하지 않는다.

세 level이 `17 x 21`, `9 x 11`, `5 x 6`이고 $A=3$, $K=4$이면 다음과 같다.

| level | feature | anchor 수 | class raw | box raw |
| --- | --- | --- | --- | --- |
| `P3` | `[N,C,17,21]` | 1,071 | `[N,12,17,21]` | `[N,12,17,21]` |
| `P4` | `[N,C,9,11]` | 297 | `[N,12,9,11]` | `[N,12,9,11]` |
| `P5` | `[N,C,5,6]` | 90 | `[N,12,5,6]` | `[N,12,5,6]` |
| 합계 | - | 1,458 | `[N,1458,4]` | `[N,1458,4]` |

이 예제에서는 우연히 $AK=4A=12$다. raw channel 수가 같다고 두 head를 혼동하면 안 된다.

## 5. 실행 가능한 PyTorch 학습·테스트 구현

다음은 **실행 가능한 교육용 구현**이다. 실제 backbone·matcher·anchor decode는 Part 2의 계약을 재사용하고, 오늘은 head, loss, ignore, prior, 재현성에 집중한다.

```python
import math
import torch
import torch.nn.functional as F
from torch import nn


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    if not torch.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("targets must be binary after applying the ignore mask")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = targets * probability + (1.0 - targets) * (1.0 - probability)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return alpha_t * (1.0 - pt).pow(gamma) * bce


class SharedRetinaHead(nn.Module):
    def __init__(self, channels, anchors, classes, prior=0.01):
        super().__init__()
        if not 0.0 < prior < 1.0:
            raise ValueError("prior must be in (0, 1)")
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
        prior_bias = math.log(prior / (1.0 - prior))
        nn.init.zeros_(self.cls_tower[-1].weight)
        nn.init.constant_(self.cls_tower[-1].bias, prior_bias)

    def flatten_cls(self, tensor):
        n, _, height, width = tensor.shape
        tensor = tensor.reshape(n, self.anchors, self.classes, height, width)
        return tensor.permute(0, 3, 4, 1, 2).reshape(
            n, height * width * self.anchors, self.classes
        )

    def flatten_box(self, tensor):
        n, _, height, width = tensor.shape
        tensor = tensor.reshape(n, self.anchors, 4, height, width)
        return tensor.permute(0, 3, 4, 1, 2).reshape(
            n, height * width * self.anchors, 4
        )

    def forward(self, pyramid):
        cls_levels = []
        box_levels = []
        for feature in pyramid:
            cls_levels.append(self.flatten_cls(self.cls_tower(feature)))
            box_levels.append(self.flatten_box(self.box_tower(feature)))
        return torch.cat(cls_levels, dim=1), torch.cat(box_levels, dim=1)


def retina_loss(
    cls_logits,
    box_delta,
    class_target,
    box_target,
    anchor_state,
    alpha=0.25,
    gamma=2.0,
):
    if anchor_state.shape != cls_logits.shape[:2]:
        raise ValueError("anchor_state shape mismatch")
    if class_target.shape != anchor_state.shape:
        raise ValueError("class_target shape mismatch")
    valid = anchor_state >= 0
    positive = anchor_state == 1
    targets = torch.zeros_like(cls_logits)
    positive_index = positive.nonzero(as_tuple=False)
    if positive_index.numel() > 0:
        labels = class_target[positive].long()
        if not torch.all((labels >= 0) & (labels < cls_logits.shape[-1])):
            raise ValueError("positive class id out of range")
        targets[positive_index[:, 0], positive_index[:, 1], labels] = 1.0

    per_class = sigmoid_focal_loss(cls_logits, targets, alpha, gamma)
    classification_sum = per_class[valid].sum()
    positive_count = positive.sum().to(cls_logits.dtype)
    denominator = positive_count.clamp(min=1.0)

    if positive.any():
        regression_sum = F.smooth_l1_loss(
            box_delta[positive], box_target[positive], reduction="sum"
        )
    else:
        regression_sum = box_delta.sum() * 0.0
    return (
        classification_sum / denominator,
        regression_sum / denominator,
        positive_count,
    )


def run_once(seed):
    set_deterministic(seed)
    head = SharedRetinaHead(channels=4, anchors=2, classes=3, prior=0.01)
    initial_prior = torch.sigmoid(head.cls_tower[-1].bias.detach())
    torch.testing.assert_close(initial_prior, torch.full_like(initial_prior, 0.01))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    pyramid = [
        torch.randn(2, 4, 5, 7),
        torch.randn(2, 4, 3, 4),
        torch.randn(2, 4, 2, 2),
    ]
    anchor_count = 2 * (5 * 7 + 3 * 4 + 2 * 2)
    class_target = torch.full((2, anchor_count), -1, dtype=torch.long)
    box_target = torch.zeros(2, anchor_count, 4)
    anchor_state = torch.zeros(2, anchor_count, dtype=torch.long)
    anchor_state[0, 1] = 1
    class_target[0, 1] = 2
    box_target[0, 1] = torch.tensor([0.2, -0.1, 0.05, 0.3])
    anchor_state[0, 2] = -1
    anchor_state[1, 0] = -1

    history = []
    for _ in range(3):
        cls_logits, box_delta = head(pyramid)
        cls_loss, box_loss, count = retina_loss(
            cls_logits,
            box_delta,
            class_target,
            box_target,
            anchor_state,
        )
        total = cls_loss + box_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        history.append(
            (
                float(cls_loss.detach()),
                float(box_loss.detach()),
                int(count.detach()),
            )
        )
    state = {name: value.detach().clone() for name, value in head.state_dict().items()}
    return history, state, head, pyramid


history_a, state_a, head, pyramid = run_once(17)
history_b, state_b, _, _ = run_once(17)
assert history_a == history_b
assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)

cls_logits, box_delta = head(pyramid)
expected = 2 * (5 * 7 + 3 * 4 + 2 * 2)
assert cls_logits.shape == (2, expected, 3)
assert box_delta.shape == (2, expected, 4)

# ignore anchor는 logit을 크게 바꿔도 loss에 영향을 주지 않는다.
test_logits = torch.zeros(1, 3, 2, dtype=torch.float64)
test_boxes = torch.zeros(1, 3, 4, dtype=torch.float64)
test_class = torch.tensor([[1, -1, -1]])
test_box_target = torch.zeros_like(test_boxes)
test_state = torch.tensor([[1, -1, 0]])
base = retina_loss(
    test_logits, test_boxes, test_class, test_box_target, test_state
)[0]
changed = test_logits.clone()
changed[0, 1] = torch.tensor([100.0, -100.0])
masked = retina_loss(
    changed, test_boxes, test_class, test_box_target, test_state
)[0]
torch.testing.assert_close(base, masked)

# positive가 없는 image도 유효 negative classification gradient를 만든다.
empty_logits = torch.zeros(1, 4, 2, requires_grad=True)
empty_boxes = torch.zeros(1, 4, 4, requires_grad=True)
empty_state = torch.zeros(1, 4, dtype=torch.long)
empty_class = torch.full((1, 4), -1, dtype=torch.long)
empty_target = torch.zeros_like(empty_boxes)
empty_cls, empty_box, empty_count = retina_loss(
    empty_logits,
    empty_boxes,
    empty_class,
    empty_target,
    empty_state,
)
(empty_cls + empty_box).backward()
assert empty_count.item() == 0
assert empty_logits.grad.abs().sum() > 0
assert empty_boxes.grad.abs().sum() == 0

# 극단 logit도 finite다.
extreme = torch.tensor([-1000.0, 1000.0], requires_grad=True)
extreme_target = torch.tensor([0.0, 1.0])
extreme_loss = sigmoid_focal_loss(extreme, extreme_target).sum()
extreme_loss.backward()
assert torch.isfinite(extreme_loss)
assert torch.isfinite(extreme.grad).all()
print("retina tests passed:", history_a)
```

### 구현 범위

이 코드는 head·loss contract를 실제 실행한다. 완전한 RetinaNet 학습에는 Part 2의 anchor generator, IoU matcher, box encode/decode, 데이터 loader와 평가기가 추가되어야 한다. 교육용 feature를 직접 만들었으므로 backbone 정확도 benchmark를 주장하지 않는다.

## 6. hard-negative mining과 Focal Loss ablation

두 방법을 비교할 때는 모델, initialization, batch, optimizer, positive normalization을 같게 둔다.

### 6.1 비교 대상

| 설정 | negative 처리 | 분류 loss |
| --- | --- | --- |
| `BCE-all` | 모든 유효 negative | BCE |
| `BCE-HNM` | loss가 큰 negative를 positive당 3개 선택 | BCE |
| `Focal` | 모든 유효 negative | $\alpha=0.25$, $\gamma=2$ |
| `Focal-g0` | 모든 유효 negative | alpha-balanced BCE |

### 6.2 반드시 기록할 지표

- positive·negative·ignore anchor 수
- 선택된 negative 수와 score 분포
- positive와 negative의 loss 합 및 gradient norm
- classification recall, false positives per image, AP
- 같은 seed 재실행의 loss history와 parameter hash
- step latency와 peak memory

HNM은 선택 경계에서 작은 logit 변화가 sample 포함 여부를 바꿀 수 있다. Focal Loss는 연속적인 weight를 주지만 모든 유효 anchor의 elementwise loss를 계산하므로 중간 tensor 메모리가 더 들 수 있다.

### 6.3 해석의 함정

초기 prior가 HNM에는 `0.01`, Focal에는 `0.5`라면 loss 비교는 무효다. 또한 `mean()` BCE와 positive-normalized Focal Loss를 비교하면 gradient scale 차이를 방법 차이로 오해한다. 최종 평가는 loss 숫자가 아니라 같은 evaluator의 AP와 운영 threshold에서 비교한다.

## 7. C++17 scalar golden 구현

다음은 **C++17에서 실행 가능한 예제**다. tensor runtime을 대체하는 배포 코드는 아니며 scalar loss golden을 고정한다.

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

double sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    const double value = std::exp(z);
    return value / (1.0 + value);
}

double focal(double z, int target, double alpha = 0.25, double gamma = 2.0) {
    assert(target == 0 || target == 1);
    const double y = static_cast<double>(target);
    const double probability = sigmoid(z);
    const double pt = y * probability + (1.0 - y) * (1.0 - probability);
    const double alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha);
    const double bce = std::max(z, 0.0) - y * z
        + std::log1p(std::exp(-std::abs(z)));
    return alpha_t * std::pow(1.0 - pt, gamma) * bce;
}

int main() {
    const double easy_positive = focal(2.197224577, 1);
    const double easy_negative = focal(-2.197224577, 0);
    const double hard_positive = focal(-2.197224577, 1);
    const double hard_negative = focal(2.197224577, 0);
    assert(easy_positive < hard_positive);
    assert(easy_negative < hard_negative);
    assert(std::abs(easy_negative / easy_positive - 3.0) < 1e-5);
    std::cout << std::fixed << std::setprecision(9)
              << easy_positive << " " << easy_negative << " "
              << hard_positive << " " << hard_negative << "\n";
}
```

## 8. C# scalar golden 구현

다음은 **C#에서 실행 가능한 예제**다. C++과 같은 double precision golden을 사용한다.

```csharp
using System;

public static class FocalGolden
{
    static double Sigmoid(double z)
    {
        if (z >= 0.0) return 1.0 / (1.0 + Math.Exp(-z));
        double value = Math.Exp(z);
        return value / (1.0 + value);
    }

    static double Focal(
        double z,
        int target,
        double alpha = 0.25,
        double gamma = 2.0)
    {
        if (target != 0 && target != 1) throw new ArgumentException("binary target");
        double y = target;
        double probability = Sigmoid(z);
        double pt = y * probability + (1.0 - y) * (1.0 - probability);
        double alphaT = y * alpha + (1.0 - y) * (1.0 - alpha);
        double bce = Math.Max(z, 0.0) - y * z
            + Math.Log(1.0 + Math.Exp(-Math.Abs(z)));
        return alphaT * Math.Pow(1.0 - pt, gamma) * bce;
    }

    public static void Main()
    {
        double easyPositive = Focal(2.197224577, 1);
        double easyNegative = Focal(-2.197224577, 0);
        double hardPositive = Focal(-2.197224577, 1);
        double hardNegative = Focal(2.197224577, 0);
        if (!(easyPositive < hardPositive && easyNegative < hardNegative))
            throw new Exception("difficulty ordering failed");
        if (Math.Abs(easyNegative / easyPositive - 3.0) >= 1e-5)
            throw new Exception("alpha_t ratio failed");
        Console.WriteLine(
            $"{easyPositive:F9} {easyNegative:F9} " +
            $"{hardPositive:F9} {hardNegative:F9}");
    }
}
```

## 9. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch Python | C++/LibTorch 또는 ONNX Runtime | C# runtime | 계약 |
| --- | --- | --- | --- | --- |
| feature | `NCHW`, `float32` | 보통 `NCHW` | API별 확인 | level 순서 `P3...P7` 고정 |
| class raw | `[N,AK,H,W]` | 같은 logical shape | flattened buffer 가능 | anchor-major channel |
| class logical | `[N,M,K]` | contiguous tensor | row-major array | background channel 없음 |
| box logical | `[N,M,4]` | contiguous tensor | row-major array | `tx,ty,tw,th` 순서 |
| anchor state | `int64` | `int64` 또는 `int32` | `long` 또는 `int` | `-1/0/1` 의미 고정 |
| loss accumulate | `float32` 또는 `float64` | training runtime만 | training runtime만 | positive 수 정규화 |
| inference score | sigmoid `float32` | fused sigmoid 가능 | 동일 | threshold 전에 sigmoid 한 번 |

ONNX export에는 보통 head의 raw logits와 box delta까지만 포함한다. Focal Loss, matcher, positive normalization은 학습 graph의 책임이다. 후처리를 graph 밖에 두면 anchor metadata, variance, score threshold, NMS 규칙을 model artifact와 함께 versioning한다.

## 10. 테스트와 디버깅

### 10.1 최소 단위 테스트

- $\gamma=0$에서 alpha-balanced BCE와 같은가?
- positive와 negative의 같은 $p_t$에 서로 다른 $\alpha_t$가 적용되는가?
- `ignore` logit을 바꿔도 loss와 gradient가 같은가?
- positive가 0일 때 NaN 없이 negative gradient가 남는가?
- box regression은 positive에만 gradient를 주는가?
- prior bias의 sigmoid가 정확히 $\pi$인가?
- level 순서를 바꾸면 contract test가 실패하는가?
- 동일 seed 재실행이 같은 history와 parameter를 만드는가?

### 10.2 증상별 디버깅 지도

| 증상 | 먼저 볼 값 | 흔한 원인 |
| --- | --- | --- |
| 첫 step loss 폭발 | 초기 class probability | prior bias 누락 또는 부호 반대 |
| positive recall 0 | matcher와 class target | positive가 ignore 또는 background로 변환됨 |
| loss가 batch마다 크게 출렁임 | worker별 $N_+$ | local positive 수로 각각 정규화 |
| background score가 내려가지 않음 | empty image gradient | positive 0일 때 전체 loss를 0으로 만듦 |
| box head gradient 0 | positive mask와 target | matcher index와 flatten 순서 불일치 |
| AP는 낮고 loss는 감소 | decode와 evaluator | anchor 순서, variance, class별 NMS 불일치 |
| fp16에서 loss 0 | modulating factor dtype | 쉬운 표본의 $(1-p_t)^\gamma$ underflow |
| 희귀 class recall 악화 | class별 positive 수 | foreground-background 문제와 long tail 혼동 |

### 10.3 gradient 검증

작은 `float64` tensor에서는 `torch.autograd.gradcheck`를 사용할 수 있다. 단, mask와 target은 고정하고 logit만 미분한다. matcher의 `argmax` 같은 이산 선택을 gradcheck 대상에 넣지 않는다.

## 11. 성능·메모리·수치 안정성

### 11.1 중간 tensor 메모리

`[N,M,K]`의 BCE, $p_t$, modulating factor, $\alpha_t$를 모두 저장하면 class logit의 여러 배가 된다. 예를 들어 $N=2$, $M=100000$, $K=80$, `float32` tensor 하나는 약 64 MB다. elementwise 식을 compile·fuse하거나 chunk 단위로 계산하면 peak memory를 줄일 수 있다.

### 11.2 mixed precision

큰 음수·양수 logit의 BCE는 `binary_cross_entropy_with_logits`로 안정화한다. 그러나 $(1-p_t)^\gamma$는 `float16`에서 쉬운 표본에 대해 0으로 underflow할 수 있다. 이것이 항상 오류는 아니지만 gradient 통계가 달라질 수 있다. loss 내부만 `float32`로 승격하는 ablation을 측정한다.

### 11.3 분산 정규화

worker $w$의 loss 합을 $S_w$, positive 수를 $n_w$라 하자. 각 worker가 $S_w/\max(1,n_w)$를 만든 뒤 평균하면 positive가 적은 worker가 과대 대표될 수 있다. 원하는 global objective가 다음이라면 합과 count의 all-reduce 의미를 맞춰야 한다.

$$
L_{global}
=
\frac{\sum_w S_w}{\max(1,\sum_w n_w)}
$$

DDP가 gradient를 worker 수로 평균하는 동작까지 포함해 scale을 단위 테스트한다.

### 11.4 profile 항목

- level별 head convolution latency
- flatten·concatenate copy 유무
- Focal Loss forward·backward latency
- saved tensor peak memory
- score threshold 전후 후보 수
- NMS class 수와 latency

## 12. 실무 실패 사례

### 사례 A: ignore를 background로 학습

IoU 경계 anchor와 crowd 영역을 0 target으로 바꾸자 background false confidence는 줄었지만 정답 주변 recall도 함께 무너졌다.

**방지:** state tensor와 one-hot target을 분리하고 valid mask 적용 전후 개수를 log한다.

### 사례 B: worker별 local normalization

빈 image가 많은 worker는 분모가 1이고 다른 worker는 수백이라 gradient 기여가 불균형해졌다.

**방지:** global positive count를 쓰거나 sampler가 worker별 밀도를 균등하게 만들고 수학적 objective를 테스트한다.

### 사례 C: prior bias만 checkpoint에서 누락

head weight는 복원했지만 마지막 bias key를 strict하지 않게 놓쳤다. 재시작 첫 step의 negative loss가 다시 폭발했다.

**방지:** `strict=True`, schema version, sigmoid(bias) smoke test를 배포 전 수행한다.

### 사례 D: Focal Loss가 noisy label에 집중

잘못 붙은 상자는 계속 hard example로 남는다. 쉬운 정상 표본이 줄어들수록 annotation noise의 상대 영향이 커질 수 있다.

**방지:** high-loss sample audit, box quality score, robust loss ablation을 별도로 한다.

### 사례 E: 학습 score를 확률로 해석

Focal Loss 모델에 threshold `0.5`를 관성적으로 적용해 recall이 급락했다.

**방지:** validation precision-recall curve로 class별 threshold를 선택하고 ECE·reliability diagram으로 calibration을 점검한다.

## 13. 배포 관점

학습 artifact에는 다음을 저장한다.

- `class_count`, class ID mapping
- FPN level과 anchor 순서
- anchor scale·ratio·variance
- head channel interleave schema
- prior $\pi$, $\alpha$, $\gamma$, box loss weight
- matcher threshold와 ignore 정책
- normalization과 distributed world-size 의미
- preprocessing, resize, padding, dtype
- evaluator와 NMS version

추론 artifact는 raw logits와 box delta를 낸다. runtime은 sigmoid, decode, clip, threshold, class별 NMS를 같은 metadata로 실행한다. Focal Loss 자체는 추론에 필요하지 않다.

모니터링에서는 전체 score 평균만 보지 않는다. class별 후보 수, score quantile, NMS 전후 개수, 빈 prediction 비율, 객체 크기별 recall proxy, 입력 해상도와 padding 비율을 함께 본다.

## 14. 체크리스트

### 수학과 target

- [ ] $\alpha_t$가 positive에는 $\alpha$, negative에는 $1-\alpha$인가?
- [ ] ignore anchor가 모든 class loss에서 빠지는가?
- [ ] positive anchor의 한 class만 1인가?
- [ ] regression은 positive에만 적용되는가?
- [ ] positive 0 정책과 분모가 명시됐는가?

### shape와 구현

- [ ] classification과 box의 flatten 순서가 같은가?
- [ ] 모든 FPN level에서 subnet instance를 공유하는가?
- [ ] class와 box subnet parameter는 분리됐는가?
- [ ] prior bias sigmoid가 $\pi$인가?
- [ ] extreme logit과 mixed precision이 finite인가?

### 재현성과 평가

- [ ] seed, manifest, deterministic 설정을 저장했는가?
- [ ] same-seed replay를 parameter까지 비교했는가?
- [ ] HNM/BCE/Focal이 같은 normalization으로 비교됐는가?
- [ ] AP evaluator와 NMS 규칙이 versioning됐는가?
- [ ] class별 calibration과 threshold를 검증했는가?

### 배포

- [ ] ONNX output의 layout·dtype·dynamic axis가 문서화됐는가?
- [ ] anchor metadata가 model과 원자적으로 배포되는가?
- [ ] sigmoid를 정확히 한 번 적용하는가?
- [ ] 학습과 추론의 resize·padding이 같은가?
- [ ] 운영 drift metric과 rollback 기준이 있는가?

## 15. 연습문제

### 문제 1

$y=0$, foreground 확률 $p=0.1$, $\alpha=0.25$, $\gamma=2$일 때 $p_t$, $\alpha_t$, modulating factor를 구하라.

### 문제 2

`[N, A*K, H, W] = [2, 15, 10, 12]`, $A=3$일 때 $K$와 flatten shape를 구하라.

### 문제 3

positive 4개, 유효 negative 100개, ignore 20개가 있다. 분류 loss 합이 12이고 regression loss 합이 2일 때 $\lambda_{box}=1$인 전체 loss를 구하라.

### 문제 4

$\pi=0.05$인 prior bias를 구하라.

### 문제 5

positive가 없는 image에서 negative classification까지 0으로 만들면 어떤 학습 신호를 잃는가?

### 문제 6

worker A의 loss 합과 positive 수가 `(10, 1)`, worker B가 `(30, 9)`다. local-normalize 후 평균과 global-normalize 값을 비교하라.

### 문제 7

Focal Loss와 hard-negative mining의 핵심 구현 차이를 한 문장씩 설명하라.

### 문제 8

Focal Loss 학습 모델의 score threshold를 `0.5`로 고정하면 안 되는 이유를 설명하라.

## 16. 해답

### 해답 1

negative의 정답 확률은 $p_t=1-p=0.9$다. $\alpha_t=1-\alpha=0.75$이고 modulating factor는 $(1-0.9)^2=0.01$이다.

### 해답 2

$K=15/3=5$다. flatten shape는 `[2, 10*12*3, 5]`, 즉 `[2,360,5]`다.

### 해답 3

분모는 $\max(1,4)=4$다. 따라서 $L_{cls}=12/4=3$, $L_{box}=2/4=0.5$, 전체는 `3.5`다. ignore 수는 분모에 들어가지 않는다.

### 해답 4

$$
b=\log\left(\frac{0.05}{0.95}\right)\approx-2.94444
$$

### 해답 5

객체가 없는 image에서 foreground score를 낮추는 신호를 잃는다. dataset에 빈 image가 많다면 false positive 억제가 특히 약해질 수 있다.

### 해답 6

local-normalize 후 평균은 다음과 같다.

$$
\frac{10/1+30/9}{2}\approx6.6667
$$

global-normalize는 다음과 같다.

$$
\frac{10+30}{1+9}=4
$$

두 objective는 같지 않다.

### 해답 7

hard-negative mining은 ranking으로 일부 negative를 이산 선택한다. Focal Loss는 모든 유효 negative를 유지하면서 현재 $p_t$에 따라 연속적인 weight를 곱한다.

### 해답 8

Focal Loss와 prior bias는 score calibration을 바꿀 수 있다. validation precision-recall trade-off와 운영 비용으로 class별 threshold를 다시 정해야 한다.

## 핵심 요약

1. Focal Loss는 stable BCE에 target별 $\alpha_t$와 $(1-p_t)^\gamma$를 곱한다.
2. 원본의 모든 표본에 동일한 `alpha`를 곱하는 구현은 표준 $\alpha_t$가 아니다.
3. positive, negative, ignore를 target 값 하나에 섞지 말고 state와 one-hot target을 분리한다.
4. classification과 regression loss는 positive 수로 정규화하되 빈 image 정책을 명시한다.
5. RetinaNet subnet은 FPN level 사이에서 공유되며 classification과 box subnet끼리는 분리된다.
6. prior bias $\log(\pi/(1-\pi))$는 초기 negative loss 폭발을 줄인다.
7. HNM과 Focal Loss ablation은 initialization, normalization, evaluator를 같게 해야 한다.
8. mixed precision underflow, distributed count, flatten 순서가 실무의 주요 실패 지점이다.
9. Focal Loss는 label noise, long-tail, calibration을 자동 해결하지 않는다.
10. 학습 graph의 loss와 추론 graph의 sigmoid·decode·NMS 계약을 분리해 versioning한다.

## 다음 학습 예고

다음 실행은 2회차 구현 7/18 `02-06.FCOS_DETR.md`의 Part 1/2다. FCOS의 위치별 `LTRB` target, center sampling, FPN regression range, center-ness, loss normalization과 class별 NMS를 완전한 학습·평가 코드로 연결한다.
