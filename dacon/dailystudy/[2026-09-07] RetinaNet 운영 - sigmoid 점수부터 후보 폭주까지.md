<!-- curriculum: cycle=3; level=production-engineering; source_index=6/18; source=02-05.YOLO.md; part=3/3 -->

# RetinaNet 운영: sigmoid 점수부터 후보 폭주까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-07 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 6/18 · `02-05.YOLO.md` |
| Part | 3/3 · Focal Loss·RetinaNet 운영 계약 |
| 이전 소스 | `02-05.YOLO.md` Part 2/3 · SSD/FPN 다중 출력 ABI |
| 다음 소스 | 7/18 · `02-06.FCOS_DETR.md` Part 1/2 · FCOS 운영 |

## 오늘의 운영 질문

학습이 끝난 RetinaNet을 배포했더니 AP는 같은데 후보 수와 p99 지연이 급증했다. 모델 파일만 되돌리면 해결될까? 반드시 그렇지는 않다. 서비스 결과는 다음 전체 경로의 함수다.

```text
image -> backbone/FPN -> class logits -> sigmoid/calibration
      -> level별 threshold/top-k -> box decode -> NMS -> response
```

오늘은 Focal Loss를 다시 구현하는 데 그치지 않고, 분류 logit이 운영 비용과 품질로 바뀌는 경계를 versioned contract로 만든다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. RetinaNet의 독립 sigmoid 출력과 softmax 출력을 manifest 수준에서 구분한다.
2. Focal Loss와 초기 foreground prior bias를 stable logit 수식으로 연결한다.
3. FPN level별 class tensor의 shape, flatten 순서, candidate 수를 추적한다.
4. FP16·INT8에서 sigmoid와 threshold가 흔들리는 경로를 진단한다.
5. class·level별 threshold를 calibration set에서 정하고 versioning한다.
6. 후보 폭주를 메모리·NMS 복잡도·p99 지연으로 환산한다.
7. Python, C++17, C#에서 동일한 score admission 결과를 검증한다.
8. canary metric과 rollback 단위를 model·calibrator·postprocess bundle로 확장한다.

## 선수 지식과 기호

- binary cross entropy, sigmoid, logit
- Focal Loss의 $p_t$, $\alpha_t$, $\gamma$
- RetinaNet의 FPN과 anchor
- precision, recall, calibration, quantile
- FP16, INT8, ONNX Runtime 같은 추론 runtime

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $K$ | foreground class 수 |
| $A_l$ | level $l$의 위치당 anchor 수 |
| $H_l,W_l$ | level $l$의 공간 크기 |
| $M_l=H_lW_lA_l$ | level $l$의 anchor 수 |
| $z_{ilc}$ | anchor $i$, level $l$, class $c$의 raw logit |
| $p_{ilc}=\sigma(z_{ilc})$ | sigmoid score |
| $\tau_{lc}$ | level·class별 admission threshold |
| $K_l$ | level별 pre-NMS candidate cap |
| $\pi$ | classification head의 초기 foreground prior |
| $Q_l$ | threshold를 통과한 level $l$의 후보 수 |

RetinaNet 분류 head는 background channel을 포함한 softmax가 아니라 $K$개의 독립 sigmoid logit을 낸다. 한 anchor가 여러 class threshold를 동시에 통과할 수 있다는 사실이 후보 예산에 직접 영향을 준다.

## 1. 원본과 앞선 두 회차에서 이번에 확장하는 것

원본 [02-05.YOLO.md](../05.ImageClassification/02-05.YOLO.md)는 쉬운 negative의 총량을 Focal Loss로 줄이는 직관을 제공한다. 1회차 문서는 $p_t$, $\alpha_t$, gradient와 prior bias를 유도했고, 2회차 문서는 ignore mask, positive-count normalization, shared subnet과 재현 학습을 구현했다. 이번 글은 그 학습 산출물을 서비스에 연결한다.

| 원본·과거 회차의 초점 | 오늘의 추가 운영 계약 |
| --- | --- |
| 쉬운 negative의 학습 기여 감소 | 배포 후보 수는 자동으로 작아지지 않으므로 별도 budget을 둔다. |
| raw head와 anchor flatten | raw logit·sigmoid·calibrated score 중 어느 것을 export하는지 고정한다. |
| prior bias로 초기 학습 안정화 | bias와 calibrator를 checkpoint·manifest·양자화 artifact에서 함께 검증한다. |
| FP32 stable loss | FP16 loss reduction과 INT8 inference threshold 이동을 분리해 검사한다. |
| class별 score 관찰 | class·level·size bucket별 score와 후보 funnel을 같은 trace에 남긴다. |
| model checkpoint 배포 | model, anchor, calibrator, threshold, NMS를 하나의 immutable bundle로 배포한다. |

원문의 다음 표현은 운영 관점에서 교정해야 한다.

1. Focal Loss는 쉬운 negative를 수학적으로 완전히 제거하지 않는다. 유한 정밀도에서 underflow로 0이 될 수 있지만 이는 알고리즘의 보장이 아니다.
2. `gamma=2`, `alpha=0.25`는 모든 데이터의 고정 최적값이 아니다. 재학습과 배포 score 분포를 함께 비교해야 한다.
3. Focal Loss로 학습했다고 score가 보정된 확률이 되는 것은 아니다. threshold `0.5`도 보편 기본값이 아니다.
4. RetinaNet은 단순히 SSD의 loss만 바꾼 모델이 아니다. FPN, subnet 공유, independent sigmoid, prior bias와 정규화가 함께 작동한다.
5. 원본 예제처럼 모든 표본에 같은 `alpha`를 곱하면 표준 $\alpha_t$가 아니다. positive에는 $\alpha$, negative에는 $1-\alpha$를 쓴다.
6. `exp(-BCE)`로 $p_t$를 복원하는 방법은 가중치나 label smoothing이 없는 unreduced BCE일 때만 직접 성립한다.

## 2. 직관: loss가 줄인 것은 학습 기여이지 서비스 후보가 아니다

Focal Loss는 학습 중 쉬운 background의 gradient를 줄인다. 추론에서는 loss를 계산하지 않는다. head가 만든 모든 $M_lK$개 logit은 threshold를 적용하기 전까지 여전히 후보다.

초기 prior가 $\pi=0.01$이면 모든 class bias는 대략 `-4.595`다. 초기 sigmoid score는 0.01이다. 서비스 threshold를 실수로 `0.005`로 배포하면 학습이 덜 된 모델에서도 거의 모든 anchor-class 쌍이 통과한다. $100{,}000$ anchors와 80 classes라면 최대 800만 score가 후처리로 밀려온다.

따라서 다음 두 명제를 분리한다.

- Focal Loss는 **학습 gradient 예산**을 재분배한다.
- threshold와 top-k는 **추론 candidate 예산**을 제한한다.

둘은 관련되어 있지만 서로 대체하지 않는다.

## 3. 수학: logit에서 admission까지

### 3.1 stable sigmoid와 BCE

logit $z$의 sigmoid는 다음과 같다.

$$
\sigma(z)=\frac{1}{1+\exp(-z)}
$$

큰 음수에서 `exp(-z)` overflow를 피하려면 부호에 따라 계산을 나눈다.

$$
\sigma(z)=
\begin{cases}
\dfrac{1}{1+\exp(-z)}, & z\ge 0 \\
\dfrac{\exp(z)}{1+\exp(z)}, & z<0
\end{cases}
$$

binary target $y\in\{0,1\}$의 stable BCE는 다음과 같다.

$$
\operatorname{BCE}(z,y)
=\max(z,0)-zy+\log(1+\exp(-|z|))
$$

정답 확률과 class balancing factor는 다음과 같다.

$$
p_t=yp+(1-y)(1-p)
$$

$$
\alpha_t=\alpha y+(1-\alpha)(1-y)
$$

Focal Loss는 다음과 같다.

$$
\operatorname{FL}(z,y)
=\alpha_t(1-p_t)^\gamma\operatorname{BCE}(z,y)
$$

학습에서는 BCE와 reduction을 FP32로 계산하는 것이 안전하다. inference에서는 Focal Loss가 graph에 없어야 하며, raw logit과 sigmoid 중 graph 책임을 manifest에 기록한다.

### 3.2 prior bias

초기 foreground 확률을 $\pi$로 만들려면 bias $b$는 다음 조건을 만족한다.

$$
\sigma(b)=\pi
$$

따라서 다음을 얻는다.

$$
b=\log\left(\frac{\pi}{1-\pi}\right)
$$

$\pi=0.01$이면 다음과 같다.

$$
b=\log\left(\frac{0.01}{0.99}\right)\approx-4.59512
$$

이 값은 학습 초기화 정보이면서 동시에 checkpoint integrity 신호다. classification bias만 기본값 0으로 누락되면 초기 score가 0.5가 되어 candidate가 폭주한다.

### 3.3 calibration과 threshold

검증 데이터에서 temperature $T_c>0$를 class별로 학습했다면 보정 score는 다음과 같다.

$$
\tilde p_{ilc}=\sigma\left(\frac{z_{ilc}}{T_c}\right)
$$

admission indicator는 다음과 같다.

$$
a_{ilc}=\mathbb{1}[\tilde p_{ilc}\ge\tau_{lc}]
$$

level 후보 수는 다음과 같다.

$$
Q_l=\sum_{i=1}^{M_l}\sum_{c=1}^{K}a_{ilc}
$$

같은 positive scalar temperature는 한 class 안의 순위를 바꾸지 않지만 threshold 통과 수는 바꾼다. class마다 $T_c$가 다르면 전역 top-k의 class 간 순서도 달라질 수 있다. calibrator revision과 threshold revision을 분리 배포하면 안 된다.

### 3.4 후보 예산과 복잡도

threshold 뒤 level별 cap을 적용한 수를 다음처럼 둔다.

$$
R_l=\min(Q_l,K_l)
$$

전체 pre-NMS 후보 수는 다음과 같다.

$$
R=\sum_lR_l
$$

단순 pairwise NMS의 최악 비교 횟수는 $O(R^2)$다. score sort도 $O(R\log R)$ 비용을 만든다. 후보 하나가 box 4개, score 1개를 FP32로, class·level·index를 int32로 저장하면 최소 payload는 32 bytes다.

$$
M_{candidate}\ge32R\ \mathrm{bytes}
$$

$R=100{,}000$이면 payload만 약 3.05 MiB다. allocator overhead, sort workspace, device-to-host copy는 별도다.

## 4. tensor shape와 flatten 계약

### 4.1 level별 raw output

FPN feature는 다음 shape를 가진다.

$$
P_l\in\mathbb{R}^{N\times C\times H_l\times W_l}
$$

classification과 box head의 raw output은 다음과 같다.

$$
Z_l^{cls}\in\mathbb{R}^{N\times(A_lK)\times H_l\times W_l}
$$

$$
Z_l^{box}\in\mathbb{R}^{N\times(4A_l)\times H_l\times W_l}
$$

논리 view는 각각 `[N,H_l,W_l,A_l,K]`, `[N,H_l,W_l,A_l,4]`다. `permute` 뒤 reshape해야 하며 channel을 바로 `[A_l,K]`로 해석하려면 export가 anchor-major인지 먼저 확인한다.

### 4.2 수작업 shape 추적

$N=1$, $K=3$, $A_l=2$이고 다음 세 level을 생각하자.

| level | feature shape | class raw | box raw | anchor 수 | score 수 |
| --- | --- | --- | --- | --- | --- |
| `P3` | `[1,8,8,10]` | `[1,6,8,10]` | `[1,8,8,10]` | 160 | 480 |
| `P4` | `[1,8,4,5]` | `[1,6,4,5]` | `[1,8,4,5]` | 40 | 120 |
| `P5` | `[1,8,2,3]` | `[1,6,2,3]` | `[1,8,2,3]` | 12 | 36 |

전체 anchor는 212개, score는 636개다. 같은 anchor가 class 두 개의 threshold를 통과하면 candidate도 두 개다. anchor 수와 candidate 수를 같은 metric으로 부르면 안 된다.

### 4.3 flatten index

anchor-major layout에서 논리 index는 다음과 같다.

$$
j=(((rW_l+c)A_l+a)K)+k
$$

역변환은 다음 순서다.

$$
k=j\bmod K
$$

$$
u=\left\lfloor\frac{j}{K}\right\rfloor
$$

$$
a=u\bmod A_l
$$

$$
v=\left\lfloor\frac{u}{A_l}\right\rfloor,qquad
r=\left\lfloor\frac{v}{W_l}\right\rfloor,qquad
c=v\bmod W_l
$$

Python, C++, C#의 golden test는 score뿐 아니라 이 index 왕복도 검사해야 한다.

## 5. release bundle과 score manifest

```text
retinanet-release-2026-09-07/
  model.onnx
  manifest.json
  anchors.npz
  classes.txt
  calibration.json
  thresholds.json
  golden-logits.npy
  golden-candidates.json
  checksums.sha256
```

이는 **설명용 구조**다. 실제 artifact에는 접근 제어, 서명, content hash가 필요하다.

```json
{
  "schema": "retinanet-serving/v3",
  "model_output": "raw_logits",
  "classification": "independent_sigmoid",
  "class_count": 3,
  "levels": [
    {"name": "P3", "anchors": 2, "candidate_cap": 1200},
    {"name": "P4", "anchors": 2, "candidate_cap": 600},
    {"name": "P5", "anchors": 2, "candidate_cap": 300}
  ],
  "raw_layout": "NCHW-anchor-major",
  "flatten_order": "level-row-column-anchor-class",
  "score_transform": "sigmoid(logit/temperature[class])",
  "calibration_revision": "cal-20260907-a",
  "threshold_revision": "thr-20260907-a",
  "tie_break": "score-desc-level-index-class",
  "nms": "class-wise-iou-0.5",
  "global_detection_cap": 100
}
```

loader는 모르는 schema, class 수 불일치, 누락된 level, 0 이하 temperature, 범위 밖 threshold를 추측해 보정하지 않고 거부한다.

## 6. NumPy·PyTorch 실행 예제

다음은 **실행 가능한 Python 예제**다. 외부 데이터나 비밀정보가 필요 없다. NumPy oracle, mini RetinaNet head, prior bias, shape, Focal Loss, backward, calibration, stable admission을 한 번에 검증한다.

```python
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


torch.manual_seed(20260907)
np.random.seed(20260907)


def stable_sigmoid_numpy(values):
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def stable_admission(logits, temperatures, thresholds, cap):
    logits = np.asarray(logits, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must be [anchor, class]")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("thresholds must be probabilities")
    scores = stable_sigmoid_numpy(logits / temperatures[None, :])
    anchor_index, class_index = np.nonzero(scores >= thresholds[None, :])
    selected_scores = scores[anchor_index, class_index]
    # primary key score descending, then anchor, then class ascending
    order = np.lexsort((class_index, anchor_index, -selected_scores))
    order = order[:cap]
    return np.stack(
        [anchor_index[order], class_index[order]], axis=1
    ), selected_scores[order]


def focal_loss(logits, targets, valid, positive_count, alpha=0.25, gamma=2.0):
    # loss의 핵심 연산과 reduction은 FP32로 승격한다.
    logits = logits.float()
    targets = targets.float()
    valid = valid.bool()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(targets == 1.0, probability, 1.0 - probability)
    alpha_t = torch.where(targets == 1.0, alpha, 1.0 - alpha)
    per_class = alpha_t * (1.0 - pt).pow(gamma) * bce
    return per_class.masked_fill(~valid, 0.0).sum() / max(1, positive_count)


@dataclass(frozen=True)
class Level:
    name: str
    height: int
    width: int
    anchors: int


LEVELS = (
    Level("P3", 8, 10, 2),
    Level("P4", 4, 5, 2),
    Level("P5", 2, 3, 2),
)
CLASS_COUNT = 3
CHANNELS = 8
PRIOR = 0.01


class TinyRetinaHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_conv = nn.Conv2d(CHANNELS, 2 * CLASS_COUNT, 3, padding=1)
        self.box_conv = nn.Conv2d(CHANNELS, 2 * 4, 3, padding=1)
        prior_bias = math.log(PRIOR / (1.0 - PRIOR))
        nn.init.constant_(self.class_conv.bias, prior_bias)

    def forward(self, features):
        outputs = {}
        for level in LEVELS:
            feature = features[level.name]
            outputs[f"cls_{level.name}"] = self.class_conv(feature)
            outputs[f"box_{level.name}"] = self.box_conv(feature)
        return outputs


def flatten_class(raw, anchors, classes):
    batch, channels, height, width = raw.shape
    if channels != anchors * classes:
        raise ValueError("classification channel mismatch")
    return raw.view(batch, anchors, classes, height, width).permute(
        0, 3, 4, 1, 2
    ).contiguous().view(batch, height * width * anchors, classes)


features = {
    level.name: torch.zeros(1, CHANNELS, level.height, level.width)
    for level in LEVELS
}
head = TinyRetinaHead()
outputs = head(features)
flat_logits = []
for level in LEVELS:
    cls = outputs[f"cls_{level.name}"]
    box = outputs[f"box_{level.name}"]
    assert cls.shape == (1, level.anchors * CLASS_COUNT, level.height, level.width)
    assert box.shape == (1, level.anchors * 4, level.height, level.width)
    flat_logits.append(flatten_class(cls, level.anchors, CLASS_COUNT))

flat_logits = torch.cat(flat_logits, dim=1)
assert flat_logits.shape == (1, 212, 3)
initial_probability = torch.sigmoid(head.class_conv.bias.detach())
torch.testing.assert_close(initial_probability, torch.full_like(initial_probability, PRIOR))

# 하나의 positive, 나머지는 negative, 마지막 anchor는 ignore다.
targets = torch.zeros_like(flat_logits)
targets[0, 0, 1] = 1.0
valid = torch.ones_like(flat_logits, dtype=torch.bool)
valid[:, -1, :] = False
loss = focal_loss(flat_logits, targets, valid, positive_count=1)
loss.backward()
assert math.isfinite(loss.item())
assert torch.isfinite(head.class_conv.bias.grad).all()

# 수작업 admission golden: 동점이면 anchor, class 순서다.
golden_logits = np.array([
    [0.0, 2.0, -2.0],
    [2.0, 2.0, -8.0],
    [0.0, -1.0, 3.0],
])
pairs, scores = stable_admission(
    golden_logits,
    temperatures=np.array([1.0, 2.0, 1.0]),
    thresholds=np.array([0.5, 0.7, 0.9]),
    cap=4,
)
assert pairs.tolist() == [[2, 2], [1, 0], [0, 1], [1, 1]]
np.testing.assert_allclose(
    scores,
    [0.9525741268, 0.8807970780, 0.7310585786, 0.7310585786],
    rtol=1e-9,
)

# threshold가 prior 아래면 212 x 3 후보가 모두 통과한다.
prior_logits = np.full((212, 3), math.log(PRIOR / (1.0 - PRIOR)))
all_pairs, _ = stable_admission(
    prior_logits,
    temperatures=np.ones(3),
    thresholds=np.full(3, 0.005),
    cap=1000,
)
assert len(all_pairs) == 636

# per-tensor INT8를 흉내 내어 logit 오차가 threshold 결정을 바꿀 수 있음을 확인한다.
scale = 0.05
near_threshold = np.array([[0.024, -0.024]])
quantized = np.clip(np.rint(near_threshold / scale), -127, 127) * scale
before = stable_sigmoid_numpy(near_threshold)
after = stable_sigmoid_numpy(quantized)
assert (before >= 0.5).tolist() != (after >= 0.5).tolist()

print("flat shape:", tuple(flat_logits.shape))
print("initial probability:", f"{initial_probability[0].item():.6f}")
print("loss:", f"{loss.item():.6f}")
print("golden pairs:", pairs.tolist())
print("prior candidates:", len(all_pairs))
print("quantized logits:", quantized.tolist())
```

예상 핵심 출력은 `flat shape: (1, 212, 3)`, 초기 확률 `0.010000`, prior 아래 threshold의 후보 `636`이다. 마지막 양자화 예제는 0.5 경계 근처에서 두 logit이 0으로 모여 admission이 달라지는 의도적인 실패 재현이다.

## 7. C++17 예제: score ABI와 안정적인 tie-break

다음은 **실행 가능한 C++17 예제**다. inference library와 독립적으로 sigmoid, calibration, threshold, 정렬 계약을 검증한다.

```cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

struct Candidate {
    double score;
    std::size_t anchor;
    std::size_t class_index;
};

double StableSigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exp_value = std::exp(value);
    return exp_value / (1.0 + exp_value);
}

int main() {
    const std::vector<std::vector<double>> logits = {
        {0.0, 2.0, -2.0},
        {2.0, 2.0, -8.0},
        {0.0, -1.0, 3.0},
    };
    const std::vector<double> temperature = {1.0, 2.0, 1.0};
    const std::vector<double> threshold = {0.5, 0.7, 0.9};
    std::vector<Candidate> candidates;
    for (std::size_t anchor = 0; anchor < logits.size(); ++anchor) {
        for (std::size_t class_index = 0; class_index < 3; ++class_index) {
            assert(temperature[class_index] > 0.0);
            const double score = StableSigmoid(
                logits[anchor][class_index] / temperature[class_index]
            );
            if (score >= threshold[class_index]) {
                candidates.push_back({score, anchor, class_index});
            }
        }
    }
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            if (left.score != right.score) return left.score > right.score;
            if (left.anchor != right.anchor) return left.anchor < right.anchor;
            return left.class_index < right.class_index;
        }
    );
    assert(candidates.size() == 6);
    assert(candidates[0].anchor == 2 && candidates[0].class_index == 2);
    assert(candidates[1].anchor == 1 && candidates[1].class_index == 0);
    assert(candidates[2].anchor == 0 && candidates[2].class_index == 1);
    assert(candidates[3].anchor == 1 && candidates[3].class_index == 1);
    std::cout << std::fixed << std::setprecision(6)
              << candidates[0].score << " "
              << candidates[1].score << " "
              << candidates[2].score << " "
              << candidates[3].score << "\n";
}
```

예상 출력은 `0.952574 0.880797 0.731059 0.731059`다.

## 8. C# 예제: 같은 admission 계약

다음은 **실행 가능한 C# 예제**다. 비교 key를 명시해 runtime의 불안정한 기본 정렬에 의존하지 않는다.

```csharp
using System;
using System.Collections.Generic;

public sealed class Candidate
{
    public readonly double Score;
    public readonly int Anchor;
    public readonly int ClassIndex;

    public Candidate(double score, int anchor, int classIndex)
    {
        Score = score;
        Anchor = anchor;
        ClassIndex = classIndex;
    }
}

public static class ScoreContract
{
    static double StableSigmoid(double value)
    {
        if (value >= 0.0)
            return 1.0 / (1.0 + Math.Exp(-value));
        double expValue = Math.Exp(value);
        return expValue / (1.0 + expValue);
    }

    public static void Main()
    {
        double[,] logits = {
            {0.0, 2.0, -2.0},
            {2.0, 2.0, -8.0},
            {0.0, -1.0, 3.0},
        };
        double[] temperature = {1.0, 2.0, 1.0};
        double[] threshold = {0.5, 0.7, 0.9};
        var candidates = new List<Candidate>();
        for (int anchor = 0; anchor < logits.GetLength(0); ++anchor)
            for (int classIndex = 0; classIndex < 3; ++classIndex)
            {
                if (temperature[classIndex] <= 0.0)
                    throw new ArgumentOutOfRangeException("temperature");
                double score = StableSigmoid(
                    logits[anchor, classIndex] / temperature[classIndex]
                );
                if (score >= threshold[classIndex])
                    candidates.Add(new Candidate(score, anchor, classIndex));
            }
        candidates.Sort((left, right) => {
            int scoreOrder = right.Score.CompareTo(left.Score);
            if (scoreOrder != 0) return scoreOrder;
            int anchorOrder = left.Anchor.CompareTo(right.Anchor);
            if (anchorOrder != 0) return anchorOrder;
            return left.ClassIndex.CompareTo(right.ClassIndex);
        });
        if (candidates.Count != 6 ||
            candidates[0].Anchor != 2 || candidates[0].ClassIndex != 2 ||
            candidates[1].Anchor != 1 || candidates[1].ClassIndex != 0 ||
            candidates[2].Anchor != 0 || candidates[2].ClassIndex != 1 ||
            candidates[3].Anchor != 1 || candidates[3].ClassIndex != 1)
            throw new Exception("candidate order mismatch");
        Console.WriteLine(
            "{0:F6} {1:F6} {2:F6} {3:F6}",
            candidates[0].Score,
            candidates[1].Score,
            candidates[2].Score,
            candidates[3].Score
        );
    }
}
```

예상 출력은 C++과 같은 `0.952574 0.880797 0.731059 0.731059`다.

## 9. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch Python | C++ runtime | C# runtime | 고정할 계약 |
| --- | --- | --- | --- | --- |
| 입력 | `[N,3,H,W]` `float32` 또는 AMP | tensor view | `DenseTensor<float>` 등 | RGB/BGR, range, normalization |
| class raw | `[N,A_lK,H_l,W_l]` | output-name lookup | output metadata lookup | raw logit 여부, channel order |
| logical class | `[N,H_l,W_l,A_l,K]` | explicit stride/index | checked `long` index | flatten 순서 |
| box raw | `[N,4A_l,H_l,W_l]` | `float` 또는 `half` | `float` 또는 `Half` | delta field order, anchor revision |
| temperature | FP32 vector `[K]` | `double`/`float` artifact | `double[]`/`float[]` | 양수, class map revision |
| threshold | FP32 `[L,K]` | immutable table | immutable table | score 공간과 inclusive 비교 |
| candidate index | `int64` | `std::int64_t` | `long` | overflow, level prefix |
| final box | 원본 pixel FP32 | 동일 | 동일 | half-open `xyxy`, clip 시점 |

### 9.1 logits와 probability를 이름으로도 구분한다

`cls_P3` 같은 모호한 이름보다 `cls_logits_P3` 또는 `cls_prob_P3`를 사용한다. manifest와 runtime output metadata가 다르면 이중 sigmoid를 적용하거나 sigmoid를 누락할 수 있다.

logit 0에 sigmoid를 한 번 적용하면 0.5지만 두 번 적용하면 약 0.622459다. 빈 장면 후보가 갑자기 늘 수 있는 큰 차이다.

### 9.2 layout conversion

execution provider가 내부적으로 `NHWC`를 써도 외부 ABI는 output metadata로 확인한다. `NCHW` raw를 메모리 복사 없이 잘못 `NHWC`로 읽으면 값은 finite하고 shape 곱도 같아서 단순 검사가 통과할 수 있다. index-pattern golden tensor가 필요하다.

### 9.3 dtype 경계

- 학습: convolution은 autocast 가능하지만 BCE, focal factor, 합계, global positive normalization은 FP32 경로를 검사한다.
- FP16 inference: sigmoid는 극단 logit에서 0 또는 1로 반올림될 수 있다. threshold 근처 logit parity를 별도 측정한다.
- INT8 inference: classification과 box head에 같은 quantization scale을 공유하지 않는다. 가능하면 channel별 scale을 평가한다.
- C#: `Half` 지원이 runtime마다 다르므로 input/output cast 위치를 명시한다.

## 10. FP16·INT8 수치 안정성

### 10.1 FP16 학습

Focal factor $(1-p_t)^\gamma$는 쉬운 표본에서 매우 작다. FP16에서 먼저 0으로 반올림되면 해당 표본의 gradient가 예상보다 일찍 사라진다. 해결 순서는 다음과 같다.

1. logits를 FP32로 승격해 BCE와 sigmoid를 계산한다.
2. unreduced loss를 FP32로 합산한다.
3. global positive count를 FP32 분모로 사용한다.
4. gradient scaler의 overflow·skip count를 기록한다.

### 10.2 INT8 inference

logit quantization을 다음처럼 쓰자.

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{z}{s}\right)+z_0,q_{min},q_{max}\right)
$$

$$
\hat z=s(q-z_0)
$$

sigmoid의 국소 민감도는 다음과 같다.

$$
\frac{d\sigma}{dz}=p(1-p)
$$

logit 오차가 $|\Delta z|$일 때 작은 오차 범위의 score 변화는 대략 다음과 같다.

$$
|\Delta p|\approx p(1-p)|\Delta z|
$$

$p=0.5$에서 민감도가 최대 0.25다. score threshold 부근의 logit을 calibration dataset이 충분히 포함하지 않으면 전체 tensor MSE는 작아도 candidate churn은 클 수 있다.

### 10.3 양자화 release metric

단순 logit MAE 외에 다음을 측정한다.

- threshold decision flip rate
- float와 quantized candidate set의 Jaccard similarity
- class·level별 $Q_l$ 변화율
- NMS 전후 detection count 차이
- class별 precision·recall과 small/medium/large AP 변화
- postprocess p95·p99 latency와 peak workspace

## 11. calibration과 threshold 선택

### 11.1 데이터 분할

temperature와 threshold를 training set에서 맞추지 않는다. model 학습에 사용하지 않은 calibration split을 두고, 최종 품질 보고용 test split은 다시 분리한다. camera, site, 시간대 같은 group leakage도 막는다.

### 11.2 목적 함수

확률 calibration에는 binary NLL 또는 Brier score를 사용할 수 있다. detection threshold는 서비스 비용을 반영해 정한다.

$$
J(\tau)=c_{FN}\operatorname{FN}(\tau)+c_{FP}\operatorname{FP}(\tau)+c_{lat}T_{post}(\tau)
$$

의료 경보처럼 false negative 비용이 크면 낮은 threshold를 택할 수 있지만 candidate cap과 latency guard를 함께 설계해야 한다. threshold는 정확도 숫자 하나가 아니라 운영 정책이다.

### 11.3 calibration이 할 수 없는 것

temperature scaling은 score mapping을 조정할 뿐 localization 오류, class map mismatch, 잘못된 anchor, NMS bug를 고치지 않는다. calibration 후 AP가 크게 변했다면 global cap이나 cross-class ordering이 개입했는지 확인한다.

## 12. 성능·메모리 예산

### 12.1 raw class output

FP32 class logit 메모리는 다음과 같다.

$$
M_{cls}=4N\sum_lH_lW_lA_lK
$$

예를 들어 $N=1$, 전체 anchor가 100,000개, $K=80$이면 32,000,000 bytes, 약 30.52 MiB다. GPU에서 CPU로 전부 복사한 뒤 threshold를 적용하면 PCIe 전송과 host memory를 낭비한다.

### 12.2 후처리 위치

가능하면 device에서 sigmoid·threshold·level top-k를 수행하고 compact candidate만 host로 옮긴다. 다만 provider별 `NonMaxSuppression` semantics, 동점 정렬, class 처리 방식이 다를 수 있으므로 golden parity gate가 먼저다.

### 12.3 latency 분해

$$
T_{e2e}=T_{queue}+T_{pre}+T_{model}+T_{score}+T_{topk}+T_{decode}+T_{nms}+T_{copy}
$$

`T_model`만 안정적이고 `T_nms` p99가 증가한다면 GPU model rollback보다 threshold·calibrator·candidate cap revision을 먼저 대조한다.

### 12.4 admission control

다음 guard를 함께 둔다.

- 입력 pixel 수와 batch 크기 상한
- level별 pre-NMS cap $K_l$
- global candidate cap
- class별 final detection cap
- postprocess time budget 초과 시 명시적 degraded response
- cap 도달률과 dropped-candidate metric

cap은 조용히 품질을 바꾸므로 `cap_hit_total{level,class}`를 반드시 관찰한다.

## 13. 테스트와 디버깅

### 13.1 release-gate 테스트 행렬

| 테스트 | 고장 주입 | 기대 결과 |
| --- | --- | --- |
| score semantics | output을 probability로 거짓 표기 | manifest loader 거부 |
| prior bias | bias를 0으로 교체 | 초기 score golden 실패 |
| layout | anchor-major를 field-major로 해석 | index-pattern golden 실패 |
| calibration | temperature 0 또는 class 수 불일치 | 시작 전 거부 |
| threshold | 확률 threshold를 logit에 직접 비교 | candidate set parity 실패 |
| quantization | coarse scale 적용 | decision flip rate gate 실패 |
| tie-break | stable key 제거 | C++·C# candidate 순서 불일치 |
| candidate cap | 작은 `P3` cap 적용 | cap-hit·small recall gate 실패 |
| empty scene | 배경 이미지 batch | 후보 수와 p99 상한 통과 |
| extreme logit | `-1000`, `1000` 입력 | finite score, NaN 0개 |

### 13.2 디버깅 순서

1. release bundle의 checksum과 schema를 확인한다.
2. runtime output 이름, shape, dtype를 기록한다.
3. raw logit 몇 개를 Python golden과 비교한다.
4. sigmoid가 정확히 한 번 적용됐는지 확인한다.
5. calibration revision과 class map 순서를 확인한다.
6. threshold 전후 class·level별 count를 비교한다.
7. top-k와 tie-break 뒤 candidate ID를 비교한다.
8. decode와 NMS를 마지막에 비교한다.

최종 box만 비교하면 score transform과 NMS가 서로의 오류를 가려 원인 분리가 어렵다.

### 13.3 tolerance

raw logit에는 absolute·relative tolerance를 둘 수 있지만 threshold 결정에는 별도 정책이 필요하다. threshold에서 아주 가까운 score는 작은 dtype 차이로 결과가 바뀐다. 다음 두 지표를 함께 둔다.

- margin 밖 candidate는 exact ID parity
- margin 안 candidate는 decision flip rate와 최종 metric budget

margin은 임의 상수가 아니라 target runtime의 logit error 분포로 정한다.

## 14. 운영 관측 대시보드

### 14.1 입력

- 해상도, aspect ratio, padding 비율
- brightness·blur proxy
- camera/site/app version
- empty-scene proxy 비율

### 14.2 score funnel

- `raw_scores_total{level}`
- `finite_scores_total{level}`
- `threshold_pass_total{level,class}`
- `topk_kept_total{level,class}`
- `nms_kept_total{class}`
- `cap_hit_total{level,class}`

### 14.3 score 분포

- raw logit p01, p50, p95, p99
- sigmoid score quantile
- threshold margin 안의 score 비율
- calibrator 적용 전후 ECE와 Brier score
- class별 positive-rate proxy

### 14.4 시스템

- model·score·top-k·decode·NMS latency p50/p95/p99
- device/host peak memory
- device-to-host byte
- request timeout·OOM·fallback rate
- model, calibrator, threshold, postprocess revision

## 15. 실무 실패 사례

### 사례 A: 모델은 같지만 후보가 80배 증가했다

**상황:** 새 client가 raw logit을 probability로 오해해 threshold `0.05`와 직접 비교했다.

**원인:** logit 0은 probability 0.5지만, score transform이 누락된 비교에서는 의미가 완전히 다르다.

**방지:** output 이름에 `logits`를 넣고 manifest의 `model_output`과 runtime pipeline을 검증한다.

### 사례 B: INT8 AP는 통과했지만 p99가 실패했다

**상황:** 전체 AP 변화는 허용 범위였지만 threshold 주변 score가 양자화 bin 하나에 몰렸다.

**원인:** calibration objective가 tensor MSE만 최소화했고 candidate flip과 NMS 비용을 보지 않았다.

**방지:** decision flip rate, $Q_l$, cap hit, postprocess p99를 INT8 gate에 포함한다.

### 사례 C: rare class만 사라졌다

**상황:** class별 temperature 파일은 새 버전인데 threshold 파일은 이전 버전이었다.

**원인:** 두 artifact를 독립 배포했다.

**방지:** calibration과 threshold를 하나의 호환성 revision으로 묶고 class map checksum을 검사한다.

### 사례 D: 빈 장면에서 후보 폭주

**상황:** checkpoint 변환기가 classification head bias를 누락해 0으로 초기화했다.

**원인:** weight shape만 검사하고 initial prior golden을 확인하지 않았다.

**방지:** known input raw logit과 bias checksum을 release bundle에 포함한다.

### 사례 E: GPU와 CPU가 같은 box를 다른 순서로 반환

**상황:** score 동점에서 provider마다 `topk` 순서가 달랐다.

**원인:** tie-break가 API 계약에 없었다.

**방지:** `(score descending, level order, flat index, class)` key를 명시하고 같은 golden을 실행한다.

### 사례 F: `P3` cap을 낮춰 latency만 고쳤다

**상황:** 작은 물체 후보가 많은 `P3`의 cap을 줄여 p99는 회복했지만 small recall이 하락했다.

**원인:** cap을 시스템 knob로만 보고 품질 정책으로 versioning하지 않았다.

**방지:** size bucket별 recall proxy와 cap hit를 canary gate에 포함한다.

## 16. 배포와 rollback

### 16.1 graph 경계

선택지는 세 가지다.

1. raw logits와 box deltas만 export하고 service가 후처리를 담당한다.
2. sigmoid·threshold·top-k까지 graph에 넣고 NMS는 service가 담당한다.
3. NMS까지 graph에 넣어 최종 detection만 반환한다.

첫 방식은 디버깅과 정책 교체가 쉽지만 전송량과 client parity 부담이 크다. 셋째 방식은 interface가 작지만 provider-specific NMS와 dynamic output 검증이 어렵다. 어느 선택이든 책임 경계를 manifest에 고정한다.

### 16.2 배포 순서

1. artifact checksum과 schema를 offline 검증한다.
2. FP32 reference에서 golden logits·candidate ID를 저장한다.
3. target provider에서 raw logit parity를 확인한다.
4. score·calibration·threshold·top-k 단계별 parity를 확인한다.
5. empty, dense, small-object, extreme-aspect slice를 shadow traffic으로 실행한다.
6. 1% canary에서 quality proxy와 postprocess p99를 함께 본다.
7. cap hit·timeout·OOM이 안정적일 때 점진 확대한다.

### 16.3 rollback 단위

model만 rollback하면 새 calibrator와 old logits가 결합될 수 있다. 다음을 하나의 immutable release ID로 되돌린다.

- model graph
- anchor artifact
- class map
- calibration parameters
- threshold table
- top-k·NMS policy
- preprocessing and coordinate contract

## 17. 운영 체크리스트

### 학습·수치

- [ ] $\alpha_t$가 positive와 negative에 다르게 적용되는가?
- [ ] ignore mask와 global positive normalization을 검증했는가?
- [ ] BCE, focal factor, reduction이 FP32인가?
- [ ] prior bias의 sigmoid가 $\pi$와 일치하는가?
- [ ] AMP overflow·skipped step을 기록하는가?

### shape·score ABI

- [ ] output이 raw logit인지 probability인지 명시했는가?
- [ ] independent sigmoid와 softmax를 구분했는가?
- [ ] level·anchor·class flatten 순서가 고정됐는가?
- [ ] sigmoid를 정확히 한 번 적용하는가?
- [ ] calibration과 threshold class 순서가 class map과 같은가?

### 성능·양자화

- [ ] level별 candidate cap과 global cap이 있는가?
- [ ] INT8 decision flip rate와 candidate Jaccard를 측정했는가?
- [ ] threshold 전후·top-k·NMS count를 기록하는가?
- [ ] model뿐 아니라 postprocess p95·p99를 측정하는가?
- [ ] cap hit가 품질 slice와 함께 보이는가?

### 릴리스·운영

- [ ] model·calibrator·threshold·NMS가 하나의 release ID인가?
- [ ] Python·C++·C# golden candidate 순서가 같은가?
- [ ] empty scene과 dense scene을 canary에 포함했는가?
- [ ] alert에 revision label이 포함되는가?
- [ ] 전체 bundle rollback을 연습했는가?

## 18. 연습문제

### 문제 1

초기 prior $\pi=0.02$를 만드는 classification bias를 구하라.

### 문제 2

$H=20$, $W=30$, $A=9$, $K=5$인 level의 anchor 수, score 수, FP32 class raw byte를 구하라.

### 문제 3

한 anchor의 두 class sigmoid score가 모두 threshold를 통과했다. anchor 후보는 몇 개인가? NMS는 어떻게 적용해야 하는가?

### 문제 4

class별 temperature를 적용한 뒤 예전 threshold 파일을 재사용하면 위험한 이유를 설명하라.

### 문제 5

INT8 logit 오차가 0.04이고 threshold 부근 $p=0.5$라면 1차 근사 score 오차는 얼마인가?

### 문제 6

level별 threshold 통과 수가 `P3=5000`, `P4=800`, `P5=100`이고 cap이 각각 1200, 600, 300일 때 pre-NMS 후보 수 $R$을 구하라.

### 문제 7

raw logit은 동일한데 새 release의 NMS p99가 증가했다. 확인할 artifact와 metric을 세 가지 이상 적어라.

### 문제 8

model만 rollback하는 것이 충분하지 않은 이유를 설명하라.

## 19. 해답

### 해답 1

$$
b=\log\left(\frac{0.02}{0.98}\right)\approx-3.89182
$$

release test는 `sigmoid(b)`가 0.02인지 확인한다.

### 해답 2

anchor 수는 다음과 같다.

$$
M=20\times30\times9=5400
$$

score 수는 다음과 같다.

$$
MK=5400\times5=27000
$$

FP32 class raw byte는 다음과 같다.

$$
27000\times4=108000\ \mathrm{bytes}
$$

이는 약 105.47 KiB다.

### 해답 3

class-anchor pair 기준으로 후보가 2개다. 일반적인 RetinaNet 후처리는 class별 NMS를 적용한다. class-agnostic NMS를 쓰려면 그 정책과 품질 차이를 별도 versioning해야 한다.

### 해답 4

temperature는 logit-to-score mapping을 바꾼다. 같은 확률 threshold라도 통과하는 raw logit 경계가 달라지므로 후보 수, precision, recall, latency가 모두 달라질 수 있다. 두 파일은 호환성 revision으로 묶어야 한다.

### 해답 5

$$
|\Delta p|\approx p(1-p)|\Delta z|=0.5\times0.5\times0.04=0.01
$$

threshold가 0.5 근처라면 0.01은 decision flip을 만들기에 충분할 수 있다.

### 해답 6

$$
R=\min(5000,1200)+\min(800,600)+\min(100,300)=1900
$$

`P3`와 `P4`는 cap에 닿으므로 해당 slice의 recall도 확인해야 한다.

### 해답 7

calibrator revision, threshold table, level별 top-k cap, tie-break, NMS IoU와 class-wise 여부를 확인한다. metric은 level·class별 threshold 통과 수, cap hit, NMS 입력 수, `T_topk`, `T_nms`, device-to-host byte가 유용하다.

### 해답 8

배포 결과는 model, anchors, class map, calibration, threshold, top-k, NMS, preprocessing의 조합이다. old model과 new calibrator처럼 검증되지 않은 조합을 만들지 않도록 전체 bundle을 atomic하게 rollback해야 한다.

## 핵심 요약

1. Focal Loss는 학습 gradient를 재분배하지만 추론 후보 수를 제한하지 않는다.
2. RetinaNet class output은 background softmax가 아니라 $K$개의 independent sigmoid logit이다.
3. raw logit, sigmoid score, calibrated score 중 export 의미를 manifest에 고정해야 한다.
4. prior bias $\log(\pi/(1-\pi))$가 누락되면 초기 score와 후보 수가 크게 달라진다.
5. calibration parameter와 threshold는 하나의 호환성 revision으로 배포한다.
6. FP16 학습에서는 stable BCE와 reduction을 FP32로 유지한다.
7. INT8 gate는 tensor 오차뿐 아니라 threshold flip, candidate set, NMS p99를 측정해야 한다.
8. level별 candidate cap은 성능 knob이자 small-object 품질 정책이다.
9. Python·C++·C#은 stable sigmoid와 tie-break까지 같은 golden을 통과해야 한다.
10. rollback 단위는 model 하나가 아니라 anchor·calibrator·threshold·NMS를 포함한 immutable bundle이다.

## 다음 학습 예고

다음 문서는 3회차 실무 엔지니어 7/18 `02-06.FCOS_DETR.md` Part 1/2다. anchor 없는 FCOS의 위치별 `l,t,r,b` 출력, center sampling, centerness, multi-level assignment를 serving ABI로 만들고, RetinaNet에서 관리한 anchor·candidate 운영 복잡성이 어떻게 바뀌는지 비교한다.
