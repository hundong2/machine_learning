<!-- curriculum: cycle=3; level=production-engineering; source_index=6/18; source=02-05.YOLO.md; part=2/3 -->

# SSD와 FPN 운영: 다중 출력 ABI와 레벨별 예산

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-06 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 6/18 · `02-05.YOLO.md` |
| Part | 2/3 · SSD/FPN 다중 스케일 운영 |
| 이전 소스 | `02-05.YOLO.md` Part 1/3 · YOLOv1 운영 |
| 다음 소스 | `02-05.YOLO.md` Part 3/3 · Focal Loss/RetinaNet 운영 |

## 오늘의 운영 질문

SSD와 FPN은 작은 물체를 위해 여러 해상도의 feature map에서 동시에 예측한다. 그러나 이 장점은 배포 경계를 여러 개로 늘린다.

- export 뒤 `P3`, `P4`, `P5`의 순서가 바뀌어도 decoder가 안전한가?
- 서로 다른 stride와 anchor schema를 어떤 기계 판독 가능한 manifest로 고정하는가?
- 작은 물체를 살리려고 고해상도 level을 늘렸을 때 메모리와 NMS 지연은 얼마나 증가하는가?
- dynamic input에서 실제 feature shape과 선언한 stride가 어긋나면 어디서 실패시킬 것인가?
- Python, C++, C#이 같은 flat index를 같은 `(level,row,column,anchor)`로 해석하는가?
- 전체 후보 수만 보지 않고 특정 level의 폭주와 작은 물체 recall 저하를 어떻게 관측하는가?

1회차는 SSD의 다중 해상도와 FPN의 top-down 직관을 다뤘다. 2회차는 default box, 강제 매칭, hard-negative mining, 학습 loss를 구현했다. 오늘은 이를 반복하지 않고 **이름 기반 다중 출력 ABI, level routing, 자원 예산, 크로스런타임 parity, canary와 장애 격리**로 확장한다.

원본 [02-05.YOLO.md](../05.ImageClassification/02-05.YOLO.md)는 YOLOv1, SSD/FPN, Focal Loss/RetinaNet을 한 파일에 담고 있다. 이번 회차도 세 Part로 나누며, 오늘 Part 2는 SSD/FPN 범위를 완결한다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

- feature level을 list 위치가 아니라 이름, stride, shape equation으로 식별한다.
- level별 anchor 수와 raw head channel 수를 배포 전에 검증한다.
- flat index와 tensor layout을 Python, C++, C#에서 같은 방식으로 계산한다.
- 작은 객체가 어느 level에 배정되고 어느 단계에서 탈락하는지 telemetry로 추적한다.
- FPN activation, raw output, candidate buffer, NMS 비용을 요청별로 예산화한다.
- dynamic shape, odd shape, missing output, output reorder를 release test에 포함한다.
- ONNX multi-output graph와 decoder를 하나의 versioned bundle로 배포한다.
- 장애 시 raw output, decode, filtering, NMS를 분리해 rollback 여부를 판단한다.

## 선수 지식과 기호

- SSD default box와 `cxcywh`/`xyxy` 변환
- FPN의 lateral connection과 top-down upsampling
- `NCHW`, `NHWC`, contiguous flat buffer
- stride, receptive field, feature pyramid level
- IoU, score threshold, top-k, NMS
- p50·p95·p99 latency, canary, SLO, rollback

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $C_f$ | FPN feature channel 수 |
| $H_k,W_k$ | level $P_k$의 공간 크기 |
| $s_k$ | level $P_k$의 nominal stride |
| $A_k$ | 해당 level의 cell당 anchor 수 |
| $C$ | foreground class 수 |
| $B_k=H_kW_kA_k$ | level $k$의 box 수 |
| $K_k$ | level별 pre-NMS candidate 상한 |
| $D=C+4$ | anchor당 class와 box raw 값 수의 단순화된 합 |

## 1. 직관: 피라미드는 여러 모델이 아니라 하나의 좌표계 묶음이다

`P3`, `P4`, `P5`를 단순히 크기가 다른 tensor라고 보면 운영 오류를 놓치기 쉽다. 각 level에는 다음 의미가 함께 붙는다.

1. 어느 backbone stage에서 왔는가
2. 입력 pixel에 대한 nominal stride는 얼마인가
3. 어떤 scale과 aspect ratio의 anchor가 연결되는가
4. raw class와 box channel이 어떤 순서로 배열되는가
5. 어느 candidate budget과 threshold를 적용하는가

따라서 level은 tensor position이 아니라 **이름 있는 좌표계**다. `outputs[0]`이 우연히 `P3`였다는 사실은 ABI가 아니다. 이름, shape, stride, anchor schema, flatten order를 함께 검증해야 한다.

작은 물체 recall을 지키는 것도 model 구조만의 문제가 아니다. 고해상도 level의 output이 빠졌거나, `P3`에 `P4` anchor를 연결했거나, 전역 top-k가 큰 물체 후보로 가득 차면 학습된 작은 물체 신호는 배포 단계에서 사라진다.

## 2. 원본에서 유지할 것과 교정할 것

### 2.1 유지할 핵심

원본의 SSD/FPN 부분에서 유지할 핵심은 다음과 같다.

- SSD는 여러 해상도의 feature map에서 동시에 box와 class를 예측한다.
- 고해상도 feature는 작은 물체를 위한 더 촘촘한 spatial location을 제공한다.
- FPN은 깊은 feature의 정보를 top-down 경로와 lateral connection으로 높은 해상도에 전달한다.
- level마다 anchor scale과 역할이 다르므로 전체 pyramid가 하나의 detector를 이룬다.

### 2.2 “작은 물체는 얕은 층이 담당한다”는 충분조건이 아니다

고해상도는 더 많은 sample point를 제공하지만 작은 물체 정확도를 자동 보장하지 않는다. 입력 resize, annotation 품질, 최소 box filter, anchor scale, assignment 정책, feature semantics, score threshold가 모두 맞아야 한다.

운영 지표는 `AP_small` 하나로 끝내지 않는다. 최소한 다음 funnel을 크기 구간별로 본다.

$$
N_{\mathrm{GT}}
\rightarrow N_{\mathrm{assigned}}
\rightarrow N_{\mathrm{decoded}}
\rightarrow N_{\mathrm{thresholded}}
\rightarrow N_{\mathrm{topk}}
\rightarrow N_{\mathrm{NMS}}
$$

어느 화살표에서 작은 물체가 줄었는지 알아야 학습, decoder, threshold, NMS 중 어디를 고칠지 결정할 수 있다.

### 2.3 FPN의 `3 x 3` convolution은 단순한 alias 제거 장치가 아니다

원본은 마지막 convolution을 계단 현상을 매끄럽게 하는 필터로 설명한다. 실제로는 top-down과 lateral을 합친 feature를 학습 가능한 방식으로 정제해 각 pyramid output을 만드는 역할이다. aliasing 완화만으로 기능을 한정하면 안 된다.

### 2.4 FPN과 SSD는 같은 것이 아니다

SSD는 서로 다른 깊이의 feature에 직접 prediction head를 붙이는 다중 스케일 detector 설계다. FPN은 bottom-up feature와 top-down feature를 lateral connection으로 결합하는 feature extractor다. FPN은 SSD식 head, RetinaNet head, two-stage detector 등 여러 head와 조합될 수 있다.

### 2.5 `scale_factor=2`는 dynamic shape ABI가 아니다

odd input에서는 깊은 level을 두 배로 키운 결과가 lateral shape과 1 pixel 다를 수 있다. 안전한 식은 목표 tensor의 실제 공간 크기를 명시하는 것이다.

```python
# 설명용: odd shape에서도 덧셈 가능한 FPN upsample 계약
top_down = torch.nn.functional.interpolate(
    deeper,
    size=lateral.shape[-2:],
    mode="nearest",
)
fused = lateral + top_down
```

### 2.6 SSD scale 식은 특정 설계의 시작점이지 보편 법칙이 아니다

원본의 선형 scale 식은 SSD300 설계를 이해하는 데 유용하다.

$$
s_k=s_{\min}+\frac{s_{\max}-s_{\min}}{m-1}(k-1)
$$

하지만 실제 bundle은 식만 저장해서는 부족하다. extra square box, aspect ratio 중복 제거, center offset, clip 여부, variance까지 포함한 **전개 완료된 level schema**와 hash를 저장해야 한다.

## 3. shape와 index를 배포 수식으로 고정하기

### 3.1 feature shape

padding convention이 같은 stride-$s_k$ convolution stack이라면 흔히 다음 형태가 된다.

$$
H_k=\left\lceil\frac{H_{\mathrm{in}}}{s_k}\right\rceil,
\qquad
W_k=\left\lceil\frac{W_{\mathrm{in}}}{s_k}\right\rceil
$$

그러나 architecture마다 downsampling padding이 다를 수 있으므로 이 식을 추측하지 말고 export fixture로 검증한다. 입력 `513 x 769`와 stride 8이면 식상 `65 x 97`이다.

### 3.2 raw head shape

anchor-major `NCHW` head를 가정하면 level $k$의 출력은 다음과 같다.

$$
Z_k\in\mathbb{R}^{N\times A_kD\times H_k\times W_k}
$$

논리적인 view는 다음과 같다.

$$
\widetilde{Z}_k\in\mathbb{R}^{N\times H_k\times W_k\times A_k\times D}
$$

단순 `reshape`만으로 두 식을 연결할 수 있는지는 channel ordering에 달려 있다. `anchor-major`인지 `field-major`인지 manifest에 적지 않으면 shape는 맞아도 값은 섞인다.

### 3.3 level 안의 flat index

level 안에서 `row -> column -> anchor` 순서라면 다음과 같다.

$$
i_k=((rW_k)+c)A_k+a
$$

전체 pyramid에서 level 순서가 `P3`, `P4`, `P5`라면 prefix는 다음과 같다.

$$
o_k=\sum_{j<k}H_jW_jA_j
$$

따라서 global index는 다음과 같다.

$$
g_k=o_k+i_k
$$

예를 들어 `P3=(4,5,A=2)`, `P4=(2,3,A=3)`, `P5=(1,2,A=1)`이면 box 수는 각각 40, 18, 2이고 prefix는 `[0,40,58,60]`이다. `P4`의 `(r=1,c=2,a=2)`는 level index 17, global index 57이다.

### 3.4 inverse index

flat index를 다시 논리 좌표로 복원하면 decoder와 generator의 순서를 검증할 수 있다.

$$
a=i_k\bmod A_k
$$

$$
q=\left\lfloor\frac{i_k}{A_k}\right\rfloor,
\qquad
r=\left\lfloor\frac{q}{W_k}\right\rfloor,
\qquad
c=q\bmod W_k
$$

첫 값, 경계값, 마지막 값만 확인하지 말고 모든 index의 round trip을 property test로 검사한다.

## 4. release bundle과 manifest

### 4.1 bundle 구조

```text
ssd-fpn-release-2026-09-06/
  model.onnx
  manifest.json
  classes.txt
  anchors.npz
  golden-input.npy
  golden-p3-class.npy
  golden-p3-box.npy
  golden-candidates.json
  checksums.sha256
```

위 파일명은 **설명용 bundle 예시**다. 실제 registry에서는 각 artifact의 content hash와 접근 권한을 별도로 관리한다.

### 4.2 manifest 예시

```json
{
  "schema": "multiscale-detector/v4",
  "architecture": "ssd-fpn",
  "input": {
    "name": "images",
    "layout": "NCHW",
    "dtype": "float32",
    "dynamic_hw": {"min": [320, 320], "max": [800, 1333]}
  },
  "levels": [
    {
      "name": "P3",
      "stride": 8,
      "anchors_per_cell": 3,
      "class_output": "cls_P3",
      "box_output": "box_P3",
      "candidate_cap": 1000
    },
    {
      "name": "P4",
      "stride": 16,
      "anchors_per_cell": 6,
      "class_output": "cls_P4",
      "box_output": "box_P4",
      "candidate_cap": 500
    },
    {
      "name": "P5",
      "stride": 32,
      "anchors_per_cell": 6,
      "class_output": "cls_P5",
      "box_output": "box_P5",
      "candidate_cap": 300
    }
  ],
  "raw_layout": "NCHW-anchor-major",
  "flatten_order": "level-row-column-anchor",
  "box_encoding": "cxcywh-variance-v1",
  "anchor_artifact_sha256": "example-only",
  "score_activation": "softmax-with-background-0",
  "global_post_nms_cap": 100
}
```

manifest loader는 모르는 schema version을 추측해 실행하지 않고 거부해야 한다. `P3`가 없거나 output 이름이 중복되거나 channel 수가 $A_kC$ 또는 $4A_k$와 다르면 inference 전에 실패시킨다.

### 4.3 이름으로 연결하고 순서는 검증한다

runtime이 반환하는 map을 다음처럼 다룬다.

```text
runtime outputs
  cls_P5 -> manifest level P5 class
  box_P3 -> manifest level P3 box
  cls_P3 -> manifest level P3 class
  box_P5 -> manifest level P5 box
  cls_P4 -> manifest level P4 class
  box_P4 -> manifest level P4 box
```

반환 순서가 섞여도 이름으로 연결하면 안전하다. 다만 이름이 맞아도 stride와 anchor artifact revision이 틀릴 수 있으므로 shape와 checksum 검사는 여전히 필요하다.

## 5. 작은 물체 라우팅을 수식과 telemetry로 연결하기

### 5.1 객체 크기와 level

물체 box의 대표 크기를 다음처럼 둘 수 있다.

$$
d=\sqrt{wh}
$$

학습 assignment가 anchor IoU 기반이라면 하나의 정답이 여러 level에 positive를 만들 수 있다. 운영에서는 각 정답의 최종 matched level을 단순 규칙으로 가정하지 않고 실제 matcher 결과에서 기록한다.

level별 비율은 다음과 같다.

$$
r_k=\frac{N_{\mathrm{positive},k}}{\sum_jN_{\mathrm{positive},j}}
$$

새 데이터에서 `P3`의 $r_3$가 급감하면 작은 물체 감소일 수도 있지만 resize 정책, 최소 box filter, annotation 손실, anchor config 변경일 수도 있다.

### 5.2 배포 candidate funnel

각 level에서 다음 count를 기록한다.

| 단계 | metric 예 | 의미 |
| --- | --- | --- |
| raw | `raw_boxes{level}` | $H_kW_kA_k$ 계약 확인 |
| finite | `finite_boxes{level}` | decode NaN·Inf 탐지 |
| score | `score_pass{level}` | threshold 전후 분포 |
| top-k | `topk_kept{level}` | cap에 계속 닿는지 확인 |
| NMS | `nms_kept{level}` | level 내부 중복 정도 |
| final | `final_boxes{size_bucket}` | 실제 크기별 결과 |

`topk_kept{level=P3}=K_3`가 지속되면 `P3`가 포화된 것이다. 전역 후보 수가 정상이어도 작은 물체 후보끼리 경쟁해 recall이 떨어질 수 있다.

### 5.3 level별 top-k와 전역 top-k

전역 top-k만 사용하면 box 수가 많은 고해상도 level이 후보를 독점할 수 있다. 반대로 level별 cap만 사용하면 품질이 낮은 후보를 불필요하게 많이 남길 수 있다.

실무에서는 다음 두 단계를 조합할 수 있다.

1. 각 level에서 안정적인 score 정렬로 최대 $K_k$개를 유지한다.
2. 합친 뒤 global cap $K$를 다시 적용한다.

동점 key는 최소한 `(score descending, level order, flat index ascending)`으로 고정한다. `topk` kernel의 임의 동점 순서에 release parity를 맡기지 않는다.

## 6. 메모리와 지연 예산

### 6.1 FPN activation

FP32 level 하나의 activation byte는 다음과 같다.

$$
M_{\mathrm{feature},k}=4NC_fH_kW_k
$$

예를 들어 $N=1$, $C_f=256$, 입력 `800 x 1280`, stride가 8, 16, 32라면 공간 크기는 `100 x 160`, `50 x 80`, `25 x 40`이다.

$$
M_{P3}=4\times1\times256\times100\times160=16{,}384{,}000\ \mathrm{bytes}
$$

세 level 합은 20.51 MiB다. 이는 output만의 크기이며 framework workspace, backbone activation, allocator fragmentation은 포함하지 않는다.

### 6.2 raw output

class와 box를 분리하고 foreground $C=20$, background 포함 class channel 21, $A_k=6$이라면 cell당 raw float 수는 다음과 같다.

$$
A_k(21+4)=150
$$

같은 `P3=100 x 160`의 FP32 raw output은 다음과 같다.

$$
4\times100\times160\times150=9{,}600{,}000\ \mathrm{bytes}
$$

작은 물체를 위해 `P2`를 추가하면 spatial area가 `P3`의 약 4배가 된다. 정확도 실험과 함께 GPU activation, output transfer, decode time, candidate count를 모두 다시 측정해야 한다.

### 6.3 candidate buffer

candidate 하나가 `xyxy` 4개, score 1개를 FP32로 가지고 class, level, flat index를 int32로 가진다면 최소 payload는 다음과 같다.

$$
5\times4+3\times4=32\ \mathrm{bytes}
$$

실제 구조체 padding과 allocator overhead는 더 클 수 있다. 후보 100,000개면 payload만 약 3.05 MiB다. struct-of-arrays와 array-of-structs 중 어느 쪽이 runtime kernel에 유리한지 profile한다.

### 6.4 지연 모델

요청의 end-to-end 지연을 다음처럼 분리한다.

$$
T_{\mathrm{e2e}}=T_{\mathrm{queue}}+T_{\mathrm{pre}}+T_{\mathrm{backbone}}+T_{\mathrm{FPN}}+T_{\mathrm{head}}+T_{\mathrm{decode}}+T_{\mathrm{NMS}}+T_{\mathrm{copy}}
$$

GPU forward만 재면 output download와 CPU NMS 폭주를 놓친다. level별 decode time과 candidate count를 같은 trace id로 연결한다.

## 7. NumPy·PyTorch 실행 예제

다음은 **실행 가능한 Python 예제**다. 외부 데이터 없이 다음을 검증한다.

- output dictionary의 순서가 섞여도 이름으로 연결
- odd input의 FPN target-size upsampling
- level별 shape와 channel contract
- index round trip과 stable candidate selection
- feature/output byte 예산
- finite backward

```python
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


torch.manual_seed(20260906)
np.random.seed(20260906)


@dataclass(frozen=True)
class Level:
    name: str
    stride: int
    anchors: int
    cap: int


LEVELS = (
    Level("P3", 8, 2, 5),
    Level("P4", 16, 3, 4),
    Level("P5", 32, 1, 2),
)
CLASS_COUNT = 3


def expected_hw(height, width, stride):
    return math.ceil(height / stride), math.ceil(width / stride)


def level_index(row, column, anchor, width, anchors):
    return (row * width + column) * anchors + anchor


def inverse_level_index(index, width, anchors):
    anchor = index % anchors
    spatial = index // anchors
    return spatial // width, spatial % width, anchor


def stable_level_topk(scores, cap):
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    indices = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((indices, -np.where(finite, scores, -np.inf)))
    return order[finite[order]][:cap]


class TinyFPN(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.c3 = nn.Conv2d(3, channels, 3, stride=8, padding=1)
        self.c4 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.c5 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        self.out3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.out4 = nn.Conv2d(channels, channels, 3, padding=1)
        self.out5 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, images):
        c3 = self.c3(images)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        m5 = c5
        m4 = c4 + F.interpolate(m5, size=c4.shape[-2:], mode="nearest")
        m3 = c3 + F.interpolate(m4, size=c3.shape[-2:], mode="nearest")
        return {"P5": self.out5(m5), "P3": self.out3(m3), "P4": self.out4(m4)}


class MultiHead(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.cls = nn.ModuleDict({
            level.name: nn.Conv2d(channels, level.anchors * CLASS_COUNT, 3, padding=1)
            for level in LEVELS
        })
        self.box = nn.ModuleDict({
            level.name: nn.Conv2d(channels, level.anchors * 4, 3, padding=1)
            for level in LEVELS
        })

    def forward(self, features):
        outputs = {}
        for level in LEVELS:
            outputs[f"cls_{level.name}"] = self.cls[level.name](features[level.name])
            outputs[f"box_{level.name}"] = self.box[level.name](features[level.name])
        return outputs


def validate_outputs(outputs, input_hw):
    height, width = input_hw
    prefixes = [0]
    logical = {}
    for level in LEVELS:
        cls = outputs[f"cls_{level.name}"]
        box = outputs[f"box_{level.name}"]
        expected = expected_hw(height, width, level.stride)
        assert tuple(cls.shape[-2:]) == expected
        assert tuple(box.shape[-2:]) == expected
        assert cls.shape[1] == level.anchors * CLASS_COUNT
        assert box.shape[1] == level.anchors * 4
        count = expected[0] * expected[1] * level.anchors
        prefixes.append(prefixes[-1] + count)
        logical[level.name] = {
            "boxes": count,
            "class_shape": tuple(cls.shape),
            "box_shape": tuple(box.shape),
        }
    return prefixes, logical


height, width = 65, 81
images = torch.randn(2, 3, height, width)
fpn = TinyFPN()
head = MultiHead()
features = fpn(images)
assert list(features) == ["P5", "P3", "P4"]  # 반환 순서는 의도적으로 뒤섞였다.
outputs = head(features)  # head는 manifest 순서로 이름을 조회한다.
prefixes, logical = validate_outputs(outputs, (height, width))

# 모든 level index를 왕복 검증한다.
for level in LEVELS:
    level_height, level_width = expected_hw(height, width, level.stride)
    for row in range(level_height):
        for column in range(level_width):
            for anchor in range(level.anchors):
                index = level_index(row, column, anchor, level_width, level.anchors)
                assert inverse_level_index(index, level_width, level.anchors) == (
                    row,
                    column,
                    anchor,
                )

# score 동점은 작은 flat index가 먼저다. NaN은 후보에서 제외한다.
selected = stable_level_topk([0.7, 0.9, 0.9, float("nan"), 0.8], cap=3)
assert selected.tolist() == [1, 2, 4]

# loss는 설명용 smoke test이며 실제 SSD loss 전체 구현이 아니다.
loss = sum(value.float().square().mean() for value in outputs.values())
loss.backward()
assert all(parameter.grad is not None for parameter in fpn.parameters())
assert all(torch.isfinite(parameter.grad).all() for parameter in fpn.parameters())

feature_bytes = sum(tensor.numel() * tensor.element_size() for tensor in features.values())
output_bytes = sum(tensor.numel() * tensor.element_size() for tensor in outputs.values())
assert prefixes == [0, 198, 288, 297]

print("feature shapes:", {key: tuple(value.shape) for key, value in features.items()})
print("prefixes:", prefixes)
print("stable top-k:", selected.tolist())
print("bytes:", feature_bytes, output_bytes)
print("loss:", f"{loss.item():.6f}")
print("logical:", logical)
```

예상되는 핵심 shape은 `P3=(2,8,9,11)`, `P4=(2,8,5,6)`, `P5=(2,8,3,3)`이다. `P3` box 수는 $9\times11\times2=198$이고 올바른 prefix는 `[0,198,288,297]`이다. 이 값을 `[0,180,270,276]`처럼 바꾼 negative test는 즉시 실패해야 한다. nominal stride만으로 shape를 추측하지 않고 실제 convolution 식과 runtime output을 검사하는 것이 핵심이다.

일반 convolution 출력은 다음과 같다.

$$
H_{\mathrm{out}}=\left\lfloor\frac{H+2p-d(k-1)-1}{s}+1\right\rfloor
$$

위 예제에서 $H=65$, $p=1$, $d=1$, $k=3$, $s=8$이면 $H_{\mathrm{out}}=9$다. 코드 블록 자체는 positive test로 그대로 실행 가능하다. 별도 negative test에서는 예상 prefix 하나를 바꾸어 release gate가 잘못된 shape 추측을 거부하는지 확인한다.

## 8. C++17 예제: prefix와 index ABI

다음은 **실행 가능한 C++17 예제**다. runtime tensor library 없이 flat index 계약만 독립적으로 검증한다.

```cpp
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

struct Level {
    std::string name;
    std::size_t height;
    std::size_t width;
    std::size_t anchors;
};

std::size_t LevelIndex(
    std::size_t row,
    std::size_t column,
    std::size_t anchor,
    const Level& level
) {
    assert(row < level.height);
    assert(column < level.width);
    assert(anchor < level.anchors);
    return (row * level.width + column) * level.anchors + anchor;
}

std::tuple<std::size_t, std::size_t, std::size_t> InverseIndex(
    std::size_t index,
    const Level& level
) {
    assert(index < level.height * level.width * level.anchors);
    const std::size_t anchor = index % level.anchors;
    const std::size_t spatial = index / level.anchors;
    return {spatial / level.width, spatial % level.width, anchor};
}

int main() {
    const std::vector<Level> levels = {
        {"P3", 4, 5, 2},
        {"P4", 2, 3, 3},
        {"P5", 1, 2, 1},
    };
    std::vector<std::size_t> prefix = {0};
    for (const auto& level : levels) {
        prefix.push_back(
            prefix.back() + level.height * level.width * level.anchors
        );
        for (std::size_t row = 0; row < level.height; ++row) {
            for (std::size_t column = 0; column < level.width; ++column) {
                for (std::size_t anchor = 0; anchor < level.anchors; ++anchor) {
                    const auto index = LevelIndex(row, column, anchor, level);
                    assert(InverseIndex(index, level) ==
                           std::make_tuple(row, column, anchor));
                }
            }
        }
    }
    assert((prefix == std::vector<std::size_t>{0, 40, 58, 60}));
    const auto p4_index = LevelIndex(1, 2, 2, levels[1]);
    assert(p4_index == 17);
    assert(prefix[1] + p4_index == 57);
    std::cout << "prefix 0 40 58 60; P4-global "
              << prefix[1] + p4_index << "\n";
}
```

예상 출력은 `prefix 0 40 58 60; P4-global 57`이다.

## 9. C# 예제: 같은 index ABI

다음은 **실행 가능한 C# 예제**다. `long`을 사용해 큰 dynamic shape에서 32비트 곱셈 overflow가 나지 않게 한다.

```csharp
using System;
using System.Collections.Generic;

public sealed class Level
{
    public readonly string Name;
    public readonly long Height;
    public readonly long Width;
    public readonly long Anchors;

    public Level(string name, long height, long width, long anchors)
    {
        Name = name;
        Height = height;
        Width = width;
        Anchors = anchors;
    }
}

public static class PyramidAbi
{
    static long LevelIndex(long row, long column, long anchor, Level level)
    {
        if (row < 0 || row >= level.Height ||
            column < 0 || column >= level.Width ||
            anchor < 0 || anchor >= level.Anchors)
            throw new ArgumentOutOfRangeException("logical index");
        checked
        {
            return (row * level.Width + column) * level.Anchors + anchor;
        }
    }

    static Tuple<long, long, long> InverseIndex(long index, Level level)
    {
        checked
        {
            long count = level.Height * level.Width * level.Anchors;
            if (index < 0 || index >= count)
                throw new ArgumentOutOfRangeException("flat index");
            long anchor = index % level.Anchors;
            long spatial = index / level.Anchors;
            return Tuple.Create(
                spatial / level.Width,
                spatial % level.Width,
                anchor
            );
        }
    }

    public static void Main()
    {
        var levels = new[] {
            new Level("P3", 4, 5, 2),
            new Level("P4", 2, 3, 3),
            new Level("P5", 1, 2, 1),
        };
        var prefix = new List<long> {0};
        foreach (Level level in levels)
        {
            checked
            {
                prefix.Add(prefix[prefix.Count - 1] +
                           level.Height * level.Width * level.Anchors);
            }
            for (long row = 0; row < level.Height; ++row)
                for (long column = 0; column < level.Width; ++column)
                    for (long anchor = 0; anchor < level.Anchors; ++anchor)
                    {
                        long index = LevelIndex(row, column, anchor, level);
                        Tuple<long, long, long> restored = InverseIndex(index, level);
                        if (restored.Item1 != row || restored.Item2 != column ||
                            restored.Item3 != anchor)
                            throw new Exception("index round trip failed");
                    }
        }
        if (string.Join(" ", prefix) != "0 40 58 60")
            throw new Exception("prefix mismatch");
        long p4Index = LevelIndex(1, 2, 2, levels[1]);
        if (prefix[1] + p4Index != 57)
            throw new Exception("global index mismatch");
        Console.WriteLine("prefix 0 40 58 60; P4-global 57");
    }
}
```

예상 출력은 C++과 같은 `prefix 0 40 58 60; P4-global 57`이다.

## 10. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch Python | C++ runtime | C# runtime | 반드시 고정할 것 |
| --- | --- | --- | --- | --- |
| 입력 | `[N,3,H,W]` `float32` | contiguous buffer 또는 device tensor | `DenseTensor<float>` 등 | RGB/BGR, range, normalization |
| FPN feature | 이름별 `[N,C_f,H_k,W_k]` | output name lookup | output metadata lookup | 이름, stride, actual shape |
| class raw | `[N,A_kC,H_k,W_k]` | backend layout 확인 | flat memory stride 확인 | anchor-major/field-major |
| box raw | `[N,4A_k,H_k,W_k]` | `float`/`half` | `float`/`Half` 지원 여부 | field order, variance |
| anchor | `[B_k,4]` `float32` | immutable artifact | immutable artifact | coordinate space, checksum |
| prefix | Python `int`/`int64` | `std::size_t` 또는 `int64_t` | `long` | overflow 검사 |
| candidate class | `int64` | `int32_t`/`int64_t` | `int`/`long` | class map revision |
| final box | 원본 pixel `float32` | 동일 | 동일 | 반열린 `xyxy`, clip 시점 |

### 10.1 `NCHW`를 논리 shape로 바꾸는 두 schema

anchor-major channel 순서는 대략 다음과 같다.

```text
[a0:c0, a0:c1, ..., a1:c0, a1:c1, ...]
```

field-major channel 순서는 다음과 같다.

```text
[c0:a0, c0:a1, ..., c1:a0, c1:a1, ...]
```

둘 다 channel 수는 $A_kC$다. 따라서 단순 shape test는 이 오류를 찾지 못한다. channel index 자체를 값으로 채운 synthetic output을 모든 runtime decoder에 넣어 golden을 비교한다.

### 10.2 dtype 경계

- model output이 FP16이어도 score activation, box `exp`, NMS IoU는 FP32 승격을 기본으로 검토한다.
- anchor artifact는 model output dtype에 맞춰 암묵 변환하지 말고 기준 FP32 값을 보관한다.
- index, shape product, byte size는 64비트에서 계산하고 할당 전에 상한을 검사한다.
- INT8 head는 class와 box의 quantization scale을 분리하는 편이 일반적이며, box decode parity를 별도 승인한다.

## 11. 테스트와 디버깅

### 11.1 release-gate 테스트 행렬

| 테스트 | 입력/변형 | 기대 결과 |
| --- | --- | --- |
| output reorder | runtime 반환 순서를 역순으로 변경 | 이름 연결로 동일 detection |
| missing level | `box_P3` 제거 | 시작 시 명확한 오류 |
| wrong channel | `cls_P4` channel 1개 감소 | inference 전 거부 |
| odd shape | `513 x 769` | actual target-size upsample 성공 |
| anchor mismatch | artifact hash 변경 | bundle load 거부 |
| layout mismatch | anchor-major를 field-major로 해석 | golden test 실패 |
| large shape | 최대 허용 `H,W,N` | 64비트 byte budget 통과/거부 |
| tied score | 같은 score 여러 개 | 모든 runtime에서 같은 순서 |
| empty scene | 모든 score가 threshold 미만 | 빈 결과, finite latency |
| crowded scene | 많은 overlapping candidate | cap 준수, p99 예산 내 처리 |

### 11.2 디버깅 순서

최종 AP부터 거꾸로 추측하지 않는다. 같은 입력과 trace id로 다음 artifact를 비교한다.

1. 전처리 tensor checksum
2. `P3/P4/P5` actual shape와 feature checksum
3. 이름별 class/box raw tensor 일부
4. anchor count, prefix, fingerprint
5. decode 직후 finite box와 score
6. level별 threshold와 top-k 통과 index
7. NMS 입력과 최종 원본 좌표

### 11.3 증상에서 원인으로 가는 표

| 증상 | 먼저 볼 metric/artifact | 흔한 원인 |
| --- | --- | --- |
| 작은 물체만 급감 | `topk_kept{level=P3}`, small funnel | P3 cap 포화, resize 변경, P3 output 누락 |
| box가 규칙적으로 이동 | anchor fingerprint, flat index | flatten order 불일치 |
| 큰 물체와 작은 물체 scale 뒤바뀜 | output name-stride map | level reorder |
| odd image만 실패 | actual feature shape | `scale_factor=2` 가정 |
| GPU는 빠른데 p99 증가 | decode/NMS/copy span | 후보 폭주 또는 CPU fallback |
| raw parity지만 final 불일치 | thresholded index, NMS input | tie-break, class grouping, clip 순서 |
| INT8에서 가느다란 box 붕괴 | box scale, width/height histogram | box head 과도한 양자화 |
| 배치가 커질 때 OOM | 요청별 shape·candidate budget | max shape 곱, workspace, fragmentation |

## 12. 성능·메모리·수치 안정성

### 12.1 admission control

요청을 batch에 넣기 전에 예상 feature와 raw output byte를 계산한다.

$$
M_{\mathrm{estimate}}=\sum_k b_fNC_fH_kW_k+\sum_k b_oNH_kW_kA_k(C+4)+M_{\mathrm{workspace}}
$$

$b_f$, $b_o$는 feature와 output element byte다. 실제 peak와의 오차를 측정해 safety factor를 둔다. 상한을 넘으면 작은 bucket으로 resize, 작은 batch로 분리, CPU queue 전환, 요청 거부 중 명시한 정책을 적용한다.

### 12.2 계산량을 줄이는 순서

- class activation 전에 안전하게 가능한 logit prefilter가 있는지 검토한다.
- level별 top-k로 device-to-host candidate transfer를 제한한다.
- decode와 NMS를 device에 둘지 CPU에 둘지 end-to-end로 비교한다.
- pinned memory와 async copy는 lifetime bug 없이 trace로 확인한다.
- dynamic shape profile 수를 무제한 늘리지 않고 실제 traffic bucket에 맞춘다.

### 12.3 수치 guard

- box width와 height decode에 쓰는 `exp` 입력을 승인된 범위로 clamp하고 count를 기록한다.
- softmax는 max logit을 빼고 FP32에서 계산한다.
- NaN score는 높은 점수로 정렬되지 않게 명시적으로 제외한다.
- IoU union은 epsilon으로 보호하되 퇴화 box를 먼저 제거한다.
- coordinate clip은 resize padding 제거 뒤 원본 image boundary에서 수행한다.

### 12.4 benchmark 보고 형식

평균 하나 대신 다음을 같은 hardware와 warm-up 조건에서 기록한다.

```text
input bucket, batch size, precision, provider
queue/preprocess/backbone/FPN/head/decode/NMS/copy p50/p95/p99
P3/P4/P5 actual shapes and raw candidate counts
level caps hit rate and post-NMS count
peak allocated/reserved memory and host RSS
fallback op list, NaN/Inf/clamp counts
small/medium/large quality metrics on delayed labels
```

## 13. 실무 실패 사례

### 사례 A: ONNX output 최적화가 순서를 바꿈

애플리케이션은 `outputs[0:6]`을 `P3/P4/P5` class와 box로 믿었다. 새 exporter가 graph output 순서를 바꿨지만 shape가 우연히 호환되어 crash 없이 scale이 뒤섞였다. 큰 box는 작게, 작은 box는 크게 decode되었다.

대응은 output name lookup, name별 shape 검증, anchor checksum, synthetic channel fingerprint다. output 순서 변경은 canary 전에 반드시 negative test로 주입한다.

### 사례 B: `P3` cap이 늘 포화됨

새 camera가 더 넓은 장면을 보내면서 작은 texture가 많은 이미지에서 `P3` score 후보가 급증했다. 전역 p99는 아직 SLO 안이었지만 level별 cap에 계속 닿아 작은 실제 물체가 top-k 전에 밀려났다.

대응은 cap hit rate와 크기별 recall을 연결해 보고, calibration·threshold·학습 negative 분포를 점검한다. cap만 키우면 NMS 지연과 메모리를 옮겨 놓을 뿐일 수 있다.

### 사례 C: dynamic shape에서 1 pixel mismatch

고정 `640 x 640` 테스트는 통과했지만 `513 x 769`에서 `scale_factor=2` 결과와 lateral feature가 달랐다. 일부 runtime은 오류를 냈고 일부 wrapper는 crop해 좌표가 조용히 이동했다.

대응은 target-size interpolation, odd-shape test matrix, actual output shape telemetry다. 조용한 crop이나 pad를 decoder 안에서 허용하지 않는다.

### 사례 D: C#에서 index product overflow

shape metadata는 `int`였고 `H*W*A*C`를 32비트로 계산했다. 비정상적으로 큰 요청에서 overflow가 작은 양수로 돌아와 buffer 크기 검사를 우회했다.

대응은 checked 64-bit arithmetic, 입력 shape 상한, allocation 전 byte budget 검증이다. shape는 신뢰할 수 없는 요청 입력으로 취급한다.

### 사례 E: quantized box head의 scale 공유

class logit과 box regression output에 같은 calibration 전략을 적용했다. class AP는 유지됐지만 작은 box의 width·height 오차가 커졌다. 전체 mAP만 봐서 canary에서 놓쳤다.

대응은 head별 calibration, decoded coordinate parity, size-bucket metric, thin-object slice를 release gate에 둔다.

### 사례 F: anchor artifact만 이전 revision

model과 manifest는 새 버전인데 edge image에 오래된 `anchors.bin`이 남았다. count가 같아 shape test는 통과했지만 ratio 순서가 달라 box가 틀어졌다.

대응은 bundle 전체의 immutable version과 checksum 검증이다. model SHA만으로 release identity를 정의하지 않는다.

## 14. 배포 설계

### 14.1 graph 경계 선택

두 선택이 흔하다.

| 경계 | 장점 | 위험 |
| --- | --- | --- |
| graph가 raw level output 반환 | decoder 교체와 디버깅이 쉬움 | host copy와 다중 출력 ABI 부담 |
| graph가 decode/top-k까지 수행 | transfer 감소, 중앙화된 의미 | custom op, dynamic shape, provider portability 부담 |

어느 쪽이든 golden fixture는 raw와 final 둘 다 보관한다. graph 내부 후처리도 version과 threshold를 manifest에 남긴다.

### 14.2 release 순서

1. training checkpoint와 export code revision을 고정한다.
2. manifest와 anchor artifact를 생성하고 checksum을 계산한다.
3. PyTorch eager와 ONNX CPU provider의 이름별 raw output을 비교한다.
4. 실제 production provider에서 dynamic/odd shape를 비교한다.
5. Python, C++, C# decoder의 candidate index와 box를 비교한다.
6. 최대 shape·혼잡 장면의 p99와 peak memory를 측정한다.
7. shadow traffic에서 level별 funnel과 기존 release의 차이를 본다.
8. 작은 비율 canary 뒤 자동 rollback condition을 확인한다.

오늘 환경에서 `onnx`와 `onnxruntime`이 설치되지 않았다면 이를 성공으로 간주하지 않는다. NumPy·PyTorch와 C++·C# 경계까지만 검증하고 ONNX 및 대상 execution provider parity는 **미검증**으로 문서와 실행 보고에 남긴다.

### 14.3 canary와 rollback

다음 조건은 예시이며 실제 threshold는 baseline 분포와 SLO에서 정한다.

- bundle load 또는 output schema 오류 1건: 즉시 배포 중단
- NaN/Inf detection 비율 증가: 즉시 rollback 후보
- p99 또는 OOM rate가 예산 초과: traffic 확대 중단
- `P3` cap hit rate 급증: 작은 물체 slice 검토 전 확대 중단
- raw parity는 맞지만 final count drift: decoder/NMS bundle rollback
- delayed label의 small-object recall 하락: model/resize/anchor revision 분석

model과 decoder를 따로 rollback하면 호환되지 않는 조합이 생길 수 있다. immutable bundle 단위로 되돌린다.

## 15. 운영 관측 대시보드

### 입력과 shape

- image height, width, aspect ratio, batch size histogram
- resize scale과 padding 분포
- level별 actual `H_k,W_k`와 schema mismatch count
- 허용 범위 밖 shape reject count

### model과 candidate

- level별 raw box, finite box, threshold pass, cap hit, NMS keep
- class별 score와 detection count
- decoded width, height, aspect ratio, clip ratio
- NaN, Inf, exponent clamp count

### 시스템

- queue, preprocess, inference, output copy, decode, NMS p50/p95/p99
- GPU allocated/reserved, host RSS, OOM/retry count
- provider별 fallback op와 graph compile/cache miss
- bundle SHA, model SHA, manifest SHA, anchor SHA

### 품질

- delayed label 기준 small/medium/large precision·recall·AP
- camera, 조도, blur, occlusion, density slice
- level별 matched positive와 최종 true positive의 lineage
- 이전 release 대비 paired disagreement sample

cardinality가 큰 image id를 metric label로 넣지 않는다. trace나 sampling된 debug artifact에 두고 metric은 bounded label을 사용한다.

## 16. 체크리스트

### schema와 artifact

- [ ] level을 이름과 stride로 식별하는가?
- [ ] class/box output 이름과 channel 식이 manifest에 있는가?
- [ ] anchor scale·ratio·variance·순서 artifact의 checksum을 검증하는가?
- [ ] 모르는 schema version을 거부하는가?
- [ ] model과 decoder를 immutable bundle로 묶는가?

### shape와 layout

- [ ] odd·dynamic shape에서 actual feature size를 검사하는가?
- [ ] upsample target을 lateral tensor의 정확한 size로 지정하는가?
- [ ] anchor-major/field-major channel fingerprint가 있는가?
- [ ] flat index round trip을 전 범위에서 검증하는가?
- [ ] shape product와 byte 계산을 checked 64-bit로 하는가?

### 성능과 안정성

- [ ] level별 candidate cap과 global cap을 구분하는가?
- [ ] score 동점과 NaN 정렬 정책이 결정적인가?
- [ ] 최대 shape·혼잡·빈 장면을 benchmark하는가?
- [ ] feature, raw output, candidate, workspace를 함께 예산화하는가?
- [ ] box decode와 IoU의 FP32 경계를 검증하는가?

### 관측과 배포

- [ ] level별 candidate funnel과 cap hit rate를 기록하는가?
- [ ] small-object 품질을 resize·level lineage와 연결하는가?
- [ ] raw output parity와 final detection parity를 분리하는가?
- [ ] production provider와 C++·C# decoder까지 승인했는가?
- [ ] bundle 단위 rollback을 연습했는가?

## 17. 연습문제

### 문제 1

`P3=(80,120,A=3)`, `P4=(40,60,A=6)`, `P5=(20,30,A=6)`일 때 level별 box 수와 prefix를 구하라.

### 문제 2

`P4=(2,3,A=3)`에서 `(row=1,column=2,anchor=2)`의 level index를 구하라. 앞선 `P3` box가 40개면 global index는 얼마인가?

### 문제 3

입력 높이 513, kernel 3, padding 1, dilation 1, stride 8인 convolution의 출력 높이를 구하라. 단순히 $\lceil513/8\rceil$한 값과 같은가?

### 문제 4

$N=1$, $C_f=256$, `P3=100 x 160`, FP32일 때 feature activation payload는 몇 byte와 몇 MiB인가?

### 문제 5

Python과 C++의 raw output checksum은 같지만 작은 box만 큰 box scale로 decode된다. 가장 먼저 비교할 세 artifact를 순서대로 쓰라.

### 문제 6

`P3`의 level cap hit rate가 80%인데 전체 post-NMS detection 수는 변하지 않았다. 왜 정상이라고 단정할 수 없는가?

### 문제 7

ONNX runtime이 outputs를 다른 순서로 반환해도 안전하게 연결하는 방법과, 그 방법만으로 충분하지 않은 이유를 설명하라.

### 문제 8

FP16 model output에서 box `exp`와 NMS IoU만 FP32로 승격하는 이유를 각각 설명하라.

## 18. 해답

### 해답 1

level별 box 수는 다음과 같다.

$$
B_3=80\times120\times3=28{,}800
$$

$$
B_4=40\times60\times6=14{,}400
$$

$$
B_5=20\times30\times6=3{,}600
$$

prefix는 `[0,28800,43200,46800]`이다.

### 해답 2

$$
i_4=((1\times3)+2)\times3+2=17
$$

global index는 $40+17=57$이다.

### 해답 3

$$
H_{\mathrm{out}}=\left\lfloor\frac{513+2-2-1}{8}+1\right\rfloor=65
$$

$\lceil513/8\rceil=65$로 이 구성에서는 같다. 그러나 kernel, padding, dilation이 바뀌면 같다는 보장이 없으므로 runtime shape를 검증해야 한다.

### 해답 4

$$
4\times1\times256\times100\times160=16{,}384{,}000\ \mathrm{bytes}
$$

이는 $16{,}384{,}000/2^{20}=15.625$ MiB다.

### 해답 5

첫째 output name과 stride mapping, 둘째 level별 anchor artifact와 checksum, 셋째 flat index와 variance를 비교한다. raw가 같으므로 backbone보다 decoder schema를 먼저 본다.

### 해답 6

cap 전에 좋은 작은 물체 후보가 잘릴 수 있고 이후 NMS가 다른 후보를 남겨 최종 개수만 같을 수 있다. level별 true-positive lineage나 delayed label의 small-object recall을 함께 봐야 한다.

### 해답 7

manifest의 output name으로 tensor를 조회한다. 그러나 이름만 맞아도 잘못된 stride, anchor revision, channel layout을 연결할 수 있으므로 actual shape, channel 식, anchor checksum, synthetic fingerprint가 추가로 필요하다.

### 해답 8

`exp`는 작은 logit 오차를 multiplicative box 크기 오차로 키우고 overflow할 수 있다. IoU는 거의 맞닿은 box에서 교집합과 합집합의 차이를 계산하므로 낮은 정밀도 반올림에 민감하다. 둘 다 FP32로 승격해 decode와 NMS 경계를 안정화한다.

## 핵심 요약

- SSD/FPN의 배포 단위는 여러 tensor가 아니라 이름, stride, anchor, layout이 묶인 versioned pyramid ABI다.
- 원본의 다중 스케일 직관은 유효하지만 고해상도 feature만으로 작은 물체 recall이 자동 보장되지는 않는다.
- `P3/P4/P5`는 runtime 반환 위치가 아니라 output 이름과 검증된 shape로 연결한다.
- flat index는 `level -> row -> column -> anchor` 순서와 prefix를 모든 언어에서 같은 64비트 식으로 계산한다.
- odd·dynamic shape에서는 `scale_factor=2` 대신 lateral tensor의 실제 target size를 사용한다.
- 고해상도 level은 feature memory뿐 아니라 raw output, transfer, decode, top-k, NMS 비용을 함께 늘린다.
- level별 cap과 전역 cap은 다른 실패 모드를 가지므로 cap hit rate와 작은 물체 funnel을 함께 관측한다.
- model, manifest, anchors, decoder, golden fixture를 immutable bundle로 묶어야 부분 rollback의 호환성 사고를 막을 수 있다.
- raw output, decoded candidate, final detection parity를 분리하면 model 장애와 후처리 장애를 빠르게 격리할 수 있다.
- ONNX CPU와 실제 production provider, Python/C++/C# decoder가 모두 통과해야 크로스런타임 배포가 완료된다.

## 다음 학습 예고

다음 글은 같은 `02-05.YOLO.md`의 3회차 실무 엔지니어 Part 3/3이다. Focal Loss와 RetinaNet의 sigmoid ABI, foreground prior bias, FP16 안정성, level별 normalization, threshold calibration, 양자화와 candidate 폭주, production monitoring·rollback을 다룬다. Part 3을 끝내면 7/18 `02-06.FCOS_DETR.md`로 이동한다.
