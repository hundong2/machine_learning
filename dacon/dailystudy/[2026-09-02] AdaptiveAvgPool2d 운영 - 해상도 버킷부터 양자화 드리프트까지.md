<!-- curriculum: cycle=3; level=production-engineering; source_index=3/18; source=02-02.AdaptiveAvgPooll2d.md; part=1/1 -->

# AdaptiveAvgPool2d 운영: 해상도 버킷부터 양자화 드리프트까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-02 |
| 회차·수준 | 3회차 · 실무 엔지니어 |
| 현재 소스 | 3/18 · `02-02.AdaptiveAvgPooll2d.md` |
| Part | 1/1 |
| 이전 소스 | `02-1.LazyLinear.md` |
| 다음 소스 | `02-03.SPP.md` |
| 예상 학습 시간 | 140~180분 |
| 실행 검증 환경 | Python 3.12.12 · NumPy 2.3.5 · PyTorch 2.13.0 · clang C++17 · Mono C# |

## 1. 오늘의 운영 질문

1회차에는 floor·ceil bin 경계와 overlap을 유도했고, 2회차에는 독립 NumPy forward·backward와 2,304개 shape 전수 테스트를 만들었다. 이번 회차는 그 설명과 구현을 반복하지 않고 다음 질문에 답한다.

> `AdaptiveAvgPool2d`가 가변 shape를 받아도, 왜 실제 서비스에는 해상도 버킷·패딩 계약·크로스런타임 골든·양자화 게이트·shape 드리프트 모니터링이 따로 필요한가?

적응형 풀링이 고정하는 것은 **pool 이후의 공간 shape**다. 다음 항목은 자동으로 해결되지 않는다.

- 크기가 다른 이미지를 하나의 dense batch로 묶는 문제
- pool 이전 backbone의 연산량과 activation memory
- padding이 convolution과 평균에 끼치는 영향
- compiler가 동적 shape마다 새 graph를 만드는 문제
- Python, C++, C# runtime의 layout·dtype·반올림 차이
- 임의 출력 grid의 ONNX export와 accelerator 지원
- 입력 해상도 분포 변화가 latency와 정확도에 만드는 드리프트

오늘의 결과물은 모델 코드 한 줄이 아니라 versioning 가능한 **shape release contract**다.

## 2. 학습 목표

학습을 마치면 다음을 할 수 있어야 한다.

1. 입력 해상도를 제한된 bucket으로 routing하고 초과 요청을 거부한다.
2. bin 경계를 64비트 정수로 계산해 Python·C++·C# 결과를 맞춘다.
3. padding이 평균을 오염시키는 경로와 masked pooling의 한계를 설명한다.
4. NCHW·NHWC와 float32·float16·정수 양자화 계약을 분리한다.
5. 해상도 bucket별 latency·memory를 재현 가능한 방법으로 측정한다.
6. arbitrary adaptive grid와 GAP의 ONNX 배포 위험을 구분한다.
7. shape drift, padding ratio, compile cache miss, quantization saturation을 운영 지표로 만든다.
8. shadow·canary·rollback이 가능한 release bundle을 설계한다.

## 3. 선수 지식과 기호

### 3.1 텐서 계약

backbone이 만든 NCHW feature를 다음처럼 둔다.

$$
X \in \mathbb{R}^{N \times C \times H_i \times W_i}
$$

목표 grid가 $(H_o,W_o)$이면 출력은 다음과 같다.

$$
Y \in \mathbb{R}^{N \times C \times H_o \times W_o}
$$

- $N$: batch 크기
- $C$: channel 수
- $H_i,W_i$: pool 직전 feature 높이와 너비
- $H_o,W_o$: 고정할 출력 grid
- $A_{p,q}$: 출력 위치 $(p,q)$가 읽는 실제 bin 면적
- $M$: 유효 위치가 1, padding이 0인 mask

입력 이미지의 $H,W$와 backbone feature의 $H_i,W_i$를 혼용하지 않는다. stride 16 backbone이라 해도 padding·ceil mode·stem 구조에 따라 단순히 $H/16,W/16$이 아닐 수 있다.

### 3.2 운영 용어

| 용어 | 이 문서의 의미 |
| --- | --- |
| 해상도 bucket | 같은 compiled graph와 batch를 공유하도록 정한 대표 입력 크기 |
| occupancy | bucket 전체 pixel 중 실제 이미지가 차지하는 비율 |
| shape ABI | layout, dtype, channel, 허용 $H,W$, pool grid를 묶은 입출력 계약 |
| golden | runtime이 달라도 같아야 하는 작은 입력·출력 기준값 |
| release bundle | model, preprocessing, manifest, golden, benchmark를 함께 versioning한 묶음 |
| shadow | 실사용 응답에는 영향 없이 새 모델을 병렬 실행하는 단계 |
| canary | 일부 실제 요청에 새 release를 적용하는 단계 |

## 4. 직관: 자유로운 shape가 아니라 통제된 shape 집합

`AdaptiveAvgPool2d((2,3))`는 다음 두 입력을 같은 head로 보낼 수 있다.

```text
(N, C, 16, 16) -> adaptive pool -> (N, C, 2, 3)
(N, C, 20, 32) -> adaptive pool -> (N, C, 2, 3)
```

그러나 production compiler는 `(16,16)`과 `(20,32)`에 서로 다른 optimized graph를 만들 수 있다. 요청마다 새로운 shape가 들어오면 compile cache가 커지고 첫 요청 latency가 튄다. GPU batch도 같은 shape끼리만 효율적으로 묶인다.

따라서 실무 계약은 보통 다음처럼 제한한다.

```text
원본 H,W
   |
   | validate: channel, pixel budget, aspect ratio
   v
bucket 선택: 256x256 / 320x512 / 512x512
   |
   | resize + pad + mask
   v
고정 shape batch와 사전 컴파일 graph
   |
   v
backbone -> AdaptiveAvgPool2d(2,3) -> fixed head
```

adaptive pooling은 이 파이프라인의 마지막 shape adapter다. 입구의 무제한 shape 허가증이 아니다.

## 5. bin 경계를 크로스런타임 ABI로 만들기

출력 높이 인덱스 $p$의 시작과 끝은 다음과 같다.

$$
h_s(p)=\left\lfloor\frac{pH_i}{H_o}\right\rfloor
$$

$$
h_e(p)=\left\lceil\frac{(p+1)H_i}{H_o}\right\rceil
$$

너비도 동일하다.

$$
w_s(q)=\left\lfloor\frac{qW_i}{W_o}\right\rfloor
$$

$$
w_e(q)=\left\lceil\frac{(q+1)W_i}{W_o}\right\rceil
$$

양의 정수 나눗셈에서 ceil은 다음처럼 계산한다.

$$
\left\lceil\frac{a}{b}\right\rceil
=
\left\lfloor\frac{a+b-1}{b}\right\rfloor
$$

production helper는 중간 곱을 64비트로 계산해야 한다. `output_index * input_size`가 32비트에서 overflow하면 작은 unit test는 통과해도 큰 panorama나 feature tensor에서 경계가 음수가 될 수 있다.

출력 값은 다음과 같다.

$$
Y_{n,c,p,q}
=
\frac{1}{A_{p,q}}
\sum_{h=h_s(p)}^{h_e(p)-1}
\sum_{w=w_s(q)}^{w_e(q)-1}
X_{n,c,h,w}
$$

$$
A_{p,q}
=
\bigl(h_e(p)-h_s(p)\bigr)
\bigl(w_e(q)-w_s(q)\bigr)
$$

`kernel_size`와 `stride` 하나를 manifest에 기록하는 것으로는 충분하지 않다. 정확한 ABI는 입력 shape, 출력 grid, 위 경계식, end-exclusive 규칙이다.

## 6. 손계산 golden: $5 \times 7 \rightarrow 3 \times 2$

입력을 1부터 35까지 row-major로 채운다.

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 & 7 \\
8 & 9 & 10 & 11 & 12 & 13 & 14 \\
15 & 16 & 17 & 18 & 19 & 20 & 21 \\
22 & 23 & 24 & 25 & 26 & 27 & 28 \\
29 & 30 & 31 & 32 & 33 & 34 & 35
\end{bmatrix}
$$

높이 구간은 `[0:2]`, `[1:4]`, `[3:5]`이고 너비 구간은 `[0:4]`, `[3:7]`이다. 결과는 다음과 같다.

$$
Y=
\begin{bmatrix}
6 & 9 \\
16.5 & 19.5 \\
27 & 30
\end{bmatrix}
$$

이 예제는 높이와 너비가 모두 나누어떨어지지 않고, 정수가 아닌 평균도 포함한다. layout 축 교환, end-inclusive 오류, 정수 나눗셈, 잘못된 분모를 동시에 잡기에 좋다.

## 7. 해상도 bucket을 선택하는 수학

요청 크기를 $(H,W)$, bucket $b$를 $(H_b,W_b)$라 하자. crop하지 않고 pad할 수 있는 후보는 다음 조건을 만족해야 한다.

$$
H \le H_b,
\qquad
W \le W_b
$$

유효 pixel 점유율은 다음과 같다.

$$
\rho(H,W;b)=\frac{HW}{H_bW_b}
$$

padding 비율은 $1-\rho$다. 후보 중 단순히 면적이 가장 작은 bucket을 선택할 수 있다.

$$
b^*
=
\underset{b}{\operatorname{argmin}}
\left(H_bW_b-HW\right)
$$

실무에서는 여기에 aspect ratio 왜곡, batch queue 길이, device memory, SLA penalty를 더한 cost를 쓴다. bucket이 너무 많으면 padding은 줄지만 compiled graph와 queue가 늘고 batch 결합률이 낮아진다.

## 8. 실행 가능한 Python 운영 골든

다음 코드는 독립 실행 가능하다. NumPy 기준 구현, PyTorch parity, NHWC 변환, bucket routing, padding-aware GAP, 간단한 정수 양자화 기준을 한 번에 검사한다.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def start_index(index: int, output_size: int, input_size: int) -> int:
    return (index * input_size) // output_size


def end_index(index: int, output_size: int, input_size: int) -> int:
    numerator = (index + 1) * input_size
    return (numerator + output_size - 1) // output_size


def pool_numpy(x: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError("expected NCHW rank-4 input")
    n, c, input_h, input_w = x.shape
    output_h, output_w = grid
    if min(n, c, input_h, input_w, output_h, output_w) <= 0:
        raise ValueError("all dimensions must be positive")
    y = np.empty((n, c, output_h, output_w), dtype=np.float64)
    x64 = x.astype(np.float64, copy=False)
    for oh in range(output_h):
        hs = start_index(oh, output_h, input_h)
        he = end_index(oh, output_h, input_h)
        for ow in range(output_w):
            ws = start_index(ow, output_w, input_w)
            we = end_index(ow, output_w, input_w)
            y[:, :, oh, ow] = x64[:, :, hs:he, ws:we].mean(
                axis=(-2, -1)
            )
    return y


@dataclass(frozen=True)
class Bucket:
    height: int
    width: int

    @property
    def pixels(self) -> int:
        return self.height * self.width


BUCKETS = (Bucket(256, 256), Bucket(320, 512), Bucket(512, 512))


def choose_bucket(height: int, width: int) -> Bucket:
    if height <= 0 or width <= 0:
        raise ValueError("invalid image size")
    candidates = [
        bucket
        for bucket in BUCKETS
        if height <= bucket.height and width <= bucket.width
    ]
    if not candidates:
        raise ValueError("image exceeds the release pixel contract")
    return min(candidates, key=lambda bucket: bucket.pixels)


def masked_gap(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or mask.ndim != 4:
        raise ValueError("x and mask must be NCHW")
    if mask.shape[0] != x.shape[0] or mask.shape[-2:] != x.shape[-2:]:
        raise ValueError("mask shape mismatch")
    if mask.shape[1] not in (1, x.shape[1]):
        raise ValueError("mask channel must be 1 or C")
    numerator = F.adaptive_avg_pool2d(x * mask, (1, 1))
    denominator = F.adaptive_avg_pool2d(mask, (1, 1))
    return numerator / denominator.clamp_min(1e-12)


def run_contract() -> None:
    x_np = np.arange(1, 36, dtype=np.float32).reshape(1, 1, 5, 7)
    expected = np.array(
        [[[[6.0, 9.0], [16.5, 19.5], [27.0, 30.0]]]],
        dtype=np.float64,
    )
    manual = pool_numpy(x_np, (3, 2))
    actual = F.adaptive_avg_pool2d(torch.from_numpy(x_np), (3, 2))
    np.testing.assert_array_equal(manual, expected)
    np.testing.assert_array_equal(actual.numpy(), expected.astype(np.float32))

    # mobile NHWC adapter가 channel을 공간 축으로 오해하지 않는지 확인한다.
    nhwc = np.stack(
        [x_np[0, 0], x_np[0, 0] + 100.0, x_np[0, 0] + 200.0],
        axis=-1,
    )[None, ...]
    nchw = np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))
    layout_output = F.adaptive_avg_pool2d(torch.from_numpy(nchw), (3, 2))
    assert tuple(layout_output.shape) == (1, 3, 3, 2)
    torch.testing.assert_close(
        layout_output[:, 1] - layout_output[:, 0],
        torch.full((1, 3, 2), 100.0),
    )

    # 3x4 유효 영역을 5x7에 zero-pad하면 plain GAP는 오염된다.
    padded = torch.zeros(1, 1, 5, 7, dtype=torch.float64)
    mask = torch.zeros_like(padded)
    padded[:, :, :3, :4] = torch.arange(1, 13, dtype=torch.float64).reshape(3, 4)
    mask[:, :, :3, :4] = 1.0
    plain = F.adaptive_avg_pool2d(padded, 1).item()
    corrected = masked_gap(padded, mask).item()
    assert abs(plain - 2.2285714285714286) < 1e-12
    assert corrected == 6.5

    # scale=0.25, zero-point=0인 정수 저장값도 같은 golden을 만든다.
    scale = 0.25
    quantized = np.rint(x_np / scale).astype(np.int32)
    dequantized_pool = pool_numpy(quantized, (3, 2)) * scale
    np.testing.assert_array_equal(dequantized_pool, expected)

    routed = {
        shape: choose_bucket(*shape)
        for shape in [(224, 224), (300, 500), (480, 360)]
    }
    assert routed[(224, 224)] == Bucket(256, 256)
    assert routed[(300, 500)] == Bucket(320, 512)
    assert routed[(480, 360)] == Bucket(512, 512)
    try:
        choose_bucket(900, 1200)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized request must be rejected")

    print(manual.reshape(3, 2))
    print(f"plain_gap={plain:.9f} masked_gap={corrected:.9f}")
    print({shape: (bucket.height, bucket.width) for shape, bucket in routed.items()})
    print("portable contract passed")


if __name__ == "__main__":
    run_contract()
```

예상 핵심 출력은 다음과 같다.

```text
[[ 6.   9. ]
 [16.5 19.5]
 [27.  30. ]]
plain_gap=2.228571429 masked_gap=6.500000000
{(224, 224): (256, 256), (300, 500): (320, 512), (480, 360): (512, 512)}
portable contract passed
```

## 9. padding-aware 평균의 유도와 한계

mask $M_{n,1,h,w}$가 유효 위치에서 1, padding에서 0이라 하자. 한 bin의 masked average는 다음과 같다.

$$
Y^{\mathrm{masked}}_{n,c,p,q}
=
\frac{
\displaystyle
\sum_{(h,w)\in R_{p,q}}
X_{n,c,h,w}M_{n,1,h,w}
}{
\displaystyle
\max\left(
\sum_{(h,w)\in R_{p,q}}M_{n,1,h,w},
\epsilon
\right)
}
$$

`pool(x * mask) / pool(mask)`가 이 식과 같은 이유는 분자와 분모가 동일한 bin 면적 $A_{p,q}$로 나뉘어 상쇄되기 때문이다.

하지만 feature 끝에서 mask를 곱하는 것만으로 모든 padding 오염이 사라지지는 않는다. 앞선 convolution의 receptive field가 0 padding과 유효 pixel을 함께 읽었다면 경계 feature 자체가 이미 달라졌다. 다음 전략을 task와 backend에 맞게 비교한다.

- 유사 aspect ratio끼리 bucket을 묶어 padding을 줄인다.
- resize·center crop처럼 padding 없는 정책을 쓴다.
- 각 stage에서 mask를 downsample하고 masked operator를 쓴다.
- padding을 학습과 추론에서 동일하게 노출한다.
- 유효 영역 crop 뒤 pooling하는 별도 kernel을 쓴다.

빈 bin의 분모가 0인 경우 `clamp`로 조용히 0을 만드는 것보다 요청 또는 bucket 계약을 거부하는 편이 안전할 수 있다. 정책을 manifest에 명시해야 한다.

## 10. PyTorch serving head와 shape guard

다음 코드는 독립 실행 가능한 최소 serving head다. adaptive pool 앞에서 channel·dtype·공간 범위를 검증하고 pool 뒤 feature 길이를 assert한다.

```python
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ShapeContract:
    channels: int = 8
    min_height: int = 4
    min_width: int = 4
    max_height: int = 64
    max_width: int = 64
    output_height: int = 2
    output_width: int = 3


class ServingHead(nn.Module):
    def __init__(self, classes: int, contract: ShapeContract) -> None:
        super().__init__()
        self.contract = contract
        self.pool = nn.AdaptiveAvgPool2d(
            (contract.output_height, contract.output_width)
        )
        features = (
            contract.channels
            * contract.output_height
            * contract.output_width
        )
        self.classifier = nn.Linear(features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.contract
        if x.ndim != 4 or x.shape[1] != c.channels:
            raise ValueError("expected release NCHW channel contract")
        if x.dtype not in (torch.float16, torch.float32):
            raise TypeError("unsupported release dtype")
        height, width = x.shape[-2:]
        if not (c.min_height <= height <= c.max_height):
            raise ValueError("height outside release contract")
        if not (c.min_width <= width <= c.max_width):
            raise ValueError("width outside release contract")
        pooled = self.pool(x)
        flattened = pooled.flatten(start_dim=1)
        expected = c.channels * c.output_height * c.output_width
        if flattened.shape[1] != expected:
            raise RuntimeError("pool ABI mismatch")
        return self.classifier(flattened)


torch.manual_seed(20260902)
head = ServingHead(classes=5, contract=ShapeContract()).eval()
with torch.no_grad():
    small = head(torch.randn(2, 8, 11, 13))
    large = head(torch.randn(3, 8, 31, 47))
assert tuple(small.shape) == (2, 5)
assert tuple(large.shape) == (3, 5)
try:
    head(torch.randn(1, 8, 100, 100))
except ValueError:
    pass
else:
    raise AssertionError("oversized feature must be rejected")
print(tuple(small.shape), tuple(large.shape), "shape guard passed")
```

guard는 모델 정확도를 높이지 않는다. 대신 잘못된 요청이 allocator OOM, compiler 재컴파일, `Linear` 오류로 늦게 터지기 전에 API 경계에서 설명 가능한 오류로 바꾼다.

## 11. tensor shape 추적

입력 bucket과 backbone stride가 다르더라도 adaptive pool 뒤는 같은 shape이어야 한다.

| 단계 | 정사각 bucket | 가로형 bucket | 계약 |
| --- | --- | --- | --- |
| decode | `(N, 256, 256, 3)` uint8 NHWC | `(N, 320, 512, 3)` uint8 NHWC | mobile·web 입력 |
| layout·normalize | `(N, 3, 256, 256)` float32 NCHW | `(N, 3, 320, 512)` float32 NCHW | color order와 scale 고정 |
| backbone 출력 예 | `(N, 8, 16, 16)` | `(N, 8, 20, 32)` | 실제 model graph로 확인 |
| adaptive pool | `(N, 8, 2, 3)` | `(N, 8, 2, 3)` | 공간 shape 고정 |
| flatten | `(N, 48)` | `(N, 48)` | $8 \times 2 \times 3$ |
| classifier | `(N, 5)` | `(N, 5)` | class order 고정 |

`decode`와 model input의 layout이 다르다는 사실을 manifest에 명시한다. transpose 뒤 non-contiguous view를 runtime이 암묵적으로 copy하면 예상하지 못한 latency가 생길 수 있으므로 변환 위치와 ownership도 측정한다.

## 12. framework 간 layout·dtype 대응

| 환경 | 대표 layout | 경계 정수 | 누적 dtype | 출력 dtype | 운영 확인 |
| --- | --- | --- | --- | --- | --- |
| PyTorch Python | NCHW | 내부 구현 | backend별 확인 | 보통 입력과 동일 | CPU·GPU parity |
| NumPy golden | NCHW로 강제 | Python 정수 | float64 | float64 | 엄격한 기준값 |
| C++ golden | HW plane | `std::int64_t` | double | double | 경계·값 ABI |
| C# golden | HW plane | `long` | double | double | mobile adapter ABI |
| ONNX runtime | model graph 계약 | operator별 | provider별 | graph dtype | provider별 parity |
| NHWC accelerator | NHWC | kernel별 | device별 | device별 | transpose 포함 latency |

float16·bfloat16에서 pool 결과가 float32 기준과 조금 다른 것은 곧바로 버그가 아니다. 다음 두 한도를 함께 둔다.

1. tensor 한도: 최대 절대 오차와 상대 오차
2. task 한도: top-1 agreement, calibration, slice별 recall 변화

bitwise equality를 모든 device에 강제하면 정상적인 reduction 순서 차이를 장애로 오인할 수 있다. 반대로 tolerance를 크게 잡아 class 순서가 바뀌는 문제를 숨겨도 안 된다.

## 13. 정수 양자화 평균의 단계별 유도

입력 실수 $x$를 scale $s_x$, zero-point $z_x$로 양자화하면 다음과 같다.

$$
x \approx s_x(q_x-z_x)
$$

bin 안 정수의 합을 $S_q=\sum q_x$라 하면 실수 평균은 다음과 같다.

$$
\bar{x}
\approx
s_x\left(\frac{S_q}{A}-z_x\right)
$$

출력 scale과 zero-point가 $s_y,z_y$이면 requantization은 다음과 같다.

$$
q_y
=
\operatorname{clip}
\left(
\operatorname{round}
\left[
\frac{s_x}{s_y}
\left(\frac{S_q}{A}-z_x\right)
+z_y
\right]
\right)
$$

여기서 backend마다 확인할 항목은 다음과 같다.

- `round-to-nearest-even`인지 away-from-zero인지
- 합산 accumulator가 32비트인지 64비트인지
- division과 requantization 순서
- saturation 범위
- per-tensor와 per-channel scale 지원
- GAP와 arbitrary grid의 quantized kernel 지원 차이

최악의 accumulator 크기를 단순하게 상계하면 다음과 같다.

$$
|S_q-Az_x|
\le
A\max(|q_{\min}-z_x|,|q_{\max}-z_x|)
$$

큰 feature map의 GAP를 int32로 합산할 때는 이 상계가 int32 범위를 넘지 않는지 release gate에서 계산한다. calibration dataset에는 평균적인 이미지뿐 아니라 최대 해상도·높은 activation·padding이 많은 slice를 포함한다.

## 14. C++17 portable golden

다음 코드는 외부 tensor library 없이 실행 가능하다. production LibTorch 또는 accelerator kernel을 대체하는 코드가 아니라 경계와 값 ABI를 고정하는 oracle이다.

```cpp
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

std::int64_t start_index(
    std::int64_t output_index,
    std::int64_t output_size,
    std::int64_t input_size) {
    return output_index * input_size / output_size;
}

std::int64_t end_index(
    std::int64_t output_index,
    std::int64_t output_size,
    std::int64_t input_size) {
    const std::int64_t numerator = (output_index + 1) * input_size;
    return (numerator + output_size - 1) / output_size;
}

std::vector<double> adaptive_avg_pool2d(
    const std::vector<double>& input,
    std::int64_t input_h,
    std::int64_t input_w,
    std::int64_t output_h,
    std::int64_t output_w) {
    if (input_h <= 0 || input_w <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::invalid_argument("sizes must be positive");
    }
    if (static_cast<std::int64_t>(input.size()) != input_h * input_w) {
        throw std::invalid_argument("input size mismatch");
    }
    std::vector<double> output(
        static_cast<std::size_t>(output_h * output_w), 0.0);
    for (std::int64_t oh = 0; oh < output_h; ++oh) {
        const auto hs = start_index(oh, output_h, input_h);
        const auto he = end_index(oh, output_h, input_h);
        for (std::int64_t ow = 0; ow < output_w; ++ow) {
            const auto ws = start_index(ow, output_w, input_w);
            const auto we = end_index(ow, output_w, input_w);
            double sum = 0.0;
            for (std::int64_t h = hs; h < he; ++h) {
                for (std::int64_t w = ws; w < we; ++w) {
                    sum += input[static_cast<std::size_t>(h * input_w + w)];
                }
            }
            const auto area = (he - hs) * (we - ws);
            output[static_cast<std::size_t>(oh * output_w + ow)] =
                sum / static_cast<double>(area);
        }
    }
    return output;
}

int main() {
    std::vector<double> input(35);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<double>(index + 1);
    }
    const auto actual = adaptive_avg_pool2d(input, 5, 7, 3, 2);
    const std::vector<double> expected{6.0, 9.0, 16.5, 19.5, 27.0, 30.0};
    assert(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        assert(std::abs(actual[index] - expected[index]) < 1e-12);
    }
    std::cout << std::fixed << std::setprecision(1);
    for (double value : actual) {
        std::cout << value << ' ';
    }
    std::cout << "checksum=108.0\n";
}
```

컴파일 명령은 다음과 같다.

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -pedantic adaptive_pool_release.cpp -o adaptive_pool_release
./adaptive_pool_release
```

## 15. C# portable golden

다음 코드는 C++과 같은 $5 \times 7 \rightarrow 3 \times 2$ 결과를 검사한다. 경계 곱은 `long`으로 계산한다.

```csharp
using System;

public static class AdaptivePoolRelease
{
    private static long StartIndex(long index, long outputSize, long inputSize)
    {
        return index * inputSize / outputSize;
    }

    private static long EndIndex(long index, long outputSize, long inputSize)
    {
        long numerator = (index + 1) * inputSize;
        return (numerator + outputSize - 1) / outputSize;
    }

    private static double[,] Pool(double[,] input, int outputH, int outputW)
    {
        int inputH = input.GetLength(0);
        int inputW = input.GetLength(1);
        if (inputH <= 0 || inputW <= 0 || outputH <= 0 || outputW <= 0)
        {
            throw new ArgumentOutOfRangeException("sizes must be positive");
        }
        var output = new double[outputH, outputW];
        for (int oh = 0; oh < outputH; ++oh)
        {
            int hs = checked((int)StartIndex(oh, outputH, inputH));
            int he = checked((int)EndIndex(oh, outputH, inputH));
            for (int ow = 0; ow < outputW; ++ow)
            {
                int ws = checked((int)StartIndex(ow, outputW, inputW));
                int we = checked((int)EndIndex(ow, outputW, inputW));
                double sum = 0.0;
                for (int h = hs; h < he; ++h)
                {
                    for (int w = ws; w < we; ++w)
                    {
                        sum += input[h, w];
                    }
                }
                output[oh, ow] = sum / ((he - hs) * (we - ws));
            }
        }
        return output;
    }

    public static void Main()
    {
        var input = new double[5, 7];
        int value = 1;
        for (int h = 0; h < 5; ++h)
        {
            for (int w = 0; w < 7; ++w)
            {
                input[h, w] = value++;
            }
        }
        double[,] expected = {
            { 6.0, 9.0 },
            { 16.5, 19.5 },
            { 27.0, 30.0 }
        };
        double[,] actual = Pool(input, 3, 2);
        double checksum = 0.0;
        for (int h = 0; h < 3; ++h)
        {
            for (int w = 0; w < 2; ++w)
            {
                if (Math.Abs(actual[h, w] - expected[h, w]) > 1e-12)
                {
                    throw new Exception("portable golden mismatch");
                }
                checksum += actual[h, w];
                Console.Write(actual[h, w].ToString("F1") + " ");
            }
        }
        Console.WriteLine("checksum=" + checksum.ToString("F1"));
    }
}
```

컴파일·실행 명령은 다음과 같다.

```bash
csc -nologo -optimize+ -out:AdaptivePoolRelease.exe AdaptivePoolRelease.cs
mono AdaptivePoolRelease.exe
```

## 16. 성능 측정: pool 단독 수치가 전부가 아니다

### 16.1 계산량과 메모리

naive adaptive pooling의 읽기 비용은 bin 면적 합에 비례한다.

$$
\operatorname{work}
\propto
NC
\sum_{p=0}^{H_o-1}
\sum_{q=0}^{W_o-1}
A_{p,q}
$$

pool 출력 저장 크기는 dtype byte 수를 $d$라 할 때 다음과 같다.

$$
M_{\mathrm{out}}=dNCH_oW_o
$$

하지만 peak memory는 pool 전 activation이 지배하는 경우가 많다.

$$
M_{\mathrm{in}}=dNCH_iW_i
$$

입력 해상도를 두 배로 키우면 $H_iW_i$는 대략 네 배가 된다. 마지막 adaptive pool이 작아도 backbone의 FLOPs와 memory는 그대로 커진다.

### 16.2 실행 가능한 CPU microbenchmark

다음 코드는 독립 실행 가능하다. 결과 숫자는 하드웨어와 부하에 따라 달라지므로 절대 threshold를 코드에 assert하지 않고 shape와 finite 결과만 검사한다.

```python
from time import perf_counter_ns

import numpy as np
import torch
import torch.nn.functional as F


def benchmark(shape: tuple[int, int, int, int]) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(20260902)
    x = torch.randn(shape, generator=generator)
    for _ in range(20):
        F.adaptive_avg_pool2d(x, (2, 3))
    samples_ms = []
    for _ in range(200):
        start = perf_counter_ns()
        y = F.adaptive_avg_pool2d(x, (2, 3))
        samples_ms.append((perf_counter_ns() - start) / 1_000_000)
    assert tuple(y.shape) == (shape[0], shape[1], 2, 3)
    assert torch.isfinite(y).all()
    return float(np.percentile(samples_ms, 50)), float(
        np.percentile(samples_ms, 95)
    )


torch.set_num_threads(1)
for feature_shape in [(8, 64, 16, 16), (8, 64, 20, 32), (8, 64, 32, 32)]:
    p50, p95 = benchmark(feature_shape)
    print(feature_shape, f"p50={p50:.4f}ms", f"p95={p95:.4f}ms")
```

microbenchmark에는 최소한 다음 메타데이터를 붙인다.

- CPU·GPU·accelerator 모델과 thread 수
- runtime와 compiler version
- eager·compiled·quantized 여부
- batch, dtype, layout, input bucket, output grid
- warm-up 횟수와 sample 수
- p50·p95·p99, peak memory, compile 시간을 분리한 값

pool만 빨라도 전처리 transpose, padding, backbone, device transfer가 느리면 end-to-end SLA는 실패한다. 최종 gate는 전체 request path를 측정한다.

## 17. ONNX와 accelerator 배포 계약

### 17.1 GAP와 arbitrary grid를 구분한다

`AdaptiveAvgPool2d((1,1))`은 global spatial average라서 `GlobalAveragePool`처럼 표현되기 쉽다. 반면 `(2,3)` 같은 arbitrary grid는 input shape와 output shape의 조합에 따라 고정 `AveragePool`, 여러 slice·reduce subgraph, 또는 미지원 연산으로 처리될 수 있다.

다음 코드는 설명용이다. 현재 실행 환경에는 `onnx`와 `onnxruntime` package가 없어 실제 export와 target parity는 **미검증**이다.

```python
# 설명용: release exporter 환경에서 onnx와 target runtime을 설치한 뒤 실행한다.
import torch
from torch import nn

model = nn.AdaptiveAvgPool2d((2, 3)).eval()
example = torch.randn(1, 8, 20, 32)
torch.onnx.export(
    model,
    (example,),
    "adaptive_pool_2x3.onnx",
    input_names=["features"],
    output_names=["pooled"],
    dynamic_axes={
        "features": {0: "batch", 2: "height", 3: "width"},
        "pooled": {0: "batch"},
    },
    dynamo=True,
)
```

export 파일 생성만으로 release를 통과시키지 않는다.

1. graph의 실제 operator와 fallback node를 검사한다.
2. 모든 허용 bucket을 target execution provider에서 실행한다.
3. Python golden과 tensor 오차·top-1 agreement를 비교한다.
4. cold compile과 steady-state latency를 분리한다.
5. float와 quantized model을 각각 검사한다.
6. 지원하지 않는 bucket은 API gateway에서 거부한다.

### 17.2 release manifest 예시

다음 JSON은 설명용이지만 기계 판독 가능한 형태다.

```json
{
  "schema_version": 1,
  "layout": "NCHW",
  "input_dtype": "float32",
  "channels": 3,
  "color_order": "RGB",
  "image_buckets": [[256, 256], [320, 512], [512, 512]],
  "pool_grid": [2, 3],
  "feature_channels": 8,
  "flatten_features": 48,
  "oversize_policy": "reject",
  "padding_value": 0.0,
  "mask_policy": "feature_mask_v1",
  "golden_id": "adaptive-pool-5x7-to-3x2-v1",
  "quantization": "none",
  "runtime": "target-runtime-and-provider-version"
}
```

manifest의 hash를 model artifact 이름과 telemetry에 넣으면 요청 로그만으로도 어떤 shape 계약이 실행됐는지 추적할 수 있다.

## 18. 테스트와 release gate

### 18.1 계층별 테스트

| 층 | 검사 | 잡는 실패 |
| --- | --- | --- |
| 수학 golden | `5 x 7 -> 3 x 2` 값·checksum | 경계, 분모, 정수 truncation |
| layout golden | 비대칭 shape와 channel offset | NHWC·NCHW 축 교환 |
| padding golden | plain GAP와 masked GAP 비교 | 0 padding 평균 오염 |
| bucket test | 모든 경계·초과 요청 | 잘못된 routing과 무제한 shape |
| dtype test | float64·float32·저정밀 비교 | accumulator·반올림 drift |
| cross-runtime | Python·C++·C# 동일 값 | 언어별 integer·layout 차이 |
| export parity | 모든 bucket의 target 실행 | graph 분해·provider 미지원 |
| load test | bucket별 queue와 p99 | batching 붕괴·compile spike |
| canary metric | accuracy·latency·OOM | 실제 분포의 회귀 |

### 18.2 negative test

- rank가 4가 아니면 거부한다.
- channel, color order, dtype이 manifest와 다르면 거부한다.
- $H,W$가 0이거나 최대 bucket을 넘으면 거부한다.
- mask에 유효 pixel이 없는 bin을 거부한다.
- 알려지지 않은 bucket에서 compiler fallback을 허용하지 않는다.
- quantized accumulator 상계를 넘는 shape를 거부한다.
- release bundle의 model·manifest·golden hash가 다르면 시작하지 않는다.

### 18.3 cached graph gate

모든 허용 `(batch,bucket,dtype)` 조합을 시작 전에 warm-up할지, 첫 요청에 compile할지 정책을 정한다. warm-up 목록과 실제 runtime cache key가 다르면 준비했다고 생각한 shape가 production에서 다시 compile될 수 있다.

## 19. 디버깅 플레이북

### 증상 A: 특정 aspect ratio에서만 정확도가 떨어진다

1. 원본 $H,W$, 선택 bucket, resize scale, padding ratio를 조회한다.
2. 학습과 serving의 resize·pad anchor가 같은지 확인한다.
3. plain pool과 masked pool 결과를 비교한다.
4. padding 경계가 backbone receptive field에 미친 영향을 feature map으로 본다.
5. bucket별 confusion matrix와 작은 객체 slice를 비교한다.

### 증상 B: 첫 요청만 매우 느리다

1. compiler cache miss와 graph compile 시간을 분리한다.
2. 요청 shape가 manifest bucket과 정확히 같은지 확인한다.
3. batch dimension도 cache key인지 확인한다.
4. transpose·contiguous copy가 cold path에 있는지 확인한다.
5. startup warm-up과 readiness 순서를 수정한다.

### 증상 C: Python과 mobile logits가 다르다

1. decode color order와 EXIF orientation을 비교한다.
2. NHWC·NCHW 변환 후 channel별 golden을 비교한다.
3. resize, padding 값, normalization을 비교한다.
4. pool 단독 `5 x 7 -> 3 x 2` golden을 실행한다.
5. quantization scale·zero-point·rounding과 accumulator를 비교한다.
6. layer별 최대 절대 오차를 찾아 최초 divergence를 좁힌다.

### 증상 D: 고해상도 traffic 증가 뒤 OOM이 난다

1. bucket별 요청 비율과 동시성을 확인한다.
2. backbone peak activation과 workspace를 측정한다.
3. 큰 bucket의 batch cap과 queue를 분리한다.
4. admission control이 raw shape가 아니라 resize 뒤 shape를 잘못 보고 있지 않은지 확인한다.
5. 직전 안정 release와 bucket manifest를 rollback한다.

## 20. 실무 실패 사례

### 사례 1: “가변 입력 지원”이 graph 폭증을 만들었다

API가 모든 양의 $H,W$를 허용했다. adaptive pool 뒤 head는 정상 동작했지만 compiler는 새 shape마다 graph를 만들었고 cache eviction과 cold latency가 반복됐다.

대응은 세 개 bucket으로 입력을 정규화하고, 알려진 graph만 serving하며, `unknown_shape_reject_total`을 모니터링하는 것이었다.

### 사례 2: padding mask가 너무 늦게 적용됐다

pool 직전에 mask를 적용해 평균의 0 padding은 제거했지만, 여러 convolution이 이미 padding 경계를 읽었다. 작은 이미지 bucket의 feature 분포가 큰 이미지와 달랐고 calibration이 나빠졌다.

대응은 aspect ratio bucket을 추가하고 학습에도 같은 padding 정책을 적용했다. masked pool 하나가 전체 backbone을 padding-invariant로 만들지는 않는다는 회귀 테스트도 추가했다.

### 사례 3: int8 GAP accumulator가 최대 bucket에서만 문제를 냈다

calibration은 작은 평균 해상도에서 수행했고, 최대 bucket의 큰 bin과 높은 activation 조합을 포함하지 않았다. target kernel의 accumulator·requantization 오차가 커져 작은 class logit 순서가 바뀌었다.

대응은 accumulator 상계 검사, 최대 bucket stress vector, float shadow parity, top-1 agreement gate를 추가하는 것이었다.

### 사례 4: export 성공 뒤 accelerator fallback이 숨어 있었다

ONNX 파일은 생성됐지만 `(2,3)` adaptive grid가 accelerator native kernel이 아니라 CPU subgraph로 실행됐다. 기능 test는 통과했지만 device copy와 fallback 때문에 p99가 SLA를 넘었다.

대응은 graph placement 검사, fallback node 수 0 gate, device trace, GAP 대안 ablation을 release 전에 수행하는 것이었다.

## 21. 모니터링과 SLO

### 21.1 입력·routing 지표

- 원본 `height`, `width`, aspect ratio, pixel 수 histogram
- 선택 bucket별 요청 수와 queue depth
- padding ratio p50·p95·p99
- oversize·unknown shape 거부 수
- bucket occupancy와 batch fill rate

### 21.2 runtime 지표

- bucket·batch·dtype별 p50·p95·p99 latency
- cold compile 시간, graph cache hit rate, fallback node 수
- device memory peak와 OOM
- layout transpose·host-device copy byte 수
- quantization saturation ratio와 accumulator guard failure

### 21.3 model 지표

- pool 전후 activation 평균·표준편차·최댓값
- bucket별 confidence, ECE, macro F1, class recall
- float shadow와 quantized serving의 logit 오차·top-1 agreement
- padding ratio slice별 error rate
- training baseline 대비 $H,W$ 분포 PSI

shape drift가 감지되면 무조건 모델을 재학습하기 전에 routing 정책, camera update, client resize 변경, traffic mix를 함께 조사한다.

## 22. rollout과 rollback

권장 순서는 다음과 같다.

1. offline golden: Python·C++·C#과 target runtime parity
2. offline replay: 실제 traffic의 모든 bucket과 tail shape
3. shadow: latency·logit·bucket routing 비교
4. canary: 작은 traffic에서 error·OOM·calibration 확인
5. 점진 확대: bucket별로 rollout 비율 증가
6. full rollout: 이전 bundle을 즉시 복구 가능하게 보존

rollback 단위는 model weight만이 아니다. 다음을 함께 되돌린다.

- preprocessing과 resize·padding 정책
- bucket 목록과 batch cap
- pool grid와 head shape
- quantization parameter
- runtime graph와 compiler cache
- class order와 calibration

## 23. 원문과 앞선 회차에서 바로잡고 확장한 내용

| 구분 | 원문 또는 앞선 설명 | 이번 회차의 보정·확장 |
| --- | --- | --- |
| 역사 | SPP-Net의 통찰이 곧 평균 adaptive pooling | SPP의 고정 길이 발상은 연결되지만 원 논문의 pyramid pooling과 현대 평균 API를 동일 연산으로 보지 않는다 |
| fixed input | 과거에는 항상 `224 x 224`만 가능 | convolution은 가변 공간 크기를 받을 수 있어도 batching·compute·head·compiler 제약은 시대와 구조마다 다르다 |
| resize 회피 | adaptive pool이면 crop·warp가 불필요 | 서비스는 compute와 batch를 통제하려고 여전히 resize·crop·pad를 사용한다 |
| partition | overlap도 완벽한 분할 | coverage는 보장하지만 겹치므로 엄밀한 partition은 아니다 |
| kernel·stride | 동적으로 하나의 kernel과 stride를 계산 | 정확한 계약은 출력 bin별 floor·ceil 반열린 경계다 |
| RoI | RoI Pooling·RoIAlign은 adaptive pool 응용 | box 좌표, quantization, sampling이 있는 별도 operator다 |
| shape 지원 | head가 가변 입력을 해결 | ragged batch, backbone memory, compiler specialization, SLA는 별도다 |
| 구현 검증 | forward·backward 골든 | bucket·mask·quantization·runtime placement·운영 telemetry까지 release gate로 확장한다 |

원문 파일명은 `02-02.AdaptiveAvgPooll2d.md`처럼 `l`이 하나 더 들어가지만 PyTorch API는 `nn.AdaptiveAvgPool2d`다. curriculum 메타데이터에는 실제 source 파일명을 그대로 기록했다.

## 24. 성능·정확도 의사결정 표

| 요구 | 우선 선택 | trade-off |
| --- | --- | --- |
| 가장 단순한 분류 head | GAP `(1,1)` | 공간 정보와 작은 신호 희석 가능 |
| 거친 위치 보존 | `(2,2)` 또는 `(2,3)` | head parameter와 runtime 지원 부담 증가 |
| padding이 많은 batch | aspect bucket + mask | bucket·mask 관리 복잡도 증가 |
| accelerator 호환 최우선 | native GAP 또는 고정 input pool | model 표현력·입력 유연성 감소 가능 |
| int8 latency 최우선 | native quantized kernel | scale·rounding·accumulator 검증 필요 |
| 동적 traffic | 제한된 bucket + admission control | 일부 요청 resize 또는 거부 |

## 25. 배포 체크리스트

### shape와 전처리

- [ ] layout, channel, color order, dtype을 manifest에 기록했다.
- [ ] 허용 bucket과 oversize 정책을 기록했다.
- [ ] resize·crop·pad anchor와 값을 학습·추론에서 맞췄다.
- [ ] pool grid와 flatten feature 수를 assert한다.
- [ ] padding mask의 생성·downsample·빈 bin 정책을 정했다.

### 수치와 크로스런타임

- [ ] `5 x 7 -> 3 x 2` golden이 Python·C++·C#에서 같다.
- [ ] NHWC·NCHW layout golden을 통과한다.
- [ ] dtype·device별 tensor와 task tolerance를 정했다.
- [ ] quantized scale·zero-point·rounding·accumulator를 검증했다.
- [ ] 최대 bucket stress vector를 포함했다.

### 성능과 운영

- [ ] bucket별 batch cap, p50·p95·p99, peak memory를 측정했다.
- [ ] cold compile과 steady-state latency를 분리했다.
- [ ] target graph의 fallback과 device placement를 검사했다.
- [ ] shape·padding·cache·quantization 지표를 dashboard에 연결했다.
- [ ] 이전 release bundle과 rollback 절차를 보존했다.

## 26. 연습문제

### 문제 1

입력 feature가 `(4,32,20,31)`이고 grid가 `(2,3)`이다. pool 출력과 flatten shape, 10-class `Linear` weight shape을 구하라.

### 문제 2

원본이 `300 x 500`이고 후보 bucket이 `256 x 256`, `320 x 512`, `512 x 512`다. crop을 허용하지 않을 때 선택 bucket과 occupancy, padding 비율을 구하라.

### 문제 3

유효한 `3 x 4` feature의 값이 1부터 12이고 이를 `5 x 7`에 0으로 pad했다. plain GAP와 mask-aware GAP를 구하라.

### 문제 4

int8 입력의 $q_{\min}=-128$, $q_{\max}=127$, $z_x=0$이고 GAP bin 면적이 1,048,576이다. 단순 절댓값 상계가 signed int32 최댓값보다 작은지 판단하라.

### 문제 5

adaptive pool이 있으므로 모든 $H,W$를 API에서 허용했더니 p99가 불안정해졌다. model accuracy 문제가 아닐 수 있는 원인 세 가지를 쓰라.

### 문제 6

ONNX export가 성공했다. production release 전에 arbitrary `(2,3)` grid에 대해 확인할 최소 네 가지를 쓰라.

## 27. 연습문제 해답

### 해답 1

pool 출력은 `(4,32,2,3)`이다. flatten feature 수는 다음과 같다.

$$
32 \times 2 \times 3=192
$$

flatten shape은 `(4,192)`, `Linear(192,10)`의 weight shape은 `(10,192)`다.

### 해답 2

`256 x 256`은 두 축 모두 담지 못하고, `320 x 512`가 가장 작은 유효 bucket이다. occupancy는 다음과 같다.

$$
\rho
=
\frac{300 \times 500}{320 \times 512}
\approx 0.9155
$$

padding 비율은 약 $1-0.9155=0.0845$, 즉 8.45%다.

### 해답 3

유효 값의 합은 78이고 유효 원소 수는 12다. mask-aware GAP는 $78/12=6.5$다. plain GAP는 전체 35칸으로 나누므로 다음과 같다.

$$
\frac{78}{35}
\approx 2.228571429
$$

### 해답 4

가장 큰 절댓값은 128로 잡을 수 있으므로 상계는 다음과 같다.

$$
1{,}048{,}576 \times 128
=134{,}217{,}728
$$

이는 signed int32 최댓값 2,147,483,647보다 작다. 따라서 이 단순 상계만 보면 overflow하지 않는다. 다만 실제 kernel의 partial sum, bias, requantization 구현은 별도로 확인한다.

### 해답 5

새 shape마다 graph compile이 발생할 수 있고, 같은 shape끼리 batch가 묶이지 않아 device utilization이 떨어질 수 있다. 또한 큰 shape의 backbone activation과 workspace가 커져 memory pressure·OOM·allocator 지연이 생길 수 있다. transpose·padding copy가 새로운 hot path가 되었을 가능성도 있다.

### 해답 6

graph operator와 fallback placement, 모든 허용 bucket의 target runtime 실행, Python과의 값·top-1 parity, cold·warm latency와 peak memory를 확인한다. quantized release라면 scale·rounding·saturation도 별도로 검증한다.

## 28. 핵심 요약

1. adaptive pooling은 pool 뒤 shape를 고정할 뿐 입력 traffic 전체를 자유롭게 만들지 않는다.
2. production은 제한된 해상도 bucket으로 batch·compiler·memory를 통제한다.
3. bin 경계는 64비트 정수 floor·ceil 규칙으로 언어 간 공유한다.
4. `5 x 7 -> 3 x 2` 비가분 golden은 layout·분모·정수 오류를 함께 잡는다.
5. mask-aware pooling은 평균의 padding을 제외하지만 앞선 convolution 오염까지 되돌리지는 못한다.
6. NCHW·NHWC, dtype, quantization scale·rounding·accumulator는 별도 ABI다.
7. GAP와 arbitrary adaptive grid는 ONNX·accelerator 지원성이 다를 수 있다.
8. pool 단독 latency보다 bucket별 end-to-end p95·p99와 peak memory가 release 기준이다.
9. shape drift, padding ratio, compile cache miss, fallback, saturation을 운영 중 관찰한다.
10. rollback은 weight뿐 아니라 preprocessing·bucket·pool grid·runtime graph를 함께 되돌린다.

## 29. 다음 학습 예고

다음 소스는 `02-03.SPP.md`다. 3회차 실무 엔지니어 관점에서는 여러 pyramid level의 출력 순서와 offset을 versioned ABI로 만들고, level별 병렬화·메모리·크로스런타임 동등성·ONNX graph·서비스 장애 격리를 다룬다. 오늘 만든 bucket, 64비트 경계, golden, runtime placement, shape drift 계약을 다중 level로 확장한다.
