<!-- curriculum: cycle=1; level=foundation; source_index=3/18; source=02-02.AdaptiveAvgPooll2d.md; part=1/1 -->

# AdaptiveAvgPool2d: 가변 해상도를 고정 격자로 접는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-07-19 |
| 회차·수준 | 1회차 · 기초 |
| 현재 소스 | 3/18 · `02-02.AdaptiveAvgPooll2d.md` |
| Part | 1/1 |
| 이전 소스 | `02-1.LazyLinear.md` |
| 다음 소스 | `02-03.SPP.md` |
| 예상 학습 시간 | 110~150분 |
| 실행 검증 환경 | Python 3.12.12 · PyTorch 2.13.0 · C++17 코드 교차 검토 · C# Mono 6.12 |

## 1. 오늘 해결할 문제

어제 배운 `LazyLinear`은 첫 입력에서 flatten 길이를 알아낼 수 있지만, 그 길이는 첫 forward 뒤에 고정된다. 입력 해상도가 달라지면 합성곱 feature map의 높이와 너비도 달라지고, flatten 길이도 달라진다.

```text
(N, C, 32, 32) -> flatten -> (N, 1024C)
(N, C, 48, 48) -> flatten -> (N, 2304C)
```

하나의 고정된 `Linear`는 두 feature 길이를 동시에 받을 수 없다. `AdaptiveAvgPool2d`는 입력 공간 크기와 무관하게 사용자가 지정한 출력 높이와 너비를 만든다.

```text
(N, C, H, W)
        |
        | AdaptiveAvgPool2d((3, 2))
        v
(N, C, 3, 2)
        |
        | flatten(start_dim=1)
        v
(N, 6C)
```

오늘의 핵심 문장은 다음과 같다.

> 적응형 평균 풀링은 입력을 단순히 같은 크기로 자르는 연산이 아니라, 출력 bin마다 floor와 ceil로 정한 입력 구간의 평균을 계산하는 shape 어댑터다.

## 2. 학습 목표

학습을 마치면 다음을 할 수 있어야 한다.

1. 일반 평균 풀링과 적응형 평균 풀링의 입력 계약 차이를 설명한다.
2. 출력 인덱스에서 입력 구간의 시작과 끝을 직접 계산한다.
3. 나누어떨어지지 않는 입력에서 bin이 겹치는 이유를 설명한다.
4. NCHW 텐서에 대한 2차원 출력을 손으로 검산한다.
5. `nn.AdaptiveAvgPool2d`를 가변 해상도 분류기에 안전하게 연결한다.
6. Python, C++, C#에서 동일한 floor/ceil 규칙을 구현한다.
7. layout, dtype, 수치 오차, 메모리, export 제약을 점검한다.

## 3. 선수 지식과 기호

### 3.1 텐서 shape

PyTorch 이미지 batch의 기본 layout은 NCHW다.

$$
X \in \mathbb{R}^{N \times C \times H_{\mathrm{in}} \times W_{\mathrm{in}}}
$$

- $N$: batch 크기
- $C$: 채널 수
- $H_{\mathrm{in}}$: 입력 feature map 높이
- $W_{\mathrm{in}}$: 입력 feature map 너비

목표 출력 크기를 $(H_{\mathrm{out}},W_{\mathrm{out}})$로 두면 출력은 다음 shape을 가진다.

$$
Y \in \mathbb{R}^{N \times C \times H_{\mathrm{out}} \times W_{\mathrm{out}}}
$$

평균은 공간 축에서만 계산한다. batch와 channel은 서로 섞이지 않는다.

### 3.2 floor, ceil, 반열린 구간

$\lfloor z \rfloor$는 $z$ 이하의 가장 큰 정수이고, $\lceil z \rceil$는 $z$ 이상의 가장 작은 정수다.

반열린 구간 $[a,b)$는 $a$는 포함하고 $b$는 제외한다. Python의 `range(a, b)` 및 slice `x[a:b]`와 같은 규칙이다. 구간에 들어 있는 정수 인덱스 수는 $b-a$다.

### 3.3 평균 풀링

일반 `AvgPool2d`는 kernel과 stride를 정하고 출력 크기를 얻는다. padding과 dilation을 생략한 1차원 식은 다음과 같다.

$$
O=\left\lfloor\frac{I-K}{S}\right\rfloor+1
$$

- $I$: 입력 길이
- $K$: kernel 크기
- $S$: stride
- $O$: 출력 길이

반대로 adaptive pooling은 $I$와 원하는 $O$를 알고, 각 출력 인덱스가 읽을 입력 구간을 정한다.

## 4. 직관: 출력 격자를 입력 위에 투영하기

입력 길이 $I$를 연속 구간 $[0,I]$라고 생각하자. 이를 출력 개수 $O$에 맞추어 이상적으로 나누면 경계는 다음 위치에 놓인다.

$$
0,\frac{I}{O},\frac{2I}{O},\ldots,\frac{OI}{O}=I
$$

하지만 텐서 인덱스는 정수다. 출력 bin $j$가 입력 원소를 빠뜨리지 않으려면 왼쪽 경계는 아래로 내리고 오른쪽 경계는 위로 올린다.

$$
s(j)=\left\lfloor\frac{jI}{O}\right\rfloor
$$

$$
e(j)=\left\lceil\frac{(j+1)I}{O}\right\rceil
$$

출력 $j$는 반열린 구간 $[s(j),e(j))$의 평균이다.

$$
y_j=\frac{1}{e(j)-s(j)}
\sum_{k=s(j)}^{e(j)-1}x_k
$$

이 방식은 resize의 bilinear interpolation처럼 새 좌표의 값을 주변 점에서 보간하지 않는다. 각 bin에 포함된 실제 입력 값들의 산술평균을 낸다.

## 5. 1차원 경계식 단계별 유도

### 5.1 이상적인 실수 경계

출력 bin $j$가 담당하는 이상적인 구간은 다음과 같다.

$$
\left[\frac{jI}{O},\frac{(j+1)I}{O}\right]
$$

이 구간과 닿는 정수 셀을 포함하기 위해 시작점에는 floor, 끝점에는 ceil을 쓴다. 예를 들어 $I=7$, $O=3$이면 이상적인 경계는 $0$, $7/3$, $14/3$, $7$이다.

### 5.2 $I=7$, $O=3$ 계산

| 출력 $j$ | $s(j)$ | $e(j)$ | 입력 인덱스 | bin 크기 |
| ---: | ---: | ---: | --- | ---: |
| 0 | 0 | 3 | `0, 1, 2` | 3 |
| 1 | 2 | 5 | `2, 3, 4` | 3 |
| 2 | 4 | 7 | `4, 5, 6` | 3 |

입력 인덱스 2와 4는 각각 이웃한 두 bin에 들어간다. 이는 구현 실수가 아니라 floor/ceil 경계 규칙의 결과다.

### 5.3 원문의 중요한 수정: 겹치는 bin은 partition이 아니다

원문은 이 예를 “정확한 분배”와 “partitioning”으로 설명한다. 그러나 수학에서 partition은 보통 부분집합들이 서로 겹치지 않아야 한다. 위 구간은 인덱스 2와 4가 겹치므로 엄밀한 의미의 partition이 아니다.

정확한 표현은 다음과 같다.

> 모든 입력 위치를 덮는 coverage를 유지하면서, 나누어떨어지지 않을 때 이웃 bin의 경계 원소가 겹칠 수 있는 구간 모음이다.

이 차이는 역전파에서 중요하다. 겹친 입력은 둘 이상의 출력에 영향을 주므로 gradient도 여러 경로에서 더해진다.

### 5.4 입력을 빠뜨리지 않는 이유

첫 시작점과 마지막 끝점은 다음과 같다.

$$
s(0)=\left\lfloor 0\right\rfloor=0
$$

$$
e(O-1)=\left\lceil\frac{OI}{O}\right\rceil=I
$$

또한 이웃한 bin 사이에는 다음 관계가 성립한다.

$$
s(j+1)=\left\lfloor\frac{(j+1)I}{O}\right\rfloor
\le
\left\lceil\frac{(j+1)I}{O}\right\rceil=e(j)
$$

따라서 $s(j+1)>e(j)$인 빈틈은 생기지 않는다. 실수 경계가 정수면 두 값이 같고, 정수가 아니면 floor가 ceil보다 1 작아 경계 셀이 겹친다.

### 5.5 bin 크기

bin $j$가 읽는 원소 수는 다음과 같다.

$$
K_j=e(j)-s(j)
$$

$I$가 $O$로 나누어떨어지면 모든 $K_j=I/O$이고 구간이 겹치지 않는다. 나누어떨어지지 않으면 bin 크기와 overlap이 위치에 따라 달라질 수 있다. 따라서 adaptive pooling 전체를 언제나 하나의 고정 `kernel_size`와 `stride`를 가진 `AvgPool2d`로 바꿀 수 있는 것은 아니다.

## 6. 2차원으로 확장하기

출력 위치 $(p,q)$에 대해 높이와 너비 경계를 독립적으로 계산한다.

$$
h_s(p)=\left\lfloor\frac{pH_{\mathrm{in}}}{H_{\mathrm{out}}}\right\rfloor,
\qquad
h_e(p)=\left\lceil\frac{(p+1)H_{\mathrm{in}}}{H_{\mathrm{out}}}\right\rceil
$$

$$
w_s(q)=\left\lfloor\frac{qW_{\mathrm{in}}}{W_{\mathrm{out}}}\right\rfloor,
\qquad
w_e(q)=\left\lceil\frac{(q+1)W_{\mathrm{in}}}{W_{\mathrm{out}}}\right\rceil
$$

출력 원소는 직사각형 영역의 평균이다.

$$
Y_{n,c,p,q}
=
\frac{
\displaystyle
\sum_{h=h_s(p)}^{h_e(p)-1}
\sum_{w=w_s(q)}^{w_e(q)-1}
X_{n,c,h,w}
}{
\bigl(h_e(p)-h_s(p)\bigr)
\bigl(w_e(q)-w_s(q)\bigr)
}
$$

분모는 해당 bin의 실제 원소 수다. 가장자리와 내부 bin의 면적이 다를 수 있으므로 모든 위치에 같은 상수로 나누면 틀린다.

## 7. 손으로 검산하는 $3 \times 4 \rightarrow 2 \times 2$ 예제

입력 feature map 하나를 다음과 같이 두자.

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}
$$

높이는 $3 \rightarrow 2$이므로 구간이 `[0:2]`, `[1:3]`이다. 너비는 $4 \rightarrow 2$로 나누어떨어지므로 `[0:2]`, `[2:4]`이다.

| 출력 위치 | 입력 slice | 값 | 평균 |
| --- | --- | --- | ---: |
| `(0, 0)` | `X[0:2, 0:2]` | `1, 2, 5, 6` | 3.5 |
| `(0, 1)` | `X[0:2, 2:4]` | `3, 4, 7, 8` | 5.5 |
| `(1, 0)` | `X[1:3, 0:2]` | `5, 6, 9, 10` | 7.5 |
| `(1, 1)` | `X[1:3, 2:4]` | `7, 8, 11, 12` | 9.5 |

따라서 출력은 다음과 같다.

$$
Y=
\begin{bmatrix}
3.5 & 5.5 \\
7.5 & 9.5
\end{bmatrix}
$$

가운데 입력 행은 위쪽 bin과 아래쪽 bin에 모두 참여한다.

## 8. NumPy 수작업 구현과 PyTorch 대조

다음 코드는 독립 실행 가능하다. 정수 나눗셈만으로 floor와 ceil 경계를 계산하는 NumPy 참조 구현을 PyTorch 결과와 비교한다.

```python
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def start_index(out_index: int, out_size: int, in_size: int) -> int:
    return (out_index * in_size) // out_size


def end_index(out_index: int, out_size: int, in_size: int) -> int:
    # 양의 정수 a, b에 대해 ceil(a / b) = (a + b - 1) // b
    numerator = (out_index + 1) * in_size
    return (numerator + out_size - 1) // out_size


def adaptive_avg_pool2d_numpy(
    x: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError("x must be NCHW")

    n, c, in_h, in_w = x.shape
    out_h, out_w = output_size
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output sizes must be positive")

    # 평균 결과는 정수가 아닐 수 있으므로 float64 참조값을 만든다.
    y = np.empty((n, c, out_h, out_w), dtype=np.float64)
    x64 = x.astype(np.float64, copy=False)

    for oh in range(out_h):
        hs = start_index(oh, out_h, in_h)
        he = end_index(oh, out_h, in_h)
        for ow in range(out_w):
            ws = start_index(ow, out_w, in_w)
            we = end_index(ow, out_w, in_w)
            y[:, :, oh, ow] = x64[:, :, hs:he, ws:we].mean(
                axis=(-2, -1)
            )
    return y


x_np = np.arange(1, 13, dtype=np.float32).reshape(1, 1, 3, 4)
expected = np.array([[[[3.5, 5.5], [7.5, 9.5]]]])
manual = adaptive_avg_pool2d_numpy(x_np, (2, 2))

x_torch = torch.from_numpy(x_np)
actual = nn.AdaptiveAvgPool2d((2, 2))(x_torch).numpy()

np.testing.assert_allclose(manual, expected, rtol=0.0, atol=0.0)
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
np.testing.assert_allclose(actual, manual, rtol=0.0, atol=0.0)

print(manual.reshape(2, 2))
print("NumPy and PyTorch agree")
```

예상 출력은 다음과 같다.

```text
[[3.5 5.5]
 [7.5 9.5]]
NumPy and PyTorch agree
```

`math.ceil(float_value)` 대신 정수식 `(a + b - 1) // b`를 쓰면 큰 정수에서도 부동소수점 반올림에 의존하지 않는다.

## 9. PyTorch 가변 해상도 분류기

다음 코드는 독립 실행 가능하다. 서로 다른 해상도 두 개가 같은 분류 head로 들어가는지 확인한다.

```python
from __future__ import annotations

import torch
from torch import nn


class VariableResolutionClassifier(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 3))
        self.classifier = nn.Linear(16 * 2 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


torch.manual_seed(19)
model = VariableResolutionClassifier(num_classes=5).eval()

with torch.no_grad():
    y_small = model(torch.randn(2, 3, 64, 80))
    y_large = model(torch.randn(2, 3, 96, 112))

assert y_small.shape == (2, 5)
assert y_large.shape == (2, 5)
assert model.classifier.in_features == 96

print("small logits:", tuple(y_small.shape))
print("large logits:", tuple(y_large.shape))
print("linear in_features:", model.classifier.in_features)
```

예상 핵심 출력은 다음과 같다.

```text
small logits: (2, 5)
large logits: (2, 5)
linear in_features: 96
```

### 9.1 shape 추적

| 단계 | 작은 입력 | 큰 입력 | 불변 조건 |
| --- | --- | --- | --- |
| 입력 | `(2, 3, 64, 80)` | `(2, 3, 96, 112)` | channel 3 |
| 첫 `Conv2d` | `(2, 8, 64, 80)` | `(2, 8, 96, 112)` | 공간 크기 유지 |
| `MaxPool2d(2)` | `(2, 8, 32, 40)` | `(2, 8, 48, 56)` | 공간 크기 절반 |
| 둘째 `Conv2d` | `(2, 16, 32, 40)` | `(2, 16, 48, 56)` | channel 16 |
| adaptive pool | `(2, 16, 2, 3)` | `(2, 16, 2, 3)` | 공간 shape 고정 |
| flatten | `(2, 96)` | `(2, 96)` | $16 \times 2 \times 3$ |
| `Linear(96, 5)` | `(2, 5)` | `(2, 5)` | 클래스 logits |

### 9.2 `output_size`의 `None`

PyTorch는 한 축의 크기를 유지하는 `None`도 허용한다.

```python
import torch
from torch import nn

x = torch.randn(2, 4, 7, 11)
keep_width = nn.AdaptiveAvgPool2d((3, None))
y = keep_width(x)

assert y.shape == (2, 4, 3, 11)
print(tuple(y.shape))
```

이 코드는 독립 실행 가능하다. 다만 고정 길이 `Linear` 앞에서 너비를 유지하면 입력 너비에 따라 flatten 길이가 변할 수 있으므로 가변 해상도 분류 문제를 완전히 해결하지 못한다.

## 10. Global Average Pooling은 특수한 경우다

`AdaptiveAvgPool2d((1, 1))`은 각 채널의 모든 공간 위치를 평균내는 Global Average Pooling, 즉 GAP이다.

$$
Y_{n,c,0,0}
=
\frac{1}{H_{\mathrm{in}}W_{\mathrm{in}}}
\sum_{h=0}^{H_{\mathrm{in}}-1}
\sum_{w=0}^{W_{\mathrm{in}}-1}
X_{n,c,h,w}
$$

동일한 결과를 다음 두 방식으로 계산할 수 있다.

```python
import torch
from torch import nn

torch.manual_seed(3)
x = torch.randn(2, 5, 7, 11)

pooled = nn.AdaptiveAvgPool2d((1, 1))(x)
reduced = x.mean(dim=(-2, -1), keepdim=True)

torch.testing.assert_close(pooled, reduced)
assert pooled.shape == (2, 5, 1, 1)
print("GAP equivalence passed")
```

이 코드는 독립 실행 가능하다. GAP 뒤에 flatten하면 feature 길이는 $C$뿐이므로 해상도와 무관하다. 반면 $(2,3)$처럼 여러 bin을 남기면 대략적인 공간 배치를 보존하는 대신 head의 입력 feature가 $6C$로 커진다.

| 출력 격자 | flatten 길이 | 공간 정보 | head 파라미터 |
| --- | ---: | --- | --- |
| `(1, 1)` | $C$ | 가장 많이 압축 | 가장 작음 |
| `(2, 2)` | $4C$ | 사분면 수준 | 증가 |
| `(3, 3)` | $9C$ | 더 세밀함 | 더 크게 증가 |

## 11. 역전파: 겹친 입력의 gradient

출력 $y_j$가 $K_j$개 입력의 평균이면 해당 bin 내부의 입력에 대한 편미분은 다음과 같다.

$$
\frac{\partial y_j}{\partial x_k}
=
\begin{cases}
\dfrac{1}{K_j}, & s(j)\le k<e(j) \\
0, & \text{otherwise}
\end{cases}
$$

loss를 $L$이라 할 때 chain rule로 입력 gradient를 구하면 다음과 같다.

$$
\frac{\partial L}{\partial x_k}
=
\sum_{j:\,s(j)\le k<e(j)}
\frac{1}{K_j}
\frac{\partial L}{\partial y_j}
$$

입력 $k$가 여러 bin에 겹치면 합에 여러 항이 들어간다. $I=7$, $O=3$이고 모든 출력 gradient가 1이라면, 인덱스 2와 4는 각각 $1/3+1/3=2/3$을 받지만 나머지 인덱스는 $1/3$을 받는다.

다음 코드는 독립 실행 가능하다.

```python
import torch
from torch import nn

x = torch.arange(7, dtype=torch.float64, requires_grad=True)
y = nn.AdaptiveAvgPool1d(3)(x.reshape(1, 1, 7))
y.sum().backward()

expected = torch.tensor(
    [1 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3, 1 / 3],
    dtype=torch.float64,
)
torch.testing.assert_close(x.grad, expected)
print(x.grad)
```

이 성질 때문에 adaptive pooling을 단순한 “정보를 중복 없이 묶는 평균”으로 이해하면 안 된다.

## 12. C++17 참조 구현

다음 코드는 외부 라이브러리 없이 실행 가능하다. 단일 $H \times W$ plane을 row-major로 저장하고 $2 \times 2$로 pooling한다. 실제 libtorch 모델에서는 `torch::nn::AdaptiveAvgPool2d`를 사용하되, 이 코드는 경계 알고리즘을 독립 검증하기 위한 것이다.

```cpp
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

std::size_t start_index(
    std::size_t out_index,
    std::size_t out_size,
    std::size_t in_size) {
    return (out_index * in_size) / out_size;
}

std::size_t end_index(
    std::size_t out_index,
    std::size_t out_size,
    std::size_t in_size) {
    const std::size_t numerator = (out_index + 1) * in_size;
    return (numerator + out_size - 1) / out_size;
}

std::vector<double> adaptive_avg_pool2d(
    const std::vector<double>& input,
    std::size_t in_h,
    std::size_t in_w,
    std::size_t out_h,
    std::size_t out_w) {
    if (input.size() != in_h * in_w) {
        throw std::invalid_argument("input size does not match H*W");
    }
    if (out_h == 0 || out_w == 0) {
        throw std::invalid_argument("output sizes must be positive");
    }

    std::vector<double> output(out_h * out_w, 0.0);
    for (std::size_t oh = 0; oh < out_h; ++oh) {
        const std::size_t hs = start_index(oh, out_h, in_h);
        const std::size_t he = end_index(oh, out_h, in_h);
        for (std::size_t ow = 0; ow < out_w; ++ow) {
            const std::size_t ws = start_index(ow, out_w, in_w);
            const std::size_t we = end_index(ow, out_w, in_w);

            double sum = 0.0;
            std::size_t count = 0;
            for (std::size_t h = hs; h < he; ++h) {
                for (std::size_t w = ws; w < we; ++w) {
                    sum += input[h * in_w + w];
                    ++count;
                }
            }
            output[oh * out_w + ow] = sum / static_cast<double>(count);
        }
    }
    return output;
}

int main() {
    const std::vector<double> input{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    const auto output = adaptive_avg_pool2d(input, 3, 4, 2, 2);
    const std::vector<double> expected{3.5, 5.5, 7.5, 9.5};

    assert(output.size() == expected.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
        assert(std::abs(output[i] - expected[i]) < 1e-12);
    }

    std::cout << "C++ adaptive pooling passed\n";
    return 0;
}
```

컴파일 명령은 다음과 같다.

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic adaptive_pool.cpp -o adaptive_pool
./adaptive_pool
```

이 실행 환경에서는 C++ 코드 컴파일을 시도했으나 Apple `clang++`은 표준 헤더 `cassert`를 찾지 못했고 Homebrew GCC는 macOS SDK의 `stdio.h`와 충돌했다. 따라서 C++ 예제의 로컬 바이너리 실행은 **미검증**이다. 대신 같은 정수 경계식과 기대값을 사용하는 Python·C# 구현을 실행했고 모두 통과했으며, C++ 코드는 `-Wall -Wextra -pedantic` 기준으로 수동 검토했다. 정상적인 C++17 toolchain이 있는 환경에서 위 명령으로 다시 컴파일해야 한다.

## 13. C# .NET 참조 구현

다음 코드는 독립 실행 가능하다. 2차원 rectangular array는 개념 확인에 읽기 쉽지만, 고성능 추론 코드에서는 연속 `float[]`와 명시적 stride가 더 예측 가능한 경우가 많다.

```csharp
using System;

public static class AdaptivePoolDemo
{
    private static int StartIndex(int outputIndex, int outputSize, int inputSize)
    {
        return outputIndex * inputSize / outputSize;
    }

    private static int EndIndex(int outputIndex, int outputSize, int inputSize)
    {
        int numerator = (outputIndex + 1) * inputSize;
        return (numerator + outputSize - 1) / outputSize;
    }

    public static double[,] AdaptiveAvgPool2D(
        double[,] input,
        int outputHeight,
        int outputWidth)
    {
        if (outputHeight <= 0 || outputWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(outputHeight),
                "Output sizes must be positive.");
        }

        int inputHeight = input.GetLength(0);
        int inputWidth = input.GetLength(1);
        var output = new double[outputHeight, outputWidth];

        for (int oh = 0; oh < outputHeight; oh++)
        {
            int hs = StartIndex(oh, outputHeight, inputHeight);
            int he = EndIndex(oh, outputHeight, inputHeight);
            for (int ow = 0; ow < outputWidth; ow++)
            {
                int ws = StartIndex(ow, outputWidth, inputWidth);
                int we = EndIndex(ow, outputWidth, inputWidth);
                double sum = 0.0;
                int count = 0;

                for (int h = hs; h < he; h++)
                {
                    for (int w = ws; w < we; w++)
                    {
                        sum += input[h, w];
                        count++;
                    }
                }
                output[oh, ow] = sum / count;
            }
        }
        return output;
    }

    public static void Main()
    {
        double[,] input =
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 },
        };
        double[,] expected =
        {
            { 3.5, 5.5 },
            { 7.5, 9.5 },
        };

        double[,] actual = AdaptiveAvgPool2D(input, 2, 2);
        for (int h = 0; h < 2; h++)
        {
            for (int w = 0; w < 2; w++)
            {
                if (Math.Abs(actual[h, w] - expected[h, w]) > 1e-12)
                {
                    throw new Exception("C# result mismatch");
                }
            }
        }
        Console.WriteLine("C# adaptive pooling passed");
    }
}
```

.NET SDK로 실행할 때는 console project의 `Program.cs`에 넣고 다음 명령을 사용한다.

```bash
dotnet run --configuration Release
```

현재 환경에서는 동일 소스를 Mono C# compiler 3.9로 컴파일하고 Mono 6.12에서 실행해 `C# adaptive pooling passed`를 확인했다. `dotnet run` 명령은 target 배포 환경에서 한 번 더 확인한다.

## 14. 프레임워크 간 shape·layout·dtype 대응

동일한 숫자를 얻으려면 연산 이름만 맞추지 말고 layout, 축, dtype을 함께 맞춰야 한다.

| 환경 | 대표 입력 layout | shape 예 | 평균 축 | 주의점 |
| --- | --- | --- | --- | --- |
| PyTorch | NCHW | `(N, C, H, W)` | 마지막 두 축 | `AdaptiveAvgPool2d` 기본 계약 |
| NumPy 참조 | 코드에서 선택 | `(N, C, H, W)` | `axis=(-2, -1)` | 정수 입력은 float accumulator 권장 |
| C++ row-major 예제 | HW plane | `(H, W)` | 공간 전체 또는 bin | N, C loop를 바깥에 추가 |
| C# 예제 | HW rectangular array | `(H, W)` | 공간 bin | 배포 runtime의 실제 tensor layout 확인 |
| ONNX 계열 runtime | 보통 NCHW | `(N, C, H, W)` | operator에 따름 | 임의 output grid 지원 여부 확인 |

NHWC 텐서 `(N,H,W,C)`를 NCHW용 구현에 그대로 넘기면 마지막 두 축이 $W,C$로 해석될 수 있다. shape가 우연히 실행 가능해도 channel을 공간 축처럼 평균내는 조용한 오류가 된다.

### 14.1 dtype 계약

| dtype | 장점 | 위험 | 검증 기준 예 |
| --- | --- | --- | --- |
| `float64` | 참조 계산 오차가 작음 | 느리고 메모리 큼 | 엄격한 수작업 검산 |
| `float32` | 일반 학습·추론 기본 | 큰 bin 누적 반올림 | `rtol`, `atol` 허용 |
| `float16` | GPU 처리량·메모리 이점 | 누적 정밀도와 overflow | runtime accumulator 확인 |
| `bfloat16` | 넓은 지수 범위 | 유효숫자 적음 | 분포 기반 오차 평가 |
| 정수 | 저장 효율 | 평균에 scale과 rounding 필요 | quantization 계약 확인 |

평균은 정수 입력을 정수 출력으로 자연스럽게 닫지 않는다. 예를 들어 $(1+2)/2=1.5$다. 참조 구현에서는 입력을 `float64`로 승격했고, 실제 양자화 배포에서는 zero-point, scale, rounding mode를 runtime과 맞춰야 한다.

## 15. 테스트 전략

### 15.1 값·shape·gradient를 분리해 검사하기

shape만 맞는 잘못된 평균 구현도 많다. 다음 세 층의 테스트가 필요하다.

1. shape test: 임의 입력 크기가 목표 출력 크기가 되는가?
2. value test: 작은 결정론적 입력이 손계산과 같은가?
3. gradient test: overlap 위치의 gradient가 합산되는가?

다음 코드는 독립 실행 가능한 최소 테스트 모음이다.

```python
import torch
from torch import nn


def test_shapes() -> None:
    pool = nn.AdaptiveAvgPool2d((3, 2))
    for height, width in [(7, 11), (12, 8), (3, 2)]:
        x = torch.randn(2, 4, height, width)
        assert pool(x).shape == (2, 4, 3, 2)


def test_known_values() -> None:
    x = torch.arange(1, 13, dtype=torch.float64).reshape(1, 1, 3, 4)
    expected = torch.tensor(
        [[[[3.5, 5.5], [7.5, 9.5]]]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        nn.AdaptiveAvgPool2d((2, 2))(x),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_gap_equivalence() -> None:
    torch.manual_seed(11)
    x = torch.randn(2, 3, 5, 9, dtype=torch.float64)
    expected = x.mean(dim=(-2, -1), keepdim=True)
    actual = nn.AdaptiveAvgPool2d(1)(x)
    torch.testing.assert_close(actual, expected)


test_shapes()
test_known_values()
test_gap_equivalence()
print("three tests passed")
```

### 15.2 property 기반 불변조건

여러 무작위 shape에 대해 다음을 검사하면 경계 버그를 빨리 찾을 수 있다.

- 출력 shape은 언제나 요청한 크기다.
- 상수 입력의 출력은 같은 상수다.
- 입력에 상수 $a$를 더하면 출력에도 $a$가 더해진다.
- 입력에 양수 $b$를 곱하면 출력에도 $b$가 곱해진다.
- GAP는 명시적 공간 평균과 같다.
- 모든 bin 경계는 $0 \le s < e \le I$를 만족한다.

단, 출력 크기가 입력 크기보다 큰 upsampling 형태도 API에서 가능하므로 “항상 정보를 압축한다”를 불변조건으로 두면 안 된다.

## 16. 디버깅 가이드

### 16.1 자주 만나는 증상

| 증상 | 가능한 원인 | 빠른 확인 | 수정 방향 |
| --- | --- | --- | --- |
| `Linear` shape 오류 | pool 뒤 flatten 길이 오계산 | pool 출력과 `in_features` 출력 | $C H_{\mathrm{out}}W_{\mathrm{out}}$ 사용 |
| accuracy 급락 | GAP가 작은 객체 위치 정보를 과도하게 제거 | `(1,1)`과 `(2,2)` 비교 | grid 크기 ablation |
| CPU/GPU 값 차이 | dtype·누적 순서 차이 | `float64` 참조와 오차 출력 | 합리적 tolerance 설정 |
| channel이 사라짐 | NHWC/NCHW 혼동 | 각 축의 의미 기록 | transpose 또는 permute |
| export 실패 | runtime이 임의 adaptive grid 미지원 | 최소 모델 단독 export | 지원 operator로 재설계 |
| batch 묶기 실패 | 원본 이미지 크기가 달라 stack 불가 | pooling 전 batch shape 확인 | resize, pad, bucketing, batch 1 |

### 16.2 “가변 해상도 지원”의 범위

모델 내부에 adaptive pooling이 있어도 일반적인 dense batch tensor는 모든 sample이 같은 $H,W$를 가져야 한다. 크기가 다른 원본 이미지를 한 batch로 만들려면 다음 중 하나가 필요하다.

- 같은 크기로 resize 또는 crop
- batch 안의 최대 크기로 padding하고 padding 영향 관리
- 비슷한 해상도끼리 bucketing
- sample별 실행 또는 batch size 1

adaptive pooling은 **모델이 서로 다른 실행에서 다른 해상도를 받을 수 있게 하는 것**과 **한 tensor 안에 ragged image를 담는 것**을 구분해야 한다.

### 16.3 hook으로 shape 추적하기

다음 코드는 설명용이다. 큰 모델에서 특정 모듈의 입출력 shape을 확인하는 패턴이다.

```python
def shape_hook(module, inputs, output):
    input_shape = tuple(inputs[0].shape)
    output_shape = tuple(output.shape)
    print(type(module).__name__, input_shape, "->", output_shape)


handle = model.pool.register_forward_hook(shape_hook)
_ = model(torch.randn(1, 3, 73, 91))
handle.remove()
```

hook을 계속 등록해 두면 로그 증가와 객체 참조 문제가 생길 수 있으므로 디버깅 뒤에는 `remove()`한다.

## 17. 성능과 메모리

### 17.1 연산량의 직관

각 출력이 bin의 모든 입력을 읽으므로 단순 구현의 연산량은 bin 면적의 합에 비례한다. overlap이 없으면 채널별로 입력 원소를 거의 한 번씩 읽는다. overlap이 있으면 경계 원소를 여러 번 읽을 수 있다.

개념적인 비용은 다음과 같이 쓸 수 있다.

$$
\mathcal{O}\left(
NC
\sum_{p=0}^{H_{\mathrm{out}}-1}
\sum_{q=0}^{W_{\mathrm{out}}-1}
K^{(h)}_p K^{(w)}_q
\right)
$$

여기서 $K^{(h)}_p=h_e(p)-h_s(p)$이고 $K^{(w)}_q=w_e(q)-w_s(q)$다.

### 17.2 head 파라미터 절감

pool 전 feature map이 $(N,256,14,14)$이고 클래스가 1000개라고 하자. 바로 flatten해 선형층에 넣으면 weight 수는 다음과 같다.

$$
256 \times 14 \times 14 \times 1000
=50{,}176{,}000
$$

GAP 뒤에는 다음과 같이 줄어든다.

$$
256 \times 1000=256{,}000
$$

bias를 제외하면 약 196배 차이다. 다만 파라미터가 적다는 사실이 항상 정확도가 높다는 뜻은 아니다. 위치 정보가 중요한 task에서는 더 큰 grid나 다른 head가 필요할 수 있다.

### 17.3 contiguous와 layout

PyTorch kernel은 비연속 view도 처리할 수 있지만, layout 변환이나 예상치 못한 copy가 전체 latency에 영향을 줄 수 있다. 배포 전에는 다음을 실제 target에서 측정한다.

- NCHW와 channels-last의 end-to-end latency
- pool 단독이 아니라 전후 convolution을 포함한 latency
- batch와 해상도 bucket별 p50, p95
- peak activation memory
- dtype별 오차와 처리량

## 18. 수치 안정성

평균은 합을 원소 수로 나눈다. 큰 bin에서 동일 부호의 큰 값이 많이 누적되거나 크기가 매우 다른 값이 섞이면 낮은 정밀도에서 반올림 오차가 커질 수 있다.

### 18.1 비교 허용오차

손계산 결과가 정확히 이진수로 표현되는 3.5 같은 값은 exact comparison이 가능하지만, 일반 난수와 큰 bin에서는 다음처럼 허용오차를 둔다.

```python
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

허용오차를 무조건 크게 잡아 테스트를 통과시키지 말고, 다음을 기록한다.

1. reference dtype
2. target dtype
3. 최대 절대 오차
4. 최대 상대 오차
5. 오차가 task metric에 미치는 영향

### 18.2 mixed precision

GPU kernel이 저정밀 입력을 더 높은 정밀도의 accumulator로 합산하는지는 device와 backend 구현에 따라 달라질 수 있다. 문서의 수학식만으로 실제 accumulator dtype을 단정하지 말고, 사용하는 PyTorch·ONNX runtime·가속기 조합에서 검증한다.

## 19. 실무 실패 사례

### 사례 A: 작은 결함이 GAP에서 사라짐

제조 이미지에서 결함이 전체 feature map의 극히 작은 부분만 차지했다. `(1,1)` GAP가 결함 신호를 정상 배경과 평균내어 분류 confidence가 낮아졌다.

대응은 단순히 해상도를 키우는 것이 아니었다.

- `(1,1)`, `(2,2)`, `(4,4)` grid ablation
- max와 average feature 결합 실험
- crop 또는 patch sampling 개선
- localization 보조 loss 검토

### 사례 B: padding이 평균을 오염시킴

가변 크기 이미지를 0으로 padding해 batch를 만든 뒤 adaptive average pooling을 적용했다. 작은 이미지는 padding 비율이 커서 평균 feature 크기가 체계적으로 작아졌다.

대응 방법은 다음과 같다.

- 유사 해상도끼리 bucket 구성
- padding mask를 고려한 masked average 구현
- crop/resize 정책 재검토
- padding 비율을 모니터링 지표로 기록

`AdaptiveAvgPool2d` 자체는 padding 위치를 모르므로 자동으로 제외하지 않는다.

### 사례 C: 학습과 배포의 operator 불일치

학습에서는 $(3,3)$ adaptive pooling을 사용했지만 target runtime은 GAP만 효율적으로 지원했다. export가 실패하거나 여러 slice와 reduce로 분해되어 latency가 급증했다.

대응은 모델 완성 뒤가 아니라 설계 초기에 target runtime의 operator coverage를 확인하는 것이다. 필요하면 GAP 기반 head, 고정 입력 shape, 지원되는 pooling 조합을 정확도와 함께 비교한다.

### 사례 D: 큰 이미지가 들어오니 OOM

마지막에 adaptive pooling이 있으니 어떤 해상도도 안전하다고 오해했다. 그러나 pool 이전의 convolution activation은 입력 면적에 비례해 커진다.

대응은 입력 최대 크기 제한, 해상도 bucket, tiling, activation memory 추정, admission control이다. adaptive pooling은 head shape을 고정할 뿐 앞단 계산량과 메모리를 고정하지 않는다.

## 20. 배포 관점

### 20.1 모델 계약

배포 명세에는 “가변 입력 지원” 한 줄 대신 다음을 기록한다.

- 허용 batch 범위
- 허용 $H,W$ 최소·최대
- channel 수와 color order
- NCHW/NHWC layout
- 입력 dtype과 정규화
- adaptive pool output grid
- target runtime과 operator version
- 해상도별 latency·memory 한도

### 20.2 ONNX와 동적 shape

GAP는 보통 global average pooling operator로 자연스럽게 표현할 수 있다. 반면 임의의 $(H_{\mathrm{out}},W_{\mathrm{out}})$ adaptive pooling은 입력과 출력 크기의 관계에 따라 exporter나 backend 지원이 제한될 수 있다.

따라서 다음 순서로 검증한다.

1. pool만 포함한 최소 모델 export
2. 두 개 이상의 동적 입력 해상도로 runtime 실행
3. PyTorch와 값 비교
4. 실제 backend에서 latency 측정
5. 최종 모델 전체에서 재검증

이 문서는 특정 버전의 exporter 지원 범위를 고정된 사실로 가정하지 않는다. 배포 환경의 정확한 버전으로 테스트하는 것이 계약이다.

### 20.3 운영 모니터링

adaptive pooling을 사용해도 입력 분포 변화는 남는다. 다음 지표가 유용하다.

- 입력 높이·너비와 aspect ratio 분포
- padding 비율
- 해상도 bucket별 latency와 error rate
- pool 전 activation 평균·분산·최댓값
- confidence와 calibration 변화
- 최대 허용 해상도 거부 건수

## 21. 원문에서 바로잡거나 구분할 내용

원문 `02-02.AdaptiveAvgPooll2d.md`의 교육적 직관은 유용하지만 다음은 구분해야 한다.

| 원문 취지 | 보정된 설명 |
| --- | --- |
| SPP-Net 통찰이 곧 `AdaptiveAvgPool2d` 구현이다 | SPP의 고정 길이 아이디어와 adaptive pooling은 연결되지만, 현대 API 하나를 논문 아이디어와 동일시하면 역사와 연산 차이를 흐릴 수 있다 |
| 겹친 구간도 완벽한 partition이다 | coverage는 완전하지만 겹치므로 엄밀한 partition은 아니다 |
| adaptive pooling은 kernel과 stride를 동적으로 계산한다 | 입문 직관으로는 가능하지만 정확한 계약은 출력 bin별 시작·끝 인덱스다. 단일 kernel/stride로 표현되지 않는 경우가 있다 |
| 가변 해상도 문제를 해결한다 | head shape은 해결하지만 batch raggedness, 앞단 메모리, runtime 지원은 별도 문제다 |
| RoI pooling과 같은 응용이다 | 둘 다 고정 shape을 만들지만 RoI Pooling/RoIAlign은 box 좌표와 sampling 규칙을 포함하는 별도 연산이다 |

또한 소스 파일명은 `AdaptiveAvgPooll2d`처럼 `l`이 하나 더 들어가지만 PyTorch 클래스의 정확한 이름은 `nn.AdaptiveAvgPool2d`다. curriculum 메타데이터의 `source`는 실제 파일명을 그대로 기록했다.

## 22. 선택 가이드

| 요구사항 | 우선 검토 | 이유 |
| --- | --- | --- |
| 분류에서 해상도 독립적인 작은 head | GAP `(1,1)` | feature 길이가 $C$로 고정 |
| 거친 위치 정보 보존 | `(2,2)` 또는 더 큰 adaptive grid | 여러 공간 bin 유지 |
| 작은 객체 존재 여부 강조 | max pooling 또는 혼합 | 평균에서 신호가 희석될 수 있음 |
| 정확한 위치 복원 | pooling만으로 해결하지 않음 | segmentation·detection 구조 필요 |
| padding이 많은 batch | masked average 또는 bucketing | 0 padding 평균 오염 방지 |
| 제한된 edge runtime | 지원 operator 우선 설계 | export·latency 위험 감소 |

## 23. 구현·리뷰 체크리스트

### 수학과 shape

- [ ] 입력 layout과 각 축의 의미를 기록했다.
- [ ] 출력 grid를 $(H_{\mathrm{out}},W_{\mathrm{out}})$ 순서로 지정했다.
- [ ] `Linear.in_features`를 $C H_{\mathrm{out}}W_{\mathrm{out}}$로 계산했다.
- [ ] 나누어떨어지지 않는 작은 예제를 손으로 검산했다.
- [ ] overlap을 partition으로 잘못 설명하지 않았다.

### 코드와 테스트

- [ ] shape뿐 아니라 결정론적 값도 테스트했다.
- [ ] GAP와 명시적 공간 평균의 동등성을 검사했다.
- [ ] dtype별 합리적인 허용오차를 정했다.
- [ ] NHWC/NCHW 변환을 테스트했다.
- [ ] output size가 0이거나 잘못된 입력을 거부한다.

### 성능과 배포

- [ ] pool 이전 activation memory를 입력 최대 해상도로 측정했다.
- [ ] padding 비율이 평균을 오염하지 않는지 확인했다.
- [ ] target exporter와 runtime에서 동적 shape을 실행했다.
- [ ] 실제 device에서 해상도별 p95 latency를 측정했다.
- [ ] 입력 크기 범위와 초과 요청 정책을 명시했다.

## 24. 연습문제

### 문제 1

$I=8$, $O=3$일 때 각 출력 bin의 시작점, 끝점, 입력 인덱스를 구하라.

### 문제 2

입력이 `(4, 32, 15, 21)`이고 `AdaptiveAvgPool2d((3, 5))`를 적용한 뒤 batch 축을 보존해 flatten했다. flatten 결과 shape과 10개 클래스 선형층의 `in_features`를 구하라.

### 문제 3

모든 값이 7인 입력에 임의 크기의 adaptive average pooling을 적용했다. 출력은 무엇인가? 이 성질을 어떤 테스트로 만들 수 있는가?

### 문제 4

$I=7$, $O=3$인 1차원 예제에서 loss가 $L=y_0+2y_1+3y_2$다. 입력 인덱스 2가 받는 gradient를 구하라.

### 문제 5

서로 다른 크기의 이미지 8장을 `AdaptiveAvgPool2d`가 있는 모델에 한 번에 전달하려 했더니 `torch.stack`에서 실패했다. 원인과 해결책 두 가지를 설명하라.

### 문제 6

GAP를 적용하자 작은 결함 분류 recall이 하락했다. adaptive grid 관점에서 가장 먼저 할 ablation 두 가지를 제안하라.

## 25. 연습문제 해답

### 해답 1

경계식은 $s(j)=\lfloor 8j/3\rfloor$, $e(j)=\lceil 8(j+1)/3\rceil$이다.

| $j$ | $s(j)$ | $e(j)$ | 입력 인덱스 |
| ---: | ---: | ---: | --- |
| 0 | 0 | 3 | `0, 1, 2` |
| 1 | 2 | 6 | `2, 3, 4, 5` |
| 2 | 5 | 8 | `5, 6, 7` |

인덱스 2와 5가 겹친다.

### 해답 2

pool 출력은 `(4, 32, 3, 5)`다. flatten 길이는 다음과 같다.

$$
32 \times 3 \times 5=480
$$

따라서 flatten 결과는 `(4, 480)`이고 `Linear`의 `in_features`는 480이다. 클래스 10개면 출력은 `(4, 10)`이다.

### 해답 3

어떤 bin도 값 7만 평균내므로 출력의 모든 값은 7이다. 여러 입력·출력 shape을 무작위로 만들고 `torch.full` 입력에 대한 출력이 같은 상수인지 검사하는 property test로 만들 수 있다.

### 해답 4

인덱스 2는 bin 0과 bin 1에 들어가며 두 bin 크기는 모두 3이다.

$$
\frac{\partial L}{\partial x_2}
=
1\cdot\frac{1}{3}
+2\cdot\frac{1}{3}
=1
$$

### 해답 5

adaptive pooling은 모델 내부 연산이다. 그 전에 `torch.stack`으로 dense batch를 만들려면 모든 image tensor shape가 같아야 한다. 비슷한 크기끼리 bucket하고 각 bucket 안에서 padding하거나, resize/crop으로 크기를 통일할 수 있다. batch size 1도 가능하지만 처리량 trade-off가 있다.

### 해답 6

첫째, GAP `(1,1)`과 `(2,2)`, `(4,4)` adaptive grid를 비교해 위치 정보 보존이 recall을 회복하는지 본다. 둘째, average와 max 또는 두 결과의 결합을 비교해 작은 강한 신호의 평균 희석 여부를 확인한다. 파라미터 수가 달라지므로 비교 실험의 seed, 학습 budget, head 구조도 기록한다.

## 26. 핵심 요약

1. `AdaptiveAvgPool2d`는 목표 출력 공간 크기를 먼저 정한다.
2. 각 bin은 floor 시작점과 ceil 끝점으로 정의한다.
3. 입력 크기가 출력 크기로 나누어떨어지지 않으면 이웃 bin이 겹칠 수 있다.
4. coverage는 유지되지만 겹친 구간은 엄밀한 partition이 아니다.
5. 2차원 출력은 높이와 너비 경계를 독립적으로 구해 직사각형 평균을 낸다.
6. GAP는 output size가 `(1,1)`인 특수한 경우다.
7. adaptive pooling은 head의 feature 길이를 고정하지만 ragged batch와 앞단 메모리는 해결하지 않는다.
8. shape, 값, gradient를 각각 테스트해야 한다.
9. 배포에서는 layout, dtype, dynamic shape, operator 지원을 실제 runtime으로 검증한다.
10. grid가 작을수록 head는 가벼워지지만 공간 정보와 작은 신호가 희석될 수 있다.

## 27. 다음 학습 예고

다음 소스는 `02-03.SPP.md`다. 하나의 adaptive grid만 사용하는 대신 여러 해상도의 bin, 예를 들어 $1 \times 1$, $2 \times 2$, $4 \times 4$ 출력을 이어 붙이면 무엇을 얻고 어떤 비용을 치르는지 살펴본다. 오늘 배운 bin 경계, overlap, flatten 길이 계산이 Spatial Pyramid Pooling의 기초가 된다.
