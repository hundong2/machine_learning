<!-- curriculum: cycle=2; level=implementation; source_index=3/18; source=02-02.AdaptiveAvgPooll2d.md; part=1/1 -->

# AdaptiveAvgPool2d: 출력과 역전파를 하나의 골든 계약으로 검증하는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-09 |
| 회차·수준 | 2회차 · 구현 |
| 현재 소스 | 3/18 · `02-02.AdaptiveAvgPooll2d.md` |
| Part | 1/1 |
| 이전 소스 | `02-1.LazyLinear.md` |
| 다음 소스 | `02-03.SPP.md` |
| 예상 학습 시간 | 120~160분 |
| 실행 검증 환경 | Python 3.12.12 · NumPy 2.3.5 · PyTorch 2.13.0 · clang C++17 · Mono C# |

## 1. 이번 회차의 구현 질문

1회차에는 출력 bin의 시작점을 floor, 끝점을 ceil로 정한다는 수학과 shape을 익혔다. 이번에는 설명을 반복하지 않고 다음 질문에 답한다.

> 프레임워크 구현을 믿기 전에, forward 값과 backward gradient를 독립적인 골든 구현으로 어떻게 검증할 것인가?

`AdaptiveAvgPool2d`는 학습 parameter가 없지만 구현 계약은 단순하지 않다. 입력 길이가 출력 길이로 나누어떨어지지 않으면 bin이 겹치고, 겹친 입력에는 여러 출력 gradient가 누적된다. shape만 맞는 잘못된 구현도 얼마든지 만들 수 있다.

이번 문서의 산출물은 다음 네 가지다.

- 정수 연산만 쓰는 NumPy forward 기준 구현
- 겹침을 정확히 누적하는 NumPy backward 기준 구현
- 작은 shape 공간을 전수 조사하는 property harness
- 학습·평가·ablation·배포 전 검사를 포함한 PyTorch 모듈

## 2. 학습 목표

학습을 마치면 다음을 할 수 있어야 한다.

1. 출력 bin 경계를 부동소수점 없이 계산한다.
2. 2차원 forward와 backward를 수식에서 코드로 옮긴다.
3. 값, gradient, 상수 보존, coverage를 서로 다른 테스트로 검증한다.
4. `gradcheck`와 독립 기준 구현이 잡는 버그의 차이를 설명한다.
5. GAP와 공간 grid head를 같은 training harness에서 ablation한다.
6. NCHW·NHWC, float32·float64 계약을 프레임워크 사이에서 맞춘다.
7. 동적 입력 export가 실제 target runtime에서 검증되기 전에는 지원된다고 선언하지 않는다.

## 3. 선수 지식과 기호

### 3.1 텐서와 인덱스

입력과 출력은 NCHW layout을 기준으로 둔다.

$$
X \in \mathbb{R}^{N \times C \times H_i \times W_i}
$$

$$
Y \in \mathbb{R}^{N \times C \times H_o \times W_o}
$$

- $N$: batch 크기
- $C$: channel 수
- $H_i,W_i$: 입력 높이와 너비
- $H_o,W_o$: 목표 출력 높이와 너비
- $p,q$: 출력의 높이·너비 인덱스
- $h,w$: 입력의 높이·너비 인덱스

평균은 $H,W$ 축에서만 일어난다. $N,C$ 축의 원소는 섞이지 않는다.

### 3.2 정수 나눗셈으로 경계 계산하기

출력 높이 인덱스 $p$의 시작과 끝은 다음과 같다.

$$
h_s(p)=\left\lfloor\frac{pH_i}{H_o}\right\rfloor
$$

$$
h_e(p)=\left\lceil\frac{(p+1)H_i}{H_o}\right\rceil
$$

양의 정수 $a,b$에 대해 다음 항등식을 쓸 수 있다.

$$
\left\lceil\frac{a}{b}\right\rceil
=
\left\lfloor\frac{a+b-1}{b}\right\rfloor
$$

따라서 코드에서는 `math.ceil(a / b)` 대신 `(a + b - 1) // b`를 쓴다. 큰 shape에서 부동소수점 반올림에 기대지 않고 C++·C#과 같은 규칙을 공유할 수 있다.

너비 경계도 같은 방식이다.

$$
w_s(q)=\left\lfloor\frac{qW_i}{W_o}\right\rfloor,
\qquad
w_e(q)=\left\lceil\frac{(q+1)W_i}{W_o}\right\rceil
$$

## 4. 직관: 연산자가 아니라 실행 명세를 만든다

프레임워크 함수 호출 한 줄은 구현이 아니라 의존이다. 구현 수준에서 필요한 것은 독립적으로 검사할 수 있는 명세다.

```text
입력 shape + 목표 shape
        |
        v
정수 경계표 생성 -----> coverage/범위 property
        |
        v
bin별 평균 -----------> forward golden
        |
        v
bin별 gradient 분배 --> backward golden
        |
        v
PyTorch 결과와 값·gradient 교차 검증
```

같은 경계표를 forward와 backward가 공유해야 한다. forward만 맞고 backward에서 overlap 누적을 덮어쓰는 버그, 또는 두 구현이 서로 다른 반올림 규칙을 쓰는 버그를 막기 위해서다.

## 5. 단계별 forward 유도

출력 위치 $(p,q)$가 읽는 직사각형을 다음처럼 둔다.

$$
R_{p,q}
=
[h_s(p),h_e(p)) \times [w_s(q),w_e(q))
$$

그 면적은 다음과 같다.

$$
A_{p,q}
=
\bigl(h_e(p)-h_s(p)\bigr)
\bigl(w_e(q)-w_s(q)\bigr)
$$

출력은 영역 평균이다.

$$
Y_{n,c,p,q}
=
\frac{1}{A_{p,q}}
\sum_{h=h_s(p)}^{h_e(p)-1}
\sum_{w=w_s(q)}^{w_e(q)-1}
X_{n,c,h,w}
$$

### 5.1 손으로 검산: $4 \times 5 \rightarrow 3 \times 2$

입력을 다음과 같이 둔다.

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20
\end{bmatrix}
$$

높이 경계는 `[0:2]`, `[1:3]`, `[2:4]`이고 너비 경계는 `[0:3]`, `[2:5]`다. 출력 `(0,0)`은 `1,2,3,6,7,8`의 평균 4.5다. 출력 `(1,1)`은 `8,9,10,13,14,15`의 평균 11.5다.

전체 결과는 다음과 같다.

$$
Y=
\begin{bmatrix}
4.5 & 6.5 \\
9.5 & 11.5 \\
14.5 & 16.5
\end{bmatrix}
$$

높이의 1행과 2행, 너비의 2열은 이웃 bin 사이에서 겹친다. 이 중복은 backward에서 핵심이 된다.

## 6. 단계별 backward 유도

상류 gradient를 다음처럼 둔다.

$$
G_{n,c,p,q}=\frac{\partial L}{\partial Y_{n,c,p,q}}
$$

입력 $X_{n,c,h,w}$가 영역 $R_{p,q}$에 포함되면 해당 평균의 미분은 $1/A_{p,q}$이고, 포함되지 않으면 0이다.

$$
\frac{\partial Y_{n,c,p,q}}{\partial X_{n,c,h,w}}
=
\begin{cases}
\dfrac{1}{A_{p,q}}, & (h,w) \in R_{p,q} \\
0, & \text{otherwise}
\end{cases}
$$

연쇄 법칙을 적용하면 다음과 같다.

$$
\frac{\partial L}{\partial X_{n,c,h,w}}
=
\sum_{p=0}^{H_o-1}
\sum_{q=0}^{W_o-1}
G_{n,c,p,q}
\frac{\partial Y_{n,c,p,q}}{
\partial X_{n,c,h,w}}
$$

따라서 backward 구현은 각 bin의 상류 gradient를 면적으로 나눈 뒤 입력 slice에 **더해야** 한다.

```python
grad_x[..., hs:he, ws:we] += grad_y[..., oh, ow] / area
```

`=`로 대입하면 겹친 위치의 앞선 기여가 사라진다. 이 버그는 입력과 출력 크기가 나누어떨어지는 테스트만으로는 드러나지 않는다.

## 7. 실행 가능한 NumPy 골든 구현

다음 코드는 독립 실행 가능하다. forward, backward, 경계 property, PyTorch 값·gradient 교차 검증을 하나의 파일에서 수행한다.

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Bin:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


def make_bins(input_size: int, output_size: int) -> list[Bin]:
    if input_size <= 0 or output_size <= 0:
        raise ValueError("input_size and output_size must be positive")
    bins = []
    for out_index in range(output_size):
        start = (out_index * input_size) // output_size
        numerator = (out_index + 1) * input_size
        end = (numerator + output_size - 1) // output_size
        bins.append(Bin(start, end))
    return bins


def adaptive_avg_pool2d_forward(
    x: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError("x must use NCHW layout")
    n, c, input_h, input_w = x.shape
    output_h, output_w = output_size
    h_bins = make_bins(input_h, output_h)
    w_bins = make_bins(input_w, output_w)
    result_dtype = np.result_type(x.dtype, np.float32)
    y = np.empty((n, c, output_h, output_w), dtype=result_dtype)
    for oh, h_bin in enumerate(h_bins):
        for ow, w_bin in enumerate(w_bins):
            region = x[
                :, :, h_bin.start : h_bin.end, w_bin.start : w_bin.end
            ]
            y[:, :, oh, ow] = region.mean(axis=(-2, -1))
    return y


def adaptive_avg_pool2d_backward(
    grad_y: np.ndarray,
    input_shape: tuple[int, int, int, int],
) -> np.ndarray:
    if grad_y.ndim != 4:
        raise ValueError("grad_y must use NCHW layout")
    n, c, input_h, input_w = input_shape
    if grad_y.shape[:2] != (n, c):
        raise ValueError("batch and channel dimensions must agree")
    output_h, output_w = grad_y.shape[-2:]
    h_bins = make_bins(input_h, output_h)
    w_bins = make_bins(input_w, output_w)
    grad_x = np.zeros(input_shape, dtype=grad_y.dtype)
    for oh, h_bin in enumerate(h_bins):
        for ow, w_bin in enumerate(w_bins):
            area = h_bin.size * w_bin.size
            contribution = grad_y[:, :, oh, ow, None, None] / area
            grad_x[
                :, :, h_bin.start : h_bin.end, w_bin.start : w_bin.end
            ] += contribution
    return grad_x


def assert_bin_properties(input_size: int, output_size: int) -> None:
    bins = make_bins(input_size, output_size)
    assert bins[0].start == 0
    assert bins[-1].end == input_size
    coverage = np.zeros(input_size, dtype=np.int64)
    for index, current in enumerate(bins):
        assert 0 <= current.start < current.end <= input_size
        if index:
            previous = bins[index - 1]
            assert previous.start <= current.start
            assert previous.end <= current.end
            assert current.start <= previous.end
        coverage[current.start : current.end] += 1
    assert np.all(coverage >= 1)


def run_exhaustive_properties() -> None:
    # 작은 shape 공간은 난수 표본 대신 전부 검사한다.
    generator = np.random.default_rng(20260809)
    cases = 0
    for input_h in range(1, 9):
        for input_w in range(1, 9):
            for output_h in range(1, 7):
                for output_w in range(1, 7):
                    assert_bin_properties(input_h, output_h)
                    assert_bin_properties(input_w, output_w)
                    x_np = generator.normal(
                        size=(1, 2, input_h, input_w)
                    ).astype(np.float64)
                    y_np = adaptive_avg_pool2d_forward(
                        x_np, (output_h, output_w)
                    )
                    x_torch = torch.tensor(x_np, requires_grad=True)
                    y_torch = F.adaptive_avg_pool2d(
                        x_torch, (output_h, output_w)
                    )
                    np.testing.assert_allclose(
                        y_np, y_torch.detach().numpy(), rtol=1e-12, atol=1e-12
                    )
                    grad_y = generator.normal(size=y_np.shape)
                    grad_x_np = adaptive_avg_pool2d_backward(
                        grad_y, x_np.shape
                    )
                    y_torch.backward(torch.from_numpy(grad_y))
                    np.testing.assert_allclose(
                        grad_x_np,
                        x_torch.grad.numpy(),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    constant = np.full_like(x_np, 3.25)
                    constant_y = adaptive_avg_pool2d_forward(
                        constant, (output_h, output_w)
                    )
                    np.testing.assert_array_equal(
                        constant_y, np.full_like(constant_y, 3.25)
                    )
                    cases += 1
    print(f"exhaustive cases: {cases}")


def run_hand_check() -> None:
    x = np.arange(1, 21, dtype=np.float64).reshape(1, 1, 4, 5)
    expected = np.array(
        [[[[4.5, 6.5], [9.5, 11.5], [14.5, 16.5]]]]
    )
    actual = adaptive_avg_pool2d_forward(x, (3, 2))
    np.testing.assert_array_equal(actual, expected)
    grad_y = np.ones_like(actual)
    grad_x = adaptive_avg_pool2d_backward(grad_y, x.shape)
    # 모든 출력 합의 gradient 합은 출력 원소 수와 같다.
    np.testing.assert_allclose(grad_x.sum(), grad_y.sum())
    print(actual.reshape(3, 2))
    print("gradient sum:", grad_x.sum())


if __name__ == "__main__":
    run_hand_check()
    run_exhaustive_properties()
```

예상 핵심 출력은 다음과 같다.

```text
[[ 4.5  6.5]
 [ 9.5 11.5]
 [14.5 16.5]]
gradient sum: 6.0
exhaustive cases: 2304
```

### 7.1 왜 전수 테스트인가

`hypothesis` 같은 property-based testing 도구가 없어도 작은 정수 shape를 전부 훑을 수 있다. $H_i,W_i \in [1,8]$, $H_o,W_o \in [1,6]$이면 2,304개 조합이다. 특히 다음 경우를 자동으로 포함한다.

- 입력과 출력이 같은 경우
- 정확히 나누어떨어지는 downsampling
- 나누어떨어지지 않는 downsampling
- 출력이 입력보다 큰 upsampling 형태
- 높이만 커지거나 너비만 커지는 비대칭 형태

이 테스트는 난수 seed도 고정하므로 실패를 재현할 수 있다.

## 8. gradient 검증의 세 층

### 8.1 독립 backward와 autograd 비교

위 테스트는 NumPy backward와 PyTorch autograd를 비교한다. 두 구현이 같은 실수를 공유할 가능성을 줄이기 위해 NumPy 코드는 프레임워크 pooling API를 호출하지 않는다.

### 8.2 보존 법칙

loss를 모든 출력의 합으로 두면 각 출력의 상류 gradient가 1이다. 각 bin이 입력에 나누어 주는 gradient의 합은 1이므로 전체 입력 gradient 합은 출력 원소 수와 같다.

$$
\sum_{n,c,h,w}\frac{\partial L}{\partial X_{n,c,h,w}}
=
NCH_oW_o
$$

이 invariant는 특정 golden 배열보다 넓은 shape에서 동작하며, backward의 누락과 잘못된 분모를 빠르게 잡는다.

### 8.3 `gradcheck`

다음 코드는 실행 가능하며 PyTorch 연산 자체의 수치 미분 검사를 수행한다.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(20260809)
x = torch.randn(1, 1, 4, 5, dtype=torch.float64, requires_grad=True)

passed = torch.autograd.gradcheck(
    lambda value: F.adaptive_avg_pool2d(value, (3, 2)),
    (x,),
    eps=1e-6,
    atol=1e-6,
    rtol=1e-4,
)
assert passed
print("gradcheck:", passed)
```

`gradcheck`는 유한 차분과 autograd를 비교하지만 우리 서비스 코드의 layout 변환이나 참조 backward까지 검증하지 않는다. 독립 golden, invariant, `gradcheck`를 서로 대체 관계로 보지 않는다.

## 9. 완전한 PyTorch 학습·평가 구현

다음 코드는 실행 가능한 최소 학습 파이프라인이다. 입력 해상도가 달라도 같은 classifier head를 쓰며, GAP와 `2 x 2` grid를 동일한 데이터·seed·optimizer 조건에서 비교한다.

```python
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class StripeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, count: int, height: int, width: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        images = torch.randn(count, 1, height, width, generator=generator) * 0.05
        labels = torch.arange(count) % 2
        for index, label in enumerate(labels.tolist()):
            if label == 0:
                images[index, :, :, : width // 3] += 1.0
            else:
                images[index, :, :, -width // 3 :] += 1.0
        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class PoolClassifier(nn.Module):
    def __init__(self, grid: tuple[int, int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.classifier = nn.Linear(8 * grid[0] * grid[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled.flatten(start_dim=1))


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
) -> Metrics:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            batch_size = labels.numel()
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size
    return Metrics(loss_sum / sample_count, correct / sample_count)


def experiment(grid: tuple[int, int]) -> tuple[Metrics, tuple[int, ...]]:
    seed_everything(20260809)
    train_data = StripeDataset(32, 9, 13, seed=1)
    valid_data = StripeDataset(16, 11, 15, seed=2)
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(3),
    )
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False)
    model = PoolClassifier(grid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    for _ in range(12):
        run_epoch(model, train_loader, optimizer)
    metrics = run_epoch(model, valid_loader, optimizer=None)
    with torch.no_grad():
        shape = tuple(model(torch.randn(3, 1, 17, 19)).shape)
    return metrics, shape


if __name__ == "__main__":
    for candidate in [(1, 1), (2, 2)]:
        metrics, output_shape = experiment(candidate)
        print(candidate, metrics, output_shape)
        assert output_shape == (3, 2)
        assert metrics.accuracy >= 0.95
```

실행 결과는 다음과 같았다.

```text
(1, 1) Metrics(loss=0.13634394854307175, accuracy=1.0) (3, 2)
(2, 2) Metrics(loss=0.0, accuracy=1.0) (3, 2)
```

이 합성 과제는 좌우 위치가 class를 결정하지만 GAP도 정확도 1.0을 얻었다. GAP가 마지막 feature map의 공간 좌표를 평균내더라도, zero padding을 쓰는 convolution은 경계와의 관계를 feature 값에 인코딩할 수 있다. 즉 “GAP를 쓰면 위치 정보를 전혀 이용할 수 없다”는 주장은 이 실험으로 입증되지 않는다.

이 결과는 실패한 실험이 아니라 ablation 설계의 허점을 드러낸다. pooling grid의 효과만 분리하려면 convolution을 제거한 고정 feature, circular padding, 위치를 상쇄한 합성 데이터 등을 추가해야 한다. 여러 seed에서 정확도뿐 아니라 loss, parameter 수, latency를 비교해야 한다. ablation의 목적은 예상한 결론을 만드는 것이 아니라 어떤 정보 경로가 남았는지 확인하는 것이다.

## 10. tensor shape 추적

`grid=(2,2)`, batch 8인 학습 입력의 shape은 다음과 같다.

| 단계 | 학습 입력 | 검증 입력 | 계약 |
| --- | --- | --- | --- |
| 입력 | `(8, 1, 9, 13)` | `(8, 1, 11, 15)` | NCHW, float32 |
| 첫 convolution | `(8, 4, 9, 13)` | `(8, 4, 11, 15)` | 공간 크기 유지 |
| 둘째 convolution | `(8, 8, 9, 13)` | `(8, 8, 11, 15)` | channel 8 |
| adaptive pool | `(8, 8, 2, 2)` | `(8, 8, 2, 2)` | 출력 공간 고정 |
| flatten | `(8, 32)` | `(8, 32)` | $8 \times 2 \times 2$ |
| classifier | `(8, 2)` | `(8, 2)` | class logits |

GAP를 쓰면 pool은 `(8,8,1,1)`, flatten은 `(8,8)`이다. grid ablation은 classifier parameter 수도 바꾸므로 정확도뿐 아니라 head 크기와 latency도 함께 보고해야 한다.

## 11. framework 간 shape·layout·dtype 계약

| 환경 | 기본 예제 layout | 입력 dtype | 누적·출력 정책 | 주의점 |
| --- | --- | --- | --- | --- |
| PyTorch | NCHW | float32 또는 float64 | 입력 dtype 유지 | `AdaptiveAvgPool2d`의 축은 마지막 두 축 |
| NumPy 골든 | NCHW로 명시 | float32 또는 float64 | 최소 float32 결과 | integer 평균의 truncation 금지 |
| C++ 예제 | 평면 HW | double | double | 실제 NCHW는 `(n,c)`마다 같은 함수를 호출 |
| C# 예제 | 2차원 배열 | double | double | row-major index를 명시적으로 계산 |
| 일반 NHWC runtime | NHWC | backend 계약 | backend 계약 | pool 전후 transpose 또는 native axis 설정 필요 |

NHWC 입력 `(N,H,W,C)`를 NCHW 기준 구현에 그대로 전달하면 마지막 두 축을 `W,C`로 오해한다. shape가 우연히 정상이어도 channel 평균이 일어날 수 있으므로, 값이 channel별로 다른 layout golden test를 둔다.

## 12. C++17 골든 예제

다음 코드는 독립 실행 가능하다. 외부 tensor library 없이 단일 channel `4 x 5 -> 3 x 2`를 검산한다.

```cpp
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int start_index(int output_index, int output_size, int input_size) {
    return output_index * input_size / output_size;
}

int end_index(int output_index, int output_size, int input_size) {
    const int numerator = (output_index + 1) * input_size;
    return (numerator + output_size - 1) / output_size;
}

std::vector<double> adaptive_avg_pool2d(
    const std::vector<double>& input,
    int input_h,
    int input_w,
    int output_h,
    int output_w
) {
    if (input_h <= 0 || input_w <= 0 || output_h <= 0 || output_w <= 0) {
        throw std::invalid_argument("sizes must be positive");
    }
    if (input.size() != static_cast<std::size_t>(input_h * input_w)) {
        throw std::invalid_argument("input size mismatch");
    }
    std::vector<double> output(output_h * output_w, 0.0);
    for (int oh = 0; oh < output_h; ++oh) {
        const int hs = start_index(oh, output_h, input_h);
        const int he = end_index(oh, output_h, input_h);
        for (int ow = 0; ow < output_w; ++ow) {
            const int ws = start_index(ow, output_w, input_w);
            const int we = end_index(ow, output_w, input_w);
            double sum = 0.0;
            for (int h = hs; h < he; ++h) {
                for (int w = ws; w < we; ++w) {
                    sum += input[h * input_w + w];
                }
            }
            const int area = (he - hs) * (we - ws);
            output[oh * output_w + ow] = sum / area;
        }
    }
    return output;
}

int main() {
    std::vector<double> input(20);
    for (int index = 0; index < 20; ++index) {
        input[index] = index + 1.0;
    }
    const auto output = adaptive_avg_pool2d(input, 4, 5, 3, 2);
    const std::vector<double> expected{4.5, 6.5, 9.5, 11.5, 14.5, 16.5};
    for (std::size_t index = 0; index < output.size(); ++index) {
        assert(std::abs(output[index] - expected[index]) < 1e-12);
    }
    std::cout << std::fixed << std::setprecision(1);
    for (double value : output) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}
```

예상 출력은 `4.5 6.5 9.5 11.5 14.5 16.5`다. 실제 production C++에서는 `int` 곱셈 overflow를 피하려고 shape와 중간 곱을 `std::int64_t`로 승격해야 한다.

## 13. C# 골든 예제

다음 코드는 독립 실행 가능하며 C++ 예제와 같은 값을 검사한다.

```csharp
using System;

public static class AdaptivePoolGolden
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

    private static double[,] AdaptiveAvgPool2D(double[,] input, int outH, int outW)
    {
        int inH = input.GetLength(0);
        int inW = input.GetLength(1);
        if (inH <= 0 || inW <= 0 || outH <= 0 || outW <= 0)
        {
            throw new ArgumentOutOfRangeException("sizes must be positive");
        }
        var output = new double[outH, outW];
        for (int oh = 0; oh < outH; ++oh)
        {
            int hs = StartIndex(oh, outH, inH);
            int he = EndIndex(oh, outH, inH);
            for (int ow = 0; ow < outW; ++ow)
            {
                int ws = StartIndex(ow, outW, inW);
                int we = EndIndex(ow, outW, inW);
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
        var input = new double[4, 5];
        int value = 1;
        for (int h = 0; h < 4; ++h)
        {
            for (int w = 0; w < 5; ++w)
            {
                input[h, w] = value++;
            }
        }
        double[,] expected = {
            {4.5, 6.5},
            {9.5, 11.5},
            {14.5, 16.5}
        };
        var actual = AdaptiveAvgPool2D(input, 3, 2);
        for (int h = 0; h < 3; ++h)
        {
            for (int w = 0; w < 2; ++w)
            {
                if (Math.Abs(actual[h, w] - expected[h, w]) > 1e-12)
                {
                    throw new Exception("golden mismatch");
                }
                Console.Write(actual[h, w].ToString("F1") + " ");
            }
        }
        Console.WriteLine();
    }
}
```

## 14. 테스트 전략

### 14.1 테스트 피라미드

| 층 | 대표 검사 | 잡는 실패 |
| --- | --- | --- |
| 예제 기반 | `4 x 5 -> 3 x 2` golden | 축 교환, 평균 분모 오류 |
| property | 작은 shape 2,304개 전수 | 경계·coverage·upsampling 오류 |
| gradient | NumPy backward 대 autograd | overlap 누적, 잘못된 area |
| 수치 미분 | `gradcheck` | autograd backward 불일치 |
| 모델 통합 | 두 해상도 train/eval | flatten 길이, head 연결 오류 |
| 배포 통합 | runtime별 두 동적 shape | exporter 분해, layout, dtype 오류 |

### 14.2 반드시 포함할 negative test

- 입력 rank가 4가 아니면 거부한다.
- 입력 또는 출력 공간 크기가 0이면 거부한다.
- backward의 batch·channel이 입력 shape와 다르면 거부한다.
- integer 입력은 평균 결과 dtype 정책을 명시한다.
- `NaN` 또는 `Inf` 입력의 전파 정책을 확인한다.
- 잘못된 NHWC 입력이 조용히 통과하지 않도록 adapter 경계에서 검사한다.

### 14.3 shape-only test가 부족한 이유

시작점을 ceil, 끝점을 floor로 뒤집어도 일부 shape는 빈 slice 없이 같은 출력 shape를 만든다. `AvgPool2d`의 고정 kernel·stride로 근사해도 나누어떨어지는 case는 통과한다. 그러므로 최소 한 개의 비가분 shape golden과 gradient 검사가 필요하다.

## 15. 디버깅 플레이북

### 증상 A: shape는 맞지만 값이 PyTorch와 다르다

1. 높이·너비별 경계표를 출력한다.
2. end가 반열린 구간인지 확인한다.
3. 각 위치의 분모가 실제 bin 면적인지 확인한다.
4. NCHW와 NHWC를 혼동하지 않았는지 확인한다.
5. integer division으로 평균이 잘리지 않는지 확인한다.

### 증상 B: forward는 맞지만 gradient가 다르다

1. 입력과 출력이 나누어떨어지지 않는 case인지 확인한다.
2. backward slice가 `+=`인지 확인한다.
3. 상류 gradient shape와 batch·channel을 확인한다.
4. bin마다 다른 area를 사용했는지 확인한다.
5. float64의 작은 tensor로 autograd와 비교한다.

### 증상 C: 학습에서는 되는데 export runtime에서 실패한다

1. pool 하나만 남긴 최소 모델을 export한다.
2. 고정 shape와 동적 shape를 분리해 시험한다.
3. target runtime이 임의 output grid를 지원하는지 확인한다.
4. exporter가 여러 slice·reduce 연산으로 분해했는지 graph를 본다.
5. GAP 또는 지원되는 고정 pool로 바꾼 ablation을 실행한다.

## 16. 성능·메모리·수치 안정성

### 16.1 계산량

naive 구현은 각 출력 bin의 모든 원소를 읽는다. 겹침 때문에 일부 입력을 여러 번 읽을 수 있다. 대략적인 연산량은 다음과 같다.

$$
\mathcal{O}
\left(
NC
\sum_{p=0}^{H_o-1}
\sum_{q=0}^{W_o-1}
A_{p,q}
\right)
$$

GAP는 각 입력을 사실상 한 번 읽으므로 memory bandwidth 지배 연산이 되기 쉽다. 더 큰 adaptive grid와 overlap은 읽기 횟수를 늘릴 수 있다.

### 16.2 activation memory

pool 출력의 원소 수는 $NCH_oW_o$다. 그러나 peak memory는 pool 전 feature map $NCH_iW_i$가 지배할 수 있다. adaptive pooling이 head shape을 고정한다고 해서 입력 해상도의 메모리 비용이 사라지지 않는다.

float32 activation의 단순 저장 크기는 다음과 같다.

$$
M_{\text{bytes}}=4NCH_iW_i
$$

학습에서는 gradient, 저장된 중간 activation, allocator workspace까지 더해진다.

### 16.3 dtype

- 검산에는 float64를 써서 반올림 잡음을 줄인다.
- production float32는 backend별 reduction 순서 차이를 고려해 tolerance를 둔다.
- float16·bfloat16은 큰 bin 합산 오차와 backend accumulation dtype을 실제 device에서 측정한다.
- integer 입력을 자체 구현이 허용한다면 출력 dtype과 rounding을 API로 명시한다.

### 16.4 결정론

CPU의 작은 예제가 결정적이어도 GPU parallel reduction 순서에 따라 마지막 비트가 달라질 수 있다. 배포 parity test는 bitwise equality보다 dtype·backend별 `rtol`, `atol`과 task metric 한도를 함께 사용한다.

## 17. 실무 실패 사례

### 사례 1: backward가 overlap을 덮어썼다

사내 kernel은 forward benchmark를 통과했지만 학습 loss가 기준 모델보다 느리게 줄었다. backward에서 각 bin의 gradient를 입력에 대입해 겹친 위치의 앞선 기여를 지우고 있었다. 나누어떨어지는 `8 -> 4` test만 있어서 발견하지 못했다.

대응은 `7 -> 3`, `4 -> 3` 같은 비가분 shape와 gradient sum invariant를 회귀 테스트에 추가하는 것이다.

### 사례 2: mobile adapter가 NHWC를 NCHW로 해석했다

입출력 rank와 원소 수가 맞아 inference가 중단되지는 않았지만 channel 축이 공간 평균에 섞여 정확도가 급락했다. 모든 channel에 같은 값을 넣은 기존 golden은 축 교환을 감추었다.

대응은 channel마다 상이한 상수와 비대칭 $H,W$를 사용하는 layout golden을 추가하고 adapter에서 layout enum을 강제하는 것이다.

### 사례 3: 가변 해상도 지원 선언 뒤 OOM이 발생했다

head가 어떤 크기도 받는다는 사실을 전체 모델이 무제한 크기를 처리한다는 뜻으로 해석했다. 고해상도 요청에서 pool 이전 backbone activation이 GPU memory를 소진했다.

대응은 허용 $H,W$ 범위, pixel budget, bucket별 batch 크기, 초과 요청 거부를 모델 계약에 넣는 것이다.

### 사례 4: export 성공을 runtime 지원으로 오해했다

모델 파일은 생성되었지만 target runtime에서 임의 grid가 지원되지 않거나 비효율적인 subgraph로 분해되었다. 첫 요청에서 compile 시간이 길어지고 steady-state latency도 목표를 넘었다.

대응은 최소 모델 export, graph 검사, 두 입력 shape parity, warm-up 뒤 p50·p95 benchmark를 배포 gate로 만든다.

## 18. 원문과 1회차에서 확장한 내용

| 구분 | 기존 내용 | 이번 구현 회차의 확장 |
| --- | --- | --- |
| 경계 | floor·ceil 직관 | 부동소수점 없는 공통 정수 helper |
| forward | NumPy와 PyTorch 값 비교 | 2,304개 shape 전수 교차 검증 |
| backward | overlap에서 gradient가 더해짐 | 독립 backward 코드와 보존 invariant |
| 테스트 | 몇 개의 예제와 shape test | golden·property·gradcheck·통합 test 피라미드 |
| 학습 | 가변 해상도 classifier forward | 재현 가능한 train/eval과 grid ablation |
| 배포 | runtime 확인 필요 | 최소 export·graph·parity·latency gate |

원문의 “동적으로 kernel과 stride를 계산한다”는 입문 직관은 단일 고정 kernel과 stride로 모든 bin을 설명할 수 있다는 뜻으로 사용하면 안 된다. 정확한 실행 계약은 bin마다 계산한 반열린 구간이다. 또한 겹치는 bin은 coverage는 보장하지만 수학적 partition은 아니다.

소스 파일명은 `AdaptiveAvgPooll2d`로 `l`이 하나 더 들어가지만 PyTorch API의 정확한 이름은 `AdaptiveAvgPool2d`다. curriculum의 `source`에는 실제 파일명을 그대로 기록했다.

## 19. 배포 관점

### 19.1 모델 입력 계약

다음 항목을 artifact와 함께 versioning한다.

- 입력 layout, channel order, dtype
- 허용 batch와 $H,W$ 최소·최대
- normalization과 padding 값
- 목표 pooling grid
- 해상도 bucket별 latency·memory 한도
- output logits의 class order
- exporter·runtime·device 조합

### 19.2 ONNX 검증 계획

현재 로컬 환경에는 `onnx` package가 설치되어 있지 않아 이 문서의 ONNX export 코드는 실행 검증하지 않았다. 임의 설치로 환경을 바꾸지 않고 다음 gate를 명세한다.

```python
# 설명용: onnx 및 target runtime 설치 환경에서 실행해야 한다.
import torch
from torch import nn

model = nn.AdaptiveAvgPool2d((2, 3)).eval()
example = torch.randn(1, 8, 11, 13)

torch.onnx.export(
    model,
    (example,),
    "adaptive_pool.onnx",
    input_names=["features"],
    output_names=["pooled"],
    dynamic_axes={
        "features": {0: "batch", 2: "height", 3: "width"},
        "pooled": {0: "batch"},
    },
    dynamo=True,
)
```

export 파일 생성만으로 통과시키지 않는다. target runtime에서 적어도 `(1,8,11,13)`과 `(3,8,17,19)`를 실행하고 PyTorch 출력과 비교해야 한다. exporter와 runtime의 지원 범위는 버전에 따라 달라질 수 있으므로 실제 배포 조합의 결과가 유일한 근거다.

### 19.3 모니터링

- 입력 $H,W$, aspect ratio, pixel 수 분포
- 해상도 bucket별 p50·p95 latency와 OOM
- pool 전후 activation의 평균·표준편차·최댓값
- padding 비율과 masked sample 비율
- grid별 accuracy·macro F1·calibration
- runtime fallback 또는 graph recompilation 횟수

## 20. 구현·리뷰 체크리스트

### 수학·shape

- [ ] 시작은 floor, 끝은 ceil이며 end-exclusive다.
- [ ] 높이와 너비 경계를 독립적으로 계산한다.
- [ ] bin마다 실제 면적을 분모로 쓴다.
- [ ] 출력 shape은 `(N,C,H_o,W_o)`다.
- [ ] overlap을 partition이라고 부르지 않는다.

### 코드·테스트

- [ ] 비가분 shape golden을 포함한다.
- [ ] forward 값과 backward gradient를 모두 비교한다.
- [ ] backward slice는 `+=`로 누적한다.
- [ ] 상수 보존과 gradient sum invariant를 검사한다.
- [ ] 동일 seed로 GAP와 공간 grid를 ablation한다.
- [ ] layout test는 비대칭 shape와 channel별 다른 값을 쓴다.

### 성능·배포

- [ ] 최대 입력에서 pool 이전 activation memory를 측정한다.
- [ ] dtype·device별 parity tolerance를 정한다.
- [ ] 최소 모델과 전체 모델을 각각 export한다.
- [ ] 두 개 이상의 동적 입력 shape를 target runtime에서 실행한다.
- [ ] graph 분해와 warm-up 뒤 latency를 확인한다.
- [ ] 지원하지 않은 조합을 문서에서 명확히 거부한다.

## 21. 연습문제

### 문제 1

$I=5$, $O=4$일 때 네 bin의 반열린 구간을 구하라.

### 문제 2

입력 shape이 `(6,16,23,31)`이고 목표가 `(3,5)`다. pool 출력, flatten 출력, 7-class `Linear`의 weight shape을 구하라.

### 문제 3

loss가 모든 pool 출력의 합이고 입력 shape이 `(2,3,4,5)`, 출력 shape이 `(2,3,3,2)`다. 모든 입력 gradient의 합은 얼마인가?

### 문제 4

backward에서 `grad_x[s:e] = contribution`을 사용했을 때 왜 `8 -> 4` test는 통과하고 `7 -> 3` test는 실패할 수 있는가?

### 문제 5

GAP 모델과 `2 x 2` grid 모델을 비교할 때 정확도만 보고 결론 내리면 안 되는 이유를 두 가지 쓰라.

### 문제 6

ONNX 파일 생성은 성공했지만 동적 입력 지원을 선언할 수 없는 이유와 최소 추가 검증을 쓰라.

## 22. 연습문제 해답

### 해답 1

경계는 다음과 같다.

| 출력 인덱스 | 시작 | 끝 | 구간 |
| ---: | ---: | ---: | --- |
| 0 | 0 | 2 | `[0:2]` |
| 1 | 1 | 3 | `[1:3]` |
| 2 | 2 | 4 | `[2:4]` |
| 3 | 3 | 5 | `[3:5]` |

이웃 bin이 한 원소씩 겹친다.

### 해답 2

pool 출력은 `(6,16,3,5)`, flatten 출력은 `(6,240)`이다. `nn.Linear(240,7)`의 weight shape은 `(7,240)`이고 bias shape은 `(7,)`다.

### 해답 3

출력 원소 수와 같으므로 다음과 같다.

$$
2 \times 3 \times 3 \times 2=36
$$

### 해답 4

`8 -> 4`에서는 `[0:2]`, `[2:4]`, `[4:6]`, `[6:8]`처럼 bin이 겹치지 않는다. 각 입력 위치가 한 번만 쓰이므로 대입과 누적이 우연히 같다. `7 -> 3`에서는 경계 위치가 두 bin에 포함되어 두 gradient를 더해야 하므로 대입은 앞의 기여를 지운다.

### 해답 5

첫째, `2 x 2` head는 flatten 길이와 classifier parameter가 GAP보다 네 배이므로 model capacity가 다르다. 둘째, latency와 memory도 달라질 수 있다. 동일 seed·학습 budget 외에도 parameter 수, 추론 비용, 여러 seed의 분산을 함께 보고해야 pooling grid의 효과를 분리할 수 있다.

### 해답 6

exporter가 연산을 target runtime이 지원하지 않는 graph로 만들 수 있고, 고정 example shape만 성공했을 수 있다. 최소한 graph를 검사하고 두 동적 입력 shape를 target runtime에서 실행해 PyTorch와 값·shape를 비교하며 latency까지 측정해야 한다.

## 23. 핵심 요약

1. adaptive average pooling의 정확한 계약은 출력 bin별 정수 경계와 영역 평균이다.
2. ceil 경계는 양의 정수식으로 계산해 언어 간 반올림 차이를 없앨 수 있다.
3. backward는 상류 gradient를 bin 면적으로 나누어 입력 slice에 누적한다.
4. 비가분 shape의 overlap 때문에 `+=`가 필수다.
5. 독립 NumPy forward·backward는 프레임워크 결과의 골든 기준이 된다.
6. 작은 shape 전수 검사는 random 몇 개보다 경계 case를 안정적으로 포괄한다.
7. golden, property, invariant, `gradcheck`, 모델 통합 test는 서로 다른 실패를 잡는다.
8. GAP와 더 큰 grid의 비교에는 정확도뿐 아니라 parameter·latency trade-off가 있다.
9. adaptive pooling은 head shape을 고정하지만 ragged batch와 backbone memory를 해결하지 않는다.
10. export 성공과 target runtime의 동적 shape 지원은 같은 말이 아니다.

## 24. 다음 학습 예고

다음 소스는 `02-03.SPP.md`다. 2회차 구현에서는 여러 adaptive grid의 출력을 이어 붙이는 SPP 모듈을 완성하고, level 순서·flatten offset·gradient·가변 shape batch·ablation을 단위 테스트한다. 오늘 만든 bin 골든 구현을 각 pyramid level의 oracle로 재사용할 수 있다.
