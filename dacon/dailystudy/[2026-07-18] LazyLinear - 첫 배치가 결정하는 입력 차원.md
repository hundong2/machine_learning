<!-- curriculum: cycle=1; level=foundation; source_index=2/18; source=02-1.LazyLinear.md; part=1/1 -->

# LazyLinear: 첫 배치가 결정하는 입력 차원과 가변 해상도의 함정

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-07-18 |
| 회차·수준 | 1회차 · 기초 |
| 현재 소스 | 2/18 · `02-1.LazyLinear.md` |
| Part | 1/1 |
| 이전 소스 | `02.ClassificationForHands.md` |
| 다음 소스 | `02-02.AdaptiveAvgPooll2d.md` |
| 예상 학습 시간 | 100~140분 |
| 실행 검증 환경 | Python 3.12.12 · PyTorch 2.13.0 · C++17 · .NET 8 이상 |

## 1. 오늘 해결할 문제

CNN의 합성곱 블록이 출력한 텐서를 완전 연결층에 넣으려면 보통 다음과 같이 작성한다.

```python
x = torch.flatten(x, start_dim=1)
x = nn.Linear(32 * 16 * 16, 64)(x)
```

여기에는 두 가지 서로 다른 문제가 숨어 있다.

1. 모델을 작성하는 시점에 `32 * 16 * 16`을 직접 계산해야 한다.
2. 첫 번째 입력과 다른 해상도가 들어오면 flatten 길이가 달라질 수 있다.

`nn.LazyLinear`는 첫 번째 문제를 해결한다. 첫 forward에서 입력의 마지막 차원을 관찰해 `in_features`를 정하기 때문이다. 그러나 첫 forward가 끝난 뒤에는 평범한 `Linear`처럼 입력 길이가 고정된다. 따라서 두 번째 문제, 즉 **실행 중 가변 해상도 지원 문제는 해결하지 못한다.**

오늘의 핵심 문장은 다음과 같다.

> Lazy initialization은 차원 결정을 늦출 뿐, 한 번 결정된 차원을 계속 바꾸는 동적 계층이 아니다.

## 2. 학습 목표

학습을 마치면 다음을 할 수 있어야 한다.

1. `Linear`의 차원 일치 조건을 행렬 곱으로 설명한다.
2. CNN feature map을 flatten했을 때 선형층의 파라미터 수와 메모리를 계산한다.
3. `LazyLinear`의 미초기화 상태와 첫 forward 이후 상태를 구분한다.
4. optimizer, checkpoint, 분산 학습, export 전에 초기화가 필요한 이유를 설명한다.
5. `LazyLinear`과 Global Average Pooling 가운데 요구사항에 맞는 방식을 선택한다.
6. Python, C++, C#에서 같은 지연 초기화 계약을 구현하고 테스트한다.

## 3. 선수 지식

### 3.1 텐서 shape 표기

CNN feature map은 보통 NCHW layout을 사용한다.

$$
X \in \mathbb{R}^{N \times C \times H \times W}
$$

- $N$: batch 크기
- $C$: 채널 수
- $H$: 높이
- $W$: 너비

`torch.flatten(x, start_dim=1)`은 batch 축을 보존하고 나머지 축을 하나로 합친다.

$$
(N,C,H,W) \longrightarrow (N,F),
\qquad F=CHW
$$

`start_dim=0`을 사용하면 batch 축까지 섞이므로 분류 모델에서는 대부분 잘못된 선택이다.

### 3.2 행렬 곱의 안쪽 차원

두 행렬 $A \in \mathbb{R}^{m \times n}$과 $B \in \mathbb{R}^{n \times p}$를 곱하려면 안쪽 차원 $n$이 같아야 한다.

$$
AB \in \mathbb{R}^{m \times p}
$$

선형층도 같은 규칙을 따른다.

## 4. Linear를 수학으로 이해하기

입력 batch와 선형층 파라미터를 다음과 같이 두자.

$$
X \in \mathbb{R}^{N \times F},
\qquad
W \in \mathbb{R}^{O \times F},
\qquad
b \in \mathbb{R}^{O}
$$

- $F$: 입력 feature 수, 즉 `in_features`
- $O$: 출력 feature 수, 즉 `out_features`

PyTorch의 계산은 다음과 같다.

$$
Y=XW^{\mathsf{T}}+b
$$

따라서 출력 shape은

$$
Y \in \mathbb{R}^{N \times O}
$$

이다. $X$의 마지막 차원과 $W$의 두 번째 차원이 모두 $F$여야 한다. 첫 입력의 $F$가 8192여서 $W$가 $(64,8192)$로 만들어졌다면, 다음 입력의 $F$가 32768일 때 같은 행렬 곱을 수행할 수 없다.

### 4.1 파라미터 수

bias를 사용하는 선형층의 전체 파라미터 수는

$$
P=OF+O=O(F+1)
$$

이다. feature map이 `(N, 32, 16, 16)`이고 출력 뉴런이 64개라면

$$
F=32 \times 16 \times 16=8192
$$

$$
P=64(8192+1)=524{,}352
$$

이다. 가중치를 `float32`로만 저장해도 대략

$$
524{,}352 \times 4\ \text{bytes}
\approx 2.00\ \text{MiB}
$$

가 필요하다. Adam 계열 optimizer의 gradient와 두 상태 텐서까지 단순 합산하면 이보다 여러 배 큰 메모리가 필요하다.

해상도가 $64 \times 64$에서 $128 \times 128$로 두 배가 되고 합성곱 블록의 축소 비율이 같다면 $H$와 $W$가 각각 두 배가 될 수 있다. 이때 $F=CHW$는 네 배가 되고 선형층 파라미터 수도 거의 네 배가 된다.

## 5. 지연 초기화의 상태 기계

`nn.LazyLinear(out_features)`는 생성 시점에 `out_features`만 안다. `in_features`와 weight의 실제 shape은 첫 입력이 들어올 때 정한다.

```text
[생성]
UninitializedParameter
        |
        | 첫 forward: 입력의 마지막 차원 F 관찰
        v
[materialize]
weight shape = (out_features, F)
        |
        | 이후 forward
        v
[고정된 Linear 계약]
모든 입력의 마지막 차원은 F여야 함
```

이 과정을 **materialization**, 즉 실제 shape과 저장 공간을 갖는 파라미터로 구체화하는 과정이라고 부른다.

### 5.1 첫 forward에서 일어나는 일

첫 입력이 $X \in \mathbb{R}^{N \times F}$이면 다음 순서로 생각할 수 있다.

1. 입력의 마지막 차원 $F$를 읽는다.
2. weight를 $(O,F)$ shape으로 materialize한다.
3. bias를 사용하면 bias를 $(O)$ shape으로 materialize한다.
4. 새 파라미터를 초기화한다.
5. 평범한 affine transformation을 계산한다.

PyTorch `Linear`의 기본 weight 초기화 범위는 결과적으로 fan-in $F$에 의존한다. 단순화하면 각 weight가 대략 다음 구간에서 초기화된다고 이해할 수 있다.

$$
W_{ij} \sim \mathcal{U}\left(-\frac{1}{\sqrt{F}},\frac{1}{\sqrt{F}}\right)
$$

즉, 입력 차원을 모르면 올바른 초기화 범위도 확정할 수 없다. 지연 모듈은 shape 결정과 초기화를 함께 늦춘다.

### 5.2 마지막 차원만 추론한다

`LazyLinear`은 입력 전체 shape이 아니라 **마지막 차원**을 `in_features`로 사용한다. 입력이 `(2, 5, 7)`이면 $F=7$이고 출력은 `(2, 5, O)`가 된다.

$$
(*,F) \longrightarrow (*,O)
$$

CNN에서는 먼저 `(N,C,H,W)`를 `(N,CHW)`로 flatten하므로 마지막 차원이 $CHW$가 된다.

## 6. Python 최소 실행 예제

다음 코드는 독립 실행 가능하다. 첫 forward 전후의 상태와 다른 feature 길이를 넣었을 때의 오류를 확인한다.

```python
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter


class LazyClassifier(nn.Module):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


torch.manual_seed(7)
model = LazyClassifier()

assert isinstance(model.classifier.weight, UninitializedParameter)

x64 = torch.randn(2, 3, 64, 64)
y64 = model(x64)

assert y64.shape == (2, 4)
assert model.classifier.in_features == 8 * 32 * 32
assert tuple(model.classifier.weight.shape) == (4, 8192)

try:
    model(torch.randn(2, 3, 128, 128))
except RuntimeError as error:
    print("expected shape error:", str(error).splitlines()[0])
else:
    raise AssertionError("다른 flatten 길이는 실패해야 한다")

print("output:", tuple(y64.shape))
print("materialized weight:", tuple(model.classifier.weight.shape))
```

예상 핵심 출력은 다음과 같다.

```text
expected shape error: mat1 and mat2 shapes cannot be multiplied ...
output: (2, 4)
materialized weight: (4, 8192)
```

오류 메시지의 세부 숫자와 표현은 PyTorch 버전에 따라 달라질 수 있으므로 테스트에서는 전체 문자열보다 오류 타입과 shape 불변조건을 검증하는 편이 안전하다.

## 7. shape을 손으로 추적하기

위 예제의 입력이 `(2, 3, 64, 64)`일 때 shape 흐름은 다음과 같다.

| 단계 | 연산 | 출력 shape | 설명 |
| --- | --- | --- | --- |
| 1 | 입력 | `(2, 3, 64, 64)` | NCHW |
| 2 | `Conv2d(3, 8, 3, padding=1)` | `(2, 8, 64, 64)` | 공간 크기 유지 |
| 3 | `MaxPool2d(2)` | `(2, 8, 32, 32)` | 높이·너비 절반 |
| 4 | `flatten(start_dim=1)` | `(2, 8192)` | $8 \times 32 \times 32$ |
| 5 | `LazyLinear(4)` | `(2, 4)` | 첫 입력에서 $F=8192$ 추론 |

입력이 `(2, 3, 128, 128)`이면 flatten 결과가 `(2, 32768)`이 된다. 이미 만들어진 weight는 `(4, 8192)`이므로 다음 곱의 안쪽 차원이 다르다.

$$
(2 \times 32768)(8192 \times 4)
$$

따라서 계산이 실패한다. 이것이 `LazyLinear`과 가변 해상도 지원을 구분해야 하는 이유다.

## 8. 더 안전한 초기화 패턴

운영 코드에서는 실제 학습을 시작하기 전에 대표 shape의 더미 batch로 모델을 명시적으로 materialize하는 편이 좋다.

```python
import torch


def initialize_with_example(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
) -> None:
    model.to(device)
    example = torch.zeros(input_shape, device=device)
    training = model.training
    model.eval()
    with torch.no_grad():
        model(example)
    model.train(training)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LazyClassifier(num_classes=4)
initialize_with_example(model, (1, 3, 64, 64), device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

assert all(
    not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
    for parameter in model.parameters()
)
```

이 패턴의 장점은 다음과 같다.

- shape 오류가 첫 실제 batch보다 앞에서 발생한다.
- 파라미터 수와 모델 summary를 안전하게 계산할 수 있다.
- optimizer, 분산 wrapper, compile, export 전에 모델 구조를 확정할 수 있다.
- checkpoint의 weight shape을 명확히 기록할 수 있다.

더미 입력은 실제 전처리 파이프라인과 같은 채널 수, dtype, device, 해상도를 사용해야 한다. Batch Normalization이 있다면 더미 입력으로 running statistics가 오염되지 않도록 `eval()`과 `torch.no_grad()`를 사용하고 원래 train/eval 상태를 복원한다.

### 8.1 optimizer를 먼저 만들어도 되는가

현재 PyTorch의 많은 optimizer는 `UninitializedParameter` 객체가 materialize될 때 같은 파라미터 객체를 계속 추적할 수 있다. 그러므로 “optimizer를 먼저 만들면 언제나 실패한다”는 주장은 정확하지 않다.

하지만 다음과 같은 shape 의존 작업은 초기화 전 실패하거나 잘못된 가정을 할 수 있다.

- 파라미터별 `numel()` 집계
- shape 기반 optimizer parameter group 구성
- 일부 외부 모델 summary 또는 pruning 도구
- 분산 학습 wrapper와 graph capture
- ONNX 또는 다른 정적 graph export

따라서 팀 규칙으로 **materialize 후 optimizer 및 실행 wrapper 구성** 순서를 정하면 버전과 도구 차이에서 오는 위험을 줄일 수 있다.

## 9. 수작업 affine 검증

작은 숫자로 `Linear`의 계산을 직접 검증해 보자.

$$
X=
\begin{bmatrix}
1 & 2 & 3
\end{bmatrix},
\qquad
W=
\begin{bmatrix}
1 & 0 & -1 \\
0.5 & 0.5 & 0.5
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
1 & -1
\end{bmatrix}
$$

첫 번째 출력은

$$
1(1)+2(0)+3(-1)+1=-1
$$

이고, 두 번째 출력은

$$
1(0.5)+2(0.5)+3(0.5)-1=2
$$

이다.

```python
import torch
from torch import nn

x = torch.tensor([[1.0, 2.0, 3.0]])
layer = nn.Linear(3, 2)

with torch.no_grad():
    layer.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]))
    layer.bias.copy_(torch.tensor([1.0, -1.0]))

expected = torch.tensor([[-1.0, 2.0]])
torch.testing.assert_close(layer(x), expected)
print(layer(x))
```

`LazyLinear`의 특별한 부분은 이 affine 계산이 아니라 첫 입력으로 weight의 두 번째 차원을 정하는 준비 단계다. 초기화가 끝난 뒤 계산 자체는 `Linear`와 같다.

## 10. 학습 가능한 완전한 Python 예제

다음 예제는 합성 데이터 한 batch로 forward, loss, backward, optimizer step을 실행한다.

```python
import torch
from torch import nn


class TrainableLazyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


torch.manual_seed(17)
model = TrainableLazyCNN(num_classes=5)

# 구조 확정 단계
with torch.no_grad():
    warmup_logits = model(torch.zeros(1, 3, 32, 32))
assert warmup_logits.shape == (1, 5)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

images = torch.randn(4, 3, 32, 32)
targets = torch.tensor([0, 1, 2, 3])

optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()

assert logits.shape == (4, 5)
assert torch.isfinite(loss)
print(f"loss={loss.item():.6f}")
```

이 예제의 warm-up과 실제 학습 입력은 모두 `32 x 32`다. 실제 서비스가 여러 해상도를 받는다면 전처리에서 resize/crop으로 통일하거나, 뒤에서 설명할 Global Average Pooling 같은 구조를 사용해야 한다.

## 11. C++17로 지연 초기화 계약 구현하기

아래 코드는 외부 딥러닝 라이브러리 없이 컴파일할 수 있는 교육용 구현이다. 첫 입력의 feature 수로 weight를 만들고, 이후 feature 수가 바뀌면 즉시 예외를 던진다. 자동 미분이나 고성능 행렬 연산을 제공하는 운영용 계층은 아니다.

```cpp
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

class LazyDense {
public:
    explicit LazyDense(std::size_t out_features)
        : out_features_(out_features) {
        if (out_features_ == 0) {
            throw std::invalid_argument("out_features must be positive");
        }
    }

    std::vector<float> forward(const std::vector<float>& input) {
        if (!initialized_) {
            initialize(input.size());
        }
        if (input.size() != in_features_) {
            throw std::invalid_argument("input feature size changed after initialization");
        }

        std::vector<float> output(out_features_, 0.0F);
        for (std::size_t out = 0; out < out_features_; ++out) {
            float value = bias_[out];
            for (std::size_t in = 0; in < in_features_; ++in) {
                value += weight_[out * in_features_ + in] * input[in];
            }
            output[out] = value;
        }
        return output;
    }

    std::size_t in_features() const { return in_features_; }

private:
    void initialize(std::size_t in_features) {
        if (in_features == 0) {
            throw std::invalid_argument("input must not be empty");
        }
        in_features_ = in_features;
        weight_.resize(out_features_ * in_features_);
        bias_.resize(out_features_);

        std::mt19937 generator(7);
        const float bound = 1.0F / std::sqrt(static_cast<float>(in_features_));
        std::uniform_real_distribution<float> distribution(-bound, bound);
        for (float& value : weight_) value = distribution(generator);
        for (float& value : bias_) value = distribution(generator);
        initialized_ = true;
    }

    std::size_t in_features_ = 0;
    std::size_t out_features_;
    bool initialized_ = false;
    std::vector<float> weight_;
    std::vector<float> bias_;
};

int main() {
    LazyDense layer(2);
    const auto output = layer.forward({1.0F, 2.0F, 3.0F});
    assert(layer.in_features() == 3);
    assert(output.size() == 2);

    try {
        layer.forward({1.0F, 2.0F, 3.0F, 4.0F});
        return 1;
    } catch (const std::invalid_argument&) {
        std::cout << "shape guard passed\n";
    }
}
```

컴파일 명령은 다음과 같다.

```bash
c++ -std=c++17 -O2 lazy_dense.cpp -o lazy_dense
./lazy_dense
```

이 문서 작성 환경에서는 Apple Clang의 C++ 표준 라이브러리 헤더 설치가 불완전해 기본 `c++` 명령으로는 환경 오류가 발생했다. 동일 코드를 설치된 GCC 15의 `g++-15 -std=c++17 -O2`로 컴파일하고 실행해 `shape guard passed`와 종료 코드 0을 확인했다.

실무 LibTorch 모델에서는 가능한 경우 대표 입력으로 모듈을 초기화한 뒤 checkpoint를 저장하고, C++ 추론 서비스는 이미 shape이 확정된 weight를 읽게 하는 방식이 단순하다. 추론 요청마다 weight shape을 바꾸는 설계는 동시성, 메모리 할당, 모델 버전 관리에 불리하다.

## 12. C#으로 지연 초기화 계약 구현하기

다음 .NET 콘솔 예제도 첫 호출에서 입력 feature 수를 결정하고 이후 변경을 거부한다. 배열 기반 교육용 코드이며 TorchSharp의 자동 미분 계층을 대체하지 않는다.

```csharp
using System;

public sealed class LazyDense
{
    private readonly int _outFeatures;
    private readonly Random _random = new(7);
    private float[,]? _weight;
    private float[]? _bias;

    public LazyDense(int outFeatures)
    {
        if (outFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(outFeatures));
        _outFeatures = outFeatures;
    }

    public int? InFeatures { get; private set; }

    public float[] Forward(float[] input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (InFeatures is null)
            Initialize(input.Length);
        if (input.Length != InFeatures)
            throw new ArgumentException("Input feature size changed after initialization.");

        var output = new float[_outFeatures];
        for (var o = 0; o < _outFeatures; o++)
        {
            var value = _bias![o];
            for (var i = 0; i < InFeatures.Value; i++)
                value += _weight![o, i] * input[i];
            output[o] = value;
        }
        return output;
    }

    private void Initialize(int inFeatures)
    {
        if (inFeatures <= 0)
            throw new ArgumentException("Input must not be empty.");

        InFeatures = inFeatures;
        _weight = new float[_outFeatures, inFeatures];
        _bias = new float[_outFeatures];
        var bound = 1.0 / Math.Sqrt(inFeatures);

        for (var o = 0; o < _outFeatures; o++)
        {
            _bias[o] = NextWeight(bound);
            for (var i = 0; i < inFeatures; i++)
                _weight[o, i] = NextWeight(bound);
        }
    }

    private float NextWeight(double bound) =>
        (float)(_random.NextDouble() * 2.0 * bound - bound);
}

public static class Program
{
    public static void Main()
    {
        var layer = new LazyDense(2);
        var output = layer.Forward(new[] { 1.0F, 2.0F, 3.0F });

        if (layer.InFeatures != 3 || output.Length != 2)
            throw new InvalidOperationException("Initialization test failed.");

        try
        {
            layer.Forward(new[] { 1.0F, 2.0F, 3.0F, 4.0F });
            throw new InvalidOperationException("Shape mismatch was not detected.");
        }
        catch (ArgumentException)
        {
            Console.WriteLine("shape guard passed");
        }
    }
}
```

프로젝트 실행 예시는 다음과 같다.

```bash
dotnet new console --framework net8.0 --output LazyDenseDemo
# 위 C# 코드로 LazyDenseDemo/Program.cs를 교체
dotnet run --project LazyDenseDemo
```

Python 학습 모델을 C# 서비스로 옮길 때는 C#에서 계층을 다시 지연 초기화하기보다, Python에서 materialize하고 검증한 모델을 ONNX 등 합의된 형식으로 export하는 편이 weight 재현성과 장애 추적에 유리하다.

## 13. 언어와 프레임워크 사이의 계약

| 항목 | PyTorch Python | C++ 예제 | C# 예제 |
| --- | --- | --- | --- |
| 입력 표현 | 보통 `(N,F)` tensor | 한 sample의 `vector<float>` | 한 sample의 `float[]` |
| weight 논리 shape | `(O,F)` | row-major 1차원 `(O,F)` | 2차원 `[O,F]` |
| 초기화 시점 | 첫 forward | 첫 `forward` | 첫 `Forward` |
| dtype | 모델 설정에 따름 | `float` | `float` |
| shape 변경 | 행렬 곱 오류 | 명시적 예외 | 명시적 예외 |
| 자동 미분 | 지원 | 예제는 미지원 | 예제는 미지원 |

서로 다른 런타임에서 결과를 비교하려면 다음 조건을 고정해야 한다.

- weight와 bias 값을 파일에서 동일하게 로드한다.
- 입력 flatten 순서가 모두 같아야 한다.
- NCHW와 NHWC 변환 시 메모리 순서를 확인한다.
- `float32`와 `float64`를 섞지 않는다.
- 난수 생성기가 다르면 같은 seed라도 weight가 같다고 가정하지 않는다.

## 14. LazyLinear, 명시적 Linear, GAP 비교

| 선택 | 장점 | 한계 | 적합한 상황 |
| --- | --- | --- | --- |
| 명시적 `Linear(F,O)` | 구조가 즉시 확정되고 도구 호환성이 높음 | $F$를 직접 계산해야 함 | 입력 해상도가 고정된 운영 모델 |
| `LazyLinear(O)` | 실험 코드에서 $F$ 계산을 생략 | 첫 입력 이후 $F$ 고정, 초기화 전 도구 제약 | 빠른 prototype, 복잡한 feature extractor 연결 |
| GAP 후 `Linear(C,O)` | 공간 크기와 classifier 파라미터 수를 분리 | 위치 정보를 평균으로 압축 | 가변 해상도 분류, 현대 CNN head |
| Adaptive Pool 후 Linear | head 입력 공간 크기를 원하는 값으로 고정 | pooling bin의 의미와 정보 손실 검토 필요 | 고정 크기 head가 필요한 다양한 입력 |

### 14.1 의사결정 규칙

다음 질문을 순서대로 확인한다.

1. 서비스 입력 해상도가 항상 같은가?
2. 해상도가 다르면 전처리 resize/crop으로 통일해도 되는가?
3. 공간 위치 정보를 classifier가 직접 사용해야 하는가?
4. 모델 export와 여러 언어 배포가 중요한가?

입력 해상도가 고정이고 빠른 실험이 목적이면 `LazyLinear`이 편리하다. 입력 해상도가 달라질 수 있고 이미지 전체의 채널별 존재 증거가 중요하면 GAP가 더 자연스럽다. 공간 배치를 일정 격자로 보존해야 하면 `AdaptiveAvgPool2d((p,p))` 뒤에 명시적 `Linear`을 두는 방법을 검토한다.

## 15. 실무 실패 사례

### 15.1 작은 이미지로 우연히 materialize됨

문제: 데이터 검사 코드가 썸네일 `(1,3,32,32)`을 모델에 먼저 통과시켰다. 실제 학습 이미지는 `(N,3,224,224)`라 첫 학습 batch에서 행렬 곱 오류가 발생했다.

예방:

- 모델 초기화 함수를 한 곳에 두고 대표 입력 계약을 명시한다.
- 첫 forward 직후 `in_features`를 assertion으로 확인한다.
- 예상하지 않은 코드 경로에서 모델을 호출하지 못하게 초기화 단계를 분리한다.

### 15.2 rank마다 다른 shape으로 초기화됨

문제: 분산 학습에서 각 process가 서로 다른 해상도의 첫 batch를 받으면 rank별 weight shape이 달라질 수 있다.

예방:

- 분산 wrapper를 구성하기 전에 모든 rank에서 같은 shape으로 materialize한다.
- 모델 구조와 파라미터 shape hash를 rank 간 비교한다.
- 다중 scale 학습은 lazy head가 아니라 구조적으로 가변 shape을 지원하는 head를 사용한다.

### 15.3 checkpoint shape가 현재 전처리와 다름

문제: 과거 모델은 $64 \times 64$ 입력으로 초기화했지만 현재 전처리는 $96 \times 96$이다. 이름이 같은 weight라도 shape이 달라 load에 실패한다.

예방:

- checkpoint metadata에 입력 layout, 채널, 해상도, dtype, class 수를 기록한다.
- load 후 대표 입력 smoke test를 수행한다.
- `strict=False`로 오류를 숨기기 전에 누락·초과 key와 shape 차이를 보고한다.

### 15.4 모델 summary가 첫 forward를 대신 수행함

문제: 외부 summary 도구가 사용자가 예상하지 않은 shape으로 dummy forward를 실행해 lazy 계층을 먼저 초기화했다.

예방:

- summary 도구의 입력 shape을 코드 리뷰 대상에 포함한다.
- summary 전용 모델 인스턴스를 사용하거나 명시적 초기화 뒤 summary를 생성한다.
- materialize 전후의 parameter type과 shape을 테스트한다.

## 16. 체크포인트와 배포 체크

### 16.1 저장 전

- 모든 `UninitializedParameter`가 사라졌는지 확인한다.
- 대표 입력으로 forward가 성공하는지 확인한다.
- 입력 계약과 classifier `in_features`를 metadata에 기록한다.
- train/eval 모드와 dtype, device를 의도대로 설정한다.

### 16.2 로드 후

- missing key, unexpected key, shape mismatch를 모두 확인한다.
- 학습 때와 같은 flatten 순서인지 확인한다.
- batch 크기 1과 운영 batch 크기로 smoke test한다.
- 기준 Python 출력과 배포 런타임 출력을 허용 오차 안에서 비교한다.

### 16.3 export 전

ONNX와 정적 graph 도구는 실제 shape을 가진 파라미터를 필요로 한다. 다음 순서를 권장한다.

```text
모델 생성
  -> 대표 입력으로 materialize
  -> checkpoint load 또는 학습
  -> eval 모드
  -> 대표 입력으로 기준 출력 저장
  -> export
  -> 대상 런타임 출력 비교
```

## 17. 디버깅 체크리스트

- [ ] `torch.flatten(x, 1)`로 batch 축을 보존했는가?
- [ ] 첫 forward 입력이 실제 데이터 계약과 같은가?
- [ ] materialize 후 `in_features`와 weight shape을 기록했는가?
- [ ] 입력 해상도가 바뀔 가능성을 `LazyLinear`로 잘못 해결하려 하지 않았는가?
- [ ] 분산 wrapper, compile, export 전에 초기화했는가?
- [ ] checkpoint의 입력 해상도와 현재 전처리가 같은가?
- [ ] NCHW와 NHWC, RGB와 BGR을 구분했는가?
- [ ] C++ 또는 C# 런타임과 Python이 같은 weight와 flatten 순서를 사용하는가?
- [ ] 가변 해상도가 요구사항이면 GAP 또는 Adaptive Pool을 검토했는가?

## 18. 테스트 전략

최소한 다음 테스트를 자동화한다.

```python
import pytest
import torch


def test_lazy_classifier_materializes_expected_shape() -> None:
    model = LazyClassifier(num_classes=4)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 4)
    assert model.classifier.in_features == 8192


def test_lazy_classifier_rejects_new_resolution() -> None:
    model = LazyClassifier(num_classes=4)
    model(torch.randn(1, 3, 64, 64))
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 128, 128))


def test_batch_size_may_change() -> None:
    model = LazyClassifier(num_classes=4)
    model(torch.randn(1, 3, 64, 64))
    output = model(torch.randn(7, 3, 64, 64))
    assert output.shape == (7, 4)
```

batch 크기는 행렬 곱의 바깥 차원이므로 1에서 7로 바뀌어도 괜찮다. 반면 flatten feature 수는 안쪽 차원이므로 바뀌면 안 된다.

## 19. 연습문제

### 문제 1

feature map shape이 `(8, 24, 10, 10)`이고 `LazyLinear(6)`을 처음 통과한다. materialize된 weight와 출력 shape, bias 포함 파라미터 수를 구하라.

### 문제 2

첫 입력이 `(4, 3, 64, 64)`이고 합성곱 블록의 출력이 `(4, 16, 8, 8)`이다. 두 번째 입력에서 batch 크기만 10으로 바뀌면 실행 가능한가? 공간 크기가 `(16,16)`으로 바뀌면 실행 가능한가?

### 문제 3

입력 해상도가 다양하지만 classifier가 정확한 픽셀 위치보다 각 채널의 전체 활성 정도를 사용하면 된다. `LazyLinear`과 GAP 중 무엇이 더 적합한가?

### 문제 4

왜 서로 다른 언어에서 같은 seed만 설정하는 것으로 동일한 초기 weight를 보장할 수 없는가?

## 20. 해답

### 해답 1

flatten feature 수는

$$
F=24 \times 10 \times 10=2400
$$

이다. weight shape은 `(6, 2400)`, 출력 shape은 `(8, 6)`이다. 파라미터 수는

$$
6(2400+1)=14{,}406
$$

이다.

### 해답 2

batch 크기만 4에서 10으로 바뀌면 실행할 수 있다. feature 수 $16 \times 8 \times 8=1024$는 같기 때문이다. 공간 크기가 `(16,16)`이 되면 feature 수가 $4096$으로 바뀌므로 이미 `(O,1024)`로 초기화된 weight와 곱할 수 없다.

### 해답 3

GAP가 더 적합하다. GAP는 $H,W$ 전체를 평균 내어 각 채널을 하나의 값으로 만들므로 classifier 입력 길이가 $C$로 고정된다. `LazyLinear`은 첫 입력의 $CHW$를 고정할 뿐이다.

### 해답 4

언어와 라이브러리마다 난수 생성 알고리즘, 값 변환, 초기화 순서가 다를 수 있기 때문이다. 교차 런타임 동등성 테스트에서는 seed에 의존하지 말고 동일한 weight 파일을 로드해야 한다.

## 21. 핵심 요약

1. `Linear`은 $Y=XW^{\mathsf{T}}+b$이며 입력 마지막 차원과 weight의 `in_features`가 같아야 한다.
2. `LazyLinear`은 첫 forward까지 `in_features` 결정을 늦춘다.
3. 첫 forward 뒤에는 입력 feature 수가 고정되므로 가변 해상도를 직접 지원하지 않는다.
4. batch 크기는 바뀔 수 있지만 flatten 길이는 바뀌면 안 된다.
5. 대표 입력으로 명시적으로 materialize한 뒤 optimizer, 분산 wrapper, export를 구성하면 안전하다.
6. 가변 해상도 분류에는 GAP 또는 Adaptive Pool이 더 적합한 경우가 많다.
7. C++와 C# 배포에서는 학습 시 확정한 shape, weight, layout, dtype 계약을 그대로 유지해야 한다.

## 22. 다음 학습 예고

다음 편에서는 `02-02.AdaptiveAvgPooll2d.md`를 바탕으로 `AdaptiveAvgPool2d`가 입력 크기와 무관하게 목표 출력 격자를 만드는 원리를 다룬다. 단순히 “알아서 크기를 맞춘다”에서 멈추지 않고, 각 출력 bin의 시작·끝 index를 계산하고, 겹치는 bin이 생기는 이유와 일반 평균 풀링과의 차이를 Python, C++, C# 코드로 검증한다.

## 23. 원본 연결과 정정 사항

- 학습 원본: `dacon/05.ImageClassification/02-1.LazyLinear.md`
- 이전 문서: `[2026-07-18] CNN의 출발점 - 2D 합성곱과 출력 크기.md`
- 원본의 핵심인 차원 일치, `LazyLinear`, GAP 비교를 유지하면서 초기화 상태, optimizer, checkpoint, 분산 학습, export 계약을 확장했다.
- 원본의 “Lazy Module은 모델 생성 직후에는 파라미터가 비어 있다”는 표현은 저장 공간이 구체화되지 않은 `UninitializedParameter` 상태라는 뜻으로 구체화했다.
- 원본에서 GAP를 FCN의 특성과 동일시할 수 있는 대목은 구분해서 읽어야 한다. GAP는 공간 축을 줄이는 연산이고, FCN은 일반적으로 공간 위치별 예측을 유지하는 완전 합성곱 구조를 뜻할 수도 있어 문맥에 따라 의미가 다르다.
