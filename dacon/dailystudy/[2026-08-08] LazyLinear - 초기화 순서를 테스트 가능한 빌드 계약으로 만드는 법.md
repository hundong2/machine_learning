<!-- curriculum: cycle=2; level=implementation; source_index=2/18; source=02-1.LazyLinear.md; part=1/1 -->

# LazyLinear: 초기화 순서를 테스트 가능한 빌드 계약으로 만드는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-08 |
| 회차·수준 | 2회차 · 구현 |
| 현재 소스 | 2/18 · `02-1.LazyLinear.md` |
| Part | 1/1 |
| 이전 소스 | `02.ClassificationForHands.md` |
| 다음 소스 | `02-02.AdaptiveAvgPooll2d.md` |
| 예상 학습 시간 | 160~220분 |
| 실행 검증 환경 | Python 3.12.12 · PyTorch 2.13.0 · NumPy 2.3.5 · C++17 · C# |

## 1. 오늘의 구현 목표

1회차에는 `LazyLinear`이 첫 입력의 마지막 차원으로 가중치 shape을 정하며, 이후에는 그 shape이 고정된다는 원리를 배웠다. 이번에는 그 지식을 학습 시스템의 **빌드 계약**으로 만든다.

학습을 마치면 다음을 구현할 수 있어야 한다.

1. 대표 입력 사양으로 모든 lazy parameter를 명시적으로 materialize한다.
2. 잘못된 첫 batch가 모델 shape을 영구히 고정하지 못하게 한다.
3. materialize, checkpoint load, optimizer 생성, compile·export 순서를 코드로 강제한다.
4. `state_dict`를 새 lazy 모델에 로드할 때 생기는 상태 차이를 테스트한다.
5. flatten head와 GAP head를 같은 조건에서 ablation한다.
6. shape, dtype, parameter identity, optimizer state, checkpoint round trip을 단위 테스트한다.
7. Python, C++, C#의 초기화 방식과 layout 계약을 맞춘다.

오늘의 핵심은 다음 한 문장이다.

> Lazy module은 모델 설계를 대신하지 않는다. 입력 사양을 확정하는 시점을 생성자에서 검증된 build 단계로 옮길 뿐이다.

## 2. 선수 지식과 기호

합성곱 backbone의 출력을 다음과 같이 둔다.

$$
A \in \mathbb{R}^{N \times C \times H \times W}
$$

- $N$: batch 크기
- $C$: feature channel 수
- $H,W$: feature map의 공간 크기

flatten하면 다음 shape이 된다.

$$
X=\operatorname{flatten}(A,1)
\in \mathbb{R}^{N \times F},
\qquad
F=CHW
$$

출력 class 수가 $K$일 때 선형 분류기는 다음 affine transform이다.

$$
Z=XW^{\mathsf T}+b,
\qquad
W \in \mathbb{R}^{K \times F},
\qquad
b \in \mathbb{R}^{K}
$$

따라서 $X$의 마지막 차원 $F$를 모르면 $W$의 저장 공간과 초기화 범위를 정할 수 없다. `LazyLinear(K)`는 이 결정을 첫 입력까지 미룬다.

## 3. 원본과 1회차에서 한 단계 더 나아가기

원본은 수동 `in_features`, lazy initialization, GAP이라는 세 선택지를 소개한다. 1회차 문서는 첫 forward 이후 shape이 고정된다는 점, 파라미터 수, 분산·checkpoint·export 위험을 설명했다. 이번에는 다음 구현 질문에 답한다.

| 질문 | 이번 구현의 답 |
| --- | --- |
| 누가 첫 입력 shape을 결정하는가? | 임의의 train batch가 아니라 검증된 `InputSpec`이 결정한다. |
| lazy parameter가 남았는지 어떻게 아는가? | build 직후 전체 module을 순회해 `UninitializedParameter`가 0개인지 검사한다. |
| optimizer는 언제 만드는가? | 모델 build와 checkpoint load가 끝난 뒤 만든다. |
| resume 시 optimizer state는 언제 로드하는가? | 동일 topology와 parameter shape을 확정한 뒤 로드한다. |
| 다른 해상도를 받아야 하는가? | 요구사항이면 flatten head 대신 GAP·adaptive pooling을 선택한다. |
| export가 가능한가? | concrete weight, 고정 dtype·layout, 대표 입력으로 export smoke test를 통과해야 한다. |

원본의 “파라미터가 비어 있다”는 표현은 개념적으로는 맞지만 구현상 더 정확한 표현이 필요하다. PyTorch는 parameter 이름을 없애는 것이 아니라 `UninitializedParameter` placeholder를 등록한다. 첫 forward에서 같은 parameter 객체가 concrete storage를 얻을 수 있으므로, 단순히 `len(list(model.parameters()))`만 검사해서는 초기화 여부를 알 수 없다.

또한 원본의 GAP를 “공간 크기에 완벽히 독립적인 모델”이라고 부르는 것은 범위를 좁혀 읽어야 한다. head의 입력 feature 수는 해상도와 독립적이지만, 앞선 convolution의 최소 크기, stride, padding, positional encoding, 메모리 사용량까지 자동으로 독립적이 되는 것은 아니다.

## 4. 상태 기계로 보는 모델 수명 주기

학습 코드에서 lazy model은 다음 상태를 갖는다.

```text
CREATED
  lazy parameter 존재
      |
      | build(validated InputSpec)
      v
MATERIALIZED
  모든 parameter shape 확정
      |
      | checkpoint load 또는 새 초기화 유지
      v
WEIGHTS_READY
      |
      | optimizer / DDP / compile 구성
      v
TRAINABLE
      |
      | eval + export smoke test
      v
DEPLOYABLE
```

핵심 불변조건은 다음과 같다.

$$
\operatorname{count\_uninitialized}(M)=0
$$

이 조건이 거짓인 동안에는 parameter shape에 의존하는 다음 작업을 시작하지 않는다.

- layer별 learning rate와 weight decay group 구성
- optimizer state 생성 또는 복원
- DDP·FSDP 같은 분산 wrapper 구성
- graph compile과 정적 export
- parameter 수, FLOP, 메모리 보고서 생성

## 5. 잘못된 첫 batch가 만드는 영구 오류

다음 전처리 버그를 생각해 보자.

1. 설정 파일의 입력은 `(3, 64, 64)`이다.
2. 첫 train batch만 resize가 빠져 `(3, 80, 80)`으로 들어온다.
3. 두 번의 stride-2 pooling 뒤 feature map은 `(16, 20, 20)`이 된다.
4. `LazyLinear`은 $F=16\times20\times20=6400$으로 materialize된다.
5. 정상 batch의 feature map `(16,16,16)`은 $F=4096$이므로 두 번째 step에서 실패한다.

첫 batch에서 오류가 나지 않았다는 사실은 input contract가 맞다는 증거가 아니다. 오히려 잘못된 shape을 parameter schema로 저장할 수 있다.

따라서 build 입력은 `DataLoader`의 우연한 첫 batch가 아니라 다음 조건을 통과한 별도 tensor여야 한다.

- channel 수가 모델 계약과 같은가?
- 높이·너비가 양수이며 최소 downsampling 크기 이상인가?
- dtype이 학습 전 입력 dtype과 같은가?
- layout이 NCHW인가?
- device가 모델 parameter와 같은가?

## 6. shape을 단계별로 추적하기

오늘 사용할 backbone은 두 번의 stride-2 convolution을 사용한다. 입력이 `(N,3,32,40)`일 때 shape은 다음과 같다.

합성곱 출력 크기는 다음 식으로 계산한다.

$$
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+2p-d(k-1)-1}{s}+1
\right\rfloor
$$

너비도 같은 식을 쓴다. 여기서 kernel $k=3$, stride $s=2$, padding $p=1$, dilation $d=1$이다.

| 단계 | 연산 | 출력 shape |
| --- | --- | --- |
| 입력 | RGB NCHW | `(N, 3, 32, 40)` |
| block 1 | `Conv2d(3, 8, 3, stride=2, padding=1)` | `(N, 8, 16, 20)` |
| block 2 | `Conv2d(8, 16, 3, stride=2, padding=1)` | `(N, 16, 8, 10)` |
| flatten | `flatten(1)` | `(N, 1280)` |
| head | `LazyLinear(5)` | `(N, 5)` |

따라서 materialize된 weight shape은 `(5, 1280)`이고 bias를 포함한 head parameter 수는 다음과 같다.

$$
P_{\mathrm{flat}}
=5\times1280+5
=6405
$$

같은 backbone에 GAP를 적용하면 `(N,16,1,1) -> (N,16)`이므로 다음과 같다.

$$
P_{\mathrm{gap}}
=5\times16+5
=85
$$

이 설정에서 flatten head는 GAP head보다 다음 배수만큼 많은 parameter를 가진다.

$$
\frac{6405}{85}
\approx75.35
$$

## 7. NumPy로 affine과 gradient를 독립 검증하기

다음 코드는 **실행 가능한 독립 검증 예제**다. 프레임워크와 무관하게 affine forward와 weight gradient를 계산하고 finite difference로 확인한다.

```python
import numpy as np


def affine(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("expected x[N,F], weight[K,F], bias[K]")
    if x.shape[1] != weight.shape[1] or weight.shape[0] != bias.shape[0]:
        raise ValueError("affine shape mismatch")
    return x @ weight.T + bias


x = np.array([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=np.float64)
weight = np.array([[2.0, -1.0, 0.5], [-3.0, 2.0, 1.0]], dtype=np.float64)
bias = np.array([0.25, -0.5], dtype=np.float64)
target = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float64)


def loss_fn(w: np.ndarray) -> float:
    error = affine(x, w, bias) - target
    return float(np.mean(error**2))


prediction = affine(x, weight, bias)
error = prediction - target
analytic = (2.0 / error.size) * error.T @ x

epsilon = 1e-6
numeric = np.zeros_like(weight)
for row in range(weight.shape[0]):
    for col in range(weight.shape[1]):
        plus = weight.copy()
        minus = weight.copy()
        plus[row, col] += epsilon
        minus[row, col] -= epsilon
        numeric[row, col] = (loss_fn(plus) - loss_fn(minus)) / (2.0 * epsilon)

np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(prediction, [[4.5, -7.0], [-3.25, 4.5]])
print("prediction:", prediction.tolist())
print("max gradient error:", float(np.max(np.abs(analytic - numeric))))
```

이 검증은 `LazyLinear`의 상태 전이까지 검사하지는 않는다. 대신 materialize 이후 실제 계산이 어떤 수학을 따라야 하는지 독립적인 golden oracle을 제공한다.

## 8. PyTorch 모델 팩토리: build를 API로 만든다

다음 코드는 **실행 가능한 구현 예제**다. `InputSpec`을 검증하고, 임시 batch로 materialize한 뒤, lazy parameter가 남지 않았는지 검사한다.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter


@dataclass(frozen=True)
class InputSpec:
    channels: int
    height: int
    width: int
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        if self.channels != 3:
            raise ValueError("this model requires three RGB channels")
        if min(self.height, self.width) < 8:
            raise ValueError("height and width must be at least 8")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("build uses float32 or float64 input")


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, head_kind: str = "lazy_flatten") -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if head_kind not in {"lazy_flatten", "gap"}:
            raise ValueError("unsupported head_kind")

        self.head_kind = head_kind
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        if head_kind == "lazy_flatten":
            self.pool = nn.Identity()
            self.head = nn.LazyLinear(num_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(16, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("input must be NCHW")
        x = self.pool(self.features(x))
        return self.head(torch.flatten(x, 1))


def uninitialized_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]


@torch.no_grad()
def materialize(model: nn.Module, spec: InputSpec, device: torch.device) -> None:
    spec.validate()
    model.to(device=device, dtype=spec.dtype)
    was_training = model.training
    model.eval()
    dummy = torch.zeros(
        2,
        spec.channels,
        spec.height,
        spec.width,
        dtype=spec.dtype,
        device=device,
    )
    output = model(dummy)
    if output.ndim != 2 or output.shape[0] != 2:
        raise RuntimeError("classifier output must be [N,K]")
    remaining = uninitialized_names(model)
    if remaining:
        raise RuntimeError(f"uninitialized parameters remain: {remaining}")
    model.train(was_training)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for parameter in model.parameters():
        if parameter.requires_grad:
            yield parameter


def build_model(
    spec: InputSpec,
    num_classes: int,
    head_kind: str,
    device: torch.device,
) -> ImageClassifier:
    model = ImageClassifier(num_classes=num_classes, head_kind=head_kind)
    materialize(model, spec, device)
    return model


def save_checkpoint(
    path: Path,
    model: ImageClassifier,
    optimizer: torch.optim.Optimizer,
    spec: InputSpec,
    step: int,
) -> None:
    if uninitialized_names(model):
        raise RuntimeError("cannot save an unmaterialized model")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "input_spec": {
                "channels": spec.channels,
                "height": spec.height,
                "width": spec.width,
                "dtype": str(spec.dtype),
            },
            "head_kind": model.head_kind,
            "step": step,
        },
        path,
    )


torch.manual_seed(17)
device = torch.device("cpu")
spec = InputSpec(channels=3, height=32, width=40)
model = build_model(spec, num_classes=5, head_kind="lazy_flatten", device=device)

assert not uninitialized_names(model)
assert tuple(model.head.weight.shape) == (5, 1280)
assert model.head.in_features == 1280

optimizer = torch.optim.AdamW(trainable_parameters(model), lr=3e-3)
x = torch.randn(4, 3, 32, 40)
target = torch.tensor([0, 1, 2, 3])

optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = nn.functional.cross_entropy(logits, target)
loss.backward()
optimizer.step()

assert logits.shape == (4, 5)
assert torch.isfinite(loss)
assert len(optimizer.state) > 0
print("head:", tuple(model.head.weight.shape))
print("loss:", f"{loss.item():.6f}")
```

### 8.1 왜 `no_grad`를 쓰고 `inference_mode`를 쓰지 않는가

lazy parameter를 `torch.inference_mode()` 안에서 처음 materialize하면 새 weight 자체가 inference tensor가 될 수 있다. 실제 검증에서도 이후 학습 forward가 `Inference tensors cannot be saved for backward` 오류로 실패했다. 따라서 **학습할 lazy model의 build에는 `torch.no_grad()`를 사용한다.** 이미 concrete한 추론 전용 모델을 실행할 때의 `inference_mode()`와 parameter를 처음 만드는 build 단계를 구분해야 한다.

### 8.2 왜 dummy batch 크기를 2로 두는가

이 예제에는 BatchNorm이 없으므로 1도 가능하다. 그러나 실제 모델에 BatchNorm이 있고 build 중 실수로 train mode가 유지되면 batch 크기 1과 매우 작은 공간 크기가 오류나 오염된 running statistics를 만들 수 있다. build 함수는 `eval()`과 `inference_mode()`를 함께 사용하고 원래 mode를 복원한다.

더 엄격한 팀에서는 shape propagation 전용 API나 meta tensor를 사용할 수 있다. 다만 모든 custom operation이 meta device를 지원하는 것은 아니므로, 작은 concrete dummy input이 더 단순하고 신뢰할 만한 경우가 많다.

### 8.3 dtype과 device 순서

모델과 dummy input의 dtype·device가 달라지면 첫 forward 자체가 실패한다. mixed precision 학습을 하더라도 master parameter는 보통 `float32`로 materialize한 뒤 autocast 경계를 학습 step에 둔다. `float16` CPU 연산 지원이나 초기화 분산까지 우연에 맡기지 않는다.

## 9. checkpoint 복원 순서

가장 예측 가능한 resume 순서는 다음과 같다.

```text
checkpoint metadata 읽기
  -> config와 InputSpec 일치 검사
  -> 같은 model topology 생성
  -> InputSpec으로 materialize
  -> model state_dict 로드
  -> optimizer 생성
  -> optimizer state_dict 로드
  -> scheduler·scaler·step 복원
```

모델 state만 로드하는 경우에도 입력 사양 검사를 생략하지 않는다. weight shape이 맞더라도 channel order, normalization, class index mapping이 다르면 조용히 잘못된 예측을 만들 수 있다.

### 9.1 새 lazy model에 state_dict를 바로 로드할 때의 함정

현재 검증 환경인 PyTorch 2.13.0에서는 materialize된 `LazyLinear`의 state를 새 `LazyLinear`에 직접 로드하면 weight는 concrete `(K,F)`가 되지만 첫 forward 전 `in_features` 속성이 0으로 남았다. 같은 shape의 첫 forward 뒤에는 일반 `Linear`로 전환되며 `in_features=F`가 되었다.

이 동작은 버전별 구현 세부사항일 수 있다. 따라서 다음과 같은 취약한 검사를 피한다.

```python
# 취약한 코드: load 직후 버전에 따라 0일 수 있다.
assert model.head.in_features == expected_features
```

대신 build 후 load 순서를 사용하고, weight shape과 smoke forward를 모두 검사한다.

```python
model = build_model(spec, num_classes=5, head_kind="lazy_flatten", device=device)
result = model.load_state_dict(checkpoint["model"], strict=True)
assert not result.missing_keys and not result.unexpected_keys
assert tuple(model.head.weight.shape) == (5, 1280)
assert model(torch.zeros(2, 3, 32, 40)).shape == (2, 5)
```

## 10. optimizer 생성 시점과 parameter identity

PyTorch 2.13.0에서는 optimizer를 materialize 전에 만들어도 `UninitializedParameter` 객체가 같은 identity를 유지하며 concrete parameter로 바뀌었다. 따라서 “optimizer를 먼저 만들면 반드시 parameter를 놓친다”는 설명은 정확하지 않다.

그러나 안전한 순서를 늦출 이유도 없다.

- 이름·shape별 parameter group은 materialize 전에 만들기 어렵다.
- optimizer state checkpoint는 concrete shape을 전제로 한다.
- parameter 수와 decay 제외 규칙을 감사하기 어렵다.
- 외부 wrapper가 PyTorch와 같은 identity 보존을 보장하지 않을 수 있다.

따라서 구현 규칙은 다음과 같다.

> 가능 여부와 권장 순서를 구분한다. 현재 PyTorch에서 되는 경로보다 여러 도구에서 검증하기 쉬운 경로를 표준으로 삼는다.

## 11. 완전한 단위 테스트

다음 코드는 **`pytest`용 실행 가능한 테스트 예제**다. 이 문서의 검증 환경에는 `pytest` 패키지가 설치되어 있지 않아 동일 함수를 직접 호출하는 harness로 대체 검증한다.

```python
import copy
from pathlib import Path

import torch
from torch import nn


def test_build_materializes_expected_shape() -> None:
    spec = InputSpec(3, 32, 40)
    model = build_model(spec, 5, "lazy_flatten", torch.device("cpu"))
    assert uninitialized_names(model) == []
    assert tuple(model.head.weight.shape) == (5, 1280)
    assert model(torch.zeros(2, 3, 32, 40)).shape == (2, 5)


def test_wrong_resolution_fails_after_build() -> None:
    model = build_model(InputSpec(3, 32, 40), 5, "lazy_flatten", torch.device("cpu"))
    try:
        model(torch.zeros(2, 3, 36, 40))
    except RuntimeError as error:
        assert "cannot be multiplied" in str(error)
    else:
        raise AssertionError("flatten contract must reject a different feature length")


def test_gap_head_accepts_two_resolutions() -> None:
    model = build_model(InputSpec(3, 32, 40), 5, "gap", torch.device("cpu"))
    assert model(torch.zeros(2, 3, 32, 40)).shape == (2, 5)
    assert model(torch.zeros(2, 3, 36, 44)).shape == (2, 5)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(23)
    spec = InputSpec(3, 32, 40)
    source = build_model(spec, 5, "lazy_flatten", torch.device("cpu"))
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)

    x = torch.randn(3, 3, 32, 40)
    target = torch.tensor([0, 1, 2])
    loss = nn.functional.cross_entropy(source(x), target)
    loss.backward()
    optimizer.step()

    path = tmp_path / "lazy.pt"
    save_checkpoint(path, source, optimizer, spec, step=1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    restored = build_model(spec, 5, checkpoint["head_kind"], torch.device("cpu"))
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])

    source.eval()
    restored.eval()
    torch.testing.assert_close(source(x), restored(x), rtol=0.0, atol=0.0)
    assert checkpoint["step"] == 1
    assert len(restored_optimizer.state) == len(optimizer.state)


def test_materialization_is_reproducible() -> None:
    spec = InputSpec(3, 32, 40)
    torch.manual_seed(101)
    first = build_model(spec, 5, "lazy_flatten", torch.device("cpu"))
    first_state = copy.deepcopy(first.state_dict())

    torch.manual_seed(101)
    second = build_model(spec, 5, "lazy_flatten", torch.device("cpu"))
    for key, value in second.state_dict().items():
        torch.testing.assert_close(value, first_state[key], rtol=0.0, atol=0.0)
```

테스트의 목적은 framework 오류 문자열을 전부 고정하는 것이 아니다. 다음 불변조건을 고정하는 것이다.

- build된 weight shape
- 허용·거부해야 할 입력 shape
- checkpoint 전후 동일 logits
- optimizer state 복원
- 같은 seed와 같은 materialization 순서에서 같은 초기값

## 12. ablation: lazy flatten과 GAP를 공정하게 비교하기

head 선택은 convenience 비교가 아니라 모델 가설 비교다.

| 항목 | lazy flatten | GAP + explicit Linear |
| --- | --- | --- |
| 공간 정보 | 위치별 weight를 둘 수 있음 | 채널별 공간 평균으로 압축 |
| head parameter | $K(CHW+1)$ | $K(C+1)$ |
| 다른 해상도 | 같은 $CHW$가 아니면 실패 | backbone이 허용하면 가능 |
| 과적합 위험 | 작은 데이터에서 상대적으로 큼 | 상대적으로 작은 head |
| 위치 민감 분류 | 유리할 수 있음 | 정보 손실 가능 |
| export schema | build 후 고정 | 생성 시점부터 고정 |

공정한 ablation은 다음을 고정한다.

- 동일 train·validation split
- 동일 augmentation과 normalization
- 동일 backbone 초기값
- 동일 optimizer budget과 epoch 수
- 여러 seed의 평균과 표준편차
- accuracy뿐 아니라 macro F1, calibration, latency, peak memory

head parameter 수가 크게 다르므로 weight decay나 learning rate의 최적값도 다를 수 있다. 1회 비교에서 GAP가 이겼다고 구조적으로 항상 우월하다고 결론 내리지 않는다.

## 13. C++17: 명시적 materialize 계약

다음 코드는 **실행 가능한 표준 C++17 예제**다. 외부 tensor library 없이 지연 affine head의 상태 전이와 shape guard를 보여 준다. 실제 LibTorch 배포에서는 Python에서 이미 materialize·export한 모델을 읽는 방식을 우선한다.

```cpp
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

class LazyAffine {
public:
    explicit LazyAffine(std::size_t out_features)
        : out_features_(out_features), in_features_(0), initialized_(false) {
        if (out_features == 0) {
            throw std::invalid_argument("out_features must be positive");
        }
    }

    void materialize(std::size_t in_features) {
        if (in_features == 0) {
            throw std::invalid_argument("in_features must be positive");
        }
        if (initialized_) {
            if (in_features != in_features_) {
                throw std::invalid_argument("cannot rematerialize with another shape");
            }
            return;
        }
        in_features_ = in_features;
        weight_.assign(out_features_ * in_features_, 0.0);
        bias_.assign(out_features_, 0.0);
        initialized_ = true;
    }

    void set_parameters(const std::vector<double>& weight,
                        const std::vector<double>& bias) {
        if (!initialized_ || weight.size() != weight_.size() || bias.size() != bias_.size()) {
            throw std::invalid_argument("parameter shape mismatch");
        }
        weight_ = weight;
        bias_ = bias;
    }

    std::vector<double> forward(const std::vector<double>& x) const {
        if (!initialized_) {
            throw std::logic_error("materialize before forward");
        }
        if (x.size() != in_features_) {
            throw std::invalid_argument("input feature mismatch");
        }
        std::vector<double> y(out_features_, 0.0);
        for (std::size_t row = 0; row < out_features_; ++row) {
            y[row] = bias_[row];
            for (std::size_t col = 0; col < in_features_; ++col) {
                y[row] += weight_[row * in_features_ + col] * x[col];
            }
        }
        return y;
    }

private:
    std::size_t out_features_;
    std::size_t in_features_;
    bool initialized_;
    std::vector<double> weight_;
    std::vector<double> bias_;
};

int main() {
    LazyAffine head(2);
    head.materialize(3);
    head.set_parameters({2.0, -1.0, 0.5, -3.0, 2.0, 1.0}, {0.25, -0.5});
    const auto y = head.forward({1.0, -2.0, 0.5});
    if (std::abs(y[0] - 4.5) > 1e-12 || std::abs(y[1] + 7.0) > 1e-12) {
        throw std::runtime_error("golden output mismatch");
    }
    std::cout << y[0] << " " << y[1] << "\n";
}
```

C++의 `std::vector`는 연속 1차원 저장소다. weight를 row-major `(K,F)`로 펴서 `row * F + col`로 접근했다. 실제 ONNX Runtime이나 LibTorch tensor의 stride·contiguous 여부를 별도로 확인해야 한다.

## 14. C#: 명시적 materialize 계약

다음 코드는 **실행 가능한 C# 예제**다. C++ 예제와 같은 state machine과 golden output을 사용한다.

```csharp
using System;

public sealed class LazyAffine
{
    private readonly int outFeatures;
    private int inFeatures;
    private double[,] weight = new double[0, 0];
    private double[] bias = Array.Empty<double>();
    private bool initialized;

    public LazyAffine(int outFeatures)
    {
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
        this.outFeatures = outFeatures;
    }

    public void Materialize(int features)
    {
        if (features <= 0) throw new ArgumentOutOfRangeException(nameof(features));
        if (initialized)
        {
            if (features != inFeatures) throw new InvalidOperationException("shape is already fixed");
            return;
        }
        inFeatures = features;
        weight = new double[outFeatures, inFeatures];
        bias = new double[outFeatures];
        initialized = true;
    }

    public void SetParameters(double[,] newWeight, double[] newBias)
    {
        if (!initialized || newWeight.GetLength(0) != outFeatures ||
            newWeight.GetLength(1) != inFeatures || newBias.Length != outFeatures)
            throw new ArgumentException("parameter shape mismatch");
        weight = (double[,])newWeight.Clone();
        bias = (double[])newBias.Clone();
    }

    public double[] Forward(double[] x)
    {
        if (!initialized) throw new InvalidOperationException("materialize before forward");
        if (x.Length != inFeatures) throw new ArgumentException("input feature mismatch");
        var y = new double[outFeatures];
        for (int row = 0; row < outFeatures; row++)
        {
            y[row] = bias[row];
            for (int col = 0; col < inFeatures; col++)
                y[row] += weight[row, col] * x[col];
        }
        return y;
    }
}

public static class Program
{
    public static void Main()
    {
        var head = new LazyAffine(2);
        head.Materialize(3);
        head.SetParameters(
            new double[,] { { 2.0, -1.0, 0.5 }, { -3.0, 2.0, 1.0 } },
            new double[] { 0.25, -0.5 });
        double[] y = head.Forward(new double[] { 1.0, -2.0, 0.5 });
        if (Math.Abs(y[0] - 4.5) > 1e-12 || Math.Abs(y[1] + 7.0) > 1e-12)
            throw new Exception("golden output mismatch");
        Console.WriteLine($"{y[0]:F2} {y[1]:F2}");
    }
}
```

Python 학습 모델을 C#에서 다시 lazy initialize하면 random initialization, dtype, weight layout을 다시 맞춰야 한다. 실무에서는 Python에서 shape을 확정하고 checkpoint 또는 ONNX로 전달하며, C# 서비스는 입력 계약과 출력 golden만 검증하는 편이 단순하다.

## 15. 프레임워크 간 shape·layout·dtype 대응

| 경계 | Python·PyTorch | C++ | C#·서비스 |
| --- | --- | --- | --- |
| 이미지 입력 | `(N,C,H,W)` NCHW | runtime tensor descriptor 확인 | `DenseTensor<float>` 등의 dimensions 확인 |
| flatten | batch 축 보존 `flatten(1)` | stride가 연속인지 확인 후 reshape | 명시적 index 또는 runtime reshape |
| weight 논리 shape | `(K,F)` | row-major flat buffer 가능 | `double[K,F]` 또는 runtime tensor |
| affine | `X @ W.T + b` | loop·BLAS의 transpose flag 확인 | runtime `Gemm` 계약 확인 |
| 학습 dtype | 주로 `float32`, autocast 별도 | 보통 `float` | ONNX 입력은 흔히 `float` |
| 예제 golden dtype | NumPy `float64` | `double` | `double` |

ONNX의 `Gemm` node가 `transB=1`인지, exporter가 weight를 별도 transpose했는지는 graph를 보고 확인한다. shape이 `(K,F)`라는 논리 계약과 파일의 물리 storage order를 혼동하지 않는다.

## 16. 성능과 메모리

### 16.1 parameter와 optimizer 메모리

head parameter가 $P$개이고 모든 tensor를 `float32`로 둔다고 하자. SGD momentum은 대략 parameter, gradient, momentum buffer로 $12P$ bytes가 필요할 수 있다. AdamW는 parameter, gradient, first moment, second moment만 단순 합산해도 대략 $16P$ bytes다. mixed precision master weight나 allocator overhead는 별도다.

오늘의 flatten head $P=6405$는 작지만 고해상도 feature에서 빠르게 커진다. 예를 들어 $C=256$, $H=W=14$, $K=1000$이면 다음과 같다.

$$
P=1000(256\times14\times14+1)=50{,}177{,}000
$$

weight만 `float32`로 약 191.4 MiB다. GAP head는 $1000(256+1)=257{,}000$ parameter에 그친다.

### 16.2 materialization peak

첫 forward는 parameter 할당과 activation 할당을 함께 수행한다. 큰 실제 batch로 materialize하면 초기 peak memory가 불필요하게 커진다. 대표 shape는 유지하되 작은 batch와 `no_grad()`를 사용해 autograd graph를 만들지 않는다. 학습할 lazy weight를 처음 만드는 단계에는 `inference_mode()`를 사용하지 않는다.

### 16.3 compile cache와 dynamic shape

lazy parameter가 materialize된 뒤에도 입력 해상도가 바뀌면 backbone graph가 재compile되거나 guard failure가 날 수 있다. GAP가 head shape 오류를 없애도 compiler cache, kernel selection, activation memory의 변동은 남는다. 지원 해상도 bucket을 정하고 각 bucket을 benchmark한다.

## 17. 수치 안정성과 재현성

### 17.1 fan-in과 초기화 순서

선형층 초기화 scale은 fan-in $F$에 의존한다. 같은 seed를 써도 다른 shape으로 먼저 materialize하면 난수 소비량과 initialization bound가 달라진다. 모델 비교에서는 다음을 고정한다.

- module 생성 순서
- build input shape
- dtype와 device
- random seed 설정 시점
- build 이후 DataLoader worker seed

### 17.2 mixed precision

logits가 매우 크면 직접 `softmax` 후 `log`를 계산하는 구현이 overflow·underflow를 만들 수 있다. 학습에는 logits를 그대로 `cross_entropy`에 전달한다. AMP를 쓰더라도 optimizer state와 loss scaling 복원 여부를 checkpoint에 포함한다.

### 17.3 exact reproducibility의 범위

같은 CPU, 같은 PyTorch build, 같은 연산 순서에서는 exact match 테스트가 유용하다. 다른 GPU·driver·kernel까지 bitwise 동일성을 일반화하지 않는다. cross-device 테스트는 합리적인 `rtol`, `atol`과 metric 허용 범위를 별도로 정한다.

## 18. 테스트와 디버깅 순서

shape 오류가 나면 다음 순서로 범위를 좁힌다.

1. raw image가 HWC인지 CHW인지 확인한다.
2. collate 후 batch가 NCHW인지 확인한다.
3. backbone 각 stage의 `(N,C,H,W)`를 기록한다.
4. `flatten(1)` 직전과 직후 shape을 기록한다.
5. `head.weight.shape[1]`과 flatten 길이를 비교한다.
6. checkpoint metadata의 입력 사양과 현재 config를 비교한다.
7. model summary나 warm-up 요청이 먼저 materialize했는지 확인한다.

production log에는 전체 tensor 값을 남기지 않는다. 다음 schema만으로도 많은 장애를 진단할 수 있다.

```text
model_version, input_contract_version, request_shape,
request_dtype, feature_shape, head_weight_shape, device
```

민감한 이미지나 사용자 데이터는 shape 진단을 이유로 로그에 저장하지 않는다.

## 19. 실무 실패 사례

### 19.1 health check가 모델을 잘못 고정했다

서비스 시작 시 health check가 비용을 줄이려고 `(1,3,16,16)` dummy를 보냈다. 첫 실제 요청은 `(1,3,224,224)`였고 flatten head에서 실패했다.

교정은 health check와 model build를 분리하는 것이다. build는 versioned input contract를 사용하고, health check는 그 계약과 같은 shape으로 이미 build된 모델을 호출한다.

### 19.2 resume가 되었지만 class 의미가 바뀌었다

class 수 $K$가 같아 weight shape 검사는 통과했지만 label index 순서가 바뀌었다. 모델은 정상 shape의 잘못된 의미를 출력했다.

checkpoint에 class name 순서와 hash를 저장하고 현재 dataset manifest와 비교해야 한다. shape 검사는 semantic schema 검사를 대신하지 않는다.

### 19.3 GAP로 바꾼 뒤 작은 물체 성능이 하락했다

parameter와 latency는 줄었지만 위치가 중요한 손동작의 국소 신호가 전역 평균에서 희석되었다. adaptive pooling을 `(2,2)`로 두거나 attention pooling을 ablation하고, 입력 crop 정책도 함께 점검한다.

### 19.4 export만 별도 환경에서 실패했다

학습 process에서는 첫 batch가 이미 모델을 materialize했지만 export job은 새 모델을 만든 직후 export를 호출했다. export entry point가 암묵적 학습 이력에 의존한 것이 원인이다.

export command 자체가 `InputSpec -> build -> load -> smoke forward -> export -> runtime golden`의 전체 순서를 수행해야 한다.

## 20. 배포 관점

배포 artifact에는 적어도 다음을 함께 둔다.

- model weight와 format version
- 입력 layout, channel, 높이·너비 범위, dtype
- normalization mean·standard deviation과 RGB/BGR 순서
- class index와 class name mapping
- head 종류와 feature dimension
- framework·opset·runtime version
- 대표 입력·출력 golden fixture

ONNX package가 현재 검증 환경에 설치되어 있지 않아 이 문서에서는 실제 export를 수행하지 않았다. 대신 concrete parameter 검사, smoke forward, C++·C# 독립 affine golden으로 경계를 검증했다. 실제 배포 게이트에는 ONNX checker와 target runtime inference 비교가 추가되어야 한다.

## 21. 구현 체크리스트

- [ ] `InputSpec`이 version control과 checkpoint metadata에 있는가?
- [ ] build 입력이 임의의 첫 train batch와 분리되어 있는가?
- [ ] build 후 `UninitializedParameter`가 0개인지 검사하는가?
- [ ] head weight shape과 feature shape을 함께 검사하는가?
- [ ] checkpoint load 전에 topology와 입력 사양을 확인하는가?
- [ ] optimizer는 concrete model state가 준비된 뒤 만드는가?
- [ ] optimizer·scheduler·scaler state까지 resume하는가?
- [ ] flatten과 GAP를 여러 seed로 ablation했는가?
- [ ] 지원하는 모든 해상도를 smoke test하는가?
- [ ] export job이 독립적으로 build·load·검증하는가?
- [ ] C++·C# runtime의 layout과 dtype이 Python과 같은가?
- [ ] class index mapping을 shape와 별도로 검사하는가?

## 22. 연습문제

### 문제 1

입력 `(N,3,31,37)`이 오늘의 stride-2 convolution 두 개를 지난다. 각 stage와 flatten shape을 구하라.

### 문제 2

`LazyLinear(10)`이 $F=2048$로 materialize되었다. bias 포함 parameter 수와 `float32` weight+bias 용량을 구하라.

### 문제 3

checkpoint를 새 lazy model에 직접 로드했더니 weight shape은 맞지만 `in_features`가 0이다. 어떤 검사를 신뢰하고 어떤 복원 순서를 사용할 것인가?

### 문제 4

GAP head가 서로 다른 해상도를 받으면 shape 안전성이 완전히 보장되는가? 반례가 될 수 있는 backbone 조건 두 가지를 쓰라.

### 문제 5

같은 seed로 두 모델을 만들었지만 lazy weight가 다르다. 가능한 원인을 세 가지 쓰라.

## 23. 해답

### 해답 1

첫 convolution은 다음과 같다.

$$
31 \to \left\lfloor\frac{31+2-2-1}{2}+1\right\rfloor=16,
\qquad
37 \to 19
$$

두 번째는 $16\to8$, $19\to10$이다. 따라서 shape은 `(N,8,16,19)`, `(N,16,8,10)`, flatten `(N,1280)`이다.

### 해답 2

$$
P=10\times2048+10=20{,}490
$$

용량은 $20{,}490\times4=81{,}960$ bytes, 약 80.04 KiB다. gradient와 optimizer state는 포함하지 않은 값이다.

### 해답 3

버전별 lazy 속성 갱신 시점보다 concrete `weight.shape`, 미초기화 parameter 유무, smoke forward 출력을 신뢰한다. 더 예측 가능한 복원은 metadata 검사, 대표 입력으로 build, model state load, optimizer 생성과 state load 순서다.

### 해답 4

GAP 앞의 convolution·pooling이 작은 입력에서 0 크기 출력을 만들 수 있다. 또한 절대 positional embedding이나 고정 token 수를 요구하는 block이 backbone에 있으면 GAP 이전에 실패할 수 있다.

### 해답 5

build shape이 다르면 fan-in과 난수 소비량이 달라진다. seed 설정 전에 다른 random operation이 실행되었을 수 있다. module 생성·materialization 순서 또는 device·backend가 달라졌을 수도 있다.

## 24. 핵심 요약

1. `LazyLinear`은 입력 feature 설계를 없애지 않고 결정 시점을 첫 forward로 옮긴다.
2. 첫 train batch 대신 검증된 `InputSpec`으로 명시적으로 materialize한다.
3. build 후 모든 `UninitializedParameter`가 사라졌는지 검사한다.
4. 학습할 lazy weight의 build에는 `no_grad()`를 쓰며 `inference_mode()`를 피한다.
5. 권장 순서는 build, model state load, optimizer 생성, optimizer state load다.
6. checkpoint shape뿐 아니라 입력 전처리와 class mapping의 semantic schema도 검사한다.
7. flatten head는 위치별 표현력이 있지만 parameter와 해상도 결합이 크다.
8. GAP head는 작고 해상도에 유연하지만 backbone 제약과 정보 손실은 남는다.
9. NumPy, PyTorch, C++, C#에서 같은 affine golden을 검증하면 배포 경계를 추적하기 쉽다.

## 25. 다음 학습 예고

다음은 `02-02.AdaptiveAvgPooll2d.md`를 2회차 구현 수준으로 다룬다. 정수로 나누어떨어지지 않는 입력에서 bin의 시작·끝 index를 독립 구현하고, PyTorch forward·backward와 교차 검증한다. empty bin, overlap, dtype, ONNX export 차이와 property-based test까지 연결한다.

## 26. 원본 연결과 정정 기록

- 원본: [`02-1.LazyLinear.md`](../05.ImageClassification/02-1.LazyLinear.md)
- 1회차는 lazy 상태와 GAP 선택의 기초를 다뤘고, 이번 문서는 versioned build API, checkpoint round trip, optimizer state, 단위 테스트, ablation으로 확장했다.
- 원본의 “파라미터가 비어 있음”은 PyTorch의 등록된 `UninitializedParameter` placeholder라는 구현으로 구체화했다.
- GAP는 head의 공간 shape 의존성을 없애지만 전체 backbone의 입력 제약과 계산량까지 없애지는 않는다고 범위를 바로잡았다.
- PyTorch 2.13.0에서 새 lazy model에 state를 직접 로드한 직후 `in_features=0`이 남는 동작을 확인했으며, 버전 독립적인 build-then-load 패턴을 제시했다.
