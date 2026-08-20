<!-- curriculum: cycle=2; level=implementation; source_index=10/18; source=02-09.ConvNeXt.md; part=1/1 -->

# ConvNeXt: layout과 stochastic depth를 재현하는 구현 계약

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-21 |
| 회차·수준 | 2회차·구현 |
| 현재 소스 | 10/18 `02-09.ConvNeXt.md` |
| Part | 1/1 |
| 이전 소스 | 9/18 `02-08.SwimTransformer.md`: 이동 창 stage 구현 |
| 다음 소스 | 11/18 `02-10.SSL(Self-Supervised-Learning.md`: FCMAE·GRN 구현 |

원본은 ResNet을 현대화한 흐름과 depthwise convolution, inverted bottleneck, GELU, LayerNorm을 소개한다. 1회차 문서는 각 연산의 직관과 `NCHW -> NHWC -> NCHW` shape를 설명했다. 이번 2회차는 그 설명을 반복하지 않고 ConvNeXt block, downsample, classifier, stochastic depth, 학습·평가 loop, 재현성, ablation과 배포 검증을 하나의 실행 계약으로 묶는다.

## 학습 목표

이 글을 마치면 다음을 구현하고 검증할 수 있다.

1. `groups=C`인 depthwise convolution과 channel MLP를 분리해 구현한다.
2. `LayerNorm(C)`가 실제로 정규화하는 축을 수식과 코드로 일치시킨다.
3. layer scale과 sample별 stochastic depth를 residual branch에 올바른 순서로 적용한다.
4. patchify stem, stage, downsample, global pooling을 연결한 작은 분류기를 만든다.
5. 홀수 입력을 포함한 tensor shape와 layout 변환을 단위 테스트한다.
6. 같은 seed의 loss history와 모든 parameter가 정확히 재현되는지 검사한다.
7. `3x3`과 `7x7` depthwise kernel의 parameter·정확도 ablation을 실행한다.
8. NumPy, PyTorch, C++와 C#에서 LayerNorm·layer scale의 golden 값을 맞춘다.
9. layout copy, AMP, export와 운영 실패를 release gate로 바꾼다.

## 선수 지식과 기호

convolution, residual connection, automatic differentiation, cross entropy와 optimizer의 기본 사용법을 알고 있다고 가정한다.

| 기호 | 정의 |
| --- | --- |
| $B$ | batch 크기 |
| $C$ | block 입력·출력 channel 수 |
| $H,W$ | feature map 높이와 너비 |
| $k$ | depthwise kernel 한 변 |
| $r$ | channel MLP 확장 비율 |
| $K$ | class 수 |
| $p_l$ | $l$번째 block의 drop-path 확률 |
| $\gamma\in\mathbb{R}^{C}$ | channel별 layer-scale parameter |
| $\epsilon$ | LayerNorm 분모 안정화 상수 |

이미지와 convolution feature는 논리 shape `BCHW`, channel MLP 구간은 `BHWC`로 표기한다. PyTorch의 `channels_last` memory format과 논리적 `BHWC` shape는 서로 다른 개념이다.

## 1. 이번 회차의 구현 경계

ConvNeXt block의 수식은 짧지만 실제 구현은 다음 계약에서 자주 깨진다.

- `LayerNorm(C)`를 `BCHW`에 바로 적용해 너비 $W$를 정규화한다.
- `permute` 뒤 원래 layout으로 돌아오지 않아 residual 덧셈이 실패한다.
- `view`가 가능한지 확인하지 않고 non-contiguous tensor를 재해석한다.
- depthwise convolution의 `groups`를 빠뜨려 공간·channel mixing을 다시 결합한다.
- layer scale을 checkpoint와 다른 초기값으로 두어 초기 최적화 동역학을 바꾼다.
- stochastic depth를 sample이 아니라 pixel이나 channel마다 독립 적용한다.
- `eval()`을 잊어 validation과 export에서도 residual branch가 무작위로 사라진다.
- batch loss 평균을 다시 단순 평균해 마지막 작은 batch에 과도한 가중치를 준다.
- `channels_last`를 켰다는 사실만으로 빨라졌다고 결론 내리고 실제 copy와 kernel 시간을 재지 않는다.

이번 구현은 작은 2-stage 분류기로 핵심 계약을 검증한다. 논문의 전체 ConvNeXt-Tiny 규모를 복제하려는 코드는 아니지만 stem, downsample, block, head, 학습과 평가의 의미는 그대로 유지한다.

## 2. 원본과 1회차 설명에서 바로잡을 구현 세부

| 원본의 표현 또는 축약 | 이번 구현의 정확한 계약 |
| --- | --- |
| depthwise convolution은 self-attention과 거의 동일 | 공간·channel mixing을 분리한다는 설계 비유만 유사하다. convolution kernel은 입력과 무관하고 local이며 attention weight는 입력 의존적이다. |
| ConvNeXt-Tiny stage 비율은 `1:1:9:1` | 대표 depth는 `[3,3,9,3]`이고 이를 약분하면 `1:1:3:1`이다. 세 번째 stage에 9개 block이 있다는 사실과 비율을 혼동하면 안 된다. |
| depthwise convolution은 축소된 channel에서 실행 | ConvNeXt block은 $C$ channel에서 depthwise convolution한 뒤 channel MLP에서 $rC$로 확장한다. |
| `gamma`를 1로 초기화 | 대표 구현의 layer scale은 보통 $10^{-6}$으로 시작한다. 1은 기능상 가능하지만 pretrained recipe와 초기 동역학이 다르다. |
| LayerNorm은 한 이미지의 모든 channel pixel을 사용 | `BHWC` 구현의 `LayerNorm(C)`는 각 $(b,h,w)$ 위치에서 마지막 channel $C$개만 정규화한다. |
| BatchNorm은 작은 batch에서 극도로 불안정 | 작은 batch는 통계 잡음을 키울 수 있지만 항상 실패하는 것은 아니다. 핵심 차이는 BN이 batch·공간 통계와 running state를 사용한다는 점이다. |
| `7x7` kernel이 Swin window를 똑같이 모사 | 둘의 local 범위가 비슷할 뿐 weight 공유, 입력 의존성, channel 처리와 경계 동작은 다르다. |

정확한 원본 파일명을 curriculum 메타데이터에 유지하고, 모델 이름과 코드 식별자는 `ConvNeXt`로 쓴다.

## 3. block을 수식에서 코드 순서로 내리기

입력은 $X\in\mathbb{R}^{B\times C\times H\times W}$다. depthwise convolution은 channel별로 공간만 섞는다.

$$
Z_{b,c,i,j}
=
d_c+
\sum_{u=0}^{k-1}
\sum_{v=0}^{k-1}
D_{c,u,v}X_{b,c,i+u-p,j+v-p}
$$

여기서 $D\in\mathbb{R}^{C\times1\times k\times k}$이고 `groups=C`다. stride 1, dilation 1, $p=(k-1)/2$인 홀수 kernel이면 shape는 보존된다.

$$
Z\in\mathbb{R}^{B\times C\times H\times W}
$$

`LayerNorm(C)`와 `Linear`를 적용하려고 마지막 축을 channel로 옮긴다.

$$
U=\operatorname{permute}_{BCHW\rightarrow BHWC}(Z)
\in\mathbb{R}^{B\times H\times W\times C}
$$

### 3.1 LayerNorm

각 위치 $(b,h,w)$에서 평균과 분산을 계산한다.

$$
\mu_{b,h,w}
=
\frac{1}{C}\sum_{c=1}^{C}U_{b,h,w,c}
$$

$$
\sigma^2_{b,h,w}
=
\frac{1}{C}\sum_{c=1}^{C}
\left(U_{b,h,w,c}-\mu_{b,h,w}\right)^2
$$

$$
V_{b,h,w,c}
=
a_c
\frac{U_{b,h,w,c}-\mu_{b,h,w}}
{\sqrt{\sigma^2_{b,h,w}+\epsilon}}
+q_c
$$

$a_c$와 $q_c$는 학습 가능한 affine scale과 bias다. variance는 표본 분산이 아니라 $C$로 나누는 population variance다.

### 3.2 channel MLP와 layer scale

첫 linear가 $C$에서 $rC$로 확장하고 두 번째 linear가 다시 $C$로 줄인다.

$$
M
=
W_2\operatorname{GELU}(W_1V+b_1)+b_2
$$

$$
W_1\in\mathbb{R}^{rC\times C},
\qquad
W_2\in\mathbb{R}^{C\times rC}
$$

channel별 layer scale을 곱한 뒤 `BCHW`로 되돌린다.

$$
R
=
\operatorname{permute}_{BHWC\rightarrow BCHW}
(\gamma\odot M)
$$

초기 $\gamma_c=10^{-6}$이면 학습 초기에 block은 identity에 가깝다. 그렇다고 branch gradient가 모두 0인 것은 아니다. $\gamma$ 자체는 즉시 gradient를 받고, $\gamma$가 0이 아니라면 branch parameter에도 작은 gradient가 흐른다.

### 3.3 sample별 stochastic depth

keep 확률을 $q_l=1-p_l$라 하고 sample $b$의 mask를 $m_b\sim\operatorname{Bernoulli}(q_l)$라 하자.

$$
Y_b
=
X_b+
\frac{m_b}{q_l}R_b
$$

mask shape는 `(B,1,1,1)`이다. 따라서 같은 sample의 모든 channel과 pixel이 branch를 함께 유지하거나 함께 버린다. 학습 중 기대값은 보존된다.

$$
\mathbb{E}\left[\frac{m_b}{q_l}R_b\right]=R_b
$$

평가에서는 mask를 뽑지 않고 $Y=X+R$을 그대로 사용한다. dropout처럼 평가 시 다시 $q_l$을 곱하면 이중 보정이다.

## 4. stage와 분류기의 shape 추적

작은 검증 모델은 stem 뒤 두 stage를 사용한다. 입력은 `(B,3,32,32)`, channel은 `[16,32]`, stage depth는 `[1,1]`이다.

| 단계 | 연산 | 출력 shape |
| --- | --- | --- |
| 입력 | RGB batch | `(B,3,32,32)` |
| stem convolution | `4x4`, stride 4, padding 0 | `(B,16,8,8)` |
| stem LayerNorm | channel별 위치 정규화 | `(B,16,8,8)` |
| stage 0 | ConvNeXt block 1개 | `(B,16,8,8)` |
| downsample norm | `LayerNorm2d(16)` | `(B,16,8,8)` |
| downsample convolution | `2x2`, stride 2 | `(B,32,4,4)` |
| stage 1 | ConvNeXt block 1개 | `(B,32,4,4)` |
| global average | 공간 평균 | `(B,32)` |
| head norm | `LayerNorm(32)` | `(B,32)` |
| classifier | `Linear(32,K)` | `(B,K)` |

padding 없는 convolution의 출력 크기는 다음이다.

$$
H_{out}
=
\left\lfloor
\frac{H+2p-d(k-1)-1}{s}
\right\rfloor+1
$$

예를 들어 홀수 입력 `(B,3,19,23)`은 stem 뒤 `(B,16,4,5)`, downsample 뒤 `(B,32,2,2)`가 된다. 모델은 실행되지만 오른쪽과 아래의 남는 pixel이 사용되지 않을 수 있다. production preprocessing이 resize인지 pad인지 반드시 고정해야 한다.

## 5. NumPy golden: LayerNorm과 residual scale

다음은 실행 가능한 독립 검증이다. PyTorch를 사용하지 않으므로 정규화 축, variance 정의와 layer scale을 별도로 확인할 수 있다.

```python
import numpy as np

x = np.array([[[[1.0, 3.0]]]], dtype=np.float64)  # B,H,W,C
mean = x.mean(axis=-1, keepdims=True)
var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
norm = (x - mean) / np.sqrt(var + 1e-6)

gamma = np.array([0.5, 0.5], dtype=np.float64)
residual = x + gamma * norm

np.testing.assert_allclose(mean, 2.0, atol=0.0)
np.testing.assert_allclose(var, 1.0, atol=0.0)
np.testing.assert_allclose(norm, [[[[-0.9999995, 0.9999995]]]], atol=1e-7)
np.testing.assert_allclose(residual, [[[[0.50000025, 3.49999975]]]], atol=1e-7)
print(np.round(residual.reshape(-1), 6))
```

예상 출력은 `[0.5 3.5]`다. `axis=(1,2,3)`으로 바꾸면 이미지 전체를 정규화하는 다른 연산이 되며, 이 golden test가 실패해야 한다.

## 6. PyTorch 전체 구현

다음 코드는 실행 가능한 완전 예제다. 작은 합성 분류 문제를 만들고 두 모델을 학습·평가하며 단위 테스트와 exact replay를 수행한다.

```python
import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected BCHW, got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(mask_shape, device=x.device, dtype=x.dtype)
        mask.bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion: int = 4,
        layer_scale_init: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("shape-preserving kernel must be odd")
        hidden = expansion * channels
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pw1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class TinyConvNeXt(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dims: tuple[int, int] = (16, 32),
        depths: tuple[int, int] = (1, 1),
        kernel_size: int = 7,
        max_drop_path: float = 0.1,
    ):
        super().__init__()
        if len(dims) != len(depths):
            raise ValueError("dims and depths must have the same length")
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        total_blocks = sum(depths)
        rates = torch.linspace(0.0, max_drop_path, total_blocks).tolist()
        cursor = 0
        stages = []
        downsamples = []
        for stage_index, (channels, depth) in enumerate(zip(dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        channels,
                        kernel_size=kernel_size,
                        drop_path=rates[cursor],
                    )
                )
                cursor += 1
            stages.append(nn.Sequential(*blocks))
            if stage_index + 1 < len(dims):
                downsamples.append(
                    nn.Sequential(
                        LayerNorm2d(channels),
                        nn.Conv2d(channels, dims[stage_index + 1], 2, stride=2),
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return self.head_norm(x.mean(dim=(-2, -1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def make_dataset(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(count) % 2
    images = 0.15 * torch.randn(count, 3, 32, 32, generator=generator)
    for index, label in enumerate(labels.tolist()):
        if label == 0:
            images[index, :, :, 10:14] += 1.5
        else:
            images[index, :, 10:14, :] += 1.5
    return images, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    logits = model(images)
    loss = F.cross_entropy(logits, labels).item()
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def train_model(
    kernel_size: int,
    seed: int,
    epochs: int = 10,
) -> tuple[nn.Module, list[float], tuple[float, float]]:
    set_seed(seed)
    train_x, train_y = make_dataset(48, seed=101)
    valid_x, valid_y = make_dataset(16, seed=202)
    model = TinyConvNeXt(kernel_size=kernel_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    shuffle = torch.Generator().manual_seed(seed + 1)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=shuffle)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(order), 8):
            indices = order[start:start + 8]
            logits = model(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
            sample_count += len(indices)
        history.append(loss_sum / sample_count)
    return model, history, evaluate(model, valid_x, valid_y)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def assert_same_state(left: nn.Module, right: nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)


def run_contract_tests() -> None:
    set_seed(7)

    # 홀수 입력의 floor shape와 finite backward.
    model = TinyConvNeXt()
    odd = torch.randn(2, 3, 19, 23, requires_grad=True)
    logits = model(odd)
    assert logits.shape == (2, 2)
    logits.square().mean().backward()
    assert odd.grad is not None and torch.isfinite(odd.grad).all()

    # LayerNorm2d와 명시적 BHWC LayerNorm의 동등성.
    feature = torch.randn(2, 4, 5, 7, dtype=torch.float64)
    norm2d = LayerNorm2d(4).double()
    expected = F.layer_norm(
        feature.permute(0, 2, 3, 1),
        (4,),
        norm2d.weight,
        norm2d.bias,
        1e-6,
    ).permute(0, 3, 1, 2)
    torch.testing.assert_close(norm2d(feature), expected, rtol=0, atol=0)

    # gamma=0이면 평가 시 block은 정확한 identity.
    identity = ConvNeXtBlock(4, layer_scale_init=0.0, drop_path=0.5).eval()
    torch.testing.assert_close(identity(feature.float()), feature.float(), rtol=0, atol=0)

    # DropPath mask는 sample별 상수이며 token별 dropout이 아니다.
    drop = DropPath(0.5).train()
    dropped = drop(torch.ones(32, 3, 4, 5))
    per_sample = dropped.flatten(1)
    assert torch.equal(per_sample, per_sample[:, :1].expand_as(per_sample))
    assert set(torch.unique(dropped).tolist()).issubset({0.0, 2.0})
    assert len(torch.unique(dropped)) == 2

    # depthwise group 계약.
    first_block = model.stages[0][0]
    assert first_block.dwconv.groups == first_block.dwconv.in_channels
    assert first_block.dwconv.in_channels == first_block.dwconv.out_channels


run_contract_tests()

model_7a, history_7a, metrics_7a = train_model(kernel_size=7, seed=17)
model_7b, history_7b, metrics_7b = train_model(kernel_size=7, seed=17)
assert history_7a == history_7b
assert metrics_7a == metrics_7b
assert_same_state(model_7a, model_7b)

model_3, history_3, metrics_3 = train_model(kernel_size=3, seed=17)

print("contracts=PASS")
print(
    f"k7 loss={history_7a[0]:.6f}->{history_7a[-1]:.6f} "
    f"valid_loss={metrics_7a[0]:.6f} valid_acc={metrics_7a[1]:.3f} "
    f"params={parameter_count(model_7a)}"
)
print(
    f"k3 loss={history_3[0]:.6f}->{history_3[-1]:.6f} "
    f"valid_loss={metrics_3[0]:.6f} valid_acc={metrics_3[1]:.3f} "
    f"params={parameter_count(model_3)}"
)
print("exact_replay=PASS")
```

### 6.1 구현 선택의 이유

`LayerNorm2d`는 논리 shape를 `BHWC`로 바꿔 표준 `layer_norm`을 호출한다. `permute`는 view일 수 있지만, 다음 kernel이 요구하는 layout에 따라 backend가 copy를 삽입할 수 있다. 기능 검증과 profiler 검증을 분리해야 한다.

`DropPath`의 mask는 sample 차원만 다르고 나머지 축은 1이다. `nn.Dropout`으로 대체하면 element별 mask가 되어 다른 regularizer가 된다.

학습 loss는 `loss.item() * batch_size`를 누적한 뒤 전체 sample 수로 나눈다. 마지막 batch가 작아도 각 sample의 가중치가 같다.

validation은 `@torch.no_grad()`와 `model.eval()`을 모두 사용한다. 전자는 gradient graph를 만들지 않고, 후자는 stochastic depth 같은 module 동작을 평가 모드로 바꾼다. 둘 중 하나가 다른 하나를 대신하지 않는다.

## 7. 테스트를 실패시키며 디버깅하기

### 7.1 최소 단위 테스트 목록

| 테스트 | 입력 | 통과 조건 | 잡아내는 버그 |
| --- | --- | --- | --- |
| shape 보존 | block에 `(2,C,5,7)` | 출력 shape 동일 | padding·permute 오류 |
| odd input | model에 `(2,3,19,23)` | logits `(2,K)`와 finite gradient | stem·downsample floor 착각 |
| LN parity | 독립 `BHWC` 계산 | bitwise 또는 tight tolerance 일치 | 정규화 축 오류 |
| identity gate | `gamma=0` | 출력과 입력 exact match | layer-scale 위치 오류 |
| drop-path granularity | all-one batch | sample 안의 모든 값 동일 | element dropout 사용 |
| depthwise groups | 첫 block weight | `groups=in=out=C` | 일반 convolution 사용 |
| train/eval | 같은 입력 반복 | eval 동일, train은 mask에 따라 변화 가능 | `eval()` 누락 |
| exact replay | 같은 seed 두 학습 | history·state exact match | seed·shuffle·kernel 비결정성 |

### 7.2 음성 테스트

다음 항목은 실패해야 한다.

- 짝수 kernel로 shape-preserving block을 만들면 `ValueError`가 난다.
- `LayerNorm2d`에 3차원 tensor를 넣으면 `ValueError`가 난다.
- drop-path 확률이 1 이상이면 생성자에서 거부된다.
- 입력 channel이 stem의 3과 다르면 convolution이 실패한다.
- `dims`와 `depths` 길이가 다르면 stage 생성 전에 거부된다.

오류를 조용히 broadcast하거나 자동 reshape하지 않는 이유는 shape가 맞아 보이는 semantic bug가 checkpoint와 export까지 전파되는 것을 막기 위해서다.

### 7.3 흔한 증상별 추적 순서

| 증상 | 먼저 기록할 값 | 원인 후보 | 수정 |
| --- | --- | --- | --- |
| residual add shape 오류 | branch·residual의 shape와 stride | 마지막 `permute` 누락 | `BHWC -> BCHW` 복원 |
| loss가 seed마다 크게 다름 | seed, sample order, drop-path mask | shuffle generator 공유 실패 | generator와 seed를 checkpoint |
| validation이 흔들림 | `model.training` | `eval()` 누락 | 평가 진입 직후 assert |
| 학습 초반 정체 | `gamma`와 branch grad norm | 너무 작은 scale·낮은 LR | recipe 단위로 조정하고 기록 |
| pretrained 정확도 붕괴 | norm eps, preprocessing, state key | checkpoint 계약 불일치 | 모델 설정 hash 비교 |
| CPU 추론이 느림 | transpose·copy·depthwise kernel time | layout 변환 비용 | backend별 profile 후 선택 |
| AMP에서 NaN | logits, norm variance, grad norm | overflow·잘못된 loss scaling | finite gate와 scaler log |

## 8. ablation을 해석하는 법

위 코드는 같은 데이터, seed, optimizer와 epoch에서 `k=7`과 `k=3`만 바꾼다. 이것이 최소한의 통제 ablation이다.

depthwise convolution의 weight parameter 수는 bias를 빼면 다음과 같다.

$$
P_{dw}=Ck^2
$$

channel MLP의 weight parameter 수는 다음과 같다.

$$
P_{mlp}=C(rC)+(rC)C=2rC^2
$$

한 위치의 곱셈 누산 수도 같은 leading term을 가진다.

$$
\operatorname{MAC}_{block}
\approx
BHW\left(Ck^2+2rC^2\right)
$$

$C=96$, $r=4$일 때 `7x7` depthwise는 $4{,}704$ weight, 두 linear는 $73{,}728$ weight다. `7x7`을 `3x3`으로 바꾸면 공간 kernel parameter는 줄지만 channel MLP는 그대로다.

합성 stripe 데이터는 매우 쉬우므로 두 kernel이 모두 accuracy 1.0에 도달할 수 있다. 그 결과는 “두 kernel이 실전에서도 동등하다”가 아니라 **이 fixture가 receptive-field 차이를 식별하지 못한다**는 뜻이다. 실제 선택에는 다음 실험이 추가로 필요하다.

1. 작은 물체와 긴 공간 의존성을 포함한 validation set
2. class별 recall과 calibration
3. 같은 총 학습 step과 augmentation
4. parameter뿐 아니라 실제 latency와 peak memory
5. 최소 세 seed의 평균과 분산

## 9. C++ golden 구현

다음은 실행 가능한 검증용 C++17 예제다. production convolution engine이 아니라 `BHWC`의 한 위치에서 LayerNorm과 layer scale residual을 계산해 Python golden과 맞춘다.

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
  constexpr double eps = 1e-6;
  const std::array<double, 2> x{1.0, 3.0};
  const std::array<double, 2> gamma{0.5, 0.5};

  double mean = 0.0;
  for (double value : x) mean += value;
  mean /= static_cast<double>(x.size());

  double variance = 0.0;
  for (double value : x) variance += (value - mean) * (value - mean);
  variance /= static_cast<double>(x.size());

  std::array<double, 2> output{};
  for (std::size_t channel = 0; channel < x.size(); ++channel) {
    const double normalized = (x[channel] - mean) / std::sqrt(variance + eps);
    output[channel] = x[channel] + gamma[channel] * normalized;
  }

  assert(std::abs(output[0] - 0.50000025) < 1e-7);
  assert(std::abs(output[1] - 3.49999975) < 1e-7);
  std::cout << std::fixed << std::setprecision(6)
            << output[0] << ' ' << output[1] << '\n';
}
```

예상 출력은 `0.500000 3.500000`이다. variance를 `C-1`로 나누면 이 테스트가 실패한다.

## 10. C# golden 구현

다음 C# 예제도 실행 가능하며 같은 `double` 연산과 tolerance를 사용한다.

```csharp
using System;

public static class ConvNeXtGolden
{
    public static void Main()
    {
        const double epsilon = 1e-6;
        double[] x = { 1.0, 3.0 };
        double[] gamma = { 0.5, 0.5 };

        double mean = 0.0;
        foreach (double value in x)
        {
            mean += value;
        }
        mean /= x.Length;

        double variance = 0.0;
        foreach (double value in x)
        {
            double centered = value - mean;
            variance += centered * centered;
        }
        variance /= x.Length;

        double[] output = new double[x.Length];
        for (int channel = 0; channel < x.Length; channel++)
        {
            double normalized = (x[channel] - mean) / Math.Sqrt(variance + epsilon);
            output[channel] = x[channel] + gamma[channel] * normalized;
        }

        if (Math.Abs(output[0] - 0.50000025) >= 1e-7 ||
            Math.Abs(output[1] - 3.49999975) >= 1e-7)
        {
            throw new Exception("LayerNorm or layer-scale contract mismatch");
        }

        Console.WriteLine("{0:F6} {1:F6}", output[0], output[1]);
    }
}
```

두 언어 예제에서 `double`을 쓴 것은 손계산 tolerance를 엄격히 비교하기 위해서다. 실제 모델 weight가 `float32`라면 C++의 `float`, C#의 `System.Single`과 runtime별 tolerance를 별도로 정한다.

## 11. 프레임워크 간 shape·layout·dtype 계약

| 경계 | PyTorch eager | ONNX·일반 runtime | C++·C# 직접 배열 | 검증 항목 |
| --- | --- | --- | --- | --- |
| 이미지 입력 | 논리 `BCHW` | 보통 `NCHW` | API stride를 직접 정의 | RGB/BGR, resize, mean/std |
| depthwise weight | `(C,1,k,k)`, `groups=C` | `Conv`의 group 속성 $C$ | kernel index 직접 정의 | cross-correlation 방향 |
| block norm 입력 | `BHWC`로 permute | transpose가 graph에 남을 수 있음 | 마지막 channel contiguous 여부 | 정규화 축과 $\epsilon$ |
| pointwise mixing | `Linear(C,rC)` | `MatMul` 또는 fused op | row-major weight 계약 | weight transpose |
| layer scale | `(C,)` broadcast | reshape·broadcast | channel offset | residual 전에 적용 |
| drop path | train에서만 `(B,1,1,1)` | inference graph에서는 제거 | 보통 구현하지 않음 | export가 eval mode인지 |
| global average | `mean((-2,-1))` | `ReduceMean` | $H\times W$ loop | padding 포함 여부 |
| logits | `(B,K)` `float32` | `(B,K)` | flat `B*K` | class index mapping |

논리 shape가 같아도 physical stride는 다를 수 있다. PyTorch의 다음 두 tensor는 모두 `(B,C,H,W)`이지만 후자는 channels-last memory format일 수 있다.

```python
# 설명용: backend profile 전에 확인할 진단 코드
x = torch.randn(8, 96, 56, 56)
y = x.contiguous(memory_format=torch.channels_last)
print(x.shape, x.stride())
print(y.shape, y.stride())
```

`channels_last`는 shape를 `(B,H,W,C)`로 바꾸지 않는다. 반대로 `permute(0,2,3,1)`는 논리 축 순서를 바꾸지만 반드시 새로운 contiguous buffer를 만들지는 않는다.

## 12. 성능과 메모리

### 12.1 무엇을 측정할 것인가

다음 값을 같은 warm-up, batch, dtype, 입력 shape에서 측정한다.

- end-to-end p50·p95·p99 latency
- images per second
- peak allocated memory와 resident memory
- `permute` 뒤 발생한 contiguous copy 시간
- depthwise convolution과 두 pointwise linear의 kernel 시간
- preprocessing과 host-to-device transfer 시간

FLOPs가 적어도 작은 depthwise kernel을 backend가 비효율적으로 실행하면 wall-clock은 느릴 수 있다. layout 변환을 여러 block에서 반복하면 arithmetic 감소보다 copy가 더 클 수도 있다.

### 12.2 activation memory

channel MLP의 중간 activation shape는 `(B,H,W,rC)`다. 원소 수는 다음과 같다.

$$
A_{mlp}=BHWrC
$$

예를 들어 `float32`라면 이 tensor만 대략 $4BHWrC$ byte다. 학습에서는 backward를 위해 여러 activation을 보관하므로 실제 peak는 더 크다. gradient checkpointing은 저장을 줄이는 대신 forward 재계산을 늘린다.

### 12.3 stochastic depth schedule

전체 block 수가 $L$이고 최대 확률이 $p_{max}$라면 선형 schedule은 다음처럼 둘 수 있다.

$$
p_l
=
\frac{l}{L-1}p_{max},
\qquad
l=0,\ldots,L-1
$$

$L=1$이면 분모가 0이므로 구현에서는 확률 0 또는 명시한 단일 값을 사용해야 한다. 위 PyTorch 코드는 `torch.linspace`로 이 경계를 처리한다.

## 13. 수치 안정성과 mixed precision

LayerNorm은 분산이 0인 위치에서도 $\epsilon$ 덕분에 finite 값을 낸다. 하지만 dtype이 낮으면 mean·variance reduction의 반올림 오차가 커진다. AMP를 사용할 때 다음을 검사한다.

1. norm과 softmax 같은 민감한 reduction이 backend에서 충분한 accumulator precision을 쓰는가
2. `gamma`가 $10^{-6}$에서 `float16` 업데이트로 사라지지 않는가
3. loss scaler의 scale 감소 횟수와 skipped optimizer step이 증가하지 않는가
4. logits, loss, gradient norm에 NaN·Inf가 없는가
5. `float32` fixture와 AMP logits의 최대 절대·상대 오차가 허용 범위 안인가

`bfloat16`은 `float16`보다 exponent 범위가 넓지만 mantissa가 짧다. “AMP 사용” 하나로 묶지 말고 device, autocast dtype, runtime와 tolerance를 모델 카드에 기록한다.

## 14. 실무 실패 사례

### 14.1 학습은 되지만 checkpoint가 호환되지 않음

팀 A는 pointwise mixing을 `Linear`로, 팀 B는 `1x1 Conv2d`로 구현했다. 수학적으로 변환 가능하지만 weight shape는 각각 `(out,in)`과 `(out,in,1,1)`이다. 자동 key load만 믿으면 missing key 또는 잘못된 transpose가 생긴다.

대응은 명시적 변환 함수, 변환 전후 state schema hash와 고정 입력 logits parity다.

### 14.2 작은 validation batch에서 결과가 달라짐

LayerNorm에는 BatchNorm running statistic이 없으므로 batch 크기 자체가 같은 sample의 logits를 바꾸면 안 된다. 한 장 단독 logits와 큰 batch 안의 같은 장 logits가 다르면 preprocessing, stochastic augmentation, train mode 또는 batch 의존 custom op를 의심한다.

### 14.3 export했더니 latency만 증가

graph에 `Transpose -> MatMul -> Transpose`가 block마다 남고 runtime가 이를 fuse하지 못했다. FLOPs와 accuracy는 같지만 memory movement가 커졌다.

대응은 graph node 수만 보는 것이 아니라 profiler trace에서 copy와 kernel 시간을 분리하고, `Linear` 경로와 `1x1 Conv2d` 경로를 target backend에서 비교하는 것이다.

### 14.4 홀수 입력에서 가장자리 객체를 놓침

padding 없는 stem과 downsample이 나머지 행·열을 버렸다. 학습은 224 고정 crop이라 드러나지 않았지만 서비스의 동적 해상도에서 가장자리 정보가 사라졌다.

대응은 지원 입력을 stride 배수로 pad하고 valid region을 기록하거나, resize 정책을 고정한 뒤 가장자리 fixture를 회귀 테스트에 넣는 것이다.

### 14.5 재현 seed는 같지만 결과가 다름

Python, NumPy와 Torch seed는 고정했지만 data shuffle generator state, worker seed, augmentation RNG 또는 device의 비결정 kernel이 빠졌다.

대응은 seed만 저장하지 않고 epoch, sampler state, optimizer, scheduler, AMP scaler와 RNG state를 checkpoint에 넣는다. exact replay를 지원하지 않는 backend에서는 허용 tolerance와 통계 기준을 미리 정한다.

## 15. 배포와 모니터링 계약

### 15.1 export 전

- `model.eval()`을 호출해 drop path를 제거한다.
- 학습과 같은 RGB 순서, resize, crop, interpolation, mean/std를 고정한다.
- supported batch·height·width와 stride 배수 정책을 명시한다.
- class index와 label 문자열 mapping을 artifact에 포함한다.
- framework, opset, runtime, device와 dtype version을 기록한다.

### 15.2 parity gate

golden image 집합에 대해 eager와 target runtime를 비교한다.

$$
\Delta_{abs}
=
\max_i\left|z_i^{eager}-z_i^{runtime}\right|
$$

$$
\Delta_{rel}
=
\max_i
\frac{\left|z_i^{eager}-z_i^{runtime}\right|}
{\max\left(\left|z_i^{eager}\right|,\tau\right)}
$$

$\tau$는 0 근처에서 relative error가 폭주하지 않게 하는 작은 상수다. top-1 일치만으로는 작은 logit drift와 calibration 회귀를 놓칠 수 있으므로 absolute·relative error도 함께 본다.

### 15.3 운영 지표

- 입력 해상도·batch별 p50·p99 latency와 timeout
- CPU·GPU memory, OOM, runtime fallback 횟수
- NaN·Inf logits와 preprocessing 실패율
- class 분포, entropy, confidence와 calibration drift
- 이미지 밝기·해상도·aspect ratio drift
- model SHA, preprocessing version과 runtime version

canary에서 지표가 기준을 넘으면 이전 artifact로 되돌릴 수 있어야 한다. rollback은 code만이 아니라 model, label mapping과 preprocessing을 하나의 version 단위로 수행한다.

이 환경에는 `onnx`와 `onnxruntime` package가 설치되어 있지 않으므로 이 문서의 ONNX parity는 실행 검증하지 못했다. 대신 eager의 shape, gradient, exact replay와 독립 언어 golden을 검증하며, 실제 배포 전에는 target runtime에서 parity gate를 별도로 통과해야 한다.

## 16. 구현 체크리스트

- [ ] depthwise convolution이 `groups=C`이고 입력·출력 channel이 모두 $C$다.
- [ ] 홀수 kernel에서 padding이 $(k-1)/2$라 block shape가 보존된다.
- [ ] `LayerNorm(C)` 직전 논리 shape의 마지막 축이 channel이다.
- [ ] norm의 variance 분모와 $\epsilon$이 checkpoint recipe와 같다.
- [ ] pointwise expansion과 projection weight shape를 확인했다.
- [ ] layer scale이 residual 덧셈 전 branch에만 적용된다.
- [ ] drop-path mask shape가 `(B,1,1,1)`이고 eval에서는 비활성화된다.
- [ ] batch loss를 sample 수로 가중 평균한다.
- [ ] validation에서 `eval()`과 no-grad를 모두 사용한다.
- [ ] 홀수 입력, identity gate, finite backward와 exact replay를 테스트한다.
- [ ] `3x3`·`7x7` ablation에서 바뀐 변수가 kernel뿐인지 확인한다.
- [ ] layout별 latency와 copy 비용을 target device에서 측정한다.
- [ ] eager와 target runtime의 logits parity를 golden fixture로 검사한다.
- [ ] model·preprocessing·label mapping을 함께 versioning한다.

## 17. 연습문제

### 문제 1

$C=128$, $r=4$, $k=7$인 block에서 bias, norm과 layer scale을 제외한 depthwise convolution과 두 linear의 parameter 수를 구하라.

### 문제 2

입력 shape가 `(8,96,56,56)`일 때 depthwise convolution, `BHWC` 변환, 첫 linear, 두 번째 linear와 residual 합 직전 shape를 순서대로 쓰라. $r=4$다.

### 문제 3

drop-path 확률이 0.2일 때 살아남은 residual branch에 왜 $1/0.8$을 곱하는가? 학습 중 기대값으로 설명하라.

### 문제 4

`LayerNorm(96)`를 `(B,96,56,56)` tensor에 바로 호출하면 왜 실패하거나 잘못된 의미가 되는가?

### 문제 5

`3x3`과 `7x7` 모델이 합성 데이터에서 모두 accuracy 1.0을 얻었다. 이 결과만으로 kernel을 선택할 수 없는 이유와 추가할 측정 두 가지를 쓰라.

### 문제 6

validation 함수에 `torch.no_grad()`만 있고 `model.eval()`이 없다. 이 모델에서 어떤 문제가 생기는가?

## 18. 해답

### 해답 1

depthwise weight는 다음과 같다.

$$
128\cdot7^2=6{,}272
$$

두 linear weight는 다음과 같다.

$$
128\cdot512+512\cdot128=131{,}072
$$

channel MLP가 parameter 대부분을 차지한다.

### 해답 2

shape 순서는 `(8,96,56,56)`, `(8,56,56,96)`, `(8,56,56,384)`, `(8,56,56,96)`, 다시 `(8,96,56,56)`이다. 마지막 shape가 residual과 같아야 덧셈할 수 있다.

### 해답 3

$m\sim\operatorname{Bernoulli}(0.8)$이면 $\mathbb{E}[m]=0.8$이다. branch를 $mR/0.8$로 두면 기대값은 $0.8R/0.8=R$이 되어 eval branch와 scale이 맞는다.

### 해답 4

PyTorch `LayerNorm(96)`은 마지막 축 크기가 96이기를 기대한다. `BCHW`의 마지막 축은 56이므로 shape 오류가 난다. 우연히 $W=96$이면 실행되지만 너비를 정규화하는 semantic bug가 된다.

### 해답 5

쉬운 stripe fixture가 두 kernel의 receptive-field 차이를 요구하지 않았을 수 있다. 실제 validation의 class별 recall·calibration, 여러 seed, target device latency·memory와 더 긴 공간 의존성 데이터 등을 추가해야 한다.

### 해답 6

autograd graph는 만들지 않지만 module은 여전히 train mode다. stochastic depth가 무작위 branch mask를 적용하므로 같은 validation 입력의 logits와 metric이 흔들릴 수 있다.

## 핵심 요약

- ConvNeXt block은 `depthwise spatial mixing -> channel LayerNorm -> channel MLP -> layer scale -> stochastic depth -> residual` 순서다.
- `LayerNorm(C)`는 각 spatial 위치의 channel만 정규화하며 `BHWC` 축 계약이 핵심이다.
- layer scale은 대표 recipe에서 $10^{-6}$으로 시작하고, drop path는 sample별 mask를 쓴다.
- 완전한 구현은 block뿐 아니라 stem, downsample, global pooling, 학습·평가와 checkpoint 재현성을 포함한다.
- parameter 수는 대개 channel MLP가 지배하지만 실제 latency는 layout copy와 backend kernel 효율까지 profile해야 한다.
- 합성 ablation의 동률은 실전 동등성 증거가 아니라 fixture의 식별력 한계를 보여준다.
- 배포 전에는 `eval()` export, preprocessing version과 eager/runtime logits parity를 release gate로 둔다.

## 다음 학습 예고

다음은 `02-10.SSL(Self-Supervised-Learning.md`의 2회차 구현이다. FCMAE의 mask가 encoder의 active position과 loss denominator에 어떻게 연결되는지, GRN이 `NCHW`와 `NHWC`에서 같은 값을 내는지, pretraining·linear probing·fine-tuning을 어떻게 재현 가능한 실험으로 분리하는지 다룬다.
