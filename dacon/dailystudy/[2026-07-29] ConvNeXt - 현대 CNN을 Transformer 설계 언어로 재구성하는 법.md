<!-- curriculum: cycle=1; level=foundation; source_index=10/18; source=02-09.ConvNeXt.md; part=1/1 -->

# ConvNeXt - 현대 CNN을 Transformer 설계 언어로 재구성하는 법

## 학습 진도

| 날짜 | 회차·수준 | 현재 소스 | Part | 이전 소스 | 다음 소스 |
| --- | --- | --- | --- | --- | --- |
| 2026-07-29 | 1회차 · 기초 | 10/18 · `02-09.ConvNeXt.md` | 1/1 | `02-08.SwimTransformer.md` | `02-10.SSL(Self-Supervised-Learning.md` |

## 학습 목표

`ConvNeXt`는 self-attention을 CNN으로 바꾼 모델이 아니다. 저자들이 ResNet을 Swin Transformer 시대의 **훈련법과 설계 선택**으로 차례로 현대화해, 순수 합성곱도 강한 기준선이 될 수 있음을 보인 아키텍처다. 이 글을 마치면 다음을 할 수 있다.

- depthwise convolution과 pointwise channel mixing의 역할을 분리해 설명한다.
- ConvNeXt block의 모든 tensor shape와 `NHWC` 전환 이유를 추적한다.
- large kernel, inverted bottleneck, LayerNorm, layer scale, stochastic depth의 trade-off를 계산한다.
- PyTorch, C++, C#에서 채널별 2D convolution의 동일한 의미를 검증한다.
- 학습·추론·ONNX 배포에서 layout, 정규화, padding이 달라져 생기는 실패를 막는다.

## 선수 지식과 표기

입력 feature map은 기본적으로 $X \in \mathbb{R}^{N \times C \times H \times W}$, 즉 PyTorch의 `NCHW`다. 배치 크기는 $N$, 채널은 $C$, 공간 크기는 $H \times W$다. ConvNeXt 내부의 `LayerNorm(C)`와 `Linear`는 채널을 마지막 축에 두는 $NHWC$ 표현을 편리하게 사용한다. 이 글에서 $k$는 커널 크기, $r$은 채널 확장 비율이며 기본 block은 $k=7$, $r=4$다.

> 원문 교정: depthwise convolution이 multi-head self-attention과 “수학적으로 거의 동일”한 것은 아니다. 둘 다 공간 혼합과 채널 혼합을 분리한다는 **구조적 비유**는 맞지만, attention은 입력 의존적인 가중치로 먼 토큰을 섞고, depthwise convolution은 위치 공유 고정 커널로 국소 이웃을 섞는다.

## 1. ResNet을 현대화한 순서

ConvNeXt-Tiny의 대표적 큰 설계는 `4x4, stride 4` stem과 stage depth $[3,3,9,3]$이다. 초반 stem은 $224 \times 224$ 이미지를 $56 \times 56$으로 만들고, stage 사이의 `2x2, stride 2` downsample은 해상도를 절반·채널을 두 배로 바꾼다.

| 단계 | 입력 shape | 연산 | 출력 shape | 의도 |
| --- | --- | --- | --- | --- |
| stem | $(N,3,224,224)$ | $4 \times 4$, stride 4 convolution | $(N,96,56,56)$ | patchify와 같은 coarse tokenization |
| stage 1 | $(N,96,56,56)$ | block 3개 | $(N,96,56,56)$ | 얕은 공간 특징 |
| downsample | $(N,96,56,56)$ | LN + $2 \times 2$, stride 2 | $(N,192,28,28)$ | 계층적 표현 |
| stage 3 | $(N,384,14,14)$ | block 9개 | $(N,384,14,14)$ | 계산량 집중 |
| head | $(N,768,7,7)$ | global average pool + LN + linear | $(N,K)$ | 분류 logits |

`4x4` stem은 ViT의 `16x16` patch embedding을 그대로 모사하지 않는다. Swin의 초기 patch 크기와 비슷한 **해상도 감소율**을 택해 계층형 CNN의 시작점을 현대화한 것이다. 입력이 $H,W$의 배수가 아니면 stride convolution의 출력 크기는 다음과 같다.

$$
H_{out} = \left\lfloor \frac{H + 2p - d(k-1) - 1}{s} \right\rfloor + 1
$$

따라서 서비스 입력 크기를 고정하거나, preprocessing에서 resize·pad 정책을 모델 카드와 함께 고정해야 한다.

## 2. Depthwise convolution: 공간만 섞기

일반 convolution의 출력 $Y \in \mathbb{R}^{N \times C_{out} \times H \times W}$는 다음과 같다.

$$
Y_{n,o,i,j} = b_o + \sum_{c=1}^{C_{in}} \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} W_{o,c,u,v} X_{n,c,i+u-p,j+v-p}
$$

일반 $k \times k$ convolution의 parameter 수는 $k^2 C_{in} C_{out}$이다. depthwise convolution은 `groups=C`로 각 입력 채널에 커널 하나만 배정한다.

$$
Z_{n,c,i,j} = b_c + \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} D_{c,u,v} X_{n,c,i+u-p,j+v-p}
$$

여기서 parameter 수는 $k^2 C$다. $C_{in}=C_{out}=C$일 때 parameter 비는 $1/C$이지만, ConvNeXt의 block 전체 비용은 뒤의 $1 \times 1$ channel MLP가 지배한다. 그러므로 “block이 $1/C$로 싸다”는 결론은 틀리다.

### 손계산 검증

한 채널의 $3 \times 3$ 입력과 $2 \times 2$ kernel을 생각하자.

$$
X = \begin{bmatrix}1 & 2 & 3\\4 & 5 & 6\\7 & 8 & 9\end{bmatrix}, \quad
D = \begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}
$$

padding 없이 좌상단 출력은 $1\cdot1 + 2\cdot0 + 4\cdot0 + 5\cdot(-1)=-4$다. 두 채널이라면 이 계산을 채널별로 독립 수행하고, 아직 채널을 더하지 않는다.

```python
import numpy as np

x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
k = np.array([[1, 0], [0, -1]], dtype=np.float32)
out = np.array([
    (x[i:i + 2, j:j + 2] * k).sum()
    for i in range(2) for j in range(2)
], dtype=np.float32).reshape(2, 2)
assert np.array_equal(out, np.full((2, 2), -4, dtype=np.float32))
print(out)
```

실행 가능한 예제다. convolution 라이브러리가 수학적 convolution의 kernel flip 대신 cross-correlation을 쓰는 관례도 이 코드와 같다.

## 3. ConvNeXt block의 수식과 shape

하나의 block은 다음 순서다. $\operatorname{DWConv}_{7}$는 `NCHW`, 나머지 channel-wise 연산은 `NHWC`에서 보자.

$$
\begin{aligned}
Z &= \operatorname{DWConv}_{7}(X) &&\in \mathbb{R}^{N \times C \times H \times W} \\
U &= \operatorname{permute}(Z) &&\in \mathbb{R}^{N \times H \times W \times C} \\
V &= \operatorname{LN}(U) &&\in \mathbb{R}^{N \times H \times W \times C} \\
Q &= W_2\,\operatorname{GELU}(W_1V+b_1)+b_2 &&\in \mathbb{R}^{N \times H \times W \times C} \\
Y &= X + \operatorname{permute}^{-1}(\gamma \odot Q) &&\in \mathbb{R}^{N \times C \times H \times W}
\end{aligned}
$$

여기서 $W_1 \in \mathbb{R}^{C \times rC}$, $W_2 \in \mathbb{R}^{rC \times C}$, $\gamma \in \mathbb{R}^{C}$다. 이름이 `inverted bottleneck`인 까닭은 ResNet bottleneck의 축소-공간합성곱-확장과 달리, 채널을 먼저 $rC$로 확장한 뒤 다시 줄이기 때문이다. 단, 원문의 “축소된 상태에서 depthwise convolution”은 잘못이다. ConvNeXt는 **확장 전에** $C$ 채널에서 depthwise convolution을 한다.

### LayerNorm의 축

각 위치 $(n,h,w)$에서 채널 평균과 분산은 다음이다.

$$
\mu_{n,h,w}=\frac{1}{C}\sum_{c=1}^{C}U_{n,h,w,c}, \qquad
\sigma^2_{n,h,w}=\frac{1}{C}\sum_{c=1}^{C}(U_{n,h,w,c}-\mu_{n,h,w})^2
$$

$$
V_{n,h,w,c}=a_c\frac{U_{n,h,w,c}-\mu_{n,h,w}}{\sqrt{\sigma^2_{n,h,w}+\epsilon}}+b_c
$$

이는 “한 이미지의 모든 채널 픽셀”을 한꺼번에 정규화한다는 설명보다 정확하다. 이 구현에서 `LayerNorm(C)`는 **각 spatial location의 마지막 채널 축**만 정규화한다. BatchNorm은 학습 때 batch·공간 축의 통계를 쓰고 running statistic을 유지하므로, 작은 batch에서만이 아니라 train/eval 통계 불일치도 운영 위험이다.

## 4. large kernel은 attention의 복제품이 아니다

`7x7` depthwise kernel의 receptive field는 한 block에서 $7 \times 7$의 고정 이웃이다. 같은 해상도에서 stride 1 block $L$개를 쌓으면 이론적 receptive field는 대략 $1+6L$로 증가한다. 반면 global attention은 한 층에서 모든 $HW$ 위치를 볼 수 있고 비용은 대략 $O((HW)^2C)$다. depthwise convolution 비용은 $O(HWCk^2)$다.

| 방법 | 공간 가중치 | 한 층의 연결 범위 | 공간 비용 경향 | 적합한 경우 |
| --- | --- | --- | --- | --- |
| depthwise `7x7` | 학습 후 입력과 무관 | local $7x7$ | $HWCk^2$ | 모바일·고해상도 CNN |
| window attention | 입력 의존 | local window | $HWCw^2$ | 계층형 transformer |
| global attention | 입력 의존 | 전체 | $(HW)^2C$ | 짧은 token sequence |

ConvNeXt의 성과를 “7x7이면 attention이 된다”로 해석하면 안 된다. 학습 증강, optimizer, depth 배치, 정규화, residual 안정화가 함께 바뀐 통제 실험의 결과다.

## 5. PyTorch 구현과 실행 검증

아래 코드는 실행 가능한 최소 block이다. `channels_last` memory format은 선택적 최적화이며 tensor의 논리 shape를 `NHWC`로 바꾸는 `permute`와 다른 개념이다.

```python
import torch
from torch import nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)                    # N,C,H,W
        x = x.permute(0, 2, 3, 1)             # N,H,W,C
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        x = self.gamma * x
        return residual + x.permute(0, 3, 1, 2)

torch.manual_seed(7)
block = ConvNeXtBlock(8)
x = torch.randn(2, 8, 11, 13, requires_grad=True)
y = block(x)
assert y.shape == (2, 8, 11, 13)
loss = y.square().mean()
loss.backward()
assert torch.isfinite(x.grad).all()
assert block.dwconv.groups == 8
print(f"shape={tuple(y.shape)}, loss={loss.item():.6f}")
```

`gamma`를 작은 값으로 시작하는 layer scale은 깊은 네트워크 초기에 residual branch를 작게 만들어 최적화를 안정화한다. 값이 너무 작으면 초반 학습이 느려질 수 있으므로, pretrained checkpoint와 동일한 설정을 유지한다.

### NumPy와 PyTorch의 LayerNorm 교차 검증

```python
import numpy as np
import torch

a = np.array([[[[1., 3., 5.]]]], dtype=np.float32)  # N,H,W,C
mean = a.mean(axis=-1, keepdims=True)
var = ((a - mean) ** 2).mean(axis=-1, keepdims=True)
expected = (a - mean) / np.sqrt(var + 1e-6)
actual = torch.nn.LayerNorm(3, eps=1e-6)(torch.tensor(a)).detach().numpy()
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
print(actual.reshape(-1))
```

실행 가능한 예제다. 출력은 대략 `[-1.2247448, 0, 1.2247448]`이며, 축을 잘못 지정해 `NCHW`에 바로 `LayerNorm(C)`를 적용하면 마지막 축이 $W$라서 오류 또는 의미 변경이 발생한다.

## 6. C++과 C#의 channel-layout 대응

프레임워크 없이 검증할 때에는 메모리 layout을 명시해야 한다. 다음 두 예제는 단일 `N=1`, `C=2`, `H=W=3` NCHW 배열에 channel별 $2x2$ cross-correlation을 수행한다. 모두 `[-4, -4, -4, -4]`를 두 채널에 출력한다.

```cpp
#include <cassert>
#include <iostream>
#include <vector>

int main() {
  const int C = 2, H = 3, W = 3;
  std::vector<float> x(C * H * W);
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H * W; ++i) x[c * H * W + i] = float(i + 1 + 10 * c);
  const float k[4] = {1, 0, 0, -1};
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        float s = 0;
        for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v)
          s += x[c * H * W + (i + u) * W + j + v] * k[u * 2 + v];
        assert(s == -4.0f); std::cout << s << ' ';
      }
}
```

```csharp
using System;

class DepthwiseCheck {
  static void Main() {
    const int C = 2, H = 3, W = 3;
    var x = new float[C * H * W];
    for (int c = 0; c < C; c++) for (int i = 0; i < H * W; i++) x[c * H * W + i] = i + 1 + 10 * c;
    float[] k = { 1, 0, 0, -1 };
    for (int c = 0; c < C; c++) for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
      float s = 0;
      for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) s += x[c * H * W + (i + u) * W + j + v] * k[u * 2 + v];
      if (s != -4) throw new Exception("depthwise index mismatch");
      Console.Write(s + " ");
    }
  }
}
```

| 환경 | 논리 shape | 기본 layout 관례 | dtype | 주의점 |
| --- | --- | --- | --- | --- |
| PyTorch `Conv2d` | `N,C,H,W` | NCHW | `float32` | `groups=C`가 depthwise |
| PyTorch `LayerNorm` | `N,H,W,C` | channel-last logical view | `float32` 또는 AMP | `normalized_shape=C` |
| ONNX `Conv` | `N,C,H,W` | NCHW | `float32` 권장 | transpose를 명시적으로 export |
| C++/C# 배열 | flat buffer | 인덱스를 직접 정의 | `float` | stride와 channel offset을 테스트 |

이 C++·C# 예제는 실행 가능한 최소 검증용이며, production inference engine을 대체하지 않는다.

## 7. 테스트와 디버깅

다음 불변식을 CI의 작은 단위 테스트로 둔다.

```python
def test_convnext_contracts():
    block = ConvNeXtBlock(4)
    x = torch.randn(1, 4, 5, 7, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.mean().backward()
    assert torch.isfinite(x.grad).all()
    try:
        block(torch.randn(1, 3, 5, 7))
        raise AssertionError("channel mismatch must fail")
    except RuntimeError:
        pass

test_convnext_contracts()
print("shape and backward checks passed")
```

| 증상 | 흔한 원인 | 확인·수정 |
| --- | --- | --- |
| residual add가 실패 | `permute` 뒤 `N,H,W,C`를 되돌리지 않음 | add 전 두 shape를 log |
| 정확도가 train만 높음 | BatchNorm checkpoint를 LN 모델에 섞음 | state dict key와 norm type 검사 |
| ONNX 결과가 다름 | resize·padding 또는 layout transpose 차이 | 입력 한 장의 logits golden test |
| NaN | mixed precision에서 분산·loss 폭주 | finite check, grad clip, AMP scaler log |
| 느린 추론 | 잦은 `permute().contiguous()` 복사 | profiler로 transpose copy와 kernel time 분리 |

## 8. 성능, 메모리, 수치 안정성

pointwise MLP parameter는 $2rC^2$, depthwise parameter는 $k^2C$다. 예를 들어 $C=96$, $r=4$, $k=7$이면 각각 $73{,}728$와 $4{,}704$다. 따라서 “depthwise가 싸다”는 이점은 channel MLP를 없애는 근거가 아니라, large spatial kernel을 현실적인 비용에 쓰게 하는 근거다.

학습에서는 activation memory가 $NCHW$에 비례한다. gradient checkpointing은 메모리를 줄이는 대신 forward 재계산을 늘린다. inference에서는 batch와 입력 해상도를 고정할 수 있으면 compiler가 더 잘 fuse할 수 있다. 단, `channels_last`는 모든 CPU·GPU backend에서 항상 빠르지 않으므로 profile로 결정한다.

LayerNorm의 $\epsilon$은 0 분산에서 나눗셈을 안전하게 한다. 임의로 checkpoint보다 큰 `eps`로 바꾸면 작은 특징의 스케일이 달라지고 정확도가 이동할 수 있다. preprocessing의 RGB/BGR 순서, mean/std, interpolation도 같은 수준의 배포 계약이다.

## 9. 실무 실패 사례와 배포 관점

1. **잘못된 weight 이식**: `Conv2d`의 weight shape는 `(out,in/groups,kH,kW)`이고 `Linear`는 `(out,in)`이다. TensorFlow의 `HWIO`를 NCHW weight로 transpose하지 않으면 shape가 맞아도 의미가 틀릴 수 있다.
2. **동적 입력의 착각**: ConvNeXt 본체는 대개 가변 $H,W$를 통과시키지만 classifier의 global pooling·후처리·서빙 batcher가 고정 shape를 요구할 수 있다. 지원 해상도 집합과 pad 규칙을 API 계약에 명시한다.
3. **학습/서빙 preprocess 불일치**: center crop 224로 학습하고 단순 stretch로 서빙하면 객체 종횡비가 달라진다. 원본 이미지 hash, transform version, model SHA를 요청 로그에 남긴다.
4. **성능 회귀를 accuracy만으로 판단**: p50/p99 latency, peak memory, 입력 해상도별 throughput, class별 calibration을 release gate로 둔다.

배포 전에는 PyTorch eager와 exported runtime에 같은 RGB fixture를 넣어 top-1 class, logits 최대 절대오차, shape, dtype을 비교한다. 모델 교체에는 canary traffic, latency·error rate·confidence drift 대시보드, 빠른 rollback artifact가 필요하다.

## 체크리스트

- [ ] 입력이 `float32`, RGB 순서, 학습 mean/std와 일치한다.
- [ ] depthwise convolution의 `groups`가 channel 수와 같다.
- [ ] `LayerNorm(C)` 직전에는 마지막 축이 channel이다.
- [ ] residual 합의 양쪽 shape와 dtype이 정확히 같다.
- [ ] checkpoint의 `eps`, layer scale, stochastic depth 설정을 보존했다.
- [ ] exported ONNX/runtime를 golden input으로 eager와 비교했다.
- [ ] latency, peak memory, NaN/Inf, confidence drift를 운영 지표로 기록한다.

## 연습문제와 해답

### 문제 1

$C=64$, $k=7$, $r=4$ block에서 depthwise convolution과 두 Linear의 parameter 수를 각각 구하라. bias와 norm parameter는 무시한다.

### 해답 1

depthwise는 $7^2\cdot64=3{,}136$개다. MLP는 $64\cdot256+256\cdot64=32{,}768$개다. spatial convolution보다 channel MLP가 약 10.4배 크다.

### 문제 2

`x`가 `(2, 96, 56, 56)`일 때 `x.permute(0, 2, 3, 1)` 뒤 shape와 `LayerNorm(96)`의 정규화 축을 쓰라.

### 해답 2

shape는 `(2, 56, 56, 96)`이고 마지막 축 96개 channel 값이다. batch나 모든 spatial 위치를 함께 평균내지 않는다.

### 문제 3

ConvNeXt의 `7x7` depthwise convolution이 global attention과 동등하지 않은 한 가지 이유를 쓰라.

### 해답 3

depthwise kernel은 위치 공유되고 입력과 무관한 고정 계수로 local 이웃만 섞는다. global attention의 가중치는 query·key에서 매 입력마다 달라지고, 한 층에서 모든 토큰에 연결될 수 있다.

## 핵심 요약

- ConvNeXt는 transformer의 모든 연산을 복제한 CNN이 아니라, 현대적인 macro·micro design을 ResNet 계열에 적용한 결과다.
- depthwise convolution은 공간 혼합, `1x1`/Linear MLP는 채널 혼합을 맡는다.
- `NCHW -> NHWC -> NCHW` 전환은 `LayerNorm(C)`와 Linear의 축 계약을 맞춘다.
- $7x7$ local kernel은 attention과 구조적 유사점이 있으나 입력 의존적 전역 연결을 제공하지 않는다.
- 성능 판단에는 정확도뿐 아니라 layout copy, memory, preprocess, exported-runtime golden test가 필요하다.

## 다음 학습 예고

다음은 `02-10.SSL(Self-Supervised-Learning.md`의 self-supervised learning이다. 라벨 없이 augment된 두 view의 표현을 어떻게 맞추고, collapse를 왜 막아야 하는지 contrastive·non-contrastive 목표와 shape 관점에서 다룬다.
