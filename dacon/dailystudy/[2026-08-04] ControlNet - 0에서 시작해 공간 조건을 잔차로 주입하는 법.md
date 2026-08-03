<!-- curriculum: cycle=1; level=foundation; source_index=16/18; source=02-15.ControlNet.md; part=1/1 -->

# ControlNet - 0에서 시작해 공간 조건을 잔차로 주입하는 법

## 학습 진도

| 날짜 | 회차·수준 | 현재 소스 | Part | 이전 소스 | 다음 소스 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-04 | 1회차 · 기초 | 16/18 · `02-15.ControlNet.md` | 1/1 | `02-14.CFG(Classifier-Free guidance).md` | `02-16.LoRA.md` |

## 학습 목표

ControlNet은 이미 학습된 diffusion model의 생성 능력을 보존하면서 edge, depth, pose 같은 **공간 조건**을 추가하는 구조다. 이 글을 마치면 다음을 할 수 있다.

- text condition과 spatial condition이 맡는 역할을 구분한다.
- locked backbone, trainable copy, condition encoder, zero convolution의 연결을 설명한다.
- zero-initialized $1\times1$ convolution의 순전파와 역전파를 index 표기로 유도한다.
- 첫 backward에서 어느 파라미터가 즉시 학습되고 어느 파라미터의 gradient가 0인지 설명한다.
- 각 resolution에서 residual tensor의 shape가 맞는지 추적한다.
- NumPy 손계산과 PyTorch 미니 모델로 초기 불변성 및 gradient 흐름을 검증한다.
- C++와 C#에서 framework-independent zero convolution 한 step을 재현한다.
- layout, dtype, preprocessing, residual scale이 다른 runtime 사이에서 어긋나는 문제를 진단한다.
- 학습 메모리, mixed precision, zero-conv 초기화, ONNX 배포 실패를 예방한다.

## 선수 지식과 기호

2D convolution, chain rule, residual connection, U-Net의 encoder-decoder 구조와 DDPM noise prediction을 알고 있으면 충분하다.

| 기호 | 뜻 | 대표 shape |
| --- | --- | --- |
| $x_t$ | timestep $t$의 noisy latent | $(N,C_z,H_z,W_z)$ |
| $c_t$ | text embedding | $(N,L,D)$ |
| $c_s$ | edge, depth, pose 같은 spatial condition | $(N,C_s,H_s,W_s)$ |
| $h_i$ | locked backbone의 $i$번째 feature | $(N,C_i,H_i,W_i)$ |
| $\tilde{h}_i$ | trainable copy의 $i$번째 feature | $(N,C_i,H_i,W_i)$ |
| $Z_i$ | zero-initialized $1\times1$ convolution | channel projection |
| $r_i$ | decoder에 더하는 control residual $Z_i(\tilde{h}_i)$ | $(N,C_i,H_i,W_i)$ |
| $s_i$ | resolution $i$의 residual scale | scalar |
| $\epsilon_\theta$ | U-Net이 예측한 noise | $(N,C_z,H_z,W_z)$ |

이 글의 tensor 표기는 PyTorch의 `NCHW`를 기준으로 한다. 이미지 공간 조건의 해상도와 latent 해상도는 보통 다르므로 condition encoder가 크기와 channel을 맞춘다.

## 1. 직관: 의미와 기하를 서로 다른 통로로 보낸다

`말을 탄 우주비행사`라는 문장은 무엇이 있어야 하는지는 잘 말하지만, 관절 좌표나 윤곽선의 정확한 위치까지 지정하지는 않는다. 반대로 pose skeleton은 인물의 배치를 잘 지정하지만 옷감, 조명, 화풍을 설명하지 못한다.

ControlNet은 이 둘을 다음처럼 나눈다.

1. pretrained diffusion backbone은 text condition과 noisy latent를 받아 기존 생성 지식을 사용한다.
2. trainable control branch는 noisy latent와 spatial condition을 함께 읽는다.
3. branch가 만든 multi-resolution residual을 backbone decoder의 대응 feature에 더한다.
4. residual을 내보내는 convolution을 0으로 초기화해 학습 시작점에는 control 경로의 영향이 없게 한다.

핵심은 별도의 완성 이미지를 두 모델이 평균내는 것이 아니다. U-Net 내부의 여러 해상도에서 **feature residual**을 주입한다.

## 2. 원문에서 바로잡을 점

### 2.1 trainable copy는 무작위 네트워크가 아니다

원문은 “초기화된 복제 모델이 무작위 노이즈를 뿜는다”고 설명한다. 원래 ControlNet 설계에서 trainable copy는 pretrained U-Net의 encoder와 middle block 가중치를 복제해 시작한다. 즉 branch의 feature extractor 자체가 무작위 초기화되는 것이 아니다.

새 경로가 즉시 backbone 출력을 바꾸지 않는 직접적인 이유는 복제본의 출력이 원래부터 작아서가 아니라, backbone으로 연결하는 zero convolution의 weight와 bias가 정확히 0이기 때문이다.

### 2.2 전체 Stable Diffusion을 통째로 복제하지 않는다

개념 그림에서 “모델을 두 개로 나눈다”고 말할 수는 있지만, 표준 구조는 U-Net 전체 decoder까지 학습 가능한 사본으로 하나 더 만드는 방식이 아니다. locked U-Net을 유지하고, 주로 encoder와 middle block의 trainable copy에서 여러 control residual을 만든다. 이 residual들이 locked decoder의 skip 경로와 middle feature에 주입된다.

### 2.3 zero convolution은 마법이 아니라 초기 조건이다

zero convolution은 일반적인 convolution layer를 0으로 초기화한 것이다. optimizer step이 진행되면 weight와 bias는 0을 벗어난다. 0 제약을 영원히 유지하는 layer가 아니며, control strength도 자동으로 항상 안전하게 유지되지는 않는다.

### 2.4 “첫 결과가 100% 동일”에는 조건이 붙는다

control residual은 step 0에서 정확히 0이다. 따라서 동일한 backbone input, timestep, text embedding, scheduler state, random seed를 사용하고 dropout 같은 비결정성을 제거하면 control을 연결하기 전과 같은 network output을 얻는다.

그러나 서로 다른 random noise, stochastic sampler, dropout state, dtype 또는 kernel을 사용한 두 전체 pipeline의 최종 이미지를 비교해 “항상 bitwise 동일”하다고 말할 수는 없다. 검증 대상은 먼저 같은 입력에서의 U-Net output이어야 한다.

### 2.5 첫 backward의 gradient는 모든 곳에 동시에 흐르지 않는다

원문은 zero convolution의 weight가 첫 backward부터 학습된다는 점은 맞게 설명한다. 다만 zero convolution **앞쪽**에 있는 trainable copy의 깊은 파라미터는 첫 step에서 gradient가 0일 수 있다. 출력 projection의 weight가 0이어서 그 앞 입력으로 전달되는 gradient가 차단되기 때문이다. zero-conv가 한 번 업데이트된 뒤부터 upstream gradient가 열리는 구조다.

## 3. 구조를 block 단위로 읽기

### 3.1 locked path와 trainable path

resolution $i$의 locked block을 $F_i$, 복제한 trainable block을 $\tilde{F}_i$라 하자. 단순화하면 다음과 같다.

$$
h_i
=
F_i(h_{i-1};\theta_i)
$$

$$
\tilde{h}_i
=
\tilde{F}_i(\tilde{h}_{i-1}+q_i;\tilde{\theta}_i)
$$

여기서 $q_i$는 condition encoder에서 온 공간 조건 feature다. 초기에는 $\tilde{\theta}_i=\theta_i$지만, $\theta_i$는 frozen이고 $\tilde{\theta}_i$만 optimizer가 갱신한다.

### 3.2 zero convolution과 residual injection

trainable feature를 decoder가 기대하는 channel로 투영한다.

$$
r_i
=
Z_i(\tilde{h}_i;W_i,b_i)
$$

$$
d_i^{\mathrm{in}}
=
h_i+s_i r_i
$$

초기 조건은 다음과 같다.

$$
W_i=0,
\qquad
b_i=0
$$

따라서 모든 유한 입력 $\tilde{h}_i$에 대해 $r_i=0$이다. $s_i$가 어떤 유한 값이어도 첫 forward의 decoder 입력은 $h_i$와 같다.

### 3.3 condition encoder

spatial condition $c_s$는 RGB edge map일 수도 있고 single-channel depth map일 수도 있다. condition encoder는 이를 latent feature resolution과 channel에 맞춘다.

$$
q_0
=
E_\phi(c_s)
$$

실제 구현에서는 stride convolution과 activation으로 해상도를 단계적으로 낮춘다. 중요한 계약은 특정 architecture의 channel 숫자를 외우는 것이 아니라 다음 세 가지다.

- condition의 좌표계가 원본 이미지 crop, resize, flip과 정확히 함께 움직여야 한다.
- condition feature의 최종 spatial size가 주입받는 latent feature와 같아야 한다.
- condition encoder의 마지막 연결도 zero-initialized projection을 거쳐 초기 영향을 차단해야 한다.

## 4. zero convolution의 수학

### 4.1 순전파를 index로 쓰기

$1\times1$ convolution은 각 공간 위치에서 같은 선형변환을 적용한다.

$$
Y_{n,o,h,w}
=
\sum_{i=1}^{C_{\mathrm{in}}}
W_{o,i}X_{n,i,h,w}
+b_o
$$

$W_{o,i}=0$이고 $b_o=0$이면 다음이 성립한다.

$$
Y_{n,o,h,w}=0
$$

kernel이 $1\times1$이므로 spatial size는 stride 1, padding 0에서 보존된다.

$$
(N,C_{\mathrm{in}},H,W)
\longrightarrow
(N,C_{\mathrm{out}},H,W)
$$

### 4.2 weight와 bias gradient

상류 gradient를 $G_{n,o,h,w}=\partial\mathcal{L}/\partial Y_{n,o,h,w}$라 하자.

$$
\frac{\partial\mathcal{L}}{\partial W_{o,i}}
=
\sum_{n,h,w}
G_{n,o,h,w}X_{n,i,h,w}
$$

$$
\frac{\partial\mathcal{L}}{\partial b_o}
=
\sum_{n,h,w}
G_{n,o,h,w}
$$

이 두 식에는 현재 $W$ 값이 곱해지지 않는다. 입력 $X$와 상류 gradient $G$의 상관이 0이 아니면 zero-initialized weight도 첫 optimizer step에서 0을 벗어난다.

### 4.3 입력과 upstream parameter의 gradient

입력 gradient는 다르다.

$$
\frac{\partial\mathcal{L}}{\partial X_{n,i,h,w}}
=
\sum_o
W_{o,i}G_{n,o,h,w}
$$

첫 step에는 $W=0$이므로 다음이 된다.

$$
\frac{\partial\mathcal{L}}{\partial X}=0
$$

따라서 $X$를 만든 trainable copy의 파라미터 $\tilde{\theta}$도 이 출력 경로만 고려하면 첫 backward에서 gradient가 0이다.

$$
\frac{\partial\mathcal{L}}{\partial\tilde{\theta}}
=
\frac{\partial\mathcal{L}}{\partial X}
\frac{\partial X}{\partial\tilde{\theta}}
=0
$$

첫 optimizer step으로 $W\ne0$이 되면 다음 backward부터 upstream gradient가 전달된다. 이것은 버그가 아니라 zero gate가 서서히 열리는 결과다.

## 5. scalar 손계산

입력 feature $x=2$, zero-conv weight $w=0$, bias $b=0$, target residual $y^*=1$이라 하자.

$$
y=wx+b=0
$$

손실을 다음처럼 둔다.

$$
\mathcal{L}
=
\frac{1}{2}(y-y^*)^2
$$

출력 gradient는 $y-y^*=-1$이다.

$$
\frac{\partial\mathcal{L}}{\partial w}
=
(y-y^*)x
=
-2
$$

$$
\frac{\partial\mathcal{L}}{\partial b}
=
y-y^*
=
-1
$$

$$
\frac{\partial\mathcal{L}}{\partial x}
=
(y-y^*)w
=0
$$

learning rate가 $0.1$인 SGD 한 step 뒤에는 $w=0.2$, $b=0.1$이 된다. 같은 입력의 다음 출력은 $0.5$다.

## 6. NumPy 검산

다음 코드는 **실행 가능한 예제**다. $1\times1$ convolution을 matrix multiplication으로 보고 forward와 analytic gradient를 검산한다.

```python
import numpy as np

x = np.array([[2.0, -1.0]], dtype=np.float64)  # (P=1, Cin=2)
w = np.zeros((1, 2), dtype=np.float64)         # (Cout=1, Cin=2)
b = np.zeros((1,), dtype=np.float64)
target = np.array([[1.0]], dtype=np.float64)

y = x @ w.T + b
grad_y = y - target
grad_w = grad_y.T @ x
grad_b = grad_y.sum(axis=0)
grad_x = grad_y @ w

np.testing.assert_allclose(y, [[0.0]])
np.testing.assert_allclose(grad_w, [[-2.0, 1.0]])
np.testing.assert_allclose(grad_b, [-1.0])
np.testing.assert_allclose(grad_x, [[0.0, 0.0]])

lr = 0.1
w -= lr * grad_w
b -= lr * grad_b
y_next = x @ w.T + b
np.testing.assert_allclose(y_next, [[0.6]])
print("initial=0, grad_w=", grad_w.tolist(), "next=", y_next.item())
```

scalar 절과 달리 두 번째 input channel이 추가되어 다음 출력은 $0.6$이다.

## 7. tensor shape 추적

예시 latent를 $(N,4,64,64)$, condition image를 $(N,3,512,512)$라 하자. 실제 channel 수와 block 수는 backbone마다 다르지만 계약은 다음과 같다.

| 단계 | 연산 | 입력 shape | 출력 shape |
| --- | --- | --- | --- |
| condition 입력 | resize·normalize | $(N,3,512,512)$ | $(N,3,512,512)$ |
| condition encoder | stride downsample | $(N,3,512,512)$ | $(N,320,64,64)$ |
| encoder level 0 | control block | $(N,320,64,64)$ | $(N,320,64,64)$ |
| zero projection 0 | $1\times1$ conv | $(N,320,64,64)$ | $(N,320,64,64)$ |
| encoder level 1 | down block | $(N,320,64,64)$ | $(N,640,32,32)$ |
| zero projection 1 | $1\times1$ conv | $(N,640,32,32)$ | $(N,640,32,32)$ |
| middle | control middle block | $(N,1280,8,8)$ | $(N,1280,8,8)$ |
| decoder injection | elementwise add | backbone과 residual 동일 | 동일 shape |
| noise prediction | output conv | decoder final feature | $(N,4,64,64)$ |

elementwise add 직전에는 batch, channel, height, width가 모두 같아야 한다. broadcasting으로 우연히 실행되도록 두면 잘못된 batch나 channel이 조용히 복제될 수 있으므로 명시적으로 assert한다.

## 8. PyTorch 미니 구현

다음은 **실행 가능한 교육용 축소 구현**이다. 실제 Stable Diffusion U-Net 전체가 아니라, frozen block과 trainable copy, condition adapter, zero projection의 핵심 gradient 계약만 재현한다.

```python
import copy
import torch
from torch import nn

torch.manual_seed(7)


class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MiniControl(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        pretrained = Block(channels)
        self.locked = pretrained
        self.trainable = copy.deepcopy(pretrained)
        self.condition = nn.Conv2d(1, channels, 3, padding=1)
        self.zero = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)
        for parameter in self.locked.parameters():
            parameter.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.locked(x)
        control_feature = self.trainable(x + self.condition(condition))
        residual = self.zero(control_feature)
        if base.shape != residual.shape:
            raise RuntimeError(f"shape mismatch: {base.shape} vs {residual.shape}")
        return base + residual, residual


model = MiniControl()
x = torch.randn(2, 4, 8, 8)
condition = torch.randn(2, 1, 8, 8)
target = torch.randn_like(x)

base = model.locked(x).detach()
output, residual = model(x, condition)
torch.testing.assert_close(residual, torch.zeros_like(residual))
torch.testing.assert_close(output, base)

loss = (output - target).square().mean()
loss.backward()

assert model.zero.weight.grad is not None
assert model.zero.weight.grad.abs().sum() > 0
assert model.zero.bias.grad is not None
assert model.trainable.net[0].weight.grad is not None
assert torch.count_nonzero(model.trainable.net[0].weight.grad) == 0
assert model.condition.weight.grad is not None
assert torch.count_nonzero(model.condition.weight.grad) == 0
assert all(parameter.grad is None for parameter in model.locked.parameters())

optimizer = torch.optim.SGD(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.05,
)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
output2, residual2 = model(x, condition)
assert residual2.abs().sum() > 0
output2.square().mean().backward()
assert model.trainable.net[0].weight.grad.abs().sum() > 0
assert model.condition.weight.grad.abs().sum() > 0

print("shape:", tuple(output.shape))
print("step0 residual:", float(residual.detach().abs().max()))
print("step1 residual sum:", float(residual2.detach().abs().sum()))
print("finite:", bool(torch.isfinite(output2).all()))
```

### 이 코드가 확인하는 것

- `copy.deepcopy`로 locked block과 trainable block이 같은 pretrained 출발점을 갖는다.
- 첫 forward에서 residual이 정확히 0이고 output은 locked path와 같다.
- 첫 backward에서 zero projection의 weight와 bias에는 gradient가 생긴다.
- 첫 backward에서 trainable block과 condition adapter의 gradient tensor는 존재하지만 값은 0이다.
- 한 optimizer step 뒤에는 zero gate가 열려 upstream branch에도 non-zero gradient가 흐른다.
- frozen parameters에는 gradient가 생성되지 않는다.

실제 training loop에서는 optimizer를 만들 때 `requires_grad=True`인 parameter만 전달한다. frozen parameter를 optimizer에 넣어도 보통 업데이트되지는 않지만, state 관리와 메모리 사용이 불필요하게 복잡해진다.

## 9. C++ 예제

다음은 **실행 가능한 C++17 예제**다. 외부 tensor library 없이 scalar zero convolution의 첫 SGD step을 검증한다.

```cpp
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    const double x = 2.0;
    const double target = 1.0;
    const double learning_rate = 0.1;
    double weight = 0.0;
    double bias = 0.0;

    const double y = weight * x + bias;
    const double grad_y = y - target;
    const double grad_weight = grad_y * x;
    const double grad_bias = grad_y;
    const double grad_x = grad_y * weight;

    assert(y == 0.0);
    assert(grad_weight == -2.0);
    assert(grad_bias == -1.0);
    assert(grad_x == 0.0);

    weight -= learning_rate * grad_weight;
    bias -= learning_rate * grad_bias;
    const double next = weight * x + bias;
    assert(std::abs(next - 0.5) < 1e-12);

    std::cout << std::fixed << std::setprecision(1)
              << "initial=" << y << " next=" << next << '\n';
    return 0;
}
```

컴파일 예시는 `clang++ -std=c++17 controlnet_zero.cpp -o controlnet_zero`다.

## 10. C# 예제

다음은 **실행 가능한 C# 예제**이며 C++ 예제와 같은 golden value를 확인한다.

```csharp
using System;

public static class ControlNetZero
{
    public static void Main()
    {
        const double x = 2.0;
        const double target = 1.0;
        const double learningRate = 0.1;
        double weight = 0.0;
        double bias = 0.0;

        double y = weight * x + bias;
        double gradY = y - target;
        double gradWeight = gradY * x;
        double gradBias = gradY;
        double gradX = gradY * weight;

        if (y != 0.0 || gradWeight != -2.0 ||
            gradBias != -1.0 || gradX != 0.0)
        {
            throw new InvalidOperationException("step-0 invariant failed");
        }

        weight -= learningRate * gradWeight;
        bias -= learningRate * gradBias;
        double next = weight * x + bias;
        if (Math.Abs(next - 0.5) > 1e-12)
        {
            throw new InvalidOperationException("SGD golden value failed");
        }

        Console.WriteLine($"initial={y:F1} next={next:F1}");
    }
}
```

`csc ControlNetZero.cs` 또는 `dotnet` project에서 컴파일할 수 있다.

## 11. framework 간 shape·layout·dtype 대응

| 항목 | PyTorch | C++ runtime 예 | C# runtime 예 | 확인할 계약 |
| --- | --- | --- | --- | --- |
| image tensor | `NCHW`가 일반적 | ONNX Runtime은 model metadata 기준 | ML.NET·ONNX metadata 기준 | 축 순서를 이름으로 고정 |
| UI bitmap | 변환 필요 | `HWC`, `uint8`가 흔함 | `HWC`, `byte`가 흔함 | transpose와 normalize 순서 |
| condition | `float32` 또는 AMP | 흔히 `float32` | 흔히 `float32` | edge가 `0/1`인지 `0/255`인지 |
| latent | 보통 `float16`·`float32` | provider에 따라 다름 | provider에 따라 다름 | model input dtype과 일치 |
| timestep | 구현별 scalar·vector | `int64` 또는 float tensor | `Int64` 또는 `Single` | exported graph schema 확인 |
| text embedding | $(N,L,D)$ | contiguous buffer | multidimensional tensor | padding length와 dtype |
| residual list | resolution별 tensor | 여러 named output | 여러 named output | 이름·순서·shape 모두 검증 |

`NHWC` condition을 `NCHW`로 바꿀 때는 다음 대응을 사용한다.

$$
X^{\mathrm{NCHW}}_{n,c,h,w}
=
X^{\mathrm{NHWC}}_{n,h,w,c}
$$

buffer의 원소 수만 같다고 shape 계약이 맞는 것은 아니다. 예를 들어 $(1,3,512,512)$와 $(1,512,512,3)$은 같은 수의 원소를 가지므로 단순 length 검사는 오류를 잡지 못한다.

## 12. 테스트 전략

### 12.1 초기 불변성 테스트

같은 $x_t$, timestep, text embedding을 넣고 다음을 확인한다.

$$
\max
\left|
\epsilon_{\mathrm{with\ control}}-
\epsilon_{\mathrm{backbone}}
\right|
\le\tau
$$

`float32` eager mode에서는 exact zero residual을 기대할 수 있다. 서로 다른 exported runtime을 비교한다면 kernel 차이를 고려해 작은 tolerance $\tau$를 둔다.

### 12.2 gradient 테스트

첫 backward 직후 네 종류를 따로 검사한다.

| 파라미터 | 첫 backward 기대값 |
| --- | --- |
| locked backbone | `grad is None` |
| residual zero-conv weight·bias | non-zero 가능, 일반적으로 non-zero |
| zero-conv보다 앞선 trainable copy | zero gradient |
| condition encoder | 해당 zero gate만 경로라면 zero gradient |

두 번째 step 이후에는 data와 loss가 퇴화하지 않았다면 trainable copy와 condition encoder에 non-zero gradient가 나타나야 한다.

### 12.3 shape negative test

다음 입력은 조용한 resize나 broadcasting 대신 즉시 실패시킨다.

- batch size가 다른 condition
- crop 후 width 또는 height가 1 pixel 다른 condition
- single-channel depth를 3-channel로 잘못 선언한 input
- residual list의 resolution 순서가 뒤집힌 export
- backbone variant가 기대하는 channel과 다른 ControlNet checkpoint

### 12.4 zero state audit

checkpoint load 직후 다음을 기록한다.

```python
for name, parameter in model.named_parameters():
    if ".zero." in name:
        print(name, float(parameter.detach().abs().max()))
```

이 코드는 **설명용 점검 조각**이다. 이미 학습된 ControlNet checkpoint에서는 zero-conv가 0이 아니므로, 새로 초기화한 학습 시작점에만 zero를 기대해야 한다.

## 13. 디버깅 순서

### 증상 A: control을 켜자 첫 step부터 출력이 크게 바뀐다

1. zero-conv를 만든 뒤 generic initializer가 다시 덮어썼는지 확인한다.
2. bias까지 0인지 확인한다.
3. condition feature가 zero projection을 우회해 backbone에 직접 더해지는지 확인한다.
4. 비교 실행이 같은 latent, timestep, text embedding, scheduler state인지 확인한다.

### 증상 B: zero-conv만 학습되고 branch는 계속 멈춰 있다

첫 step의 zero upstream gradient는 정상이다. 여러 optimizer step 뒤에도 branch gradient가 계속 0이면 다음을 확인한다.

- optimizer가 zero-conv를 실제로 포함하는가.
- learning rate가 0이거나 scheduler가 너무 일찍 0으로 만들지 않았는가.
- mixed precision overflow로 optimizer step이 skip되지 않았는가.
- residual scale $s_i$가 0으로 고정되어 있지 않은가.
- `detach()`가 control feature에 잘못 들어가 있지 않은가.

### 증상 C: edge와 생성물 위치가 어긋난다

모델보다 preprocessing을 먼저 의심한다. 원본 이미지에 random crop과 horizontal flip을 적용하면서 condition에는 같은 transform parameter를 적용하지 않으면 supervision 자체가 모순된다. 두 tensor를 별도로 random transform하지 말고 하나의 sample 단위 transform으로 묶는다.

### 증상 D: 여러 ControlNet을 합치면 과포화된다

각 control residual을 단순 합산하면 전체 norm이 커질 수 있다.

$$
r_i^{\mathrm{total}}
=
\sum_{k=1}^{K}s_{i,k}r_{i,k}
$$

각 branch scale과 timestep schedule을 따로 기록하고, residual norm 및 최종 latent norm을 관찰한다. “ControlNet 2개니까 각각 1.0”은 안전한 보편 규칙이 아니다.

## 14. 성능과 메모리

### 14.1 무엇이 비싼가

$1\times1$ zero convolution 자체보다 U-Net encoder와 middle block을 한 벌 더 실행하고 activation을 저장하는 비용이 크다. training에서는 trainable copy의 backward activation도 필요하다.

feature map $(N,C,H,W)$ 하나를 dtype byte 수 $b$로 저장하는 최소 크기는 다음과 같다.

$$
M
=
NCHWb
$$

예를 들어 $(2,320,64,64)$ `float32` tensor 하나는 다음 크기다.

$$
2\times320\times64\times64\times4
=
10{,}485{,}760\ \mathrm{bytes}
$$

이는 약 $10.0$ MiB이며, 실제 training peak에는 여러 block activation, gradient, optimizer state와 temporary workspace가 더해진다.

### 14.2 실용적인 절감 방법

- locked backbone forward는 gradient를 만들 필요가 없는 구간을 명확히 분리한다.
- trainable branch에는 gradient checkpointing을 적용해 compute와 memory를 교환한다.
- `float16` 또는 `bfloat16`을 쓰되 loss scaling과 overflow를 관찰한다.
- condition image를 매 step CPU에서 반복 추출하지 말고 재사용 가능한 edge·depth map은 cache한다.
- 여러 ControlNet을 동시에 쓸 때 residual을 오래 보관하지 말고 소비 순서에 맞춰 lifetime을 줄인다.
- batch를 줄이기 전에 activation checkpoint와 optimizer state dtype을 함께 점검한다.

### 14.3 수치 안정성

zero-conv는 초기에는 안전하지만 학습이 진행되면 residual norm이 커질 수 있다. 다음 비율을 layer별로 기록하면 유용하다.

$$
\rho_i
=
\frac{\lVert s_i r_i\rVert_2}
{\lVert h_i\rVert_2+\varepsilon}
$$

$\rho_i$가 갑자기 폭증하면 learning rate, loss scale, condition outlier 또는 checkpoint mismatch를 확인한다. `float16`에서 norm을 직접 누적할 때 overflow가 날 수 있으므로 metric 계산은 `float32`로 cast한다.

## 15. 실무 실패 사례

### 사례 1: Canny threshold가 학습·운영에서 다르다

학습 데이터는 얇은 edge를 사용했는데 서비스는 두꺼운 edge나 다른 threshold를 사용하면 condition distribution이 바뀐다. 모델 version과 함께 preprocessor 종류, threshold, resize mode를 배포 artifact로 고정해야 한다.

### 사례 2: depth 단위와 방향이 뒤집힌다

한 pipeline은 가까운 값을 1, 먼 값을 0으로 만들고 다른 pipeline은 반대로 만들 수 있다. min-max normalization은 scene마다 scale을 바꾸기도 한다. depth estimator version과 normalization 식을 metadata에 기록한다.

### 사례 3: pose crop만 좌우 반전된다

image를 flip했는데 left/right joint label 또는 skeleton map을 같이 바꾸지 않으면 모델은 상충하는 좌표를 학습한다. visual overlay test로 condition과 target이 pixel 단위로 겹치는지 검사한다.

### 사례 4: 다른 backbone용 checkpoint를 억지로 load한다

channel 수가 우연히 맞아도 block ordering, cross-attention dimension, latent scaling이 다르면 의미가 맞지 않는다. checkpoint에 base model identifier와 config hash를 저장하고 load 전에 비교한다.

### 사례 5: “frozen”인데 BatchNorm 통계가 변한다

파라미터의 `requires_grad=False`만으로 module state가 모두 고정되는 것은 아니다. BatchNorm running statistics와 dropout은 train/eval mode의 영향을 받는다. Stable Diffusion 계열 U-Net은 보통 GroupNorm을 쓰지만, 다른 backbone으로 일반화할 때는 stateful layer를 따로 감사한다.

## 16. 배포 관점

### 16.1 두 graph인가 한 graph인가

배포 선택지는 크게 두 가지다.

1. ControlNet graph가 resolution별 residual을 출력하고 base U-Net이 이를 입력받는다.
2. Control branch와 base U-Net을 하나의 graph로 합친다.

분리형은 base U-Net 재사용과 control 조합이 쉽지만 tensor transfer와 I/O binding이 중요하다. 통합형은 최적화 기회가 있지만 control variant마다 큰 graph를 관리할 수 있다.

### 16.2 export contract

최소한 다음을 versioned manifest에 넣는다.

- base model과 ControlNet checkpoint identifier
- latent `NCHW` shape와 dynamic axis 허용 범위
- condition 종류, channel 수, normalization, resize와 crop 규칙
- timestep dtype과 batching 규칙
- text embedding length와 dimension
- residual output 이름, resolution 순서와 channel
- residual scale 기본값과 허용 범위
- prediction type이 `epsilon`, `v_prediction` 또는 다른 형식인지

### 16.3 운영 모니터링

이미지 품질은 단일 scalar로 잡기 어렵다. 최소한 시스템 metric과 control metric을 함께 본다.

- latency의 condition encoder, control branch, base U-Net 분해
- peak GPU memory와 OOM 비율
- NaN·Inf output 비율
- condition이 비어 있거나 거의 상수인 sample 비율
- residual norm ratio $\rho_i$의 분위수
- pose keypoint나 edge alignment 같은 task-specific proxy
- base-only 결과와 control 결과의 품질 회귀용 고정 seed golden set

개인정보가 포함될 수 있는 pose, depth, 원본 이미지를 그대로 장기 저장하지 말고 필요한 집계 metric과 보존 정책을 별도로 설계한다.

## 17. 설계 의사결정

| 상황 | 우선 선택 | 이유 |
| --- | --- | --- |
| 윤곽·자세를 강하게 제어 | ControlNet | multi-resolution spatial residual이 직접적 |
| style이나 작은 domain 적응 | LoRA 검토 | parameter-efficient weight adaptation이 목적에 가까움 |
| text 조건만 더 강하게 반영 | CFG 조정 | 별도 spatial condition branch가 불필요 |
| 여러 base model에서 같은 control 재사용 | 호환성 먼저 검증 | architecture·latent contract가 다를 수 있음 |
| VRAM이 매우 제한됨 | adapter·경량 control 또는 offload 검토 | full trainable copy 비용이 큼 |
| 실시간 서비스 | preprocessor cache·graph 분리 benchmark | condition 추출과 residual 전달 비용을 분리 측정 |

ControlNet, CFG, LoRA는 서로 배타적이지 않다. ControlNet은 공간 구조, CFG는 condition strength, LoRA는 parameter-efficient adaptation을 주로 담당한다. 함께 쓸 때는 scale들의 상호작용을 독립적으로 ablation한다.

## 18. 체크리스트

### 학습 전

- [ ] base model identifier와 preprocessing version을 고정했다.
- [ ] trainable copy가 pretrained encoder·middle weight에서 시작한다.
- [ ] locked backbone의 parameter와 module mode를 확인했다.
- [ ] 모든 zero-conv의 weight와 bias가 0이다.
- [ ] image와 spatial condition에 동일한 geometric transform을 적용한다.
- [ ] condition channel, range, layout, dtype을 문서화했다.

### 첫 두 optimizer step

- [ ] step 0 residual이 0이고 base output 불변성이 통과한다.
- [ ] first backward에서 zero-conv gradient가 non-zero다.
- [ ] first backward에서 zero-conv upstream gradient가 0임을 이해한다.
- [ ] optimizer step 뒤 zero-conv parameter가 0을 벗어난다.
- [ ] second backward에서 trainable copy와 condition encoder에 gradient가 흐른다.
- [ ] locked backbone에는 gradient가 없다.

### 배포 전

- [ ] residual output 이름·순서·shape를 golden test로 고정했다.
- [ ] preprocessing을 training과 같은 코드 또는 versioned 구현으로 재사용한다.
- [ ] control scale 0에서 base-only output과 비교한다.
- [ ] 여러 control 합산 시 residual norm을 관찰한다.
- [ ] target hardware에서 dtype, memory, latency를 측정했다.
- [ ] 고정 seed·고정 scheduler golden sample을 회귀 테스트에 넣었다.

## 19. 연습문제

### 문제 1

zero-conv의 weight와 bias가 모두 0일 때 output이 0인 이유와 weight gradient가 0일 필요가 없는 이유를 설명하라.

### 문제 2

첫 backward에서 zero-conv 앞 trainable block의 gradient가 0이 되는 이유를 수식으로 설명하라.

### 문제 3

backbone feature가 $(2,640,32,32)$이고 control residual이 $(2,640,31,32)$다. elementwise add 전에 무엇을 해야 하는가?

### 문제 4

$(2,320,64,64)$ `float16` feature 하나의 최소 저장 크기를 MiB로 계산하라. $1\ \mathrm{MiB}=2^{20}$ bytes로 둔다.

### 문제 5

control scale을 0으로 했는데 base-only 결과와 다르다. 가장 먼저 고정해야 할 네 가지 입력 또는 상태를 쓰라.

### 문제 6

학습 후 저장한 ControlNet checkpoint를 load했더니 zero-conv가 0이 아니다. 이것이 항상 오류인가?

## 20. 해답

### 해답 1

순전파는 $Y=WX+b$이고 $W=b=0$이라 $Y=0$이다. 반면 $\partial\mathcal{L}/\partial W=G X^T$이므로 현재 $W$가 아니라 입력 $X$와 상류 gradient $G$에 의해 결정된다. 둘의 상관이 0이 아니면 weight gradient는 non-zero다.

### 해답 2

입력 gradient가 $\partial\mathcal{L}/\partial X=W^T G$인데 첫 step의 $W=0$이므로 0이다. chain rule에 의해 그 앞 parameter의 gradient도 이 경로에서는 0이다. zero-conv가 한 번 업데이트된 뒤 $W\ne0$이 되면 gradient가 열린다.

### 해답 3

즉시 shape mismatch로 실패시킨 뒤 preprocessing, stride, padding, odd-size 처리 또는 residual 순서를 수정한다. 의미를 확인하지 않은 interpolation이나 crop으로 억지로 맞추면 좌표가 어긋날 수 있다.

### 해답 4

`float16`은 2 bytes다.

$$
\frac{2\times320\times64\times64\times2}{2^{20}}
=5\ \mathrm{MiB}
$$

이는 tensor payload 하나만의 최소값이며 allocator, activation, gradient는 별도다.

### 해답 5

같은 initial latent $x_t$, timestep, text embedding, scheduler state 또는 random generator state를 먼저 고정한다. dropout 등 module mode와 dtype·kernel도 같아야 엄격한 비교가 가능하다.

### 해답 6

아니다. 0 초기화는 학습 시작점의 조건이다. 학습이 진행된 checkpoint의 zero-conv는 control residual을 만들기 위해 0에서 벗어나는 것이 정상이다. 새로 만든 model의 초기화 직후인지, 학습된 checkpoint load 직후인지 구분해야 한다.

## 핵심 요약

- ControlNet은 pretrained diffusion backbone을 frozen 상태로 유지하고 encoder·middle의 trainable copy에서 spatial control residual을 만든다.
- trainable copy는 무작위가 아니라 pretrained weight를 복제해 출발한다.
- zero-initialized $1\times1$ convolution 때문에 step 0의 residual은 정확히 0이다.
- zero-conv weight와 bias는 첫 backward에서 학습될 수 있지만, 그 앞 branch는 첫 step에 zero gradient를 받을 수 있다.
- control residual과 backbone feature는 모든 축의 shape, layout, dtype이 맞아야 한다.
- 실제 비용은 zero-conv보다 trainable encoder copy의 forward, activation과 backward가 지배한다.
- preprocessing과 base-model 호환성은 학습 정확도뿐 아니라 배포 계약의 일부다.
- ControlNet은 공간 구조, CFG는 guidance strength, LoRA는 parameter-efficient adaptation이라는 서로 다른 축을 다룬다.

## 다음 학습 예고

다음 소스는 `02-16.LoRA.md`다. ControlNet이 별도 control branch와 multi-resolution residual로 공간 조건을 주입했다면, LoRA는 큰 weight matrix의 업데이트를 낮은 rank의 두 행렬로 제한해 적은 trainable parameter로 모델을 적응시킨다. 다음 글에서는 $\Delta W=BA$의 shape, rank와 parameter 수, scaling, merge·unmerge, 초기화와 배포 계약을 추적한다.
