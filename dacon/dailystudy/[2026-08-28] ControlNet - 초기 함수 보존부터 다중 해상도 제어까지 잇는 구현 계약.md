<!-- curriculum: cycle=2; level=implementation; source_index=16/18; source=02-15.ControlNet.md; part=1/1 -->

# ControlNet - 초기 함수 보존부터 다중 해상도 제어까지 잇는 구현 계약

## 학습 진도

| 날짜 | 회차·수준 | 현재 소스 | Part | 이전 소스 | 다음 소스 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-28 | 2회차 · 구현 | 16/18 · `02-15.ControlNet.md` | 1/1 | `02-14.CFG(Classifier-Free guidance).md` | `02-16.LoRA.md` |

## 학습 목표

1회차에는 frozen backbone, trainable copy, zero convolution의 직관과 첫 역전파를 배웠다. 이번에는 그 설명을 재현 가능한 학습·평가·배포 코드로 바꾼다. 이 글을 마치면 다음을 할 수 있다.

- 잠긴 denoiser와 학습 가능한 control branch의 파라미터 경계를 코드로 고정한다.
- zero-initialized projection이 step 0의 함수를 보존하는 조건을 테스트한다.
- 첫 optimizer step 전후의 gradient가 어느 계층까지 흐르는지 검증한다.
- condition image를 여러 해상도의 residual로 변환하고 대응 feature에 주입한다.
- crop, resize, flip을 image와 condition에 함께 적용하는 데이터 계약을 세운다.
- 작은 PyTorch 모델을 끝까지 학습하고 checkpoint를 동일 출력으로 복원한다.
- zero initialization과 random initialization을 공정한 ablation으로 비교한다.
- NumPy, PyTorch, C++와 C#의 shape·layout·dtype 대응을 정리한다.
- mixed precision, 메모리, ONNX export와 운영 실패를 진단한다.

## 선수 지식과 기호

2D convolution, residual connection, chain rule, U-Net의 encoder·decoder, DDPM noise-prediction loss를 알고 있으면 된다.

| 기호 | 정의 | 대표 shape |
| --- | --- | --- |
| $x_t$ | timestep $t$의 noisy latent | $(N,C_z,H,W)$ |
| $c_s$ | edge, depth, pose 같은 공간 조건 | $(N,C_s,H_c,W_c)$ |
| $c_t$ | text condition | $(N,L,D)$ |
| $h_i$ | frozen backbone의 $i$번째 feature | $(N,C_i,H_i,W_i)$ |
| $u_i$ | trainable copy의 $i$번째 feature | $(N,C_i,H_i,W_i)$ |
| $q_i$ | condition encoder의 $i$번째 feature | $(N,Q_i,H_i,W_i)$ |
| $Z_i$ | 0으로 초기화한 $1\times1$ projection | $C_i\to C_i$ |
| $r_i$ | control residual $Z_i(u_i)$ | $(N,C_i,H_i,W_i)$ |
| $s_i$ | level별 control scale | scalar 또는 $(N,1,1,1)$ |
| $\epsilon_\theta$ | denoiser의 noise prediction | $(N,C_z,H,W)$ |

본문의 tensor는 PyTorch `NCHW`를 기준으로 한다. 배포 manifest에는 layout을 생략하지 않는다.

## 1. 원본을 구현 관점에서 다시 읽기

원본은 zero convolution의 핵심 아이디어를 잘 보여 주지만, 실제 시스템에서 그대로 받아들이면 위험한 표현이 있다.

| 원본의 표현 | 구현에서의 교정 |
| --- | --- |
| fine-tuning하면 파괴적 망각이 발생한다 | 가능한 실패 모드이지 모든 fine-tuning의 필연적 결과는 아니다. 학습률, 데이터 혼합, regularization과 평가로 판단한다. |
| 복제 모델이 무작위 노이즈를 낸다 | trainable copy는 pretrained encoder·middle block weight를 복사해 시작한다. 새 영향이 0인 직접 원인은 출력 zero projection이다. |
| Stable Diffusion을 두 개로 나눈다 | 표준 설계는 locked U-Net 전체와 encoder·middle 중심의 trainable copy를 결합한다. decoder 전체를 두 벌 학습하는 설명은 과도한 단순화다. |
| step 0 결과가 100% 동일하다 | 같은 입력·상태에서 denoiser 출력이 같다는 뜻이다. dropout, RNG, dtype, scheduler가 다르면 전체 이미지의 bitwise 동일성은 보장되지 않는다. |
| zero-conv 앞의 모든 계층이 첫 backward부터 학습된다 | zero output weight에는 gradient가 생기지만, 그 앞 feature로 가는 gradient는 첫 backward에 0이다. projection이 한 번 갱신된 뒤 열린다. |
| condition을 강제로 주입한다 | scale, preprocessing, 학습 분포에 따라 무시되거나 과잉 제어될 수 있다. 강도는 평가할 hyperparameter다. |
| 자세 조건이면 캐릭터 일관성이 완벽하다 | pose는 기하를 제한할 뿐 identity·texture·시간 일관성을 보장하지 않는다. 별도 조건과 temporal 설계가 필요하다. |

이번 구현의 경계는 다음과 같다.

```text
paired image + spatial condition
  -> synchronized augmentation
  -> latent/noise/timestep + text context
  -> frozen denoiser feature pyramid
  -> copied trainable encoder + condition encoder
  -> zero projections at matching resolutions
  -> scaled residual injection
  -> noise-prediction loss
  -> control-only checkpoint + deployment manifest
```

## 2. 구조를 함수 계약으로 분해하기

### 2.1 frozen path

frozen encoder block $F_i$와 decoder $D$를 다음처럼 둔다.

$$
h_i
=
F_i(h_{i-1};\theta_i),
\qquad
\operatorname{requires\_grad}(\theta_i)=\mathrm{False}
$$

제어가 없을 때 기준 함수는 다음과 같다.

$$
y_{\mathrm{base}}
=
D(h_1,\ldots,h_K;\theta_D)
$$

`eval()`과 freeze는 다른 연산이다. `requires_grad_(False)`는 weight gradient를 막고, `eval()`은 dropout과 batch normalization의 동작을 바꾼다. pretrained denoiser의 원래 normalization이 GroupNorm이라도 두 개념을 구분해 둔다.

### 2.2 trainable copy와 condition path

복제한 block의 초기 weight는 frozen block과 같다.

$$
\tilde{\theta}_i^{(0)}
=
\theta_i
$$

공간 조건은 condition encoder $E_\phi$를 거쳐 branch 입력에 더해진다.

$$
q_0
=
E_\phi(c_s)
$$

$$
u_1
=
\tilde{F}_1(x_t+q_0;\tilde{\theta}_1)
$$

더 깊은 level은 stride와 channel을 맞춰 진행한다.

$$
u_i
=
\tilde{F}_i(u_{i-1}+q_i;\tilde{\theta}_i)
$$

condition adapter의 마지막 projection도 0으로 초기화할 수 있다. 이 경우 시작점에는 복제 branch가 noisy latent만 읽고, 학습되면서 condition 통로가 열린다.

### 2.3 다중 해상도 residual injection

각 level의 trainable feature를 zero projection에 통과시킨다.

$$
r_i
=
Z_i(u_i;W_i,b_i)
$$

decoder가 소비하는 feature는 다음과 같다.

$$
h_i^{\prime}
=
h_i+s_i r_i
$$

따라서 controlled output은 다음 함수다.

$$
y_{\mathrm{ctrl}}
=
D(h_1+s_1r_1,\ldots,h_K+s_Kr_K;\theta_D)
$$

주입 직전에 반드시 세 조건을 검사한다.

$$
\operatorname{shape}(h_i)
=
\operatorname{shape}(r_i)
$$

$$
\operatorname{dtype}(h_i)
=
\operatorname{dtype}(r_i)
$$

$$
\operatorname{device}(h_i)
=
\operatorname{device}(r_i)
$$

broadcast로 우연히 실행되는 shape는 허용하지 않는다. 특히 `(N, C, 1, 1)` residual이 전체 공간에 broadcast되면 공간 제어가 아니라 channel bias가 된다.

## 3. zero projection의 forward·backward 유도

### 3.1 step 0의 함수 보존

$1\times1$ convolution을 index로 쓰면 다음과 같다.

$$
r_{n,o,h,w}
=
\sum_{j=1}^{C_{\mathrm{in}}}
W_{o,j}u_{n,j,h,w}
+b_o
$$

초기 조건은 다음과 같다.

$$
W^{(0)}=0,
\qquad
b^{(0)}=0
$$

그러면 모든 유한한 $u$에 대해 $r_i^{(0)}=0$이고, 모든 유한한 scale $s_i$에 대해 다음이 성립한다.

$$
y_{\mathrm{ctrl}}^{(0)}
=
y_{\mathrm{base}}
$$

이는 network function에 대한 등식이다. optimizer state나 최종 sampling trajectory까지 자동으로 같다는 뜻은 아니다.

### 3.2 출력 weight는 첫 backward부터 움직인다

상류 gradient를 다음처럼 정의한다.

$$
G_{n,o,h,w}
=
\frac{\partial\mathcal{L}}{\partial r_{n,o,h,w}}
$$

weight와 bias gradient는 다음과 같다.

$$
\frac{\partial\mathcal{L}}{\partial W_{o,j}}
=
\sum_{n,h,w}
G_{n,o,h,w}u_{n,j,h,w}
$$

$$
\frac{\partial\mathcal{L}}{\partial b_o}
=
\sum_{n,h,w}
G_{n,o,h,w}
$$

현재 $W$ 값이 두 식에 곱해지지 않는다. feature와 error가 상관되어 있으면 첫 backward에서 output projection이 0을 벗어날 준비를 한다.

### 3.3 복제 branch는 한 박자 늦게 gradient를 받는다

입력 feature에 대한 gradient는 다음과 같다.

$$
\frac{\partial\mathcal{L}}{\partial u_{n,j,h,w}}
=
\sum_o
W_{o,j}G_{n,o,h,w}
$$

첫 backward에서는 $W=0$이므로 다음이 성립한다.

$$
\frac{\partial\mathcal{L}}{\partial u}=0
$$

따라서 output zero projection만 경로로 가진 복제 block의 첫 gradient도 0이다.

$$
\frac{\partial\mathcal{L}}{\partial\tilde{\theta}}
=
\frac{\partial\mathcal{L}}{\partial u}
\frac{\partial u}{\partial\tilde{\theta}}
=0
$$

첫 optimizer step에서 $W\ne0$이 되면 두 번째 backward부터 upstream gradient가 열린다. 이 순서를 unit test로 고정하면 잘못된 random initialization이나 detached residual을 빨리 찾을 수 있다.

### 3.4 여러 zero projection이 동시에 있을 때

level별 loss gradient 크기는 feature의 scale과 spatial size에 좌우된다.

$$
\left\|
\frac{\partial\mathcal{L}}{\partial W_i}
\right\|_F
\propto
\left\|
\sum_{n,h,w}G_i u_i^{\mathsf{T}}
\right\|_F
$$

고해상도 level은 합산 위치가 많아 gradient norm이 커질 수 있다. level마다 같은 learning rate를 쓴다고 실제 update 크기가 같지는 않다. gradient norm을 level별로 기록하되, 무조건 같은 값으로 정규화하지 말고 품질 metric과 함께 본다.

## 4. tensor shape 추적

$N=2$, latent가 `(2, 4, 32, 48)`, condition image가 `(2, 3, 256, 384)`인 예를 보자.

| 경계 | tensor | shape | 비고 |
| --- | --- | --- | --- |
| 입력 | noisy latent | `(2, 4, 32, 48)` | `NCHW`, model dtype |
| 입력 | condition image | `(2, 3, 256, 384)` | 정규화 범위를 manifest에 기록 |
| condition stem | $q_1$ | `(2, 320, 32, 48)` | 총 stride 8 |
| control level 1 | $u_1$, $r_1$ | `(2, 320, 32, 48)` | frozen skip 1과 동일 |
| control level 2 | $u_2$, $r_2$ | `(2, 640, 16, 24)` | stride 2 |
| control level 3 | $u_3$, $r_3$ | `(2, 1280, 8, 12)` | stride 4 |
| middle | $u_4$, $r_4$ | `(2, 1280, 4, 6)` | stride 8 |
| 출력 | predicted noise | `(2, 4, 32, 48)` | latent와 동일 |

숫자는 특정 checkpoint의 예일 뿐 보편 규칙이 아니다. 실제 계약은 backbone이 내는 skip 목록의 length, channel, spatial shape와 control residual 목록이 정확히 같다는 것이다.

홀수 해상도에서는 downsample rounding과 upsample target을 명시한다. `scale_factor=2`만 믿지 말고 skip의 실제 크기를 사용한다.

```python
# 설명용: 실제 decoder에서 권장하는 spatial alignment
up = torch.nn.functional.interpolate(
    low_resolution,
    size=skip.shape[-2:],
    mode="nearest",
)
assert up.shape[-2:] == skip.shape[-2:]
```

## 5. NumPy 수작업 검증

아래 코드는 **실행 가능한 독립 golden test**다. zero gate 한 개의 첫 SGD step과 두 해상도 residual 합성을 손계산과 비교한다.

```python
import numpy as np

x = np.array([1.0, -2.0], dtype=np.float64)
target = np.array([0.5, -0.5], dtype=np.float64)
w = np.float64(0.0)
b = np.float64(0.0)

y0 = w * x + b
error = y0 - target
grad_w = np.sum(error * x)
grad_b = np.sum(error)

assert np.array_equal(y0, np.zeros_like(x))
assert grad_w == -1.5
assert grad_b == 0.0

learning_rate = 0.1
w -= learning_rate * grad_w
b -= learning_rate * grad_b
y1 = w * x + b
np.testing.assert_allclose(y1, [0.15, -0.30], rtol=0.0, atol=1e-12)

base_high = np.array([[1.0, 2.0], [3.0, 4.0]])
residual_high = np.array([[0.1, -0.1], [0.2, -0.2]])
base_low = np.array([[10.0]])
residual_low = np.array([[0.5]])

controlled_high = base_high + 2.0 * residual_high
controlled_low = base_low + 0.25 * residual_low
np.testing.assert_allclose(controlled_high, [[1.2, 1.8], [3.4, 3.6]])
np.testing.assert_allclose(controlled_low, [[10.125]])

print(f"w={w:.6f} y1={y1.tolist()}")
print("multi-scale golden passed")
```

이 예제에서 zero output은 학습 불능을 뜻하지 않는다. $w$의 gradient는 `-1.5`이고 첫 step 뒤 출력은 즉시 0을 벗어난다.

## 6. PyTorch 완전 구현

아래 코드는 **CPU에서 실행 가능한 축소 구현**이다. 실제 diffusion U-Net 대신 작은 frozen denoiser를 쓰지만, 다음 계약은 그대로 유지한다.

- frozen path와 copied path의 weight가 시작할 때 같다.
- condition adapter와 두 output projection은 0으로 초기화한다.
- 두 해상도의 residual을 대응 feature에 더한다.
- step 0 출력 동일성, 첫·두 번째 backward의 gradient 순서를 검사한다.
- toy control target을 학습하고 같은 seed의 exact replay를 검사한다.

```python
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class FrozenDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(2, 4, 3, padding=1)
        self.enc2 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.middle = nn.Conv2d(8, 8, 3, padding=1)
        self.up = nn.Conv2d(8, 4, 3, padding=1)
        self.out = nn.Conv2d(4, 1, 1)

    def encode(self, x):
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        middle = F.silu(self.middle(h2))
        return h1, middle

    def decode(self, h1, middle):
        up = F.interpolate(middle, size=h1.shape[-2:], mode="nearest")
        return self.out(F.silu(self.up(up) + h1))

    def forward(self, x, residuals=None):
        h1, middle = self.encode(x)
        if residuals is not None:
            r1, r2 = residuals
            assert r1.shape == h1.shape
            assert r2.shape == middle.shape
            h1 = h1 + r1
            middle = middle + r2
        return self.decode(h1, middle)


class TinyControlNet(nn.Module):
    def __init__(self, backbone: FrozenDenoiser, zero_init=True):
        super().__init__()
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)

        self.copy_enc1 = copy.deepcopy(backbone.enc1)
        self.copy_enc2 = copy.deepcopy(backbone.enc2)
        self.copy_middle = copy.deepcopy(backbone.middle)
        for layer in (self.copy_enc1, self.copy_enc2, self.copy_middle):
            layer.requires_grad_(True)

        self.cond_stem = nn.Conv2d(1, 2, 3, padding=1)
        self.cond_gate = ZeroConv2d(2, 2)
        self.zero1 = ZeroConv2d(4, 4)
        self.zero2 = ZeroConv2d(8, 8)

        if not zero_init:
            for layer in (self.cond_gate, self.zero1, self.zero2):
                nn.init.kaiming_uniform_(layer.weight, a=5 ** 0.5)
                nn.init.zeros_(layer.bias)

    def control_residuals(self, x, condition, scales=(1.0, 1.0)):
        assert x.ndim == condition.ndim == 4
        assert x.shape[0] == condition.shape[0]
        assert x.shape[-2:] == condition.shape[-2:]

        q = self.cond_gate(F.silu(self.cond_stem(condition)))
        u1 = F.silu(self.copy_enc1(x + q))
        u2 = F.silu(self.copy_enc2(u1))
        u2 = F.silu(self.copy_middle(u2))
        return scales[0] * self.zero1(u1), scales[1] * self.zero2(u2)

    def forward(self, x, condition, scales=(1.0, 1.0)):
        residuals = self.control_residuals(x, condition, scales)
        return self.backbone(x, residuals)


def make_batch():
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 9),
        torch.linspace(-1.0, 1.0, 11),
        indexing="ij",
    )
    condition = torch.stack([
        (grid_x > 0).to(torch.float64),
        (grid_y > 0).to(torch.float64),
        ((grid_x + grid_y) > 0).to(torch.float64),
        ((grid_x - grid_y) > 0).to(torch.float64),
    ])[:, None]
    latent = torch.cat([condition, 1.0 - condition], dim=1)
    return latent, condition


def gradient_gate_test():
    seed_everything(7)
    backbone = FrozenDenoiser()
    model = TinyControlNet(backbone, zero_init=True)
    x, condition = make_batch()
    target = backbone(x).detach() + 0.25 * (condition - 0.5)

    with torch.no_grad():
        base = backbone(x)
        controlled = model(x, condition)
        assert torch.equal(base, controlled)
        assert all(not p.requires_grad for p in backbone.parameters())

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    loss = F.mse_loss(model(x, condition), target)
    loss.backward()

    first_zero_grad = model.zero1.weight.grad.norm().item()
    first_copy_grad = model.copy_enc1.weight.grad.norm().item()
    assert first_zero_grad > 0.0
    assert first_copy_grad == 0.0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    F.mse_loss(model(x, condition), target).backward()
    second_copy_grad = model.copy_enc1.weight.grad.norm().item()
    assert second_copy_grad > 0.0

    return first_zero_grad, first_copy_grad, second_copy_grad


def train_once(seed: int, zero_init=True, steps=30):
    seed_everything(seed)
    backbone = FrozenDenoiser()
    model = TinyControlNet(backbone, zero_init=zero_init)
    x, condition = make_batch()
    with torch.no_grad():
        base = backbone(x)
        target = base + 0.25 * (condition - 0.5)
        initial_drift = (model(x, condition) - base).abs().max().item()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-2,
        weight_decay=0.0,
    )
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, condition)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        history.append(loss.detach().item())

    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    final = model(x, condition).detach()
    return torch.tensor(history), state, final, initial_drift


g0, c0, c1 = gradient_gate_test()
history_a, state_a, output_a, drift_zero = train_once(123, zero_init=True)
history_b, state_b, output_b, _ = train_once(123, zero_init=True)
_, _, _, drift_random = train_once(123, zero_init=False, steps=1)

assert torch.equal(history_a, history_b)
assert torch.equal(output_a, output_b)
assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
assert history_a[-1] < history_a[0]
assert drift_zero == 0.0
assert drift_random > 0.0

print(f"first zero/copy grad={g0:.6f}/{c0:.6f}")
print(f"second copy grad={c1:.6f}")
print(f"loss={history_a[0]:.6f}->{history_a[-1]:.6f}")
print(f"initial drift zero/random={drift_zero:.6f}/{drift_random:.6f}")
print("exact replay passed")
```

### 6.1 이 축소 구현에서 실제 모델로 바뀌는 부분

| 축소 구현 | 실제 diffusion 학습 |
| --- | --- |
| 2-channel toy latent | VAE latent channel 수 |
| 1-channel binary condition | Canny, depth, pose, segmentation 등 |
| timestep 없음 | sinusoidal 또는 learned timestep embedding |
| text 없음 | cross-attention context와 attention mask |
| MSE target이 base output의 이동 | sampled noise 또는 prediction type에 따른 target |
| 2개 residual level | 모든 down block residual과 middle residual |
| 작은 frozen CNN | pretrained conditional diffusion U-Net |

구조가 커져도 가장 중요한 API는 `down_block_additional_residuals`와 `mid_block_additional_residual`처럼 residual 목록과 middle tensor를 명시적으로 넘기는 방식이다. 위치 순서를 이름이나 index로 고정한다.

### 6.2 diffusion 학습 loss

`epsilon` prediction이면 정방향 noise를 target으로 사용한다.

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

$$
\mathcal{L}_{\mathrm{ctrl}}
=
\frac{1}{NCHW}
\left\|
\epsilon
-
\epsilon_{\theta,\psi}(x_t,t,c_t,c_s)
\right\|_2^2
$$

$\theta$는 frozen backbone, $\psi$는 condition encoder·trainable copy·zero projection parameter다. `v_prediction` checkpoint라면 target을 noise로 고정하면 안 된다. scheduler의 prediction type과 학습 target을 함께 저장한다.

## 7. 데이터 파이프라인 구현 계약

### 7.1 geometric transform은 항상 쌍으로 적용한다

원본 이미지와 condition은 같은 좌표계를 공유해야 한다.

```text
sample random crop parameters once
  -> crop image with bilinear interpolation
  -> crop edge/pose/label with modality-appropriate interpolation
sample horizontal flip once
  -> flip both tensors
  -> for pose, swap left/right joint channels or labels
```

RGB image와 continuous depth는 bilinear를 쓸 수 있지만 segmentation id나 one-hot edge는 nearest가 안전하다. interpolation이 condition 의미를 바꾸는지 modality별로 정한다.

### 7.2 값 범위와 missing condition

| condition | 권장 확인 | 흔한 실패 |
| --- | --- | --- |
| Canny edge | `0/1` 또는 `0/255` 중 하나로 고정 | train은 `0/1`, serving은 `0/255` |
| depth | near/far 방향, clipping, normalization | 센서마다 scale이 달라 절대 깊이 drift |
| pose | joint order, 좌우 정의, confidence | flip 뒤 left/right channel 미교환 |
| segmentation | class id와 palette 분리 | RGB palette를 연속값처럼 보간 |
| normal map | 좌표계와 단위 벡터 | `BGR/RGB`, 카메라 축 부호 불일치 |

condition이 없는 sample을 zero image로 대체할지, sample을 제외할지, 별도 presence mask를 넣을지 결정한다. zero image가 실제로 유효한 blank condition일 수 있으므로 presence 의미를 암묵적으로 만들지 않는다.

### 7.3 reproducibility

다음 RNG stream을 분리한다.

- data order와 augmentation
- timestep sampling
- diffusion noise
- text condition dropout
- dropout 또는 stochastic depth

checkpoint에는 model·optimizer·scheduler뿐 아니라 각 RNG state와 sampler position을 저장한다. batch를 다시 만드는 시점이 달라지면 seed 하나만 같아도 exact replay가 깨진다.

## 8. C++ 예제

아래 코드는 **표준 C++17로 실행 가능한 zero gate 한 step**이다. framework binding 전에 scalar golden을 고정할 때 쓸 수 있다.

```cpp
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    const std::array<double, 2> x{1.0, -2.0};
    const std::array<double, 2> target{0.5, -0.5};
    double weight = 0.0;
    double bias = 0.0;
    double grad_weight = 0.0;
    double grad_bias = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        const double output = weight * x[i] + bias;
        const double error = output - target[i];
        grad_weight += error * x[i];
        grad_bias += error;
    }

    assert(std::abs(grad_weight + 1.5) < 1e-12);
    assert(std::abs(grad_bias) < 1e-12);

    const double learning_rate = 0.1;
    weight -= learning_rate * grad_weight;
    bias -= learning_rate * grad_bias;

    std::array<double, 2> base{10.0, 20.0};
    std::array<double, 2> controlled{};
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double residual = weight * x[i] + bias;
        controlled[i] = base[i] + 2.0 * residual;
    }

    assert(std::abs(controlled[0] - 10.3) < 1e-12);
    assert(std::abs(controlled[1] - 19.4) < 1e-12);
    std::cout << std::fixed << std::setprecision(6)
              << weight << " " << controlled[0] << " " << controlled[1] << "\n";
}
```

실제 LibTorch 또는 ONNX Runtime 구현에서는 `NCHW`, contiguous 여부, float type과 residual 목록 순서를 추가로 assert한다.

## 9. C# 예제

아래 코드는 **.NET/Mono에서 실행 가능한 같은 golden**이다.

```csharp
using System;

public static class ControlNetGolden
{
    private static void Require(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    public static void Main()
    {
        double[] x = { 1.0, -2.0 };
        double[] target = { 0.5, -0.5 };
        double weight = 0.0;
        double bias = 0.0;
        double gradWeight = 0.0;
        double gradBias = 0.0;

        for (int i = 0; i < x.Length; i++)
        {
            double output = weight * x[i] + bias;
            double error = output - target[i];
            gradWeight += error * x[i];
            gradBias += error;
        }

        Require(Math.Abs(gradWeight + 1.5) < 1e-12, "weight gradient");
        Require(Math.Abs(gradBias) < 1e-12, "bias gradient");

        const double learningRate = 0.1;
        weight -= learningRate * gradWeight;
        bias -= learningRate * gradBias;

        double[] baseline = { 10.0, 20.0 };
        double[] controlled = new double[2];
        for (int i = 0; i < x.Length; i++)
        {
            double residual = weight * x[i] + bias;
            controlled[i] = baseline[i] + 2.0 * residual;
        }

        Require(Math.Abs(controlled[0] - 10.3) < 1e-12, "first output");
        Require(Math.Abs(controlled[1] - 19.4) < 1e-12, "second output");
        Console.WriteLine(
            $"{weight:F6} {controlled[0]:F6} {controlled[1]:F6}"
        );
    }
}
```

Unity/Barracuda, ML.NET 또는 ONNX Runtime C# binding으로 옮길 때 `NHWC` texture와 `NCHW` model tensor 사이의 transpose를 명시적으로 테스트한다.

## 10. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch | C++ runtime | C# runtime | 검사 |
| --- | --- | --- | --- | --- |
| latent | `NCHW`, `float16/32` | model metadata 확인 | `DenseTensor<float>` 차원 명시 | `(N,C,H,W)` |
| condition image | 보통 `NCHW` | OpenCV는 기본 `HWC/BGR` | bitmap은 보통 `HWC/BGRA` | channel order·range |
| timestep | scalar 또는 `(N,)` | `int64`/float contract | `long[]`/`float[]` contract | batch broadcast 금지 |
| text context | `NLD` | contiguous buffer | row-major buffer | `L`, `D`, mask |
| residual list | Python `tuple` | 이름별 output binding | 이름별 dictionary | index·name 고정 |
| accumulation | FP32 권장 | kernel별 precision | provider별 precision | golden tolerance |
| output | latent와 같은 shape | output metadata | output metadata | finite·dtype |

condition preprocessing을 graph 밖에서 하면 versioned manifest가 필요하다.

```json
{
  "condition_type": "edge",
  "condition_layout": "NCHW",
  "condition_channel_order": "RGB",
  "condition_range": [0.0, 1.0],
  "latent_dtype": "float16",
  "residual_accumulation_dtype": "float32",
  "residual_names": ["down_0", "down_1", "mid"],
  "control_scales": [1.0, 1.0, 1.0],
  "prediction_type": "epsilon"
}
```

이 JSON은 설명용 manifest 예시이며 특정 framework가 자동으로 읽는 표준 형식은 아니다.

## 11. 테스트와 디버깅

### 11.1 필수 unit test

1. **initial identity:** 같은 입력에서 base output과 zero-init controlled output이 정확히 같다.
2. **frozen invariant:** backward와 optimizer step 뒤 backbone parameter와 buffer가 변하지 않는다.
3. **first-gradient gate:** 첫 backward에서 zero projection gradient는 0이 아니고 copy gradient는 0이다.
4. **second-gradient opening:** 첫 optimizer step 뒤 copy gradient가 0이 아니다.
5. **shape contract:** 모든 residual이 대응 frozen feature와 정확히 같은 shape다.
6. **condition sensitivity:** 학습 뒤 condition만 바꾸면 출력이 바뀐다.
7. **zero-scale:** 모든 $s_i=0$이면 학습된 branch가 있어도 base output과 같다.
8. **checkpoint parity:** 저장·복원 전후 출력이 tolerance 안에서 같다.
9. **exact replay:** 같은 seed·data order·RNG state에서 loss history와 parameter가 같다.
10. **negative tests:** 잘못된 batch, spatial size, residual count, dtype을 조용히 broadcast하지 않고 거부한다.

### 11.2 증상에서 원인으로 가는 표

| 증상 | 먼저 확인할 원인 | 관측값 |
| --- | --- | --- |
| step 0부터 base와 다름 | zero bias 누락, random output projection, train mode 차이 | residual별 max absolute value |
| zero projection도 학습 안 됨 | residual `detach`, `no_grad`, optimizer 누락 | `grad is None`과 grad norm 구분 |
| copy가 첫 step부터 큰 gradient | projection이 실제로 0이 아님, 우회 loss 연결 | zero weight max, graph path |
| condition을 바꿔도 동일 | pair mismatch, condition gate 미학습, scale 0 | condition shuffle sensitivity |
| geometry가 밀림 | crop·resize·flip 비동기 | image-condition overlay |
| 얇은 edge가 사라짐 | bilinear resize, dtype cast, threshold 순서 | preprocessing 단계별 histogram |
| 특정 해상도에서 crash | stride rounding, residual 순서 | level별 `(C,H,W)` trace |
| FP16에서 얼룩·NaN | residual overflow, normalization, scheduler mismatch | finite check, FP32 reference |
| checkpoint가 너무 큼 | frozen backbone 중복 저장 | trainable key allowlist |

### 11.3 logging 최소 집합

- 전체 loss와 timestep bucket별 loss
- condition type·missing ratio·value histogram
- level별 residual RMS와 max absolute value
- level별 zero projection·copy gradient norm
- base 대비 controlled output delta
- control scale과 guidance scale
- peak allocated memory, step time, samples per second
- checkpoint hash, preprocessing version, dataset revision

gradient norm만 보고 품질을 판단하지 않는다. edge alignment, pose keypoint error, depth consistency와 text-image quality를 함께 기록한다.

## 12. ablation 설계

### 12.1 zero initialization 대 random initialization

같은 backbone, copied weight, optimizer, batch order와 seed를 사용하고 output projection initialization만 바꾼다.

| 항목 | zero init | random init |
| --- | --- | --- |
| step 0 base delta | 정확히 0 | 일반적으로 0이 아님 |
| 첫 copy gradient | 0 | 일반적으로 0이 아님 |
| 초기 안정성 | 기준 함수 보존 | scale에 따라 흔들림 |
| 최종 품질 | 평가 필요 | 평가 필요 |

zero init의 이점은 최종 품질의 자동 우월성이 아니라 **알려진 함수에서 시작하는 안전한 경계 조건**이다.

### 12.2 multi-scale 대 single-scale

middle residual만 쓰는 실험과 모든 down level을 쓰는 실험을 비교한다. 고해상도 residual은 윤곽 정렬에, 낮은 해상도 residual은 전역 구조에 기여할 수 있지만 이는 architecture와 dataset에 의존한다.

### 12.3 scale sweep

$s\in\{0,0.25,0.5,1,2\}$를 같은 initial noise와 prompt로 비교한다.

- $s=0$: base regression test
- 낮은 $s$: text 자유도 증가 가능
- 높은 $s$: condition alignment 증가 가능
- 지나친 $s$: texture 손상, saturation, 중복 edge 가능

CFG scale과 control scale을 동시에 바꾸면 원인을 분리하기 어렵다. 먼저 하나를 고정하고 다른 하나를 sweep한다.

## 13. 성능·메모리·수치 안정성

### 13.1 메모리 예산

frozen backbone도 backward graph가 control residual을 거쳐 입력까지 필요하면 일부 activation이 남을 수 있다. weight gradient가 없다고 activation 비용이 모두 사라지는 것은 아니다.

대략적인 peak memory를 다음 항목으로 나눠 측정한다.

$$
M_{\mathrm{peak}}
\approx
M_{\mathrm{frozen\ weights}}
+M_{\mathrm{trainable\ weights}}
+M_{\mathrm{optimizer}}
+M_{\mathrm{activations}}
+M_{\mathrm{workspace}}
$$

- frozen weight: inference dtype로 유지 가능 여부 확인
- trainable weight: master FP32 copy 존재 여부 확인
- optimizer: Adam 계열이면 moment 두 벌을 고려
- activation: resolution과 batch size에 거의 선형 증가
- workspace: attention·convolution algorithm에 따라 변동

gradient checkpointing, activation offload, memory-efficient attention을 각각 켜고 step time과 peak memory를 함께 기록한다.

### 13.2 residual 누적 precision

FP16 residual에 큰 control scale을 곱하면 overflow나 cancellation이 생길 수 있다. 안전한 경계 구현은 다음과 같다.

```python
# 설명용: residual 합산만 FP32로 수행
merged = (
    frozen_feature.float()
    + control_scale.float() * control_residual.float()
).to(frozen_feature.dtype)
```

이 방식도 최종 cast에서 정보가 줄어든다. target GPU의 fused kernel과 FP32 reference 사이 오차를 실제로 측정한다.

### 13.3 condition encoder 병목

고해상도 condition을 매 diffusion step마다 다시 encode하면 낭비다. condition encoder output이 timestep이나 latent에 독립적이면 sampling loop 밖에서 cache할 수 있다. 단, augmentation·control scale·batch가 바뀌면 cache key도 바뀌어야 한다.

여러 ControlNet을 동시에 쓸 때는 branch 수만큼 compute와 memory가 늘어난다. residual을 합칠 때 level·shape를 확인하고 branch별 scale을 기록한다.

## 14. 실무 실패 사례

### 실패 1: pose flip 뒤 왼팔과 오른팔이 뒤바뀜

이미지와 pose heatmap을 함께 horizontal flip했지만 joint channel 의미를 교환하지 않았다. 공간 위치는 맞아 보여도 semantic joint id가 틀려 학습이 불안정했다.

**교정:** transform 후 `left_shoulder <-> right_shoulder` 같은 permutation을 적용하고 colored overlay golden을 저장한다.

### 실패 2: checkpoint에 frozen U-Net까지 중복 저장

control-only 배포를 의도했지만 전체 wrapper의 `state_dict`를 저장해 artifact가 두 배 가까이 커졌다.

**교정:** trainable key allowlist와 base model revision·hash를 별도로 저장한다. load 시 base hash가 다르면 중단한다.

### 실패 3: residual index가 한 칸 밀림

export 과정에서 dictionary가 list로 바뀌며 `down_1` residual이 `down_2`에 들어갔다. channel이 우연히 같아 shape test를 통과했지만 해상도 보간이 추가되어 품질만 악화했다.

**교정:** output name, level stride, expected spatial shape를 manifest에 함께 넣고 impulse condition golden으로 위치를 확인한다.

### 실패 4: training edge와 serving edge가 다름

학습은 RGB 변환 뒤 Canny를 계산했지만 serving은 BGR buffer에 같은 threshold를 적용했다. condition density가 달라져 control strength가 크게 흔들렸다.

**교정:** preprocessor를 versioned component로 배포하고 입력 image에서 condition tensor까지 golden hash를 비교한다.

### 실패 5: 학습 재시작 뒤 갑자기 loss spike

model과 optimizer만 복원하고 data sampler와 noise RNG state를 복원하지 않았다. 같은 global step이지만 전혀 다른 timestep·noise 조합으로 이어졌다.

**교정:** sampler position, 모든 RNG state, scaler와 scheduler state를 atomic checkpoint에 포함한다.

## 15. checkpoint와 배포

### 15.1 control-only checkpoint

artifact에는 다음을 포함한다.

- condition encoder
- trainable encoder·middle copy
- 모든 zero projection
- base model identifier와 immutable revision
- text encoder·VAE·scheduler revision
- prediction type와 timestep convention
- condition preprocessing manifest
- residual name·shape·scale convention
- training dtype와 recommended inference dtype

load 후에는 missing key와 unexpected key를 무시하지 않는다. base hash mismatch는 warning이 아니라 호환성 실패로 다루는 편이 안전하다.

### 15.2 ONNX export 경계

두 가지 배포 구성이 있다.

1. **분리 graph:** ControlNet이 residual 목록을 출력하고 base U-Net이 이를 입력으로 받는다.
2. **통합 graph:** control branch와 base U-Net을 하나의 graph로 묶는다.

분리 graph는 branch 교체와 cache가 쉽지만 tensor 전달 비용과 복잡한 I/O가 늘어난다. 통합 graph는 optimization 기회가 있지만 artifact가 커지고 branch 조합이 고정된다.

dynamic axis는 batch만 열지, height·width도 열지 target runtime과 함께 결정한다. U-Net downsample factor의 배수가 아닌 해상도, residual sequence output, external data format 지원을 export 전에 확인한다.

### 15.3 배포 parity gate

고정 입력에 대해 다음 경계를 저장한다.

- preprocessed condition tensor
- level별 control residual
- controlled U-Net output
- scheduler 한 step 뒤 latent

PyTorch FP32를 reference로 하고 PyTorch mixed precision, ONNX CPU, target accelerator의 최대·평균 오차를 각각 기록한다. 최종 이미지 하나의 눈대중 비교만으로는 residual 순서 오류를 찾기 어렵다.

## 16. 구현 체크리스트

### 학습 전

- [ ] base model revision과 checksum을 고정했다.
- [ ] trainable copy가 base encoder·middle weight와 정확히 같은지 검사했다.
- [ ] 모든 output projection의 weight와 bias가 0인지 검사했다.
- [ ] frozen parameter가 optimizer에 들어가지 않았는지 검사했다.
- [ ] image와 condition의 paired augmentation golden을 확인했다.
- [ ] condition range, channel order, interpolation을 기록했다.
- [ ] prediction type과 scheduler timestep convention을 고정했다.

### 학습 중

- [ ] 첫 forward의 base delta가 0이다.
- [ ] 첫·두 번째 backward의 gradient gate test가 통과한다.
- [ ] residual RMS와 gradient norm을 level별로 기록한다.
- [ ] condition shuffle과 zero-scale negative control을 주기적으로 평가한다.
- [ ] NaN·Inf를 loss뿐 아니라 residual과 output에서 검사한다.
- [ ] checkpoint가 RNG와 sampler position까지 복원한다.

### 배포 전

- [ ] control-only checkpoint의 base hash가 일치한다.
- [ ] Python·C++·C# preprocessing golden이 일치한다.
- [ ] residual name·order·shape가 runtime manifest와 일치한다.
- [ ] mixed precision parity가 정한 tolerance를 통과한다.
- [ ] dynamic shape와 batch 경계를 negative test로 확인했다.
- [ ] scale 0이 base output으로 돌아가는지 target runtime에서 검사했다.
- [ ] latency, peak memory, cache hit rate와 품질 metric을 함께 측정했다.

## 17. 연습문제

### 문제 1

zero projection의 weight와 bias가 0인데 첫 backward에서 weight gradient가 0이 아닐 수 있는 이유를 식으로 설명하라.

### 문제 2

첫 backward에서 zero projection 앞의 copy block gradient가 0인 이유와, 언제 0이 아니게 되는지 설명하라.

### 문제 3

frozen feature가 `(2, 320, 31, 47)`이고 residual이 `(2, 320, 32, 48)`이다. broadcast가 가능한가? 올바른 교정 방법은 무엇인가?

### 문제 4

학습 뒤 condition을 shuffle해도 alignment metric이 거의 변하지 않았다. 가능한 원인 세 가지와 확인할 logging을 제시하라.

### 문제 5

PyTorch는 `NCHW/RGB/[0,1]`, C# 전처리는 `HWC/BGRA/[0,255]` bitmap을 받는다. model 입력까지 필요한 변환 순서를 쓰라.

### 문제 6

zero init과 random init ablation에서 random init run만 다른 seed를 사용했다. 이 결과로 initialization 효과를 주장할 수 있는가?

## 18. 해답

### 해답 1

다음 식에는 현재 weight 값이 곱해지지 않는다.

$$
\frac{\partial\mathcal{L}}{\partial W}
=
\sum G u^{\mathsf{T}}
$$

따라서 feature $u$와 상류 gradient $G$의 상관이 0이 아니면 $W=0$이어도 gradient가 생긴다.

### 해답 2

feature gradient에는 현재 projection weight가 곱해진다.

$$
\frac{\partial\mathcal{L}}{\partial u}
=
W^{\mathsf{T}}G
$$

첫 backward에는 $W=0$이므로 copy gradient가 0이다. 첫 optimizer step에서 $W$가 0을 벗어난 뒤 다음 backward부터 gradient가 열린다.

### 해답 3

두 spatial dimension이 모두 달라 elementwise addition도 일반 broadcasting도 허용되지 않는다. residual을 임의로 resize하기보다 두 branch가 같은 downsample rounding을 사용하도록 고치고, decoder upsample은 실제 target `size=(31,47)`을 사용한다. resize가 architecture 계약이라면 mode와 align policy를 명시하고 테스트한다.

### 해답 4

가능한 원인은 condition-image pairing 붕괴, condition scale 0, condition gate·projection의 optimizer 누락, model이 condition을 무시한 상태다. condition tensor hash, scale, residual RMS, projection gradient norm, 원래·shuffle condition의 output delta와 alignment metric을 확인한다.

### 해답 5

BGRA에서 alpha를 정책대로 제거하고 BGR을 RGB로 reorder한다. `HWC`를 `CHW`로 transpose하고 float로 변환한 뒤 255로 나눠 `[0,1]`로 만든다. 마지막에 batch dimension을 추가해 `NCHW`가 되었는지 shape와 channel별 golden pixel을 확인한다.

### 해답 6

주장할 수 없다. initialization 외에 RNG가 달라 copied base weight, batch order 또는 stochastic 연산도 달라질 수 있다. 같은 seed와 초기 backbone, data order, optimizer를 사용하고 output projection initialization만 바꿔야 한다. 여러 seed의 평균과 분산도 보고한다.

## 핵심 요약

1. ControlNet의 안전한 시작점은 pretrained copy 자체가 아니라 0으로 초기화한 output projection이 만든다.
2. step 0에는 residual이 0이므로 같은 상태의 base denoiser 함수를 보존한다.
3. 첫 backward에서 zero projection은 학습되지만 그 앞 copy는 gradient를 받지 못하고, 첫 update 뒤부터 열린다.
4. control residual은 frozen skip과 channel·height·width·dtype·device가 모두 같아야 한다.
5. image와 condition의 crop·resize·flip은 하나의 paired transform이며 modality별 interpolation이 필요하다.
6. 재현성은 seed 하나가 아니라 data, noise, timestep, dropout RNG와 sampler position의 복원 문제다.
7. zero init, multi-scale level, control scale은 같은 seed와 품질 metric으로 ablation한다.
8. control-only checkpoint에는 base revision, preprocessing, residual 순서와 prediction type이 포함되어야 한다.
9. 배포 parity는 condition tensor, level별 residual, U-Net output, scheduler output의 네 경계에서 검사한다.
10. C++·C# runtime에서는 layout·channel order·range와 residual binding 이름을 명시적으로 고정한다.

## 다음 학습 예고

다음 소스는 2회차 구현 17/18 `02-16.LoRA.md`다. 낮은 rank의 두 행렬로 weight update를 표현하고, initialization, scaling, merge·unmerge parity, optimizer state, rank ablation과 ONNX 배포 계약을 완전한 코드로 연결한다.
