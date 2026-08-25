<!-- curriculum: cycle=2; level=implementation; source_index=14/18; source=02-13.DDIM (Denoising Diffusion Implicit Models).md; part=1/1 -->

# DDIM - prediction type부터 inversion까지 재현하는 구현 계약

## 학습 진도

| 날짜 | 회차·수준 | 현재 소스 | Part | 이전 소스 | 다음 소스 |
| --- | --- | --- | --- | --- | --- |
| 2026-08-26 | 2회차 · 구현 | 14/18 · `02-13.DDIM (Denoising Diffusion Implicit Models).md` | 1/1 | `02-12.Diffusion.md` | `02-14.CFG(Classifier-Free guidance).md` |

## 학습 목표

1회차에는 DDIM의 marginal, 임의 시점 도약, $\eta$의 의미를 익혔다. 이번에는 설명을 실행 가능한 계약으로 바꾼다. 이 글을 마치면 다음을 할 수 있다.

- training index와 clean endpoint를 분리한 schedule을 만든다.
- `epsilon`, `sample`, `v_prediction` 출력을 공통 $(\hat{x}_0,\hat{\epsilon})$ 표현으로 변환한다.
- DDIM의 deterministic·stochastic step과 전체 sampling loop를 구현한다.
- 인접 step에서 $\eta=1$인 DDIM과 DDPM posterior step이 일치함을 수치 검증한다.
- deterministic replay, timestep 축소 ablation, inversion 오차를 테스트한다.
- Python·C++·C# 사이의 shape, layout, dtype, random-number 계약을 맞춘다.
- 모델과 scheduler를 한 배포 artifact처럼 versioning하고 운영 실패를 진단한다.

## 선수 지식과 기호

정규분포, DDPM 정방향 과정, 누적 곱, noise-prediction loss, PyTorch의 broadcasting을 알고 있으면 된다.

| 기호 | 정의 | 코드 shape |
| --- | --- | --- |
| $K$ | 학습 transition 수 | scalar |
| $t$ | 현재 noisy 시점, $1\leq t\leq K$ | `(N,)`, `int64` |
| $s$ | 다음 시점, $0\leq s<t$ | `(N,)`, `int64` |
| $\beta_t$ | $t-1\to t$ 정방향 분산 | schedule scalar |
| $\alpha_t$ | $1-\beta_t$ | schedule scalar |
| $\bar{\alpha}_t$ | $\prod_{i=1}^{t}\alpha_i$, 단 $\bar{\alpha}_0=1$ | `(N,1,1,1)` |
| $x_0$ | clean sample | `(N,C,H,W)` |
| $x_t$ | noisy sample | `(N,C,H,W)` |
| $m_\theta$ | 모델의 raw output | `(N,C,H,W)` |
| $\hat{x}_0$ | 모델 출력에서 복원한 clean estimate | `(N,C,H,W)` |
| $\hat{\epsilon}$ | 모델 출력에서 복원한 noise estimate | `(N,C,H,W)` |
| $\eta$ | DDIM stochasticity 계수 | scalar |
| $z$ | step마다 새로 넣는 표준 Gaussian noise | `(N,C,H,W)` |

이 문서의 schedule 배열은 길이 `K + 1`이다. index `0`은 학습 step이 아니라 clean endpoint이며 $\bar{\alpha}_0=1$이다. 이 정의를 끝까지 유지하면 마지막 step을 별도 마법 값 `-1`로 처리할 필요가 없다.

## 1. 원본을 구현 관점에서 다시 읽기

원본의 중요한 통찰은 DDPM 학습의 simple loss가 인접한 $x_{t-1}$을 직접 입력으로 요구하지 않는다는 점이다.

$$
\mathcal{L}_{\mathrm{simple}}
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$

하지만 구현에 그대로 옮기기 전에 다음 표현을 교정해야 한다.

| 원본 표현 | 구현에서의 수정 |
| --- | --- |
| DDPM은 Markov chain이라 한 step도 건너뛸 수 없다 | DDIM은 같은 학습 모델을 재사용하는 non-Markovian family와 subsequence sampler를 제시한다. Markov라는 사실만으로 모든 accelerated solver가 금지되지는 않는다. |
| $\sigma=0$이면 완벽한 ODE가 된다 | discrete deterministic DDIM update와 probability-flow ODE는 관련되지만 동일한 객체는 아니다. |
| 50 step이면 오차 없이 도착한다 | 모델 오차, 큰 timestep 간격, guidance, clipping, finite precision 때문에 품질과 reconstruction 오차를 측정해야 한다. |
| deterministic이면 정확히 reversible하다 | DDIM inversion은 수치적 근사다. 조건·guidance·모델 오차가 있으면 round-trip error가 남는다. |
| U-Net은 항상 noise를 출력한다 | checkpoint는 `epsilon`, `sample`, `v_prediction` 중 하나일 수 있다. shape가 같아도 해석이 틀리면 결과가 붕괴한다. |

이번 구현의 핵심은 마지막 행이다. scheduler는 raw model output을 곧바로 DDIM 식에 넣지 않고, 먼저 공통 표현으로 변환해야 한다.

## 2. 전체 실행 계약

한 요청의 흐름을 다음처럼 고정한다.

```text
model artifact + scheduler config
        -> descending timestep 생성
        -> model(x_t, t) = raw output
        -> prediction type 변환
        -> (pred_x0, pred_epsilon)
        -> DDIM update(t -> s, eta, RNG)
        -> finite/shape 검사
        -> x_s
```

모델 artifact와 scheduler config 사이에는 최소 다음 불변식이 필요하다.

- 같은 $\beta_t$ 또는 $\bar{\alpha}_t$ table을 사용한다.
- 학습 때 사용한 timestep 범위·scale·embedding 규칙을 유지한다.
- prediction type과 clipping·thresholding 정책을 보존한다.
- latent scaling, channel 수, normalization, VAE version을 맞춘다.
- stochastic sampling의 RNG seed와 backend를 기록한다.

## 3. 단계별 수학 유도

### 3.1 정방향 sample

정방향 폐쇄형은 다음과 같다.

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$

`alpha_bar`는 batch에 맞춰 `(N,1,1,1)`로 reshape한다. `(N,)`를 그대로 곱하면 PyTorch가 마지막 축과 broadcast하려 하므로 batch 크기와 width가 우연히 같을 때 조용히 잘못 계산될 수 있다.

### 3.2 세 prediction type을 공통 표현으로 바꾸기

#### `epsilon`

모델이 $m_\theta=\hat{\epsilon}$을 출력하면 다음과 같다.

$$
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}
}{
\sqrt{\bar{\alpha}_t}
}
$$

#### `sample`

모델이 $m_\theta=\hat{x}_0$을 직접 출력하면 noise를 역산한다.

$$
\hat{\epsilon}
=
\frac{
x_t-\sqrt{\bar{\alpha}_t}\hat{x}_0
}{
\sqrt{1-\bar{\alpha}_t}
}
$$

#### `v_prediction`

velocity target을 다음처럼 정의한다.

$$
v
=
\sqrt{\bar{\alpha}_t}\epsilon
-
\sqrt{1-\bar{\alpha}_t}x_0
$$

$x_t$ 식과 함께 두 식을 풀면 division 없이 다음 변환을 얻는다.

$$
\hat{x}_0
=
\sqrt{\bar{\alpha}_t}x_t
-
\sqrt{1-\bar{\alpha}_t}\hat{v}
$$

$$
\hat{\epsilon}
=
\sqrt{1-\bar{\alpha}_t}x_t
+
\sqrt{\bar{\alpha}_t}\hat{v}
$$

세 경로는 perfect target을 넣었을 때 동일한 $x_0$와 $\epsilon$을 복원해야 한다. 이것이 prediction type adapter의 가장 중요한 golden test다.

### 3.3 임의의 $t\to s$ DDIM step

$0\leq s<t$일 때 stochastic noise의 표준편차는 다음과 같다.

$$
\sigma_{t\to s}
=
\eta
\sqrt{
\frac{1-\bar{\alpha}_s}{1-\bar{\alpha}_t}
}
\sqrt{
1-\frac{\bar{\alpha}_t}{\bar{\alpha}_s}
}
$$

다음 상태는 세 항의 합이다.

$$
x_s
=
\sqrt{\bar{\alpha}_s}\hat{x}_0
+
\sqrt{
1-\bar{\alpha}_s-\sigma_{t\to s}^2
}\hat{\epsilon}
+
\sigma_{t\to s}z
$$

여기서 $z\sim\mathcal{N}(0,I)$다. $\eta=0$이면 같은 초기 noise, condition, 모델, timestep, backend에서 update가 deterministic하다. 시작 noise까지 사라지는 것은 아니다.

### 3.4 clean endpoint

$s=0$, $\bar{\alpha}_0=1$을 대입하면 다음과 같다.

$$
\sigma_{t\to0}=0
$$

$$
\sqrt{1-\bar{\alpha}_0-\sigma_{t\to0}^2}=0
$$

따라서 마지막 출력은 다음과 같다.

$$
x_0=\hat{x}_0
$$

분기문 없이도 식 자체가 clean endpoint를 처리한다. 단, table index `0`의 의미를 model의 training timestep `0`과 섞지 않아야 한다.

### 3.5 인접 step에서 DDPM과의 parity

$s=t-1$, $\eta=1$이면 DDIM 분산은 DDPM posterior variance가 된다.

$$
\sigma_{t\to t-1}^2
=
\beta_t
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}
=
\tilde{\beta}_t
$$

동일한 $\hat{x}_0$와 동일한 $z$를 쓰면 DDIM update와 DDPM posterior mean·variance update가 수치 오차 안에서 같아야 한다. 이 검사는 두 구현이 같은 schedule index를 읽는지 한 번에 확인한다.

## 4. tensor shape 추적

latent diffusion의 batch를 예로 든다.

| 단계 | tensor | shape | dtype·layout |
| --- | --- | --- | --- |
| 1 | `x_t` | `(N,4,H/8,W/8)` | model dtype, `NCHW` |
| 2 | `t`, `s` | `(N,)` | `int64` |
| 3 | condition | `(N,L,D)` | model dtype |
| 4 | raw model output | `(N,4,H/8,W/8)` | `x_t`와 같음 |
| 5 | `alpha_t`, `alpha_s` | `(N,1,1,1)` | 계산 dtype |
| 6 | `pred_x0`, `pred_epsilon` | `(N,4,H/8,W/8)` | 계산 dtype |
| 7 | `sigma`, direction scale | `(N,1,1,1)` | 계산 dtype |
| 8 | `x_s` | `(N,4,H/8,W/8)` | model dtype |
| 9 | VAE decode | `(N,3,H,W)` | decoder 계약 |

classifier-free guidance는 보통 U-Net batch를 `(2N,C,H,W)`로 늘린다. unconditional·conditional 출력을 `(N,C,H,W)` 하나로 결합한 뒤 scheduler에 전달해야 한다. 다음 소스에서 이 결합을 구현한다.

## 5. NumPy 수작업·parity 검증

다음 코드는 **실행 가능**하다. 세 prediction type의 round trip과 scalar DDIM step을 독립적으로 검증한다.

```python
import numpy as np


def convert_output(x_t, output, alpha_bar, prediction_type):
    sqrt_a = np.sqrt(alpha_bar)
    sqrt_oma = np.sqrt(1.0 - alpha_bar)
    if prediction_type == "epsilon":
        eps = output
        x0 = (x_t - sqrt_oma * eps) / sqrt_a
    elif prediction_type == "sample":
        x0 = output
        eps = (x_t - sqrt_a * x0) / sqrt_oma
    elif prediction_type == "v_prediction":
        x0 = sqrt_a * x_t - sqrt_oma * output
        eps = sqrt_oma * x_t + sqrt_a * output
    else:
        raise ValueError(prediction_type)
    return x0, eps


x0 = np.array([0.25, -0.50], dtype=np.float64)
eps = np.array([0.75, -1.25], dtype=np.float64)
alpha_bar = 0.64
x_t = np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps
targets = {
    "epsilon": eps,
    "sample": x0,
    "v_prediction": np.sqrt(alpha_bar) * eps
    - np.sqrt(1.0 - alpha_bar) * x0,
}

for kind, output in targets.items():
    got_x0, got_eps = convert_output(x_t, output, alpha_bar, kind)
    np.testing.assert_allclose(got_x0, x0, atol=1e-12)
    np.testing.assert_allclose(got_eps, eps, atol=1e-12)

alpha_t, alpha_s, eta = 0.64, 0.81, 0.5
sigma_sq = (
    eta**2
    * (1.0 - alpha_s)
    / (1.0 - alpha_t)
    * (1.0 - alpha_t / alpha_s)
)
pred_x0, pred_eps = convert_output(-0.2, -1.0, alpha_t, "epsilon")
x_s = (
    np.sqrt(alpha_s) * pred_x0
    + np.sqrt(1.0 - alpha_s - sigma_sq) * pred_eps
    + np.sqrt(sigma_sq) * 0.25
)
np.testing.assert_allclose(pred_x0, 0.5, atol=1e-12)
print(f"sigma={np.sqrt(sigma_sq):.9f} x_s={x_s:.9f}")
```

검증 출력은 다음과 같다.

```text
sigma=0.166409266 x_s=0.088727701
```

## 6. PyTorch 완전 구현

다음 코드는 **실행 가능**하다. 작은 이미지 denoiser의 학습·sampling·재현성 test와 DDPM parity를 한 파일에 포함한다. production 모델 대신 $8\times8$ toy pattern을 써 scheduler 오류가 GPU 비용에 가려지지 않게 했다.

```python
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class Schedule:
    betas: torch.Tensor
    alpha_bars: torch.Tensor

    @classmethod
    def linear(cls, steps, beta_start=1e-4, beta_end=2e-2):
        betas = torch.linspace(
            beta_start, beta_end, steps, dtype=torch.float64
        )
        if steps < 2 or not bool(torch.all((0 < betas) & (betas < 1))):
            raise ValueError("invalid beta schedule")
        alpha_bars = torch.cat([
            torch.ones(1, dtype=torch.float64),
            torch.cumprod(1.0 - betas, dim=0),
        ])
        return cls(betas, alpha_bars)

    def inference_timesteps(self, transitions):
        total = self.betas.numel()
        if not (1 <= transitions <= total):
            raise ValueError("invalid transition count")
        times = torch.round(
            torch.linspace(total, 0, transitions + 1, dtype=torch.float64)
        ).long()
        if not bool(torch.all(times[:-1] > times[1:])):
            raise AssertionError("timesteps must be strictly descending")
        return times


def extract(table, t, like):
    if t.dtype != torch.long or t.shape != (like.shape[0],):
        raise ValueError("t must be int64 with shape (N,)")
    return table.to(like.device).gather(0, t).reshape(
        -1, *([1] * (like.ndim - 1))
    ).to(like.dtype)


def target_from_epsilon(x0, epsilon, alpha_bar, prediction_type):
    sqrt_a = torch.sqrt(alpha_bar)
    sqrt_oma = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    if prediction_type == "epsilon":
        return epsilon
    if prediction_type == "sample":
        return x0
    if prediction_type == "v_prediction":
        return sqrt_a * epsilon - sqrt_oma * x0
    raise ValueError(prediction_type)


def convert_model_output(x_t, output, alpha_bar, prediction_type):
    sqrt_a = torch.sqrt(alpha_bar.clamp_min(1e-20))
    sqrt_oma = torch.sqrt((1.0 - alpha_bar).clamp_min(1e-20))
    if prediction_type == "epsilon":
        epsilon = output
        x0 = (x_t - sqrt_oma * epsilon) / sqrt_a
    elif prediction_type == "sample":
        x0 = output
        epsilon = (x_t - sqrt_a * x0) / sqrt_oma
    elif prediction_type == "v_prediction":
        x0 = sqrt_a * x_t - sqrt_oma * output
        epsilon = sqrt_oma * x_t + sqrt_a * output
    else:
        raise ValueError(prediction_type)
    return x0, epsilon


def ddim_step(x_t, output, t, s, schedule, prediction_type, eta, noise=None):
    if x_t.shape != output.shape or t.shape != s.shape:
        raise ValueError("shape contract failed")
    if not bool(torch.all(t > s)) or not (0.0 <= eta <= 1.0):
        raise ValueError("invalid transition")
    alpha_t = extract(schedule.alpha_bars, t, x_t)
    alpha_s = extract(schedule.alpha_bars, s, x_t)
    if not bool(torch.all(alpha_s > alpha_t)):
        raise ValueError("alpha_bar order failed")
    pred_x0, pred_epsilon = convert_model_output(
        x_t, output, alpha_t, prediction_type
    )
    sigma_sq = (
        eta**2
        * (1.0 - alpha_s)
        / (1.0 - alpha_t).clamp_min(1e-20)
        * (1.0 - alpha_t / alpha_s.clamp_min(1e-20))
    )
    if float(sigma_sq.min()) < -1e-6:
        raise FloatingPointError("negative DDIM variance")
    sigma_sq = sigma_sq.clamp_min(0.0)
    direction_sq = 1.0 - alpha_s - sigma_sq
    if float(direction_sq.min()) < -1e-6:
        raise FloatingPointError("negative direction variance")
    if noise is None:
        noise = torch.zeros_like(x_t)
    x_s = (
        torch.sqrt(alpha_s) * pred_x0
        + torch.sqrt(direction_sq.clamp_min(0.0)) * pred_epsilon
        + torch.sqrt(sigma_sq) * noise
    )
    return x_s, pred_x0, pred_epsilon


def ddpm_posterior_step(x_t, pred_x0, t, schedule, noise):
    alpha_bar_t = extract(schedule.alpha_bars, t, x_t)
    alpha_bar_prev = extract(schedule.alpha_bars, t - 1, x_t)
    beta_t = extract(schedule.betas, t - 1, x_t)
    alpha_t = 1.0 - beta_t
    variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_prev)
        / (1.0 - alpha_bar_t)
    )
    mean = coef_x0 * pred_x0 + coef_xt * x_t
    return mean + torch.sqrt(variance.clamp_min(0.0)) * noise


class TinyDenoiser(nn.Module):
    def __init__(self, steps, width=24):
        super().__init__()
        self.time = nn.Embedding(steps + 1, width)
        self.net = nn.Sequential(
            nn.Conv2d(1 + width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x, t):
        emb = self.time(t).reshape(x.shape[0], -1, 1, 1)
        emb = emb.expand(-1, -1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, emb], dim=1))


def make_patterns(count):
    x = torch.full((count, 1, 8, 8), -1.0)
    for i in range(count):
        if i % 2 == 0:
            x[i, 0, :, 2:4] = 1.0
        else:
            x[i, 0, 4:6, :] = 1.0
    return x


def train_once(seed, prediction_type="epsilon"):
    seed_everything(seed)
    schedule = Schedule.linear(20, 1e-4, 0.15)
    model = TinyDenoiser(20)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    data = make_patterns(64)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    for _ in range(120):
        indices = torch.randint(0, 64, (16,), generator=generator)
        x0 = data[indices]
        t = torch.randint(1, 21, (16,), generator=generator)
        epsilon = torch.randn(x0.shape, generator=generator)
        alpha_bar = extract(schedule.alpha_bars, t, x0)
        x_t = (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * epsilon
        )
        target = target_from_epsilon(
            x0, epsilon, alpha_bar, prediction_type
        )
        loss = torch.mean((model(x_t, t) - target) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return model.eval(), schedule, history


@torch.no_grad()
def sample(model, schedule, initial_noise, transitions, prediction_type, eta, seed):
    x = initial_noise.clone()
    times = schedule.inference_timesteps(transitions)
    generator = torch.Generator().manual_seed(seed)
    for t_scalar, s_scalar in zip(times[:-1], times[1:]):
        t = torch.full((x.shape[0],), int(t_scalar), dtype=torch.long)
        s = torch.full_like(t, int(s_scalar))
        output = model(x, t)
        noise = (
            torch.randn(x.shape, generator=generator)
            if eta > 0
            else None
        )
        x, _, _ = ddim_step(
            x, output, t, s, schedule, prediction_type, eta, noise
        )
    return x


schedule = Schedule.linear(20, 1e-4, 0.15)
assert torch.equal(
    schedule.inference_timesteps(5),
    torch.tensor([20, 16, 12, 8, 4, 0]),
)

# 세 prediction type round trip
x0 = torch.tensor([[[[0.25, -0.50]]]], dtype=torch.float64)
epsilon = torch.tensor([[[[0.75, -1.25]]]], dtype=torch.float64)
t = torch.tensor([13])
alpha_bar = extract(schedule.alpha_bars, t, x0)
x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * epsilon
for kind in ("epsilon", "sample", "v_prediction"):
    output = target_from_epsilon(x0, epsilon, alpha_bar, kind)
    got_x0, got_eps = convert_model_output(x_t, output, alpha_bar, kind)
    torch.testing.assert_close(got_x0, x0, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(got_eps, epsilon, atol=1e-12, rtol=1e-12)

# eta=1, 인접 step의 DDPM·DDIM parity
s = t - 1
z = torch.tensor([[[[-0.3, 0.2]]]], dtype=torch.float64)
ddim, pred_x0, _ = ddim_step(
    x_t, epsilon, t, s, schedule, "epsilon", 1.0, z
)
ddpm = ddpm_posterior_step(x_t, pred_x0, t, schedule, z)
torch.testing.assert_close(ddim, ddpm, atol=1e-12, rtol=1e-12)

# 학습과 exact replay
model_a, schedule_a, history_a = train_once(20260826)
model_b, _, history_b = train_once(20260826)
assert history_a == history_b
for key, value in model_a.state_dict().items():
    torch.testing.assert_close(value, model_b.state_dict()[key], atol=0, rtol=0)

initial = torch.randn(
    (4, 1, 8, 8), generator=torch.Generator().manual_seed(9)
)
out20 = sample(model_a, schedule_a, initial, 20, "epsilon", 0.0, 44)
out5 = sample(model_a, schedule_a, initial, 5, "epsilon", 0.0, 44)
out20_repeat = sample(
    model_a, schedule_a, initial, 20, "epsilon", 0.0, 999
)
torch.testing.assert_close(out20, out20_repeat, atol=0, rtol=0)
assert torch.isfinite(out5).all() and torch.isfinite(out20).all()

print(f"loss={history_a[0]:.6f}->{history_a[-1]:.6f}")
print(f"ddpm_ddim_max_error={(ddpm-ddim).abs().max().item():.3e}")
print(f"sample_5_mean={out5.mean().item():.6f}")
print(f"sample_20_mean={out20.mean().item():.6f}")
print(f"step_ablation_l1={(out5-out20).abs().mean().item():.6f}")
print("replay=exact")
```

로컬 실행 결과는 다음과 같다.

```text
loss=1.141575->0.346779
ddpm_ddim_max_error=2.220e-16
sample_5_mean=-0.301970
sample_20_mean=-0.300839
step_ablation_l1=0.047757
replay=exact
```

`step_ablation_l1`은 5-step과 20-step 결과의 평균 절대 차이다. 이것은 FID나 선호도 같은 품질 지표가 아니다. 같은 초기 noise에서 timestep 축소가 trajectory를 바꿨는지 확인하는 회귀 지표다.

## 7. inversion을 테스트 가능한 문제로 바꾸기

$\eta=0$에서 $x_s$로부터 더 noisy한 $x_t$로 이동하는 흔한 근사는 다음과 같다.

$$
\tilde{x}_t
=
\sqrt{\bar{\alpha}_t}
\hat{x}_0(x_s,s)
+
\sqrt{1-\bar{\alpha}_t}
\hat{\epsilon}(x_s,s)
$$

이 식은 future point $x_t$에서 평가한 noise predictor를 정확히 역으로 푸는 것이 아니다. 현재 point $x_s$에서의 예측을 재사용하는 수치 step이다. 따라서 다음 두 test를 분리한다.

1. **oracle test:** 고정된 진짜 $(x_0,\epsilon)$을 쓰면 어떤 subsequence에서도 round trip이 tolerance 안에서 맞아야 한다.
2. **model test:** 학습된 denoiser로 `x0 -> inverted latent -> reconstructed x0`의 MAE·LPIPS·identity metric을 측정한다.

편집 시스템은 reconstruction metric뿐 아니라 editability도 본다. inversion 오차가 작아도 prompt를 바꿨을 때 구조가 고정되거나, 반대로 편집은 잘 되지만 identity가 무너질 수 있다.

## 8. C++17 golden 구현

다음 코드는 **실행 가능**하다. tensor runtime을 붙이기 전에 `v_prediction` 변환과 deterministic step을 scalar double로 검증한다.

```cpp
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

struct Prediction {
    double x0;
    double epsilon;
};

Prediction ConvertV(double xt, double v, double alpha_bar) {
    if (!(0.0 < alpha_bar && alpha_bar < 1.0)) {
        throw std::invalid_argument("alpha_bar must be in (0, 1)");
    }
    const double sqrt_a = std::sqrt(alpha_bar);
    const double sqrt_oma = std::sqrt(1.0 - alpha_bar);
    return {
        sqrt_a * xt - sqrt_oma * v,
        sqrt_oma * xt + sqrt_a * v,
    };
}

double DdimDeterministic(
    const Prediction& pred,
    double alpha_bar_s) {
    return std::sqrt(alpha_bar_s) * pred.x0 +
           std::sqrt(1.0 - alpha_bar_s) * pred.epsilon;
}

int main() {
    const double x0 = 0.25;
    const double epsilon = 0.75;
    const double alpha_t = 0.64;
    const double xt = std::sqrt(alpha_t) * x0 +
                      std::sqrt(1.0 - alpha_t) * epsilon;
    const double v = std::sqrt(alpha_t) * epsilon -
                     std::sqrt(1.0 - alpha_t) * x0;
    const Prediction pred = ConvertV(xt, v, alpha_t);
    assert(std::abs(pred.x0 - x0) < 1e-12);
    assert(std::abs(pred.epsilon - epsilon) < 1e-12);
    const double xs = DdimDeterministic(pred, 0.81);
    assert(std::abs(xs - (0.9 * 0.25 + std::sqrt(0.19) * 0.75)) < 1e-12);
    std::cout << std::fixed << std::setprecision(9)
              << "x0=" << pred.x0
              << " eps=" << pred.epsilon
              << " xs=" << xs << '\n';
}
```

예상 출력은 `x0=0.250000000 eps=0.750000000 xs=0.551917421`이다.

LibTorch나 ONNX Runtime C++ API를 붙일 때 model output tensor는 `NCHW`, coefficient는 `{N,1,1,1}`로 만든다. host `double` table을 매 step GPU로 복사하지 말고 device tensor를 한 번 cache한다.

## 9. C# golden 구현

다음 코드는 **실행 가능**하다. C++과 같은 scalar oracle을 독립 구현한다.

```csharp
using System;

public readonly struct Prediction
{
    public readonly double X0;
    public readonly double Epsilon;

    public Prediction(double x0, double epsilon)
    {
        X0 = x0;
        Epsilon = epsilon;
    }
}

public static class Program
{
    public static Prediction ConvertV(double xt, double v, double alphaBar)
    {
        if (!(0.0 < alphaBar && alphaBar < 1.0))
            throw new ArgumentException("alphaBar must be in (0, 1)");
        double sqrtA = Math.Sqrt(alphaBar);
        double sqrtOneMinusA = Math.Sqrt(1.0 - alphaBar);
        return new Prediction(
            sqrtA * xt - sqrtOneMinusA * v,
            sqrtOneMinusA * xt + sqrtA * v);
    }

    public static double DdimDeterministic(
        Prediction pred,
        double alphaBarS)
    {
        return Math.Sqrt(alphaBarS) * pred.X0 +
               Math.Sqrt(1.0 - alphaBarS) * pred.Epsilon;
    }

    public static void Main()
    {
        double x0 = 0.25;
        double epsilon = 0.75;
        double alphaT = 0.64;
        double xt = Math.Sqrt(alphaT) * x0 +
                    Math.Sqrt(1.0 - alphaT) * epsilon;
        double v = Math.Sqrt(alphaT) * epsilon -
                   Math.Sqrt(1.0 - alphaT) * x0;
        Prediction pred = ConvertV(xt, v, alphaT);
        if (Math.Abs(pred.X0 - x0) >= 1e-12 ||
            Math.Abs(pred.Epsilon - epsilon) >= 1e-12)
            throw new Exception("prediction conversion failed");
        double xs = DdimDeterministic(pred, 0.81);
        double expected = 0.9 * 0.25 + Math.Sqrt(0.19) * 0.75;
        if (Math.Abs(xs - expected) >= 1e-12)
            throw new Exception("DDIM step failed");
        Console.WriteLine(
            $"x0={pred.X0:F9} eps={pred.Epsilon:F9} xs={xs:F9}");
    }
}
```

C# ONNX Runtime의 timestep input은 export 계약에 따라 `DenseTensor<long>`이 필요할 수 있다. 이미지가 Unity texture의 `HWC`·`byte`라면 `NCHW`·`float` 변환과 latent scaling을 scheduler 밖의 전처리 계약으로 분리한다.

## 10. 프레임워크 간 shape·layout·dtype 대응

| 개념 | PyTorch Python | C++ | C# |
| --- | --- | --- | --- |
| latent | `torch.Tensor`, `NCHW` | LibTorch tensor 또는 ORT value, `NCHW` | `DenseTensor<float>`, `NCHW` |
| timestep | `torch.long`, `(N,)` | `int64_t`, `{N}` | `long`, `{N}` |
| schedule 생성 | `float64` 후 model dtype cast | host `double` 후 device cache | `double[]` 후 input dtype cast |
| model output | input과 같은 shape | output name·rank 검사 | named output·dimensions 검사 |
| RNG | `torch.Generator` | backend generator 또는 host noise input | seed 고정 Gaussian generator |
| layout 변환 | 필요할 때만 `permute` | stride·contiguous 검사 | texture `HWC`에서 명시적 transpose |
| half precision | scheduler invariant 별도 검사 | FP16 kernel 지원 확인 | provider별 FP16 지원 확인 |

`System.Random.NextDouble()`은 균일분포다. Gaussian $z$가 필요하면 Box-Muller 같은 변환을 검증하거나 backend의 normal RNG를 사용해야 한다. 서로 다른 언어에서 같은 seed가 같은 Gaussian sequence를 준다고 가정하지 말고, cross-language golden test에서는 noise tensor 자체를 fixture로 저장한다.

## 11. 테스트·디버깅 전략

### 11.1 필수 단위 테스트

- schedule 길이가 `K + 1`이고 `alpha_bars[0] == 1`인지 확인한다.
- timestep이 endpoint를 포함하며 strict descending인지 확인한다.
- 세 prediction type이 동일한 $(x_0,\epsilon)$을 복원하는지 확인한다.
- $\eta=0$에서 sampler가 RNG seed와 무관하게 exact replay되는지 확인한다.
- $\eta>0$에서 같은 noise fixture는 같고 다른 fixture는 달라지는지 확인한다.
- $s=t-1$, $\eta=1$에서 DDPM·DDIM 결과를 비교한다.
- 마지막 $s=0$에서 direction·noise 항이 0인지 확인한다.
- batch size 1과 3, 홀수 spatial shape에서 broadcasting을 확인한다.
- empty batch는 허용할지 fail-fast할지 API 계약을 정한다.

### 11.2 property test

무작위 schedule·shape에 대해 다음 불변식을 검사한다.

$$
0<\bar{\alpha}_t<\bar{\alpha}_s\leq1
$$

$$
\sigma_{t\to s}^2\geq0
$$

$$
1-\bar{\alpha}_s-\sigma_{t\to s}^2\geq0
$$

작은 음수는 floating-point 반올림일 수 있지만, `-1e-3` 같은 큰 위반까지 `clamp(0)`로 숨기면 schedule index 오류를 놓친다. tolerance 밖이면 먼저 실패시키고, tolerance 안의 값만 0으로 clamp한다.

### 11.3 증상별 첫 확인

| 증상 | 가능한 원인 | 첫 검사 |
| --- | --- | --- |
| 첫 step에서 `NaN` | alpha 순서 반대, 분모 0 | `t > s`, `alpha_s > alpha_t` |
| 형체는 있지만 색이 포화됨 | prediction type·latent scale 오류 | checkpoint scheduler config |
| batch 1만 정상 | coefficient broadcast 오류 | `(N,1,1,1)` shape |
| 같은 seed인데 결과가 다름 | 초기 noise, condition, backend nondeterminism | request replay manifest |
| 5-step 결과가 크게 달라짐 | 큰 discretization·model error | 5·10·20-step ablation |
| inversion 후 identity 손실 | guidance·prompt·clipping 불일치 | round-trip 구성 전체 diff |
| Python·C# 결과 불일치 | RNG나 dtype·layout 차이 | 고정 noise tensor golden |

## 12. ablation과 재현성

이번 toy 실행은 다음 두 질문만 답한다.

| 실험 | 결과 | 해석 |
| --- | ---: | --- |
| 첫 loss → 마지막 loss | `1.141575 → 0.346779` | 학습 loop가 gradient를 전달했다. 품질 보장은 아니다. |
| DDPM·DDIM 최대 오차 | `2.220e-16` | `float64` 인접-step parity 통과 |
| 5-step sample mean | `-0.301970` | regression fingerprint |
| 20-step sample mean | `-0.300839` | regression fingerprint |
| 5-step 대 20-step L1 | `0.047757` | timestep 축소가 trajectory를 바꿈 |
| 동일 seed 재학습 | exact | history와 state dict 동일 |
| $\eta=0$ sampler 반복 | exact | step RNG seed가 결과에 영향 없음 |

실제 모델에서는 step별 latency p50/p95, peak VRAM, FID·CLIP score·task metric, blind preference, reconstruction metric을 함께 기록한다. 한 지표만으로 배포 sampler를 고르지 않는다.

## 13. 성능·메모리·수치 안정성

### 13.1 계산량

U-Net 한 번의 비용을 $C_\theta$라 하면 $S$ transition의 주된 비용은 다음과 같다.

$$
\operatorname{Cost}\approx S C_\theta
$$

그러나 wall-clock speedup은 학습 step 수와 inference step 수의 단순 비가 아니다. text encoder, VAE, scheduler kernel launch, memory transfer, safety checker 같은 고정 비용이 남는다.

### 13.2 메모리

일반 생성에는 현재 `x_t`, raw output, `pred_x0`, `pred_epsilon`, `x_s`만 있으면 된다. 모든 중간 latent를 보관하면 메모리가 $O(SNCHW)$로 늘어난다. UI preview나 inversion 분석이 필요할 때만 지정된 step을 저장한다.

classifier-free guidance의 unconditional·conditional 입력을 batch로 합치면 U-Net 호출은 한 번이지만 activation peak가 커진다. 별도 호출은 peak memory를 줄일 수 있으나 latency가 늘 수 있다.

### 13.3 precision

- `beta`와 누적 곱은 `float64`로 만든 뒤 runtime dtype으로 cast한다.
- $\bar{\alpha}_t$가 매우 작을 때 `epsilon -> x0` division이 model error를 증폭한다.
- FP16에서 $1-\bar{\alpha}_t$가 0으로 반올림될 수 있다.
- schedule invariant를 `float64`로 검사하고 elementwise update는 최소 `float32`와 비교한다.
- `pred_x0` clipping과 dynamic thresholding은 안전장치이면서 sampler 의미를 바꾸는 설정이다.
- mixed precision parity는 절대 오차뿐 아니라 최종 품질 metric으로 평가한다.

## 14. 실무 실패 사례

### 사례 1: prediction type만 빠진 C# 포트

Python scheduler는 config에서 `v_prediction`을 읽었지만 C# host는 output을 noise로 해석했다. tensor shape와 dtype이 같아서 예외가 나지 않았고, 이미지는 여러 step 뒤에야 무너졌다. model manifest에 prediction type을 필수 enum으로 넣고 golden latent 한 step을 CI에서 비교해야 한다.

### 사례 2: clean endpoint off-by-one

Python table은 `0..K-1` training index였고 C++ 코드는 index `0`을 $\bar{\alpha}_0=1$로 가정했다. 두 코드가 같은 숫자 index에 다른 SNR을 사용했다. manifest에 `alpha_bars` hash뿐 아니라 index convention을 기록하고 endpoint test를 둬야 한다.

### 사례 3: deterministic을 bitwise portable로 오해

$\eta=0$은 scheduler가 새 noise를 넣지 않는다는 뜻이다. GPU architecture, attention kernel, compiler, mixed precision이 달라지면 bitwise 결과는 달라질 수 있다. 재현성 SLO를 exact bytes, latent tolerance, perceptual hash 중 하나로 명시한다.

### 사례 4: inversion latent를 원본 백업으로 사용

round-trip이 한 데이터셋에서 좋아 보여 원본을 보관하지 않았다. model update, prompt change, guidance change 뒤 복원이 달라졌다. inversion은 편집 초기값이지 lossless archive가 아니다.

### 사례 5: step 축소가 p95를 개선하지 못함

U-Net 호출 수는 줄었지만 요청마다 schedule tensor를 CPU에서 GPU로 복사하고, 작은 kernel을 동기 실행했다. alpha table과 timestep을 device에 cache하고 profiler trace로 transfer·synchronization을 분리해야 한다.

## 15. 배포·모니터링 관점

model bundle에는 다음 manifest를 포함한다.

```json
{
  "model_sha256": "...",
  "vae_sha256": "...",
  "scheduler_version": "ddim-host-v2",
  "alpha_bars_sha256": "...",
  "index_convention": "clean_zero_training_1_to_K",
  "prediction_type": "epsilon",
  "latent_layout": "NCHW",
  "latent_dtype": "float16",
  "timestep_dtype": "int64",
  "clip_sample": false
}
```

요청 log에는 model·scheduler version, timestep 배열 또는 spacing 정책, step 수, $\eta$, seed, guidance, resolution, batch, backend와 precision을 남긴다. 개인 prompt나 원본 이미지는 privacy 정책에 따라 원문 대신 hash·category 같은 최소 정보만 기록한다.

운영 metric은 다음을 포함한다.

- end-to-end latency p50/p95/p99와 U-Net step latency
- step 수·resolution·batch별 throughput과 peak VRAM
- OOM, timeout, `NaN`, safety rejection 비율
- scheduler/model manifest mismatch 거부 횟수
- 고정 prompt·seed canary의 latent tolerance와 품질 drift
- inversion·editing 서비스의 round-trip error 분포

rollout은 golden one-step parity, 고정 canary, shadow traffic, 소규모 canary, 전체 배포 순으로 진행한다. scheduler만 바뀌어도 생성 결과가 바뀌므로 model 변경과 같은 수준으로 versioning한다.

## 16. 체크리스트

### 수학·구현

- [ ] `alpha_bars[0] == 1`과 index convention을 명시했다.
- [ ] timestep이 endpoint를 포함하고 strict descending이다.
- [ ] prediction type을 manifest에서 읽고 enum 외 값은 거부한다.
- [ ] `epsilon`, `sample`, `v_prediction` round trip test가 있다.
- [ ] coefficient shape가 `(N,1,1,1)`이다.
- [ ] $\eta=1$ 인접-step DDPM parity를 확인했다.
- [ ] 마지막 clean endpoint를 golden test로 검증했다.
- [ ] negative variance를 큰 clamp로 숨기지 않는다.

### 학습·평가

- [ ] data RNG, timestep RNG, noise RNG를 재현 가능하게 관리한다.
- [ ] 동일 seed의 loss history와 state dict replay를 확인했다.
- [ ] step·spacing·$\eta$ ablation에서 latency와 품질을 함께 본다.
- [ ] inversion은 oracle과 learned-model test를 분리한다.
- [ ] clipping·thresholding 정책을 실험 설정에 기록한다.

### 배포·운영

- [ ] model·VAE·scheduler·alpha table hash를 함께 배포한다.
- [ ] Python·C++·C# one-step golden fixture를 공유한다.
- [ ] RNG sequence 대신 고정 noise tensor로 cross-language parity를 본다.
- [ ] backend·precision별 재현성 SLO를 정의했다.
- [ ] canary와 rollback 가능한 scheduler version을 유지한다.

## 17. 연습문제

### 문제 1

$\bar{\alpha}_t=0.64$, $x_t=0.65$, `v_prediction` output이 $v=0.45$일 때 $\hat{x}_0$와 $\hat{\epsilon}$을 구하라.

### 문제 2

길이 `K + 1`인 `alpha_bars`에서 `alpha_bars[0]`을 1로 두는 이유와 training timestep 범위를 쓰라.

### 문제 3

$s=t-1$, $\eta=1$ parity test가 잡을 수 있는 버그를 세 가지 쓰라.

### 문제 4

$\eta=0$ sampler가 서로 다른 RNG seed에도 같은 결과를 냈다. 이것이 정상인 이유와 그래도 달라질 수 있는 입력 두 가지를 쓰라.

### 문제 5

5-step과 20-step 출력의 L1 차이가 작으면 5-step sampler를 바로 배포해도 되는가? 필요한 추가 평가를 쓰라.

### 문제 6

Python과 C#에서 같은 integer seed를 사용했지만 stochastic DDIM 결과가 다르다. 안전한 cross-language parity 방법은 무엇인가?

## 18. 해답

### 해답 1

$\sqrt{\bar{\alpha}_t}=0.8$, $\sqrt{1-\bar{\alpha}_t}=0.6$이다.

$$
\hat{x}_0
=
0.8(0.65)-0.6(0.45)
=
0.25
$$

$$
\hat{\epsilon}
=
0.6(0.65)+0.8(0.45)
=
0.75
$$

### 해답 2

index `0`을 clean endpoint로 두면 $s=0$인 마지막 DDIM 식이 자연스럽게 $x_0=\hat{x}_0$가 된다. 이 문서의 model training timestep은 `1..K`다.

### 해답 3

`beta[t]`와 `beta[t-1]` 혼동, clean endpoint 때문에 생긴 alpha table offset, posterior variance 또는 DDIM sigma 식 오류를 잡을 수 있다. 동일한 noise fixture를 쓰지 않은 RNG wiring 오류도 드러난다.

### 해답 4

$\eta=0$이면 step마다 새 $z$를 사용하지 않으므로 sampler 내부 generator seed는 결과에 영향이 없다. 초기 noise, condition·guidance, model weight, timestep 배열, backend 연산이 달라지면 결과는 달라질 수 있다.

### 해답 5

안 된다. L1은 같은 초기 noise의 trajectory 차이일 뿐 품질 지표가 아니다. latency p50/p95, peak memory, FID·CLIP·task metric, human preference, safety와 여러 seed의 분산을 함께 비교한다.

### 해답 6

언어별 RNG algorithm과 Gaussian 변환이 다를 수 있다. seed만 공유하지 말고 미리 생성한 noise tensor를 fixture로 저장해 각 구현에 같은 값·shape·dtype으로 입력한다.

## 핵심 요약

- DDIM 구현은 raw model output을 prediction type에 맞게 $(\hat{x}_0,\hat{\epsilon})$으로 변환한 뒤 시작한다.
- clean endpoint를 index `0`, training transition을 `1..K`로 정의하면 off-by-one을 줄일 수 있다.
- $\eta=0$은 scheduler trajectory를 deterministic하게 하고, $\eta=1$의 인접 step은 DDPM posterior와 parity를 이룬다.
- 세 prediction type round trip, DDPM parity, endpoint, exact replay가 scheduler CI의 핵심이다.
- 적은 step은 U-Net 호출을 줄이지만 품질·속도·정확한 inversion을 보장하지 않는다.
- model과 scheduler, alpha table, prediction type, index convention은 하나의 versioned artifact로 배포해야 한다.

## 다음 학습 예고

다음 소스는 `02-14.CFG(Classifier-Free guidance).md`다. unconditional·conditional prediction을 한 모델에서 얻고 guidance scale로 결합하는 구현, batch doubling, prediction type과 scheduler의 경계, negative prompt와 saturation 실패를 다룬다.

$$
\hat{\epsilon}_{\mathrm{cfg}}
=
\hat{\epsilon}_{\mathrm{uncond}}
+
w
\left(
\hat{\epsilon}_{\mathrm{cond}}
-
\hat{\epsilon}_{\mathrm{uncond}}
\right)
$$
