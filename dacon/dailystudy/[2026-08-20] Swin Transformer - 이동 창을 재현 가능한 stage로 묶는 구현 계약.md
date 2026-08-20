<!-- curriculum: cycle=2; level=implementation; source_index=9/18; source=02-08.SwimTransformer.md; part=1/1 -->

# Swin Transformer: 이동 창을 재현 가능한 stage로 묶는 구현 계약

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-08-20 |
| 회차·수준 | 2회차·구현 |
| 현재 소스 | 9/18 `02-08.SwimTransformer.md` |
| Part | 1/1 |
| 이전 소스 | 8/18 `02-07.ViT.md`: 가변 해상도·마스크 구현 |
| 다음 소스 | 10/18 `02-09.ConvNeXt.md`: ConvNeXt block·학습 구현 |

원본은 전역 attention의 비용, window attention, shifted window, cyclic shift, patch merging을 소개한다. 1회차 문서는 이 구조의 직관과 수식을 작은 tensor로 확인했다. 이번 2회차는 설명을 반복하지 않고 window 순서, relative position bias index, padding과 shifted mask의 결합, stochastic depth, stage 학습·평가, checkpoint와 배포 계약을 실행 가능한 코드와 테스트로 고정한다.

원본 파일명은 `SwimTransformer`이지만 모델의 정확한 이름은 **Swin Transformer**다. 커리큘럼 추적을 위해 메타데이터의 `source`에는 실제 파일명을 그대로 기록한다.

## 학습 목표

이 글을 마치면 다음을 구현하고 검증할 수 있다.

1. `BHWC` feature를 raster 순서의 window batch로 나누고 되돌린다.
2. shifted-window region mask와 padding key mask를 결합한다.
3. 2차원 상대 위치를 bias table의 1차원 index로 변환한다.
4. W-MSA와 SW-MSA를 번갈아 쓰는 block을 구현한다.
5. sample별 stochastic depth와 `train()`·`eval()` 차이를 테스트한다.
6. 홀수 spatial shape를 보존하는 patch merging을 구현한다.
7. 작은 Swin classifier를 완전한 학습·평가 loop로 실행한다.
8. shift 유무의 연결성 ablation과 동일 seed 재현성을 검사한다.
9. NumPy, PyTorch, C++와 C#의 layout·dtype 계약을 맞춘다.
10. export와 운영에서 dynamic shape, mask cache, 입력 상한을 관리한다.

## 선수 지식과 기호

행렬 곱, softmax, multi-head attention, residual connection, LayerNorm, cross entropy를 알고 있다고 가정한다.

| 기호 | 정의 |
| --- | --- |
| $B$ | batch 크기 |
| $H,W$ | feature 높이와 너비 |
| $C$ | channel 차원 |
| $M$ | 정사각 window 한 변 |
| $T=M^2$ | window 하나의 token 수 |
| $s$ | shift 크기, 보통 $\lfloor M/2\rfloor$ |
| $h$ | attention head 수 |
| $d=C/h$ | head 하나의 차원 |
| $n_w$ | 이미지 한 장의 padded window 수 |
| $H_p,W_p$ | window 배수로 padding한 높이와 너비 |
| $p_l$ | block $l$의 drop-path 확률 |
| $K$ | class 수 |

입력 이미지는 `NCHW`, convolution 출력도 `NCHW`, Swin block 내부 feature는 `BHWC`, window token은 `(B*n_w,T,C)`로 표기한다.

## 1. 이번 회차의 구현 경계

window attention은 수식보다 tensor 계약에서 자주 깨진다.

- `reshape` 전에 필요한 `permute`를 빠뜨리면 shape만 맞고 window 값은 틀린다.
- cyclic roll만 적용하면 이미지 반대편 경계가 가짜 이웃이 된다.
- odd shape를 pad한 뒤 padding token을 key로 숨기지 않으면 값 0도 정보로 섞인다.
- relative bias index의 행·열 방향이 바뀌면 checkpoint를 읽어도 다른 모델이 된다.
- patch merging의 네 위치 concatenate 순서가 runtime마다 다르면 weight가 호환되지 않는다.
- mask를 매 forward마다 Python loop로 만들면 dynamic shape가 적어도 latency가 커진다.
- stochastic depth를 token별로 적용하면 residual branch 정규화와 달라진다.

이번 구현은 정사각 window와 2차원 image feature를 사용한다. 입력 해상도는 가변이지만, stage의 channel 수와 head 수는 checkpoint 계약으로 고정한다.

## 2. 원본에서 구현 관점으로 바로잡을 점

| 원본의 표현 또는 축약 | 이번 구현의 정확한 계약 |
| --- | --- |
| 모델명을 `Swim Transformer`로 표기 | 정확한 이름은 `Swin Transformer`이며 `Swin`은 shifted window를 가리킨다. |
| 해상도가 4배면 global attention이 256배 | 가로와 세로가 각각 4배라는 뜻이면 token 수가 16배, quadratic attention 항이 256배다. pixel 수가 4배라는 뜻이면 16배다. |
| shifted window는 feature를 우측 하단으로 이동 | 논리적 목표는 window 경계를 옮기는 것이다. 효율적 구현은 feature를 보통 `(-s,-s)`로 roll한 뒤 고정 partition한다. |
| cyclic shift가 cross-window 연결을 만든다 | 연결을 만드는 것은 shifted grouping이다. cyclic shift는 균일한 dense batch를 유지하는 구현 기법이다. |
| mask의 `-100`은 음의 무한대 | `-100`은 유한하다. dtype에 맞는 큰 음수 또는 boolean mask를 쓰고 all-masked row를 피한다. |
| patch merging 뒤 channel이 4배 | concatenate 직후만 $4C$이고 LayerNorm과 linear reduction 뒤 보통 $2C$다. |
| 기존 detection head와 완벽히 결합 | feature stride·channel·normalization·checkpoint naming을 adapter와 함께 맞춰야 한다. |
| `torch.roll` round trip이면 검증 완료 | partition, region mask, padding mask, attention, reverse, crop까지 검증해야 한다. |

## 3. window 직렬화 계약

### 3.1 partition

$H_p$와 $W_p$가 $M$의 배수라 하자. window grid 좌표를 $(a,b)$, 내부 좌표를 $(i,j)$라 하면 원래 feature 좌표는 다음과 같다.

$$
y=aM+i,\qquad x=bM+j
$$

이미지 하나에서 window index와 token index를 raster 순서로 정의한다.

$$
w_{\mathrm{idx}}=a\frac{W_p}{M}+b
$$

$$
t_{\mathrm{idx}}=iM+j
$$

shape 흐름은 다음과 같다.

```text
(B,H_p,W_p,C)
-> (B,H_p/M,M,W_p/M,M,C)
-> (B,H_p/M,W_p/M,M,M,C)
-> (B*n_w,M*M,C)
```

두 번째 화살표의 `permute(0,1,3,2,4,5)`가 window row와 column을 붙인다.

### 3.2 reverse

reverse는 partition의 정확한 역함수여야 한다.

$$
\operatorname{reverse}(\operatorname{partition}(X))=X
$$

값이 모두 같은 tensor는 잘못된 순서도 통과하므로 `arange` golden tensor를 쓴다. batch와 channel도 1보다 큰 case를 포함한다.

## 4. shifted mask를 만드는 순서

### 4.1 padding

원래 shape가 $(H,W)$라면 window 배수까지 오른쪽과 아래를 pad한다.

$$
H_p=\left\lceil\frac{H}{M}\right\rceil M
$$

$$
W_p=\left\lceil\frac{W}{M}\right\rceil M
$$

$$
p_H=H_p-H,\qquad p_W=W_p-W
$$

feature와 함께 `valid` boolean map도 pad한다. 원래 위치는 참, padding 위치는 거짓이다.

### 4.2 cyclic shift와 region id

shifted block은 padded feature와 valid map을 같은 방향으로 roll한다.

$$
\widetilde{X}_{y,x}=X_{(y+s)\bmod H_p,(x+s)\bmod W_p}
$$

cyclic wrap으로 붙은 서로 다른 논리 영역을 구분하려고 $3\times3$ slice에 region id를 부여한다. 같은 shifted window 안에서도 id가 다르면 attention을 차단한다.

window $u$의 region id vector를 $r_u\in\mathbb{Z}^{T}$라 하면 허용 행렬은 다음과 같다.

$$
A^{\mathrm{region}}_{u,i,j}
=
\mathbb{1}[r_{u,i}=r_{u,j}]
$$

### 4.3 padding key mask와 결합

shift 후 window로 나눈 valid vector를 $v_{b,u,j}$라 하자.

$$
A_{b,u,i,j}
=
A^{\mathrm{region}}_{u,i,j}
\land
v_{b,u,j}
$$

padding query의 출력은 마지막 crop에서 제거된다. production fused kernel에서는 query mask까지 명시해 불필요한 계산을 줄일 수 있다.

## 5. relative position bias index

window 내부 token $i$와 $j$의 좌표를 각각 $(y_i,x_i)$, $(y_j,x_j)$라 하자.

$$
\Delta y=y_i-y_j,\qquad \Delta x=x_i-x_j
$$

offset 범위는 $[-(M-1),M-1]$다. 음수를 없애고 row-major index로 바꾼다.

$$
q(i,j)
=
(\Delta y+M-1)(2M-1)+(\Delta x+M-1)
$$

bias table의 shape는 다음과 같다.

$$
\left((2M-1)^2,h\right)
$$

table에서 index를 모으면 head-first bias shape는 $(h,T,T)$다. query와 key를 뒤집어도 shape가 같으므로 center·corner index를 golden test로 고정한다.

## 6. attention과 residual 수식

window token $X_w\in\mathbb{R}^{T\times C}$를 head별로 projection한다.

$$
Q=X_wW_Q,\qquad K=X_wW_K,\qquad V=X_wW_V
$$

head $a$의 logit은 다음과 같다.

$$
L^{(a)}_{ij}
=
\frac{Q^{(a)}_i(K^{(a)}_j)^\top}{\sqrt{d}}
+R^{(a)}_{ij}
$$

허용되지 않은 위치를 mask하고 마지막 축으로 softmax한다.

$$
P^{(a)}_{ij}
=
\operatorname{softmax}_j
\left(
L^{(a)}_{ij}
+\operatorname{mask}_{ij}
\right)
$$

pre-norm residual은 다음과 같다.

$$
\widehat{X}
=
X+\operatorname{DropPath}
\left(
\operatorname{WMSA}(\operatorname{LN}(X))
\right)
$$

$$
Y
=
\widehat{X}
+\operatorname{DropPath}
\left(
\operatorname{MLP}(\operatorname{LN}(\widehat{X}))
\right)
$$

W-MSA block은 $s=0$, 다음 SW-MSA block은 보통 $s=\lfloor M/2\rfloor$를 사용한다.

## 7. stochastic depth 계약

drop path는 residual branch 전체를 sample별로 끈다. 4차원 feature의 mask shape는 $(B,1,1,1)$이다. keep probability를 $q=1-p_l$라 하면 다음과 같다.

$$
\widetilde{F}(X)
=
\frac{Z}{q}F(X),
\qquad
Z\sim\operatorname{Bernoulli}(q)
$$

$$
\mathbb{E}[\widetilde{F}(X)]=F(X)
$$

평가 모드에서는 identity다. token마다 다른 mask를 만들면 sample별 stochastic depth와 다른 연산이다.

## 8. patch merging 계약

입력이 홀수 크기면 오른쪽과 아래를 0으로 pad하고 네 격자를 다음 순서로 뽑는다.

$$
X_{00}=X[:,0::2,0::2,:]
$$

$$
X_{10}=X[:,1::2,0::2,:]
$$

$$
X_{01}=X[:,0::2,1::2,:]
$$

$$
X_{11}=X[:,1::2,1::2,:]
$$

concatenate 순서는 checkpoint 계약이다.

$$
X_{\mathrm{cat}}
=
\operatorname{Concat}
\left[
X_{00},X_{10},X_{01},X_{11}
\right]
\in
\mathbb{R}^{B\times\lceil H/2\rceil\times\lceil W/2\rceil\times4C}
$$

LayerNorm과 linear reduction 뒤 출력은 다음 shape다.

$$
X_{\mathrm{out}}
\in
\mathbb{R}^{B\times\lceil H/2\rceil\times\lceil W/2\rceil\times2C}
$$

## 9. stage shape 추적

입력 image가 `(B,3,31,35)`이고 patch embedding이 `kernel=2, stride=2`라 하자.

| 단계 | 내부 shape | 비고 |
| --- | --- | --- |
| image pad | `(B,3,32,36)` | patch 배수 |
| patch embedding | `(B,16,18,16)` | `BHWC` |
| W-MSA용 pad | `(B,16,20,16)` | $M=4$ |
| window tokens | `(B*20,16,16)` | 이미지당 20 windows |
| stage 1 crop | `(B,16,18,16)` | attention 뒤 crop |
| patch merging | `(B,8,9,32)` | $4C\to2C$ |
| stage 2 pad | `(B,8,12,32)` | $M=4$ |
| global average | `(B,32)` | spatial 평균 |
| classifier | `(B,K)` | class logits |

block의 temporary padding은 즉시 crop한다. patch merging의 odd-shape padding은 stage 출력 크기의 일부다.

## 10. NumPy 수작업 검증

다음 코드는 설명용이면서 **실행 가능**하다. window round trip과 relative bias index를 외부 framework 없이 검사한다.

```python
import numpy as np


def partition(x: np.ndarray, m: int) -> np.ndarray:
    b, h, w, c = x.shape
    if h % m or w % m:
        raise ValueError("H and W must be divisible by M")
    return (
        x.reshape(b, h // m, m, w // m, m, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, m * m, c)
    )


def reverse(
    windows: np.ndarray, m: int, b: int, h: int, w: int
) -> np.ndarray:
    c = windows.shape[-1]
    expected = b * (h // m) * (w // m)
    if windows.shape != (expected, m * m, c):
        raise ValueError("window shape mismatch")
    return (
        windows.reshape(b, h // m, w // m, m, m, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(b, h, w, c)
    )


def relative_index(m: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    coords = np.stack([yy.reshape(-1), xx.reshape(-1)])
    delta = coords[:, :, None] - coords[:, None, :]
    delta[0] += m - 1
    delta[1] += m - 1
    delta[0] *= 2 * m - 1
    return delta.sum(axis=0)


x = np.arange(2 * 4 * 4 * 2).reshape(2, 4, 4, 2)
windows = partition(x, 2)
np.testing.assert_array_equal(reverse(windows, 2, 2, 4, 4), x)
np.testing.assert_array_equal(windows[0, :, 0], [0, 2, 8, 10])

index = relative_index(2)
np.testing.assert_array_equal(
    index,
    np.array(
        [
            [4, 3, 1, 0],
            [5, 4, 2, 1],
            [7, 6, 4, 3],
            [8, 7, 5, 4],
        ]
    ),
)
assert index.min() == 0 and index.max() == 8

print("windows:", windows.shape)
print("relative-index range:", int(index.min()), int(index.max()))
```

예상 출력은 window shape `(8,4,2)`와 relative index 범위 `0 8`이다.

## 11. 실행 가능한 PyTorch 구현

다음 코드는 교육용으로 작게 만든 **실행 가능한 완전 예제**다. relative bias, region·padding mask, drop path, patch merging, train/eval loop와 재현성 test를 포함한다.

```python
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def window_partition(x: torch.Tensor, m: int) -> torch.Tensor:
    b, h, w, c = x.shape
    if h % m or w % m:
        raise ValueError("padded H and W must be divisible by window")
    return (
        x.reshape(b, h // m, m, w // m, m, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, m * m, c)
    )


def window_reverse(
    windows: torch.Tensor, m: int, b: int, h: int, w: int
) -> torch.Tensor:
    c = windows.shape[-1]
    expected = b * (h // m) * (w // m)
    if windows.shape != (expected, m * m, c):
        raise ValueError("window count or token count mismatch")
    return (
        windows.reshape(b, h // m, w // m, m, m, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, h, w, c)
    )


def relative_position_index(m: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
    )
    coords = coords.flatten(1)
    delta = coords[:, :, None] - coords[:, None, :]
    delta = delta.permute(1, 2, 0).contiguous()
    delta[:, :, 0] += m - 1
    delta[:, :, 1] += m - 1
    delta[:, :, 0] *= 2 * m - 1
    return delta.sum(-1)


def region_allowed(hp: int, wp: int, m: int, shift: int) -> torch.Tensor:
    if shift == 0:
        nw = (hp // m) * (wp // m)
        return torch.ones(nw, m * m, m * m, dtype=torch.bool)

    labels = torch.zeros((1, hp, wp, 1), dtype=torch.int64)
    h_slices = (slice(0, -m), slice(-m, -shift), slice(-shift, None))
    w_slices = (slice(0, -m), slice(-m, -shift), slice(-shift, None))
    region = 0
    for hs in h_slices:
        for ws in w_slices:
            labels[:, hs, ws, :] = region
            region += 1
    ids = window_partition(labels, m).squeeze(-1)
    return ids[:, :, None] == ids[:, None, :]


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype)
        mask.bernoulli_(keep)
        return x * mask / keep


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        table_size = (2 * window - 1) ** 2
        self.relative_bias = nn.Parameter(torch.zeros(table_size, heads))
        self.register_buffer(
            "relative_index",
            relative_position_index(window),
            persistent=False,
        )
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(
        self, windows: torch.Tensor, allowed: torch.Tensor
    ) -> torch.Tensor:
        bnw, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            bnw, tokens, 3, self.heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_bias[self.relative_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        logits = logits + bias.unsqueeze(0)
        logits = logits.masked_fill(
            ~allowed[:, None, :, :], torch.finfo(logits.dtype).min
        )
        attention = logits.softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(
            bnw, tokens, channels
        )
        return self.proj(output)


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window: int,
        shift: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        if not 0 <= shift < window:
            raise ValueError("shift must satisfy 0 <= shift < window")
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads, window)
        self.path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.path2 = DropPath(drop_path)

    def _attention_branch(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        m = self.window
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
        hp, wp = h + ph, w + pw

        valid = torch.ones((b, h, w, 1), dtype=torch.bool, device=x.device)
        valid = F.pad(valid, (0, 0, 0, pw, 0, ph), value=False)
        if self.shift:
            x = torch.roll(
                x, shifts=(-self.shift, -self.shift), dims=(1, 2)
            )
            valid = torch.roll(
                valid, shifts=(-self.shift, -self.shift), dims=(1, 2)
            )

        windows = window_partition(x, m)
        valid_keys = window_partition(valid, m).squeeze(-1)
        region = region_allowed(hp, wp, m, self.shift).to(x.device)
        region = region.repeat(b, 1, 1)
        allowed = region & valid_keys[:, None, :]
        windows = self.attention(windows, allowed)
        x = window_reverse(windows, m, b, hp, wp)

        if self.shift:
            x = torch.roll(
                x, shifts=(self.shift, self.shift), dims=(1, 2)
            )
        return x[:, :h, :w, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.path1(self._attention_branch(self.norm1(x)))
        x = x + self.path2(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x00 = x[:, 0::2, 0::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x11 = x[:, 1::2, 1::2, :]
        x = torch.cat([x00, x10, x01, x11], dim=-1)
        return self.reduction(self.norm(x))


class TinySwin(nn.Module):
    def __init__(self, classes: int = 2, use_shift: bool = True) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, 16, kernel_size=2, stride=2)
        shift = 2 if use_shift else 0
        self.stage1 = nn.Sequential(
            SwinBlock(16, 4, 4, 0, 0.0),
            SwinBlock(16, 4, 4, shift, 0.05),
        )
        self.merge = PatchMerging(16)
        self.stage2 = nn.Sequential(
            SwinBlock(32, 4, 4, 0, 0.10),
            SwinBlock(32, 4, 4, shift, 0.15),
        )
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        ph = image.shape[-2] % 2
        pw = image.shape[-1] % 2
        image = F.pad(image, (0, pw, 0, ph))
        x = self.patch(image).permute(0, 2, 3, 1)
        x = self.stage1(x)
        x = self.merge(x)
        x = self.stage2(x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.head(x)


def toy_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    image = 0.05 * torch.randn(32, 3, 16, 16, generator=generator)
    target = torch.arange(32) % 2
    image[target == 0, :, 2:7, 2:7] += 1.0
    image[target == 1, :, 9:14, 9:14] += 1.0
    return image, target


def train_once(seed: int, use_shift: bool) -> tuple[
    list[float], dict[str, torch.Tensor], float
]:
    seed_all(seed)
    image, target = toy_data()
    model = TinySwin(use_shift=use_shift)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-4
    )
    history = []
    model.train()
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(image), target)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        prediction = model(image).argmax(dim=1)
        accuracy = float((prediction == target).float().mean())
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    return history, state, accuracy


seed_all(123)
x = torch.arange(2 * 8 * 8 * 3).reshape(2, 8, 8, 3)
windows = window_partition(x, 4)
assert torch.equal(window_reverse(windows, 4, 2, 8, 8), x)

odd = torch.randn(2, 7, 9, 16, dtype=torch.float64, requires_grad=True)
block = SwinBlock(16, 4, 4, 2, 0.0).double()
y = block(odd)
assert y.shape == odd.shape and torch.isfinite(y).all()
y.square().mean().backward()
assert odd.grad is not None and torch.isfinite(odd.grad).all()

merged = PatchMerging(16).double()(odd.detach())
assert merged.shape == (2, 4, 5, 32)

history1, state1, accuracy1 = train_once(20260820, True)
history2, state2, accuracy2 = train_once(20260820, True)
np.testing.assert_array_equal(history1, history2)
for name in state1:
    assert torch.equal(state1[name], state2[name]), name
assert accuracy1 == accuracy2
_, _, fixed_accuracy = train_once(20260820, False)

model = TinySwin().eval()
with torch.no_grad():
    dynamic_logits = model(torch.randn(2, 3, 15, 17))
assert dynamic_logits.shape == (2, 2)
assert torch.isfinite(dynamic_logits).all()

drop = DropPath(0.5).eval()
probe = torch.randn(4, 3, 5, 7)
assert torch.equal(drop(probe), probe)

print("window:", tuple(windows.shape))
print("odd block:", tuple(y.shape), "merge:", tuple(merged.shape))
print(
    "loss:",
    f"{history1[0]:.6f}",
    "->",
    f"{history1[-1]:.6f}",
    "shift/fixed accuracy:",
    f"{accuracy1:.3f}",
    f"{fixed_accuracy:.3f}",
)
print("exact replay: PASS")
```

실행 결과 shifted·fixed accuracy는 모두 `1.000`이었다. 이 synthetic task는 밝은 사각형의 절대 위치 차이가 커서 고정 window만으로도 풀린다. 따라서 이 결과는 shift가 불필요하다는 증거가 아니라, cross-window 정보가 필요한 data로 ablation을 설계해야 한다는 경고다. loss는 `0.758849 -> 0.494606`이었고 동일 seed exact replay를 통과했다.

### 11.1 구현에서 단순화한 부분

- production Swin보다 stage depth와 embedding 차원이 작다.
- shifted mask를 매 forward에 만든다. 운영 구현은 shape·device별 cache가 필요하다.
- mixed precision, distributed training과 gradient accumulation은 제외했다.
- classification용 global average를 사용한다. detection은 stage feature adapter가 필요하다.

### 11.2 학습·평가 loop의 경계

`model.train()`은 stochastic depth를 켜고 `model.eval()`은 끈다. 평가 시 `torch.no_grad()`를 함께 쓴다. 재현성 검사는 accuracy만 비교하지 않고 loss history와 모든 `state_dict` tensor를 exact replay로 비교한다. device나 mixed precision이 바뀌면 허용 오차와 환경을 기록한다.

## 12. shift ablation

parameter 수와 학습 budget을 그대로 두고 두 번째 block의 shift만 0 또는 $M/2$로 바꾸면 grouping 효과를 분리할 수 있다.

| 설정 | 첫 block | 둘째 block | 구조 |
| --- | --- | --- | --- |
| 고정 window | W-MSA | W-MSA | 같은 window 안에서만 전달 |
| shifted window | W-MSA | SW-MSA | 이전 경계 양쪽의 연결 증가 |

reachable token 수가 늘었다고 정확도가 자동으로 높아지는 것은 아니다. 위 toy ablation도 두 설정 모두 accuracy 1.0이었다. 실제 ablation은 경계 양쪽의 관계가 label을 결정하는 data를 사용하고 seed 여러 개, augmentation, parameter 수와 학습 budget을 같게 유지해 평균과 분산을 보고한다.

## 13. C++17 예제

다음 코드는 표준 라이브러리만 사용하는 **실행 가능한 C++17 예제**다. $3\times5$ scalar feature를 padding한 뒤 `[X00,X10,X01,X11]` 순서로 patch merge한다.

```cpp
#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

std::vector<float> PatchMerge(const std::vector<float>& x, int h, int w) {
    const int hp = h + h % 2;
    const int wp = w + w % 2;
    const int oh = hp / 2;
    const int ow = wp / 2;
    std::vector<float> output(oh * ow * 4, 0.0F);

    auto Read = [&](int y, int col) {
        if (y >= h || col >= w) {
            return 0.0F;
        }
        return x.at(static_cast<std::size_t>(y * w + col));
    };

    for (int y = 0; y < oh; ++y) {
        for (int col = 0; col < ow; ++col) {
            const int base = (y * ow + col) * 4;
            output[base + 0] = Read(2 * y, 2 * col);
            output[base + 1] = Read(2 * y + 1, 2 * col);
            output[base + 2] = Read(2 * y, 2 * col + 1);
            output[base + 3] = Read(2 * y + 1, 2 * col + 1);
        }
    }
    return output;
}

int main() {
    std::vector<float> x(15);
    for (int i = 0; i < 15; ++i) {
        x[i] = static_cast<float>(i);
    }
    const auto y = PatchMerge(x, 3, 5);
    assert(y.size() == 2 * 3 * 4);
    assert((std::vector<float>{y[0], y[1], y[2], y[3]})
           == (std::vector<float>{0, 5, 1, 6}));
    assert((std::vector<float>{y[20], y[21], y[22], y[23]})
           == (std::vector<float>{14, 0, 0, 0}));

    std::cout << std::fixed << std::setprecision(1)
              << y[0] << " " << y[1] << " "
              << y[2] << " " << y[3] << "\n"
              << y[20] << " " << y[21] << " "
              << y[22] << " " << y[23] << "\n";
}
```

이 예제는 attention kernel이 아니라 patch merging layout의 golden oracle이다. production C++에서는 contiguous `NHWC` stride, aligned memory와 framework tensor view를 별도로 검증한다.

## 14. C# 예제

다음 **실행 가능한 C# 예제**는 C++과 같은 입력과 출력 순서를 검사한다.

```csharp
using System;

public static class SwinPatchMerge
{
    static float Read(float[] x, int h, int w, int y, int col)
    {
        if (y >= h || col >= w)
            return 0.0f;
        return x[y * w + col];
    }

    static float[] PatchMerge(float[] x, int h, int w)
    {
        int hp = h + h % 2;
        int wp = w + w % 2;
        int oh = hp / 2;
        int ow = wp / 2;
        var output = new float[oh * ow * 4];
        for (int y = 0; y < oh; ++y)
        {
            for (int col = 0; col < ow; ++col)
            {
                int offset = (y * ow + col) * 4;
                output[offset + 0] = Read(x, h, w, 2 * y, 2 * col);
                output[offset + 1] = Read(x, h, w, 2 * y + 1, 2 * col);
                output[offset + 2] = Read(x, h, w, 2 * y, 2 * col + 1);
                output[offset + 3] = Read(x, h, w, 2 * y + 1, 2 * col + 1);
            }
        }
        return output;
    }

    static void Main()
    {
        var x = new float[15];
        for (int i = 0; i < x.Length; ++i)
            x[i] = i;
        float[] y = PatchMerge(x, 3, 5);
        if (y.Length != 24)
            throw new Exception("shape mismatch");
        float[] expectedFirst = { 0, 5, 1, 6 };
        float[] expectedLast = { 14, 0, 0, 0 };
        for (int i = 0; i < 4; ++i)
        {
            if (y[i] != expectedFirst[i] || y[20 + i] != expectedLast[i])
                throw new Exception("layout mismatch");
        }
        Console.WriteLine(
            "{0:F1} {1:F1} {2:F1} {3:F1}",
            y[0], y[1], y[2], y[3]
        );
        Console.WriteLine(
            "{0:F1} {1:F1} {2:F1} {3:F1}",
            y[20], y[21], y[22], y[23]
        );
    }
}
```

.NET image API의 buffer는 BGRA, row stride와 bottom-up 저장을 사용할 수 있다. byte buffer를 논리 `NCHW` 또는 `NHWC` tensor로 바꾸는 전처리는 독립적으로 검사한다.

## 15. 프레임워크 간 shape·layout·dtype 대응

| 환경 | image 기본 layout | block 내부 권장 layout | dtype 주의 |
| --- | --- | --- | --- |
| PyTorch | `NCHW` | 예제는 `BHWC` | FP16·BF16 mask와 softmax 확인 |
| LibTorch C++ | `NCHW` | graph 계약을 따름 | `permute` 뒤 contiguous 여부 |
| ONNX Runtime | 보통 `NCHW` | transpose 포함 | dynamic `Roll`·`Pad` 지원 |
| C# tensor runtime | API별 상이 | stride 명시 | `Half`와 accumulator dtype |
| NumPy oracle | 자유 | `BHWC` | float64와 model float32 차이 |

weight 변환에서 다음 shape를 고정한다.

| parameter | PyTorch shape | 의미 |
| --- | --- | --- |
| patch convolution | `(C_out,C_in,P,P)` | patch projection |
| QKV weight | `(3C,C)` | output-major linear |
| relative bias table | `((2M-1)^2,h)` | offset-major |
| patch reduction | `(2C,4C)` | `[X00,X10,X01,X11]` 입력 |
| classifier | `(K,C_last)` | output-major linear |

## 16. 테스트와 디버깅

### 16.1 단위 테스트

최소 test matrix는 다음과 같다.

1. `B=2,C=3`인 `arange` tensor의 partition·reverse exact round trip.
2. relative index의 diagonal과 corner index 확인.
3. shifted mask가 wrap-around token pair를 차단하는지 확인.
4. padding key를 바꿔도 원래 영역 출력이 불변인지 확인.
5. `H=7,W=9,M=4` odd shape forward·backward.
6. patch merging 출력 shape와 concatenate golden.
7. drop path가 eval에서 identity인지 확인.
8. 같은 seed 두 실행의 loss history와 parameter 비교.
9. 다른 dynamic input shape의 finite output.

### 16.2 자주 발생하는 오류

| 증상 | 가능 원인 | 가장 빠른 진단 |
| --- | --- | --- |
| window 값이 줄무늬로 섞임 | `permute` 순서 누락 | `arange` 첫 두 window |
| 경계에서 정확도 급락 | region mask 누락 | 좌우 끝 token attention |
| odd input에서 NaN | all-masked query row | allowed row 수 검사 |
| checkpoint 뒤 성능 붕괴 | relative index 방향 | corner pair golden |
| stage 전환 수치 불일치 | merge concat 순서 | reduction 전 $4C$ 비교 |
| eval 결과가 매번 변함 | `eval()` 누락 | 같은 입력 두 번 비교 |
| batch 2부터 mask 오류 | mask repeat 순서 | 단독·batch 결과 비교 |
| export만 실패 | dynamic operator 미지원 | odd-shape 최소 graph |

### 16.3 gradient 검사

double precision의 작은 window에서 attention 입력과 relative bias table을 `torch.autograd.gradcheck`로 검사할 수 있다. hard boolean mask는 미분 대상이 아니다. finite-difference step과 tolerance를 기록한다.

## 17. 성능·메모리·수치 안정성

window attention score 원소 수는 다음과 같다.

$$
B n_w h T^2
=
B h H_pW_pM^2
$$

$H_p,W_p,h$가 같을 때 window 크기를 두 배 늘리면 score memory는 4배다. projection과 MLP는 channel 차원의 영향도 크므로 score 식만으로 latency를 예측하지 않는다.

padding 낭비율은 다음과 같다.

$$
\rho_{\mathrm{pad}}
=
1-\frac{HW}{H_pW_p}
$$

작은 feature에서 $M$이 크면 낭비가 커진다. dynamic batching은 비슷한 해상도를 bucket해 pad 비율을 낮춘다.

region mask cache key에는 다음 값이 필요하다.

```text
(H_p,W_p,M,s,device)
```

서로 다른 유효 크기를 한 batch에 pad했다면 sample별 valid mask를 잃으면 안 된다.

FP16에서 `-1e9`를 cast하면 음의 무한대가 될 수 있다. boolean mask와 `torch.finfo(dtype).min`을 사용해도 all-masked row 의미는 정의해야 한다. 유효 query가 자기 자신을 볼 수 있다는 invariant를 먼저 보장한다.

LayerNorm은 마지막 channel 축 $C$에 적용한다. `NCHW` tensor에 `LayerNorm(C)`를 바로 호출하면 마지막 축 $W$가 정규화 대상이 되는 조용한 오류가 날 수 있다.

`roll -> reshape -> permute -> reshape`는 FLOPs가 작아도 bandwidth와 allocation을 쓴다. profiler에서 matmul뿐 아니라 roll, copy, contiguous, padding과 mask 생성도 본다.

## 18. 재현성 기록

checkpoint 옆에 다음을 저장한다.

- source commit과 PyTorch·CUDA·driver 버전
- seed와 deterministic algorithm 설정
- stage별 depth, channel, head, window, shift sequence
- drop-path schedule
- patch embedding과 patch merging padding 정책
- optimizer, learning-rate schedule, weight decay
- augmentation, normalization과 class mapping
- mixed-precision scaler 상태

CPU와 GPU 또는 서로 다른 kernel 사이에서는 exact equality가 보장되지 않을 수 있다. 이때 reference dtype, 최대 절대·상대 오차, 평가 metric 허용치를 함께 기록한다.

## 19. 실무 실패 사례

### 사례 1: mask cache key에서 shift 누락

W-MSA용 all-allowed mask가 SW-MSA에서도 재사용됐다. shape는 같고 loss도 감소하지만 wrap-around 정보가 섞였다.

해결은 cache key에 `shift_size`를 넣고 경계 token pair golden test를 release gate에 두는 것이다.

### 사례 2: 마지막 stage에서 window가 feature보다 큼

입력 crop이 줄어 feature가 `3 x 4`인데 window는 7이었다. 대부분이 padding이 되어 latency와 attention 분포가 달라졌다.

해결은 stage별 effective window를 쓸지, 입력 최소 크기를 강제할지 정한다. window가 바뀌면 relative bias interpolation 정책도 필요하다.

### 사례 3: C++ patch merging 순서 불일치

Python은 `[00,10,01,11]`, C++은 `[00,01,10,11]` 순서였다. output shape는 같지만 reduction weight 의미가 바뀌었다.

해결은 reduction 전 $4C$ vector를 공통 golden fixture로 비교하는 것이다.

### 사례 4: evaluation에서 stochastic depth가 켜짐

서빙 wrapper가 `model.eval()`을 호출하지 않아 같은 image의 logit이 요청마다 달라졌다.

해결은 export 전에 eval mode를 강제하고 동일 입력 반복 test를 둔다.

### 사례 5: 입력 제한 없는 dynamic shape

window attention이 global attention보다 효율적이라는 이유로 최대 해상도를 두지 않았다. 큰 image에서 window 수와 activation이 계속 늘어 OOM이 났다.

해결은 pixel 수, 각 변 길이와 batch token 상한을 두고 거부·resize·queue 정책을 명시하는 것이다.

### 사례 6: pretrained relative bias 누락

state-dict key 변환기가 relative bias table을 버리고 무작위 초기화했다. backbone 대부분은 load되어 missing key 경고를 놓치기 쉬웠다.

해결은 missing·unexpected key allowlist를 좁게 유지하고 고정 image embedding을 reference와 비교한다.

## 20. 배포 관점

### 20.1 export 전 계약

| 항목 | 고정할 내용 |
| --- | --- |
| 입력 | `NCHW`, RGB/BGR, range, mean/std |
| shape | 허용 min/max와 padding 방식 |
| window | stage별 $M,s$와 dynamic 축 |
| mask | region·padding mask와 cache |
| relative bias | table shape와 index 방향 |
| merging | `[00,10,01,11]` 순서 |
| dtype | input, weight, accumulator, output |
| 출력 | logits 또는 stage feature 이름·stride |

### 20.2 ONNX·runtime parity

이 환경에서 ONNX package가 없다면 export 성공을 주장하지 않는다. package가 있는 CI에서 다음을 검증한다.

1. PyTorch eval model을 고정 seed로 실행한다.
2. window 배수 입력과 odd input을 각각 export한다.
3. `Pad`, `Roll`, reshape와 dynamic shape의 provider 지원을 확인한다.
4. PyTorch와 runtime logit의 최대 절대·상대 오차를 비교한다.
5. batch 1·2와 최소·최대 허용 해상도를 검사한다.
6. warm-up 뒤 latency와 peak memory를 기록한다.

### 20.3 detection backbone

각 stage feature의 stride와 channel을 명시적으로 반환한다. FPN adapter가 기대하는 `C2,C3,C4,C5` 순서, channel projection과 normalization을 맞춘다. classification head를 제거한 것만으로 detector와 호환되지는 않는다.

### 20.4 운영 모니터링

- 입력 높이·너비·aspect ratio와 pad 비율
- stage별 window 수와 batch token 수
- p50·p95·p99 latency와 OOM·timeout
- NaN·Inf logit과 confidence drift
- resize·reject 비율
- model·preprocessing·runtime version

## 21. 체크리스트

### 구현

- [ ] partition과 reverse가 `arange` tensor에서 exact round trip이다.
- [ ] W-MSA와 SW-MSA shift sequence가 `0,s,0,s`다.
- [ ] region mask와 padding key mask를 모두 적용했다.
- [ ] relative bias index 방향을 golden으로 고정했다.
- [ ] drop path mask가 sample별이고 eval에서 identity다.
- [ ] odd shape patch merging과 concat 순서를 검사했다.
- [ ] train/eval loop와 optimizer zero-grad 순서가 명확하다.

### 테스트

- [ ] batch 1·2, even·odd shape를 포함했다.
- [ ] forward·backward finite test를 통과했다.
- [ ] 같은 seed의 loss와 parameter replay를 검사했다.
- [ ] shift ablation의 통제 변수를 기록했다.
- [ ] C++·C#·Python의 merge golden이 같다.
- [ ] export runtime parity를 별도 환경에서 확인했다.

### 운영

- [ ] 입력 shape와 pixel 수 상한이 있다.
- [ ] mask cache key에 shape, window, shift, device가 있다.
- [ ] preprocessing과 class mapping을 versioning한다.
- [ ] missing checkpoint key를 allowlist로 검사한다.
- [ ] latency, pad 비율, OOM과 NaN을 모니터링한다.

## 22. 연습문제

### 문제 1

$H=15,W=17,M=4$일 때 $H_p,W_p$, 이미지당 window 수, padding 비율을 구하라.

### 문제 2

$M=7,h=6$일 때 relative bias table shape를 구하라.

### 문제 3

query 좌표가 $(0,0)$, key 좌표가 $(1,2)$이고 $M=4$일 때 relative bias index를 구하라.

### 문제 4

입력이 `(B,7,9,32)`일 때 patch merging concatenate 전후 shape를 구하라.

### 문제 5

왜 shifted feature와 valid map을 같은 방향으로 roll해야 하는가?

### 문제 6

drop-path 확률이 0.2일 때 살아남은 residual branch의 scale은 얼마인가?

### 문제 7

window 크기를 두 배로 하면 같은 padded spatial shape와 head 수에서 score memory는 몇 배인가?

### 문제 8

checkpoint와 output shape가 정상인데 C++ logit이 Python과 크게 다르다. 가장 먼저 비교할 두 intermediate는 무엇인가?

## 23. 해답

### 해답 1

$$
H_p=\left\lceil\frac{15}{4}\right\rceil4=16
$$

$$
W_p=\left\lceil\frac{17}{4}\right\rceil4=20
$$

window 수는 $(16/4)(20/4)=20$이다. padding 비율은 다음과 같다.

$$
1-\frac{15\cdot17}{16\cdot20}
=
0.203125
$$

즉 약 20.31%다.

### 해답 2

offset 위치 수는 $(2M-1)^2=13^2=169$이므로 table shape는 `(169,6)`이다.

### 해답 3

$\Delta y=-1$, $\Delta x=-2$다.

$$
q=(-1+3)\cdot7+(-2+3)=15
$$

### 해답 4

padding 뒤 spatial shape는 `(8,10)`이다. concatenate shape는 `(B,4,5,128)`이고 linear reduction 뒤 `(B,4,5,64)`다.

### 해답 5

feature만 roll하면 valid map이 다른 좌표를 가리킨다. 실제 token을 padding으로 막거나 padding token을 실제 key로 허용하게 된다.

### 해답 6

keep probability는 $q=0.8$이므로 scale은 $1/q=1.25$다.

### 해답 7

score 원소 수는 $M^2$에 비례하므로 4배다.

### 해답 8

partition 직후 첫 window token 순서와 patch merging reduction 직전의 $4C$ vector를 먼저 비교한다. 같다면 relative bias index와 masked attention logit을 비교한다.

## 24. 핵심 요약

1. 핵심은 `roll` 하나가 아니라 partition·mask·attention·reverse·crop의 전체 계약이다.
2. region mask는 cyclic wrap의 가짜 이웃을, padding key mask는 가짜 token을 막는다.
3. relative bias table은 $((2M-1)^2,h)$이며 query-key 방향이 checkpoint 호환성을 결정한다.
4. W-MSA와 SW-MSA 교대는 고정 window 경계를 넘어 연결을 늘린다.
5. drop path는 residual branch를 sample별로 끄며 eval에서는 identity다.
6. patch merging의 `[00,10,01,11]` 순서는 shape에 드러나지 않는 weight 계약이다.
7. odd shape, batch 2, dynamic input과 exact replay를 단위 테스트에 포함한다.
8. 성능은 attention FLOPs뿐 아니라 roll, padding, permute, copy와 mask 생성 비용으로 결정된다.
9. 배포에서는 input limit, runtime operator 지원, checkpoint key와 preprocessing을 versioning한다.

## 25. 다음 학습 예고

다음 소스는 2회차 구현 10/18 `02-09.ConvNeXt.md`다. depthwise convolution, inverted bottleneck, channels-last LayerNorm, layer scale, stochastic depth를 완전한 학습·평가 block으로 구현하고 layout 변환 비용과 ablation을 검증한다.
