<!-- curriculum: cycle=1; level=foundation; source_index=9/18; source=02-08.SwimTransformer.md; part=1/1 -->

# Swin Transformer: 창을 옮겨 계층적으로 보는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-07-28 |
| 회차·수준 | 1회차·기초 |
| 현재 소스 | 9/18 `02-08.SwimTransformer.md` |
| Part | 1/1 |
| 이전 소스 | 8/18 `02-07.ViT.md`: 전역 self-attention을 쓰는 Vision Transformer |
| 다음 소스 | 10/18 `02-09.ConvNeXt.md`: Transformer 시대의 convolution 설계 |

원본 파일명은 `SwimTransformer`지만 모델의 정확한 이름은 **Swin Transformer**다. Swin은 `Shifted Window`의 줄임말이다. 원본은 전역 attention의 비용, window attention, shifted window, cyclic shift와 mask, patch merging을 한 흐름으로 설명한다. 오늘은 이 범위를 Part 1/1로 완결한다.

이 글에서는 작은 숫자로 window를 직접 나누고 되돌린다. 그다음 shifted-window mask를 수식과 NumPy로 확인하고, 실행 가능한 PyTorch mini block에서 attention과 patch merging의 shape를 끝까지 추적한다. 마지막에는 C++와 C#에서 같은 window partition을 구현해 layout 계약을 비교한다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. 전역 self-attention과 window self-attention의 계산량 차이를 유도한다.
2. `BHWC` tensor를 window batch로 나누고 정확히 복원한다.
3. 고정 window만 반복할 때 생기는 정보 단절을 설명한다.
4. cyclic shift와 shifted-window attention을 구분한다.
5. 순환 이동으로 붙은 가짜 이웃을 attention mask로 차단한다.
6. relative position bias의 index와 shape를 설명한다.
7. patch merging의 `4C -> 2C` 변환과 stage별 shape를 추적한다.
8. 입력 크기가 window 크기의 배수가 아닐 때 padding과 crop 계약을 세운다.
9. NumPy golden test와 PyTorch forward·backward test를 실행한다.
10. C++·C#·Python의 layout과 dtype 차이를 점검한다.
11. 메모리, 수치 안정성, 배포 실패 사례를 진단한다.
12. 원본의 축약되거나 모호한 표현을 실제 구현 관점에서 바로잡는다.

## 선수 지식과 기호

행렬 곱, softmax, multi-head self-attention, convolution의 stride, tensor의 batch와 channel 개념이 필요하다. 직전 ViT 글의 patch token과 attention shape를 알고 있으면 좋다.

| 기호 | 뜻 |
| --- | --- |
| $B$ | batch 크기 |
| $H,W$ | token feature map의 높이와 너비 |
| $C$ | token embedding channel |
| $M$ | 정사각 window 한 변의 token 수 |
| $T=M^2$ | window 하나의 token 수 |
| $N=HW$ | 전체 token 수 |
| $n_w=HW/M^2$ | 이미지 한 장의 window 수 |
| $h$ | attention head 수 |
| $d=C/h$ | head 하나의 차원 |
| $s=\lfloor M/2\rfloor$ | shift 크기 |
| $L$ | Transformer block 수 |

PyTorch image tensor는 흔히 `NCHW`를 사용하지만, window partition 설명에서는 위치 축이 붙어 있는 `BHWC`가 편리하다. 이 글은 attention block 내부에서 `(B,H,W,C)`를 사용하고, convolution 입출력에서는 `(B,C,H,W)`를 사용한다. 변환 시에는 반드시 `permute`와 `contiguous` 여부를 확인한다.

## 1. 직관: 운동장을 교실로 나누되 다음 시간에는 칸을 옮긴다

ViT의 전역 attention은 모든 token이 모든 token을 본다. 관계를 넓게 볼 수 있지만 고해상도에서는 attention score 행렬이 매우 커진다.

Swin Transformer는 먼저 feature map을 $M\times M$ window로 나눈다. 같은 window 안에서만 attention을 계산하면 비용이 전체 token 수에 선형으로 증가한다. 그러나 같은 칸막이를 모든 block에서 반복하면 서로 다른 window의 token이 만나지 못한다.

해결책은 두 종류의 block을 번갈아 사용하는 것이다.

1. W-MSA block은 정렬된 window 안에서 attention한다.
2. 다음 SW-MSA block은 window 경계를 $s$만큼 옮겨 attention한다.

첫 시간에는 4명씩 고정된 교실에서 이야기하고, 다음 시간에는 칸막이를 반 칸 옮겨 이전 교실의 이웃과 새 조를 만드는 셈이다. 한 block만으로 전역 관계가 생기는 것은 아니다. block을 거치며 지역 정보가 점진적으로 전파된다.

```text
입력 feature        (B, H, W, C)
W-MSA window        (B*n_w, M*M, C)
window attention    (B*n_w, M*M, C)
window reverse      (B, H, W, C)
cyclic shift        (B, H, W, C)
SW-MSA window       (B*n_w, M*M, C)
masked attention    (B*n_w, M*M, C)
reverse shift       (B, H, W, C)
patch merging       (B, H/2, W/2, 2C)
```

## 2. 원본에서 구분하거나 바로잡을 점

원본의 큰 방향은 맞다. 다만 구현과 수학에서는 다음을 더 정확히 구분해야 한다.

| 원본의 표현 | 정확한 관점 |
| --- | --- |
| 고해상도에서 해상도가 4배가 되면 attention 비용이 256배다 | “해상도”가 가로·세로 길이를 뜻할 때 맞다. 각 변이 4배면 token 수는 16배이고, 전역 score 비용은 token 수의 제곱이므로 256배다. 전체 pixel 수가 4배라는 뜻이라면 score 비용은 약 16배다. |
| window attention이면 이미지가 아무리 커져도 비용이 정비례한다 | $M$과 $C$를 고정하면 spatial token 수 $HW$에 선형이다. stage가 깊어지며 $C$가 변하고 projection 비용 $4HWC^2$도 있으므로 전체 모델 비용은 별도로 계산해야 한다. |
| shifted window는 격자를 우측 하단으로 이동한다 | 논리적으로 window 경계를 옮기는 것이 핵심이다. 효율적 구현은 보통 feature를 위·왼쪽으로 `(-s,-s)` cyclic roll한 뒤 고정 window로 나누고, 끝에서 반대로 roll한다. |
| cyclic shift로 cross-window 연결이 생긴다 | 연결을 정의하는 것은 shifted window grouping이다. cyclic shift는 같은 크기의 dense batch로 효율적으로 계산하기 위한 구현 기법이다. |
| mask에 `-100`이라는 음의 무한대를 준다 | `-100`은 음의 무한대가 아니라 큰 음수 근사다. dtype과 kernel에 따라 `float("-inf")`, `torch.finfo(dtype).min`, 충분히 작은 유한값을 선택하고 NaN 가능성을 테스트해야 한다. |
| patch merging은 channel을 4배로 만든 뒤 $2C$로 압축한다 | 네 위치를 concatenate한 중간 tensor가 $4C$이고, LayerNorm과 linear reduction 뒤 출력은 보통 $2C$다. |
| Swin은 기존 detection head와 완벽히 결합된다 | 계층적 multi-scale output은 FPN류와 잘 맞지만 channel adapter, stride, normalization, padding, pretrained weight 계약을 맞춰야 한다. 자동으로 완벽히 호환되는 것은 아니다. |
| window attention의 핵심은 cyclic shift다 | 학습 가능한 relative position bias도 중요한 구성 요소다. 같은 window 안에서 token 사이의 상대적 2차원 offset을 attention logit에 더한다. |

원본 예제의 `torch.roll`과 역변환 equality test는 cyclic shift 자체를 설명하는 데는 맞다. 그러나 실제 SW-MSA가 되려면 window partition, attention mask, masked softmax, window reverse까지 함께 있어야 한다.

## 3. 전역 attention의 비용

### 3.1 projection 비용

입력 token 행렬을 다음처럼 두자.

$$
X\in\mathbb{R}^{N\times C}
$$

`Q`, `K`, `V` projection과 최종 output projection은 각각 대략 $NC^2$의 곱셈-덧셈을 요구한다. 네 projection을 합치면 다음과 같다.

$$
\Omega_{\mathrm{proj}}=4NC^2
$$

이미지 token 수는 $N=HW$이므로 다음처럼 쓸 수 있다.

$$
\Omega_{\mathrm{proj}}=4HWC^2
$$

### 3.2 score와 weighted sum 비용

head들을 합쳐 생각하면 $QK^\top$ 비용은 대략 $N^2C$다. attention probability와 $V$의 곱도 $N^2C$다.

$$
\Omega_{\mathrm{attn}}=2N^2C
$$

따라서 전역 MSA의 대표적 복잡도 식은 다음과 같다.

$$
\Omega(\mathrm{Global\ MSA})
=4HWC^2+2(HW)^2C
$$

이 식은 big-O보다 구체적인 연산 항을 비교하기 위한 근사다. bias, softmax, normalization, memory access 비용은 생략한다.

### 3.3 해상도 scaling을 정확히 읽기

가로와 세로를 각각 $a$배로 늘리고 patch 또는 token stride를 유지하면 다음과 같다.

$$
H'=aH,\qquad W'=aW
$$

token 수는 $a^2$배다.

$$
N'=a^2N
$$

전역 attention score 항은 $a^4$배가 된다.

$$
(N')^2=a^4N^2
$$

$a=4$라면 token 수는 16배, score 항은 256배다. 다만 projection 항은 token 수와 같이 16배다. 실제 wall-clock time이 정확히 256배라는 뜻은 아니다.

## 4. window attention의 비용

window 한 개의 token 수는 다음과 같다.

$$
T=M^2
$$

$H,W$가 $M$으로 나누어떨어진다면 이미지 한 장의 window 수는 다음과 같다.

$$
n_w=\frac{HW}{M^2}
$$

window 한 개의 score와 weighted sum 비용은 다음과 같다.

$$
2T^2C=2M^4C
$$

모든 window에 대해 합하면 다음과 같다.

$$
\frac{HW}{M^2}\cdot2M^4C
=2M^2HWC
$$

projection 항까지 포함한 대표 식은 다음과 같다.

$$
\Omega(\mathrm{W\text{-}MSA})
=4HWC^2+2M^2HWC
$$

$M$과 $C$를 고정하면 두 항 모두 $HW$에 선형이다.

### 4.1 작은 수치 비교

$H=W=56$, $M=7$, $C=96$이라 하자. 전체 token 수는 3136이고 window token 수는 49다. attention score matrix의 원소 수만 비교하면 전역 attention은 head마다 다음 크기다.

$$
3136^2=9{,}834{,}496
$$

window attention은 window 64개의 score 원소를 합쳐 다음 크기다.

$$
64\times49^2=153{,}664
$$

비율은 다음과 같다.

$$
\frac{9{,}834{,}496}{153{,}664}=64
$$

이 예에서는 score storage가 64분의 1이다. 바로 window 개수만큼 줄어든다.

## 5. window partition과 reverse

### 5.1 shape 변환

입력을 `BHWC` layout으로 두자.

$$
X\in\mathbb{R}^{B\times H\times W\times C}
$$

$H,W$가 $M$의 배수일 때 먼저 여섯 축으로 reshape한다.

$$
X_{\mathrm{grid}}
\in
\mathbb{R}^{B\times(H/M)\times M\times(W/M)\times M\times C}
$$

window 행과 window 열 축을 붙이고, 각 window 내부의 두 위치 축을 붙이려면 축 순서를 다음처럼 바꾼다.

```text
(B, H/M, M, W/M, M, C)
          ↓ permute
(B, H/M, W/M, M, M, C)
          ↓ reshape
(B*n_w, M*M, C)
```

`reshape`만 하고 `permute`를 생략하면 memory가 이어지는 순서가 달라져 잘못된 window를 만든다. shape가 맞는 것만으로는 충분하지 않다.

### 5.2 $4\times4$ 손계산

다음 scalar feature map을 $M=2$ window로 나눈다.

```text
 0  1 |  2  3
 4  5 |  6  7
------+------
 8  9 | 10 11
12 13 | 14 15
```

raster 순서 window는 다음과 같다.

```text
w0 = [0, 1, 4, 5]
w1 = [2, 3, 6, 7]
w2 = [8, 9, 12, 13]
w3 = [10, 11, 14, 15]
```

reverse는 이 네 window를 원래 좌표에 scatter하는 연산이다. 항상 다음 round trip을 단위 테스트해야 한다.

$$
\operatorname{reverse}
\left(
\operatorname{partition}(X)
\right)=X
$$

## 6. 고정 window만 쓸 때의 단절

W-MSA를 여러 번 적용해도 window 경계가 고정되어 있다면 한 window의 token은 다른 window의 token과 직접 섞이지 않는다. MLP는 각 token에 독립적으로 적용되므로 경계를 넘기지 못한다. residual connection도 같은 좌표끼리 더하므로 경계를 넘기지 못한다.

$4\times4$, $M=2$ 예제에서 좌상단 token 0은 첫 window의 `0,1,4,5`와는 attention할 수 있지만 token 2와는 다른 window이므로 볼 수 없다. 같은 partition을 반복하면 이 관계는 계속 막힌다.

shifted window는 다음 block에서 새로운 조를 만든다. 이전에 다른 window였던 token이 같은 shifted window에 포함될 수 있다. 따라서 W-MSA와 SW-MSA를 교대로 쌓으면 정보가 경계를 넘어 단계적으로 전파된다.

한 번의 SW-MSA가 곧 전역 attention이라는 뜻은 아니다. receptive field가 block 깊이에 따라 커진다고 이해해야 한다.

## 7. cyclic shift와 attention mask

### 7.1 논리적 이동과 구현 이동

window 경계를 직접 옮기면 가장자리에 크기가 다른 window가 생긴다. 서로 다른 크기의 window는 하나의 dense batch로 묶기 어렵다.

효율적 구현은 feature를 다음처럼 순환 이동한다.

$$
\widetilde{X}
=\operatorname{Roll}(X,-s,-s)
$$

그다음 $\widetilde{X}$를 기존의 정렬된 $M\times M$ window로 나눈다. attention을 마치면 window를 reverse하고 반대 방향으로 roll한다.

$$
Y
=\operatorname{Roll}
\left(
\widetilde{Y},s,s
\right)
$$

roll은 원소를 버리지 않으므로 두 roll만 놓고 보면 완전한 역연산이다.

$$
\operatorname{Roll}
\left(
\operatorname{Roll}(X,-s,-s),s,s
\right)=X
$$

### 7.2 왜 mask가 필요한가

cyclic shift는 이미지 반대쪽 끝의 token을 memory상 인접하게 붙인다. 일반 이미지는 torus가 아니므로 왼쪽 끝과 오른쪽 끝을 이웃으로 attention하게 해서는 안 된다.

각 원본 영역에 region id를 부여하고 window로 나눈 뒤, 같은 window 안 token 두 개의 id가 다르면 attention을 막는다.

window $w$의 region id vector를 다음처럼 두자.

$$
r^{(w)}\in\mathbb{Z}^{T}
$$

mask는 다음과 같이 정의할 수 있다.

$$
A^{(w)}_{ij}
=
\begin{cases}
0,&r^{(w)}_i=r^{(w)}_j\\
-\infty,&r^{(w)}_i\ne r^{(w)}_j
\end{cases}
$$

attention probability는 다음과 같다.

$$
P^{(w)}
=
\operatorname{softmax}
\left(
\frac{Q^{(w)}(K^{(w)})^\top}{\sqrt{d}}
+A^{(w)}
\right)
$$

차단된 위치의 logit은 softmax 후 0이 된다.

### 7.3 모든 행을 막지 않도록 주의

softmax의 한 행이 전부 $-\infty$면 `0/0` 형태가 되어 NaN이 생길 수 있다. Swin mask는 token 자기 자신과 같은 region의 token을 허용하므로 정상 구성에서는 모든 행에 최소 하나의 유효 위치가 있다.

custom mask를 만들 때는 다음 invariant를 검사한다.

$$
\forall i,\quad
\exists j\ \text{such that}\ A_{ij}=0
$$

## 8. relative position bias

window attention은 locality를 제한하지만, attention 자체는 window 안에서 token의 2차원 상대 위치를 자동으로 알지 못한다. Swin은 head마다 학습 가능한 relative position bias를 attention logit에 더한다.

$M\times M$ window에서 두 token의 상대 offset은 다음 범위다.

$$
\Delta y,\Delta x
\in
\{-(M-1),\ldots,M-1\}
$$

각 축에 가능한 값이 $2M-1$개이므로 bias table의 위치 수는 다음과 같다.

$$
(2M-1)^2
$$

head 수가 $h$라면 table shape는 흔히 다음과 같다.

$$
B_{\mathrm{rel}}
\in
\mathbb{R}^{(2M-1)^2\times h}
$$

2차원 offset을 1차원 index로 바꾸는 한 방법은 다음과 같다.

$$
k
=(\Delta y+M-1)(2M-1)
+(\Delta x+M-1)
$$

head $a$의 attention logit은 다음과 같다.

$$
Z^{(a)}_{ij}
=
\frac{
q^{(a)}_i\cdot k^{(a)}_j
}{\sqrt{d}}
+B_{\mathrm{rel}}[k(i,j),a]
+A_{ij}
$$

relative bias는 window마다 같은 table을 공유한다. 그래서 window의 절대 위치가 달라도 같은 상대 offset에는 같은 bias가 적용된다.

## 9. Swin block의 계산 순서

대표적인 pre-normalization block은 다음 두 residual branch로 이루어진다.

$$
\widehat{X}^{l}
=
X^{l-1}
+
\operatorname{W\text{-}MSA}
\left(
\operatorname{LN}(X^{l-1})
\right)
$$

$$
X^{l}
=
\widehat{X}^{l}
+
\operatorname{MLP}
\left(
\operatorname{LN}(\widehat{X}^{l})
\right)
$$

다음 block은 W-MSA 대신 SW-MSA를 쓴다.

$$
\widehat{X}^{l+1}
=
X^{l}
+
\operatorname{SW\text{-}MSA}
\left(
\operatorname{LN}(X^{l})
\right)
$$

$$
X^{l+1}
=
\widehat{X}^{l+1}
+
\operatorname{MLP}
\left(
\operatorname{LN}(\widehat{X}^{l+1})
\right)
$$

attention branch와 MLP branch 모두 입출력 shape는 `(B,H,W,C)`로 유지된다. patch merging에서만 spatial resolution과 channel이 바뀐다.

## 10. patch merging 수학과 shape

### 10.1 네 격자 추출

입력이 다음 shape라고 하자.

$$
X\in\mathbb{R}^{B\times H\times W\times C}
$$

짝수·홀수 좌표로 네 tensor를 뽑는다.

```text
x00 = X[:, 0::2, 0::2, :]  # 좌상
x10 = X[:, 1::2, 0::2, :]  # 좌하
x01 = X[:, 0::2, 1::2, :]  # 우상
x11 = X[:, 1::2, 1::2, :]  # 우하
```

각 tensor shape는 `(B,H/2,W/2,C)`다. channel 축으로 concatenate하면 다음과 같다.

$$
X_{\mathrm{cat}}
\in
\mathbb{R}^{B\times H/2\times W/2\times4C}
$$

LayerNorm 뒤 shared linear reduction을 적용한다.

$$
W_{\mathrm{red}}
\in
\mathbb{R}^{4C\times2C}
$$

출력은 다음 shape다.

$$
Y
\in
\mathbb{R}^{B\times H/2\times W/2\times2C}
$$

공간 token 수는 4분의 1이 되고 channel은 2배가 된다. 전체 activation scalar 수는 절반이 된다.

$$
\frac{H}{2}\frac{W}{2}(2C)
=\frac{HWC}{2}
$$

### 10.2 홀수 spatial shape

$H$나 $W$가 홀수면 네 slice의 shape가 달라져 concatenate할 수 없다. 구현은 다음 중 하나를 명시해야 한다.

1. 오른쪽과 아래를 padding한 뒤 merge한다.
2. 마지막 행이나 열을 crop한다.
3. shape guard로 입력을 거부한다.

조용한 crop은 pixel 정보를 버리므로 권장하지 않는다. 이 글의 mini 구현은 명시적으로 padding한다.

## 11. stage 전체 shape 추적

`224 x 224` RGB 입력, patch size 4, 첫 channel $C=96$인 전형적 shape를 추적한다.

| 단계 | spatial stride | `BHWC` shape | 설명 |
| --- | ---: | --- | --- |
| patch embedding | 4 | `(B,56,56,96)` | `4 x 4` non-overlap projection |
| stage 1 | 4 | `(B,56,56,96)` | W-MSA와 SW-MSA |
| patch merging 1 | 8 | `(B,28,28,192)` | `4C -> 2C` |
| stage 2 | 8 | `(B,28,28,192)` | local attention |
| patch merging 2 | 16 | `(B,14,14,384)` | spatial 절반 |
| stage 3 | 16 | `(B,14,14,384)` | 더 넓은 유효 수용 영역 |
| patch merging 3 | 32 | `(B,7,7,768)` | 마지막 pyramid level |
| stage 4 | 32 | `(B,7,7,768)` | global average 후 분류 가능 |

각 stage output은 stride 4, 8, 16, 32의 feature pyramid를 제공한다. detection이나 segmentation neck에 연결할 때는 해당 neck이 요구하는 `NCHW`로 바꾸고 channel adapter를 맞춘다.

## 12. NumPy 수작업 검증

다음 코드는 설명용이면서 **실행 가능**하다. window partition과 reverse round trip, cyclic shift round trip, mask의 차단 확률을 검증한다.

```python
import numpy as np


def window_partition(x: np.ndarray, m: int) -> np.ndarray:
    """BHWC -> (B*num_windows, m*m, C)."""
    b, h, w, c = x.shape
    if h % m != 0 or w % m != 0:
        raise ValueError("H and W must be divisible by window size")
    x = x.reshape(b, h // m, m, w // m, m, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, m * m, c)


def window_reverse(
    windows: np.ndarray, m: int, b: int, h: int, w: int
) -> np.ndarray:
    """(B*num_windows, m*m, C) -> BHWC."""
    c = windows.shape[-1]
    expected = b * (h // m) * (w // m)
    if windows.shape[0] != expected:
        raise ValueError("window count does not match output shape")
    x = windows.reshape(b, h // m, w // m, m, m, c)
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h, w, c)


def masked_softmax(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    if not np.all(allowed.any(axis=-1)):
        raise ValueError("every query needs at least one valid key")
    masked = np.where(allowed, logits, -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    exp = np.where(allowed, exp, 0.0)
    return exp / exp.sum(axis=-1, keepdims=True)


x = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
windows = window_partition(x, m=2)
expected = np.array(
    [
        [[0], [1], [4], [5]],
        [[2], [3], [6], [7]],
        [[8], [9], [12], [13]],
        [[10], [11], [14], [15]],
    ],
    dtype=np.float32,
)
np.testing.assert_array_equal(windows, expected)
np.testing.assert_array_equal(window_reverse(windows, 2, 1, 4, 4), x)

shifted = np.roll(x, shift=(-1, -1), axis=(1, 2))
restored = np.roll(shifted, shift=(1, 1), axis=(1, 2))
np.testing.assert_array_equal(restored, x)

region_ids = np.array([0, 0, 1, 1])
allowed = region_ids[:, None] == region_ids[None, :]
prob = masked_softmax(np.zeros((4, 4), dtype=np.float64), allowed)
np.testing.assert_allclose(prob[0], [0.5, 0.5, 0.0, 0.0])
np.testing.assert_allclose(prob.sum(axis=-1), np.ones(4))

print(windows[..., 0])
print(prob)
print("NumPy window and mask checks passed")
```

예상 핵심 출력은 다음과 같다.

```text
[[ 0.  1.  4.  5.]
 [ 2.  3.  6.  7.]
 [ 8.  9. 12. 13.]
 [10. 11. 14. 15.]]
[[0.5 0.5 0.  0. ]
 [0.5 0.5 0.  0. ]
 [0.  0.  0.5 0.5]
 [0.  0.  0.5 0.5]]
NumPy window and mask checks passed
```

## 13. 실행 가능한 PyTorch mini Swin

다음 코드는 교육용으로 단순화했지만 **실행 가능**하다.

- 입력은 `BHWC`다.
- relative position bias와 dropout은 생략한다.
- 실제 Swin과 같은 방식으로 shift, region mask, window attention, reverse shift를 수행한다.
- `nn.MultiheadAttention` 대신 mask broadcast를 분명히 보이기 위해 `QKV`를 직접 계산한다.
- patch merging은 홀수 spatial shape를 오른쪽·아래에 zero padding한다.

```python
import math

import torch
from torch import nn
from torch.nn import functional as F


def partition(x: torch.Tensor, m: int) -> torch.Tensor:
    b, h, w, c = x.shape
    if h % m or w % m:
        raise ValueError("padded H and W must be divisible by window size")
    x = x.reshape(b, h // m, m, w // m, m, c)
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(-1, m * m, c)
    )


def reverse(
    windows: torch.Tensor, m: int, b: int, h: int, w: int
) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.reshape(b, h // m, w // m, m, m, c)
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(b, h, w, c)
    )


def shifted_region_mask(
    hp: int, wp: int, m: int, shift: int, device: torch.device
) -> torch.Tensor:
    """Return (num_windows, m*m, m*m) boolean allowed mask."""
    if shift == 0:
        nw = (hp // m) * (wp // m)
        return torch.ones(nw, m * m, m * m, dtype=torch.bool, device=device)

    ids = torch.zeros((1, hp, wp, 1), dtype=torch.int64, device=device)
    h_slices = (slice(0, -m), slice(-m, -shift), slice(-shift, None))
    w_slices = (slice(0, -m), slice(-m, -shift), slice(-shift, None))
    region = 0
    for hs in h_slices:
        for ws in w_slices:
            ids[:, hs, ws, :] = region
            region += 1
    ids = partition(ids, m).squeeze(-1)
    return ids[:, :, None] == ids[:, None, :]


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, allowed: torch.Tensor, batch: int
    ) -> torch.Tensor:
        bnw, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            bnw, tokens, 3, self.heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        nw = allowed.shape[0]
        logits = logits.reshape(batch, nw, self.heads, tokens, tokens)
        mask = allowed[None, :, None, :, :]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        prob = logits.softmax(dim=-1)
        prob = prob.reshape(bnw, self.heads, tokens, tokens)

        out = (prob @ v).transpose(1, 2).reshape(bnw, tokens, dim)
        return self.proj(out)


class MiniSwinBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, window: int, shift: int
    ) -> None:
        super().__init__()
        if not 0 <= shift < window:
            raise ValueError("shift must satisfy 0 <= shift < window")
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window

        shortcut = x
        x = self.norm1(x)
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w

        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        windows = partition(x, self.window)
        allowed = shifted_region_mask(
            hp, wp, self.window, self.shift, x.device
        )
        windows = self.attn(windows, allowed, batch=b)
        x = reverse(windows, self.window, b, hp, wp)

        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x[:, :h, :w, :]
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


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


torch.manual_seed(7)
x = torch.randn(2, 7, 9, 8, requires_grad=True)
w_block = MiniSwinBlock(dim=8, heads=2, window=4, shift=0)
sw_block = MiniSwinBlock(dim=8, heads=2, window=4, shift=2)
merge = PatchMerging(dim=8)

y = w_block(x)
z = sw_block(y)
merged = merge(z)

assert y.shape == (2, 7, 9, 8)
assert z.shape == (2, 7, 9, 8)
assert merged.shape == (2, 4, 5, 16)
assert torch.isfinite(merged).all()

loss = merged.square().mean()
loss.backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()

print("W-MSA:", tuple(y.shape))
print("SW-MSA:", tuple(z.shape))
print("merged:", tuple(merged.shape))
print("loss:", f"{loss.item():.6f}")
print("PyTorch forward/backward checks passed")
```

이 구현은 학습용 최소 예제다. production 구현에는 relative position bias, attention dropout, stochastic depth, pretrained weight 호환, fused kernel, mask cache가 추가될 수 있다.

## 14. C++ 예제

다음 코드는 표준 라이브러리만 사용하는 **실행 가능한 C++17 예제**다. `BHWC`의 batch와 channel이 각각 1인 $4\times4$ 입력을 $2\times2$ window로 나누고 되돌린다.

```cpp
#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<float> Partition(
    const std::vector<float>& x, int h, int w, int m) {
    if (h % m != 0 || w % m != 0) {
        throw std::invalid_argument("H and W must be divisible by M");
    }
    std::vector<float> windows;
    windows.reserve(x.size());
    for (int wr = 0; wr < h / m; ++wr) {
        for (int wc = 0; wc < w / m; ++wc) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    const int y = wr * m + i;
                    const int col = wc * m + j;
                    windows.push_back(x.at(static_cast<std::size_t>(y * w + col)));
                }
            }
        }
    }
    return windows;
}

std::vector<float> Reverse(
    const std::vector<float>& windows, int h, int w, int m) {
    if (windows.size() != static_cast<std::size_t>(h * w)) {
        throw std::invalid_argument("window element count mismatch");
    }
    std::vector<float> x(static_cast<std::size_t>(h * w));
    std::size_t index = 0;
    for (int wr = 0; wr < h / m; ++wr) {
        for (int wc = 0; wc < w / m; ++wc) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    const int y = wr * m + i;
                    const int col = wc * m + j;
                    x.at(static_cast<std::size_t>(y * w + col)) =
                        windows.at(index++);
                }
            }
        }
    }
    return x;
}

int main() {
    std::vector<float> x(16);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i);
    }
    const auto windows = Partition(x, 4, 4, 2);
    const std::vector<float> expected{
        0, 1, 4, 5, 2, 3, 6, 7,
        8, 9, 12, 13, 10, 11, 14, 15
    };
    assert(windows == expected);
    assert(Reverse(windows, 4, 4, 2) == x);
    for (float value : windows) {
        std::cout << value << ' ';
    }
    std::cout << "\nC++ window round trip passed\n";
}
```

C++ tensor runtime를 사용할 때는 raw pointer의 실제 stride를 확인해야 한다. 위 코드는 contiguous scalar `H x W`만 다룬다. channel이 여러 개면 `BHWC` offset은 다음과 같다.

$$
\operatorname{offset}(b,y,x,c)
=((bH+y)W+x)C+c
$$

## 15. C# 예제

다음 코드는 .NET에서 실행 가능한 **C# 예제**다. C++과 같은 순서의 window를 만들고 round trip을 검사한다.

```csharp
using System;

public static class SwinWindowDemo
{
    static float[] Partition(float[] x, int h, int w, int m)
    {
        if (h % m != 0 || w % m != 0)
            throw new ArgumentException("H and W must be divisible by M");

        var windows = new float[x.Length];
        int index = 0;
        for (int wr = 0; wr < h / m; wr++)
        for (int wc = 0; wc < w / m; wc++)
        for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
        {
            int y = wr * m + i;
            int col = wc * m + j;
            windows[index++] = x[y * w + col];
        }
        return windows;
    }

    static float[] Reverse(float[] windows, int h, int w, int m)
    {
        if (windows.Length != h * w)
            throw new ArgumentException("window element count mismatch");

        var x = new float[h * w];
        int index = 0;
        for (int wr = 0; wr < h / m; wr++)
        for (int wc = 0; wc < w / m; wc++)
        for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
        {
            int y = wr * m + i;
            int col = wc * m + j;
            x[y * w + col] = windows[index++];
        }
        return x;
    }

    static void AssertEqual(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new Exception("length mismatch");
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                throw new Exception($"mismatch at {i}");
    }

    public static void Main()
    {
        var x = new float[16];
        for (int i = 0; i < x.Length; i++)
            x[i] = i;

        var expected = new float[]
        {
            0, 1, 4, 5, 2, 3, 6, 7,
            8, 9, 12, 13, 10, 11, 14, 15
        };
        var windows = Partition(x, 4, 4, 2);
        AssertEqual(windows, expected);
        AssertEqual(Reverse(windows, 4, 4, 2), x);
        Console.WriteLine(string.Join(" ", windows));
        Console.WriteLine("C# window round trip passed");
    }
}
```

`System.Drawing.Bitmap`이나 camera SDK의 입력은 대개 interleaved `HWC`, BGR, row padding을 포함할 수 있다. 이를 곧바로 위의 logical tensor와 같다고 가정하지 말고 전처리 단계에서 channel order와 row stride를 검증해야 한다.

## 16. 프레임워크 간 shape·layout·dtype 대응

| 환경 | image 또는 feature 기본 관례 | window attention 권장 내부 shape | dtype 주의점 |
| --- | --- | --- | --- |
| PyTorch convolution | `NCHW` | `BHWC` 또는 `(B*n_w,T,C)` | AMP에서 mask 값과 softmax dtype 확인 |
| NumPy | 명시한 stride에 따름 | 예제는 `BHWC` | 기본 `float64`가 PyTorch `float32`와 다를 수 있음 |
| C++ raw buffer | 계약 없이는 알 수 없음 | contiguous offset을 직접 정의 | `std::size_t`와 overflow 점검 |
| C# array | row-major 1차원 배열 | offset을 직접 정의 | `float`는 FP32, `double`은 FP64 |
| ONNX | graph의 axis 계약에 따름 | exporter가 만든 transpose 포함 | provider별 FP16 mask 동작 확인 |

`NCHW -> BHWC` 변환은 다음과 같다.

```python
x_bhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
x_nchw_again = x_bhwc.permute(0, 3, 1, 2).contiguous()
```

`permute`는 대개 view만 바꾸므로 non-contiguous tensor가 된다. 이후 `view`를 쓰면 오류가 나거나, 잘못된 low-level binding이 stride를 무시할 수 있다. `reshape` 또는 명시적 `contiguous`를 사용하되 불필요한 copy 비용도 측정한다.

## 17. 테스트와 디버깅

### 17.1 필수 invariant

window 관련 코드는 숫자 정답이 있는 작은 tensor로 먼저 검증한다.

1. `reverse(partition(x)) == x`여야 한다.
2. shift 후 reverse shift 결과가 원본과 같아야 한다.
3. mask의 shape는 `(n_w,T,T)`여야 한다.
4. mask의 각 query 행에는 최소 하나의 허용 key가 있어야 한다.
5. masked probability의 차단 위치는 0이어야 한다.
6. attention probability의 마지막 축 합은 1이어야 한다.
7. block 전후 shape는 같아야 한다.
8. patch merging은 `(B,H,W,C)`를 `(B,ceil(H/2),ceil(W/2),2C)`로 바꿔야 한다.
9. forward output과 backward gradient는 모두 finite여야 한다.

### 17.2 자주 발생하는 shape 오류

| 증상 | 가능한 원인 | 확인 방법 |
| --- | --- | --- |
| window 내부 값이 섞임 | `reshape` 전 `permute` 누락 | `arange` golden tensor 출력 |
| shifted output 위치가 어긋남 | reverse roll 부호 오류 | attention을 identity로 두고 round trip |
| batch 2부터 mask가 틀림 | window mask의 batch broadcast 오류 | `B=1`과 `B=2`를 모두 테스트 |
| 홀수 입력에서 concatenate 실패 | patch merging padding 누락 | `H=7,W=9` 테스트 |
| FP16에서 NaN | 모든 key masking 또는 부적절한 sentinel | 유효 key 수와 logits finite 검사 |
| export 후 정확도 급락 | layout, resize, padding, mask 차이 | intermediate tensor를 stage별 비교 |

### 17.3 attention mask를 눈으로 확인하기

$M=2$면 $T=4$라서 mask 하나는 $4\times4$다. 0과 차단값을 heatmap이나 작은 matrix로 출력하면 region이 잘못 붙은 문제를 쉽게 찾을 수 있다. 큰 실제 입력에서 시작하지 말고 `H=W=4`, `M=2`, `s=1`로 시작한다.

## 18. 성능과 메모리

### 18.1 score memory

window attention score tensor의 대표 shape는 다음과 같다.

$$
(B,n_w,h,T,T)
$$

원소 수는 다음과 같다.

$$
B\cdot n_w\cdot h\cdot M^4
$$

$n_w=HW/M^2$를 대입하면 다음과 같다.

$$
B\cdot h\cdot HW\cdot M^2
$$

window 크기를 7에서 14로 두 배 늘리면 같은 $H,W$에서 score 원소 수는 4배가 된다.

### 18.2 mask cache

shifted mask는 입력의 padded $H,W$, window 크기, shift 크기, device에 의해 결정된다. 매 forward마다 Python loop로 다시 만들면 낭비다. production에서는 shape별 cache를 고려한다.

cache key에는 최소한 다음을 포함한다.

```text
(padded_height, padded_width, window_size, shift_size, device)
```

dtype별 큰 음수 tensor를 직접 cache한다면 dtype도 key에 포함한다. boolean allowed mask를 cache하고 forward에서 dtype에 맞춰 적용하는 방법도 있다.

### 18.3 kernel과 data movement

이론 FLOPs가 작아도 `permute`, `contiguous`, padding, roll이 memory bandwidth를 소비한다. 작은 window를 지나치게 많이 만들면 kernel launch와 reshape overhead가 커질 수 있다.

측정할 항목은 다음과 같다.

1. end-to-end latency와 stage별 latency
2. peak allocated memory
3. batch size별 throughput
4. padding 비율
5. export runtime의 transpose와 copy 개수
6. attention kernel이 실제로 fused되는지 여부

## 19. 수치 안정성과 재현성

### 19.1 softmax

안정적인 softmax는 행의 최댓값을 먼저 뺀다.

$$
\operatorname{softmax}(z)_i
=
\frac{
\exp(z_i-\max_j z_j)
}{
\sum_k\exp(z_k-\max_j z_j)
}
$$

mask를 적용한 뒤에도 유효 logit의 최댓값을 기준으로 계산해야 한다. PyTorch의 `softmax`는 안정화되어 있지만 custom kernel이나 다른 언어 구현에서는 직접 확인한다.

### 19.2 mixed precision

FP16의 최소 유한값과 FP32의 최소 유한값은 다르다. FP32에서 만든 `-1e30` mask를 FP16으로 cast하면 `-inf`가 될 수 있다. 이것 자체가 항상 오류는 아니지만, 모든 원소가 차단된 행에서는 NaN 위험이 있다.

권장 검사는 다음과 같다.

1. mask 적용 전후 logits에 NaN이 없는지 확인한다.
2. softmax 결과가 finite인지 확인한다.
3. 차단 위치 확률이 정확히 0 또는 허용 오차 이내인지 확인한다.
4. FP32 reference와 FP16 output 차이를 정한다.

### 19.3 재현성

seed만 고정해도 device, attention kernel, dropout, stochastic depth, data order에 따라 결과가 달라질 수 있다. 실험 기록에는 다음을 남긴다.

```text
model config
window and shift sizes
input and padding policy
library and accelerator versions
dtype and autocast policy
random seeds
deterministic setting
checkpoint hash
```

## 20. 실무 실패 사례

### 사례 1: window 배수가 아닌 입력을 그냥 reshape

서빙 입력이 `230 x 230`인데 학습은 `224 x 224`만 사용했다. window partition의 `reshape`가 실패하거나 마지막 token이 유실된다.

해결은 resize 규칙 또는 오른쪽·아래 padding을 명시하고, attention 후 원래 크기로 crop하는 것이다. padding ratio도 모니터링한다.

### 사례 2: cyclic roll 뒤 mask를 생략

shape와 latency는 정상인데 이미지 왼쪽과 오른쪽 끝이 가짜 이웃이 된다. 분류 정확도만 보면 놓칠 수 있고 segmentation 경계에서 artifact가 나타난다.

해결은 작은 region-id mask를 출력하고, 차단 위치 probability가 0인지 단위 테스트하는 것이다.

### 사례 3: W-MSA와 SW-MSA가 모두 같은 shift

config parser에서 `shift_size`가 모든 block에 0으로 들어갔다. 모델은 학습되지만 window 사이 정보 교환이 약해진다.

해결은 block index별 shift sequence를 로그로 남기고, 예상 `0,s,0,s,...`와 비교한다.

### 사례 4: NHWC와 NCHW를 조용히 혼동

C# 전처리가 `NHWC` buffer를 만들었지만 ONNX input은 `NCHW`였다. shape가 우연히 맞는 작은 test에서는 지나가고 실제 이미지에서 정확도가 붕괴한다.

해결은 각 축에 서로 다른 크기를 쓰는 asymmetric input과 channel별 sentinel 값으로 layout test를 만든다.

### 사례 5: patch merging concat 순서 불일치

Python은 `[x00,x10,x01,x11]`, 배포 구현은 `[x00,x01,x10,x11]` 순서를 사용했다. shape는 같지만 pretrained linear weight가 기대하는 channel 의미가 바뀐다.

해결은 `arange` tensor로 concat 순서를 고정하고, 첫 patch merging intermediate tensor를 runtime 사이에서 비교한다.

### 사례 6: window size만 키우면 항상 좋아진다고 가정

window를 키우면 더 넓은 문맥을 보지만 score memory는 $M^2$에 비례해 증가한다. 작은 dataset에서는 과적합이나 latency 증가가 이득보다 클 수 있다.

해결은 정확도뿐 아니라 throughput, peak memory, 작은 객체 성능을 함께 ablation한다.

## 21. 배포 관점

### 21.1 export 전 계약

다음 항목을 model artifact와 함께 고정한다.

| 계약 | 기록할 값 |
| --- | --- |
| 입력 | 이름, `NCHW` 또는 `NHWC`, dynamic axis |
| 전처리 | resize, crop, RGB/BGR, normalization |
| padding | 방향, 값, 원래 크기 복원 규칙 |
| window | stage별 $M$과 $s$ |
| patch merging | concat 순서와 odd-shape 정책 |
| dtype | FP32, FP16, BF16, quantized dtype |
| 출력 | stage별 stride, channel, layout |

### 21.2 ONNX와 runtime 검증

`roll`, dynamic slice, reshape, mask broadcast는 exporter와 execution provider에 따라 graph가 다르게 최적화될 수 있다. export 성공만으로 충분하지 않다.

검증 순서는 다음과 같다.

1. 고정된 golden input을 저장한다.
2. PyTorch eager의 stage별 output을 저장한다.
3. ONNX runtime의 같은 stage output을 비교한다.
4. odd shape와 window 배수 shape를 모두 검사한다.
5. FP32에서 먼저 맞춘 뒤 FP16 tolerance를 정한다.
6. 실제 target device에서 latency와 memory를 측정한다.

### 21.3 운영 모니터링

운영에서는 다음을 관찰한다.

1. 입력 높이·너비 분포
2. padding pixel 비율
3. batch별 latency와 p95·p99
4. out-of-memory와 fallback 횟수
5. output confidence와 class drift
6. detection·segmentation이면 크기별 성능 proxy
7. model version과 preprocessing version 불일치

Swin의 window 비용은 고정 $M$에서 spatial size에 선형이지만, 허용 해상도를 무제한으로 열어도 된다는 뜻은 아니다. input limit과 resource guard를 둔다.

## 22. 체크리스트

### 수학과 shape

- [ ] 전역 attention과 window attention의 $HW$ 의존성을 구분했다.
- [ ] `BHWC -> window batch -> BHWC` shape를 기록했다.
- [ ] window partition의 축 순서를 golden tensor로 검증했다.
- [ ] shifted mask의 shape와 허용 region을 확인했다.
- [ ] relative position bias table 크기를 계산했다.
- [ ] patch merging의 `4C -> 2C`를 확인했다.

### 구현과 테스트

- [ ] `reverse(partition(x)) == x`를 테스트했다.
- [ ] roll과 reverse roll을 테스트했다.
- [ ] odd spatial shape의 padding과 crop을 테스트했다.
- [ ] batch 1과 batch 2에서 mask broadcast를 테스트했다.
- [ ] forward output과 backward gradient가 finite다.
- [ ] C++·C#·Python의 window 순서가 같다.

### 성능과 배포

- [ ] window 크기별 latency와 memory를 측정했다.
- [ ] mask cache key가 shape·device를 구분한다.
- [ ] layout과 channel order를 artifact에 기록했다.
- [ ] patch merging concat 순서를 고정했다.
- [ ] ONNX 또는 target runtime과 stage별 golden output을 비교했다.
- [ ] 입력 크기와 padding ratio를 운영에서 모니터링한다.

## 23. 연습문제

### 문제 1

$H=W=28$, $M=7$일 때 이미지 한 장의 window 수와 window당 token 수를 구하라.

### 문제 2

$H=W=56$, $C=96$, $M=7$일 때 전역 attention과 window attention의 score·weighted-sum 항의 비율을 구하라.

### 문제 3

$M=7$일 때 relative position bias table의 위치 개수는 몇 개인가?

### 문제 4

입력이 `(B,15,17,96)`일 때 zero padding 후 patch merging 출력 shape를 구하라.

### 문제 5

왜 cyclic shift만 적용하고 attention mask를 생략하면 안 되는가?

### 문제 6

W-MSA block만 여러 번 반복했을 때 MLP와 residual connection이 window 경계를 넘는 정보를 만들 수 있는가?

### 문제 7

window 크기를 7에서 14로 늘릴 때 $H,W,C$가 같다면 attention score 원소 수는 몇 배가 되는가?

### 문제 8

Python과 C#의 patch merging concat 순서가 다르지만 output shape는 같았다. 어떤 테스트가 이 오류를 가장 빨리 찾는가?

## 24. 해답

### 해답 1

window 수는 다음과 같다.

$$
n_w
=\frac{28\times28}{7^2}
=16
$$

window당 token 수는 다음과 같다.

$$
T=7^2=49
$$

### 해답 2

두 방식의 projection 항은 같고, score·weighted-sum 항만 비교하면 다음과 같다.

$$
\frac{
2(HW)^2C
}{
2M^2HWC
}
=\frac{HW}{M^2}
$$

수치를 대입하면 다음과 같다.

$$
\frac{56\times56}{7^2}
=64
$$

전역 attention의 해당 항이 window attention보다 64배 크다.

### 해답 3

가능한 상대 offset 수는 다음과 같다.

$$
(2M-1)^2
=(14-1)^2
=169
$$

head가 $h$개라면 bias table shape는 `(169,h)`다.

### 해답 4

높이 15는 16으로, 너비 17은 18로 padding된다. 절반 spatial과 두 배 channel이므로 출력은 `(B,8,9,192)`다.

### 해답 5

cyclic shift는 이미지 밖으로 나간 token을 반대쪽 끝에 붙인다. mask가 없으면 원본에서 멀리 떨어진 양 끝 token이 가짜 이웃으로 attention한다.

### 해답 6

아니다. token-wise MLP는 각 위치를 독립적으로 바꾸고 residual은 같은 위치를 더한다. partition이 고정되어 있으면 attention도 window 안에서만 섞이므로 경계를 넘지 못한다.

### 해답 7

score 원소 수는 고정 $H,W$에서 $M^2$에 비례한다. $M$이 2배이므로 4배다.

### 해답 8

서로 다른 값을 가진 `arange` 또는 좌표-coded tensor를 입력하고 patch merging의 concatenate 직후 `4C` intermediate tensor를 두 runtime에서 비교한다. random end-to-end accuracy보다 빠르고 정확하다.

## 25. 핵심 요약

1. 전역 attention의 score 항은 $(HW)^2$에 비례하지만, 고정 window attention은 $M^2HW$에 비례한다.
2. window partition은 `reshape -> permute -> reshape`의 축 순서가 핵심이며 shape만 맞아서는 충분하지 않다.
3. 고정 W-MSA만 반복하면 window 경계를 넘는 정보 교환이 없다.
4. SW-MSA는 다음 block에서 window grouping을 옮겨 cross-window 연결을 만든다.
5. cyclic shift는 효율적 dense batching을 위한 구현이고, 가짜 wrap-around 이웃은 mask로 차단해야 한다.
6. relative position bias는 window 안 token의 2차원 상대 offset을 attention logit에 넣는다.
7. patch merging은 네 위치를 $4C$로 붙인 뒤 $2C$로 projection해 spatial pyramid를 만든다.
8. odd input shape의 padding, layout, concat 순서, mask dtype은 배포 전에 고정해야 할 계약이다.
9. 작은 `arange` golden tensor와 round-trip test가 window 구현 오류를 가장 빨리 찾는다.
10. 이론 FLOPs뿐 아니라 roll·transpose·copy·padding·kernel overhead를 target device에서 측정해야 한다.

## 다음 학습 예고

다음은 1회차 기초 10/18 `02-09.ConvNeXt.md`다. Swin Transformer가 계층 구조와 stage 설계로 CNN의 장점을 Transformer에 가져왔다면, ConvNeXt는 반대로 현대 Transformer의 설계 원칙을 순수 convolution network에 적용한다. depthwise convolution, inverted bottleneck, LayerNorm, stage ratio를 shape와 연산량 관점에서 비교한다.
