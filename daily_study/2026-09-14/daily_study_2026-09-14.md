# [2026-09-14] 오늘 학습: 배치·텐서 연산 & Transformer 블록 조립

> **오늘의 핵심 문장:** Transformer 블록은 특별한 마법 한 덩어리가 아니라, `배치 × 토큰 × 특징` 텐서의 마지막 축을 선형 변환하고, 토큰 축끼리 attention으로 정보를 섞은 뒤, 잔차 연결과 정규화로 그 변화를 안정적으로 누적하는 계산 그래프다.

지난 학습에서는 한 시퀀스의 토큰 표현을

$$
\mathbf{H}\in\mathbb{R}^{T\times d_{\text{model}}}
$$

로 두고, 한 개 attention head의 Query·Key·Value 연산을 배웠다. 실제 학습에서는 시퀀스 하나가 아니라 $B$개를 한꺼번에 처리하고, attention도 $h$개 head로 나눈다. 따라서 출발점은 3차원 텐서

$$
\mathbf{X}\in\mathbb{R}^{B\times T\times d}
$$

이다. 여기서 $B$는 batch size, $T$는 sequence length, $d=d_{\text{model}}$은 각 토큰 벡터의 차원이다.

각 head의 차원을 $d_h=d/h$라고 하면 multi-head attention의 핵심 shape 흐름은 다음과 같다.

$$
\begin{aligned}
\mathbf{X}
&\in\mathbb{R}^{B\times T\times d},\\
\mathbf{Q},\mathbf{K},\mathbf{V}
&\in\mathbb{R}^{B\times h\times T\times d_h},\\
\mathbf{S}=\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}}+\mathbf{M}
&\in\mathbb{R}^{B\times h\times T\times T},\\
\mathbf{A}=\operatorname{Softmax}_{\text{key}}(\mathbf{S})
&\in\mathbb{R}^{B\times h\times T\times T},\\
\mathbf{O}_{\text{heads}}=\mathbf{A}\mathbf{V}
&\in\mathbb{R}^{B\times h\times T\times d_h},\\
\mathbf{O}=\operatorname{Concat}(\mathbf{O}_{\text{heads}})\mathbf{W}_O
&\in\mathbb{R}^{B\times T\times d}.
\end{aligned}
$$

여기서 4차원 텐서 $\mathbf{K}$의 $\mathbf{K}^{\top}$는 텐서 전체를 뒤집는다는 뜻이 아니라, 각 $(b,h)$ 조각에서 마지막 두 축만 바꿔

$$
(B,h,T,d_h)
\longrightarrow
(B,h,d_h,T)
$$

로 만든다는 약속이다.

본문의 batch 선형층과 attention은 토큰을 행벡터로 쌓는 관례를 쓴다. 반면 2차원 회전을 보여 주는 RoPE와 head 하나의 recurrent state를 설명하는 GDN 절에서는 식을 간결하게 하려고 $\mathbf{q},\mathbf{k},\mathbf{v}$를 열벡터로 쓴다. 각 절의 shape를 함께 적어 두 관례를 구분한다.

오늘은 이 shape들이 왜 그렇게 되는지 인덱스로 증명하고, 위치 표현·잔차 연결·LayerNorm·MLP까지 조립해 하나의 pre-norm Transformer 블록을 완성한다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

오늘의 블록은 한 번에 발명된 부품이 아니다. 2015년 ResNet은 깊은 신경망에 항등 shortcut을 두는 residual 학습을 널리 확산시켰고, 2016년 Layer Normalization은 batch 통계에 의존하기 어려운 sequence model을 위해 한 표본 내부 특징을 정규화하는 길을 제시했다. 2017년 **Attention Is All You Need**는 recurrent 계산을 주된 경로에서 제거하고 multi-head attention·위치 표현·residual·LayerNorm·FFN을 조립했다.

이 조립의 본질적 목표는 두 가지였다. 먼 토큰끼리 짧은 계산 경로로 직접 상호작용하게 하고, 학습 때 여러 token 위치를 큰 tensor 연산으로 병렬 처리하는 것이다. 오늘 배우는 batch·축·broadcasting은 이 역사적 아이디어를 실제 코드와 가속기에서 실행 가능하게 만든 계산 언어다.

### 1.1 왜 데이터를 한 개씩 계산하지 않고 batch로 묶을까?

예제 하나의 그래디언트만 보고 가중치를 바꾸면 그 예제의 우연한 잡음에 크게 흔들릴 수 있다. 반대로 전체 데이터의 평균 그래디언트를 매번 정확히 계산하면 한 번 갱신하는 데 너무 오래 걸린다. mini-batch는 그 중간 지점이다.

표본 $B$개의 손실을

$$
\mathcal{L}_{\text{batch}}(\boldsymbol{\theta})
=
\frac{1}{B}
\sum_{b=1}^{B}
\ell_b(\boldsymbol{\theta})
$$

로 평균 내면 그래디언트도

$$
\nabla_{\boldsymbol{\theta}}
\mathcal{L}_{\text{batch}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\nabla_{\boldsymbol{\theta}}
\ell_b
$$

가 된다. 즉 batch는 여러 표본의 학습 신호를 평균내는 통계적 장치이면서, GPU가 큰 행렬곱을 병렬 처리하게 하는 계산 단위다.

중요한 구분이 있다. **같은 batch에 들어갔다고 서로의 문장을 attention하는 것은 아니다.** batch 축은 독립 표본을 나란히 쌓은 축이다. 정상적인 self-attention은 각 $b$ 안에서만 토큰을 섞는다.

### 1.2 행렬만 알면 되는데 왜 tensor가 필요할까?

행렬은 행과 열, 두 축을 가진다. 하지만 실제 모델은 동시에 여러 종류의 축을 추적해야 한다.

- 어떤 표본인가: batch 축 $B$
- 문장 안의 몇 번째 토큰인가: sequence 축 $T$
- 토큰 벡터의 몇 번째 좌표인가: feature 축 $d$
- 여러 attention 관점 중 몇 번째인가: head 축 $h$

그래서 데이터는

$$
X_{btd}
$$

처럼 세 개 이상의 첨자를 갖는다. 머신러닝 프레임워크에서 **tensor**는 이런 다차원 수 배열을 뜻한다. 엄밀한 미분기하학의 텐서 정의와 동일한 모든 구조를 전제로 하는 말은 아니다.

tensor 표기의 가장 큰 장점은 숫자를 많이 담는 것이 아니라, **각 축의 의미를 보존하는 것**이다. shape를 잃으면 서로 다른 문장, 토큰, head가 뜻하지 않게 섞일 수 있다.

### 1.3 왜 attention을 여러 head로 나눌까?

한 개 head는 각 Query에 대해 하나의 확률분포를 만든다. 이 하나의 분포가 구문 관계, 멀리 떨어진 참조, 지역 문맥, 형식적 패턴을 모두 동시에 표현해야 한다면 병목이 될 수 있다.

multi-head attention은 전체 특징 차원 $d$를 보통 $h$개의 부분공간으로 나눈다.

$$
d=h d_h
$$

각 head는 별도의 투영을 통해 자기만의 Query·Key·Value 표현과 attention 분포를 만들 수 있다. 그런 다음 모든 head의 출력을 이어 붙이고 $\mathbf{W}_O$로 다시 섞는다.

다만 “head마다 반드시 사람이 해석할 수 있는 역할 하나를 담당한다”는 보장은 없다. head는 표현 능력을 나누는 계산 구조이지, 자동으로 문법 담당·의미 담당이라는 이름표가 붙는 모듈은 아니다.

### 1.4 토큰의 내용만 있으면 왜 순서를 알 수 없을까?

마스크가 없는 self-attention에 위치 정보가 전혀 없다고 하자. 토큰 순서를 같은 방식으로 섞는 순열행렬을 $\mathbf{P}$라 하면 입력은

$$
\mathbf{X}'_{b,:,:}
=
\mathbf{P}\mathbf{X}_{b,:,:}
\qquad
(b=1,\ldots,B)
$$

가 된다. Query·Key·Value도 똑같이 순열되고, attention 출력 역시

$$
\operatorname{Attn}(\mathbf{X}')_{b,:,:}
=
\mathbf{P}\operatorname{Attn}(\mathbf{X})_{b,:,:}
$$

로 함께 순열된다. 이를 **순열 등변성(permutation equivariance)**이라고 한다. 모델은 원소가 바뀐 것은 추적하지만, “첫째·둘째”라는 순서 자체를 내용만으로 얻지는 못한다.

그래서 위치 벡터를 토큰 임베딩에 더하거나, Query와 Key를 위치에 따라 회전시키는 RoPE 같은 위치 표현이 필요하다. 인과 마스크도 과거와 미래의 비대칭을 제공하지만, 거리와 상대 위치를 풍부하게 표현하는 별도 장치와 같은 것은 아니다.

### 1.5 왜 attention 출력만 차곡차곡 쌓지 않을까?

깊은 모델에서 매 층이 입력을 완전히 새 표현으로 덮어쓰면 원래 정보와 그래디언트가 긴 변환 사슬을 모두 통과해야 한다. 잔차 연결은 부분 함수 $F$가 전체 답을 새로 만드는 대신 **현재 표현에 더할 수정량**을 배우게 한다.

$$
\mathbf{Y}
=
\mathbf{X}+F(\mathbf{X})
$$

전체 텐서를 긴 벡터 하나로 펼쳐 Jacobian을 생각하면 역전파 경로의 국소 미분은

$$
\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}
=
\mathbf{I}
+
\frac{\partial F}{\partial \mathbf{X}}
$$

가 되어 항등 경로가 남는다. 이것이 모든 그래디언트 소실을 자동으로 없앤다는 뜻은 아니지만, 깊은 네트워크가 학습할 수 있는 훨씬 직접적인 경로를 제공한다.

### 1.6 왜 LayerNorm이 필요할까?

attention과 MLP의 출력 크기는 입력·가중치·학습 단계에 따라 달라진다. 잔차 덧셈을 여러 층 반복하면 특징 좌표의 규모가 불안정해질 수 있다.

LayerNorm은 각 토큰 벡터 하나 안에서 특징 축의 평균과 분산을 계산한다.

$$
\mu_{bt}
=
\frac{1}{d}
\sum_{c=1}^{d}X_{btc},
\qquad
\sigma_{bt}^{2}
=
\frac{1}{d}
\sum_{c=1}^{d}
(X_{btc}-\mu_{bt})^2
$$

$$
\operatorname{LN}(X_{btc})
=
\gamma_c
\frac{X_{btc}-\mu_{bt}}
{\sqrt{\sigma_{bt}^{2}+\varepsilon}}
+\beta_c
$$

batch의 다른 문장이나 다른 토큰의 통계에 의존하지 않으므로, 길이가 달라지거나 batch size가 바뀌어도 같은 정의를 유지한다.

### 1.7 왜 attention 뒤에 또 MLP가 필요할까?

attention의 핵심 역할은 **토큰 축 사이의 정보 이동**이다. 어떤 위치에서 정보를 가져올지 정해 가중합한다. 반면 position-wise MLP 또는 FFN은 각 토큰 위치에서 **특징 축을 비선형 변환**한다.

$$
\operatorname{FFN}(\mathbf{x})
=
\phi(\mathbf{x}\mathbf{W}_1+\mathbf{b}_1)
\mathbf{W}_2+\mathbf{b}_2
$$

attention이 “누구에게서 읽을까?”를 해결한다면, FFN은 “읽어 온 정보를 각 위치에서 어떻게 가공할까?”를 해결한다. 두 연산의 축별 역할이 다르므로 둘 다 필요하다.

### 1.8 오늘 두 트랙은 어디에서 만날까?

오늘의 수학과 AI 부품은 다음처럼 1:1로 연결된다.

$$
\boxed{
\begin{array}{c}
\text{첨자·shape 계약}
\rightarrow
\text{QKV와 head 축을 안전하게 배치}\\
\text{축 수축}
\rightarrow
\text{Query--Key 내적과 Value 가중합}\\
\text{broadcasting}
\rightarrow
\text{batch·head 전체에 마스크 적용}\\
\text{순열행렬}
\rightarrow
\text{위치 표현이 필요한 이유 증명}\\
\text{평균·분산}
\rightarrow
\text{토큰별 LayerNorm}\\
\text{연쇄법칙·야코비안}
\rightarrow
\text{잔차 경로의 그래디언트}
\end{array}
}
$$

---

## 2. 친절한 용어 사전

### 2.1 텐서·shape 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 축 | axis | 배열에서 의미가 달라지는 방향이다. $B\times T\times d$에는 표본·토큰·특징 축이 있다. |
| 차수 | rank/order | 텐서가 가진 축의 개수다. 여기서 3차 텐서는 축이 3개라는 뜻이며, 행렬의 선형독립 차원인 matrix rank와 다른 말이다. |
| shape | $B\times T\times d$ | 각 축의 길이를 순서대로 적은 계산 계약이다. |
| 첨자 | $X_{btc}$ | batch $b$, token $t$, feature $c$에 있는 스칼라 하나를 가리킨다. |
| slicing | $\mathbf{X}_{b,t,:}$ | 일부 첨자를 고정하고 한 축 전체를 꺼내는 연산이다. 여기서는 토큰 벡터 하나다. |
| reshape |  | 원소의 총수와 순서를 보존하며 축을 묶거나 나누는 연산이다. 의미 있는 축 순서를 자동으로 바꾸지는 않는다. |
| transpose / permute |  | 원소를 해석하는 축 순서를 바꾼다. head와 token 축을 바꿀 때 사용한다. |
| contiguous |  | 메모리에서 원소가 현재 축 순서와 잘 맞게 연속 배치된 상태다. permute 뒤 view가 별도 복사나 contiguous 변환을 요구할 수 있다. |
| 축 수축 | contraction | 두 텐서가 공유하는 한 축을 곱해 더함으로써 그 축을 없애는 일반화된 내적이다. |
| batch matrix multiplication | batched matmul | 서로 broadcast 가능한 앞쪽 축을 맞춘 뒤, 마지막 두 축에서 행렬곱을 반복한다. 오늘 Q·K처럼 앞쪽 shape가 같으면 그 $B,h$ 축이 그대로 유지된다. |
| broadcasting |  | 길이가 $1$인 축을 필요한 크기만큼 논리적으로 확장해 같은 연산을 적용하는 규칙이다. |
| einsum 표기 | `bhir,bhjr->bhij` | 어떤 첨자를 유지하고 어느 첨자를 합할지 명시하는 축 계산 표기다. |

> **두 rank를 구분하자:** “4차원 tensor”의 차수는 축이 네 개라는 뜻이다. “rank-$4$ 행렬”은 독립 방향이 네 개라는 뜻이다. 서로 다른 개념이다.

### 2.2 Transformer 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| batch size | $B$ | 한 번의 순전파에서 함께 처리하는 독립 표본 수다. |
| sequence length | $T$ | 한 표본 안의 토큰 위치 수다. padding 전에는 표본마다 다를 수 있다. |
| model dimension | $d$ | 토큰 표현 한 개가 가진 특징 좌표 수다. |
| head 수 | $h$ | 병렬 attention 갈래의 수다. |
| head dimension | $d_h$ | 한 head의 Query·Key·Value 특징 차원이다. 보통 $d_h=d/h$다. |
| multi-head attention | MHA | 여러 head의 attention 출력을 이어 붙인 뒤 출력 투영으로 섞는 모듈이다. |
| concat | $\operatorname{Concat}$ | head별 마지막 특징 축을 이어 $h d_h=d$로 복원하는 연산이다. 평균이 아니다. |
| output projection | $\mathbf{W}_O$ | concat한 head 특징들을 다시 섞는 학습 가능한 선형 변환이다. |
| positional encoding |  | 토큰의 위치나 상대 거리를 모델이 구분하도록 넣는 정보다. |
| RoPE | rotary position embedding | Query·Key의 2차원 좌표쌍을 위치별 각도로 회전해 내적에 상대 위치를 반영하는 방법이다. |
| padding mask |  | 길이를 맞추기 위해 붙인 빈 토큰이 Key로 선택되지 않게 막는다. |
| causal mask |  | 다음 토큰 예측에서 위치 $i$가 미래 $j>i$를 보지 못하게 막는다. |
| residual connection | $\mathbf{x}+F(\mathbf{x})$ | 입력을 그대로 가는 항등 경로와 학습된 수정 경로를 더한다. |
| LayerNorm | LN | 각 토큰의 특징 축을 평균·분산으로 정규화한 뒤 학습 가능한 크기와 이동을 적용한다. |
| pre-norm | $\mathbf{x}+F(\operatorname{LN}(\mathbf{x}))$ | 각 부분층에 넣기 전에 정규화하는 Transformer 배치 방식이다. |
| post-norm | $\operatorname{LN}(\mathbf{x}+F(\mathbf{x}))$ | 잔차 덧셈 뒤 정규화하는 원래 Transformer의 배치 방식이다. |
| FFN / MLP |  | 각 토큰에 독립적으로 같은 비선형 특징 변환을 적용하는 부분층이다. |
| $d_{\text{ff}}$ |  | FFN의 중간 확장 차원이다. 모델 차원보다 크게 두는 경우가 많다. |
| dropout |  | 학습 중 일부 활성값 또는 연결을 무작위로 끄는 정규화다. 평가·추론 때는 끈다. |
| KV cache |  | 자기회귀 생성에서 이미 계산한 과거 토큰의 Key·Value를 저장해 재사용하는 메모리다. |

### 2.3 오늘 사용할 기호와 shape

| 기호 | shape | 의미 |
|---|---:|---|
| $\mathbf{X}$ | $B\times T\times d$ | 블록 입력 |
| $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ | $d\times d$ | 모든 head의 Q·K·V를 한 번에 만드는 투영 |
| $\mathbf{Q},\mathbf{K},\mathbf{V}$ | $B\times h\times T\times d_h$ | head 축으로 나눈 표현 |
| $\mathbf{S}$ | $B\times h\times T\times T$ | Query 위치와 Key 위치의 scaled logits |
| $\mathbf{M}_{\text{pad}}$ | $B\times1\times1\times T$ | 표본별 padding Key mask |
| $\mathbf{M}_{\text{causal}}$ | $1\times1\times T\times T$ | 모든 표본·head가 공유하는 인과 마스크 |
| $\mathbf{A}$ | $B\times h\times T\times T$ | Key 축으로 Softmax한 attention 확률 |
| $\mathbf{O}_{\text{heads}}$ | $B\times h\times T\times d_h$ | head별 Value 가중합 |
| $\mathbf{O}$ | $B\times T\times d$ | concat과 출력 투영을 마친 MHA 출력 |
| $\boldsymbol{\gamma},\boldsymbol{\beta}$ | $d$ | LayerNorm의 학습 가능한 feature별 scale과 shift |
| $\mathbf{W}_1$ | $d\times d_{\text{ff}}$ | FFN 확장 투영 |
| $\mathbf{W}_2$ | $d_{\text{ff}}\times d$ | FFN 축소 투영 |

### 2.4 오늘의 트렌드 언어

| 용어 | 초보자 해설 |
|---|---|
| open-weight | 학습된 가중치를 내려받을 수 있다는 뜻이다. 라이선스가 OSI식 오픈소스이거나 학습 데이터·전체 학습 코드까지 공개되었다는 뜻은 아니다. |
| MoE | mixture-of-experts. 모든 FFN 전문가를 매 토큰 계산하지 않고 router가 일부만 고르는 희소 구조다. |
| activated parameters | 토큰 하나를 처리할 때 실제 계산 경로에 참여하는 파라미터 수다. 저장해야 하는 총 파라미터 수와 다르다. |
| Gated DeltaNet | 과거 전체 토큰의 attention 행렬을 만들기보다, key–value 관계를 고정 크기 상태에 갱신하는 recurrent token mixer다. |
| sparse attention | 모든 Query–Key 쌍을 계산하지 않고 중요하다고 고른 일부 연결만 계산하는 attention이다. |
| QSA | Qwen Sparse Attention. 작은 indexer가 중요한 문맥 micro-block을 고른 뒤 선택된 토큰을 attention하는 구조다. |
| micro-block | 가까운 몇 개 토큰을 묶어 선택 단위로 삼은 작은 블록이다. |
| gated residual | 잔차 stream의 여러 갈래에서 무엇을 읽고 쓸지 입력 의존 gate로 조절하는 구조다. |
| n-gram embedding | 한 토큰이 아니라 연속된 두세 토큰 조합을 색인해 가져오는 embedding이다. |
| prefill | 프롬프트 전체를 처음 읽어 각 층의 상태와 KV cache를 만드는 추론 단계다. |
| decode | cache를 사용해 새 토큰을 한 개씩 생성하는 단계다. |
| kernel | MatMul·Softmax·LayerNorm 같은 한 연산을 특정 하드웨어에서 실행하는 저수준 프로그램이다. |
| WebGPU | 브라우저 JavaScript에서 여러 GPU의 계산 기능을 공통 API로 사용하는 웹 표준 계층이다. |
| WGSL | WebGPU shader가 쓰는 언어다. GPU의 병렬 작업을 어떻게 실행할지 기술한다. |
| operator contract | 입력·출력·dtype·shape 규칙처럼 한 연산이 지켜야 할 인터페이스 명세다. |
| end-to-end latency | kernel 하나뿐 아니라 데이터 전송·컴파일·모든 모델 연산·후처리를 포함해 사용자가 실제로 기다리는 전체 시간이다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 텐서는 숫자 상자가 아니라 축의 계약이다

$\mathbf{X}\in\mathbb{R}^{B\times T\times d}$에서 토큰 벡터 하나는

$$
\mathbf{x}_{bt}
=
\mathbf{X}_{b,t,:}
\in\mathbb{R}^{d}
$$

이다. 같은 선형층 $\mathbf{W}\in\mathbb{R}^{d\times d'}$를 모든 표본과 모든 토큰에 적용하면

$$
Y_{btr}
=
\sum_{c=1}^{d}
X_{btc}W_{cr}
+b_r
$$

가 된다. 합으로 사라지는 축은 입력 feature 첨자 $c$이고, $b,t,r$은 출력에 남는다. 따라서

$$
\mathbf{Y}\in\mathbb{R}^{B\times T\times d'}.
$$

이 연산은 batch나 token을 서로 섞지 않는다. 각 $(b,t)$ 위치에서 같은 가중치 $\mathbf{W}$를 공유할 뿐이다.

### 3.2 축 수축으로 보는 행렬곱

일반 행렬곱

$$
C_{ij}
=
\sum_{k}A_{ik}B_{kj}
$$

는 공유 첨자 $k$를 곱해 더해 없애는 축 수축이다. attention의 Query–Key 점수는 이를 네 축으로 확장한다.

$$
S_{bhij}
=
\frac{1}{\sqrt{d_h}}
\sum_{r=1}^{d_h}
Q_{bhir}K_{bhjr}
+M_{bhij}
$$

첨자의 의미는 다음과 같다.

- $b$: 어느 표본인지 유지한다.
- $h$: 어느 head인지 유지한다.
- $i$: Query 토큰 위치를 유지한다.
- $j$: Key 토큰 위치를 유지한다.
- $r$: Q·K 특징 좌표를 곱해 더하므로 사라진다.

따라서 결과 shape는 $B\times h\times T\times T$다. einsum식으로 쓰면

```text
bhir, bhjr -> bhij
```

이다. 두 번째 텐서의 $j$가 Key 위치라는 점이 핵심이다.

Value 가중합은

$$
O_{bhir}
=
\sum_{j=1}^{T}
A_{bhij}V_{bhjr}
$$

이다. 이번에는 Key 위치 $j$가 사라지고 Value feature $r$이 남는다.

```text
bhij, bhjr -> bhir
```

즉 attention은 첫 번째 수축에서 토큰 쌍의 점수를 만들고, 두 번째 수축에서 그 점수로 Value를 섞는다.

### 3.3 batch matmul은 batch를 섞는 곱셈이 아니다

고정된 $(b,h)$를 하나 골라 보면

$$
\mathbf{Q}_{bh}\in\mathbb{R}^{T\times d_h},
\qquad
\mathbf{K}_{bh}^{\top}\in\mathbb{R}^{d_h\times T}
$$

이므로

$$
\mathbf{Q}_{bh}\mathbf{K}_{bh}^{\top}
\in\mathbb{R}^{T\times T}
$$

이다. batched matmul은 이 행렬곱을 모든 $(b,h)$ 조합에서 병렬로 반복한다.

$$
(B,h,T,d_h)
@
(B,h,d_h,T)
\longrightarrow
(B,h,T,T)
$$

$b$가 같은 조각끼리만 곱하므로 표본 $0$의 토큰이 표본 $1$의 토큰을 보지 않는다. batch 전체를 하나의 긴 $BT$ 토큰 시퀀스로 잘못 평탄화하면 이 독립성 계약이 깨진다.

### 3.4 broadcasting은 복사처럼 보이지만 축 규칙이다

padding mask는 표본마다 다르지만, 같은 표본의 모든 head와 Query 위치에서 동일한 Key 위치를 막는다. 그래서

$$
\mathbf{M}_{\text{pad}}
\in
\mathbb{R}^{B\times1\times1\times T}
$$

이면 충분하다. 길이가 $1$인 head 축과 Query 축이 논리적으로 반복되어

$$
B\times1\times1\times T
\Longrightarrow
B\times h\times T\times T
$$

로 방송된다.

인과 마스크는 모든 표본과 head가 공유하므로

$$
M^{\text{causal}}_{11ij}
=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i
\end{cases}
$$

인 $1\times1\times T\times T$ 텐서로 둘 수 있다. 두 마스크는 더해 함께 적용할 수 있다.

$$
\mathbf{M}
=
\mathbf{M}_{\text{pad}}
+
\mathbf{M}_{\text{causal}}
$$

개념적으로는 $-\infty$를 쓰지만, 실제 저정밀도 구현은 dtype에서 안전한 매우 작은 유한값이나 검증된 fused kernel을 쓸 수 있다. 한 Query 행의 모든 Key를 막으면 Softmax 분모가 $0$꼴이 되어 NaN이 생길 수 있으므로, 유효 Query만 계산하거나 완전 마스킹 행을 안전하게 처리해야 한다.

### 3.5 head를 나누고 합치는 것은 어떤 연산일까?

Q 투영을 한 번에 계산하면

$$
\widetilde{\mathbf{Q}}
=
\mathbf{X}\mathbf{W}_Q
\in
\mathbb{R}^{B\times T\times d}
$$

이다. $d=h d_h$일 때 마지막 축을 둘로 나눈다.

$$
(B,T,d)
\xrightarrow{\text{reshape}}
(B,T,h,d_h)
\xrightarrow{\text{permute}}
(B,h,T,d_h)
$$

reshape는 $d$축을 $h$와 $d_h$로 해석할 뿐이고, permute가 실제 의미상 head 축을 token 축 앞으로 옮긴다. 역방향은

$$
(B,h,T,d_h)
\xrightarrow{\text{permute}}
(B,T,h,d_h)
\xrightarrow{\text{reshape}}
(B,T,d)
$$

이다.

이 마지막 reshape는 head별 결과를 평균내는 것이 아니다. 각 head의 $d_h$개 좌표를 이어 붙이는 concat이다. 이후

$$
\mathbf{O}
=
\operatorname{Concat}
(\mathbf{O}^{(1)},\ldots,\mathbf{O}^{(h)})
\mathbf{W}_O
$$

가 head 간 정보를 학습 가능하게 다시 섞는다.

### 3.6 multi-head attention의 완전한 식

head $a\in\{1,\ldots,h\}$마다

$$
\begin{aligned}
\mathbf{Q}^{(a)}&=\mathbf{X}\mathbf{W}_Q^{(a)},\\
\mathbf{K}^{(a)}&=\mathbf{X}\mathbf{W}_K^{(a)},\\
\mathbf{V}^{(a)}&=\mathbf{X}\mathbf{W}_V^{(a)},
\end{aligned}
$$

이며 각 투영 행렬은 보통

$$
\mathbf{W}_Q^{(a)},
\mathbf{W}_K^{(a)},
\mathbf{W}_V^{(a)}
\in\mathbb{R}^{d\times d_h}
$$

이다. head 출력은

$$
\operatorname{head}_a
=
\operatorname{Softmax}_{\text{key}}
\left(
\frac{
\mathbf{Q}^{(a)}
(\mathbf{K}^{(a)})^{\top}
}{\sqrt{d_h}}
+\mathbf{M}
\right)
\mathbf{V}^{(a)}.
$$

전체 출력은

$$
\boxed{
\operatorname{MHA}(\mathbf{X})
=
\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_h)
\mathbf{W}_O
}
$$

이다.

모든 head의 가중치를 큰 $d\times d$ 행렬 세 개로 합쳐 구현할 수 있다. bias를 제외하고 $d_h=d/h$라면 QKV와 출력 투영의 파라미터 수는

$$
3d^2+d^2
=
4d^2
$$

이다. 따라서 $d$를 고정한 채 head 수만 바꾸면 이 단순한 MHA의 투영 파라미터 수는 대체로 변하지 않는다. head 수가 늘면 head 하나의 차원 $d_h$가 줄어든다.

### 3.7 Softmax는 어느 축에 적용해야 할까?

한 Query $i$가 모든 Key $j$ 중 어디를 참고할지 정하려면

$$
A_{bhij}
=
\frac{\exp(S_{bhij})}
{\sum_{m=1}^{T}\exp(S_{bhim})}
$$

이어야 한다. 분모가 Key 첨자 $m$에 대해 합을 취하므로

$$
\sum_{j=1}^{T}A_{bhij}=1
$$

이다. 즉 마지막 Key 축에 Softmax한다.

Query 축에 Softmax하면 “각 Query가 Key를 고르는 분포”가 아니라 “각 Key가 Query들 사이에서 어떻게 배분되는가”라는 다른 연산이 된다. shape가 같아 코드가 실행되더라도 의미는 틀릴 수 있다.

수치적으로는 각 행의 최댓값을 빼도 Softmax가 변하지 않는다.

$$
\frac{e^{s_j-c}}
{\sum_m e^{s_m-c}}
=
\frac{e^{s_j}}
{\sum_m e^{s_m}}
$$

따라서 보통 $c=\max_m s_m$을 빼 overflow를 줄인다.

### 3.8 위치가 없으면 왜 순열 등변적인가?

마스크가 없는 한 head를 생각하자. 순열행렬 $\mathbf{P}$로 토큰 순서를 바꾸면

$$
\mathbf{Q}'=\mathbf{P}\mathbf{Q},
\qquad
\mathbf{K}'=\mathbf{P}\mathbf{K},
\qquad
\mathbf{V}'=\mathbf{P}\mathbf{V}.
$$

점수 행렬은

$$
\mathbf{S}'
=
\frac{
\mathbf{Q}'(\mathbf{K}')^{\top}
}{\sqrt{d_h}}
=
\mathbf{P}
\mathbf{S}
\mathbf{P}^{\top}
$$

이다. 행과 열을 같은 순열로 바꾸면 행별 Softmax도 같은 방식으로 바뀐다.

$$
\mathbf{A}'
=
\mathbf{P}
\mathbf{A}
\mathbf{P}^{\top}
$$

따라서 출력은

$$
\mathbf{O}'
=
\mathbf{A}'\mathbf{V}'
=
\mathbf{P}\mathbf{A}\mathbf{P}^{\top}
\mathbf{P}\mathbf{V}
=
\mathbf{P}\mathbf{O}.
$$

내용을 같은 순열로 바꾸면 출력도 따라 바뀔 뿐, 모델은 원래 순서를 복원할 기준이 없다. 이 증명은 위치 의존 bias나 위치 표현, 순서를 고정하는 마스크가 없는 self-attention에 대한 것이다.

### 3.9 절대 위치 덧셈과 RoPE는 어떻게 다른가?

가장 직접적인 방법은 토큰 임베딩 $\mathbf{e}_t$에 위치 벡터 $\mathbf{p}_t$를 더하는 것이다.

$$
\mathbf{x}_t
=
\mathbf{e}_t+\mathbf{p}_t
$$

두 벡터를 더하려면 같은 차원이어야 한다. 위치 벡터는 학습할 수도 있고, 사인·코사인으로 고정할 수도 있다.

RoPE는 위치 $t$에 따라 Query와 Key의 2차원 좌표쌍을 회전한다. 한 좌표쌍에서 회전행렬을

$$
\mathbf{R}(t\theta)
=
\begin{bmatrix}
\cos(t\theta)&-\sin(t\theta)\\
\sin(t\theta)&\cos(t\theta)
\end{bmatrix}
$$

라고 하자. 위치 $i,j$의 회전된 Query와 Key 내적은

$$
\begin{aligned}
(\mathbf{R}(i\theta)\mathbf{q})^{\top}
(\mathbf{R}(j\theta)\mathbf{k})
&=
\mathbf{q}^{\top}
\mathbf{R}(i\theta)^{\top}
\mathbf{R}(j\theta)
\mathbf{k}\\
&=
\mathbf{q}^{\top}
\mathbf{R}((j-i)\theta)
\mathbf{k}.
\end{aligned}
$$

회전 차이가 상대 위치 $j-i$에 의존한다. 실제 RoPE는 여러 좌표쌍에 서로 다른 주파수를 사용한다. RoPE가 Value를 반드시 회전하는 것은 아니며, 표준적인 사용에서는 주로 Query와 Key에 적용한다.

### 3.10 LayerNorm의 축과 성질

고정된 표본 $b$, 토큰 $t$의 벡터를 $\mathbf{x}\in\mathbb{R}^{d}$라 하자.

$$
\mu
=
\frac{1}{d}\sum_{c=1}^{d}x_c,
\qquad
v
=
\frac{1}{d}\sum_{c=1}^{d}(x_c-\mu)^2
$$

표준화된 좌표는

$$
\widehat{x}_c
=
\frac{x_c-\mu}{\sqrt{v+\varepsilon}}
$$

이고 최종 출력은

$$
y_c
=
\gamma_c\widehat{x}_c+\beta_c
$$

이다. $\varepsilon=0$이고 $v>0$인 이상화된 경우

$$
\frac{1}{d}\sum_c\widehat{x}_c=0,
\qquad
\frac{1}{d}\sum_c\widehat{x}_c^2=1
$$

이다. 실제로는 $\varepsilon>0$이므로 두 번째 값은 정확히 $1$보다 조금 작을 수 있다.

LayerNorm은

$$
(B,T,d)
\longrightarrow
(B,T,d)
$$

로 shape를 바꾸지 않는다. 통계를 내며 사라지는 것은 계산 중의 feature 축일 뿐, $\gamma,\beta$를 적용해 원래 feature 좌표 수를 그대로 출력한다.

BatchNorm과 달리 batch 축 $B$의 평균을 쓰지 않는다. 그래서 “batch로 묶었으니 LayerNorm도 batch 전체 평균을 쓴다”는 해석은 틀리다.

### 3.11 잔차 연결은 그래디언트에 어떤 길을 만들까?

$$
\mathbf{y}
=
\mathbf{x}+F(\mathbf{x})
$$

이고 최종 손실이 $\mathcal{L}(\mathbf{y})$라고 하자. 열벡터 그래디언트 관례와

$$
[\mathbf{J}_F]_{ij}
=
\frac{\partial F_i}{\partial x_j}
$$

를 사용하면 연쇄법칙으로

$$
\nabla_{\mathbf{x}}\mathcal{L}
=
\left(
\mathbf{I}
+
\mathbf{J}_F(\mathbf{x})
\right)^{\top}
\nabla_{\mathbf{y}}\mathcal{L}
$$

이다. $\mathbf{J}_F$는 $F$의 야코비안이다. $F$ 경로의 도함수가 작아도 $\mathbf{I}$ 경로가 존재한다.

하지만 다음처럼 과장하면 안 된다.

- $\mathbf{I}+\mathbf{J}_F$가 항상 잘 조건화된다는 보장은 없다.
- 여러 층의 곱에서 폭주나 상쇄가 전혀 없어진다는 보장도 없다.
- residual은 안정성을 돕는 핵심 구조이지만, 정규화·초기화·optimizer와 함께 작동한다.

### 3.12 계산량과 메모리는 어디에서 커질까?

QKV 투영은 대략

$$
O(BTd^2)
$$

의 연산이 든다. attention 점수와 Value 가중합은

$$
O(BhT^2d_h)
=
O(BT^2d)
$$

이다. score 또는 확률을 명시적으로 저장하면 메모리는

$$
O(BhT^2)
$$

로 자란다. 시퀀스 길이 $T$가 두 배가 되면 이 부분의 원소 수는 네 배가 된다.

인과 마스크가 행렬의 절반가량을 금지한다는 사실만으로 일반적인 dense 구현의 이론적 shape가 줄어드는 것은 아니다. 실제 계산·메모리를 줄이려면 causal 구조를 활용하는 전용 kernel, block sparsity, sliding window 같은 구현이 필요하다. FlashAttention류 알고리즘은 정확한 attention 결과를 유지하면서 중간 $T\times T$ 행렬의 메모리 이동을 줄이지만, “attention 수학 자체가 선형 복잡도로 바뀐다”는 뜻은 아니다.

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 하나의 pre-norm Transformer 블록 조립도

오늘은 설명을 위해 다음 canonical pre-norm 블록을 사용한다.

```text
입력 X  [B, T, d]
  │
  ├───────────────────────────────┐  residual path
  │                               │
  └─ LayerNorm ─ MHA ─ Dropout ─ (+) ──> Y [B, T, d]
                                          │
                                          ├──────────────────────┐
                                          │                      │
                                          └─ LayerNorm ─ FFN ─ Dropout ─ (+) ──> Z [B, T, d]
```

식으로는

$$
\boxed{
\mathbf{Y}
=
\mathbf{X}
+
\operatorname{Dropout}
\left(
\operatorname{MHA}
(\operatorname{LN}_1(\mathbf{X}))
\right)
}
$$

$$
\boxed{
\mathbf{Z}
=
\mathbf{Y}
+
\operatorname{Dropout}
\left(
\operatorname{FFN}
(\operatorname{LN}_2(\mathbf{Y}))
\right)
}
$$

이다. 잔차 덧셈을 하려면 두 항의 shape가 같아야 하므로 MHA와 FFN은 모두 최종적으로 $B\times T\times d$를 출력한다.

이는 한 가지 대표 설계다. 원래 Transformer는 post-norm을 사용했고, 현대 LLM은 LayerNorm 대신 RMSNorm, 일반 FFN 대신 SwiGLU 같은 gated MLP, 여러 Q head가 K·V를 공유하는 GQA 등을 자주 사용한다. 이 변형들은 오늘의 기본 축 계약 위에서 이해할 수 있다.

### 4.2 1단계: 토큰과 위치를 입력 텐서로 만든다

토큰 ID 텐서를

$$
\mathbf{I}\in\{0,\ldots,V-1\}^{B\times T}
$$

라 하고, 임베딩 표를

$$
\mathbf{E}\in\mathbb{R}^{V\times d}
$$

라 하자. lookup 후에는

$$
\mathbf{X}_{\text{tok}}
\in\mathbb{R}^{B\times T\times d}
$$

가 된다.

절대 위치 임베딩을 쓴다면

$$
\mathbf{P}_{\text{pos}}
\in\mathbb{R}^{1\times T\times d}
$$

를 batch 축에 broadcasting해

$$
\mathbf{X}
=
\mathbf{X}_{\text{tok}}
+
\mathbf{P}_{\text{pos}}
$$

로 더할 수 있다. RoPE를 쓰는 모델에서는 입력에 이 벡터를 더하는 대신 attention 안에서 Q·K에 회전을 적용하는 경우가 일반적이다.

padding한 위치는 값 자체를 $0$으로 만들었다고 끝나지 않는다. 선형층의 bias, 위치 표현, residual 때문에 다시 0이 아닌 값이 될 수 있다. 중요한 것은 padding 위치가 유효 Key로 선택되지 않도록 mask를 올바르게 적용하고, 필요하면 loss에서도 제외하는 것이다.

### 4.3 2단계: QKV를 한 번에 투영하고 head로 나눈다

실제 구현은 흔히

$$
\mathbf{W}_{QKV}
\in\mathbb{R}^{d\times3d}
$$

를 사용해

$$
[\widetilde{\mathbf{Q}},
\widetilde{\mathbf{K}},
\widetilde{\mathbf{V}}]
=
\operatorname{LN}(\mathbf{X})
\mathbf{W}_{QKV}
$$

를 한 번의 큰 선형연산으로 계산한다. 결과를 세 덩어리로 나눈 뒤 각각

```text
[B, T, d]
→ reshape [B, T, h, d_h]
→ permute [B, h, T, d_h]
```

로 바꾼다.

예를 들어

$$
B=2,
\quad T=3,
\quad d=8,
\quad h=2,
\quad d_h=4
$$

라면 Q는 $2\times3\times8=48$개 원소를 유지한 채 $2\times2\times3\times4=48$개 원소로 재해석된다.

### 4.4 3단계: 위치를 반영하고 attention을 계산한다

RoPE 모델이라면 head로 나눈 Q·K의 마지막 feature 축 좌표쌍에 위치별 회전을 적용한다. 그 뒤

$$
\mathbf{S}
=
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}}
+
\mathbf{M}_{\text{pad}}
+
\mathbf{M}_{\text{causal}}
$$

를 계산한다.

shape를 실제 숫자로 추적하면

$$
(2,2,3,4)
@
(2,2,4,3)
\rightarrow
(2,2,3,3)
$$

이다. 마지막 축에 Softmax한 뒤 Value를 곱하면

$$
(2,2,3,3)
@
(2,2,3,4)
\rightarrow
(2,2,3,4)
$$

가 된다.

마스크는 Softmax **전에** 로짓에 적용해야 한다. Softmax 뒤 확률을 단순히 $0$으로만 만들면 행의 합이 $1$보다 작아진다. 뒤에서 다시 정규화하면 수학적으로 같은 결과를 만들 수는 있지만, 표준 구현은 금지된 로짓을 먼저 제외해 하나의 안정적인 Softmax에서 처리한다.

### 4.5 4단계: head를 합치고 출력 투영한다

head별 출력

$$
\mathbf{O}_{\text{heads}}
\in\mathbb{R}^{B\times h\times T\times d_h}
$$

의 축을 바꿔

$$
B\times T\times h\times d_h
$$

로 만든 다음 마지막 두 축을 합친다.

$$
\operatorname{Concat}(\mathbf{O}_{\text{heads}})
\in\mathbb{R}^{B\times T\times d}
$$

출력 투영

$$
\mathbf{W}_O\in\mathbb{R}^{d\times d}
$$

를 적용해도 shape는 그대로다. 이 결과에 dropout을 적용하고 원래 $\mathbf{X}$를 더하면 첫 잔차 출력 $\mathbf{Y}$가 된다.

### 4.6 5단계: FFN은 각 토큰의 특징을 비선형 가공한다

각 토큰 벡터에 동일한 FFN을 독립 적용한다.

$$
\operatorname{FFN}(\mathbf{x})
=
\phi(
\mathbf{x}\mathbf{W}_1+\mathbf{b}_1
)
\mathbf{W}_2+\mathbf{b}_2
$$

shape는

$$
d
\xrightarrow{\mathbf{W}_1}
d_{\text{ff}}
\xrightarrow{\phi}
d_{\text{ff}}
\xrightarrow{\mathbf{W}_2}
d
$$

이다. 전체 텐서에서는

$$
(B,T,d)
\rightarrow
(B,T,d_{\text{ff}})
\rightarrow
(B,T,d).
$$

여기서는 token 축 $T$끼리 섞이지 않는다. attention이 다른 위치의 정보를 현재 토큰으로 가져온 뒤, FFN이 그 위치 안의 feature를 변환한다. ReLU, GELU, SiLU 같은 활성화가 없다면 bias까지 포함한 두 affine 층은 하나의 affine 변환으로 합쳐져 표현력이 제한된다.

### 4.7 학습과 자기회귀 추론은 tensor 흐름이 어떻게 다를까?

학습에서는 정답 문장 전체를 한 번에 넣되 causal mask로 미래를 가린다. 길이 $T$의 모든 Query 위치를 병렬 계산할 수 있다.

자기회귀 추론에서는 새 토큰을 한 개씩 만든다. 과거 $T$개 토큰의 K·V를 매번 다시 계산하면 낭비이므로 층별로 cache한다.

새 Query만 계산할 때 한 층의 전형적인 shape는

$$
\mathbf{Q}_{\text{new}}
\in\mathbb{R}^{B\times h\times1\times d_h},
$$

$$
\mathbf{K}_{\text{cache}},
\mathbf{V}_{\text{cache}}
\in\mathbb{R}^{B\times h\times T\times d_h}
$$

이다. 그러면 새 점수는

$$
B\times h\times1\times T
$$

만 계산한다. KV cache는 계산을 줄이지만 시퀀스 길이에 비례하는 메모리를 사용한다. GQA·MQA는 여러 Query head가 K·V head를 공유해 이 cache 크기를 줄이는 설계다.

### 4.8 PyTorch식 의사코드로 shape 확인하기

다음 코드는 개념을 보여 주는 의사코드다. 실전에서는 검증된 fused attention API가 수치 안정성·속도·mask 처리를 더 잘 제공할 수 있다.

```python
# x: [B, T, d], d = h * d_h
u = ln1(x)                              # [B, T, d]
qkv = qkv_proj(u)                       # [B, T, 3d]
q, k, v = qkv.chunk(3, dim=-1)          # each [B, T, d]

q = q.reshape(B, T, h, d_h).transpose(1, 2)  # [B, h, T, d_h]
k = k.reshape(B, T, h, d_h).transpose(1, 2)  # [B, h, T, d_h]
v = v.reshape(B, T, h, d_h).transpose(1, 2)  # [B, h, T, d_h]

scores = q @ k.transpose(-2, -1) / sqrt(d_h) # [B, h, T, T]
scores = scores + padding_mask + causal_mask
weights = softmax(scores, dim=-1)              # sum over Key axis
heads = weights @ v                            # [B, h, T, d_h]

merged = heads.transpose(1, 2).contiguous().reshape(B, T, d)
attn_out = out_proj(merged)                    # [B, T, d]
y = x + dropout(attn_out)                      # first residual
z = y + dropout(ffn(ln2(y)))                   # second residual
```

여기서 `.contiguous()`는 수학 연산이 아니라 permute 뒤 메모리 배치를 안전하게 정리하기 위한 구현 세부다. 프레임워크와 연산에 따라 복사가 생길 수도 있다.

### 4.9 수학 지식과 AI 부품의 정확한 매핑

| 수학 도구 | Transformer에서의 부품 | 확인해야 할 계약 |
|---|---|---|
| 첨자와 축 | $X_{btc}$, $Q_{bhir}$ | 각 첨자가 표본·head·토큰·특징 중 무엇인지 |
| 선형변환 | QKV·output·FFN 투영 | 마지막 입력 feature와 가중치 첫 축이 같은지 |
| 축 수축 | $\mathbf{Q}\mathbf{K}^{\top}$ | feature 축만 합하고 batch·head를 유지하는지 |
| 내적 | Query–Key 유사도 | $d_h$가 같고 $1/\sqrt{d_h}$로 스케일하는지 |
| 확률 정규화 | 행별 Softmax | Key 축 합이 $1$인지 |
| 가중평균 | $\mathbf{A}\mathbf{V}$ | Key 위치 축을 합해 Query별 출력을 만드는지 |
| broadcasting | padding·causal mask | 길이 $1$인 축이 의도한 방향으로만 확장되는지 |
| 순열행렬 | 위치 없는 attention | 순서 정보 없이는 출력이 입력 순열을 따라가는지 |
| 2차원 회전 | RoPE | $\mathbf{R}_i^{\top}\mathbf{R}_j$가 상대 위치에 의존하는지 |
| 평균·분산 | LayerNorm | batch가 아니라 각 토큰의 feature 축을 정규화하는지 |
| 벡터 덧셈 | residual | $F(\mathbf{x})$와 $\mathbf{x}$의 shape가 같은지 |
| 연쇄법칙 | residual 역전파 | $\mathbf{I}+\mathbf{J}_F$의 항등 경로가 있는지 |
| 비선형 함수 | FFN 활성화 | 두 선형층이 하나로 붕괴하지 않는지 |

### 4.10 초보자가 흔히 하는 오해와 주의할 점

1. **“batch 축도 attention이 섞는다.”** 표준 self-attention은 각 표본 안의 token 축만 섞는다. batch 축은 병렬 반복 축이다.

2. **“reshape와 transpose는 같은 연산이다.”** reshape는 축을 묶고 나누며, transpose는 축의 순서를 바꾼다. head split에는 둘의 의미를 모두 구분해야 한다.

3. **“multi-head는 attention 한 개를 $h$번 똑같이 복사한다.”** head별 투영과 점수가 다르다. 단, 서로 다른 역할이 자동으로 보장되지는 않는다.

4. **“head 수를 늘리면 항상 모델 파라미터도 늘어난다.”** $d$가 고정되고 $d_h=d/h$이면 표준 QKV·output 투영의 총 파라미터 수는 대체로 $4d^2$로 같다.

5. **“Softmax는 어느 축에 해도 shape가 같으니 괜찮다.”** Key 축에 해야 각 Query가 Key들을 고르는 분포가 된다. 실행 성공은 의미의 정확성을 보장하지 않는다.

6. **“mask는 Softmax 뒤 확률을 0으로만 만들면 된다.”** 그러면 합이 $1$이 아닐 수 있다. 보통 Softmax 전 로짓에서 제외한다.

7. **“padding 임베딩을 0으로 만들었으니 mask가 필요 없다.”** 위치·bias·잔차 때문에 0이 유지되지 않을 수 있고, 0인 Value라도 Key로 확률 질량을 빼앗는다.

8. **“LayerNorm은 batch 평균을 쓴다.”** Transformer의 LayerNorm은 각 토큰 안의 feature 축 통계를 쓴다. BatchNorm과 다르다.

9. **“residual이면 어떤 깊이도 자동으로 안정적이다.”** 항등 그래디언트 경로를 제공하지만 정규화, 초기화, residual scaling, optimizer 설계가 여전히 중요하다.

10. **“위치 표현이 없으면 causal mask도 아무 순서 정보를 주지 않는다.”** causal mask는 과거·미래 비대칭과 위치별 가용 prefix 크기를 만든다. 다만 일반적인 거리·상대 위치를 직접 표현하는 별도 positional mechanism과 동일하지 않다.

11. **“attention map이 곧 모델의 완전한 설명이다.”** 확률은 Value를 섞는 한 단계일 뿐이며 residual, output projection, FFN, 이후 층이 결과를 다시 바꾼다.

12. **“이 구조가 모든 현대 LLM의 정확한 구현이다.”** 오늘은 canonical pre-norm 블록이다. 실제 모델은 RMSNorm, RoPE 변형, GQA, gated MLP, mixture-of-experts, parallel residual 등 다양한 설계를 쓴다.

### 4.11 구현 전후 최소 검증 체크리스트

- 입력과 출력이 모두 $[B,T,d]$인지 확인한다.
- $d=h d_h$가 성립하는지 확인한다.
- score가 $[B,h,T,T]$인지 확인한다.
- `weights.sum(dim=-1)`이 유효 Query에서 수치 오차 범위 내 $1$인지 확인한다.
- 금지된 Key의 확률이 $0$인지 확인한다.
- 표본 하나를 바꿨을 때 다른 batch 표본의 출력이 바뀌지 않는지 확인한다.
- padding 길이를 바꿔도 유효 토큰 출력이 허용 오차 안에서 유지되는지 확인한다.
- LayerNorm의 통계 축이 마지막 feature 축인지 확인한다.
- dropout이 평가 모드에서 비활성화되는지 확인한다.
- 완전히 mask된 행과 저정밀 dtype에서 NaN·Inf가 생기지 않는지 확인한다.
- fused kernel 사용 시 비-fused 기준 구현과 작은 입력에서 출력·그래디언트를 비교한다.

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **선정 기준:** 2026-09-14 KST 현재 최근 약 2주 안팎에 공개된 1차 출처 가운데, 오늘 배운 attention·residual·tensor kernel이 실제 최신 모델과 로컬 실행 환경에서 어떻게 변형되는지 보여 주는 두 사례를 골랐다. 아래 성능 수치는 독립 재현값이 아니라 각 제작진이 지정한 조건에서 보고한 결과다.

### 5.1 Qwen3.8-Flash-Next: “고전 Transformer 블록”을 왜 하이브리드로 바꾸는가?

Qwen 팀은 2026-08-26 **Qwen3.8-Flash-Next**의 가중치를 공개했고, 2026-08-31에는 아키텍처와 ablation을 담은 기술 보고서를 arXiv에 제출했다. 텍스트와 이미지를 처리하는 multimodal sparse MoE이며, Qwen 팀은 향후 Qwen4 계열에 사용할 아키텍처의 조기 미리보기라고 설명한다.

공식 자료가 구분하는 파라미터 규모는 다음과 같다.

- main model: 총 $125$B, 토큰당 약 $6$B 활성
- accelerator 밖 host memory에 둘 수 있는 n-gram embedding: 추가 $51$B
- Hugging Face 모델 카드에 별도로 적힌 multi-token prediction(MTP) 모듈: $4$B

따라서 Hub가 전체 artifact를 $180$B로 표시하는 것과 논문이 “$125$B main model + $51$B n-gram table”이라고 설명하는 것은 집계 범위가 다르다. **토큰당 $6$B 활성**은 저장·다운로드할 가중치가 $6$B뿐이라는 뜻도 아니다.

#### 오늘의 canonical block에서 무엇이 달라졌나?

오늘 조립한 블록의 token mixer는 모든 토큰 쌍을 비교하는 dense self-attention이었다. 이 비용은

$$
O(T^2d)
$$

로 커진다. Qwen3.8-Flash-Next의 48개 층은 모델 카드 기준으로 다음 패턴을 12번 반복한다.

$$
3\times
(\text{GDN}\rightarrow\text{MoE})
+
1\times
(\text{QSA}\rightarrow\text{MoE})
$$

즉 네 층 중 세 층은 **Gated DeltaNet(GDN)**, 한 층은 **Qwen Sparse Attention(QSA)**을 쓴다.

GDN은 매 시점 $t$에서 과거를 고정 크기 상태

$$
\mathbf{S}_t
\in
\mathbb{R}^{d_k\times d_v}
$$

에 압축한다. 보고서의 head별 핵심 갱신은 다음과 같다.

$$
\widetilde{\mathbf{S}}_{t-1}
=
\alpha_t\mathbf{S}_{t-1}
$$

$$
\mathbf{e}_t
=
\mathbf{v}_t
-
\widetilde{\mathbf{S}}_{t-1}^{\top}
\mathbf{k}_t
$$

$$
\mathbf{S}_t
=
\widetilde{\mathbf{S}}_{t-1}
+
\beta_t
\mathbf{k}_t
\mathbf{e}_t^{\top},
\qquad
\mathbf{y}_t
=
\mathbf{S}_t^{\top}
\mathbf{q}_t.
$$

$\alpha_t\in(0,1)$는 이전 기억의 감쇠, $\beta_t\in(0,1)$는 새 key–value 관계를 쓰는 세기다. 단순히 $\mathbf{k}_t\mathbf{v}_t^{\top}$를 계속 더하지 않고, 현재 상태가 예측하지 못한 오차 $\mathbf{e}_t$만 갱신한다.

이 recurrent 상태는 긴 prefix를 고정 크기로 압축해 선형 비용 경로를 만들지만, 유한 상태만으로 임의의 과거 토큰을 원본 그대로 직접 찾는 능력에는 한계가 있다. 그래서 주기적으로 QSA를 둔다.

QSA는 작은 indexer가 문맥을 micro-block 단위로 점수화한 뒤 일부만 선택한다. 모델 카드의 released configuration은 QSA에

- Query head $24$개, KV head $2$개인 grouped-query 구조
- head dimension $256$
- indexer top-$K$ budget $512$ blocks, 즉 $2{,}048$ tokens와 별도로 마지막 미완성 block의 tail $0$–$3$ tokens를 더한 선택 범위

를 적고 있다. 오늘의 $[B,h,T,T]$ dense score 전체를 항상 materialize하는 대신, **먼저 어떤 token block을 읽을지 고르는 축**이 추가된 셈이다.

#### residual도 한 갈래가 아니다

오늘 배운 기본 residual은

$$
\mathbf{y}
=
\mathbf{x}+F(\mathbf{x})
$$

였다. Qwen의 Gated Residual(GR)은 residual stream을 네 갈래로 넓힌다. 입력에 따른 **element-wise read gate**가 특징별로 어느 갈래에서 읽을지 조절하고, **branch별 scalar write gate**가 부분층 출력을 각 갈래에 얼마나 쓸지 조절한다. “항등 경로를 남긴다”는 residual의 목적은 유지하면서, 어느 갈래의 정보를 다음 부분층에 노출할지를 학습하는 것이다.

보고서는 이 gate가 loss만 조금 낮추는 데 그치지 않고 downstream 평가와 고학습률 stress test의 안정성에도 영향을 주었다고 보고한다. 이는 residual 설계를 평가할 때 파라미터 수나 pretraining loss 하나만 보아서는 안 된다는 사례다.

#### 공식 보고 수치와 경계

기술 보고서의 비교에서 이 base model은 이전 $397$B-A$17$B 모델보다

- 토큰당 활성 파라미터 약 $1/3$
- 학습 토큰 약 $1/3$
- 학습 FLOPs 약 $1/9$

을 사용했다. 14개 pretraining benchmark 중 8개에서는 앞섰고, 나머지 6개에서는 최대 $2.6$점 뒤졌다고 저자들이 보고한다.

긴 문맥 kernel 실험에서는 $1$M context에서 QSA가 dense attention보다 prefill $7.6\times$, decode $4.9\times$ 빠르다고 보고했다. 공개된 비교 조건은 FlashInfer paged GQA baseline, prefill의 $16$K chunk·batch $1$, decode의 batch $4$·`next_n=4`이지만, 보고서에는 GPU 모델과 정확한 software commit이 명시되지 않았다. 따라서 이 숫자는 정확한 독립 재현 조건이 완전히 닫히지 않은 **저자 측 QSA kernel 수준** 결과다. API 요청 전체 latency, vision encoder, MoE routing, host의 $51$B n-gram table 전송까지 포함한 보편적 end-to-end 배속이 아니다.

모델 카드는 입력과 출력의 합에 대한 native context를 $262{,}144$ tokens, **static YaRN** 설정을 통한 확장 상한을 $1{,}000{,}000$으로 적는다. 그러나 “입력이 들어간다”와 “모든 종류의 1M-token 추론이 정확하다”는 다른 주장이다. static YaRN을 항상 켜면 짧은 문맥 성능에 영향을 줄 수 있다는 카드의 주의도 있다. 실제 서비스 전에는 needle retrieval뿐 아니라 짧은 입력 회귀, 장문 생성 안정성, 메모리, prefill time, task 정확도를 함께 측정해야 한다.

#### 엔지니어 인사이트 (Impact)

- **최신 block은 부품 교체보다 역할 분담에 가깝다.** GDN은 지속적 압축 기억을 맡고, released model은 전역 full dense attention을 선택적 sparse QSA로 바꿔 softmax 기반 직접 검색 경로를 주기적으로 남긴다.
- **오늘의 shape 추적이 그대로 필요하다.** QSA의 grouped-query head 수, 선택 block 축, GDN 상태의 $d_k\times d_v$ shape를 모르면 효율화가 어떤 정보를 버리는지 판단할 수 없다.
- **residual도 학습 가능한 메모리 대역이다.** 네 갈래 GR은 단순 skip connection보다 gate·branch tensor를 더 요구한다. fused 연산과 branch state의 FP8 저장은 보고서가 제시한 대역폭·overhead 최적화 경로이지 GR 수학의 필수 조건은 아니다.
- **활성 파라미터와 resident memory를 구분해야 한다.** $6$B 활성이라는 문구만 보고 소형 GPU에 들어간다고 결론 내리면 안 된다. 총 main weights, n-gram table, KV/state, quantization dtype을 따로 계산해야 한다.
- **새 구조는 kernel 성숙도가 제품 리스크다.** 표준 dense attention만 지원하는 runtime에서는 QSA·GDN의 이론적 이점을 얻지 못하거나 fallback이 필요할 수 있다. FlashQLA는 GDN kernel을 공개하지만, 위 $7.6\times$·$4.9\times$를 만든 QSA kernel과 raw benchmark는 release에서 공개되지 않았다.
- **공개 범위를 정확히 부르자.** 공개 artifact는 약 $180$B BF16 post-trained checkpoint·config·tokenizer·기술 보고서다. base weights, 전체 사전학습 데이터, 완전한 훈련·평가 pipeline은 포함되지 않았다. Hugging Face 표기는 `qwen-community-1.0`이며, $1$억 MAU 또는 월매출 미화 $20{,}000{,}000$달러를 넘는 상용 제품에는 모델명 표시 조건이 있다. 제3자에게 모델 기능을 제공하는 상용 MaaS와, 코딩·오피스 생산성이 주목적인 독립 AI Work Assistant에는 별도 라이선스가 요구되지만 내부 사용과 라이선스가 열거한 단일 목적·타 도메인·부가 기능에는 예외가 있다. Apache-2.0 코드와 같은 의미의 “완전한 오픈소스 모델”로 뭉뚱그리지 말고 원문과 사용 사례를 확인해야 한다.

**1차 출처:** [Qwen 공식 발표](https://qwen.ai/blog?id=qwen3.8-flash-next) · [기술 보고서와 제출 이력](https://arxiv.org/abs/2608.30320) · [공식 GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next) · [공식 Hugging Face 모델 카드](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) · [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) · [FlashQLA kernel 저장소](https://github.com/QwenLM/FlashQLA)

### 5.2 Hugging Face WebGPU Kernels: 수식이 브라우저에서 빨라지려면 무엇이 더 필요한가?

Hugging Face는 2026-09-01 **`@huggingface/kernels`** preview와 Apache-2.0 라이선스의 WebGPU operator repository $207$개를 공개했다. 여기서 $207$은 공개된 연산 interface·repository의 수이며 WGSL shader 파일이 정확히 $207$개라는 뜻은 아니다. 한 연산도 shape와 장치에 따라 여러 shader variant를 가질 수 있다. 오늘 배운 Transformer를 실제 GPU 프로그램으로 내리면 결국 다음과 같은 작은 연산들의 연쇄가 된다.

$$
\text{MatMul}
\rightarrow
\text{Add/Broadcast}
\rightarrow
\text{Softmax}
\rightarrow
\text{MatMul}
\rightarrow
\text{Residual Add}
\rightarrow
\text{LayerNorm}
$$

같은 수식을 구현해도 shape, dtype, GPU, 브라우저, workgroup 크기, 메모리 접근 순서에 따라 속도가 크게 달라진다. WebGPU가 여러 장치를 위한 공통 API를 제공한다고 해서 한 shader가 모든 장치에서 자동으로 빠른 것은 아니다.

#### shader 파일이 아니라 재현 가능한 연산 단위

이 공개의 핵심은 WGSL 코드만 모았다는 데 있지 않다. 각 kernel repository가 다음 artifact를 함께 갖는다.

| 파일 | 역할 |
|---|---|
| `manifest.json` | 입력·출력·attribute·dtype·shape 파생 규칙을 정의하는 contract |
| `metadata.json` | kernel 식별자, digest, provenance 기록 |
| `test.json` | 예상 출력과 비교하는 correctness 사례 |
| `bench.json` | 성능 비교와 variant 선택에 쓰는 shape 사례 |
| `*.wgsl.jinja` | 요청 shape와 장치에 맞춰 shader를 만드는 template |

예를 들어 residual과 bias에 공통으로 쓰이는 Add는

$$
(2,3)+(3)
\longrightarrow
(2,3)
$$

같은 multidirectional broadcasting을 정확히 구현해야 한다. equal-shape, vectorized broadcast, scalar, general broadcast는 수학적 출력은 같지만 최적 kernel이 다를 수 있다.

이는 오늘의 “shape는 계산 계약”이라는 원칙을 저수준 실행까지 확장한다. kernel 최적화는 contract를 바꾸어 답을 근사하는 일이 아니라, 같은 contract를 지키면서 데이터 이동과 병렬 실행을 더 효율적으로 만드는 일이다.

#### 공식 benchmark를 어떻게 읽어야 할까?

Hugging Face 팀은 Apple M4 GPU에서 자신들의 kernel과 ONNX Runtime Web `1.30.0-dev.20260826-b1f76d586a`를 비교했다. $207$개 연산의 $1{,}756$개 test case에서 양쪽 출력이 일치하고 timing이 신뢰 가능하다고 판단한 $809$개, 즉 약 $46.1\%$만 집계했다. 나머지 $947$개, 약 $53.9\%$는 이 비교 통계에서 제외되었으므로, 발표만으로 각 제외 원인이 미지원·출력 불일치·timing 불안정 중 무엇이었는지 모두 분해할 수는 없다.

그 범위의 제작진 보고 결과는 다음과 같다.

- geometric mean speedup: $2.57\times$
- median speedup: $1.90\times$
- $629$ wins, $176$ losses, $4$ ties

오늘 배운 핵심 연산의 일부 결과는 다음과 같다.

| 연산 | 비교 case 수 | HF kernel | ORT WebGPU | 보고 배속 |
|---|---:|---:|---:|---:|
| Add | $5$ | $0.064$ ms | $0.227$ ms | $3.52\times$ |
| MatMul | $29$ | $0.115$ ms | $0.131$ ms | $1.14\times$ |
| Softmax | $12$ | $0.114$ ms | $0.240$ ms | $2.11\times$ |
| LayerNormalization | $6$ | $0.061$ ms | $0.135$ ms | $2.22\times$ |

하지만 이 표는 서로 다른 shape case의 요약이며 완전한 Transformer 한 번의 시간이 아니다. 측정은 GPU가 실제 작업한 시간에 집중했고 다음을 제외했다.

- kernel 다운로드와 session 준비
- shader 컴파일
- 입력 upload와 출력 readback
- tokenizer·runtime scheduling·나머지 model graph

또한 Apple M4 한 장치의 결과이며, 발표에는 OS·browser·driver, tolerance, warmup·반복 횟수, $1{,}756$개 case 전체의 raw 결과와 제외 사유가 모두 제시되지는 않았다. 발표도 매우 짧은 workload는 timing이 어렵고 GPU cache의 영향을 받을 수 있다고 경고한다. 따라서 Windows·Android·다른 브라우저·GPU에서 같은 배속을 보장하지 않는다. 바로 이 이유로 프로젝트는 사용자의 동의 아래 여러 실제 장치에서 correctness와 성능 증거를 모으는 **Fleet** 도구도 공개했다.

#### 엔지니어 인사이트 (Impact)

- **큰-O가 같아도 실제 시간은 다르다.** MatMul과 Softmax의 수학식이 같아도 memory layout, vectorization, fusion, shape specialization이 지연을 바꾼다.
- **연산별 최고 배속을 곱하면 안 된다.** kernel 하나의 GPU 시간과 모델 end-to-end latency 사이에는 compilation·transfer·scheduling·동기화가 있다.
- **정확성이 성능보다 먼저다.** 공개 benchmark도 출력이 일치한 case만 비교했다. 저정밀 attention과 LayerNorm에서는 tolerance, NaN, mask edge case를 별도 검증해야 한다.
- **shape가 곧 workload다.** $[B,h,T,d_h]$의 구체적인 값이 달라지면 최적 workgroup과 kernel variant도 달라진다. 평균 benchmark보다 실제 제품 shape를 측정해야 한다.
- **브라우저 로컬 AI의 공급망 단위가 작아진다.** 전체 runtime 릴리스를 기다리지 않고 versioned operator contract와 kernel을 독립적으로 검토·고정할 수 있다. 동시에 원격 artifact의 digest, 버전, 라이선스를 배포 manifest에 기록해야 한다.
- **오늘의 기초가 바로 성능 디버깅 언어다.** residual Add의 broadcasting, attention MatMul의 transpose, Softmax 축, LayerNorm 축을 모르면 빠르지만 틀린 kernel을 탐지할 수 없다.

**1차 출처:** [Hugging Face 공식 발표](https://huggingface.co/blog/webgpu-kernels) · [발표 원문 GitHub](https://github.com/huggingface/blog/blob/main/webgpu-kernels.md) · [공개 WebGPU kernel 모음](https://huggingface.co/webgpu-kernels) · [`@huggingface/kernels` npm 패키지](https://www.npmjs.com/package/@huggingface/kernels)

### 5.3 두 흐름을 함께 읽기: 모델 아키텍처와 kernel은 공동 설계 대상이다

Qwen3.8-Flash-Next는 dense $T\times T$ attention을 GDN 상태와 sparse block 선택으로 바꾸고, residual stream도 네 갈래 gate로 넓힌다. Hugging Face의 WebGPU 공개는 그런 고수준 연산이 결국 장치별 kernel contract와 memory layout으로 내려가야 실제 속도가 된다는 사실을 보여 준다.

$$
\boxed{
\text{수학적 연산}
\rightarrow
\text{tensor shape와 sparsity}
\rightarrow
\text{kernel·memory layout}
\rightarrow
\text{실제 latency·정확성}
}
$$

따라서 새로운 attention 논문을 읽을 때는 benchmark 점수만 보지 말고 다음 네 질문을 함께 던져야 한다.

1. 어떤 축을 완전히 계산하고 어떤 축을 압축·선택하는가?
2. 학습과 prefill, decode에서 복잡도가 각각 어떻게 달라지는가?
3. 공개 runtime이 그 구조를 native kernel로 지원하는가?
4. kernel 수준 배속이 end-to-end 과업 정확도와 latency로 이어지는가?

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

다음 하나의 상황으로 batch·head·mask·LayerNorm·residual의 연결을 점검하자.

한 pre-norm self-attention 블록의 입력이

$$
\mathbf{X}\in\mathbb{R}^{2\times3\times8}
$$

이고, head 수는 $h=2$다. 두 번째 표본은 실제 토큰이 두 개뿐이라 마지막 위치가 padding이다. decoder형 causal attention을 사용하며, FFN 중간 차원은 $d_{\text{ff}}=32$다. bias는 파라미터 수에서 제외한다.

**하나의 핵심 질문:** “이 블록은 어느 축을 유지하고, 어느 축을 섞고, 어디서 shape를 되돌리는가?”를 다음 항목으로 답하라.

1. $d_h$와 head 분할 뒤 Q·K·V의 shape를 구하라.
2. score, attention weight, head별 출력, concat 출력, 최종 MHA 출력의 shape를 차례대로 구하라.
3. 축 의미를 명시적으로 드러내는 padding mask와 causal mask의 canonical 4D broadcastable shape를 쓰고, 두 번째 표본의 Query 위치 $i=1$이 허용받는 Key를 설명하라. 위치는 $0,1,2$로 센다.
4. Softmax가 어느 축에 적용되어야 하며 어떤 합이 $1$인지 첨자로 쓰라.
5. QKV·output 투영과 FFN의 가중치 파라미터 수를 각각 구하라.
6. 한 토큰의 LayerNorm 입력이 $\mathbf{x}=[1,1,1,1,3,3,3,3]$이고 $\boldsymbol{\gamma}=\mathbf{1}_8$, $\boldsymbol{\beta}=\mathbf{0}_8$, $\varepsilon=0$이라고 단순화할 때 출력을 구하라.
7. 위치 정보와 mask가 없는 attention에서 batch 안 각 시퀀스의 토큰 순서를 같은 순열 $\mathbf{P}$로 바꾸면 출력이 어떻게 변하는지 쓰라.
8. MHA 부분층을 $F$라 할 때 $\mathbf{y}=\mathbf{x}+F(\operatorname{LN}(\mathbf{x}))$의 residual이 역전파에 제공하는 핵심 경로를 설명하라.

### 모범 답안

#### 1. head 차원과 QKV shape

$$
d_h
=
\frac{d}{h}
=
\frac{8}{2}
=4
$$

이다. 따라서 head 분할 뒤

$$
\mathbf{Q},\mathbf{K},\mathbf{V}
\in
\mathbb{R}^{2\times2\times3\times4}
$$

이다.

#### 2. attention 내부의 shape 흐름

Query–Key 점수는

$$
(2,2,3,4)
@
(2,2,4,3)
\rightarrow
(2,2,3,3)
$$

이므로

$$
\mathbf{S},\mathbf{A}
\in\mathbb{R}^{2\times2\times3\times3}.
$$

Value 가중합은

$$
(2,2,3,3)
@
(2,2,3,4)
\rightarrow
(2,2,3,4)
$$

이므로 head별 출력은 $2\times2\times3\times4$다. head 축을 token 뒤로 옮겨 concat하면

$$
2\times3\times(2\cdot4)
=
2\times3\times8
$$

이다. $\mathbf{W}_O\in\mathbb{R}^{8\times8}$를 곱한 최종 MHA 출력도 $2\times3\times8$이어서 residual 덧셈이 가능하다.

#### 3. 두 mask와 허용 Key

padding mask의 최소 shape는

$$
2\times1\times1\times3
$$

이고, causal mask는

$$
1\times1\times3\times3
$$

이다.

두 번째 표본의 실제 토큰 위치는 $0,1$이고 위치 $2$는 padding이다. Query $i=1$은 causal 조건으로 $j\le1$만 볼 수 있으므로 $j=0,1$이 허용된다. $j=2$는 미래이면서 padding이므로 금지된다. 이 Query에서는 허용 Key가 적어도 하나 있어 Softmax가 정상적으로 정의된다.

#### 4. Softmax 축

Key 위치 $j$, 즉 마지막 축에 적용한다.

$$
A_{bhij}
=
\frac{e^{S_{bhij}}}
{\sum_{m=1}^{T}e^{S_{bhim}}}
$$

이므로 유효한 각 $(b,h,i)$에 대해

$$
\sum_{j=1}^{T}A_{bhij}=1
$$

이다.

#### 5. 파라미터 수

Q·K·V 투영은 각각 $8\times8$이고 출력 투영도 $8\times8$이다.

$$
N_{\text{MHA}}
=
3\cdot8^2+8^2
=
4\cdot64
=256
$$

이다.

FFN은 $8\times32$와 $32\times8$ 두 행렬을 쓰므로

$$
N_{\text{FFN}}
=
8\cdot32
+
32\cdot8
=512
$$

이다. 이 문항은 MHA와 FFN의 **가중치 행렬** 파라미터만 물었으므로 LayerNorm의 $\boldsymbol{\gamma},\boldsymbol{\beta}$는 별도이며, 모든 bias는 문제 조건에 따라 제외했다.

#### 6. LayerNorm 계산

평균은

$$
\mu
=
\frac{4\cdot1+4\cdot3}{8}
=2
$$

이다. 분산은

$$
v
=
\frac{4(1-2)^2+4(3-2)^2}{8}
=1
$$

이다. 따라서 $\varepsilon=0$, $\boldsymbol{\gamma}=\mathbf{1}_8$, $\boldsymbol{\beta}=\mathbf{0}_8$에서

$$
\operatorname{LN}(\mathbf{x})
=
[-1,-1,-1,-1,1,1,1,1]
$$

이다. 이 통계는 다른 batch 표본이나 다른 토큰을 사용하지 않는다.

#### 7. 위치 없는 attention의 순열

마스크와 위치 의존 항이 없고 같은 순열을 각 시퀀스의 token 축에 적용하면

$$
\operatorname{Attn}(\mathbf{X}')_{b,:,:}
=
\mathbf{P}
\operatorname{Attn}(\mathbf{X})_{b,:,:}
$$

이다. 입력 토큰 순서를 바꾸면 출력도 같은 순서로 따라 바뀐다. 순서를 원래대로 복원할 절대 기준은 생기지 않는다.

#### 8. residual의 그래디언트 경로

함수 전체를

$$
G(\mathbf{x})
=
F(\operatorname{LN}(\mathbf{x}))
$$

라고 쓰면, 표준 Jacobian 관례에서

$$
\mathbf{J}_G(\mathbf{x})
=
\mathbf{J}_F(\operatorname{LN}(\mathbf{x}))
\mathbf{J}_{\operatorname{LN}}(\mathbf{x})
$$

이다. 또한

$$
\mathbf{y}
=
\mathbf{x}+G(\mathbf{x})
$$

이고

$$
\frac{\partial\mathbf{y}}
{\partial\mathbf{x}}
=
\mathbf{I}
+
\mathbf{J}_G(\mathbf{x})
$$

이다. 열벡터 그래디언트 관례에서는

$$
\nabla_{\mathbf{x}}\mathcal{L}
=
\left(
\mathbf{I}
+
\mathbf{J}_G(\mathbf{x})
\right)^{\top}
\nabla_{\mathbf{y}}\mathcal{L}.
$$

역전파 신호에는 변환 경로 $\mathbf{J}_G$뿐 아니라 입력으로 직접 이어지는 항등 경로 $\mathbf{I}$가 있다. 이것이 residual의 핵심 이점이다. 다만 전체 깊은 네트워크의 안정성을 단독으로 보장하지는 않는다.

---

**다음 연결 고리:** 오늘은 Transformer 한 블록의 모든 축을 조립했다. 다음에는 이 블록을 여러 층 쌓을 때 학습 목표가 어떻게 만들어지는지, tokenization·언어 모델링 objective·teacher forcing·causal loss를 통해 살펴본다.
