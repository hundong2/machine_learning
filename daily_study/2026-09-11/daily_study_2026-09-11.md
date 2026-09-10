# [2026-09-11] 오늘 학습: 특잇값분해(SVD)·저랭크 근사 & Query·Key·Value 어텐션

> **오늘의 핵심 문장:** SVD는 행렬을 서로 직교하는 입력 방향, 축별 크기, 출력 방향으로 분해해 중요한 성분만 남길 수 있게 하고, attention은 Query와 Key의 scaled dot-product로 각 위치의 Value를 얼마나 섞을지 정한다.

지난 학습에서는 평균 중심화된 데이터

$$
\mathbf{X}_c\in\mathbb{R}^{N\times d}
$$

의 공분산 행렬을 고유분해해 PCA 축을 찾았다.

$$
\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
=
\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}
$$

오늘은 직사각행렬 자체를 직접 분해하는 특잇값분해, 즉 SVD로 이 결과를 다시 바라본다. 랭크가 $r$인 행렬 $\mathbf{X}\in\mathbb{R}^{m\times n}$의 compact SVD는

$$
\boxed{
\mathbf{X}
=
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}
}
$$

이다. 여기서

$$
\mathbf{U}_r\in\mathbb{R}^{m\times r},
\qquad
\boldsymbol{\Sigma}_r\in\mathbb{R}^{r\times r},
\qquad
\mathbf{V}_r\in\mathbb{R}^{n\times r}
$$

이며 $\boldsymbol{\Sigma}_r$의 대각 원소

$$
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0
$$

를 특잇값이라 한다. 앞의 $k$개 성분만 남기면

$$
\mathbf{X}_k
=
\mathbf{U}_k
\boldsymbol{\Sigma}_k
\mathbf{V}_k^{\top}
=
\sum_{i=1}^{k}
\sigma_i\mathbf{u}_i\mathbf{v}_i^{\top}
$$

이고, 이는 랭크가 $k$ 이하인 모든 행렬 가운데 $\mathbf{X}$를 가장 잘 근사한다.

AI 기초 트랙에서는 임베딩으로 만든 토큰 표현

$$
\mathbf{H}\in\mathbb{R}^{T\times d_{\text{model}}}
$$

에서 세 가지 투영을 만든다.

$$
\mathbf{Q}=\mathbf{H}\mathbf{W}_Q,
\qquad
\mathbf{K}=\mathbf{H}\mathbf{W}_K,
\qquad
\mathbf{V}_{\text{attn}}=\mathbf{H}\mathbf{W}_V
$$

그리고

$$
\boxed{
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}_{\text{attn}})
=
\operatorname{Softmax}_{\text{row}}
\left(
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}
+\mathbf{M}
\right)
\mathbf{V}_{\text{attn}}
}
$$

을 계산한다. 오늘의 목표는 이 공식을 외우는 것이 아니다. 각 행렬의 shape, $\sqrt{d_k}$의 이유, 행별 Softmax와 마스크의 역할, 그리고 SVD·저랭크 근사가 이 연산과 어디에서 만나고 어디에서 갈라지는지를 바닥부터 이해하는 것이다.

> **기호 충돌 주의:** SVD의 오른쪽 특이벡터 행렬 $\mathbf{V}_r$와 attention의 Value 행렬은 관습적으로 모두 $\mathbf{V}$라 쓴다. 오늘 문서에서는 혼동을 막기 위해 Value를 $\mathbf{V}_{\text{attn}}$라고 표기한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 고유분해만으로는 왜 부족했을까?

고유방정식

$$
\mathbf{A}\mathbf{v}
=
\lambda\mathbf{v}
$$

은 같은 공간에서 출발해 같은 공간으로 돌아오는 정사각행렬 $\mathbf{A}\in\mathbb{R}^{n\times n}$에 자연스럽다. 하지만 실제 데이터는 표본 $m$개와 특징 $n$개를 쌓은

$$
\mathbf{X}\in\mathbb{R}^{m\times n}
$$

처럼 직사각행렬인 경우가 많다. $m\ne n$이면 $\mathbf{X}\mathbf{v}$와 $\mathbf{v}$의 차원부터 다르므로 $\mathbf{X}\mathbf{v}=\lambda\mathbf{v}$를 그대로 쓸 수 없다.

또한 정사각행렬이라도 대칭이 아니면 실수 고유벡터가 충분하지 않거나, 직교기저로 깔끔하게 분해되지 않을 수 있다. 우리는 다음을 동시에 원한다.

- 직사각행렬에도 적용할 수 있어야 한다.
- 입력 공간과 출력 공간의 중요한 방향을 따로 찾아야 한다.
- 각 방향이 얼마나 강하게 전달되는지 숫자로 비교할 수 있어야 한다.
- 작은 성분을 버렸을 때 생기는 오차를 정확히 측정할 수 있어야 한다.

SVD는 이 요구를 만족한다. 모든 실수 행렬은 직교 방향과 음수가 아닌 축별 크기로 분해할 수 있다.

### 1.2 행렬에 정말 중요한 방향이 따로 있을까?

사진 한 장을 행렬로 저장했다고 하자. 이웃한 픽셀은 대개 비슷하고, 배경이나 조명처럼 여러 행과 열에 반복되는 패턴이 있다. 행렬의 모든 원소가 서로 독립적인 정보를 담는 것은 아니다.

SVD는 행렬을 랭크 $1$인 얇은 패턴들의 합으로 쓴다.

$$
\mathbf{X}
=
\sigma_1\mathbf{u}_1\mathbf{v}_1^{\top}
+\sigma_2\mathbf{u}_2\mathbf{v}_2^{\top}
+\cdots
+\sigma_r\mathbf{u}_r\mathbf{v}_r^{\top}
$$

$\mathbf{u}_i\mathbf{v}_i^{\top}$는 열 방향 패턴과 행 방향 패턴을 바깥곱으로 결합한 랭크 $1$ 행렬이다. $\sigma_i$는 그 패턴의 세기다. 큰 $\sigma_i$ 몇 개가 대부분을 설명한다면 작은 성분을 버리고도 원본을 가깝게 복원할 수 있다.

책을 한 글자씩 저장하는 대신 자주 반복되는 문장 구조를 사전처럼 저장하는 것과 비슷하다. 단, SVD의 패턴은 사람이 이름 붙인 의미 단위가 아니라 데이터가 정하는 직교 선형 방향이다.

### 1.3 왜 단순히 작은 원소를 지우면 안 될까?

행렬 원소 하나의 절댓값이 작다고 해서 중요하지 않은 것은 아니다. 작은 값들이 많은 위치에서 같은 구조를 이루면 전체적으로 큰 효과를 낼 수 있다. 반대로 큰 원소 몇 개가 잡음일 수도 있다.

저랭크 근사는 원소별 크기가 아니라 행렬이 입력 벡터를 어떤 독립 방향으로 얼마나 변환하는지를 본다. SVD의 truncated 근사

$$
\mathbf{X}_k
=
\sum_{i=1}^{k}
\sigma_i\mathbf{u}_i\mathbf{v}_i^{\top}
$$

는 큰 특잇값에 대응하는 전역적인 구조를 남긴다. 이것은 대부분을 $0$으로 만드는 희소화와 다른 압축 방식이다.

### 1.4 PCA 다음에 SVD를 배우는 이유는 무엇일까?

PCA에서는 중심화 데이터의 공분산 행렬

$$
\boldsymbol{\Sigma}_{\text{cov}}
=
\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
$$

를 만들었다. 하지만 특징 수 $d$가 크면 $d\times d$ 행렬을 명시적으로 만들고 고유분해하는 데 비용이 들고, 행렬곱 과정에서 조건수가 제곱되어 수치 오차가 커질 수 있다.

중심화 행렬을 직접 SVD하면

$$
\mathbf{X}_c
=
\mathbf{U}
\boldsymbol{\Sigma}_{\text{svd}}
\mathbf{V}^{\top}
$$

이고

$$
\mathbf{X}_c^{\top}\mathbf{X}_c
=
\mathbf{V}
\boldsymbol{\Sigma}_{\text{svd}}^2
\mathbf{V}^{\top}
$$

이므로 PCA의 주성분 방향은 $\mathbf{V}$의 열이고 공분산 고윳값은

$$
\lambda_i
=
\frac{\sigma_i^2}{N}
$$

이다. SVD는 PCA 뒤에 갑자기 등장한 별개의 마술이 아니라, 지금까지 배운 고유분해를 직사각 데이터 행렬에 확장한 계산 언어다.

### 1.5 임베딩만으로 문맥을 만들 수 있을까?

지난 학습의 임베딩 표는 같은 토큰 ID에 같은 기본 벡터를 준다. 그런데 다음 두 문장에서 “눈”의 의미는 다르다.

- 밤새 눈이 내렸다.
- 카메라의 눈 역할을 하는 렌즈다.

토큰 자체의 임베딩만으로는 현재 위치에서 어느 주변 단어를 참고해야 하는지 정할 수 없다. 모든 주변 벡터를 똑같이 평균 내면 중요한 단어와 불필요한 단어가 같은 비중을 가진다. 고정된 창만 보면 멀리 떨어진 관계를 놓치며, 한 방향으로 차례대로 읽는 구조는 긴 의존성을 전달하는 경로가 길어진다.

attention은 각 위치가 다른 위치와 직접 점수를 계산하고, 현재 입력에 따라 섞는 비율을 바꾼다.

### 1.6 왜 하나의 벡터를 Query·Key·Value 세 역할로 나눌까?

도서관 검색을 생각해 보자.

- Query는 지금 찾는 내용이다.
- Key는 각 책이 어떤 질문과 잘 맞는지를 나타내는 색인표다.
- Value는 실제로 꺼내 읽을 내용이다.

검색 기준과 전달할 내용을 같은 벡터로 강제할 이유는 없다. 예를 들어 한 토큰은 문법적으로는 동사와 잘 연결되어야 하지만, 전달해야 하는 내용에는 시제·수·의미 정보가 함께 들어갈 수 있다. 학습 가능한 $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$는 같은 입력 표현을 서로 다른 역할의 공간으로 투영한다.

중요한 점은 Key가 데이터베이스의 고정된 정수 ID가 아니고, Query가 사람이 작성한 자연어 질문만을 뜻하지 않는다는 것이다. self-attention에서는 모든 토큰 위치가 자신의 입력 표현으로 Query, Key, Value를 모두 만든다.

### 1.7 왜 내적 뒤에 $\sqrt{d_k}$로 나눌까?

Query와 Key의 각 좌표가 평균 $0$, 분산 $1$ 정도라고 단순화하자. 독립이라는 근사 아래 내적

$$
\mathbf{q}^{\top}\mathbf{k}
=
\sum_{\ell=1}^{d_k}q_{\ell}k_{\ell}
$$

의 분산은 약 $d_k$가 된다. 차원 $d_k$가 커질수록 로짓의 전형적인 크기가 $\sqrt{d_k}$에 비례해 커진다.

큰 로짓을 Softmax에 넣으면 가장 큰 항의 확률이 거의 $1$, 나머지는 거의 $0$이 되기 쉽다. 이 포화 영역에서는 그래디언트가 작아져 학습이 불안정해질 수 있다. 따라서

$$
\frac{\mathbf{q}^{\top}\mathbf{k}}{\sqrt{d_k}}
$$

로 스케일을 맞춰 로짓의 분산을 대략 $1$ 수준으로 유지한다. 이는 어떤 입력에서도 확률이 균등해진다는 보장이 아니라, 차원이 증가했다는 이유만으로 Softmax가 지나치게 날카로워지는 현상을 완화하는 초기화 관점의 동기다.

### 1.8 오늘 두 트랙은 어디에서 만날까?

SVD와 attention은 모두 내적, 직교 방향, 행렬곱을 사용하지만 목적은 다르다.

- SVD는 고정된 행렬을 가장 중요한 직교 성분으로 분해하고 저랭크 근사 오차를 최소화한다.
- attention은 입력마다 달라지는 Query–Key 점수에 Softmax를 적용해 Value의 문맥적 가중합을 만든다.
- 마스크를 더하기 전 raw 점수 $\mathbf{Q}\mathbf{K}^{\top}$는 $d_k$차원 표현을 거쳐 만들어지므로 랭크가 최대 $d_k$다.
- 그러나 행별 Softmax는 비선형이므로, 그 뒤의 attention 가중치 행렬이 반드시 랭크 $d_k$ 이하인 것은 아니다.
- SVD는 $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ 같은 큰 가중치 행렬을 근사하거나 attention 표현의 유효 차원을 진단하는 도구가 될 수 있다.
- SVD에서 작은 Frobenius 오차가 곧 언어 모델의 작은 정확도 손실을 보장하지는 않는다. 최종 과업 평가가 별도로 필요하다.

## 2. 친절한 용어 사전

### 2.1 SVD·저랭크 근사의 수학 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 행렬의 shape | $m\times n$ | 행이 $m$개, 열이 $n$개라는 뜻이다. 곱셈 가능 여부를 확인하는 첫 번째 계약이다. |
| 랭크 | $\operatorname{rank}(\mathbf{X})=r$ | 행렬이 실제로 사용하는 서로 독립인 입력·출력 방향의 수다. |
| 특잇값분해 | SVD | 직사각행렬을 왼쪽 직교 방향, 축별 크기, 오른쪽 직교 방향으로 분해하는 방법이다. |
| 왼쪽 특이벡터 | $\mathbf{u}_i$ | 입력이 행렬을 지난 뒤 나타나는 출력 공간의 단위 방향이다. |
| 오른쪽 특이벡터 | $\mathbf{v}_i$ | 행렬에 넣을 입력 공간의 단위 방향이다. |
| 특잇값 | $\sigma_i$ | $\mathbf{v}_i$ 방향이 행렬을 통과할 때 얼마나 늘어나는지 나타내는 음수가 아닌 크기다. |
| compact SVD | $\mathbf{U}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^{\top}$ | 특잇값이 양수인 $r$개 성분만 보관한 경제적인 SVD다. |
| full SVD | $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$ | 입력·출력 공간의 직교기저를 0 특잇값 방향까지 완성한 형태다. |
| 바깥곱 | $\mathbf{u}\mathbf{v}^{\top}$ | 열벡터와 행벡터를 곱해 만드는 랭크 $1$ 행렬이다. |
| truncated SVD | $\mathbf{X}_k$ | 큰 특잇값에 대응하는 앞의 $k$개 성분만 남긴 근사다. |
| 저랭크 근사 | low-rank approximation | 원래 행렬보다 적은 독립 방향으로 원본을 가깝게 표현하는 방법이다. |
| 프로베니우스 노름 | $\lVert\mathbf{A}\rVert_F$ | 모든 원소의 제곱합에 제곱근을 취한 행렬 전체 오차다. |
| 스펙트럴 노름 | $\lVert\mathbf{A}\rVert_2$ | 단위 입력을 넣었을 때 가능한 최대 출력 길이다. 가장 큰 특잇값과 같다. |
| 에카르트–영–미르스키 정리 | Eckart–Young–Mirsky | truncated SVD가 Frobenius 노름과 spectral 노름에서 최적의 랭크-$k$ 근사라는 정리다. |
| 영공간 | null space | 행렬을 곱했을 때 영벡터가 되는 입력들의 집합이다. |
| 행공간 | row space | 행렬의 행들이 만드는 입력 공간의 부분공간이다. |
| 직교 투영자 | $\mathbf{P}$ | $\mathbf{P}^{\top}=\mathbf{P}$, $\mathbf{P}^2=\mathbf{P}$를 만족하며 벡터를 부분공간 위로 내리는 행렬이다. |
| 수치 랭크 | numerical rank | 정확한 0 대신 허용 오차보다 충분히 큰 특잇값의 개수로 판단한 실용적 랭크다. |
| 유효 조건수 | $\kappa_{2,\mathrm{eff}}=\sigma_1/\sigma_r$ | 비영 특이공간에서 가장 강한 방향과 가장 약한 방향의 비율이다. 크면 역문제가 민감할 수 있다. 랭크 결손 행렬의 전체 역문제는 특이하다. |

### 2.2 Attention의 AI 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 시퀀스 길이 | $T$ | 한 입력에 있는 토큰 위치의 수다. |
| 모델 차원 | $d_{\text{model}}$ | 각 토큰 표현이 가진 기본 좌표 수다. |
| Query | $\mathbf{Q}$ | 각 위치가 지금 어떤 정보를 찾는지 표현하는 벡터 묶음이다. |
| Key | $\mathbf{K}$ | 각 위치가 어떤 Query와 잘 맞는지 비교하기 위한 색인 벡터 묶음이다. |
| Value | $\mathbf{V}_{\text{attn}}$ | 선택된 비율에 따라 실제로 전달·혼합될 내용 벡터 묶음이다. |
| Query·Key 차원 | $d_k$ | Query와 Key 한 벡터의 좌표 수다. 내적하려면 두 차원이 같아야 한다. |
| Value 차원 | $d_v$ | Value 한 벡터의 좌표 수다. $d_k$와 같을 수도, 다를 수도 있다. |
| 로짓 | $s_{ij}$ | Softmax 전 위치 $i$의 Query와 위치 $j$의 Key가 맞는 정도를 나타내는 점수다. |
| scaled dot-product | $\mathbf{q}_i^{\top}\mathbf{k}_j/\sqrt{d_k}$ | 내적을 차원에 맞게 스케일한 attention 점수다. |
| 행별 Softmax | $\operatorname{Softmax}_{\text{row}}$ | 각 Query 행 안에서 모든 Key 후보에 대한 가중치의 합이 $1$이 되도록 바꾸는 함수다. |
| attention 가중치 | $a_{ij}$ | 위치 $i$가 출력할 때 위치 $j$의 Value를 섞는 비율이다. 음수가 아니며 한 행의 합은 $1$이다. |
| 마스크 | $\mathbf{M}$ | 허용되지 않는 연결의 로짓을 보통 $-\infty$로 만들어 확률을 $0$으로 하는 장치다. |
| 인과 마스크 | causal mask | 다음 토큰 예측에서 미래 위치 $j>i$를 보지 못하게 막는다. |
| 패딩 마스크 | padding mask | 길이를 맞추기 위해 넣은 빈 토큰을 실제 정보처럼 참고하지 못하게 막는다. |
| self-attention |  | Query, Key, Value가 같은 입력 시퀀스에서 나오는 attention이다. |
| cross-attention |  | Query와 Key·Value가 서로 다른 입력에서 나오는 attention이다. |
| head |  | 독립된 투영과 attention을 계산하는 한 갈래다. 여러 head가 서로 다른 관계를 학습할 여지를 만든다. |
| 문맥 표현 | contextual representation | 주변 위치의 정보를 입력에 따라 섞어 만든, 문맥에 따라 달라지는 토큰 벡터다. |
| 잔차 연결 | residual connection | attention 출력에 원래 입력 경로를 더해 정보와 그래디언트 흐름을 돕는 연결이다. |
| KV cache |  | 자기회귀 생성에서 과거 토큰의 Key와 Value를 다시 계산하지 않도록 저장한 메모리다. |

### 2.3 기호와 shape 한눈에 보기

| 기호 | shape | 역할 |
|---|---:|---|
| $\mathbf{X}$ | $m\times n$ | SVD할 일반 행렬 |
| $\mathbf{U}_r$ | $m\times r$ | 왼쪽 특이벡터를 열로 모은 행렬 |
| $\boldsymbol{\Sigma}_r$ | $r\times r$ | 양의 특잇값을 내림차순으로 둔 대각행렬 |
| $\mathbf{V}_r$ | $n\times r$ | 오른쪽 특이벡터를 열로 모은 행렬 |
| $\mathbf{X}_k$ | $m\times n$ | 랭크-$k$ truncated SVD 근사 |
| $\mathbf{H}$ | $T\times d_{\text{model}}$ | 한 시퀀스의 입력 토큰 표현 |
| $\mathbf{W}_Q,\mathbf{W}_K$ | $d_{\text{model}}\times d_k$ | Query·Key 투영 가중치 |
| $\mathbf{W}_V$ | $d_{\text{model}}\times d_v$ | Value 투영 가중치 |
| $\mathbf{Q},\mathbf{K}$ | $T\times d_k$ | 모든 위치의 Query·Key |
| $\mathbf{V}_{\text{attn}}$ | $T\times d_v$ | 모든 위치의 Value |
| $\mathbf{S}$ | $T\times T$ | 모든 Query–Key 쌍의 로짓 |
| $\mathbf{A}$ | $T\times T$ | 행별 Softmax를 지난 attention 가중치 |
| $\mathbf{O}$ | $T\times d_v$ | 문맥을 섞은 attention 출력 |

### 2.4 오늘의 트렌드 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| VLM | vision-language model | 이미지와 텍스트를 같은 과업에서 연결해 이해하거나 생성하는 모델이다. |
| CLIP |  | 이미지와 텍스트 표현을 가까운 벡터 공간에 맞추도록 학습한 대표적인 vision-language encoder다. |
| 연속 프롬프트 | continuous prompt | 사람이 읽는 단어 대신 학습 가능한 벡터들을 입력 앞에 붙여 모델을 적응시키는 방식이다. |
| CoOp | Context Optimization | CLIP의 텍스트 문맥 벡터를 downstream 분류 데이터로 학습하는 프롬프트 적응 방법이다. |
| few-shot | 1·4·16-shot | 클래스마다 매우 적은 수의 라벨 예제만 사용해 적응·평가하는 설정이다. |
| base-to-new 일반화 |  | 일부 본 클래스에 적응한 뒤 보지 않은 새 클래스에도 지식이 유지되는지 재는 평가다. |
| 조화평균 | $H$ | 본 클래스와 새 클래스 성능 중 한쪽만 높아서는 큰 값이 나오지 않게 두 성능을 결합한 평균이다. |
| attention sink | AS | 많은 뒤쪽 Query가 특정 위치, 흔히 첫 위치에 과도한 attention을 보내는 현상이다. |
| massive activation | MA | 중간 표현의 일부 좌표가 다른 값보다 매우 큰 크기로 나타나는 현상이다. |
| self-concentration | $\alpha_{t,t}=1$ | 한 위치의 attention이 자기 자신의 Value에만 집중하는 상태다. |
| Value-non-mixing |  | attention 출력에서 서로 다른 위치의 Value가 섞이지 않고 한 Value만 통과하는 상태다. |
| RoPE | rotary position embedding | Query와 Key를 위치에 따라 회전시켜 상대 위치 정보를 점수에 반영하는 방법이다. |
| 저비트 양자화 | low-bit quantization | 가중치·활성값을 적은 비트로 표현해 메모리와 계산을 줄이는 방법이다. 큰 이상치는 오차를 키울 수 있다. |

## 3. 수학의 해부학 (증명과 원리)

### 3.1 SVD의 표기와 두 가지 형태

일반 행렬을

$$
\mathbf{X}\in\mathbb{R}^{m\times n}
$$

이라 하고 랭크를 $r$이라 하자.

#### Compact SVD

0보다 큰 특잇값에 해당하는 방향만 남기면

$$
\mathbf{X}
=
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}
$$

이다. 각 행렬의 shape는

$$
\underbrace{\mathbf{X}}_{m\times n}
=
\underbrace{\mathbf{U}_r}_{m\times r}
\underbrace{\boldsymbol{\Sigma}_r}_{r\times r}
\underbrace{\mathbf{V}_r^{\top}}_{r\times n}
$$

이다. 열들은 직교정규다.

$$
\mathbf{U}_r^{\top}\mathbf{U}_r
=
\mathbf{I}_r,
\qquad
\mathbf{V}_r^{\top}\mathbf{V}_r
=
\mathbf{I}_r
$$

다만 $r<m$이면

$$
\mathbf{U}_r\mathbf{U}_r^{\top}\ne\mathbf{I}_m
$$

이고 $r<n$이면

$$
\mathbf{V}_r\mathbf{V}_r^{\top}\ne\mathbf{I}_n
$$

이다. 각각 열공간과 행공간 위의 투영자다.

#### Full SVD

직교기저를 전체 공간까지 완성하면

$$
\mathbf{X}
=
\mathbf{U}
\boldsymbol{\Sigma}
\mathbf{V}^{\top}
$$

로 쓸 수 있다. 여기서

$$
\mathbf{U}\in\mathbb{R}^{m\times m},
\qquad
\mathbf{V}\in\mathbb{R}^{n\times n},
\qquad
\boldsymbol{\Sigma}\in\mathbb{R}^{m\times n}
$$

이고

$$
\mathbf{U}^{\top}\mathbf{U}
=
\mathbf{U}\mathbf{U}^{\top}
=
\mathbf{I}_m,
\qquad
\mathbf{V}^{\top}\mathbf{V}
=
\mathbf{V}\mathbf{V}^{\top}
=
\mathbf{I}_n
$$

이다. $\boldsymbol{\Sigma}$는 직사각 대각 형태이며 앞의 $r$개 대각 원소만 양수다.

실제 계산과 압축에서는 compact SVD가 더 경제적이지만, 이론을 설명할 때 full SVD가 편할 때도 있다. 둘은 서로 모순되는 다른 분해가 아니다.

### 3.2 SVD는 $\mathbf{X}^{\top}\mathbf{X}$의 고유분해에서 어떻게 나올까?

$\mathbf{X}^{\top}\mathbf{X}$는 항상 $n\times n$ 대칭행렬이다.

$$
\left(
\mathbf{X}^{\top}\mathbf{X}
\right)^{\top}
=
\mathbf{X}^{\top}\mathbf{X}
$$

임의의 $\mathbf{z}\in\mathbb{R}^n$에 대해

$$
\mathbf{z}^{\top}
\mathbf{X}^{\top}\mathbf{X}
\mathbf{z}
=
\left(
\mathbf{X}\mathbf{z}
\right)^{\top}
\left(
\mathbf{X}\mathbf{z}
\right)
=
\lVert\mathbf{X}\mathbf{z}\rVert_2^2
\ge0
$$

이므로 양의 준정부호다. 따라서 직교정규 고유벡터를 가지며 고윳값은 음수가 아니다.

$$
\mathbf{X}^{\top}\mathbf{X}\mathbf{v}_i
=
\sigma_i^2\mathbf{v}_i,
\qquad
\sigma_i\ge0
$$

양의 특잇값 $\sigma_i>0$에 대해

$$
\mathbf{u}_i
=
\frac{\mathbf{X}\mathbf{v}_i}{\sigma_i}
$$

라고 정의하자. 그러면

$$
\mathbf{X}\mathbf{v}_i
=
\sigma_i\mathbf{u}_i
$$

다. $\mathbf{u}_i$의 길이는

$$
\begin{aligned}
\lVert\mathbf{u}_i\rVert_2^2
&=
\frac{1}{\sigma_i^2}
\mathbf{v}_i^{\top}
\mathbf{X}^{\top}\mathbf{X}
\mathbf{v}_i\\
&=
\frac{1}{\sigma_i^2}
\mathbf{v}_i^{\top}
\left(
\sigma_i^2\mathbf{v}_i
\right)\\
&=
\mathbf{v}_i^{\top}\mathbf{v}_i\\
&=1
\end{aligned}
$$

이다. 서로 다른 $i\ne j$에 대해서도

$$
\begin{aligned}
\mathbf{u}_i^{\top}\mathbf{u}_j
&=
\frac{1}{\sigma_i\sigma_j}
\mathbf{v}_i^{\top}
\mathbf{X}^{\top}\mathbf{X}
\mathbf{v}_j\\
&=
\frac{\sigma_j^2}
{\sigma_i\sigma_j}
\mathbf{v}_i^{\top}\mathbf{v}_j\\
&=0
\end{aligned}
$$

이므로 왼쪽 특이벡터도 직교정규다.

모든 양의 특잇값 방향을 모으면

$$
\mathbf{X}\mathbf{V}_r
=
\mathbf{U}_r\boldsymbol{\Sigma}_r
$$

이고 오른쪽에 $\mathbf{V}_r^{\top}$를 곱해

$$
\mathbf{X}
=
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}
$$

를 얻는다. 마지막 등식은 $\mathbf{V}_r$가 $\mathbf{X}$의 행공간을 이루기 때문에 성립한다.

반대편에서도

$$
\mathbf{X}\mathbf{X}^{\top}\mathbf{u}_i
=
\sigma_i^2\mathbf{u}_i
$$

이므로 $\mathbf{u}_i$는 $\mathbf{X}\mathbf{X}^{\top}$의 고유벡터다. 두 대칭행렬은 같은 양의 고윳값 $\sigma_i^2$을 공유한다.

### 3.3 SVD의 기하학: 직교 좌표변환·축별 늘이기·직교 좌표변환

열벡터 $\mathbf{x}\in\mathbb{R}^n$에 full SVD를 적용하면

$$
\mathbf{X}\mathbf{x}
=
\mathbf{U}
\boldsymbol{\Sigma}
\mathbf{V}^{\top}
\mathbf{x}
$$

이다. 오른쪽부터 읽으면 다음 세 단계다.

1. $\mathbf{V}^{\top}$가 입력을 오른쪽 특이벡터 좌표로 바꾼다.
2. $\boldsymbol{\Sigma}$가 $i$번째 좌표를 $\sigma_i$배 한다.
3. $\mathbf{U}$가 결과를 출력 공간의 왼쪽 특이벡터 방향으로 놓는다.

즉 단위구를 $\mathbf{X}$로 변환하면 타원체가 된다. 타원체의 축 방향은 $\mathbf{u}_i$, 반지름은 $\sigma_i$다. $\sigma_i=0$인 입력 방향은 영벡터로 눌려 사라진다.

이 그림은 비영 특이공간에서의 유효 조건수도 설명한다. 가장 긴 축과 가장 짧은 비영 축의 비율은

$$
\kappa_{2,\mathrm{eff}}(\mathbf{X})
=
\frac{\sigma_1}{\sigma_r}
$$

이다. $\sigma_r$가 매우 작으면 그 방향의 작은 관측 오차가 역연산에서 크게 증폭될 수 있다. 랭크 결손이면 영공간 방향을 되살리는 전체 역행렬은 존재하지 않으므로, 위 비율은 비영 특이공간에 한정한 값이다.

### 3.4 바깥곱 합: 행렬을 랭크 $1$ 조각으로 읽기

대각행렬을 이용해 SVD를 펼치면

$$
\begin{aligned}
\mathbf{X}
&=
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}\\
&=
\sum_{i=1}^{r}
\sigma_i
\mathbf{u}_i
\mathbf{v}_i^{\top}
\end{aligned}
$$

이다.

$\mathbf{u}_i\in\mathbb{R}^m$와 $\mathbf{v}_i\in\mathbb{R}^n$의 바깥곱은

$$
\mathbf{u}_i\mathbf{v}_i^{\top}
=
\begin{bmatrix}
u_{i,1}v_{i,1}&\cdots&u_{i,1}v_{i,n}\\
\vdots&\ddots&\vdots\\
u_{i,m}v_{i,1}&\cdots&u_{i,m}v_{i,n}
\end{bmatrix}
$$

이다. $\mathbf{u}_i$와 $\mathbf{v}_i$가 영벡터가 아니면 이 행렬의 모든 열은 $\mathbf{u}_i$의 배수이므로 랭크가 $1$이다.

따라서 SVD는 복잡한 선형변환을 서로 직교하는 랭크-$1$ 채널들의 합으로 분리한다. 특잇값의 내림차순은 각 채널의 크기 순서다.

### 3.5 앞의 $k$개만 남기면 오차가 얼마일까?

$1\le k<r$에 대해 truncated SVD를

$$
\mathbf{X}_k
=
\sum_{i=1}^{k}
\sigma_i
\mathbf{u}_i
\mathbf{v}_i^{\top}
$$

라 하자. 잔차는

$$
\mathbf{R}_k
=
\mathbf{X}-\mathbf{X}_k
=
\sum_{i=k+1}^{r}
\sigma_i
\mathbf{u}_i
\mathbf{v}_i^{\top}
$$

이다.

랭크-$1$ 성분들이 Frobenius 내적에서 서로 직교하므로

$$
\begin{aligned}
\lVert\mathbf{X}-\mathbf{X}_k\rVert_F^2
&=
\sum_{i=k+1}^{r}\sigma_i^2
\end{aligned}
$$

이다. 가장 크게 남은 잔차 방향은 $\sigma_{k+1}$이므로

$$
\lVert\mathbf{X}-\mathbf{X}_k\rVert_2
=
\sigma_{k+1}
$$

이다.

전체 Frobenius 에너지 중 남긴 비율은

$$
\rho_k
=
\frac{
\sum_{i=1}^{k}\sigma_i^2
}{
\sum_{i=1}^{r}\sigma_i^2
}
$$

로 쓸 수 있다. 중심화 데이터의 PCA에서는 이것이 누적 설명분산비와 같다. 일반 가중치 행렬에서는 “에너지 보존율”이라고 부를 수 있지만, 데이터 분산을 설명한다는 PCA 의미를 자동으로 붙이면 안 된다.

### 3.6 왜 truncated SVD가 최적일까? — Frobenius 노름

에카르트–영–미르스키 정리의 Frobenius 노름 부분을 단계적으로 보자. 랭크가 $k$ 이하인 임의의 근사 행렬을

$$
\mathbf{B}\in\mathbb{R}^{m\times n},
\qquad
\operatorname{rank}(\mathbf{B})\le k
$$

라 하자. $\mathbf{B}$의 행공간 위 직교 투영자를 $\mathbf{P}$라 하면

$$
\mathbf{B}
=
\mathbf{B}\mathbf{P}
$$

이고 $\operatorname{rank}(\mathbf{P})\le k$다.

오차를 두 직교 부분으로 나누면

$$
\mathbf{X}-\mathbf{B}
=
\mathbf{X}(\mathbf{I}-\mathbf{P})
+
\left(
\mathbf{X}\mathbf{P}-\mathbf{B}
\right)
$$

이다. 첫 항은 $\mathbf{P}$의 직교여공간 쪽, 둘째 항은 $\mathbf{P}$의 부분공간 쪽에 있으므로 Frobenius 내적이 $0$이다. 피타고라스 정리에 따라

$$
\lVert\mathbf{X}-\mathbf{B}\rVert_F^2
=
\lVert\mathbf{X}(\mathbf{I}-\mathbf{P})\rVert_F^2
+
\lVert\mathbf{X}\mathbf{P}-\mathbf{B}\rVert_F^2
$$

이고 따라서

$$
\lVert\mathbf{X}-\mathbf{B}\rVert_F^2
\ge
\lVert\mathbf{X}(\mathbf{I}-\mathbf{P})\rVert_F^2
$$

다.

이제 $\mathbf{v}_1,\ldots,\mathbf{v}_r$를 입력 공간의 완전한 직교정규기저

$$
\mathbf{v}_1,\ldots,\mathbf{v}_n
$$

으로 확장하고, $i>r$에서는 편의상 $\sigma_i=0$이라 두자. 각 기저 방향에 대해

$$
\alpha_i
=
\lVert\mathbf{P}\mathbf{v}_i\rVert_2^2
$$

라 두자. 직교 투영이므로

$$
0\le\alpha_i\le1,
\qquad
\sum_{i=1}^{n}\alpha_i
=
\operatorname{tr}(\mathbf{P})
=
\operatorname{rank}(\mathbf{P})
\le k
$$

이다. 투영 뒤 보존되는 에너지는

$$
\lVert\mathbf{X}\mathbf{P}\rVert_F^2
=
\sum_{i=1}^{n}\sigma_i^2\alpha_i
=
\sum_{i=1}^{r}\sigma_i^2\alpha_i
$$

다. $\sigma_i^2$가 큰 순서로 정렬되어 있으므로 제한된 총량 $\sum_i\alpha_i\le k$를 앞의 $k$개 방향에 각각 $1$씩 배정할 때 최대다.

$$
\lVert\mathbf{X}\mathbf{P}\rVert_F^2
\le
\sum_{i=1}^{k}\sigma_i^2
$$

따라서 버려지는 에너지는 적어도

$$
\lVert\mathbf{X}(\mathbf{I}-\mathbf{P})\rVert_F^2
\ge
\sum_{i=k+1}^{r}\sigma_i^2
$$

다. 이를 앞의 부등식과 합치면

$$
\boxed{
\lVert\mathbf{X}-\mathbf{B}\rVert_F^2
\ge
\sum_{i=k+1}^{r}\sigma_i^2
}
$$

이다.

한편

$$
\mathbf{P}_k
=
\mathbf{V}_k\mathbf{V}_k^{\top}
$$

를 고르고

$$
\mathbf{B}
=
\mathbf{X}\mathbf{P}_k
=
\mathbf{U}_k
\boldsymbol{\Sigma}_k
\mathbf{V}_k^{\top}
=
\mathbf{X}_k
$$

로 두면 등호가 성립한다. 따라서 $\mathbf{X}_k$가 전역 최적이다. 다만 경계에서 $\sigma_k=\sigma_{k+1}$이면 같은 최솟값을 내는 랭크-$k$ 근사가 여러 개일 수 있으므로 최적해가 유일하다고 단정할 수 없다.

### 3.7 왜 truncated SVD가 최적일까? — Spectral 노름

랭크가 $k$ 이하인 $\mathbf{B}$를 다시 생각하자. $k+1$차원 부분공간

$$
\mathcal{S}
=
\operatorname{span}
\left\{
\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}
\right\}
$$

을 $\mathbf{B}$가 $k$차원 이하로 보낼 수 있으므로, $\mathcal{S}$ 안에는

$$
\mathbf{B}\mathbf{z}
=
\mathbf{0},
\qquad
\lVert\mathbf{z}\rVert_2=1
$$

을 만족하는 벡터가 적어도 하나 존재한다. $\mathbf{z}$를

$$
\mathbf{z}
=
\sum_{i=1}^{k+1}c_i\mathbf{v}_i,
\qquad
\sum_{i=1}^{k+1}c_i^2=1
$$

로 쓰면

$$
\begin{aligned}
\lVert(\mathbf{X}-\mathbf{B})\mathbf{z}\rVert_2^2
&=
\lVert\mathbf{X}\mathbf{z}\rVert_2^2\\
&=
\sum_{i=1}^{k+1}
\sigma_i^2c_i^2\\
&\ge
\sigma_{k+1}^2
\sum_{i=1}^{k+1}c_i^2\\
&=
\sigma_{k+1}^2
\end{aligned}
$$

이다. 그러므로

$$
\lVert\mathbf{X}-\mathbf{B}\rVert_2
\ge
\sigma_{k+1}
$$

다. truncated SVD는 실제로 이 하한을 달성한다.

$$
\boxed{
\min_{\operatorname{rank}(\mathbf{B})\le k}
\lVert\mathbf{X}-\mathbf{B}\rVert_2
=
\sigma_{k+1}
}
$$

### 3.8 SVD와 PCA의 정확한 관계

평균 중심화 데이터

$$
\mathbf{X}_c\in\mathbb{R}^{N\times d}
$$

의 compact SVD를

$$
\mathbf{X}_c
=
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}
$$

라 하자. 공분산 분모를 $N$으로 쓰면

$$
\begin{aligned}
\boldsymbol{\Sigma}_{\text{cov}}
&=
\frac{1}{N}
\mathbf{X}_c^{\top}\mathbf{X}_c\\
&=
\frac{1}{N}
\mathbf{V}_r
\boldsymbol{\Sigma}_r
\mathbf{U}_r^{\top}
\mathbf{U}_r
\boldsymbol{\Sigma}_r
\mathbf{V}_r^{\top}\\
&=
\mathbf{V}_r
\left(
\frac{\boldsymbol{\Sigma}_r^2}{N}
\right)
\mathbf{V}_r^{\top}
\end{aligned}
$$

이다. 따라서 PCA 주성분 방향과 고윳값은

$$
\mathbf{q}_i
=
\mathbf{v}_i,
\qquad
\lambda_i
=
\frac{\sigma_i^2}{N}
$$

이다. 표본 공분산 분모 $N-1$을 쓰면 고윳값만

$$
\lambda_i
=
\frac{\sigma_i^2}{N-1}
$$

로 바뀌고 방향은 같다.

PCA 좌표는

$$
\mathbf{Z}_k
=
\mathbf{X}_c\mathbf{V}_k
=
\mathbf{U}_k\boldsymbol{\Sigma}_k
$$

이며 중심화 공간의 재구성은

$$
\widehat{\mathbf{X}}_c
=
\mathbf{Z}_k\mathbf{V}_k^{\top}
=
\mathbf{U}_k
\boldsymbol{\Sigma}_k
\mathbf{V}_k^{\top}
=
(\mathbf{X}_c)_k
$$

이다. 즉 PCA 재구성과 중심화 데이터의 truncated SVD가 정확히 만난다.

그러나 중심화하지 않은 $\mathbf{X}$를 SVD해 앞 성분을 남기는 것과 PCA는 일반적으로 다르다. 전자는 원점 기준의 큰 에너지 방향을, 후자는 평균 주변의 큰 분산 방향을 본다.

### 3.9 $4\times2$ 예제로 SVD·PCA·저랭크 오차를 계산하자

다음 중심화 데이터를 생각하자.

$$
\mathbf{X}_c
=
\begin{bmatrix}
2&0\\
-2&0\\
0&1\\
0&-1
\end{bmatrix}
$$

각 열 평균은 $0$이다. 먼저

$$
\mathbf{X}_c^{\top}\mathbf{X}_c
=
\begin{bmatrix}
8&0\\
0&2
\end{bmatrix}
$$

이므로 오른쪽 특이벡터와 특잇값은

$$
\mathbf{v}_1
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\sigma_1=2\sqrt{2}
$$

와

$$
\mathbf{v}_2
=
\begin{bmatrix}
0\\
1
\end{bmatrix},
\qquad
\sigma_2=\sqrt{2}
$$

이다. 왼쪽 특이벡터는

$$
\mathbf{u}_1
=
\frac{\mathbf{X}_c\mathbf{v}_1}{2\sqrt{2}}
=
\begin{bmatrix}
\frac{1}{\sqrt{2}}\\
-\frac{1}{\sqrt{2}}\\
0\\
0
\end{bmatrix}
$$

와

$$
\mathbf{u}_2
=
\frac{\mathbf{X}_c\mathbf{v}_2}{\sqrt{2}}
=
\begin{bmatrix}
0\\
0\\
\frac{1}{\sqrt{2}}\\
-\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

이다.

랭크 $1$ 근사는 첫 성분만 남겨

$$
(\mathbf{X}_c)_1
=
\sigma_1\mathbf{u}_1\mathbf{v}_1^{\top}
=
\begin{bmatrix}
2&0\\
-2&0\\
0&0\\
0&0
\end{bmatrix}
$$

이다. 오차는

$$
\lVert
\mathbf{X}_c-(\mathbf{X}_c)_1
\rVert_F^2
=
\sigma_2^2
=2
$$

이고 spectral 오차는

$$
\lVert
\mathbf{X}_c-(\mathbf{X}_c)_1
\rVert_2
=
\sigma_2
=\sqrt{2}
$$

이다.

공분산 분모를 $N=4$로 쓰면

$$
\boldsymbol{\Sigma}_{\text{cov}}
=
\frac{1}{4}
\begin{bmatrix}
8&0\\
0&2
\end{bmatrix}
=
\begin{bmatrix}
2&0\\
0&\frac{1}{2}
\end{bmatrix}
$$

이고 첫 주성분의 설명분산비는

$$
\frac{2}{2+\frac{1}{2}}
=
\frac{4}{5}
=0.8
$$

이다. 같은 결과를 특잇값으로 계산하면

$$
\frac{\sigma_1^2}
{\sigma_1^2+\sigma_2^2}
=
\frac{8}{8+2}
=0.8
$$

이다.

### 3.10 저랭크 분해는 저장량과 계산량을 어떻게 바꿀까?

일반 가중치 행렬

$$
\mathbf{W}\in\mathbb{R}^{m\times n}
$$

은 $mn$개의 숫자를 저장한다. 이를

$$
\mathbf{W}
\approx
\mathbf{A}\mathbf{B},
\qquad
\mathbf{A}\in\mathbb{R}^{m\times k},
\quad
\mathbf{B}\in\mathbb{R}^{k\times n}
$$

으로 나타내면 저장량은

$$
k(m+n)
$$

개다. 압축이 되려면

$$
k(m+n)<mn
$$

이어야 한다.

열벡터 $\mathbf{x}\in\mathbb{R}^{n}$에 $\mathbf{W}$를 곱하는 비용은 대략 $O(mn)$이지만

$$
\mathbf{W}\mathbf{x}
\approx
\mathbf{A}
\left(
\mathbf{B}\mathbf{x}
\right)
$$

은 대략

$$
O(kn+mk)
$$

의 곱셈으로 계산할 수 있다. 실제 속도는 하드웨어, 배치 크기, 메모리 이동, 커널 융합에 따라 달라진다. 파라미터 수 감소가 벽시계 시간의 같은 비율 감소를 자동으로 보장하지는 않는다.

### 3.11 $k$는 어떻게 고를까?

정답 하나는 없다. 다음 기준을 함께 본다.

1. **에너지 기준**

$$
\rho_k
=
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{r}\sigma_i^2}
\ge\tau
$$

가 되도록 목표 비율 $\tau$를 정한다.

2. **오차 기준**

$$
\lVert\mathbf{X}-\mathbf{X}_k\rVert_F
$$

또는

$$
\lVert\mathbf{X}-\mathbf{X}_k\rVert_2
$$

가 허용 한계 아래인지 본다.

3. **꺾이는 지점**

특잇값 그래프에서 급격한 감소가 끝나는 지점을 후보로 삼는다. 사람 눈에 보이는 elbow는 모호할 수 있으므로 검증 지표와 함께 써야 한다.

4. **최종 과업 기준**

압축 전후의 정확도, 손실, 지연, 메모리, 안정성을 직접 측정한다. AI 모델 가중치 압축에서는 이 기준이 가장 중요하다.

5. **수치 오차 기준**

컴퓨터에서 매우 작은 특잇값은 정확한 구조가 아니라 반올림 오차일 수 있다. 절대 임계값 하나보다 행렬 크기와 $\sigma_1$, 자료형 정밀도를 고려한 상대 임계값이 낫다.

### 3.12 SVD의 비유일성과 부호

SVD 성분은 항상 한 가지 숫자 배열로만 정해지는 것이 아니다.

- 한 쌍의 부호를 동시에 바꾸어도

$$
\sigma_i
(-\mathbf{u}_i)
(-\mathbf{v}_i)^{\top}
=
\sigma_i
\mathbf{u}_i
\mathbf{v}_i^{\top}
$$

이므로 같은 행렬이다.
- 같은 특잇값이 반복되면 그 반복 부분공간 안에서 직교기저를 회전해도 같은 행렬을 얻는다.
- 따라서 구현마다 특이벡터 부호나 반복 특잇값의 기저가 달라도 재구성값과 부분공간이 같을 수 있다.

개별 좌표의 부호를 의미로 해석하기 전에 이런 비유일성을 기억해야 한다.

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Attention이 해결하는 핵심 문제

시퀀스의 $i$번째 위치가 좋은 문맥 표현을 만들려면 다른 위치의 정보가 필요하다. 가장 단순한 평균은

$$
\mathbf{o}_i
=
\frac{1}{T}
\sum_{j=1}^{T}\mathbf{v}_j
$$

처럼 모든 위치를 똑같이 취급한다. 그러나 “철수가 사과를 먹었다. 그는 배가 불렀다.”에서 “그”를 이해할 때 모든 단어가 같은 관련성을 갖지는 않는다.

Attention은 고정 평균의 계수를 입력 의존적으로 바꾼다.

$$
\mathbf{o}_i
=
\sum_{j=1}^{T}
a_{ij}\mathbf{v}_j,
\qquad
a_{ij}\ge0,
\qquad
\sum_{j=1}^{T}a_{ij}=1
$$

$a_{ij}$는 위치 $i$가 자신의 새 표현을 만들 때 위치 $j$의 Value를 얼마나 가져올지 정한다. 같은 토큰이라도 주변 문맥이 달라지면 Query–Key 점수가 달라지고 결과 벡터도 달라진다.

### 4.2 1단계: 입력에서 Query·Key·Value 만들기

한 시퀀스의 입력 표현을

$$
\mathbf{H}
=
\begin{bmatrix}
\mathbf{h}_1^{\top}\\
\vdots\\
\mathbf{h}_T^{\top}
\end{bmatrix}
\in
\mathbb{R}^{T\times d_{\text{model}}}
$$

라 하자. 각 행이 한 토큰 위치의 벡터다. 학습 가능한 가중치로

$$
\mathbf{Q}
=
\mathbf{H}\mathbf{W}_Q,
\qquad
\mathbf{K}
=
\mathbf{H}\mathbf{W}_K,
\qquad
\mathbf{V}_{\text{attn}}
=
\mathbf{H}\mathbf{W}_V
$$

를 만든다.

Shape를 모두 쓰면

$$
\underbrace{\mathbf{Q}}_{T\times d_k}
=
\underbrace{\mathbf{H}}_{T\times d_{\text{model}}}
\underbrace{\mathbf{W}_Q}_{d_{\text{model}}\times d_k}
$$

$$
\underbrace{\mathbf{K}}_{T\times d_k}
=
\underbrace{\mathbf{H}}_{T\times d_{\text{model}}}
\underbrace{\mathbf{W}_K}_{d_{\text{model}}\times d_k}
$$

$$
\underbrace{\mathbf{V}_{\text{attn}}}_{T\times d_v}
=
\underbrace{\mathbf{H}}_{T\times d_{\text{model}}}
\underbrace{\mathbf{W}_V}_{d_{\text{model}}\times d_v}
$$

이다.

Query와 Key는 내적해야 하므로 마지막 차원 $d_k$가 같아야 한다. Value는 점수 계산에 직접 들어가지 않으므로 $d_v$가 $d_k$와 꼭 같을 필요는 없다.

### 4.3 2단계: 모든 Query–Key 쌍의 점수 만들기

점수 행렬은

$$
\mathbf{S}
=
\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}
$$

이다. Shape는

$$
\underbrace{\mathbf{S}}_{T\times T}
=
\frac{
\underbrace{\mathbf{Q}}_{T\times d_k}
\underbrace{\mathbf{K}^{\top}}_{d_k\times T}
}{\sqrt{d_k}}
$$

다. $(i,j)$ 원소는

$$
s_{ij}
=
\frac{
\mathbf{q}_i^{\top}\mathbf{k}_j
}{\sqrt{d_k}}
$$

이다. 한 번의 행렬곱이 모든 $T^2$개 토큰 쌍의 내적을 동시에 계산한다.

내적은 두 벡터의 방향 정렬뿐 아니라 길이에도 영향을 받는다. 따라서 scaled dot-product attention 점수를 코사인 유사도와 같은 것으로 부르면 안 된다. 코사인 유사도는 양쪽 길이로 추가로 나누지만, 표준 attention은 그러지 않는다.

### 4.4 $\sqrt{d_k}$ 스케일링을 분산으로 유도하자

단순화된 초기화 상황에서 좌표 $q_{\ell}$과 $k_{\ell}$이 서로 독립이고

$$
\mathbb{E}[q_{\ell}]
=
\mathbb{E}[k_{\ell}]
=0,
\qquad
\operatorname{Var}(q_{\ell})
=
\operatorname{Var}(k_{\ell})
=1
$$

이라고 하자. 곱의 평균은

$$
\mathbb{E}[q_{\ell}k_{\ell}]
=0
$$

이고 독립성 아래

$$
\operatorname{Var}(q_{\ell}k_{\ell})
=
\mathbb{E}[q_{\ell}^2]
\mathbb{E}[k_{\ell}^2]
=1
$$

이다. 서로 다른 좌표의 교차 공분산도 $0$이라고 근사하면

$$
\begin{aligned}
\operatorname{Var}
\left(
\mathbf{q}^{\top}\mathbf{k}
\right)
&=
\operatorname{Var}
\left(
\sum_{\ell=1}^{d_k}
q_{\ell}k_{\ell}
\right)\\
&=
\sum_{\ell=1}^{d_k}
\operatorname{Var}(q_{\ell}k_{\ell})\\
&=
d_k
\end{aligned}
$$

이다. 따라서

$$
\operatorname{Var}
\left(
\frac{
\mathbf{q}^{\top}\mathbf{k}
}{\sqrt{d_k}}
\right)
=
\frac{d_k}{d_k}
=1
$$

이다.

실제로 학습된 좌표들이 완전히 독립이고 분산이 정확히 $1$인 것은 아니다. 이 유도는 스케일링의 설계 동기를 보여 주는 근사 모델이지, 모든 층에서 로짓 분산이 반드시 $1$이라는 정리가 아니다.

### 4.5 왜 Softmax 포화를 조심할까?

한 행의 로짓 $\mathbf{s}_i$에 대한 Softmax는

$$
a_{ij}
=
\frac{
\exp(s_{ij})
}{
\sum_{\ell=1}^{T}
\exp(s_{i\ell})
}
$$

이다. 미분은

$$
\frac{
\partial a_{ij}
}{
\partial s_{i\ell}
}
=
a_{ij}
\left(
\mathbb{1}[j=\ell]-a_{i\ell}
\right)
$$

이다. 어떤 $a_{ij}$가 거의 $1$이고 나머지가 거의 $0$이면 많은 미분값이 매우 작아진다. 차원이 커졌다는 이유만으로 로짓 차이가 커지면 학습 초기에 불필요한 포화가 생길 수 있다.

계산할 때는 오버플로를 피하려고 각 행의 최댓값을 빼도 결과가 같다.

$$
\operatorname{Softmax}(\mathbf{s})
=
\operatorname{Softmax}
\left(
\mathbf{s}
-\max_j s_j
\right)
$$

모든 원소에 같은 상수를 빼면 분자와 분모에 같은 배수가 생겨 약분되기 때문이다.

### 4.6 3단계: 마스크를 Softmax 전에 넣기

허용 여부 행렬을

$$
m_{ij}
=
\begin{cases}
0,&j\text{를 볼 수 있음},\\
-\infty,&j\text{를 볼 수 없음}
\end{cases}
$$

로 둔다. 가중치는

$$
\mathbf{A}
=
\operatorname{Softmax}_{\text{row}}
\left(
\mathbf{S}+\mathbf{M}
\right)
$$

이다. 금지된 위치는

$$
\exp(-\infty)=0
$$

이므로 정확히 $0$의 확률을 갖는다.

자기회귀 언어 모델의 인과 마스크는

$$
m_{ij}
=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i
\end{cases}
$$

다. $i$번째 토큰이 미래 정답 토큰을 미리 보지 못하게 한다.

유한정밀도 구현에서는 실제 $-\infty$ 또는 자료형이 안전하게 표현할 수 있는 매우 작은 값을 사용한다. Softmax를 계산한 뒤 금지 위치를 $0$으로 덮기만 하면 행의 합이 $1$보다 작아지므로, 다시 정규화하지 않는 한 같은 연산이 아니다.

한 행의 모든 위치가 실제 $-\infty$로 마스킹되면 정규화가 정의되지 않아 NaN이 생길 수 있다. 유한한 큰 음수 sentinel을 쓰는 구현에서는 값의 적용 방식에 따라 균등분포 등 의도하지 않은 유한 확률이 나올 수도 있다. 패딩 처리에서는 각 Query 행에 적어도 하나의 유효한 Key가 남도록 설계하고 구현의 마스크 규약을 확인해야 한다.

### 4.7 4단계: Value를 확률 가중합으로 섞기

최종 출력은

$$
\mathbf{O}
=
\mathbf{A}
\mathbf{V}_{\text{attn}}
$$

이다. Shape는

$$
\underbrace{\mathbf{O}}_{T\times d_v}
=
\underbrace{\mathbf{A}}_{T\times T}
\underbrace{\mathbf{V}_{\text{attn}}}_{T\times d_v}
$$

다. $i$번째 행은

$$
\mathbf{o}_i
=
\sum_{j=1}^{T}
a_{ij}
\mathbf{v}^{\text{attn}}_j
$$

이다.

$a_{ij}\ge0$이고 $\sum_j a_{ij}=1$이므로 단일 head의 $\mathbf{o}_i$는 Value 행들의 볼록결합이다. 그러나 뒤의 출력 투영, 잔차 연결, MLP까지 지나면 Transformer 블록 전체 출력이 원래 Value들의 볼록껍질 안에 있어야 하는 것은 아니다.

### 4.8 세 토큰 수치 예제: score에서 출력까지

계산을 투명하게 보기 위해

$$
\mathbf{Q}
=
\mathbf{K}
=
\mathbf{V}_{\text{attn}}
=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}
$$

라고 하자. 실제 모델에서는 세 행렬이 일반적으로 서로 다른 학습 투영에서 나온다. 여기서는 오직 산술을 단순하게 하려고 같게 둔다.

$d_k=2$이므로 스케일 전 내적 행렬은

$$
\mathbf{Q}\mathbf{K}^{\top}
=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
$$

이고 스케일 후에는

$$
\mathbf{S}
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
$$

이다.

인과 마스크를 적용하면

$$
\mathbf{S}+\mathbf{M}
=
\begin{bmatrix}
\frac{1}{\sqrt{2}}&-\infty&-\infty\\
0&\frac{1}{\sqrt{2}}&-\infty\\
\frac{1}{\sqrt{2}}&
\frac{1}{\sqrt{2}}&
\sqrt{2}
\end{bmatrix}
$$

이다.

첫 행은 자기 자신만 볼 수 있으므로

$$
\mathbf{a}_1
=
\begin{bmatrix}
1&0&0
\end{bmatrix}
$$

이다.

둘째 행에서는

$$
\exp(0)=1,
\qquad
\exp\left(\frac{1}{\sqrt{2}}\right)
\approx2.028
$$

이므로

$$
\mathbf{a}_2
\approx
\begin{bmatrix}
0.330&0.670&0
\end{bmatrix}
$$

이다.

셋째 행에서는

$$
\exp\left(\frac{1}{\sqrt{2}}\right)
\approx2.028,
\qquad
\exp(\sqrt{2})
\approx4.113
$$

이므로

$$
\mathbf{a}_3
\approx
\begin{bmatrix}
0.248&0.248&0.503
\end{bmatrix}
$$

이다. 반올림 때문에 합이 $0.999$로 보일 수 있지만 정확한 값의 합은 $1$이다.

따라서

$$
\mathbf{A}
\approx
\begin{bmatrix}
1&0&0\\
0.330&0.670&0\\
0.248&0.248&0.503
\end{bmatrix}
$$

이고 출력은

$$
\begin{aligned}
\mathbf{O}
&=
\mathbf{A}\mathbf{V}_{\text{attn}}\\
&\approx
\begin{bmatrix}
1&0\\
0.330&0.670\\
0.751&0.751
\end{bmatrix}
\end{aligned}
$$

이다.

셋째 출력의 첫 좌표는 첫 Value와 셋째 Value가 기여해

$$
0.248+0.503
=0.751
$$

이고 둘째 좌표도 같은 방식으로 $0.751$이다. 점수는 Key로 정했지만 실제로 더한 내용은 Value라는 구분이 숫자로 드러난다.

### 4.9 Self-attention과 cross-attention

#### Self-attention

같은 입력 $\mathbf{H}$에서

$$
\mathbf{Q}=\mathbf{H}\mathbf{W}_Q,
\qquad
\mathbf{K}=\mathbf{H}\mathbf{W}_K,
\qquad
\mathbf{V}_{\text{attn}}=\mathbf{H}\mathbf{W}_V
$$

를 모두 만든다. 한 시퀀스 내부 관계를 섞는다.

#### Cross-attention

Query는 한 표현 집합에서, Key와 Value는 다른 표현 집합에서 나온다. 예를 들어 텍스트 토큰이 이미지 패치를 참고한다면

$$
\mathbf{Q}
=
\mathbf{H}_{\text{text}}\mathbf{W}_Q,
\qquad
\mathbf{K}
=
\mathbf{H}_{\text{image}}\mathbf{W}_K,
\qquad
\mathbf{V}_{\text{attn}}
=
\mathbf{H}_{\text{image}}\mathbf{W}_V
$$

로 둘 수 있다.

텍스트 길이가 $T_q$, 이미지 토큰 수가 $T_{kv}$라면

$$
\mathbf{Q}\in\mathbb{R}^{T_q\times d_k},
\qquad
\mathbf{K}\in\mathbb{R}^{T_{kv}\times d_k}
$$

이고 점수 행렬은

$$
\mathbf{Q}\mathbf{K}^{\top}
\in
\mathbb{R}^{T_q\times T_{kv}}
$$

이다. attention이 반드시 정사각행렬이어야 하는 것은 아니다. self-attention에서 Query 수와 Key 수가 같아서 $T\times T$가 되었을 뿐이다.

### 4.10 양방향 attention과 인과 attention

양방향 encoder에서는 일반적으로 패딩을 제외한 앞뒤 모든 실제 토큰을 볼 수 있다. $i$번째 표현은 미래 위치의 입력도 참고한다.

인과 decoder에서는 $j>i$를 마스킹한다. 학습 중 전체 정답 시퀀스를 한 번에 넣어도 각 위치가 미래 정답을 훔쳐보지 못하게 한다. 생성 시에는 과거 토큰만 존재하므로 같은 규칙이 자연스럽게 유지된다.

“self-attention”과 “causal attention”은 반대말이 아니다. self-attention은 Q·K·V의 출처를, causal은 연결 허용 방향을 설명한다. 하나의 연산이 causal self-attention일 수 있다.

### 4.11 Multi-head attention은 무엇을 더할까?

한 head만으로도 attention의 핵심은 완성된다. 실제 Transformer는 보통 $H_{\text{head}}$개의 head를 병렬로 계산한다.

$h$번째 head에 대해

$$
\operatorname{head}_h
=
\operatorname{Softmax}_{\text{row}}
\left(
\frac{
\mathbf{Q}_h\mathbf{K}_h^{\top}
}{
\sqrt{d_h}
}
+\mathbf{M}
\right)
\mathbf{V}_h
$$

이고

$$
\operatorname{MHA}(\mathbf{H})
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,
\operatorname{head}_{H_{\text{head}}}
\right)
\mathbf{W}_O
$$

이다.

여러 head는 서로 다른 투영 공간에서 관계를 볼 수 있게 한다. 그러나 각 head가 자동으로 “문법 head”, “인물 head”처럼 사람이 원하는 한 가지 의미만 담당한다고 보장하지 않는다. 중복되거나 분산된 기능도 흔하다.

오늘은 단일 head의 수학을 정확히 세우는 데 집중한다. 위치 표현, 잔차 연결, LayerNorm, MLP, 전체 Transformer 블록은 다음 단계에서 조립한다.

### 4.12 계산량과 메모리는 왜 $T^2$가 될까?

표준 self-attention의 점수 행렬은

$$
\mathbf{S}\in\mathbb{R}^{T\times T}
$$

이다. 모든 Query가 모든 Key를 비교하므로 점수 계산은 대략

$$
O(T^2d_k)
$$

이고, 전체 $T\times T$ 행렬을 메모리에 물질화(materialize)하는 단순 구현에서 가중치 행렬 저장은 head당

$$
O(T^2)
$$

이다. 시퀀스 길이를 두 배로 늘리면 비교할 쌍의 수는 약 네 배가 된다. FlashAttention류의 exact tiled kernel은 타일 단위 재계산과 온라인 정규화로 전체 $T\times T$ 행렬을 저장하지 않아 메모리 복잡도를 낮출 수 있지만, dense attention의 점수 연산량 $O(T^2d_k)$ 자체는 유지된다.

다만 자기회귀 생성에서는 이미 처리한 과거 토큰의 Key와 Value를 KV cache에 저장해 새 토큰마다 과거 전체의 $\mathbf{K}$와 $\mathbf{V}_{\text{attn}}$를 다시 만들지 않는다. 새 Query가 과거 Key들과 점수를 계산하는 비용은 여전히 문맥 길이에 따라 늘고, KV cache 메모리도 레이어·head·토큰 수에 따라 커진다.

### 4.13 SVD와 attention의 1:1 연결

#### 연결 1: 투영 가중치의 저랭크 근사

예를 들어

$$
\mathbf{W}_Q
\approx
\mathbf{U}_k
\boldsymbol{\Sigma}_k
\mathbf{V}_k^{\top}
$$

로 근사하면 두 작은 선형층으로 계산할 수 있다. 특잇값을 양쪽에 나누어

$$
\mathbf{A}
=
\mathbf{U}_k\boldsymbol{\Sigma}_k^{1/2},
\qquad
\mathbf{B}
=
\boldsymbol{\Sigma}_k^{1/2}\mathbf{V}_k^{\top}
$$

라 두면

$$
\mathbf{W}_Q\approx\mathbf{A}\mathbf{B}
$$

이다.

#### 연결 2: 마스크를 더하기 전 raw 점수 행렬의 랭크

$$
\mathbf{L}
=
\mathbf{Q}\mathbf{K}^{\top}
$$

에 대해

$$
\operatorname{rank}(\mathbf{L})
\le
\min
\left(
\operatorname{rank}(\mathbf{Q}),
\operatorname{rank}(\mathbf{K})
\right)
\le d_k
$$

이다. 양의 상수 $\sqrt{d_k}$로 나누어도 이 랭크는 바뀌지 않는다. $d_k<T$이면 **마스크 전 raw 로짓**은 반드시 완전 랭크가 아니다. 이는 attention이 작은 비교 공간을 통해 모든 토큰 쌍의 점수를 만든다는 뜻이다.

하지만 유한한 실수값의 additive mask도 더하는 순간 행렬 랭크를 높일 수 있고, $-\infty$를 포함한 인과 마스크에는 보통의 실수행렬 랭크를 그대로 정의하지 않는다. 실제 가중치는

$$
\mathbf{A}
=
\operatorname{Softmax}_{\text{row}}
\left(
\frac{\mathbf{L}}{\sqrt{d_k}}
+\mathbf{M}
\right)
$$

처럼 마스크와 원소별 지수·행별 나눗셈이라는 비선형 변환을 거친다. 일반적으로

$$
\operatorname{rank}(\mathbf{A})
\le d_k
$$

라고 결론 내릴 수 없다.

#### 연결 3: 출력의 랭크

고정된 한 입력에서

$$
\mathbf{O}
=
\mathbf{A}\mathbf{V}_{\text{attn}}
$$

이므로

$$
\operatorname{rank}(\mathbf{O})
\le
\min
\left(
\operatorname{rank}(\mathbf{A}),
\operatorname{rank}(\mathbf{V}_{\text{attn}})
\right)
\le d_v
$$

이다. 더 구체적으로 고정된 한 입력에서는 $\operatorname{rowspace}(\mathbf{O})\subseteq\operatorname{rowspace}(\mathbf{V}_{\text{attn}})$다. 따라서 Value 차원은 그 입력에서 한 head가 출력할 수 있는 선형 부분공간의 상한을 정한다. 입력이 바뀌면 $\mathbf{V}_{\text{attn}}$도 바뀌므로, 이것이 모든 입력에 공통인 하나의 고정 부분공간을 뜻하지는 않는다.

#### 연결 4: 압축 오차와 과업 오차는 다르다

SVD는

$$
\lVert
\mathbf{W}-\mathbf{W}_k
\rVert_F
$$

를 최소화한다. 하지만 attention에는 로짓, 마스크, Softmax, 여러 층의 비선형성이 이어진다. 작은 가중치 행렬 오차가 특정 희귀 토큰의 순위를 뒤집을 수도 있다.

따라서 압축 전후에는 적어도 다음을 함께 측정해야 한다.

$$
\left(
\text{행렬 근사 오차},
\text{언어·비전 과업 품질},
\text{지연},
\text{메모리}
\right)
$$

### 4.14 수학 부품과 AI 동작의 정확한 매핑

| 수학 부품 | Attention에서 하는 일 | 보장하는 것 | 보장하지 않는 것 |
|---|---|---|---|
| 선형 투영 $\mathbf{H}\mathbf{W}$ | 같은 입력을 Q·K·V 역할별 좌표로 바꾼다. | shape가 맞는 선형 표현 | 사람이 해석 가능한 축 |
| 내적 $\mathbf{q}_i^{\top}\mathbf{k}_j$ | Query–Key 적합도를 계산한다. | 쌍별 실수 점수 | 코사인 유사도·인과성 |
| $\sqrt{d_k}$ 나눗셈 | 차원 증가에 따른 로짓 규모를 완화한다. | 초기화 가정 아래 분산 규모 조절 | 항상 균등한 attention |
| 마스크 $-\infty$ | 허용되지 않는 연결의 확률을 $0$으로 만든다. | 지정한 연결 차단 | 잘못 설계한 규칙 교정 |
| 행별 Softmax | 각 Query마다 Key 가중치를 확률 단체 위로 보낸다. | 비음수·행합 $1$ | 열합 $1$, 설명 충실성 |
| 행렬곱 $\mathbf{A}\mathbf{V}_{\text{attn}}$ | Value를 위치별 가중합한다. | 단일 head에서 볼록결합 | 전체 블록의 볼록결합 |
| SVD | 가중치·활성 행렬의 중요한 직교 성분을 찾는다. | 특정 행렬 노름에서 최적 저랭크 근사 | 최종 과업 품질 보존 |
| 특잇값 꼬리 $\sum_{i>k}\sigma_i^2$ | 버린 Frobenius 에너지를 정량화한다. | 행렬 수준의 정확한 오차 | 어떤 토큰이 실패할지 |

### 4.15 초보자가 흔히 하는 오해와 주의할 점

#### 오해 1: “SVD는 대칭 정사각행렬에만 쓸 수 있다.”

SVD는 모든 실수 직사각행렬에 존재한다. 대칭행렬의 고유분해보다 적용 범위가 넓다.

#### 오해 2: “특잇값은 음수일 수 있다.”

특잇값은 $\mathbf{X}^{\top}\mathbf{X}$의 음이 아닌 고윳값에 제곱근을 취한 값이므로 항상 $\sigma_i\ge0$이다.

#### 오해 3: “저랭크는 희소하다는 뜻이다.”

저랭크 행렬도 모든 원소가 0이 아닌 밀집행렬일 수 있다. 저랭크는 독립 방향이 적다는 뜻이고, 희소는 0인 원소가 많다는 뜻이다.

#### 오해 4: “가장 좋은 랭크-$k$ 근사는 모든 목적에서 가장 좋다.”

Truncated SVD는 Frobenius·spectral 노름에서 최적이다. 분류 정확도, 생성 품질, 공정성, 안전성까지 자동으로 최적인 것은 아니다.

#### 오해 5: “원본 데이터를 그냥 SVD하면 항상 PCA다.”

PCA는 평균 중심화가 먼저다. 중심화하지 않은 SVD는 원점으로부터의 에너지를 본다.

#### 오해 6: “Q, K, V는 세 종류의 토큰이다.”

보통 같은 토큰 표현을 세 가중치로 투영해 만든 세 역할이다. cross-attention에서는 출처가 다를 수 있지만 이름은 역할을 가리킨다.

#### 오해 7: “Key의 Value를 그대로 복사한다.”

Key는 가중치를 정하는 비교용 표현이고, 실제 혼합되는 내용은 대응하는 Value다. 둘은 같은 위치에서 나오더라도 서로 다른 투영이다.

#### 오해 8: “Softmax는 전체 행렬의 합을 $1$로 만든다.”

표준 attention에서는 각 Query 행마다 합이 $1$이다.

$$
\sum_{j=1}^{T}a_{ij}=1
$$

열합이 $1$일 필요는 없다.

#### 오해 9: “$\sqrt{d_k}$ 대신 항상 $\sqrt{d_{\text{model}}}$로 나눈다.”

점수 내적에 실제로 참여하는 head의 Query·Key 차원 $d_k$로 나눈다. 구현의 차원 정의를 확인해야 한다.

#### 오해 10: “인과 마스크는 Softmax 뒤에 곱해도 완전히 같다.”

뒤에서 0으로 만들면 남은 값의 합이 달라진다. 표준 방식은 Softmax 전에 금지 로짓을 $-\infty$로 보내 허용 후보끼리 정규화하는 것이다.

#### 오해 11: “큰 attention 가중치는 그 토큰이 예측의 원인이라는 증명이다.”

가중치는 한 head 한 층의 Value 혼합 계수다. 잔차 경로, 다른 head, MLP, 뒤의 여러 층이 함께 예측을 만든다. 높은 가중치만으로 충실한 설명이나 인과 관계를 단정할 수 없다.

#### 오해 12: “Softmax 전 점수 행렬이 저랭크면 Softmax 뒤도 같은 랭크다.”

Softmax는 비선형이다. 행별 확률 정규화 뒤에는 행렬 랭크가 증가할 수 있다.

#### 오해 13: “KV cache는 과거 텍스트를 그대로 저장한다.”

KV cache는 각 층에서 과거 토큰을 투영해 얻은 Key와 Value 활성값을 저장한다. 위치와 앞선 문맥의 영향을 이미 포함하므로, 다른 순서나 다른 모델 체크포인트에 무조건 붙여 쓸 수 없다.

#### 오해 14: “토큰을 많이 자르면 계산은 줄고 의미는 같은 비율로 줄어든다.”

시각 토큰에는 중복도 있지만 질문마다 중요한 위치가 다르다. 같은 절감률에서도 어떤 토큰을 남기는지가 정확도에 큰 영향을 줄 수 있다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 조사 기준 시각: 2026-09-11 02:49 (Asia/Seoul). 이전 문서에서 다룬 GPT-6 Astra, NeoMME, Gemini 3.8, K2 Horizon, FineWeb-10B, CLAMP 등의 항목은 반복하지 않았다. 오늘은 SVD·저랭크 구조와 attention의 Value 가중합을 직접 확장하며 arXiv v1 제출 타임스탬프가 2026-09-08 UTC인 논문 두 편을 원문에서 확인했고, 두 번째 연구는 공개 코드까지 점검했다. 아래 성능은 논문 저자가 보고한 값이며 독립 재현 결과와 구분해야 한다.

### 5.1 Low-Rank Prompt Learning: VLM 프롬프트의 고정 기저는 학습하지 않아도 될까?

2026-09-08 21:26:37 UTC, 즉 2026-09-09 06:26:37 KST에 arXiv v1이 제출된 **Low-Rank Prompt Learning for Vision-Language Models with Fixed-Token Bases**는 CLIP을 적은 라벨로 적응시키는 연속 프롬프트가 지나치게 많은 자유도를 갖는지 묻는다.

#### 문제: 적은 데이터로 왜 큰 프롬프트 표를 학습할까?

CoOp 방식은 사람이 쓴 “a photo of a …” 같은 템플릿 일부를 $m$개의 학습 가능한 문맥 벡터로 바꾼다. 이를 행으로 쌓은 프롬프트 행렬은

$$
\mathbf{P}
\in
\mathbb{R}^{m\times d}
$$

이다. 표준 설정

$$
m=16,
\qquad
d=512
$$

에서는

$$
md
=
16\cdot512
=
8{,}192
$$

개의 프롬프트 파라미터를 학습한다. 모델 전체에 비하면 작지만, 클래스마다 1개·4개처럼 극소수 예제만 보는 상황에는 여전히 과도한 자유도일 수 있다.

연구진은 프롬프트를

$$
\boxed{
\mathbf{P}
=
\mathbf{B}\mathbf{A}
}
$$

로 제한했다.

$$
\mathbf{B}\in\mathbb{R}^{m\times r},
\qquad
\mathbf{A}\in\mathbb{R}^{r\times d},
\qquad
r\ll\min(m,d)
$$

여기서 $\mathbf{B}$는 $r$개의 잠재 패턴을 $m$개 프롬프트 토큰 위치에 어떻게 섞을지 정하는 **token-side basis**, $\mathbf{A}$는 그 패턴을 $d$차원 CLIP 텍스트 임베딩 공간의 어떤 방향에 놓을지 정하는 **embedding-side coefficient**다.

랭크 $r=4$일 때 두 인자를 모두 학습하면 파라미터 수는

$$
r(m+d)
=
4(16+512)
=
2{,}112
$$

개다. $\mathbf{B}$를 고정하고 $\mathbf{A}$만 학습하면

$$
rd
=
4\cdot512
=
2{,}048
$$

개로 줄어든다. Dense CoOp의 $8{,}192$개와 비교하면 **학습 가능한 프롬프트 파라미터 수**가 정확히 $75\%$ 감소한다. 고정한 $\mathbf{B}$의 저장 공간이나 전체 모델 메모리까지 $75\%$ 줄어든다는 뜻은 아니다.

#### 오늘의 SVD와 같고 다른 점

행렬 곱 $\mathbf{P}=\mathbf{B}\mathbf{A}$는

$$
\operatorname{rank}(\mathbf{P})
\le r
$$

를 강제한다. 오늘 배운 저랭크 표현과 정확히 같은 선형대수다.

그러나 이 연구가 먼저 Dense CoOp 프롬프트를 학습한 뒤 truncated SVD로 Frobenius 오차를 최소화한 것이라고 이해하면 틀린다. 주 실험에서는 저랭크 인자를 분류 손실로 직접 최적화한다.

$$
\min_{\mathbf{A},\mathbf{B}}
\mathcal{L}_{\text{class}}
\left(
\mathbf{B}\mathbf{A}
\right)
$$

따라서 목표는

$$
\min_{\operatorname{rank}(\mathbf{P})\le r}
\lVert
\mathbf{P}_{\text{dense}}-\mathbf{P}
\rVert_F
$$

가 아니다. SVD-derived $\mathbf{B}$는 연구진이 비교한 여러 고정 기저 중 하나일 뿐이며, Gaussian·orthogonal·learned-then-frozen 기저도 함께 평가했다. Gaussian과 orthogonal은 무작위로 구성되며, 뒤의 transfer 대조군에 등장하는 별도 Rand $\mathbf{B}$와 구분해야 한다.

#### 연구진이 보고한 결과

평가는 다음 범위에서 이루어졌다.

- Caltech101, DTD, FGVC Aircraft, Food101, Oxford Flowers, Oxford Pets, UCF101의 7개 데이터셋
- CLIP RN50과 ViT-B/16의 2개 backbone
- 클래스당 1·4·16개 라벨을 쓰는 few-shot 설정
- 본 클래스에 적응한 뒤 새 클래스 성능도 보는 base-to-new 평가

본 클래스 정확도를 $S$, 보지 않은 새 클래스 정확도를 $U$라 하면 조화평균은

$$
H
=
\frac{2SU}{S+U}
$$

이다. 한쪽만 높고 다른 쪽이 낮으면 산술평균보다 더 크게 벌점을 준다.

논문 저자의 rank-$4$ 보고에서 factorized prompt는 ViT-B/16의 21개 데이터셋–shot 조합 모두에서 Dense CoOp보다 $H$가 높았고, RN50에서는 21개 중 17개에서 높았다. ViT-B/16에서 7개 데이터셋 평균 $H$ 개선 폭은 다음과 같다.

| 학습 예제 수 | Dense CoOp 대비 평균 $H$ 변화 |
|---:|---:|
| 1-shot | $+3.58$점 |
| 4-shot | $+4.25$점 |
| 16-shot | $+1.74$점 |

저자는 적은 데이터에서 저랭크 제약이 정규화처럼 작용했을 가능성으로 해석한다. 이는 인과적으로 증명된 유일한 설명이라기보다 결과와 일치하는 해석이다.

더 흥미로운 결과는 $\mathbf{B}$를 학습할 필요가 있었는지다. 논문은 Gaussian, orthogonal, SVD-derived, learned-then-frozen 기저를 비교했고, 각 고정 기저는 대체로 양쪽 인자를 학습한 결과의 수십 분의 몇 점 안에 있었으며 최악의 차이는 약 $1.5$%p였다. $2$개 backbone $\times$ $4$개 rank $\times$ $3$개 shot으로 만든 24개 설정에서 각각 7개 데이터셋 평균 정확도로 다섯 변형을 비교했을 때, 18개 설정에서는 학습하지 않는 기저 변형 가운데 하나가 최고 평균을 기록했다. 이는 개별 데이터셋 24개를 뜻하거나 통계적 유의성 검정을 뜻하지 않는다. 다른 데이터셋에서 학습해 가져온 $\mathbf{B}$가 무작위 $\mathbf{B}$보다 이점이 없었다는 결과도 **ViT-B/16, $r=4$, 16-shot의 지정된 source–target 쌍**에 한정된다.

#### 엔지니어 인사이트 (Impact)

- **저랭크는 압축만이 아니라 가설 공간 제어다.** 적은 데이터에서는 자유도를 줄이는 것이 본 클래스 암기보다 새 클래스 일반화에 유리할 수 있다.
- **두 인자는 대칭적이지 않을 수 있다.** $\mathbf{B}$를 고정하면 큰 $d$차원 embedding-side 계수 $\mathbf{A}$를 계속 조정할 수 있지만, $\mathbf{A}$를 고정하면 훨씬 제한적이다. 단순히 두 행렬의 크기만 보고 어느 쪽을 학습할지 정하면 안 된다.
- **SVD 최적성과 downstream 최적성을 구분해야 한다.** 원본 행렬 재구성에는 truncated SVD가 최적이지만, 분류 손실에는 무작위 고정 기저 위의 학습이 더 나을 수도 있다.
- **저장량 감소와 실행 속도 감소는 별개다.** 이 연구는 태스크별 학습 파라미터를 줄였지만 총 GPU 메모리, 에너지, 추론 지연 감소를 입증하지 않았다.
- **재현 가능성에는 공백이 있다.** 원문은 보존된 구현·설정·결과 기록을 설명하지만 완전한 과거 학습 로그와 factor checkpoint는 복구되지 않았다고 밝힌다. arXiv 항목에도 독립 공개 저장소 링크가 연결되어 있지 않다.
- **적용 범위를 넓혀 말하면 안 된다.** 결과는 두 CLIP backbone의 CoOp식 text prompt와 7개 분류 데이터셋에 관한 것이다. 깊은 multimodal prompt, 생성형 VLM, 다른 modality에서도 같은 factor asymmetry가 성립하는지는 별도 검증이 필요하다.

**원문:** [arXiv 초록·제출 이력](https://arxiv.org/abs/2609.09462) · [수식·실험·재현성 부록](https://arxiv.org/html/2609.09462)

### 5.2 It’s Not RoPE that Creates Sinks: 첫 토큰은 왜 나중 토큰의 시선을 빨아들일까?

2026-09-08 17:32:31 UTC, 즉 2026-09-09 02:32:31 KST에 arXiv v1이 제출되고 EMNLP 2026 채택으로 표시된 **It’s Not RoPE that Creates Sinks**는 LLM 내부의 두 현상을 다룬다.

- **Attention Sink:** 많은 뒤쪽 Query가 특정 위치, 흔히 첫 토큰의 Key에 지나치게 큰 가중치를 주는 현상
- **Massive Activation:** 같은 위치의 중간 hidden state 일부 좌표가 매우 큰 크기를 갖는 현상

큰 activation 이상치는 적은 비트로 값의 범위를 나누는 양자화에서 작은 값들의 해상도를 악화시킬 수 있다. 연구진은 첫 토큰이라는 위치가 가진 세 요소를 분리하려 했다.

1. BOS라는 특별 토큰의 정체
2. 첫 위치의 RoPE 위치 부호화
3. 인과 마스크가 첫 위치에 강제하는 자기집중

#### 오늘 배운 식으로 보는 Value-non-mixing

인과 attention에서 위치 $i$는 $j\le i$만 볼 수 있다.

$$
a_{ij}
=
\frac{
\exp(s_{ij})
}{
\sum_{m=1}^{i}\exp(s_{im})
}
$$

첫 위치 $i=1$에는 후보가 하나뿐이므로 로짓 값과 무관하게

$$
a_{11}
=
\frac{\exp(s_{11})}{\exp(s_{11})}
=1
$$

이다. 따라서 첫 attention 출력은

$$
\mathbf{o}_1
=
\sum_{j=1}^{1}
a_{1j}\mathbf{v}_j
=
\mathbf{v}_1
$$

이다. 여러 위치의 Value를 섞지 않는다. 논문은 이 상태를 **Value-non-mixing**이라고 부른다.

연구진의 가설은 “첫 토큰이라는 문자열이 특별해서”만도, “RoPE가 첫 위치를 만들기 때문에”만도 아니다. 인과 마스크가 만드는 self-concentration과 그 뒤의 Value-non-mixing이 초기 층에서 특수한 신호를 만들고, 이것이 층을 지나며 attention sink와 massive activation 형성에 기여한다는 것이다.

#### 개입 실험은 무엇을 보여 주었나?

연구진은 BOS를 제거한 길이 $64$ 시퀀스에서 16번째 위치가 자기 자신에게만 attention하도록

$$
\alpha_{16,16}=1
$$

을 강제했다. 이 개입은 일반적인 자연 발생 sink를 관찰한 것과 달리, 원인 후보를 시험하기 위해 attention을 인위적으로 바꾼 실험이다.

논문의 $\operatorname{Sink}^{\epsilon}_j$는 대략 다음 두 단계를 거친다.

1. 각 layer·head에서 위치 $j$가 이후 허용 Query들로부터 받은 평균 attention을 계산한다.
2. 그 평균이 임계값 $\epsilon$보다 큰 layer·head의 비율을 구한다.

Llama-3.2-3B과 Qwen2-7B에서 연구진이 보고한 16번째 위치의 sink 지표 변화는 다음과 같다.

| 모델 | 개입 전 | 모든 층에서 self-concentration 강제 |
|---|---:|---:|
| Llama-3.2-3B | $0.0012$ | $0.7424$ |
| Qwen2-7B | $0.0000$ | $0.5085$ |

초기 3개 층에만 같은 개입을 적용해도 다섯 모델 모두에서 전체 층 개입 때의 $\operatorname{Sink}^{0.3}_{16}$ 값 대비 약 $90\%$에서 $93\%$가 나타났다. 반면 중간이나 마지막 3개 층만 개입하면 지표가 거의 기본값에 머물렀다. 이는 현상의 씨앗이 초기 층에 있을 가능성을 지지한다. 이 비율은 massive activation 크기까지 $90$–$93\%$였다는 뜻은 아니다.

연구진은 첫 Key의 RoPE index만 다른 위치 index로 바꾸는 개입도 했다. 다섯 모델에서 이 조작은 첫 위치 sink를 일관되게 크게 낮추지 못했다. 예를 들어 Llama-3.2-3B의 첫 위치 지표는

$$
0.9218
\longrightarrow
0.9133
$$

이었다. 이 실험은 “시험한 Key-side RoPE 조작만으로는 현상이 사라지지 않았다”는 증거다.

하지만 논문 제목을 “RoPE는 아무 영향도 없다”는 보편 정리로 읽어서는 안 된다. 저자도 Query-side 효과와 Query–Key 상호작용을 분리하지 않았고, 다섯 모델이 모두 RoPE 기반이며, 다른 위치 부호화까지 일반화하지 않았다고 제한을 명시한다.

#### 오픈소스 재현 경로

공식 저장소는 다음을 공개한다.

- Llama-2-7B, Llama-3.2-3B, Mistral-7B, Qwen2-7B, Pythia-1B의 고정 revision
- BOS 제거·이동, Key-side RoPE index 변경, self-concentration, 반복 토큰 등 6개 조건
- sink 지표와 activation norm을 저장하는 실행·요약·시각화 명령
- 실행 설정, 데이터·모델 revision, 실제 길이, 무작위 표본, 입력 hash를 기록하는 manifest
- 개입을 끄면 수정 attention 구현이 upstream Transformers 4.57.1과 맞는지 검사하는 테스트

저장소는 원칙적으로 MIT이며, Hugging Face Transformers에서 파생된 attention mirror 코드는 Apache-2.0 조건을 따른다. Python 3.11과 CUDA GPU가 필요하고, 논문은 RTX 6000 Ada에서 주요 실험에 약 4시간이 들었다고 보고한다. 접근 제한(gated-access) Llama 모델은 Hugging Face 승인과 로그인이 필요하다. 저장소는 논문 Figure 2에 사용한 정확한 개별 표본 정보를 보존하지 못했다고 밝힌다. 제공 recipe는 같은 조건의 결정적 첫 WikiText 표본을 쓰지만 그 그림을 표본 단위로 정확히 재현하는 것은 아니다.

#### 엔지니어 인사이트 (Impact)

- **마스크는 정보 접근만 막는 장치가 아니다.** 첫 행의 확률분포를 구조적으로 $[1]$로 고정해 Value 혼합 방식과 내부 activation 분포까지 바꿀 수 있다.
- **문제의 위치를 초기 층부터 확인해야 한다.** 마지막 attention heatmap만 보면 이미 누적된 현상의 결과를 원인으로 오해할 수 있다.
- **BOS 토큰 교체만으로 해결된다고 가정하면 안 된다.** 토큰 정체와 첫 위치의 인과 구조가 분리되어야 한다.
- **양자화와의 연결은 아직 연구 질문이다.** 일반적인 activation outlier가 저비트 양자화를 어렵게 한다는 배경은 알려져 있지만, 논문은 massive activation이 같은 영향을 주는지나 새로운 완화법의 perplexity·정확도·양자화 오차 개선을 직접 측정하지 않았다.
- **모델별 차이를 보존해야 한다.** 반복 토큰을 이용한 Value-non-mixing 증거는 다섯 모델 중 세 모델에서 강했고 나머지 두 모델에서는 약하거나 입력 의존적이었다. 평균 하나로 보편 법칙을 선언하면 안 된다.
- **재현은 코드 실행만이 아니다.** 모델 라이선스·접근권한, revision, 데이터 순서, 개입 없는 기준 구현의 동등성까지 기록해야 결과를 비교할 수 있다.

**논문과 코드:** [arXiv 초록·제출 이력](https://arxiv.org/abs/2609.09085) · [논문 HTML](https://arxiv.org/html/2609.09085) · [공식 GitHub 저장소](https://github.com/kiya-raito/value-non-mixing)

### 5.3 두 연구가 함께 보여 주는 것: 구조적 제약은 작은 수식이지만 큰 편향을 만든다

첫 연구의 랭크 제약은

$$
\operatorname{rank}(\mathbf{P})\le r
$$

로 적응 가능한 프롬프트 공간을 줄인다. 이것은 적은 데이터에서 일반화에 도움이 될 수 있다.

둘째 연구의 인과 마스크는 첫 위치에서

$$
a_{11}=1
$$

을 강제한다. 이것은 정보 누출을 막지만 동시에 Value-non-mixing이라는 특수한 내부 상태를 만든다.

둘을 함께 보면 다음 원칙을 얻는다.

$$
\boxed{
\text{구조적 제약}
\longrightarrow
\text{가능한 표현·경로의 변화}
\longrightarrow
\text{일반화·안정성·효율의 변화}
}
$$

제약은 단순히 계산을 줄이는 스위치가 아니다. 어떤 방향과 연결을 허용하는지 바꾸는 귀납적 편향이다. 따라서 설계자는 “얼마나 줄였는가?”뿐 아니라 “무엇을 더 이상 표현하거나 섞을 수 없는가?”를 함께 측정해야 한다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

다음 하나의 상황으로 SVD 압축과 attention의 관계를 점검하자.

Query 투영에 쓰이는 장난감 가중치가

$$
\mathbf{W}
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
$$

이고, 랭크 $1$로 압축하려 한다. 이 문항은 열벡터 표기를 사용하므로 여기의 $\mathbf{W}$는 본문의 행벡터 표기에서 $\mathbf{W}_Q^{\top}$에 해당한다. 한 입력은

$$
\mathbf{h}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

이다. 이 입력에서 나온 Query가 두 Key

$$
\mathbf{k}_1
=
\begin{bmatrix}
0\\
1
\end{bmatrix},
\qquad
\mathbf{k}_2
=
\begin{bmatrix}
0\\
-1
\end{bmatrix}
$$

와 경쟁하고, 두 Value는

$$
\mathbf{v}_1
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\mathbf{v}_2
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

라고 하자. $d_k=2$이며 마스크는 없다.

**하나의 핵심 질문:** “SVD가 행렬을 거의 잘 보존했다면 attention의 행동도 거의 보존되었다고 말할 수 있는가?”를 다음 계산으로 답하라.

1. $\mathbf{W}$의 SVD와 최적 랭크-$1$ 근사 $\mathbf{W}_1$을 구하라.
2. Frobenius 오차 제곱, spectral 오차, 보존한 에너지 비율을 구하라.
3. 압축 전 Query $\mathbf{q}=\mathbf{W}\mathbf{h}$와 압축 후 Query $\widehat{\mathbf{q}}=\mathbf{W}_1\mathbf{h}$를 구하라.
4. 압축 전후의 scaled dot-product 로짓, Softmax 가중치, attention 출력을 계산하라.
5. 이 결과가 에카르트–영–미르스키 정리와 모순되지 않는 이유를 설명하라.
6. 같은 연산이 인과 attention의 첫 토큰 위치에서 Key 하나만 허용한다면 $a_{11}$과 출력은 무엇인가? 이것이 오늘의 attention sink 연구와 어떻게 연결되는가?
7. 오늘의 VLM 연구처럼 $m=16$, $d=512$, $r=4$인 프롬프트 $\mathbf{P}=\mathbf{B}\mathbf{A}$를 쓴다면 dense, 양쪽 factor 학습, $\mathbf{B}$ 고정 시 학습 파라미터 수를 비교하라.

### 모범 답안

#### 1. SVD와 최적 랭크-$1$ 근사

$\mathbf{W}$는 이미 음수가 아닌 대각행렬이므로 한 SVD는

$$
\mathbf{U}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
\boldsymbol{\Sigma}
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix},
\qquad
\mathbf{V}
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
$$

이다. 특잇값은

$$
\sigma_1=3,
\qquad
\sigma_2=1
$$

이다.

첫 성분만 남긴 truncated SVD는

$$
\mathbf{W}_1
=
\sigma_1
\mathbf{u}_1\mathbf{v}_1^{\top}
=
\begin{bmatrix}
3&0\\
0&0
\end{bmatrix}
$$

이다.

#### 2. 행렬 수준의 오차와 보존 에너지

버린 특잇값이 $1$ 하나이므로

$$
\lVert
\mathbf{W}-\mathbf{W}_1
\rVert_F^2
=
\sigma_2^2
=1
$$

이다. Spectral 오차는

$$
\lVert
\mathbf{W}-\mathbf{W}_1
\rVert_2
=
\sigma_2
=1
$$

이다.

보존한 Frobenius 에너지 비율은

$$
\rho_1
=
\frac{\sigma_1^2}
{\sigma_1^2+\sigma_2^2}
=
\frac{9}{10}
=0.9
$$

이다. 전체 제곱 에너지의 $90\%$를 남겼다.

#### 3. 입력 방향에 따른 Query 변화

압축 전에는

$$
\mathbf{q}
=
\mathbf{W}\mathbf{h}
=
\begin{bmatrix}
3&0\\
0&1
\end{bmatrix}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

이다.

압축 후에는

$$
\widehat{\mathbf{q}}
=
\mathbf{W}_1\mathbf{h}
=
\begin{bmatrix}
3&0\\
0&0
\end{bmatrix}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

이다.

행렬 전체 에너지에서는 작은 둘째 특이 방향이 이 입력에는 유일한 신호였다. 전역적으로 작은 방향이 특정 표본에는 결정적일 수 있다.

#### 4. Attention 행동의 변화

압축 전 scaled logits는

$$
\begin{aligned}
s_1
&=
\frac{
\mathbf{q}^{\top}\mathbf{k}_1
}{\sqrt{2}}
=
\frac{1}{\sqrt{2}},\\
s_2
&=
\frac{
\mathbf{q}^{\top}\mathbf{k}_2
}{\sqrt{2}}
=
-\frac{1}{\sqrt{2}}
\end{aligned}
$$

이다. 따라서

$$
\begin{aligned}
a_1
&=
\frac{
e^{1/\sqrt{2}}
}{
e^{1/\sqrt{2}}+e^{-1/\sqrt{2}}
}
\approx0.804,\\
a_2
&\approx0.196
\end{aligned}
$$

이다.

출력은

$$
\mathbf{o}
=
a_1\mathbf{v}_1
+a_2\mathbf{v}_2
\approx
\begin{bmatrix}
0.804\\
0.196
\end{bmatrix}
$$

이다.

압축 후 Query는 영벡터이므로 두 로짓은 모두 $0$이다.

$$
\widehat{s}_1
=
\widehat{s}_2
=0
$$

Softmax는

$$
\widehat{a}_1
=
\widehat{a}_2
=\frac{1}{2}
$$

이고 출력은

$$
\widehat{\mathbf{o}}
=
\frac{1}{2}\mathbf{v}_1
+\frac{1}{2}\mathbf{v}_2
=
\begin{bmatrix}
0.5\\
0.5
\end{bmatrix}
$$

이다.

행렬 에너지는 $90\%$ 보존했지만 이 입력의 선택 분포와 출력은 크게 달라졌다.

#### 5. 왜 SVD 최적성과 모순되지 않을까?

에카르트–영–미르스키 정리는 $\mathbf{W}_1$이 모든 랭크-$1$ 행렬 중

$$
\lVert
\mathbf{W}-\mathbf{B}
\rVert_F
$$

또는

$$
\lVert
\mathbf{W}-\mathbf{B}
\rVert_2
$$

를 최소화한다고 말한다. 다음은 말하지 않는다.

- 모든 입력 방향의 상대 오차가 작다.
- Query–Key 로짓 순위가 보존된다.
- Softmax 확률이 가깝다.
- 최종 언어·비전 과업의 정확도가 보존된다.

특히 $\mathbf{h}$는 정확히 버린 오른쪽 특이벡터 방향이었다. 전체 원소에 걸친 제곱합 오차와 단위 입력에 대한 최악의 절대 출력 오차를 최소화하는 것, 그리고 특정 표본의 의미적 중요성을 보존하는 것은 서로 다른 기준이다.

#### 6. 인과 attention 첫 위치와 Value-non-mixing

첫 Query가 볼 수 있는 Key가 자기 자신 하나뿐이면

$$
a_{11}
=
\frac{e^{s_{11}}}{e^{s_{11}}}
=1
$$

이다. 출력은

$$
\mathbf{o}_1
=
a_{11}\mathbf{v}_1
=
\mathbf{v}_1
$$

이다.

이는 로짓 크기와 관계없이 발생하는 구조적 self-concentration이며, 여러 Value가 섞이지 않는다. 오늘의 최신 연구는 이런 초기 Value-non-mixing이 초기 층을 거쳐 attention sink와 massive activation 형성에 기여한다는 가설을 개입 실험으로 지지했다. 다만 다섯 모델과 특정 개입에서 얻은 경험적 증거이므로 모든 Transformer에 대한 수학적 필연으로 확대하면 안 된다.

#### 7. 저랭크 VLM 프롬프트의 파라미터 수

Dense prompt는

$$
md
=
16\cdot512
=
8{,}192
$$

개다.

$\mathbf{B}\in\mathbb{R}^{16\times4}$와 $\mathbf{A}\in\mathbb{R}^{4\times512}$를 모두 학습하면

$$
r(m+d)
=
4(16+512)
=
2{,}112
$$

개다.

$\mathbf{B}$를 고정하고 $\mathbf{A}$만 학습하면

$$
rd
=
4\cdot512
=
2{,}048
$$

개다. 마지막 값은 Dense prompt보다

$$
1-\frac{2{,}048}{8{,}192}
=
0.75
$$

즉 $75\%$ 적다. 하지만 이 숫자만으로 총 훈련 메모리나 추론 시간이 $75\%$ 줄었다고 결론 내릴 수는 없다.

---

**다음 연결 고리:** 오늘은 한 head가 문맥을 섞는 핵심 연산을 완성했다. 다음에는 여러 head를 실제 tensor shape로 묶고, 위치 정보·잔차 연결·LayerNorm·MLP를 조립해 하나의 Transformer 블록이 되는 과정을 살펴본다.
