# [2026-09-04] 오늘 학습: 내적·정사영 & 다층 퍼셉트론·비선형 활성화

> **오늘의 핵심 문장:** 내적은 한 벡터가 특정 방향을 얼마나 향하는지 재는 점수이고, 정사영은 그 방향으로 드리운 그림자다. 다층 퍼셉트론은 여러 방향의 점수를 비선형 활성화 함수로 꺾고 조합하여 직선 하나로는 나눌 수 없는 패턴을 표현한다.

어제는 행렬이 벡터를 다른 벡터로 바꾸는 선형변환의 설계도이며, 단층 퍼셉트론의 점수가 $z=\mathbf{w}^{\top}\mathbf{x}+b$라는 사실을 배웠다. 하지만 이 점수가 기하학적으로 무엇을 뜻하는지는 아직 남아 있었다. 오늘은 **내적과 정사영**으로 그 의미를 밝히고, 단층 퍼셉트론이 풀지 못한 XOR을 **다층 퍼셉트론과 비선형 활성화 함수**가 어떻게 푸는지 확인한다.

오늘의 연결은 다음 한 줄이다.

$$
\underbrace{\mathbf{w}^{\top}\mathbf{x}}_{\text{입력과 가중치 방향의 정렬 점수}}
\quad\xrightarrow{\;+b\;}\quad
\underbrace{z}_{\text{결정경계에 대한 부호 있는 증거}}
\quad\xrightarrow{\;\phi\;}\quad
\underbrace{h=\phi(z)}_{\text{필요한 방향 증거만 통과시킨 새 특징}}
$$

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 수학의 문제: “얼마나 같은 방향인가?”를 숫자로 잴 수 없을까?

벡터의 길이만으로는 두 움직임의 관계를 알 수 없다. 동쪽으로 $5$만큼 가는 벡터와 북쪽으로 $5$만큼 가는 벡터는 길이가 같지만 방향은 전혀 다르다. 물체를 어떤 방향으로 미는 힘, 빛이 표면에 비치는 정도, 데이터를 한 축 위에서 바라본 좌표처럼 현실의 많은 문제는 다음 질문을 요구한다.

> 한 벡터 중에서 **특정 방향을 향하는 몫**은 얼마나 되는가?

**내적**은 두 벡터의 성분을 짝지어 곱한 뒤 더함으로써 방향의 정렬 정도를 하나의 스칼라로 압축한다. **정사영**은 그 점수를 이용해 벡터가 특정 직선 위에 떨어뜨리는 수직 그림자를 구한다.

```text
벡터 x
   ●
   │╲
   │ ╲  수직으로 내려간 나머지
   │  ╲
───●───●────────> 방향 u
 원점   x의 정사영
```

이 도구는 나중에 PCA가 “데이터를 어느 방향에 투영해야 퍼짐을 가장 많이 보존하는가?”를 묻는 데 그대로 사용된다.

### 1.2 AI의 문제: 직선 하나로 못 나누는 패턴은 어떻게 배울까?

단층 퍼셉트론은 입력 공간에 다음과 같은 평평한 결정경계 하나를 만든다.

$$
\mathbf{w}^{\top}\mathbf{x}+b=0
$$

가중치 벡터 $\mathbf{w}$는 경계에 수직인 방향이고, 점수의 부호는 입력이 경계의 어느 쪽에 있는지를 알려 준다. AND와 OR처럼 직선 하나로 분리되는 문제에는 충분하지만, 두 입력이 서로 다를 때만 $1$인 XOR의 네 점은 직선 하나로 나눌 수 없다.

```text
x₂
1   ●(1)        ○(0)

0   ○(0)        ●(1)
    0           1        x₁

●: XOR의 양성, ○: XOR의 음성
```

해결의 핵심은 경계를 무작정 여러 번 긋는 것이 아니다. 첫 층이 여러 방향의 증거를 측정하고, **비선형 활성화 함수**가 각 증거를 선택적으로 통과시키며, 다음 층이 그 결과를 다시 조합하게 해야 한다. 이 구조가 **다층 퍼셉트론**이다.

### 1.3 왜 선형 층만 여러 개 쌓아서는 안 될까?

두 층이 활성화 함수 없이 다음처럼 이어진다고 하자.

$$
\mathbf{h}=\mathbf{W}_1\mathbf{x}+\mathbf{b}_1
$$

$$
\mathbf{y}=\mathbf{W}_2\mathbf{h}+\mathbf{b}_2
$$

첫 식을 둘째 식에 넣으면

$$
\begin{aligned}
\mathbf{y}
&=\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2\\
&=(\mathbf{W}_2\mathbf{W}_1)\mathbf{x}
+(\mathbf{W}_2\mathbf{b}_1+\mathbf{b}_2)
\end{aligned}
$$

이다. 새 행렬 $\mathbf{W}=\mathbf{W}_2\mathbf{W}_1$과 새 편향 $\mathbf{b}=\mathbf{W}_2\mathbf{b}_1+\mathbf{b}_2$로 묶으면 결국 아핀 층 하나다. 백 층을 쌓아도 같은 방식으로 하나로 합쳐진다. 층 사이에 선형성이 아닌 함수 $\phi$를 넣어야 합성이 더 이상 한 번의 행렬곱으로 접히지 않는다.

### 1.4 오늘 두 트랙이 만나는 지점

뉴런 하나의 사전활성값을 다시 보자.

$$
z=\mathbf{w}^{\top}\mathbf{x}+b
$$

$\mathbf{w}\ne\mathbf{0}$일 때, $\mathbf{x}$를 $\mathbf{w}$ 방향으로 잰 **스칼라 정사영**은

$$
\operatorname{comp}_{\mathbf{w}}(\mathbf{x})
=\frac{\mathbf{w}^{\top}\mathbf{x}}{\lVert\mathbf{w}\rVert_2}
$$

이다. 따라서

$$
\frac{z}{\lVert\mathbf{w}\rVert_2}
=\frac{\mathbf{w}^{\top}\mathbf{x}+b}{\lVert\mathbf{w}\rVert_2}
$$

는 결정경계까지의 **부호 있는 수직 거리**다. 뉴런의 가중합은 단순한 숫자 섞기가 아니라 “입력이 이 경계의 법선 방향으로 얼마나 와 있는가?”를 재는 계산이다. 활성화 함수는 이 방향 증거를 꺾거나 눌러 다음 층이 쓸 새 특징으로 바꾼다.

## 2. 친절한 용어 사전

### 2.1 수학 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 내적 | $\mathbf{x}^{\top}\mathbf{u}$, $\mathbf{x}\cdot\mathbf{u}$ | 같은 위치의 성분을 곱해 모두 더한 스칼라다. 두 벡터의 길이와 방향 정렬을 함께 반영한다. |
| 노름 | $\lVert\mathbf{x}\rVert_2$ | 벡터 화살표의 유클리드 길이다. $\sqrt{\mathbf{x}^{\top}\mathbf{x}}$로 계산한다. |
| 단위벡터 | $\widehat{\mathbf{u}}$ | 길이가 $1$인 방향표다. $\mathbf{u}\ne\mathbf{0}$이면 $\widehat{\mathbf{u}}=\mathbf{u}/\lVert\mathbf{u}\rVert_2$다. |
| 끼인각 | $\theta$ | 두 벡터 사이의 작은 각으로, 보통 $0\le\theta\le\pi$로 잡는다. |
| 직교 | $\mathbf{x}^{\top}\mathbf{u}=0$ | 두 벡터가 수직이라는 뜻이다. 영벡터는 모든 벡터와 내적이 $0$이지만 방향과 각도는 갖지 않는다. |
| 스칼라 정사영 | $\operatorname{comp}_{\mathbf{u}}(\mathbf{x})$ | $\mathbf{x}$가 $\mathbf{u}$ 방향으로 얼마나 전진했는지를 부호 있는 숫자로 나타낸다. |
| 벡터 정사영 | $\operatorname{proj}_{\mathbf{u}}(\mathbf{x})$ | $\mathbf{x}$가 $\mathbf{u}$가 만드는 직선 위에 드리운 벡터 그림자다. |
| 잔차 | $\mathbf{r}$ | 원래 벡터에서 정사영을 뺀 나머지다. 정사영 방향과 직교한다. |
| 코사인 유사도 | $\cos\theta$ | 벡터 길이를 제거하고 방향의 유사성만 $-1$과 $1$ 사이에서 비교한다. 영벡터에는 정의되지 않는다. |
| 정사영 행렬 | $\mathbf{P}$ | 입력 벡터를 정해진 부분공간 위로 정사영하는 선형변환이다. 같은 벡터에 두 번 적용해도 결과가 더 바뀌지 않는다. |
| 법선벡터 | $\mathbf{w}$ | 직선·평면·초평면에 수직인 벡터다. 결정경계 $\mathbf{w}^{\top}\mathbf{x}+b=0$의 방향을 정한다. |
| 초평면 |  | $d$차원 공간을 한 차원 낮춘 평평한 경계다. 2차원에서는 직선, 3차원에서는 평면이다. |

### 2.2 AI 학습 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 다층 퍼셉트론 | MLP | 행렬을 이용한 아핀 층과 비선형 활성화 함수를 차례로 쌓은 순방향 신경망이다. |
| 은닉층 | $\mathbf{h}$ | 입력과 최종 출력 사이에서 새 특징을 만드는 층이다. 값이 정답으로 직접 주어지지 않아 ‘은닉’이라 부른다. |
| 깊이 |  | 입력에서 출력까지 이어지는 학습 가능한 층의 수를 가리킨다. 세는 관례가 문헌마다 다를 수 있어 구조를 함께 적는 편이 안전하다. |
| 너비 |  | 한 층에 있는 뉴런 수다. |
| 파라미터 | $\mathbf{W},\mathbf{b}$ | 데이터로부터 학습되는 가중치와 편향이다. |
| 사전활성값 | $\mathbf{z}$ | 활성화 함수에 들어가기 전의 아핀변환 결과다. 영어로 `pre-activation`이라 한다. |
| 활성화 함수 | $\phi$ | 사전활성값을 비선형으로 변환해 다음 층으로 보낼 값을 만든다. 보통 벡터의 각 성분에 따로 적용한다. |
| ReLU | $\operatorname{ReLU}(z)=\max(0,z)$ | 음수 증거는 $0$으로 막고 양수 증거는 그대로 통과시키는 함수다. |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | 모든 실수를 $0$과 $1$ 사이로 부드럽게 압축하는 함수다. |
| 포화 |  | Sigmoid처럼 입력의 절댓값이 클 때 출력이 평평해져 입력 변화가 거의 드러나지 않는 상태다. |
| 조각별 선형 |  | 구간마다 선형식이지만 구간을 합친 전체 함수는 하나의 선형식이 아닌 성질이다. ReLU가 대표적이다. |
| 순방향 계산 |  | 입력에서 은닉층을 거쳐 예측까지 값을 차례로 계산하는 과정이다. |
| 표현력 |  | 모델이 얼마나 다양한 입력–출력 관계를 나타낼 수 있는지에 관한 능력이다. 학습이 잘된다는 보장과는 다르다. |

### 2.3 오늘의 트렌드 언어

| 용어 | 초보자 해설 |
|---|---|
| 인코더 | 입력을 읽고 검색·분류에 쓸 벡터 표현으로 바꾸는 모델이다. 반드시 문장을 생성할 필요는 없다. |
| 양방향 Transformer | 각 위치가 앞과 뒤의 입력을 함께 참고해 표현을 만드는 Transformer다. 다음 토큰만 생성하는 인과적 디코더와 목적이 다르다. |
| 이미지 패치 | 이미지를 작은 사각형 조각으로 나눈 것이다. 각 조각을 벡터로 바꾸면 텍스트 토큰과 비슷한 계산 단위로 다룰 수 있다. |
| 임베딩 | 텍스트·이미지 같은 대상을 의미 비교가 가능한 숫자 벡터로 표현한 것이다. |
| 밀집 검색 | 질의와 문서 각각을 대표 벡터 하나로 만들고 유사도를 비교하는 검색 방식이다. |
| 후기 상호작용 | 질의 토큰과 문서의 패치·토큰 벡터를 개별적으로 남긴 뒤 검색 점수를 계산할 때 세밀하게 비교하는 방식이다. |
| 코사인 유사도 | 정규화된 두 임베딩의 내적으로 방향 유사성을 재는 점수다. 오늘의 수학이 실제 검색 시스템에 들어가는 자리다. |
| nDCG@$k$ | 상위 $k$개 검색 결과에서 관련 문서가 얼마나 높은 순위에 배치됐는지를 평가하는 지표다. 관련성이 높은 정답 문서를 위에 놓을수록 높아진다. |
| Visual RAG | PDF의 글자만 추출하지 않고 원래 페이지 이미지를 검색해 VLM이 표·차트·배치까지 보며 답하게 하는 검색 증강 생성 방식이다. |
| 양자화 | 벡터나 가중치의 정밀한 실숫값을 더 적은 비트로 근사해 저장량과 계산량을 줄이는 기술이다. 작게 만들수록 정보가 손실될 수 있다. |
| 근사 최근접 이웃 | 모든 문서를 일일이 비교하지 않고 질의 벡터와 가까운 후보를 빠르게 찾는 인덱스 기법이다. 영어 약자는 `ANN`이다. |
| GRPO | 여러 후보 출력을 한 그룹으로 만들고 상대적인 보상 차이를 이용해 모델의 출력 전략인 정책을 개선하는 강화학습 방법이다. |
| LoRA | 큰 모델의 기본 가중치는 고정하고, 폭이 좁은 두 행렬의 곱으로 만든 저랭크 보정만 학습하는 방법이다. |
| MoE | 여러 전문가 모듈 중 입력마다 일부만 골라 계산하는 혼합 전문가 구조다. 영어로 `Mixture of Experts`라 한다. |
| SiLU | $\operatorname{SiLU}(z)=z\sigma(z)$로, 입력에 Sigmoid 게이트를 곱하는 부드러운 비선형 활성화 함수다. |
| 보상 | 강화학습에서 어떤 결과를 얼마나 선호하는지 나타내는 스칼라 점수다. |
| 롤아웃 | 현재 모델이 한 번 과업을 수행해 출력과 결과를 만들어 낸 실행 기록이다. |
| 보상 대리자 | 사람의 진짜 선호를 직접 잴 수 없을 때 대신 사용하는 모델·규칙 점수다. 편리하지만 진짜 목표와 어긋날 수 있다. |

## 3. 수학의 해부학 (증명과 원리)

### 3.1 내적은 어떻게 계산하는가?

$d$차원 열벡터 두 개를 다음처럼 두자.

$$
\mathbf{x}
=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_d
\end{bmatrix},
\qquad
\mathbf{u}
=
\begin{bmatrix}
u_1\\
u_2\\
\vdots\\
u_d
\end{bmatrix}
$$

내적은

$$
\mathbf{x}^{\top}\mathbf{u}
=\sum_{i=1}^{d}x_i u_i
$$

로 정의한다. 결과는 벡터가 아니라 숫자 하나다. 예를 들어

$$
\mathbf{x}
=
\begin{bmatrix}
3\\
4
\end{bmatrix},
\qquad
\mathbf{u}
=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

이면

$$
\mathbf{x}^{\top}\mathbf{u}
=3\cdot1+4\cdot1
=7
$$

이다. 같은 위치끼리 곱한 값 $[3,4]^{\top}$에서 끝나는 원소별 곱과 달리, 내적은 그 결과를 **더해 스칼라 하나로 압축**한다.

자기 자신과의 내적은 길이의 제곱이다.

$$
\mathbf{x}^{\top}\mathbf{x}
=\sum_{i=1}^{d}x_i^2
=\lVert\mathbf{x}\rVert_2^2
$$

따라서

$$
\lVert\mathbf{x}\rVert_2
=\sqrt{\mathbf{x}^{\top}\mathbf{x}}
$$

이다.

### 3.2 왜 내적이 각도를 알려 주는가?

$\mathbf{x}-\mathbf{u}$의 길이 제곱을 성분식으로 전개하면

$$
\begin{aligned}
\lVert\mathbf{x}-\mathbf{u}\rVert_2^2
&=(\mathbf{x}-\mathbf{u})^{\top}(\mathbf{x}-\mathbf{u})\\
&=\mathbf{x}^{\top}\mathbf{x}
-2\mathbf{x}^{\top}\mathbf{u}
+\mathbf{u}^{\top}\mathbf{u}\\
&=\lVert\mathbf{x}\rVert_2^2
+\lVert\mathbf{u}\rVert_2^2
-2\mathbf{x}^{\top}\mathbf{u}
\end{aligned}
$$

이다. 한편 두 벡터와 그 차이가 만드는 삼각형에 코사인 법칙을 적용하면

$$
\lVert\mathbf{x}-\mathbf{u}\rVert_2^2
=\lVert\mathbf{x}\rVert_2^2
+\lVert\mathbf{u}\rVert_2^2
-2\lVert\mathbf{x}\rVert_2\lVert\mathbf{u}\rVert_2\cos\theta
$$

이다. 두 식의 마지막 항을 비교하면

$$
\boxed{
\mathbf{x}^{\top}\mathbf{u}
=\lVert\mathbf{x}\rVert_2\lVert\mathbf{u}\rVert_2\cos\theta
}
$$

를 얻는다. 두 벡터가 모두 영벡터가 아닐 때

$$
\cos\theta
=\frac{\mathbf{x}^{\top}\mathbf{u}}
{\lVert\mathbf{x}\rVert_2\lVert\mathbf{u}\rVert_2}
$$

이다.

- **두 벡터의 노름을 고정했을 때** $\theta=0$이면 같은 방향이므로 내적은 양수이고 가장 크다.
- $\theta=\pi/2$이면 수직이므로 내적은 $0$이다.
- **두 벡터의 노름을 고정했을 때** $\theta=\pi$이면 반대 방향이므로 내적은 음수이고 가장 작다.

내적은 길이까지 포함한다. 방향만 비교하려면 두 벡터를 길이 $1$로 정규화한 뒤 내적해야 하며, 그것이 코사인 유사도다.

### 3.3 스칼라 정사영과 벡터 정사영을 구분하자

$\mathbf{u}\ne\mathbf{0}$라 하자. $\mathbf{u}$ 방향의 단위벡터는

$$
\widehat{\mathbf{u}}
=\frac{\mathbf{u}}{\lVert\mathbf{u}\rVert_2}
$$

이다. $\mathbf{x}$가 이 방향으로 얼마나 뻗었는지 나타내는 스칼라 정사영은

$$
\operatorname{comp}_{\mathbf{u}}(\mathbf{x})
=\mathbf{x}^{\top}\widehat{\mathbf{u}}
=\frac{\mathbf{x}^{\top}\mathbf{u}}
{\lVert\mathbf{u}\rVert_2}
$$

이다. 이것은 부호를 가진 **길이**다.

그 길이에 방향표를 다시 곱하면 벡터 정사영을 얻는다.

$$
\begin{aligned}
\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
&=\operatorname{comp}_{\mathbf{u}}(\mathbf{x})
\widehat{\mathbf{u}}\\
&=\frac{\mathbf{x}^{\top}\mathbf{u}}
{\lVert\mathbf{u}\rVert_2}
\frac{\mathbf{u}}{\lVert\mathbf{u}\rVert_2}\\
&=\boxed{
\frac{\mathbf{u}^{\top}\mathbf{x}}
{\mathbf{u}^{\top}\mathbf{u}}\mathbf{u}
}
\end{aligned}
$$

$\mathbf{u}$가 이미 단위벡터라면 $\mathbf{u}^{\top}\mathbf{u}=1$이므로

$$
\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
=(\mathbf{u}^{\top}\mathbf{x})\mathbf{u}
$$

로 단순해진다.

앞의 숫자 예제에서는

$$
\mathbf{u}^{\top}\mathbf{u}=1^2+1^2=2
$$

이므로

$$
\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
=\frac{7}{2}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
3.5\\
3.5
\end{bmatrix}
$$

이다. 스칼라 정사영 $7/\sqrt{2}$와 벡터 정사영 $[3.5,3.5]^{\top}$는 종류부터 다르다.

### 3.4 정사영은 왜 가장 가까운 그림자인가?

$\mathbf{u}$가 만드는 직선 위의 모든 점은 $\alpha\mathbf{u}$로 쓸 수 있다. $\mathbf{x}$와 이 점 사이의 거리 제곱을 살펴보자.

$$
\begin{aligned}
\lVert\mathbf{x}-\alpha\mathbf{u}\rVert_2^2
&=(\mathbf{x}-\alpha\mathbf{u})^{\top}
(\mathbf{x}-\alpha\mathbf{u})\\
&=\mathbf{x}^{\top}\mathbf{x}
-2\alpha\mathbf{u}^{\top}\mathbf{x}
+\alpha^2\mathbf{u}^{\top}\mathbf{u}
\end{aligned}
$$

아직 미분을 배우지 않았으므로 제곱을 완성해 최소점을 찾자.

$$
\begin{aligned}
\lVert\mathbf{x}-\alpha\mathbf{u}\rVert_2^2
&=\mathbf{u}^{\top}\mathbf{u}
\left(
\alpha-
\frac{\mathbf{u}^{\top}\mathbf{x}}
{\mathbf{u}^{\top}\mathbf{u}}
\right)^2\\
&\quad+
\mathbf{x}^{\top}\mathbf{x}
-\frac{(\mathbf{u}^{\top}\mathbf{x})^2}
{\mathbf{u}^{\top}\mathbf{u}}
\end{aligned}
$$

첫 항은 음수가 될 수 없고, 정확히 $0$일 때 전체 거리가 가장 작다. 따라서

$$
\alpha^{\star}
=\frac{\mathbf{u}^{\top}\mathbf{x}}
{\mathbf{u}^{\top}\mathbf{u}}
$$

이고 가장 가까운 점은 $\alpha^{\star}\mathbf{u}=\operatorname{proj}_{\mathbf{u}}(\mathbf{x})$다.

잔차를

$$
\mathbf{r}
=\mathbf{x}-\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
$$

라 하면

$$
\begin{aligned}
\mathbf{u}^{\top}\mathbf{r}
&=\mathbf{u}^{\top}\mathbf{x}
-\mathbf{u}^{\top}
\left(
\frac{\mathbf{u}^{\top}\mathbf{x}}
{\mathbf{u}^{\top}\mathbf{u}}\mathbf{u}
\right)\\
&=\mathbf{u}^{\top}\mathbf{x}
-\frac{\mathbf{u}^{\top}\mathbf{x}}
{\mathbf{u}^{\top}\mathbf{u}}
(\mathbf{u}^{\top}\mathbf{u})\\
&=0
\end{aligned}
$$

이다. 즉, 정사영에서 원래 벡터로 향하는 가장 짧은 나머지는 직선과 수직이다. 앞의 예제에서도

$$
\mathbf{r}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}
-
\begin{bmatrix}
3.5\\
3.5
\end{bmatrix}
=
\begin{bmatrix}
-0.5\\
0.5
\end{bmatrix}
$$

이고

$$
\mathbf{u}^{\top}\mathbf{r}
=1\cdot(-0.5)+1\cdot0.5
=0
$$

이다.

정사영과 잔차는 서로 직교하므로 피타고라스 관계도 성립한다.

$$
\lVert\mathbf{x}\rVert_2^2
=
\left\lVert
\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
\right\rVert_2^2
+\lVert\mathbf{r}\rVert_2^2
$$

### 3.5 코시–슈바르츠 부등식은 왜 필요한가?

정사영 잔차의 길이 제곱은 항상 $0$ 이상이다.

$$
0
\le
\left\lVert
\mathbf{x}-\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
\right\rVert_2^2
=\lVert\mathbf{x}\rVert_2^2
-\frac{(\mathbf{u}^{\top}\mathbf{x})^2}
{\lVert\mathbf{u}\rVert_2^2}
$$

양변에 $\lVert\mathbf{u}\rVert_2^2$을 곱하고 정리하면

$$
(\mathbf{u}^{\top}\mathbf{x})^2
\le
\lVert\mathbf{u}\rVert_2^2
\lVert\mathbf{x}\rVert_2^2
$$

이고, 제곱근을 취하면

$$
\boxed{
\left|\mathbf{u}^{\top}\mathbf{x}\right|
\le
\lVert\mathbf{u}\rVert_2
\lVert\mathbf{x}\rVert_2
}
$$

를 얻는다. 이것이 코시–슈바르츠 부등식이다. $\mathbf{u}$와 $\mathbf{x}$ 중 하나가 영벡터인 경우에도 양변이 $0$이므로 성립한다. 두 벡터가 모두 영벡터가 아닐 때 양변을 노름의 곱으로 나누면

$$
-1
\le
\frac{\mathbf{u}^{\top}\mathbf{x}}
{\lVert\mathbf{u}\rVert_2\lVert\mathbf{x}\rVert_2}
\le
1
$$

이 된다. 코사인 유사도가 $[-1,1]$ 범위를 벗어나지 않는 수학적 이유다. 등호는 두 벡터가 **선형 종속**일 때 성립한다. 두 벡터가 모두 영벡터가 아니라면 이는 같은 직선 위에 있다는 뜻이다.

### 3.6 정사영도 행렬로 표현할 수 있다

벡터 정사영 식에서 $\mathbf{x}$를 오른쪽에 모으면

$$
\operatorname{proj}_{\mathbf{u}}(\mathbf{x})
=
\underbrace{
\frac{\mathbf{u}\mathbf{u}^{\top}}
{\mathbf{u}^{\top}\mathbf{u}}
}_{\mathbf{P}}
\mathbf{x}
$$

이다. $\mathbf{P}$를 $\mathbf{u}$가 만드는 직선 위로의 정사영 행렬이라 한다. 이 행렬은 두 가지 중요한 성질을 가진다.

첫째, 전치해도 같다.

$$
\mathbf{P}^{\top}
=\left(
\frac{\mathbf{u}\mathbf{u}^{\top}}
{\mathbf{u}^{\top}\mathbf{u}}
\right)^{\top}
=\frac{\mathbf{u}\mathbf{u}^{\top}}
{\mathbf{u}^{\top}\mathbf{u}}
=\mathbf{P}
$$

둘째, 두 번 정사영해도 한 번 한 것과 같다.

$$
\begin{aligned}
\mathbf{P}^2
&=\frac{\mathbf{u}\mathbf{u}^{\top}}
{\mathbf{u}^{\top}\mathbf{u}}
\frac{\mathbf{u}\mathbf{u}^{\top}}
{\mathbf{u}^{\top}\mathbf{u}}\\
&=\frac{\mathbf{u}(\mathbf{u}^{\top}\mathbf{u})\mathbf{u}^{\top}}
{(\mathbf{u}^{\top}\mathbf{u})^2}\\
&=\mathbf{P}
\end{aligned}
$$

이를 **멱등성**이라 한다. 이미 직선 위에 놓인 그림자를 같은 직선 위로 다시 내려도 움직이지 않는다는 뜻이다.

단위벡터 $\mathbf{q}$에 대해서는 $\mathbf{P}=\mathbf{q}\mathbf{q}^{\top}$이다. PCA에서는 데이터의 평균 $\boldsymbol{\mu}$를 먼저 빼고, 중심화된 데이터가 가장 넓게 퍼지는 단위 방향 $\mathbf{q}$를 찾는다. 한 개의 주성분만 남길 때 좌표와 복원은 각각 다음과 같다.

$$
s
=\mathbf{q}^{\top}(\mathbf{x}-\boldsymbol{\mu})
$$

$$
\widehat{\mathbf{x}}
=\boldsymbol{\mu}
+\mathbf{q}\mathbf{q}^{\top}
(\mathbf{x}-\boldsymbol{\mu})
$$

평균을 빼지 않으면 데이터의 분산이 아니라 원점에서의 위치까지 섞인 방향을 찾을 수 있다. 오늘의 내적과 정사영이 PCA의 핵심 문법인 이유다.

### 3.7 퍼셉트론 점수는 결정경계까지의 부호 있는 거리에 비례한다

$\mathbf{w}\ne\mathbf{0}$이라 하고 결정경계를

$$
\mathcal{H}
=\left\{
\mathbf{x}
\mid
\mathbf{w}^{\top}\mathbf{x}+b=0
\right\}
$$

라 하자. 경계 위의 한 점을 $\mathbf{x}_0$라 하면

$$
\mathbf{w}^{\top}\mathbf{x}_0+b=0
$$

이다. 단위 법선벡터는

$$
\widehat{\mathbf{w}}
=\frac{\mathbf{w}}{\lVert\mathbf{w}\rVert_2}
$$

이고, $\mathbf{x}_0$에서 $\mathbf{x}$로 가는 벡터를 이 법선 방향에 스칼라 정사영하면

$$
\begin{aligned}
d_{\text{signed}}
&=\widehat{\mathbf{w}}^{\top}(\mathbf{x}-\mathbf{x}_0)\\
&=\frac{\mathbf{w}^{\top}\mathbf{x}-\mathbf{w}^{\top}\mathbf{x}_0}
{\lVert\mathbf{w}\rVert_2}\\
&=\boxed{
\frac{\mathbf{w}^{\top}\mathbf{x}+b}
{\lVert\mathbf{w}\rVert_2}
}
\end{aligned}
$$

이다. 마지막 등식에서 $\mathbf{w}^{\top}\mathbf{x}_0=-b$를 사용했다. 절댓값을 취하면 보통의 거리가 된다.

$$
d
=\frac{\left|\mathbf{w}^{\top}\mathbf{x}+b\right|}
{\lVert\mathbf{w}\rVert_2}
$$

따라서 퍼셉트론의 $z=\mathbf{w}^{\top}\mathbf{x}+b$는 거리 자체라기보다 $\lVert\mathbf{w}\rVert_2$만큼 확대된 부호 있는 거리다. 같은 결정경계를 표현하더라도 $\mathbf{w}$와 $b$를 함께 $10$배 하면 $z$도 $10$배 되지만 실제 경계와 거리는 변하지 않는다.

### 3.8 수학 부품과 AI 부품의 일대일 연결

| 수학 부품 | AI에서 맡는 역할 | 주의점 |
|---|---|---|
| $\mathbf{w}^{\top}\mathbf{x}$ | 뉴런이 입력과 가중치 방향의 정렬 증거를 계산 | $\mathbf{w}$가 단위벡터가 아니면 정사영 길이 그 자체는 아니다. |
| $\mathbf{w}^{\top}\mathbf{x}+b$ | 결정경계를 이동시킨 사전활성값 | $b$ 때문에 원래 입력 공간에서는 아핀 함수다. |
| $z/\lVert\mathbf{w}\rVert_2$ | 경계까지의 부호 있는 거리 | $\mathbf{w}\ne\mathbf{0}$이어야 한다. |
| $\cos\theta$ | 정규화된 임베딩의 방향 유사도 | 영벡터에는 정의되지 않고, 학습된 의미의 정확성을 보장하지 않는다. |
| 여러 법선 방향 | 은닉 뉴런들이 서로 다른 패턴을 감지 | 뉴런 수가 곧 유용한 특징 수를 보장하지는 않는다. |
| 정사영 잔차 | 특정 방향으로 설명되지 않은 정보 | 최소제곱법과 PCA의 재구성 오차로 확장된다. |
| $\mathbf{P}=\mathbf{q}\mathbf{q}^{\top}$ | 한 단위 방향의 성분만 남기는 선형변환 | 활성화 함수와 달리 정사영 자체는 선형이다. |
| 비선형 함수 $\phi$ | 방향 증거를 선택·압축해 새 특징 생성 | 어떤 함수를 어느 층에 쓰는지는 목적과 학습 특성에 달려 있다. |

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 다층 퍼셉트론의 전체 구조

은닉층 하나와 출력층 하나를 가진 MLP를 다음처럼 쓸 수 있다.

$$
\mathbf{z}^{(1)}
=\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)}
$$

$$
\mathbf{h}^{(1)}
=\phi\left(\mathbf{z}^{(1)}\right)
$$

$$
\mathbf{z}^{(2)}
=\mathbf{W}^{(2)}\mathbf{h}^{(1)}+\mathbf{b}^{(2)}
$$

$$
\widehat{\mathbf{y}}
=\psi\left(\mathbf{z}^{(2)}\right)
$$

여기서 $\phi$는 은닉층 활성화 함수, $\psi$는 문제에 맞춘 출력 활성화 함수다. 두 함수가 같을 필요는 없다.

```text
입력 x
  │
  ▼
아핀변환 W⁽¹⁾x+b⁽¹⁾ ──> 방향별 점수 z⁽¹⁾
  │
  ▼
비선형 활성화 φ ─────────> 새 특징 h⁽¹⁾
  │
  ▼
아핀변환 W⁽²⁾h⁽¹⁾+b⁽²⁾ ─> 출력 점수 z⁽²⁾
  │
  ▼
출력 함수 ψ ─────────────> 예측 ŷ
```

입력이 $d$차원이고 은닉 뉴런이 $m$개, 출력이 $k$개라면

$$
\mathbf{W}^{(1)}\in\mathbb{R}^{m\times d},
\qquad
\mathbf{b}^{(1)}\in\mathbb{R}^{m}
$$

$$
\mathbf{W}^{(2)}\in\mathbb{R}^{k\times m},
\qquad
\mathbf{b}^{(2)}\in\mathbb{R}^{k}
$$

이다. 가중치 행이 영벡터가 아닌 뉴런에서는 각 행이 입력 공간의 서로 다른 법선 방향을 정한다. 활성화 함수는 각 방향 점수를 새 좌표로 바꾸고, 다음 층은 그 새 좌표 공간에 또 다른 경계를 만든다. 가중치가 영벡터인 뉴런은 입력 방향을 감지하지 않고 편향에만 반응한다.

### 4.2 ReLU는 무엇을 하고 왜 필요한가?

ReLU는 다음과 같다.

$$
\operatorname{ReLU}(z)
=\max(0,z)
=
\begin{cases}
z,&z>0,\\
0,&z\le0
\end{cases}
$$

벡터에는 성분별로 적용한다.

$$
\operatorname{ReLU}
\left(
\begin{bmatrix}
-2\\
0.5\\
3
\end{bmatrix}
\right)
=
\begin{bmatrix}
0\\
0.5\\
3
\end{bmatrix}
$$

ReLU는 $z=0$을 경첩처럼 삼아 음수 쪽을 평평하게 접는다. 구간마다 보면 직선이지만 전체를 하나의 직선으로 쓸 수 없다. 실제로 선형 함수라면 덧셈을 보존해야 하는데

$$
\operatorname{ReLU}(-1+1)=0
$$

인 반면

$$
\operatorname{ReLU}(-1)+\operatorname{ReLU}(1)
=0+1
=1
$$

이므로 덧셈을 보존하지 않는다.

ReLU는 계산이 단순하고 양수 영역에서 값을 강하게 압축하지 않아 은닉층에 널리 쓰인다. 하지만 뉴런의 사전활성값이 계속 음수 영역에 머물면 출력이 늘 $0$이 되어 학습 신호를 받기 어려운 **죽은 ReLU** 문제가 생길 수 있다. 이 문제를 완화하려고 음수에서도 작은 기울기를 남기는 Leaky ReLU 같은 변형을 사용하기도 한다.

### 4.3 Sigmoid는 무엇을 하고 언제 쓰는가?

Sigmoid는 다음과 같다.

$$
\sigma(z)
=\frac{1}{1+e^{-z}}
$$

여기서 $e\approx2.718$은 자연로그의 밑인 수학 상수다.

주요 성질은

$$
0<\sigma(z)<1,
\qquad
\sigma(0)=\frac{1}{2}
$$

이고,

$$
\lim_{z\to\infty}\sigma(z)=1,
\qquad
\lim_{z\to-\infty}\sigma(z)=0
$$

이다. 기호 $\lim_{z\to\infty}$는 $z$가 끝없이 커질 때 함수값이 가까워지는 값을 나타내며, $\lim_{z\to-\infty}$는 $z$가 음의 방향으로 끝없이 작아질 때를 뜻한다. Sigmoid는 큰 양수와 큰 음수를 부드럽게 $1$과 $0$ 근처로 압축하므로 이진 분류의 출력층에서 자주 사용한다. 그러나 Sigmoid 값 하나만 보고 곧바로 “정확한 현실 확률”이라고 단정할 수는 없다. 확률모형에 맞는 손실 함수로 학습하고 별도의 데이터에서 보정 상태를 확인해야 한다.

은닉층 전체에 Sigmoid를 쓸 수도 있지만, $|z|$가 큰 포화 영역에서는 입력이 변해도 출력이 거의 변하지 않는다. 깊은 네트워크에서는 이 평평함이 앞쪽 층의 학습을 느리게 할 수 있어 ReLU 계열이 흔한 기본 선택이 되었다. 이 현상은 편미분과 역전파를 배울 때 수식으로 다시 확인한다.

| 관점 | ReLU | Sigmoid |
|---|---|---|
| 식 | $\max(0,z)$ | $1/(1+e^{-z})$ |
| 출력 범위 | $[0,\infty)$ | $(0,1)$ |
| 주된 사용 위치 | 은닉층 | 이진 분류 출력층 |
| 장점 | 단순하고 양수 영역을 압축하지 않음 | 부드럽고 출력 범위가 확률 표현에 편리함 |
| 대표 주의점 | 음수 영역에서 뉴런이 멈출 수 있음 | 큰 $\lvert z\rvert$에서 포화될 수 있음 |

### 4.4 ReLU 뉴런 두 개로 XOR을 정확히 표현해 보자

입력이 $x_1,x_2\in\{0,1\}$라고 하자. 첫 은닉 뉴런은 “$x_1$이 $x_2$보다 큰 정도”를, 둘째는 반대 방향을 감지하게 만든다.

$$
\mathbf{W}^{(1)}
=
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix},
\qquad
\mathbf{b}^{(1)}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

$$
\mathbf{h}
=\operatorname{ReLU}
\left(
\mathbf{W}^{(1)}\mathbf{x}
\right)
=
\begin{bmatrix}
\operatorname{ReLU}(x_1-x_2)\\
\operatorname{ReLU}(x_2-x_1)
\end{bmatrix}
$$

출력 점수는 두 은닉값을 더한 뒤 출력 편향 $b^{(2)}=-1/2$를 더한다. 마지막 계단 함수 $H$는 점수가 $0$ 이상이면 $1$, 아니면 $0$을 반환한다.

$$
b^{(2)}=-\frac{1}{2},
\qquad
z_{\mathrm{out}}
=
\underbrace{
\begin{bmatrix}
1&1
\end{bmatrix}
}_{\mathbf{W}^{(2)}}
\mathbf{h}
+b^{(2)}
=\operatorname{ReLU}(x_1-x_2)
+\operatorname{ReLU}(x_2-x_1)
-\frac{1}{2}
$$

$$
\widehat{y}=H(z_{\mathrm{out}})
$$

두 ReLU 항의 합은 $|x_1-x_2|$와 같다. 네 입력을 모두 대입해 보자.

| $(x_1,x_2)$ | $(x_1-x_2,\;x_2-x_1)$ | $\mathbf{h}$ | $z_{\mathrm{out}}$ | $\widehat{y}$ |
|---|---|---|---:|---:|
| $(0,0)$ | $(0,0)$ | $(0,0)$ | $-0.5$ | $0$ |
| $(0,1)$ | $(-1,1)$ | $(0,1)$ | $0.5$ | $1$ |
| $(1,0)$ | $(1,-1)$ | $(1,0)$ | $0.5$ | $1$ |
| $(1,1)$ | $(0,0)$ | $(0,0)$ | $-0.5$ | $0$ |

정확히 XOR 진리표가 된다. 첫 행의 가중치 $[1,-1]^{\top}$는 두 입력의 차이가 첫 방향으로 양수인지 내적으로 검사하고, 둘째 행은 반대 방향을 검사한다. ReLU는 각 방향의 음수 증거를 버린다. 출력층은 살아남은 두 증거를 합친다.

중요한 점은 이 가중치를 사람이 직접 정했다는 것이다. 우리는 **표현 가능성**을 증명했을 뿐, 모델이 데이터만 보고 이 값을 자동으로 찾는 학습법은 아직 다루지 않았다. 손실 함수, 경사하강법, 역전파가 바로 그 다음 질문을 해결한다.

### 4.5 한 층의 뉴런은 공간을 어떻게 바꾸는가?

은닉 뉴런 $j$의 계산은

$$
h_j
=\operatorname{ReLU}
\left(
\mathbf{w}_j^{\top}\mathbf{x}+b_j
\right)
$$

이다. 이 식을 세 단계로 읽을 수 있다.

1. $\mathbf{w}_j\ne\mathbf{0}$이면 $\mathbf{w}_j^{\top}\mathbf{x}$가 $\mathbf{w}_j$ 방향의 정렬 증거를 잰다.
2. $b_j$가 증거를 켜는 경계의 위치를 옮긴다.
3. ReLU가 경계의 한쪽에서는 $0$, 다른 쪽에서는 거리에 비례한 값을 낸다.

뉴런 여러 개는 서로 다른 방향과 위치에서 입력 공간을 접는다. 다음 층은 이렇게 만들어진 은닉 좌표 $\mathbf{h}$에서 다시 내적을 계산한다. 결과적으로 ReLU MLP의 경계는 여러 평평한 조각이 이어진 모양이 될 수 있다. 각 조각은 선형적이어도 전체는 하나의 직선이나 평면이 아니다.

### 4.6 배치 계산에서도 구조는 같다

샘플 $N$개를 행으로 쌓은

$$
\mathbf{X}\in\mathbb{R}^{N\times d}
$$

와 첫 층의 가중치

$$
\mathbf{W}^{(1)}\in\mathbb{R}^{m\times d}
$$

가 있으면 은닉층 전체는

$$
\mathbf{H}
=\phi
\left(
\mathbf{X}(\mathbf{W}^{(1)})^{\top}
+\mathbf{1}_N(\mathbf{b}^{(1)})^{\top}
\right)
\in\mathbb{R}^{N\times m}
$$

로 계산한다. $\mathbf{1}_N\in\mathbb{R}^{N}$은 모든 성분이 $1$인 열벡터다. 곱 $\mathbf{1}_N(\mathbf{b}^{(1)})^{\top}$은 같은 편향 행벡터를 $N$개 샘플 각각에 반복해 더한다. 따라서 $\mathbf{H}$의 $i$번째 행은 $i$번째 샘플의 새 특징이고, $j$번째 열은 $j$번째 은닉 뉴런이 모든 샘플에 낸 응답이다. 실제 딥러닝은 이런 큰 내적 묶음을 행렬곱으로 병렬 처리한다.

### 4.7 초보자가 흔히 하는 오해와 주의할 점

#### 오해 1: “내적은 같은 위치끼리 곱한 벡터다.”

같은 위치끼리 곱한 뒤 반드시 모두 더해 스칼라 하나를 만드는 연산이 내적이다. 원소별 곱과 구분해야 한다.

#### 오해 2: “내적이 곧 정사영 벡터다.”

$\mathbf{x}^{\top}\mathbf{u}$는 스칼라다. $\mathbf{u}$가 단위벡터일 때도 정사영 **벡터**를 얻으려면 $(\mathbf{x}^{\top}\mathbf{u})\mathbf{u}$처럼 방향벡터를 다시 곱해야 한다.

#### 오해 3: “내적이 크면 항상 방향이 더 비슷하다.”

내적은 길이의 영향도 받는다. 방향만 비교하려면 두 노름으로 나눈 코사인 유사도를 사용해야 한다.

#### 오해 4: “코사인 유사도는 어떤 벡터에도 계산할 수 있다.”

영벡터는 노름이 $0$이라 분모가 $0$이 되므로 정의되지 않는다. 구현에서는 작은 수를 더하더라도 그것은 수치적 예외 처리이지 영벡터에 의미 있는 방향이 생긴 것이 아니다.

#### 오해 5: “층이 여러 개면 활성화 함수가 없어도 딥러닝이다.”

아핀 층만 합성하면 하나의 아핀 층으로 접힌다. 층 사이의 비선형성이 표현력을 실제로 늘린다.

#### 오해 6: “ReLU 네트워크는 조각별 선형이므로 전체도 선형이다.”

입력 구간별 식은 선형일 수 있지만 어느 뉴런이 켜지는지가 구간마다 달라진다. 모든 입력에 통하는 하나의 선형식은 아니다.

#### 오해 7: “ReLU 출력은 $0$ 이상이므로 확률이다.”

ReLU 출력에는 상한이 없고 합이 $1$일 필요도 없다. 은닉 특징의 세기이지 확률이 아니다.

#### 오해 8: “Sigmoid를 쓰면 자동으로 잘 보정된 확률이 된다.”

출력 범위만 $(0,1)$일 뿐이다. 손실 함수, 데이터 분포, 학습 상태와 사후 보정까지 점검해야 확률 해석을 신뢰할 수 있다.

#### 오해 9: “XOR 예제의 가중치를 찾았으니 학습까지 해결했다.”

가능한 파라미터가 존재한다는 것과 학습 알고리즘이 그 값을 찾는다는 것은 별개의 문제다. 다음 단계에서 손실과 최적화를 배운다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 조사 기준 시각: 2026-09-04 02:31 (Asia/Seoul). 2026-09-03 공개 글 두 편과, 이에 연결된 2026-08-31 NeoMME 사전논문·모델 카드·저장소를 서로 대조했다. 아래 수치는 별도 표시가 없는 한 공개 주체의 보고이며, 독립 재현이나 동료 심사를 통과한 결론과 구분해 읽어야 한다.

### 5.1 NeoMME: 생성용 거대 VLM 대신 검색 전용 멀티모달 인코더

H Company 연구진은 2026-08-31 arXiv 사전논문 v1을 제출하고, 2026-09-03 팀 글과 공개 모델을 통해 **NeoMME**를 소개했다. NeoMME는 $260$M과 $800$M 파라미터의 다국어·멀티모달 인코더 계열이다. 흔한 시각 문서 검색기는 사전학습된 비전 타워로 이미지를 읽고 이를 인과적 언어모델에 연결한다. 반면 NeoMME는 사전학습 비전 타워나 인과적 디코더 없이 **하나의 양방향 Transformer**가 텍스트 토큰과 원시 이미지 패치를 함께 처리한다. 다만 입력 임베딩과 투영을 묶어 재사용하는 마스크 토큰 출력 경로는 있으며, 출력 전용 파라미터는 추가되지 않는다.

이미지는 겹치지 않는 $32\times32$픽셀 패치로 나뉘고 작은 MLP를 거쳐 Transformer의 공통 표현 공간으로 들어간다. 컨텍스트 길이는 $16{,}384$토큰이며, 블록의 MLP에는 오늘 배운 ReLU를 제곱한 **squared-ReLU** 계열 활성화가 사용된다. 텍스트를 일부 가리고 복원하는 이산 마스킹 확산 목표로 처음부터 학습했으며, 모델 가중치는 Apache 2.0으로 공개됐다.

검색용 NeoMME-Retriever는 한 번의 순방향 계산에서 다음 두 표현을 함께 만든다.

1. **밀집 표현:** 질의와 문서 전체를 각각 정규화된 벡터 하나로 압축한다.
2. **후기 상호작용 표현:** 질의 토큰과 문서의 토큰·이미지 패치마다 정규화된 $128$차원 벡터를 남긴다.

정규화된 밀집 임베딩 $\mathbf{e}_q,\mathbf{e}_d$의 검색 점수는

$$
s_{\mathrm{dense}}(q,d)
=\mathbf{e}_q^{\top}\mathbf{e}_d
=\cos\theta
$$

이다. 노름이 각각 $1$이므로 내적과 코사인 유사도가 같다. 후기 상호작용의 MeanMaxSim 점수는, 질의 벡터가 $L_q$개이고 문서 벡터가 $L_d$개일 때 다음처럼 읽을 수 있다.

$$
s_{\mathrm{late}}(q,d)
=\frac{1}{L_q}
\sum_{i=1}^{L_q}
\max_{1\le j\le L_d}
\mathbf{q}_i^{\top}\mathbf{d}_j
$$

각 질의 토큰은 자신과 가장 잘 맞는 문서 토큰이나 이미지 영역을 내적으로 고르고, 그 최댓값들을 평균낸다. 예를 들어 “매출 증가”라는 질의의 각 토큰이 표의 숫자 영역과 그래프 범례처럼 서로 다른 패치에 대응할 수 있다. 오늘 배운 내적이 최신 Visual RAG 검색기의 실제 점수 함수로 곧바로 등장한 것이다.

저자 보고 기준으로 NeoMME-Retriever-260M은 ViDoRe v3에서 nDCG@$10$이 $0.523$, 800M은 $0.556$을 기록했다. $260$M 모델은 NVIDIA L40S 한 장과 $2048\times2048$ 입력 조건에서 모델 측 문서 인코딩 처리량의 중앙값이 초당 $51.3$페이지로, ColModernVBERT의 초당 $26.0$페이지보다 $1.97$배 높았다. 이 측정은 이미지 디코딩·전처리, 저장장치 입출력, 압축과 인덱스 구축을 제외하고 모델별 배치 크기를 따로 조정한 결과다.

계층적 토큰 풀링 계수 $8$, 질의 int8, 문서 이진 양자화를 조합한 한 설정에서는 후기 상호작용 저장량을 페이지당 $1536.7\pm201.1$kB에서 $6.0$kB로 $255.5$배 줄이면서 기준 nDCG@$10$ 품질의 $95.19\%$를 유지했다고 보고한다.

**엔지니어 인사이트 (Impact)**

- 검색·분류처럼 텍스트 생성이 필요 없는 작업에 생성용 VLM 전체를 쓰는 것이 언제나 효율적이지는 않다. **업무 목적에 맞춘 작은 인코더**가 인덱싱 비용과 지연을 크게 줄일 수 있다.
- PDF를 페이지 이미지로 검색하면 OCR이 평평하게 만들어 버릴 수 있는 표, 차트, 배치 정보를 보존한다. Visual RAG의 병목이 생성 모델뿐 아니라 **좋은 검색 임베딩과 인덱스 설계**에도 있음을 보여 준다.
- 모델 문서에서 말하는 패치의 “projection”과 $128$차원으로의 “projection”은 학습된 선형 사상이다. 오늘 유도한 가장 가까운 점을 구하는 **직교 정사영**이나 PCA 투영이라는 뜻은 아니다. 같은 단어라도 조건을 확인해야 한다.
- $255.5$배 압축은 공짜가 아니다. 기준 대비 $4.81\%$의 **상대 품질 손실**, nDCG@$10$으로는 약 $0.0251$의 절대 감소가 있는 특정 설정이다. 저장량 계산도 질의 벡터, ANN 라우팅과 파일시스템 부가 비용을 포함하지 않으며, 최적화된 이진 검색 지연은 측정되지 않았다. 한국어 문서, 사내 레이아웃, 목표 하드웨어에서 정확도·처리량·저장량을 함께 다시 평가해야 한다.
- 사전논문 v1이고 일부 비교 수치는 저자 자체 평가다. 대화형·인과적 생성 VLM은 아니지만 반복적인 마스크 복원으로 텍스트를 만들 수는 있다. 저자들은 이 과정이 유해하거나 사적이거나 편향된 텍스트를 만들 가능성을 경고하며, 공개 자료에는 이를 다룬 종합 정량 평가가 없다.

**직접 읽고 실행하기:** [H Company 팀 글](https://huggingface.co/blog/Hcompany/neomme) · [arXiv 사전논문](https://arxiv.org/abs/2609.01657) · [Apache 2.0 체크포인트 컬렉션](https://huggingface.co/collections/Hcompany/neomme) · [260M Retriever 모델 카드](https://huggingface.co/Hcompany/NeoMME-260M-Retriever)

### 5.2 TRL·OpenEnv 수채화 GRPO: 정답이 없는 ‘취향’을 보상으로 만들기

Hugging Face에는 같은 날 **코딩 모델을 수채화 화가로 학습하고 주요 산출물을 모두 공개한 재현 프로젝트**도 올라왔다. Qwen3.5-35B-A3B가 `p5.brush` JavaScript 코드를 쓰고, 헤드리스 브라우저가 코드를 그림으로 렌더링하며, 시각 평가기가 결과에 보상을 준다. 학습에는 TRL의 GRPO와 LoRA를 사용했다.

대표 `judge-led` 설정에는 먼저 통과 여부 $G\in\{0,1\}$를 정하는 **하드 게이트**가 있다. 게이트를 통과하지 못하면 다른 항을 계산하지 않고 전체 보상이 $0$이 된다. 통과한 경우의 상수 게이트 보상까지 포함하면 구현은 다음처럼 쓸 수 있다.

$$
R
=G\left(
0.05
+0.05R_{\mathrm{length}}
+0.60R_{\mathrm{judge}}
+0.30R_{\mathrm{HPSv3}}
\right)
$$

- $G$는 코드가 실행되고 실제로 그림을 그리며 평가기를 속이지 않는지 검사한 결과다.
- $R_{\mathrm{length}}$는 지나치게 짧은 코드를 완만하게 억제한다.
- $R_{\mathrm{judge}}$는 VLM이 후보 그림을 선별된 참조 그림과 쌍대 비교한 승률이다.
- $R_{\mathrm{HPSv3}}$는 사람들의 이미지 선호로 학습된 공개 선호 모델의 점수다.

재현자는 한 사람이 고른 히비스커스 그림 $178$개, 환경, 학습 스크립트, 세 LoRA 어댑터, 모든 롤아웃과 곡선을 공개했다. 저자 실험에서 첫 구간과 마지막 구간의 평균 그룹 보상은 `judge-led`가 $0.45$에서 $0.72$로, `hps-led`가 $0.57$에서 $0.82$로 올랐다. 다만 이 수치는 일반화 성능을 재는 표준 벤치마크가 아니라 해당 보상 체계 안에서의 개선이다.

오늘 주제와 맞닿는 흥미로운 디버깅도 있다. 보통의 LoRA 대상 목록은 밀집 모델의 층 이름을 가정했지만, 사용한 모델은 MoE라서 처음에는 $40$개 층 중 $10$개에만 어댑터가 붙었다. 재현자는 학습률, 스케줄러, 보상 스케일링과 LoRA 대상이라는 네 설정을 함께 고쳐 학습을 진행시켰다. 그중 대상을 `all-linear`로 바꾼 수정은 발견 가능한 거의 모든 선형 모듈에 학습 가능한 저랭크 행렬 $\mathbf{A},\mathbf{B}$를 덧붙였다. `all-linear`는 네트워크 전체를 선형 함수로 만들거나 기존 기본 가중치를 직접 갱신한다는 뜻이 아니다. 이 모델의 SiLU 같은 비선형 활성화 전후에서 **저랭크 어댑터가 붙은 선형 모듈**을 폭넓게 조정하며, 기본 가중치와 융합 텐서로 구현된 라우팅 전문가 가중치는 고정된다.

**엔지니어 인사이트 (Impact)**

- 수학 정답이나 테스트 통과처럼 검증 가능한 목표가 없어도, 환경과 비교 데이터로 취향을 스칼라 보상으로 바꿀 수 있다. 오픈소스 RL 생태계가 “정답 맞히기”에서 디자인·창작 작업으로 넓어지는 사례다.
- 모델은 프롬프트보다 실제 보상을 최적화한다. 출력에 $15$~$30$개 도형을 쓰라는 지시는 보상과 거의 상관이 없었고, 실제 평균은 약 $7$~$9$개에 머물렀다. 중요한 요구사항은 문장으로만 적지 말고 평가 함수와 데이터에 반영해야 한다.
- 보상은 진짜 취향이 아니라 대리자다. 참조 풀과 평가 모델의 편향이 그대로 학습 목표가 되며, 좁은 참조 풀에서는 출력 다양성이 줄 수 있다. **무엇을 보상했는가**를 감사할 수 있도록 롤아웃과 구성 요소를 함께 보존해야 한다.
- 이 결과는 한 사람의 선호로 고른, 모델 생성 히비스커스 $178$장에 집중한 단일 재현 실험이다. 인간이 그린 작품은 없고 모델은 자기 렌더를 다시 보지 못하는 단일 턴 구조다. 실제 세 실행은 H200 한 장에서 각각 $17$시간 $46$분, $32$시간 $15$분, $34$시간 $49$분 걸렸다. 실행 중에는 7B HPSv3용 `a100-large` Space, Chromium 렌더링용 `cpu-upgrade` Space가 필요하고, VLM 판정 실행에는 Qwen3-VL-30B-A3B-Instruct용 Hugging Face Inference Providers 쿼터도 든다. 저장소의 의존성 버전도 고정되지 않아 그대로 다시 실행하는 데 재현성 공백이 있다.

**직접 재현하기:** [Hugging Face 공개 글](https://huggingface.co/blog/train-to-paint-with-code) · [GitHub 코드와 결과](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour) · [하드 게이트 보상 구현](https://github.com/adithya-s-k/HuggingEnvs/blob/main/02-watercolour/envs/watercolour/core/scoring.py) · [Qwen3.5 공식 설정](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json) · [모델·데이터·롤아웃 컬렉션](https://huggingface.co/collections/HuggingEnvs/paint-with-code)

### 5.3 두 소식에서 읽어야 할 하나의 흐름

두 프로젝트는 거대한 범용 모델 하나만 고르는 시대에서 **목적에 맞는 계산과 평가를 설계하는 시대**로 이동하고 있음을 보여 준다.

- NeoMME는 문서를 생성하지 않고 잘 찾는 데 필요한 벡터와 내적에 집중한다.
- 수채화 GRPO는 코드 생성 능력 자체보다 어떤 결과를 선호할지 정하는 보상 환경에 집중한다.

행렬과 비선형 활성화가 모델의 표현을 만들지만, 실제 제품 성능은 어떤 표현을 검색 점수로 읽고 어떤 행동을 보상하는지까지 포함해 결정된다. 기초 수학, 모델 구조, 평가 설계를 따로 보지 말아야 하는 이유다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

어제 사용한 퍼셉트론의 값이 다음과 같다고 하자.

$$
\mathbf{w}
=
\begin{bmatrix}
2\\
-1
\end{bmatrix},
\qquad
b=-1,
\qquad
\mathbf{x}
=
\begin{bmatrix}
2\\
3
\end{bmatrix}
$$

다음 내용을 하나의 이야기로 설명해 보자.

1. $\mathbf{w}^{\top}\mathbf{x}$, $\lVert\mathbf{w}\rVert_2$, $\mathbf{w}$ 방향의 스칼라 정사영과 벡터 정사영을 구하라.
2. 결정경계 $\mathbf{w}^{\top}\mathbf{x}+b=0$까지의 부호 있는 거리를 구하라. 벡터 정사영이 $\mathbf{0}$이 아닌데도 이 거리가 $0$일 수 있는 이유는 무엇인가?
3. 오늘의 XOR MLP에 $(x_1,x_2)=(1,0)$을 넣어 은닉값, 출력 점수, 예측을 계산하라.
4. 그 MLP에서 ReLU를 항등함수 $\phi(z)=z$로 바꾸면 왜 여러 층을 유지해도 XOR을 표현할 수 없는가?

### 모범 답안

먼저 내적과 노름은

$$
\mathbf{w}^{\top}\mathbf{x}
=2\cdot2+(-1)\cdot3
=1
$$

$$
\lVert\mathbf{w}\rVert_2
=\sqrt{2^2+(-1)^2}
=\sqrt{5}
$$

이다. 따라서 $\mathbf{w}$ 방향의 스칼라 정사영은

$$
\operatorname{comp}_{\mathbf{w}}(\mathbf{x})
=\frac{\mathbf{w}^{\top}\mathbf{x}}
{\lVert\mathbf{w}\rVert_2}
=\frac{1}{\sqrt{5}}
$$

이고, 벡터 정사영은

$$
\begin{aligned}
\operatorname{proj}_{\mathbf{w}}(\mathbf{x})
&=\frac{\mathbf{w}^{\top}\mathbf{x}}
{\mathbf{w}^{\top}\mathbf{w}}\mathbf{w}\\
&=\frac{1}{5}
\begin{bmatrix}
2\\
-1
\end{bmatrix}\\
&=
\begin{bmatrix}
2/5\\
-1/5
\end{bmatrix}
\end{aligned}
$$

이다.

결정경계까지의 부호 있는 거리는

$$
\begin{aligned}
d_{\mathrm{signed}}
&=\frac{\mathbf{w}^{\top}\mathbf{x}+b}
{\lVert\mathbf{w}\rVert_2}\\
&=\frac{1-1}{\sqrt{5}}\\
&=0
\end{aligned}
$$

이다. $\operatorname{proj}_{\mathbf{w}}(\mathbf{x})$는 **원점을 지나는 $\mathbf{w}$ 방향 직선** 위에 내린 그림자다. 반면 거리 $0$은 $b=-1$만큼 이동한 **아핀 결정경계** 위에 $\mathbf{x}$가 놓였다는 뜻이다. 서로 다른 대상에 대한 정사영과 거리이므로 모순이 아니다.

XOR MLP에 $(1,0)$을 넣으면

$$
\mathbf{h}
=
\begin{bmatrix}
\operatorname{ReLU}(1-0)\\
\operatorname{ReLU}(0-1)
\end{bmatrix}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

이다. 출력 점수와 예측은

$$
z_{\mathrm{out}}
=1+0-\frac{1}{2}
=\frac{1}{2}
$$

$$
\widehat{y}=H(1/2)=1
$$

이므로 XOR의 정답과 같다.

마지막으로 **이 예제의 고정된 가중치**에서 ReLU를 항등함수로 바꾸면

$$
\begin{aligned}
\mathbf{W}^{(2)}\mathbf{W}^{(1)}
&=
\begin{bmatrix}
1&1
\end{bmatrix}
\begin{bmatrix}
1&-1\\
-1&1
\end{bmatrix}\\
&=
\begin{bmatrix}
0&0
\end{bmatrix}
\end{aligned}
$$

가 된다. 따라서 모든 입력에서 $z_{\mathrm{out}}=-1/2$인 상수 분류기가 되어 항상 $0$을 예측한다.

더 일반적으로 ReLU 같은 비선형성을 제거한 여러 층은 아핀변환의 합성이다.

$$
\mathbf{W}^{(2)}
(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})
+\mathbf{b}^{(2)}
=\widetilde{\mathbf{W}}\mathbf{x}
+\widetilde{\mathbf{b}}
$$

하나의 아핀 점수와 마지막 계단 함수로 축약되므로, 많아야 직선 하나의 결정경계를 가진 단층 선형 분류기와 표현력이 같다. XOR의 양성 두 점과 음성 두 점은 직선 하나로 분리할 수 없으므로, 여러 방향의 음수 증거를 서로 다르게 막아 주는 ReLU 같은 비선형성이 반드시 필요하다.

---

**다음 연결 고리:** 정사영의 최소값을 오늘은 완전제곱으로 찾았다. 다음에는 **편미분과 그래디언트**로 여러 파라미터가 있는 함수의 변화 방향을 계산하고, **MSE·교차 엔트로피 손실 함수**가 모델의 실수를 하나의 학습 목표로 만드는 방법을 배운다.
