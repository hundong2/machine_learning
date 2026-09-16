# [2026-09-16] 오늘 학습: 온도·엔트로피·확률 재정규화 & 디코딩·Instruction Fine-Tuning

> **오늘의 핵심 문장:** 사전학습은 “다음 토큰 분포”를 만들고, 디코딩은 그 분포에서 실제 토큰을 고르며, Instruction Fine-Tuning은 지시와 응답으로 구성된 예제에서 **응답 토큰의 조건부확률**이 커지도록 분포 자체를 다시 학습한다.

어제는 원문이 토큰 ID가 되고, causal Transformer가 다음 토큰 logits를 만든 뒤, Softmax와 음의 로그가능도로 학습되는 전 과정을 연결했다.

$$
\text{context}
\xrightarrow{\theta}
\mathbf{z}_t
\xrightarrow{\mathrm{Softmax}}
\mathbf{p}_t
$$

그런데 모델이

$$
\mathbf{p}_t
=
(0.45,\;0.30,\;0.15,\;0.10)
$$

이라는 다음 토큰 분포를 만들었다고 해서 화면에 표시할 토큰이 자동으로 정해지는 것은 아니다. 가장 큰 확률만 고를 수도 있고, 확률에 따라 무작위로 뽑을 수도 있으며, 위험한 꼬리 후보를 잘라 낸 뒤 다시 뽑을 수도 있다.

또한 방대한 웹 문서를 이어 쓰도록 사전학습한 모델이 곧바로 “사용자 지시를 따르는 비서”가 되는 것도 아니다. 역할 경계와 모범 응답이 담긴 대화 예제로 추가 학습해야 한다.

오늘은 이 두 문제를 분리한 뒤 다시 연결한다.

$$
\boxed{
\begin{array}{c}
\text{학습된 logits}\\
\downarrow\\
\text{temperature·top-}k\text{·top-}p\\
\downarrow\\
\text{실제 생성 token}
\end{array}
}
\qquad
\boxed{
\begin{array}{c}
\text{instruction·response examples}\\
\downarrow\\
\text{assistant-only causal loss}\\
\downarrow\\
\text{지시를 따르는 방향으로 }\theta\text{ 갱신}
\end{array}
}
$$

첫 번째 상자는 **추론 시 선택 규칙**이고, 두 번째 상자는 **학습 시 파라미터 변경**이다. 둘을 혼동하지 않는 것이 오늘의 가장 중요한 목표다.

오늘 반드시 구분할 세 가지는 다음과 같다.

1. **분포와 선택:** model이 만든 확률분포와 그 분포에서 token 하나를 고르는 algorithm은 다르다.
2. **절단과 재정규화:** 후보를 자른 뒤 남은 값의 합을 다시 $1$로 만들어야 한다.
3. **학습과 추론:** SFT는 parameter를 바꾸고, temperature·top-$k$·top-$p$는 고정된 parameter의 출력을 고른다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 가장 확률이 큰 토큰만 계속 고르면 안 될까?

다음 토큰마다 가장 큰 확률의 후보를 고르는 방법은 단순하고 빠르다. 이를 **greedy decoding**이라고 한다.

$$
x_{t+1}
=
\operatorname*{arg\,max}_{v\in\mathcal{V}}
p_{\theta}(v\mid x_{\le t})
$$

정답이 하나로 고정된 분류 문제라면 자연스러워 보인다. 하지만 언어에는 같은 뜻을 전달하는 여러 표현이 있다.

- “오늘 회의는 취소되었습니다.”
- “오늘 예정된 회의가 취소됐습니다.”
- “오늘 회의 일정은 진행되지 않습니다.”

어느 한 표현만 유일한 정답은 아니다. 매 순간 가장 큰 후보만 선택하면 재현성은 높아지지만, 문맥에 따라 반복적이거나 상투적인 경로에 갇힐 수 있다.

더 중요한 점은 greedy가 **각 시점의 국소 최선**일 뿐, 완성된 전체 수열의 확률을 가장 크게 만든다는 보장이 없다는 것이다. 지금 당장 조금 높은 선택이 다음 단계에서 막다른 길로 이어질 수 있다.

### 1.2 왜 원래 분포에서 그대로 샘플링해도 안 될까?

확률분포 그대로 토큰을 뽑으면 다양한 문장이 나온다.

$$
x_{t+1}
\sim
\operatorname{Categorical}(\mathbf{p}_t)
$$

그러나 vocabulary에는 확률이 아주 작아도 문맥상 부적절한 후보가 수만 개 남아 있을 수 있다. 긴 문장을 생성하면 작은 위험이 반복된다.

예를 들어 매 단계마다 “심각하게 이상한 후보”가 뽑힐 확률이 단지 $0.001$이라고 하자. 독립이라는 단순화 아래 $1{,}000$개 토큰 동안 한 번도 그런 후보가 나오지 않을 확률은

$$
(1-0.001)^{1000}
\approx
0.368
$$

뿐이다. 반대로 적어도 한 번 나올 확률은

$$
1-(1-0.001)^{1000}
\approx
0.632
$$

다. 그래서 분포의 “꼬리”를 제한하는 top-$k$와 top-$p$가 필요해졌다.

### 1.3 왜 temperature라는 손잡이가 필요할까?

같은 모델이라도 사용 목적은 다르다.

- JSON·SQL·도구 인자처럼 형식 정확성이 중요할 때는 흔들림을 줄이고 싶다.
- 브레인스토밍·스토리 생성에서는 여러 가능성을 살리고 싶다.
- 평가 실험에서는 확률적 변동을 통제하고 싶다.

모델을 매번 다시 학습하지 않고도 분포가 얼마나 뾰족하거나 평평한지 조절하는 손잡이가 **temperature**다. 온도가 낮으면 상위 후보에 확률이 더 몰리고, 높으면 후보들이 더 비슷한 확률을 갖는다.

다만 temperature는 사실을 새로 가르치지 않는다. 틀린 후보가 이미 가장 높은 logit을 가졌다면 온도를 낮추는 것은 그 틀린 확신을 더 강하게 만들 수도 있다.

### 1.4 왜 top-$k$ 하나만으로 부족할까?

top-$k$는 항상 후보를 정확히 $k$개 남긴다. 하지만 모델의 확신도는 문맥마다 다르다.

한 문맥에서는

$$
(0.92,\;0.03,\;0.02,\ldots)
$$

처럼 한 후보가 압도적일 수 있다. 다른 문맥에서는

$$
(0.18,\;0.16,\;0.14,\;0.12,\ldots)
$$

처럼 여러 후보가 모두 타당할 수 있다.

고정된 $k=10$은 첫 상황에서는 불필요한 꼬리 후보를 너무 많이 남기고, 두 번째 상황에서는 타당한 후보를 너무 많이 자를 수 있다. **top-$p$ 또는 nucleus sampling**은 후보 개수를 고정하지 않고, 누적 확률질량이 임계값 $p$에 도달할 만큼만 남긴다.

> 9월 10일에 나온 vector search의 top-$k$는 “가까운 문서 $k$개를 검색한다”는 뜻이었다. 오늘의 top-$k$는 “다음 token vocabulary에서 확률이 큰 $k$개만 남긴다”는 뜻이다. 이름은 같지만 대상과 목적이 다르다.

### 1.5 왜 사전학습만으로는 지시를 잘 따르지 못할까?

사전학습의 목표는 인터넷 문서의 다음 token을 맞히는 것이다. 다음과 같은 prompt가 주어졌다고 하자.

> 사용자: 물의 끓는점을 한 문장으로 설명해 줘.

base model은 답을 할 수도 있지만, 웹 문서에서 자주 본 형태를 따라

- “사용자: …”를 계속 만들어 가거나,
- 관련 없는 문단을 길게 이어 쓰거나,
- 질문·답변 형식을 흉내만 내거나,
- 어느 역할이 말해야 하는지 혼동할 수 있다.

사전학습 목표에는 “user의 요청 뒤에는 assistant가 유용한 답을 해야 한다”라는 역할 계약이 명시되어 있지 않기 때문이다.

**Instruction Fine-Tuning**, 줄여서 **IFT** 또는 문맥에 따라 **SFT**는 다음과 같은 모범 예제로 추가 학습한다.

$$
\text{system rule}
\;+\;
\text{user instruction}
\;+\;
\text{assistant response}
$$

새로운 신경망 구조가 필요한 것은 아니다. 어제 배운 causal cross-entropy를 대화 형식의 데이터에 적용한다.

### 1.6 왜 prompt까지 모두 정답으로 채점하지 않을까?

학습 예제 전체에 causal loss를 적용하면 모델은 assistant의 답뿐 아니라 system 문구와 user 질문까지 “생성해야 할 정답”으로 배운다.

우리가 원하는 핵심 행동은 보통 다음이다.

$$
\text{prompt가 주어졌을 때}
\longrightarrow
\text{좋은 assistant response를 생성}
$$

그래서 system·user token은 **조건으로 읽되**, assistant 응답 token만 직접 loss에 포함하는 **response-only loss masking**을 많이 사용한다.

이때 흔한 오해가 있다.

> prompt token을 loss에서 가리면 prompt는 모델이 보지 않는가?

아니다. prompt는 attention을 통해 응답 hidden state를 만드는 조건으로 사용된다. 단지 prompt 자체를 다음-token 정답으로 직접 채점하지 않을 뿐이다.

### 1.7 오늘의 수학과 AI 부품은 어디에서 만날까?

$$
\boxed{
\begin{array}{c}
\text{logit 차이와 비율}
\longleftrightarrow
\text{temperature가 후보 간 odds를 조절}\\
\text{Shannon entropy}
\longleftrightarrow
\text{분포의 퍼짐·불확실성 측정}\\
\text{집합 제한과 조건부확률}
\longleftrightarrow
\text{top-}k\text{·top-}p\text{ 뒤 재정규화}\\
\text{역변환 샘플링}
\longleftrightarrow
\text{범주형 분포에서 실제 token 선택}\\
\text{지시함수·masked mean}
\longleftrightarrow
\text{assistant token만 직접 SFT loss에 포함}\\
\text{연쇄법칙·오차역전파}
\longleftrightarrow
\text{응답 loss가 prompt를 처리한 가중치까지 갱신}
\end{array}
}
$$

---

## 2. 친절한 용어 사전

### 2.1 확률분포와 디코딩 용어

| 용어 | 표기 | 초보자 해설 |
|---|---:|---|
| logit | $z_v$ | token $v$에 대한 Softmax 전 점수다. 음수도 가능하며 확률이 아니다. |
| categorical distribution | $\operatorname{Cat}(\mathbf{q})$ | 유한한 후보 중 정확히 하나를 확률 $\mathbf{q}$에 따라 뽑는 분포다. 주사위의 면마다 서로 다른 확률을 붙인 것과 같다. |
| decoding |  | 모델의 다음-token 분포를 실제 token 선택으로 바꾸고, 이를 반복해 수열을 만드는 추론 절차다. tokenizer의 ID-to-text decode와 구분한다. |
| greedy decoding | $\arg\max$ | 매 단계에서 확률이 가장 큰 token 하나를 선택한다. 보통 결정적이지만 전체 수열 최적화는 아니다. |
| sampling | $x\sim\operatorname{Cat}(\mathbf{q})$ | 난수를 사용해 분포의 확률에 비례하여 token을 뽑는다. |
| temperature | $\tau>0$ | logits를 Softmax 전에 $\tau$로 나누어 분포의 뾰족함을 조절하는 양수다. |
| inverse temperature | $\beta=1/\tau$ | 온도의 역수다. 통계물리와 수식 유도에서 편리하다. |
| entropy | $H(\mathbf{q})$ | 확률이 얼마나 여러 후보에 퍼져 있는지 재는 값이다. 자연로그를 쓰면 단위는 nat다. |
| support | $\operatorname{supp}(\mathbf{q})$ | 확률이 $0$보다 큰 후보들의 집합이다. |
| top-$k$ sampling |  | 확률이 큰 $k$개 token만 남기고 다시 정규화해 샘플링한다. |
| top-$p$ sampling | nucleus sampling | 내림차순 누적 확률이 $p$ 이상이 되는 최소 후보 집합만 남긴다. 후보 수는 문맥마다 바뀐다. |
| truncation |  | 낮은 순위 후보를 잘라 확률을 $0$으로 만드는 과정이다. 문자열을 길이에서 자르는 것과 다른 뜻이다. |
| renormalization |  | 남은 확률들의 합이 다시 $1$이 되도록 같은 상수로 나누는 과정이다. |
| random seed |  | 의사난수 생성기의 시작 상태다. 같은 구현·연산 순서에서는 재현에 도움을 주지만 모든 플랫폼에서 동일 출력을 보장하지 않는다. |
| EOS | $\langle\mathrm{EOS}\rangle$ | 생성을 끝내는 특수 token이다. EOS도 vocabulary의 한 후보로 확률을 받는다. |
| max new tokens |  | prompt 뒤에 새로 생성할 수 있는 token 수의 상한이다. EOS가 안 나오는 경우의 안전장치다. |

### 2.2 확률과 집합 기호

| 기호 | 읽는 법 | 뜻 |
|---|---|---|
| $\mathcal{V}$ | vocabulary | 모델이 선택할 수 있는 전체 token 집합이다. |
| $V=\lvert\mathcal{V}\rvert$ | vocabulary size | token 종류의 수다. |
| $\mathbf{z}\in\mathbb{R}^{V}$ | logit vector | 현재 위치에서 모든 token에 준 점수 벡터다. |
| $q_v^{(\tau)}$ | temperature-scaled probability | 온도 $\tau$를 적용한 token $v$의 확률이다. |
| $\mathcal{S}$ | candidate set | 절단 뒤 남겨 둔 token 집합이다. |
| $\mathbf{1}[\cdot]$ | indicator | 조건이 참이면 $1$, 거짓이면 $0$인 함수다. |
| $\mathbb{E}_{q}[Z]$ | expectation | 분포 $q$로 가중한 $Z$의 평균이다. |
| $\operatorname{Var}_{q}(Z)$ | variance | $Z$가 그 평균에서 얼마나 퍼져 있는지 나타낸다. 항상 $0$ 이상이다. |
| $\operatorname*{arg\,max}$ | 최대점의 위치 | 가장 큰 값 자체가 아니라 그 값을 만드는 후보를 반환한다. |

### 2.3 Instruction Fine-Tuning 용어

| 용어 | 표기 | 초보자 해설 |
|---|---:|---|
| base model |  | 대규모 corpus의 다음 token 예측으로 사전학습했지만 대화 역할 수행에 특화되지 않은 모델이다. |
| pretrained model |  | 큰 데이터에서 먼저 일반 패턴을 학습한 모델이다. 이 가중치를 출발점으로 추가 학습한다. |
| fine-tuning |  | 이미 학습된 모델을 더 작은 목적별 데이터로 추가 훈련하는 과정이다. |
| supervised fine-tuning | SFT | 입력과 모범 출력 쌍을 정답으로 제공해 지도학습하는 fine-tuning이다. |
| instruction tuning | IFT | 여러 지시와 모범 응답으로 “지시를 수행하는 행동”을 가르치는 SFT다. 실무에서는 SFT와 겹쳐 부르기도 한다. |
| chat template |  | system·user·assistant 역할과 경계를 모델별 control token으로 직렬화하는 규칙이다. |
| prompt |  | 모델에게 조건으로 제공되는 system·user 내용과 필요한 대화 history다. |
| completion / response |  | prompt 뒤에서 모델이 생성해야 하는 assistant 답이다. |
| control token |  | 역할 시작·끝, tool call, turn 경계 등을 나타내는 특수 token이다. 일반 문장부호처럼 임의로 바꾸면 안 된다. |
| label | $y_t$ | 각 위치에서 모델이 맞혀야 할 다음 token ID다. |
| loss mask | $m_t\in\{0,1\}$ | target token $t$를 loss에 포함할지 정하는 스위치다. |
| ignore index | 예: $-100$ | 많은 라이브러리에서 해당 label 위치를 cross-entropy 계산에서 제외하라는 예약값이다. token ID가 아니다. |
| epoch |  | 준비한 training dataset 전체를 한 번 훑는 단위다. 3 epochs면 각 예제를 대략 세 번 본다. |
| batch |  | 한 번의 forward/backward 계산에서 함께 처리하는 여러 예제 묶음이다. |
| learning rate | $\eta$ | 한 번의 update에서 gradient 방향으로 얼마나 움직일지 정하는 크기다. |
| overfitting |  | training 예제는 외웠지만 새로운 지시에는 일반화하지 못하는 상태다. |
| parameter-efficient fine-tuning | PEFT | base parameter 대부분을 고정하고 적은 추가 parameter만 학습하는 방법의 계열이다. LoRA가 대표적이다. |

> **중요한 동명이인:** knowledge distillation에서 teacher distribution을 부드럽게 만드는 temperature와 오늘의 inference temperature는 같은 Softmax 수학을 쓸 수 있지만, 전자는 학습 신호를 만드는 용도이고 후자는 생성 token을 고르는 용도다.

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 logits에 temperature를 적용하기

현재 문맥에서 vocabulary의 logits가

$$
\mathbf{z}
=
(z_1,\ldots,z_V)
\in
\mathbb{R}^{V}
$$

라고 하자. 온도 $\tau>0$를 적용한 분포는

$$
\boxed{
q_i^{(\tau)}
=
\frac{\exp(z_i/\tau)}
{\sum_{j=1}^{V}\exp(z_j/\tau)}
}
$$

다.

수치적으로는

$$
m=\max_j z_j
$$

를 빼서

$$
q_i^{(\tau)}
=
\frac{\exp((z_i-m)/\tau)}
{\sum_j\exp((z_j-m)/\tau)}
$$

로 계산한다. 모든 logit에서 같은 값을 빼도 분자와 분모의 공통 인자가 약분되므로 분포는 변하지 않는다.

양의 상수로 나누는 것은 순서를 바꾸지 않는다.

$$
z_i>z_j
\quad\Longleftrightarrow\quad
\frac{z_i}{\tau}>
\frac{z_j}{\tau}
$$

따라서 $\tau>0$인 temperature scaling만으로는 token 순위가 바뀌지 않는다. 확률의 **간격**만 바뀐다.

### 3.2 temperature가 odds를 어떻게 바꾸는가?

후보 $i$와 $j$의 확률비를 계산하면

$$
\frac{q_i^{(\tau)}}{q_j^{(\tau)}}
=
\frac{\exp(z_i/\tau)}
{\exp(z_j/\tau)}
=
\exp\left(
\frac{z_i-z_j}{\tau}
\right).
$$

양변에 로그를 취하면

$$
\boxed{
\log
\frac{q_i^{(\tau)}}{q_j^{(\tau)}}
=
\frac{z_i-z_j}{\tau}
}
$$

다. 즉 temperature는 모든 pairwise log-odds를 $1/\tau$배 한다.

- $0<\tau<1$: logit 차이를 확대한다.
- $\tau=1$: 원래 Softmax다.
- $\tau>1$: logit 차이를 축소한다.

예를 들어

$$
\mathbf{z}
=
(2,\;1,\;0)
$$

이면 대략 다음과 같다.

| $\tau$ | $q_1^{(\tau)}$ | $q_2^{(\tau)}$ | $q_3^{(\tau)}$ | 해석 |
|---:|---:|---:|---:|---|
| $0.5$ | $0.867$ | $0.117$ | $0.016$ | 1위에 매우 집중 |
| $1$ | $0.665$ | $0.245$ | $0.090$ | 원래 Softmax |
| $2$ | $0.506$ | $0.307$ | $0.186$ | 더 평평함 |

### 3.3 $\tau\to0^+$와 $\tau\to\infty$ 극한

최대 logit이 유일하게 $z_m$이라고 하자. $i\neq m$에 대해

$$
\frac{q_i^{(\tau)}}{q_m^{(\tau)}}
=
\exp
\left(
\frac{z_i-z_m}{\tau}
\right).
$$

$z_i-z_m<0$이므로

$$
\lim_{\tau\to0^+}
\frac{q_i^{(\tau)}}{q_m^{(\tau)}}
=0.
$$

따라서

$$
\lim_{\tau\to0^+}
q_m^{(\tau)}
=1,
\qquad
\lim_{\tau\to0^+}
q_i^{(\tau)}
=0
\quad(i\neq m).
$$

이는 greedy 선택과 연결된다. 하지만 수식

$$
\frac{z_i}{\tau}
$$

에 실제로 $\tau=0$을 넣을 수는 없다. API의 “temperature 0”은 보통 이 극한을 흉내 낸 greedy 모드를 뜻하는 구현상의 약속이다.

최대 logit이 정확히 여러 개로 동률이면 $\tau\to0^+$에서 그 최대 후보들 사이에 질량이 나뉜다. tie-breaking은 구현에 따라 달라질 수 있다.

반대로 $\tau\to\infty$이면 각 $z_i/\tau\to0$이므로

$$
\exp(z_i/\tau)\to1
$$

이고

$$
\lim_{\tau\to\infty}
q_i^{(\tau)}
=
\frac{1}{V}.
$$

즉 유한 vocabulary 위의 균등분포로 간다.

### 3.4 temperature와 entropy의 관계를 증명하기

분포 $\mathbf{q}^{(\tau)}$의 Shannon entropy를

$$
H(\tau)
=
-\sum_{i=1}^{V}
q_i^{(\tau)}
\log q_i^{(\tau)}
$$

로 정의한다. 온도를 높이면 entropy가 증가한다는 직관을 증명해 보자.

역온도

$$
\beta=\frac{1}{\tau}
$$

를 쓰면

$$
q_i(\beta)
=
\frac{\exp(\beta z_i)}
{Z(\beta)},
\qquad
Z(\beta)
=
\sum_j\exp(\beta z_j).
$$

로그확률은

$$
\log q_i(\beta)
=
\beta z_i-\log Z(\beta)
$$

다. 이를 entropy에 대입하면

$$
\begin{aligned}
H(\beta)
&=
-\sum_i q_i
\left(
\beta z_i-\log Z
\right)\\
&=
-\beta
\sum_iq_i z_i
+
\log Z
\sum_iq_i\\
&=
\log Z
-
\beta\mathbb{E}_{q}[z].
\end{aligned}
$$

여기서

$$
\frac{d}{d\beta}
\log Z
=
\frac{1}{Z}
\sum_i z_i\exp(\beta z_i)
=
\mathbb{E}_{q}[z].
$$

또한 위 Softmax 식을 직접 미분하면

$$
\frac{d}{d\beta}
\mathbb{E}_{q}[z]
=
\operatorname{Var}_{q}(z)
$$

다. 직접 확인하면

$$
\begin{aligned}
\frac{d}{d\beta}
\mathbb{E}_{q}[z]
&=
\frac{d}{d\beta}
\sum_iq_i z_i\\
&=
\sum_iq_i
\left(
z_i-\mathbb{E}_{q}[z]
\right)z_i\\
&=
\mathbb{E}_{q}[z^2]
-
\mathbb{E}_{q}[z]^2\\
&=
\operatorname{Var}_{q}(z).
\end{aligned}
$$

따라서

$$
\begin{aligned}
\frac{dH}{d\beta}
&=
\mathbb{E}_{q}[z]
-
\left(
\mathbb{E}_{q}[z]
+
\beta\operatorname{Var}_{q}(z)
\right)\\
&=
-\beta\operatorname{Var}_{q}(z).
\end{aligned}
$$

그리고

$$
\frac{d\beta}{d\tau}
=
-\frac{1}{\tau^2}
$$

이므로 연쇄법칙으로

$$
\boxed{
\frac{dH}{d\tau}
=
\frac{\operatorname{Var}_{q^{(\tau)}}(z)}
{\tau^3}
\ge0
}
$$

를 얻는다.

logits가 모두 같지 않다면 분산은 양수이므로 entropy는 온도와 함께 증가한다. logits가 모두 같으면 이미 균등분포라서 어느 온도에서도 분산과 미분이 $0$이다.

> entropy가 높다는 것은 “문장이 창의적이다”와 완전히 같은 말이 아니다. 현재 한 위치의 token 확률이 퍼져 있다는 수학적 사실만 말한다.

### 3.5 greedy가 전체 수열의 최대확률을 보장하지 않는 반례

두 token만 생성한다고 하자. 첫 단계에서

$$
p(A)=0.6,
\qquad
p(B)=0.4
$$

이므로 greedy는 $A$를 고른다.

다음 단계의 최고 확률이

$$
\max_y p(y\mid A)=0.5,
\qquad
\max_y p(y\mid B)=0.9
$$

라면 greedy 경로의 최고 완성 수열 확률은

$$
0.6\times0.5
=
0.30
$$

이다. 반면 $B$로 시작하는 최고 수열은

$$
0.4\times0.9
=
0.36
$$

이다.

$$
0.36>0.30
$$

이므로 첫 단계의 국소 argmax가 전체 수열의 argmax를 놓쳤다. beam search는 여러 prefix를 동시에 유지해 이 문제를 일부 줄이지만, beam 폭이 유한하면 여전히 완전탐색은 아니며 길이 편향과 다양성 감소 같은 trade-off가 있다.

### 3.6 top-$k$: 후보 집합을 크기로 자르기

온도 적용 후 확률을 큰 순서대로 정렬한 index를

$$
\pi(1),\pi(2),\ldots,\pi(V)
$$

라 하자.

$$
q_{\pi(1)}
\ge
q_{\pi(2)}
\ge\cdots\ge
q_{\pi(V)}.
$$

top-$k$ 후보 집합은

$$
\mathcal{S}_k
=
\{\pi(1),\ldots,\pi(k)\}
$$

다. 절단 뒤 분포는

$$
\boxed{
\widetilde q_i^{(k)}
=
\frac{
q_i\mathbf{1}[i\in\mathcal{S}_k]
}{
\sum_{j\in\mathcal{S}_k}q_j
}
}
$$

다.

분모가 바로 **재정규화 상수**다. 이를 빼먹으면 남은 값의 합이 $1$보다 작아 categorical distribution이 되지 않는다.

예를 들어

$$
\mathbf{q}
=
(0.40,\;0.30,\;0.15,\;0.10,\;0.05)
$$

에서 $k=3$이면 남은 질량은

$$
0.40+0.30+0.15
=
0.85
$$

이고,

$$
\widetilde{\mathbf{q}}^{(3)}
=
\left(
\frac{0.40}{0.85},
\frac{0.30}{0.85},
\frac{0.15}{0.85},
0,
0
\right)
\approx
(0.471,\;0.353,\;0.176,\;0,\;0).
$$

### 3.7 top-$p$: 후보 집합을 누적 확률질량으로 자르기

$0<p\le1$이라 하자. 정렬된 확률의 누적합이 처음으로 $p$ 이상이 되는 최소 index를

$$
r_p
=
\min
\left\{
r:
\sum_{j=1}^{r}
q_{\pi(j)}
\ge p
\right\}
$$

로 정의한다.

그러면 nucleus는

$$
\mathcal{S}_p
=
\{
\pi(1),\ldots,\pi(r_p)
\}
$$

이고,

$$
\boxed{
\widetilde q_i^{(p)}
=
\frac{
q_i\mathbf{1}[i\in\mathcal{S}_p]
}{
\sum_{j\in\mathcal{S}_p}q_j
}
}
$$

다.

앞의 예에서 $p=0.70$이면

$$
0.40<0.70,
\qquad
0.40+0.30=0.70
$$

이므로 두 후보만 남는다. $p=0.80$이면

$$
0.40+0.30<0.80,
\qquad
0.40+0.30+0.15=0.85
$$

이므로 세 후보가 남는다.

top-$p$는 문맥이 확실하면 적은 후보를, 애매하면 많은 후보를 남기는 **적응형 후보 수**를 갖는다.

동률 처리, 임계값 포함 여부, 최소 후보 수는 library마다 세부 구현이 다를 수 있다. 따라서 실험에는 library 버전과 generation configuration을 함께 기록해야 한다.

### 3.8 temperature, top-$k$, top-$p$의 적용 순서

일반적인 pipeline은 다음과 같다.

$$
\mathbf{z}
\xrightarrow{\div\tau}
\mathbf{z}^{(\tau)}
\xrightarrow{\mathrm{Softmax}}
\mathbf{q}^{(\tau)}
\xrightarrow{\text{top-}k/\text{top-}p}
\widetilde{\mathbf{q}}
\xrightarrow{\text{sample}}
x_{t+1}.
$$

temperature는 순위를 바꾸지 않지만 확률 누적값은 바꾼다. 따라서 top-$p$ 앞에 temperature를 적용하면 nucleus의 크기도 바뀔 수 있다.

top-$k$와 top-$p$를 함께 쓰는 구현도 있다. 두 집합을 **같은 절단 전 분포에서 독립적으로 계산한다면** 남는 후보는

$$
\mathcal{S}
=
\mathcal{S}_k
\cap
\mathcal{S}_p
$$

라는 교집합이다. 그러나 top-$k$로 자르고 재정규화한 결과에 다시 top-$p$를 순차 적용하면 $\mathcal{S}_p$ 자체가 달라질 수 있다. 따라서 실제 processor 순서, tie 처리, 최소 후보 수는 API 문서를 확인해야 한다.

### 3.9 범주형 샘플링은 어떻게 실제 token 하나를 고를까?

후보가 정렬되어 있고 재정규화된 확률이

$$
\widetilde{\mathbf{q}}
=
(\widetilde q_1,\ldots,\widetilde q_r)
$$

라고 하자.

먼저

$$
U\sim\operatorname{Uniform}[0,1)
$$

인 난수 하나를 뽑는다. 그리고 누적합

$$
F(m)
=
\sum_{i=1}^{m}\widetilde q_i
$$

가 처음으로 $U$보다 커지는 index

$$
M
=
\min\{m:F(m)>U\}
$$

를 선택한다. CDF는 **cumulative distribution function**, 즉 “이 index까지의 누적확률”이다. 누적함수의 구간에서 난수의 위치를 거꾸로 찾아가는 이 방법을 inverse-CDF sampling이라고 한다.

예를 들어

$$
\widetilde{\mathbf{q}}
=
(0.5,\;0.3,\;0.2)
$$

이면 구간은

$$
[0,0.5),\quad
[0.5,0.8),\quad
[0.8,1)
$$

로 나뉜다. $U=0.73$이면 두 번째 token, $U=0.91$이면 세 번째 token을 뽑는다.

seed를 고정하면 같은 의사난수열을 재사용할 수 있다. 그러나 GPU kernel, 병렬 reduction 순서, batching, model revision, provider 내부 구현이 바뀌면 같은 seed라도 완전히 같은 출력이 보장되지 않는다.

### 3.10 SFT sequence와 response-only masked loss

chat template를 적용한 하나의 token sequence를

$$
\mathbf{s}
=
(s_1,\ldots,s_L)
$$

라 하자. 예를 들면 개념적으로

$$
\begin{aligned}
\mathbf{s}
=
[
&\langle\mathrm{system}\rangle,
\text{규칙},
\langle\mathrm{user}\rangle,
\text{질문},\\
&\langle\mathrm{assistant}\rangle,
\text{응답},
\langle\mathrm{EOS}\rangle
].
\end{aligned}
$$

모델은 여전히

$$
p_{\theta}(s_t\mid s_{<t})
$$

를 예측한다. target token $s_t$가 직접 채점할 assistant 영역이면 $m_t=1$, 아니면 $m_t=0$으로 둔다.

$$
m_t
=
\begin{cases}
1,
&
s_t\text{가 채점 대상 assistant token일 때},\\
0,
&
\text{그 외}.
\end{cases}
$$

유효 target 수를

$$
K
=
\sum_{t=2}^{L}m_t
$$

라 하자. 학습 가능한 예제로 쓰려면 $K>0$이어야 한다. response-only causal loss는

$$
\boxed{
\mathcal{L}_{\mathrm{SFT}}
=
-
\frac{1}{K}
\sum_{t=2}^{L}
m_t
\log
p_{\theta}
\left(
s_t\mid s_{<t}
\right)
}
$$

다.

분모는 전체 sequence 길이 $L$이 아니라 실제로 채점한 token 수 $K$다. 그래야 prompt 길이가 긴 예제가 단지 padding·prompt token 때문에 작은 loss를 갖는 문제가 생기지 않는다.

EOS, assistant role 종료 token, tool-call token을 loss에 포함할지는 training policy다. 종료 행동을 배우게 하려면 보통 적절한 종료 token을 target에 포함해야 한다. 어느 token을 포함했는지 명시해야 실험이 재현된다.

### 3.11 masked 위치의 logit gradient

target $s_t$를 예측하는 이전 위치의 logits를 간단히 $\mathbf{z}_t$, 그 예측분포를 $\mathbf{p}_t$, one-hot target을 $\mathbf{y}_t$라 하자. Softmax와 cross-entropy를 합치면

$$
\boxed{
\frac{\partial\mathcal{L}_{\mathrm{SFT}}}
{\partial\mathbf{z}_t}
=
\frac{m_t}{K}
\left(
\mathbf{p}_t-\mathbf{y}_t
\right)
}
$$

이다.

- $m_t=1$이면 정답 logit을 올리고 오답 logits를 내리는 직접 gradient가 생긴다.
- $m_t=0$이면 그 위치의 **직접적인 local loss gradient**는 $0$이다.

그러나 prompt token이 계산 그래프에서 제거된 것은 아니다. assistant 위치의 hidden state는 causal attention으로 앞선 prompt 표현을 읽는다.

$$
\text{prompt hidden states}
\longrightarrow
\text{assistant logits}
\longrightarrow
\mathcal{L}_{\mathrm{SFT}}
$$

따라서 assistant loss의 gradient는 이 경로를 거슬러 prompt를 처리한 **trainable** embedding·attention·FFN parameter에도 도달한다. full fine-tuning에서는 해당 base parameter가 update된다. PEFT처럼 base가 frozen이면 그 경로는 adapter의 gradient 계산에는 영향을 주지만 frozen base parameter 자체는 optimizer가 update하지 않는다. “prompt를 직접 정답으로 채점하지 않는다”와 “prompt가 학습에 아무 영향도 없다”는 전혀 다른 말이다.

### 3.12 training과 decoding을 하나의 식에서 분리하기

SFT 전후의 모델 분포는 parameter $\theta$가 달라진다.

$$
p_{\theta_{\mathrm{base}}}
\quad\longrightarrow\quad
p_{\theta_{\mathrm{SFT}}}.
$$

반면 temperature와 truncation은 주어진 parameter를 고정한 채 출력 규칙을 바꾼다.

$$
p_{\theta}
\quad\longrightarrow\quad
q_{\theta,\tau,\mathcal{S}}.
$$

이를 한 줄로 쓰면

$$
\boxed{
\text{SFT는 }\theta\text{를 바꾸고,}
\qquad
\text{decoding은 고정된 }\theta\text{의 분포를 선택 규칙으로 바꾼다.}
}
$$

낮은 temperature는 부족한 지식, 잘못된 instruction tuning, prompt injection 취약성, 사실 오류를 고치지 않는다. 높은 temperature도 모델에 새로운 창의적 지식을 추가하지 않는다. 둘은 모델 능력과 별도의 **출력 정책**이다.

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 base model에서 instruction-following model까지

오늘의 training pipeline은 다음과 같다.

$$
\boxed{
\begin{array}{c}
\text{instruction dataset}\\
\downarrow\\
\text{chat template 적용}\\
\downarrow\\
\text{tokenize·shift}\\
\downarrow\\
\text{assistant target mask}\\
\downarrow\\
\text{masked causal CE}\\
\downarrow\\
\text{backpropagation·parameter update}
\end{array}
}
$$

각 단계가 하나의 계약이다.

#### 1단계: 데이터 schema를 정한다

대화 예제는 개념적으로 다음과 같다.

~~~text
system: 너는 정확하고 간결한 과학 튜터다.
user: 왜 하늘은 파랗게 보이나요?
assistant: 짧은 파장의 파란빛이 대기 분자에 더 강하게 산란되기 때문입니다.
~~~

좋은 instruction dataset은 답변의 사실성만이 아니라 원하는 형식, 거절 경계, tool 사용 방식, 길이, 언어, tone까지 행동 예제로 담는다.

#### 2단계: 모델 고유의 chat template로 직렬화한다

실제 token sequence는 사람이 보는 role label과 다를 수 있다.

~~~text
<|system_start|>
너는 정확하고 간결한 과학 튜터다.
<|turn_end|>
<|user_start|>
왜 하늘은 파랗게 보이나요?
<|turn_end|>
<|assistant_start|>
짧은 파장의 파란빛이 대기 분자에 더 강하게 산란되기 때문입니다.
<|turn_end|>
~~~

control token과 줄바꿈 규칙은 모델마다 다르다. training 때와 inference 때 template가 다르면 모델은 역할 경계를 다른 token pattern으로 보게 된다.

> chat template는 화면 장식이 아니라 모델 입력의 일부다.

#### 3단계: label과 loss mask를 만든다

응답 전용 학습에서는 개념적으로 다음처럼 label을 둔다.

| 영역 | 입력으로 읽나? | 직접 loss를 주나? |
|---|---:|---:|
| system | 예 | 보통 아니오 |
| user | 예 | 보통 아니오 |
| assistant role 시작 | 예 | policy에 따라 다름 |
| assistant content | 예 | 예 |
| assistant 종료·EOS | 예 | 보통 예 |
| padding | batch shape에는 존재 | 아니오 |

많은 구현은 제외할 label 위치에 ignore index인 $-100$을 넣는다.

~~~text
input_ids: [SYS, s1, USR, u1, u2, AST, a1, a2, EOS]
labels:    [-100, -100, -100, -100, -100, -100, a1, a2, EOS]
~~~

$-100$은 vocabulary token이 아니라 loss 함수에 “이 위치를 무시하라”고 알리는 값이다.

많은 causal-LM API는 같은 길이의 input_ids와 labels를 받은 뒤 model 내부에서 logits와 labels를 한 칸 shift한다. 따라서 위 표에서 a1이 같은 배열 index에 적혀 있어도, 실제 loss는 바로 앞 위치의 output으로 a1을 예측한다. 직접 training loop를 쓰면 이 shift를 구현자가 명시적으로 해야 한다.

#### 4단계: batch의 유효 assistant token으로 평균한다

batch $b=1,\ldots,B$까지 포함하면

$$
\mathcal{L}_{\mathrm{batch}}
=
-
\frac{
\sum_{b=1}^{B}
\sum_t
m_{b,t}
\log p_{\theta}
\left(
s_{b,t}\mid s_{b,<t}
\right)
}{
\sum_{b=1}^{B}
\sum_t m_{b,t}
}.
$$

예제별 평균을 다시 평균하는 방식과 전체 유효 token 평균은, 응답 길이가 다르면 가중치가 달라진다. 어떤 reduction을 썼는지 기록해야 한다.

### 4.2 all-token loss와 response-only loss 비교

| 방식 | 직접 채점 영역 | 장점 | 주의점 |
|---|---|---|---|
| all-token causal loss | system·user·assistant 전체 | 모든 token에 학습 신호가 있어 단순함 | user 문장 생성까지 목적에 섞이고 긴 prompt가 loss를 지배할 수 있음 |
| response-only loss | assistant target 중심 | “주어진 지시에 답하기”와 목적이 정렬됨 | role mask 오류, assistant token 수 부족, 종료 token 누락을 주의 |
| turn-aware multi-turn loss | 여러 assistant turn·tool 결과를 선택적으로 채점 | agent 대화 행동을 세밀하게 설계 가능 | template와 역할별 policy가 복잡해짐 |

response-only가 언제나 절대적으로 우월한 것은 아니다. domain adaptation처럼 원문 자체의 언어 분포도 계속 학습하려면 all-token 또는 혼합 loss가 적절할 수 있다. 핵심은 목적과 mask가 일치하는가다.

### 4.3 full fine-tuning과 PEFT

SFT는 **어떤 데이터와 loss로 학습하는가**를 말한다. full fine-tuning과 PEFT는 **어떤 parameter를 update하는가**를 말한다.

#### Full fine-tuning

$$
\theta
\leftarrow
\theta-\eta\nabla_{\theta}\mathcal{L}
$$

모든 trainable parameter를 update한다. 표현력이 크지만 optimizer state와 gradient memory가 크고, task별 전체 checkpoint를 저장해야 한다.

#### PEFT

base parameter $\theta_0$를 고정하고 작은 adapter parameter $\phi$만 update한다.

$$
\theta_0
\text{ fixed},
\qquad
\phi
\leftarrow
\phi-\eta\nabla_{\phi}\mathcal{L}.
$$

LoRA는 frozen linear weight $\mathbf{W}$에 저랭크 변화량을 더한다.

$$
\mathbf{W}_{\mathrm{adapted}}
=
\mathbf{W}
+
\alpha\mathbf{B}\mathbf{A}.
$$

여기서 $\alpha$는 저랭크 변화량 $\mathbf{B}\mathbf{A}$가 base 출력에 미치는 크기를 조절하는 scale 계수다.

여기서 rank $r$가 작으면

$$
\mathbf{A}\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\qquad
\mathbf{B}\in\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

만 학습해 저장량을 줄일 수 있다.

어느 방식을 쓰더라도 assistant-only loss mask와 chat template 계약은 그대로 중요하다. PEFT가 잘못된 label을 자동으로 고쳐 주지는 않는다.

### 4.4 실제 autoregressive decoding loop

생성은 한 번의 forward로 끝나지 않는다.

$$
\text{prompt}
\rightarrow
x_{T+1}
\rightarrow
x_{T+2}
\rightarrow
\cdots
\rightarrow
\langle\mathrm{EOS}\rangle
$$

개념적 의사코드는 다음과 같다.

~~~python
tokens = prompt_ids

for _ in range(max_new_tokens):
    logits = model(tokens).next_token_logits

    if not do_sample or temperature == 0:
        next_id = argmax(logits)
    else:
        logits = logits / temperature
        probs = softmax(logits)
        probs = keep_top_k_or_top_p_then_renormalize(probs)
        next_id = categorical_sample(probs)

    tokens.append(next_id)

    if next_id == eos_id:
        break
~~~

실제 구현은 KV cache를 사용해 이미 처리한 prefix의 Key·Value를 재사용한다. 9월 14일에 배운 KV cache가 이 반복 생성의 비용을 줄인다.

### 4.5 목적별 decoding 전략

| 목적 | 출발 전략 | 이유 | 추가 검증 |
|---|---|---|---|
| JSON·function arguments | greedy 또는 매우 낮은 randomness | schema 일관성을 우선 | parser·schema validator·retry |
| 코드 생성 | 낮거나 중간 temperature, 여러 후보 | 결정성과 대안 탐색의 균형 | compile·test·static analysis |
| factual QA | 낮은 temperature만 믿지 않기 | 확신과 사실성은 다름 | retrieval·citation·fact check |
| 창작·아이디어 | temperature sampling + top-$p$ | 타당한 여러 경로를 살림 | 중복·안전·품질 filter |
| benchmark 재현 | 설정·seed·model revision 고정 | 변동 원인을 기록 | 여러 run의 평균·분산·Pass$^k$ |
| agent tool call | 낮은 randomness + constrained decoding | argument drift를 줄임 | 권한 제한·dry run·결과 검증 |

“권장 숫자 하나”를 모든 model과 task에 적용해서는 안 된다. 같은 $\tau$라도 logits scale과 SFT 방식이 다르면 실제 entropy가 다르다. validation set에서 task metric, 다양성, 실패율, latency를 함께 측정해야 한다.

### 4.6 SFT와 decoding을 함께 보는 작은 예제

사용자 질문이

> 2와 3을 더한 값을 숫자 하나로 답해.

라고 하자.

base model은 continuation 습관 때문에

~~~text
사용자: 2와 3을 더한 값을 숫자 하나로 답해.
assistant:
~~~

같은 role text를 반복할 수 있다. SFT는 assistant target “5”의 log-likelihood를 높이고 role boundary 뒤에 답하는 행동을 학습한다.

$$
\theta_{\mathrm{base}}
\xrightarrow{\mathrm{SFT}}
\theta_{\mathrm{instruct}}
$$

그 뒤 decoding은 $\theta_{\mathrm{instruct}}$가 만든 분포에서 token을 고른다.

$$
p_{\theta_{\mathrm{instruct}}}
(\text{“5”}\mid\text{prompt})
=0.97
$$

이라면 greedy와 낮은 temperature sampling 모두 대체로 “5”를 고를 것이다. 하지만 model이

$$
p_{\theta_{\mathrm{instruct}}}
(\text{“6”}\mid\text{prompt})
>
p_{\theta_{\mathrm{instruct}}}
(\text{“5”}\mid\text{prompt})
$$

로 잘못 배웠다면 temperature를 낮추는 것은 “6”을 더 확실하게 고를 뿐이다. 이 경우 필요한 것은 데이터·학습·추론 검증의 수정이지 temperature tuning만이 아니다.

### 4.7 초보자가 흔히 하는 오해와 주의할 점

| 오해 | 정확한 설명 |
|---|---|
| “temperature $0$을 Softmax 식에 대입한다.” | $0$으로 나눌 수 없다. 보통 greedy를 뜻하는 API 약속 또는 $\tau\to0^+$ 극한이다. |
| “temperature를 낮추면 사실성이 보장된다.” | 가장 높은 후보를 더 강화할 뿐이다. 최고 후보가 틀리면 더 일관되게 틀릴 수 있다. |
| “높은 temperature는 새로운 지식을 만든다.” | 기존 logits를 평평하게 할 뿐, parameter에 없는 지식을 추가하지 않는다. |
| “top-$p=0.9$면 token의 상위 $90\%$를 남긴다.” | token 개수 비율이 아니라 누적 확률질량 $0.9$에 도달하는 최소 후보 집합이다. |
| “top-$k$/top-$p$로 자른 값은 그대로 확률이다.” | 남은 질량으로 다시 나누어 합을 $1$로 만들어야 한다. |
| “greedy는 가장 확률 높은 문장 전체를 찾는다.” | 각 위치의 국소 argmax일 뿐이다. |
| “seed를 고정하면 어떤 서버에서도 동일하다.” | model·kernel·batching·provider가 달라지면 수치와 난수 소비 순서가 달라질 수 있다. |
| “SFT는 새 loss 함수를 발명한다.” | 보통 기존 causal cross-entropy에 chat-formatted data와 loss mask를 적용한다. |
| “response-only mask는 prompt를 삭제한다.” | prompt는 조건으로 읽힌다. 직접 label loss만 제외한다. |
| “prompt 위치의 gradient는 전부 $0$이다.” | prompt logit의 직접 loss는 $0$일 수 있지만, 뒤 assistant loss가 prompt 처리 경로로 역전파된다. |
| “all-token loss는 항상 잘못이다.” | 목적에 따라 domain adaptation과 혼합 objective에 유용할 수 있다. |
| “PEFT면 학습 compute가 거의 없다.” | trainable parameter와 optimizer state는 줄어도 frozen base의 forward·backward 경로와 adapter 연산 비용은 남는다. |
| “training template와 serving template가 달라도 글자만 같으면 된다.” | control token과 경계가 달라져 model이 전혀 다른 token sequence를 본다. |

### 4.8 실전 점검표

#### SFT 전

1. train·validation·test split 사이에 중복이나 benchmark leakage가 없는가?
2. role과 tool-call schema가 한 가지 규칙으로 정규화됐는가?
3. tokenizer와 chat template revision을 고정했는가?
4. 빈 assistant response, 깨진 Unicode, 지나치게 긴 예제를 처리했는가?
5. response-only mask가 실제 assistant token과 정확히 맞는지 눈으로 몇 개 확인했는가?
6. EOS·assistant end token을 label에 포함할지 정했는가?
7. loss 분모가 유효 assistant token 수인지 확인했는가?

#### 생성 평가 전

1. model·tokenizer·adapter revision을 기록했는가?
2. temperature, top-$k$, top-$p$, seed, max tokens, stop sequence를 기록했는가?
3. 평균 점수뿐 아니라 여러 run의 분산과 실패 유형을 보는가?
4. structured output은 실제 parser로 검증하는가?
5. 안전이 중요한 action은 model 출력만으로 실행하지 않고 권한·승인·rollback을 두는가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

오늘은 2026-09-15 공개 글 두 건을 고른다. 첫 번째는 “greedy여도 왜 agent가 매번 같은 행동을 하지 않는가?”를 측정한다. 두 번째는 instruction fine-tuning에서 학습할 작은 parameter 집합을 “저랭크 변화량”이 아니라 “독립적으로 움직이는 작은 shadow model”로 재설계한다.

### 5.1 ALTK-Evolve Consistency Analyzer: 평균 정확도 뒤의 반복 실행 불안정성

IBM Research 연구진은 2026-09-15 공식 기술 글에서 LLM agent를 같은 task에 여러 번 실행했을 때의 **consistency gap**과 이를 줄이는 ALTK-Evolve 기능을 공개했다. 연결된 기술 보고서는 2026-09-08 arXiv v1으로 제출됐고, 구현은 Apache-2.0 open-source repository에 반영됐다.

#### Mean@$k$, Pass@$k$, Pass$^k$는 서로 다른 질문이다

task $t$를 $k$번 실행한 성공 여부를

$$
X_t^{(j)}
\in
\{0,1\}
$$

라 하자.

평균 성공률은

$$
\operatorname{Mean@}k(t)
=
\frac{1}{k}
\sum_{j=1}^{k}
X_t^{(j)}.
$$

한 번이라도 성공했는지를 보는 낙관적 metric은

$$
\operatorname{Pass@}k(t)
=
\mathbf{1}
\left[
\sum_{j=1}^{k}X_t^{(j)}
\ge1
\right].
$$

반대로 모든 실행에서 성공했는지를 보는 metric은

$$
\operatorname{Pass}^{k}(t)
=
\mathbf{1}
\left[
\sum_{j=1}^{k}X_t^{(j)}
=k
\right].
$$

독립 Bernoulli 성공확률이 $p$라는 단순 모델에서는

$$
\mathbb{E}[\operatorname{Pass}^{k}]
=
p^k
\le
p
=
\mathbb{E}[\operatorname{Mean@}k]
\le
1-(1-p)^k
=
\mathbb{E}[\operatorname{Pass@}k].
$$

이름이 비슷하지만 Pass@$k$와 Pass$^k$는 거의 반대 질문이다.

연구진의 AppWorld test_normal 168-task 실험에서 ReAct/GPT-4.1 agent는

$$
\operatorname{Mean@5}
=
77.4\%
$$

였지만,

$$
\operatorname{Pass}^{5}
=
53.0\%
$$

였다. 연구진은 그 차이

$$
77.4-53.0
=
24.4
\text{ percentage points}
$$

를 consistency gap으로 부른다. percentage point를 줄여 $pp$라고 쓰며, 두 퍼센트 값의 **절대 차이**다.

#### temperature $0$도 완전한 재현성 보증서가 아니다

오늘 배운 greedy는 주어진 logits의 argmax를 고른다. 그러나 hosted endpoint의 logits 자체가 batching, floating-point reduction, platform 변화 등으로 아주 조금 달라질 수 있다. 1위와 2위가 거의 동률이면 작은 변화가 argmax를 뒤집을 수 있다.

연구의 Consistency Analyzer는 model 내부 logits 없이, 기록된 agent trajectory의 각 decision prompt를 다시 샘플링해 응답 변동을 측정한다. 논문의 평가 설정은 step마다 $N=30$, temperature $0.5$ resampling이며, tool name과 arguments 같은 categorical action은 pairwise Jaccard similarity로 비교한다. 반면 2026-09-16 현재 공개 구현의 기본 configuration은 max_samples $=5$, max_steps $=15$다. 논문의 보고 성능과 repository의 가벼운 실행 기본값을 같은 실험으로 읽으면 안 된다.

불안정한 step에서 “pagination을 끝까지 확인하라”, “검색 결과가 여러 개인지 검증하라” 같은 guideline을 만들고, 비슷한 후속 task의 prompt에 episodic memory로 주입한다. Jaccard similarity는 두 token 집합의 교집합 크기를 합집합 크기로 나눈

$$
J(A,B)
=
\frac{\lvert A\cap B\rvert}
{\lvert A\cup B\rvert}
$$

이며, 완전히 같으면 $1$, 겹침이 없으면 $0$이다.

연구진 보고에서 GPT-4.1의 같은-task 실험은

$$
\operatorname{Pass}^{5}:
53.0\%
\longrightarrow
69.0\%
$$

로 $16.0$ pp 상승했고, Mean@5도 $77.4\%\to81.0\%$로 낮아지지 않았다. sibling scenario의 similar-task 전이는 Pass$^5$가 $13.0$ pp 상승했다.

#### 엔지니어 인사이트 (Impact)

- **평균 정확도만으로 production reliability를 말할 수 없다.** retry가 허용되는 code generation은 Pass@$k$가 유용할 수 있지만, 금융 action이나 계약 확인처럼 매번 맞아야 하는 흐름은 Pass$^k$가 더 직접적이다.
- **$k$를 함께 써야 한다.** 같은 per-run 성공확률이라도 Pass$^k$는 $p^k$처럼 $k$가 커질수록 기계적으로 낮아진다. 서로 다른 $k$나 실행 조건의 값을 하나의 “신뢰성 점수”처럼 비교하면 안 된다.
- **decoding determinism과 model-distribution stability는 다르다.** greedy는 선택 규칙을 고정하지만, near-tie인 분포 자체가 작은 수치 변화에 민감한 문제는 해결하지 않는다.
- **stability는 correctness가 아니다.** Analyzer는 같은 행동을 하는지를 본다. 일관되게 틀린 action은 높은 consistency를 받을 수 있다.
- **연구 범위를 과장하면 안 된다.** 정량 평가는 AppWorld의 ReAct agent, GPT-4.1과 GPT-OSS-120B에 한정된다. similar-task도 같은 AppWorld scenario의 sibling variant다. 다른 agent stack과 넓은 OOD(out-of-distribution, 평가 조건이 기존 분포와 다른) domain에는 별도 검증이 필요하며, GPT-4.1 hard sibling에서는 Mean@5가 $0.8$ pp 낮아졌다.
- **비용이 크다.** 논문은 기본 $N=30$ resampling이 trajectory token 비용을 대략 $30$배로 만든다고 명시한다.
- **논문 설정과 repository 기본값을 섞지 말아야 한다.** 공개 구현의 sample budget은 설정으로 바꿀 수 있다. 더 작은 sample 수에서도 논문의 $+16$ pp가 그대로 재현된다고 가정하면 안 된다.
- **memory admission 검증이 아직 없다.** 현재 pipeline은 생성된 guideline을 바로 memory에 넣는다. 저자들은 guideline 생성에 쓰지 않고 따로 남겨 둔 held-out trajectory에서 개선·회귀를 검사하는 validation stage를 future work로 둔다.
- **platform noise를 직접 통제하지 못했다.** baseline과 guideline run의 non-determinism이 비교 가능하다고 가정했으며, 어려운 task에는 상당한 gap이 남았다.

**1차 출처:** [IBM Research·Hugging Face 공식 기술 글](https://huggingface.co/blog/ibm-research/altk-evolve-consistency) · [arXiv 기술 보고서](https://arxiv.org/abs/2609.08832) · [ALTK-Evolve 공식 GitHub](https://github.com/AgentToolkit/altk-evolve)

### 5.2 ShadowPEFT의 Hugging Face PEFT 통합: adapter를 작은 상태형 model로

Hugging Face는 2026-09-15 공개한 **PEFT v0.21.0 stable release**에 ShadowPEFT를 새 method로 포함했다. 같은 날 연구진의 통합 설명 글도 공개됐다. 논문은 2026-04-21 arXiv v1로 처음 제출되고 2026-09-11 v2로 개정됐으며, 이번 변화의 핵심은 연구 prototype이 널리 쓰이는 PEFT API의 first-class method가 됐다는 점이다.

따라서 오늘 시점에는 다음처럼 version을 명시해 설치할 수 있다.

~~~text
pip install "peft==0.21.0"
~~~

게시 글에는 “main에 merge됐고 다음 release에 포함될 예정”이라는 문구가 남아 있지만, 같은 날 뒤이어 나온 공식 v0.21.0 release note가 더 최신 상태다. 시시각각 바뀌는 library에서는 blog 문장보다 tagged release와 실제 설치 version을 함께 확인해야 한다.

#### LoRA와 무엇이 다른가?

LoRA는 선택한 frozen weight에 독립적인 저랭크 delta를 붙인다.

$$
\mathbf{h}'
=
\mathbf{W}\mathbf{h}
+
\alpha\mathbf{B}\mathbf{A}\mathbf{h}.
$$

ShadowPEFT는 layer depth를 따라 흐르는 shadow state

$$
\mathbf{s}^{(0)}
\rightarrow
\mathbf{s}^{(1)}
\rightarrow
\cdots
\rightarrow
\mathbf{s}^{(L)}
$$

를 유지한다. 각 Transformer block에서 개념적으로 세 단계가 일어난다.

1. shadow와 base hidden state의 차이를 저랭크 bottleneck으로 projection해 base 입력에 주입한다.
2. frozen base block이 수정된 hidden state를 처리한다.
3. base 출력으로 gated residual update를 만들어 shadow state를 다음 layer로 넘긴다.

$$
(\mathbf{h}^{(\ell)},\mathbf{s}^{(\ell)})
\longrightarrow
(\mathbf{h}^{(\ell+1)},\mathbf{s}^{(\ell+1)}).
$$

즉 task adaptation이 여러 weight에 흩어진 정적인 delta만이 아니라, 입력에 따라 layer 사이를 이동하는 작은 상태형 network가 된다.

#### 공개 비교에서 보인 trade-off

2026-09-15 글의 Llama-3.2-3B, MetaMathQA $\to$ GSM8K 비교는 다음과 같다.

| 방법 | trainable parameters | GSM8K exact match | train time | peak memory | checkpoint |
|---|---:|---:|---:|---:|---:|
| LoRA | $9.18$ M | $46.9\%$ | 15 min | 22.3 GB | 36.7 MB |
| DoRA | $9.29$ M | $46.2\%$ | 19 min | 24.5 GB | 37.2 MB |
| ShadowPEFT | $8.66$ M | $48.1\%$ | 17 min | 28.2 GB | 26.0 MB |

같은 A100 80 GB machine과 글에 연결된 default configuration에서 저자들이 보고한 수치다. ShadowPEFT는 이 설정에서 조금 높은 exact match와 작은 checkpoint를 얻었지만 peak memory는 LoRA보다

$$
28.2-22.3
=
5.9\text{ GB}
$$

더 컸다. shadow backbone의 forward와 base·shadow dual KV cache가 필요하기 때문이다.

#### 엔지니어 인사이트 (Impact)

- **parameter-efficient와 compute-free는 다르다.** trainable parameter 수와 checkpoint는 작아도 parallel shadow network 때문에 memory·latency가 늘 수 있다.
- **SFT label mask와 total objective를 구분하자.** 동일한 assistant-only label mask를 주 causal CE에 재사용할 수 있지만, ShadowPEFT는 초기 shadow state가 task를 독립적으로 풀도록 auxiliary shadow CE를 더한다.

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{base}}
+
\lambda
\mathcal{L}_{\mathrm{shadow}}.
$$

공식 ShadowConfig의 $\lambda$ 기본값은 $0.05$이고, 위 Llama-3.2-3B 비교에 연결된 configuration은 $0.01$을 썼다. 따라서 label mask는 공유할 수 있어도 total loss가 LoRA와 완전히 같다고 말할 수는 없다.
- **detached deployment는 가능성이지 완성된 routing system이 아니다.** language model의 unload_shadow()는 head(projection(backbone(x)))로 이루어진 독립 shadow model을 반환하며, base 출력이 필요한 layer별 inject·update 경로는 standalone에 남지 않는다. 외부 router를 따로 만들면 단순 요청은 edge shadow, 어려운 요청은 attached base로 보낼 수 있지만 PEFT가 그 router나 품질을 제공하지는 않는다. “분리 가능”은 “분리한 model이 충분히 정확함”을 보장하지 않으며, 공식 release note도 작은 pretrained model에서 shadow를 시작할 때 가장 잘 작동한다고 명시한다.
- **LoRA처럼 base weight에 merge할 수 없다.** input-dependent shadow trajectory이므로 merge·merge-and-unload는 명시적으로 지원되지 않는다.
- **운영 제한을 확인해야 한다.** 한 번에 하나의 Shadow adapter만 활성화할 수 있고, Diffusers model은 standalone unload가 지원되지 않는다. 현재 Flux2만 architecture-aware backend로 등록돼 있고, 그 밖의 호환 transformer-based Diffusers는 generic token-wise residual MLP fallback을 쓴다.
- **stable도 version pin이 필요하다.** v0.21.0에 정식 포함됐지만 main branch는 이후 계속 변한다. 재현 실험은 PEFT·Transformers·PyTorch version과 model revision을 함께 pin해야 한다.
- **method code와 model 사용권은 별개다.** PEFT code는 Apache-2.0이고 ShadowPEFT 논문은 CC BY 4.0이지만, 실제로 붙이는 base model·training data·배포 산출물의 license는 따로 확인해야 한다.
- **한 비교표를 보편적 우위로 읽으면 안 된다.** 저자·통합 글의 제한된 model, dataset, hyperparameter 결과이며 반복횟수와 오차막대도 보고되지 않았다. 자신의 instruction dataset에서 quality, memory, latency, adapter switching을 함께 benchmark해야 한다.

**1차 출처:** [PEFT v0.21.0 공식 release note](https://github.com/huggingface/peft/releases/tag/v0.21.0) · [ShadowPEFT 통합 발표](https://huggingface.co/blog/shadow-llm/shadowpeft-peft) · [Hugging Face PEFT 공식 문서](https://huggingface.co/docs/peft/main/en/package_reference/shadow) · [PEFT 공식 GitHub](https://github.com/huggingface/peft) · [ShadowPEFT 논문](https://arxiv.org/abs/2604.19254)

### 5.3 두 흐름을 함께 읽기: 좋은 출력은 세 층에서 결정된다

오늘 배운 내용과 두 trend를 합치면 LLM system의 행동은 세 층으로 나뉜다.

$$
\boxed{
\begin{array}{c}
\textbf{Training layer}\\
\text{SFT data·loss mask·full/PEFT가 }\theta\text{를 만든다}\\
\downarrow\\
\textbf{Decoding layer}\\
\tau\text{·top-}k\text{·top-}p\text{가 token 선택을 만든다}\\
\downarrow\\
\textbf{Agent system layer}\\
\text{memory·tool validation·retry·permissions가 trajectory를 만든다}
\end{array}
}
$$

한 층의 knob로 다른 층의 결함을 모두 고칠 수 없다.

- 틀린 training data는 temperature만으로 고칠 수 없다.
- flat한 near-tie distribution은 seed 하나만으로 production reliability가 보장되지 않는다.
- 좋은 SFT와 decoding도 tool permission·validation이 없으면 안전한 agent를 보장하지 않는다.
- 작은 adapter checkpoint도 실제 serving memory·KV cache·latency가 작다는 뜻은 아니다.

따라서 model 평가표에는 최소한 다음을 함께 기록해야 한다.

1. 어떤 model·adapter·chat template·loss mask를 썼는가?
2. 어떤 decoding configuration을 썼는가?
3. 한 번의 평균 점수뿐 아니라 반복 run의 분산과 Pass$^k$는 어떤가?
4. adapter의 trainable parameter뿐 아니라 peak memory·latency·cache 비용은 어떤가?
5. tool action의 정확성과 안정성을 별도 validator가 확인하는가?

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

어떤 instruction-tuned model이 다음 token에 대해 logits

$$
\mathbf{z}
=
(2,\;1,\;0)
$$

을 냈다. 세 후보를 각각 $A,B,C$라 하자.

동시에 SFT 예제의 token sequence가

$$
\mathbf{s}
=
[
\mathrm{SYS},
\mathrm{USER}_1,
\mathrm{USER}_2,
\mathrm{AST}_1,
\mathrm{AST}_2,
\mathrm{EOS}
]
$$

이고, $\mathrm{AST}_1,\mathrm{AST}_2,\mathrm{EOS}$만 직접 채점한다고 하자.

**하나의 핵심 질문:** “학습된 logits가 실제 출력이 되고, 그 logits를 만든 model이 response-only SFT로 update되는 과정”을 다음 항목으로 설명하라.

1. $\tau=1$과 $\tau=0.5$일 때의 확률을 계산하고 어느 쪽 entropy가 큰지 설명하라.
2. $\tau=1$ 분포에 top-$p=0.9$를 적용했을 때 남는 후보와 재정규화된 확률을 구하라.
3. $U=0.80$을 뽑았다면 top-$p$ 분포에서 어떤 token이 선택되는가?
4. greedy가 이 위치의 최고 후보를 고르더라도 전체 수열 최대확률을 보장하지 않는 이유를 말하라.
5. response-only mask $\mathbf{m}$과 SFT loss를 쓰고, 왜 prompt token도 학습에 간접 영향을 주는지 설명하라.
6. temperature를 낮추는 것과 SFT를 더 하는 것이 왜 같은 조작이 아닌지 설명하라.

### 모범 답안

#### 1. temperature에 따른 확률

$\tau=1$이면

$$
\begin{aligned}
\mathbf{q}^{(1)}
&=
\frac{(e^2,e^1,e^0)}
{e^2+e^1+e^0}\\
&\approx
(0.665,\;0.245,\;0.090).
\end{aligned}
$$

$\tau=0.5$이면 logits를 $0.5$로 나누므로

$$
\frac{\mathbf{z}}{0.5}
=
(4,\;2,\;0)
$$

이고,

$$
\begin{aligned}
\mathbf{q}^{(0.5)}
&=
\frac{(e^4,e^2,e^0)}
{e^4+e^2+e^0}\\
&\approx
(0.867,\;0.117,\;0.016).
\end{aligned}
$$

$\tau=1$ 분포가 세 후보에 더 퍼져 있으므로 entropy가 더 크다. 일반적으로

$$
\frac{dH}{d\tau}
=
\frac{\operatorname{Var}_{q^{(\tau)}}(z)}
{\tau^3}
\ge0
$$

이다.

#### 2. top-$p=0.9$와 재정규화

$\tau=1$에서 정렬된 누적확률은

$$
0.665<0.9,
\qquad
0.665+0.245
\approx
0.910
\ge0.9.
$$

따라서 최소 nucleus는

$$
\mathcal{S}_{0.9}
=
\{A,B\}
$$

다. 남은 질량으로 나누면

$$
\widetilde q_A
=
\frac{0.665}{0.665+0.245}
\approx
0.731,
$$

$$
\widetilde q_B
=
\frac{0.245}{0.665+0.245}
\approx
0.269,
$$

$$
\widetilde q_C=0.
$$

따라서

$$
\widetilde{\mathbf{q}}
\approx
(0.731,\;0.269,\;0).
$$

#### 3. inverse-CDF sampling

누적 구간은

$$
A:[0,0.731),
\qquad
B:[0.731,1).
$$

$U=0.80$은 두 번째 구간에 있으므로 $B$를 선택한다. $A$의 원래 확률이 가장 컸어도 sampling에서는 $B$가 나올 수 있다.

#### 4. greedy의 국소성

greedy는 현재 위치에서

$$
\arg\max_v p(v\mid x_{\le t})
$$

만 고른다. 전체 수열 확률은 이후 조건부확률까지 곱한

$$
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
$$

이다. 현재 확률이 조금 높은 prefix가 이후 매우 낮은 확률만 가질 수 있으므로, 국소 최대가 전체 곱의 최대를 보장하지 않는다.

#### 5. response-only mask와 gradient 경로

target token 기준 mask는

$$
\mathbf{m}
=
(0,\;0,\;0,\;1,\;1,\;1)
$$

로 둘 수 있다. 첫 token은 이전 문맥이 없어 loss 합에서는 보통 $t=2$부터 사용한다.

유효 token 수는

$$
K=3
$$

이고,

$$
\mathcal{L}_{\mathrm{SFT}}
=
-
\frac{1}{3}
\left[
\log p_{\theta}(\mathrm{AST}_1\mid s_{<4})
+
\log p_{\theta}(\mathrm{AST}_2\mid s_{<5})
+
\log p_{\theta}(\mathrm{EOS}\mid s_{<6})
\right].
$$

SYS와 USER token 자체에는 직접 target loss를 주지 않지만, assistant token을 예측하는 hidden state가 앞선 prompt를 attention한다. 따라서 응답 loss의 gradient가 prompt embedding과 prompt를 처리한 attention·FFN parameter까지 역전파된다.

#### 6. parameter 변경과 출력 정책 변경

temperature는 고정된 parameter $\theta$가 만든 logits를 나누어 inference distribution을 바꾼다.

$$
\mathbf{z}_{\theta}
\longrightarrow
\frac{\mathbf{z}_{\theta}}{\tau}.
$$

SFT는 gradient descent로 parameter 자체를 바꾼다.

$$
\theta
\longrightarrow
\theta'
=
\theta-\eta\nabla_{\theta}\mathcal{L}_{\mathrm{SFT}}.
$$

그러므로 낮은 temperature는 instruction-following 능력을 새로 학습시키지 않으며, SFT는 sampling randomness를 직접 정하는 knob가 아니다.

---

**다음 연결 고리:** 오늘은 모범 응답을 그대로 따라 배우는 SFT와 실제 token을 고르는 decoding을 분리했다. 다음에는 “여러 응답 중 사람이 어느 쪽을 더 선호하는가?”를 학습하기 위해 KL divergence·log-odds·preference pair를 사용하고, DPO와 RLHF가 SFT 다음 단계에서 무엇을 바꾸는지 연결한다.
