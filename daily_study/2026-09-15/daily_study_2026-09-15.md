# [2026-09-15] 오늘 학습: 조건부확률의 연쇄법칙·음의 로그가능도 & 토큰화·인과 언어 모델 학습

> **오늘의 핵심 문장:** 인과 언어 모델은 텍스트를 토큰 ID의 수열로 바꾼 뒤, 정답 과거 토큰만 조건으로 다음 토큰의 확률을 동시에 계산하고, 실제 다음 토큰에 준 확률의 음의 로그를 줄이도록 학습한다.

지난 학습에서는 입력

$$
\mathbf{X}\in\mathbb{R}^{B\times T\times d}
$$

가 multi-head attention, 위치 표현, 잔차 연결, LayerNorm, FFN을 통과해 같은 shape의 문맥 표현으로 바뀌는 과정을 조립했다. 그러나 아직 가장 중요한 질문이 남았다.

> Transformer는 무엇을 정답으로 삼아 가중치를 배우는가?

오늘은 원문 문자열부터 손실 하나가 나오기까지의 전체 계약을 연결한다.

$$
\boxed{
\text{raw text}
\xrightarrow{\text{tokenizer}}
\text{token IDs}
\xrightarrow{\text{shift}}
\text{inputs·targets}
\xrightarrow{\text{causal Transformer}}
\text{logits}
\xrightarrow{\text{Softmax·NLL}}
\text{loss}
}
$$

수학 트랙에서는 조건부확률의 연쇄법칙, 최대가능도추정, 로그, 음의 로그가능도를 유도한다. AI 기초 트랙에서는 subword 토큰화, teacher forcing, causal mask, 한 칸 이동한 정답, padding을 제외한 causal cross-entropy를 하나의 계산 그래프로 묶는다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 컴퓨터는 왜 문장을 그대로 읽지 못할까?

사람에게 “나는 인공지능을 배운다”는 의미 있는 문장이다. 컴퓨터의 신경망이 직접 다루는 것은 실수 배열이다. 따라서 문자열을 유한한 기호의 수열로 바꾸는 규칙이 먼저 필요하다.

가장 단순한 선택은 문장에 나온 단어마다 번호를 붙이는 것이다.

$$
\text{나는}\mapsto 17,\qquad
\text{인공지능을}\mapsto 842,\qquad
\text{배운다}\mapsto 91
$$

하지만 세상의 모든 단어와 활용형을 미리 사전에 넣을 수는 없다. 한국어의 “배운다·배웠다·배우면서”, 영어의 “learn·learned·learning”, 새 제품명, 오타, 이모지까지 모두 별도 단어로 두면 사전이 끝없이 커진다. 반대로 글자나 byte 하나씩 자르면 어떤 문자열도 표현하기 쉽지만 수열이 너무 길어지고, 모델이 자주 등장하는 의미 단위를 매번 다시 조립해야 한다.

이 두 극단 사이에서 나온 타협이 **부분단어(subword) 토큰화**다. 자주 함께 등장하는 조각은 하나로 묶고, 드문 단어는 더 작은 조각으로 나눈다.

$$
\text{token}
\neq
\text{word}
\neq
\text{character}
\neq
\text{byte}
$$

토큰은 자연이 정해 둔 단위가 아니라, tokenizer가 만든 계산 단위다.

### 1.2 왜 “문장 전체를 이해하라” 대신 “다음 토큰을 맞혀라”라고 할까?

“언어를 이해하라”는 목표는 채점 방법이 모호하다. 반면 문장의 앞부분이 주어졌을 때 실제 다음 토큰에 얼마나 높은 확률을 주었는지는 숫자로 측정할 수 있다.

예를 들어

$$
\text{“오늘 하늘은”}
$$

뒤에 올 토큰의 분포를 모델이

$$
p(\text{맑다})=0.55,\quad
p(\text{흐리다})=0.25,\quad
p(\text{먹었다})=0.001,\quad\ldots
$$

처럼 내놓는다고 하자. 실제 정답이 “맑다”였다면 $0.55$를 더 크게 만드는 방향으로 학습할 수 있다.

하나의 거대한 정답 문장을 직접 분류하는 대신, 모든 위치에서 다음 토큰 예측 문제를 만든다. 길이 $T$의 수열 하나가 거의 $T$개의 학습 신호를 제공하므로 별도 사람이 라벨을 붙이지 않은 방대한 텍스트도 학습 데이터가 된다. 이 성질 때문에 다음 토큰 예측은 **자기지도학습(self-supervised learning)** 목표다.

### 1.3 왜 과거만 보고 미래를 맞혀야 할까?

위치 $t$의 정답 $x_t$를 예측하면서 모델이 이미 $x_t$나 그 뒤의 토큰을 볼 수 있다면 시험지를 펼쳐 놓고 답을 베끼는 셈이다. 학습 손실은 낮아져도 실제 생성에서는 미래 정답이 없으므로 사용할 수 없다.

그래서 인과 언어 모델은

$$
p_{\theta}(x_t\mid x_{<t})
$$

만 계산해야 한다. 여기서

$$
x_{<t}=(x_1,\ldots,x_{t-1})
$$

는 위치 $t$보다 앞선 토큰들이다. **causal**은 “현실의 철학적 인과관계를 발견했다”는 뜻이 아니라, 계산 그래프에서 미래 위치로부터 정보가 흘러오지 못하게 했다는 뜻이다.

### 1.4 왜 학습할 때는 모델이 방금 틀린 토큰 대신 정답 토큰을 계속 넣을까?

모델이 아직 서툰 초기 학습 단계에서 자기 예측만 다음 입력으로 쓰면 첫 실수 뒤의 문맥이 빠르게 무너진다. 그러면 뒤 위치의 손실은 “올바른 과거에서 다음 토큰을 예측하는 법”보다 “우연히 망가진 문맥을 수습하는 법”을 주로 가르칠 수 있다.

**Teacher forcing**은 학습 위치마다 실제 정답 과거 토큰을 입력으로 제공하는 방식이다.

$$
\text{입력 문맥}
=
(x_1,\ldots,x_{t-1})
$$

덕분에 각 위치는 깨끗한 정답 문맥에서 다음 토큰을 배우며, causal mask와 함께 쓰면 한 시퀀스의 모든 위치를 GPU에서 병렬 계산할 수 있다.

다만 추론 때는 다음 정답을 모른다. 모델이 생성한 토큰을 다시 입력해야 한다. 이 학습·추론 조건 차이가 **exposure bias**의 출발점이다.

### 1.5 왜 확률을 그대로 곱하지 않고 로그를 취할까?

문장 확률은 여러 조건부확률의 곱이다. 각 확률이 $1$보다 작으므로 긴 문장을 곱할수록 값이 극도로 작아진다.

$$
0.1^{100}=10^{-100}
$$

유한 정밀도 컴퓨터에서는 이런 수가 $0$에 가깝게 뭉개지는 underflow가 생길 수 있다. 로그를 취하면 곱이 합으로 바뀐다.

$$
\log\left(\prod_{t=1}^{T}p_t\right)
=
\sum_{t=1}^{T}\log p_t
$$

합은 계산하기 안정적이고, 각 토큰이 전체 점수에 얼마를 기여했는지도 분리해 볼 수 있다. 확률을 크게 만들고 싶으므로 로그확률을 최대화하거나, 같은 뜻으로 **음의 로그가능도**를 최소화한다.

### 1.6 오늘의 두 트랙은 어디에서 만날까?

오늘의 수학과 AI 부품은 다음처럼 정확히 대응한다.

$$
\boxed{
\begin{array}{c}
\text{조건부확률의 연쇄법칙}
\longleftrightarrow
\text{왼쪽에서 오른쪽으로 문장 확률 분해}\\
\text{유한 집합 위 확률분포}
\longleftrightarrow
\text{어휘 전체에 대한 Softmax}\\
\text{최대가능도추정}
\longleftrightarrow
\text{관측된 다음 토큰의 확률을 키우기}\\
\text{음의 로그}
\longleftrightarrow
\text{곱을 안정적인 토큰별 loss 합으로 변환}\\
\text{지시함수·마스크}
\longleftrightarrow
\text{padding이 아닌 정답만 평균}\\
\text{연쇄법칙·그래디언트}
\longleftrightarrow
\text{오차역전파로 Transformer 가중치 갱신}
\end{array}
}
$$

---

## 2. 친절한 용어 사전

### 2.1 문자열과 토큰화 용어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| corpus | $\mathcal{C}$ | tokenizer나 언어 모델을 학습하기 위해 모은 문서 집합이다. “말뭉치”라고도 한다. |
| vocabulary | $\mathcal{V}$ | tokenizer가 사용할 수 있는 토큰 종류의 유한한 목록이다. 크기는 $\lvert\mathcal{V}\rvert$로 쓴다. |
| tokenizer | $\tau$ | 문자열을 토큰 ID 수열로 바꾸고, 가능하면 다시 문자열로 복원하는 규칙과 프로그램이다. |
| token | $x_t$ | tokenizer가 정한 한 계산 단위다. 단어 전체일 수도, 단어 조각·문장부호·byte 조각일 수도 있다. |
| token ID | $x_t\in\{0,\ldots,\lvert\mathcal{V}\rvert-1\}$ | 토큰을 어휘표의 행 번호로 표현한 정수다. ID의 숫자 크기 자체에는 의미가 없다. |
| normalization |  | Unicode 표현, 대소문자, 공백 등을 정해진 규칙으로 정리하는 전처리다. 다른 문자열이 합쳐질 수 있으므로 모델과 함께 버전을 고정해야 한다. |
| Unicode code point |  | 문자를 표준 번호로 표현한 단위다. 화면의 글자 하나와 항상 일치하지는 않는다. |
| byte |  | 파일과 메모리에서 데이터를 표현하는 $8$ bit 단위다. UTF-8의 한글 한 글자는 보통 여러 byte를 쓴다. |
| pre-tokenization |  | BPE merge 전에 공백·문장부호 등의 경계를 표시하거나 1차로 나누는 단계다. |
| BPE | byte pair encoding | 현재 기호 수열에서 자주 붙는 인접 쌍을 반복해서 하나의 새 기호로 병합하는 방식이다. 현대 구현은 byte-level 변형 등을 많이 쓴다. |
| unigram tokenizer |  | 후보 subword 사전에서 문장 전체의 확률이 높아지도록 분할을 선택하는 계열이다. 이름이 “단어 하나”라는 뜻은 아니다. |
| byte fallback |  | 사전에 없는 문자열도 원래 byte 조각으로 내려가 표현하게 하는 안전망이다. |
| encode / decode | $\tau,\delta$ | encode 함수 $\tau$는 문자열을 ID로, decode 함수 $\delta$는 ID를 문자열로 바꾼다. normalization이 정보를 합칠 수 있어 $\delta$가 항상 진짜 역함수 $\tau^{-1}$인 것은 아니다. 모델의 Transformer decoder와 tokenizer decode도 다른 말이다. |

### 2.2 특수 토큰

| 토큰 | 뜻 | 주의점 |
|---|---|---|
| $\langle\mathrm{BOS}\rangle$ | begin of sequence. 수열 시작을 표시한다. | 모든 모델이 명시적 BOS를 쓰는 것은 아니다. |
| $\langle\mathrm{EOS}\rangle$ | end of sequence. 생성 종료를 학습시키는 토큰이다. | EOS를 target에 넣어야 종료 확률도 학습된다. |
| $\langle\mathrm{PAD}\rangle$ | 길이가 다른 수열을 batch로 맞추기 위한 빈 자리다. | 보통 loss에서 제외하며, Key로도 읽지 못하게 attention mask를 둔다. |
| $\langle\mathrm{UNK}\rangle$ | unknown. tokenizer가 표현하지 못한 입력을 대신한다. | byte fallback tokenizer는 UNK 필요성을 크게 줄일 수 있다. |
| chat control token |  | system·user·assistant 경계나 tool call 구조를 나타내는 모델별 특수 토큰이다. 임의 문자열과 같은 방식으로 취급하면 안 된다. |

### 2.3 확률과 학습 목표 용어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 조건부확률 | $p(A\mid B)$ | $B$가 이미 주어졌을 때 $A$가 일어날 확률이다. |
| 확률의 연쇄법칙 | chain rule | 결합확률을 앞선 사건을 조건으로 한 확률들의 곱으로 푸는 항등식이다. 미분의 연쇄법칙과 이름만 같고 다른 정리다. |
| autoregressive |  | 이전에 나온 값을 조건으로 다음 값을 하나씩 모델링하는 방식이다. |
| parameter | $\boldsymbol{\theta}$ | embedding, attention, FFN, LM head 등 학습되는 모든 가중치의 모음이다. |
| likelihood | $p_{\theta}(\mathcal{D})$ | 고정된 관측 데이터 $\mathcal{D}$가 현재 파라미터 아래 얼마나 그럴듯한지를 파라미터의 함수로 본 값이다. |
| maximum likelihood estimation | MLE | 관측 데이터의 likelihood를 가장 크게 만드는 파라미터를 찾는 원리다. |
| log-likelihood | $\log p_{\theta}(\mathcal{D})$ | 작은 확률의 곱을 합으로 바꾼 값이다. |
| negative log-likelihood | NLL | $-\log p_{\theta}(\mathcal{D})$다. 최소화 문제로 바꾸기 위해 음수를 붙인다. |
| one-hot target | $\mathbf{y}_t$ | 정답 토큰 좌표만 $1$, 나머지는 $0$인 $\lvert\mathcal{V}\rvert$차원 벡터다. |
| cross-entropy | $H(\mathbf{y},\mathbf{p})$ | 정답 분포와 예측 분포의 차이를 정답 로그확률로 측정하는 손실이다. one-hot 정답이면 NLL과 같다. |
| perplexity | $\operatorname{PPL}$ | 평균 NLL을 다시 지수화한 값이다. 같은 평가 규약에서 낮을수록 정답 토큰에 높은 확률을 줬다는 뜻이다. |

### 2.4 Transformer 학습 용어와 shape

| 용어·기호 | shape | 초보자 해설 |
|---|---:|---|
| input IDs | $\mathbf{X}$: $B\times T$ | Transformer에 넣는 정수 토큰 ID다. |
| targets / labels | $\mathbf{Y}$: $B\times T$ | 각 입력 위치가 예측해야 할 실제 다음 토큰 ID다. |
| hidden states | $\mathbf{H}$: $B\times T\times d$ | Transformer 마지막 층이 만든 문맥 벡터다. |
| LM head | $\mathbf{W}_{\mathrm{U}}$: $d\times\lvert\mathcal{V}\rvert$ | 각 문맥 벡터를 어휘 크기의 점수로 펼치는 선형층이다. unembedding이라고도 한다. |
| logits | $\mathbf{Z}$: $B\times T\times\lvert\mathcal{V}\rvert$ | Softmax 전의 제한 없는 실수 점수다. 확률이 아니다. |
| probabilities | $\mathbf{P}$: $B\times T\times\lvert\mathcal{V}\rvert$ | 어휘 축으로 Softmax해 합이 $1$이 된 다음 토큰 분포다. |
| causal mask |  | 위치 $i$가 미래 Key $j>i$를 읽지 못하게 하는 attention용 마스크다. |
| attention mask |  | 보통 padding 같은 읽으면 안 되는 입력 위치를 가리는 마스크다. 라이브러리마다 이름과 의미가 다르다. |
| loss mask | $\mathbf{M}$: $B\times T$ | 어느 target 위치를 loss 평균에 포함할지 정하는 $0/1$ 지시자다. |
| teacher forcing |  | 학습할 때 이전 모델 예측이 아니라 실제 이전 토큰을 입력으로 주는 방식이다. |
| exposure bias |  | 학습에서는 정답 과거를 보지만 생성에서는 자기 출력을 다시 보는 조건 차이다. |

> **같은 “decode”를 구분하자:** tokenizer decode는 ID를 문자열로 되돌리는 과정이다. autoregressive decoding은 다음 토큰을 선택하고 수열에 덧붙이는 생성 과정이다.

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 토큰화는 유한한 확률 문제를 만든다

tokenizer를 함수

$$
\tau:
\text{strings}
\longrightarrow
\bigcup_{T\ge 0}\mathcal{V}^{T}
$$

로 생각하자. 문자열 $s$를 넣으면

$$
\tau(s)
=
(x_1,\ldots,x_T),
\qquad
x_t\in\mathcal{V}
$$

가 나온다. 어휘에 번호를 붙였다면 각 $x_t$는

$$
x_t\in\{0,1,\ldots,|\mathcal{V}|-1\}
$$

인 정수 ID다.

이 단계에는 아직 의미 벡터가 없다. ID $900$이 ID $100$보다 아홉 배 큰 의미를 갖는 것도 아니다. 지난 학습의 embedding table

$$
\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d}
$$

에서 ID는 단지 어느 행

$$
\mathbf{e}_{x_t}
=
\mathbf{E}[x_t,:]
\in\mathbb{R}^{d}
$$

을 가져올지 정하는 주소다.

tokenizer가 어휘 $\mathcal{V}$를 정하면 다음 토큰 예측은 $|\mathcal{V}|$개 범주 중 하나를 고르는 유한한 다중분류 문제가 된다.

### 3.2 BPE 병합을 작은 예제로 해부하기

아주 단순화한 corpus가 다음 세 단어로만 이루어진 **multiset**이라고 하자. multiset은 같은 원소가 몇 번 나왔는지를 보존하는 집합형 자료다.

$$
\mathcal{C}_{\mathrm{multi}}
=
\left\{\!\left\{
\text{low},\text{low},\text{lower}
\right\}\!\right\}
$$

단어 끝 기호를 $\langle/\mathrm{w}\rangle$라 하고 처음에는 문자 단위로 나눈다.

$$
\begin{aligned}
\text{low}
&:
l\;o\;w\;\langle/\mathrm{w}\rangle
\quad\text{(두 번)},\\
\text{lower}
&:
l\;o\;w\;e\;r\;\langle/\mathrm{w}\rangle.
\end{aligned}
$$

인접 쌍 $(l,o)$와 $(o,w)$는 각각 세 번 나온다. tie-breaking 규칙으로 $(l,o)$를 먼저 고른다고 하자.

$$
(l,o)\longrightarrow lo
$$

수열은

$$
lo\;w\;\langle/\mathrm{w}\rangle
\quad\text{(두 번)},
\qquad
lo\;w\;e\;r\;\langle/\mathrm{w}\rangle
$$

로 바뀐다. 이제 $(lo,w)$가 세 번 나오므로

$$
(lo,w)\longrightarrow low
$$

를 새 토큰으로 만들 수 있다. 충분히 자주 등장한 “low”는 한 단위가 되고, 드문 “lower”는

$$
low\;e\;r\;\langle/\mathrm{w}\rangle
$$

처럼 더 작은 조각으로 남는다.

일반화하면 매 단계 $k$에서 현재 기호 수열 전체의 인접 쌍 빈도

$$
c_k(a,b)
=
\sum_{\text{all adjacent positions}}
\mathbf{1}
\left[
(s_i,s_{i+1})=(a,b)
\right]
$$

를 세고,

$$
(a_k^{*},b_k^{*})
\in
\operatorname*{arg\,max}_{(a,b)}
c_k(a,b)
$$

를 병합한다. $\mathbf{1}[\cdot]$는 조건이 참이면 $1$, 거짓이면 $0$인 지시함수다.

실제 tokenizer는 byte-level 초기 기호, 공백 표식, Unicode normalization, 최소 빈도, 특수 토큰, tie-breaking 같은 세부 규칙을 함께 갖는다. 따라서 “BPE를 쓴다”만으로 같은 ID 수열이 보장되지 않는다. **어휘 파일과 merge 순서, normalization 설정 전체가 모델 계약**이다.

### 3.3 두 사건의 곱셈법칙에서 수열의 연쇄법칙까지

조건부확률 정의는

$$
p(B\mid A)
=
\frac{p(A,B)}{p(A)}
$$

이다. $p(A)>0$일 때 양변에 $p(A)$를 곱하면

$$
p(A,B)
=
p(A)p(B\mid A)
$$

를 얻는다.

세 토큰에는 이 식을 한 번 더 적용한다.

$$
\begin{aligned}
p(x_1,x_2,x_3)
&=
p(x_1,x_2)
p(x_3\mid x_1,x_2)\\
&=
p(x_1)
p(x_2\mid x_1)
p(x_3\mid x_1,x_2).
\end{aligned}
$$

이를 $T$개 토큰으로 반복하면

$$
\boxed{
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
}
$$

이다. 여기서

$$
x_{1:T}=(x_1,\ldots,x_T)
$$

이고 $t=1$에서는 빈 문맥이나 $\langle\mathrm{BOS}\rangle$를 조건으로 생각할 수 있다.

이 식은 데이터에 대한 가정이 아니라 모든 결합확률분포에 성립하는 항등식이다. 인과 언어 모델의 설계 선택은 각 조건부확률을 **왼쪽 문맥만 받는 하나의 신경망**으로 근사하는 데 있다.

$$
p(x_{1:T})
\approx
p_{\theta}(x_{1:T})
=
\prod_{t=1}^{T}
p_{\theta}(x_t\mid x_{<t})
$$

### 3.4 logits에서 다음 토큰 확률분포 만들기

3.3에서는 $t$를 **예측 대상 token의 위치**로 세어 $p(x_t\mid x_{<t})$라고 썼다. 이 절에서는 구현의 한 칸 shift를 분명히 하려고 입력 위치를 새 문자 $i$로 센다. 따라서 입력 위치 $i$의 출력은 다음 위치 $i+1$의 token을 예측한다. 두 표기는 같은 연쇄법칙을 한 칸 다르게 색인한 것이다.

입력 위치 $i$의 마지막 hidden state를

$$
\mathbf{h}_i\in\mathbb{R}^{d}
$$

라 하자. LM head는

$$
\mathbf{z}_i
=
\mathbf{h}_i\mathbf{W}_{\mathrm{U}}
+\mathbf{b}_{\mathrm{U}}
\in\mathbb{R}^{|\mathcal{V}|}
$$

인 logits를 만든다. 토큰 $v\in\mathcal{V}$의 확률은

$$
p_{\theta}(x_{i+1}=v\mid x_{\le i})
=
\frac{\exp(z_{i,v})}
{\sum_{u\in\mathcal{V}}\exp(z_{i,u})}
$$

이다. 분자가 양수이고 분모가 모든 어휘의 분자 합이므로

$$
p_{\theta}(v\mid x_{\le i})>0,
\qquad
\sum_{v\in\mathcal{V}}
p_{\theta}(v\mid x_{\le i})
=1
$$

을 만족한다.

수치적으로는 가장 큰 logit

$$
m_i=\max_{u}z_{i,u}
$$

를 모두에서 빼도 확률이 변하지 않는다.

$$
\frac{\exp(z_{i,v}-m_i)}
{\sum_u\exp(z_{i,u}-m_i)}
=
\frac{\exp(z_{i,v})}
{\sum_u\exp(z_{i,u})}
$$

분자와 분모에 같은 $\exp(-m_i)$가 곱해져 약분되기 때문이다. 이 **log-sum-exp trick**은 큰 지수값의 overflow를 줄인다.

### 3.5 최대가능도에서 음의 로그가능도까지

학습 데이터가 $N$개의 토큰 수열

$$
\mathcal{D}
=
\left\{
x_{1:T_n}^{(n)}
\right\}_{n=1}^{N}
$$

이라고 하자. 표본들이 독립적으로 수집되었다고 모델링하면 전체 likelihood는

$$
\mathcal{L}_{\mathrm{like}}(\boldsymbol{\theta})
=
\prod_{n=1}^{N}
p_{\theta}
\left(
x_{1:T_n}^{(n)}
\right)
$$

이다. 확률의 연쇄법칙을 대입하면

$$
\mathcal{L}_{\mathrm{like}}(\boldsymbol{\theta})
=
\prod_{n=1}^{N}
\prod_{t=1}^{T_n}
p_{\theta}
\left(
x_t^{(n)}
\mid
x_{<t}^{(n)}
\right).
$$

MLE는 이 값을 최대화한다.

$$
\hat{\boldsymbol{\theta}}_{\mathrm{MLE}}
\in
\operatorname*{arg\,max}_{\boldsymbol{\theta}}
\mathcal{L}_{\mathrm{like}}(\boldsymbol{\theta})
$$

로그는 단조증가 함수이므로 최대점을 바꾸지 않는다.

$$
\operatorname*{arg\,max}_{\boldsymbol{\theta}}
\mathcal{L}_{\mathrm{like}}
=
\operatorname*{arg\,max}_{\boldsymbol{\theta}}
\log\mathcal{L}_{\mathrm{like}}
$$

그리고 곱은 합으로 변한다.

$$
\log\mathcal{L}_{\mathrm{like}}(\boldsymbol{\theta})
=
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\log
p_{\theta}
\left(
x_t^{(n)}
\mid
x_{<t}^{(n)}
\right).
$$

최적화 관례에 맞게 음수를 붙여 최소화하면 총 NLL이다.

$$
\boxed{
\mathcal{J}_{\mathrm{NLL}}(\boldsymbol{\theta})
=
-
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\log
p_{\theta}
\left(
x_t^{(n)}
\mid
x_{<t}^{(n)}
\right)
}
$$

양의 상수로 나눈 평균 NLL도 같은 데이터에서 같은 최소점을 갖는다. 다만 보고되는 loss의 크기와 gradient scale은 달라지므로 분모가 “문장 수인지, batch 수인지, 유효 토큰 수인지” 명시해야 한다.

### 3.6 one-hot cross-entropy와 NLL은 왜 같은가?

위치 $t$의 정답 토큰 ID를 $y_t$라 하자. one-hot 벡터는

$$
y_{t,v}
=
\begin{cases}
1, & v=y_t,\\
0, & v\neq y_t
\end{cases}
$$

이다. 정답 분포 $\mathbf{y}_t$와 예측 분포 $\mathbf{p}_t$의 cross-entropy는

$$
\begin{aligned}
H(\mathbf{y}_t,\mathbf{p}_t)
&=
-
\sum_{v\in\mathcal{V}}
y_{t,v}\log p_{t,v}\\
&=
-\log p_{t,y_t}.
\end{aligned}
$$

정답 좌표 하나만 $y_{t,v}=1$이므로 나머지 항이 모두 사라진다. 따라서 one-hot target을 쓰는 next-token cross-entropy는 해당 정답 토큰의 NLL과 정확히 같다.

Softmax와 cross-entropy를 합친 logit 미분은 지난 9월 7일에 유도한 식으로 돌아온다.

$$
\boxed{
\frac{\partial\ell_t}
{\partial z_{t,v}}
=
p_{t,v}-y_{t,v}
}
$$

정답 좌표에서는 $p_{t,y_t}-1<0$이므로 경사하강법이 그 logit을 올리는 방향으로 움직인다. 오답 좌표에서는 $p_{t,v}>0$이므로 그 logit을 내리는 방향으로 움직인다. 이 신호가 LM head, Transformer 블록, embedding까지 오차역전파된다.

### 3.7 padding을 제외한 정확한 batch loss

batch에서 각 수열 길이가 다르면 $\langle\mathrm{PAD}\rangle$로 shape를 맞춘다. 위치 $(b,t)$가 실제 target이면 $M_{bt}=1$, padding이거나 학습에서 제외할 위치이면 $M_{bt}=0$인 loss mask를 두자.

$$
M_{bt}\in\{0,1\}
$$

유효 target 수는

$$
N_{\mathrm{valid}}
=
\sum_{b=1}^{B}
\sum_{t=0}^{T-1}
M_{bt}
$$

이고, $N_{\mathrm{valid}}>0$일 때 token-mean causal loss는

$$
\boxed{
\mathcal{J}_{\mathrm{causal}}
=
-
\frac{1}{N_{\mathrm{valid}}}
\sum_{b=1}^{B}
\sum_{t=0}^{T-1}
M_{bt}
\log
p_{\theta}
\left(
Y_{bt}
\mid
X_{b,\le t}
\right)
}
$$

이다.

여기서 $\mathbf{X}$와 $\mathbf{Y}$는 이미 한 칸 이동된 입력과 정답이다. 표본 $b$의 같은 원본 수열

$$
\mathbf{S}^{(b)}
=
\left(
s_0^{(b)},s_1^{(b)},\ldots,s_T^{(b)}
\right)
$$

에서

$$
X_{b,t}=s_t^{(b)},
\qquad
Y_{b,t}=s_{t+1}^{(b)},
\qquad
t=0,\ldots,T-1
$$

로 만든다. 따라서 입력 위치 $t$의 hidden state는 target $s_{t+1}$을 맞힌다.

평균을 batch의 최대 길이 $BT$로 나누면 padding이 많은 batch의 loss와 gradient가 인위적으로 작아진다. 일반적인 token mean은 실제 학습 대상인 $N_{\mathrm{valid}}$로 나눈다.

batch 전체 logit에 대한 미분도 mask와 평균 계수를 그대로 반영한다.

$$
\boxed{
\frac{\partial\mathcal{J}_{\mathrm{causal}}}
{\partial z_{bt,v}}
=
\frac{M_{bt}}{N_{\mathrm{valid}}}
\left(
p_{bt,v}
-
\mathbf{1}[v=Y_{bt}]
\right)
}
$$

$M_{bt}=0$이면 그 위치의 **출력 logit으로 들어가는 직접 CE gradient**는 정확히 $0$이다. 다만 prompt-only token처럼 loss에서만 가리고 뒤의 유효 target이 attention할 수 있게 둔 입력은, 뒤 위치의 loss를 거쳐 embedding과 hidden state에 간접 gradient를 받을 수 있다. PAD는 loss mask뿐 아니라 padding attention mask로 Key 경로도 막아야 이런 간접 영향까지 차단된다.

### 3.8 perplexity는 평균 확률의 어떤 모습인가?

자연로그를 사용한 평균 NLL이

$$
\bar{\ell}
=
-
\frac{1}{N_{\mathrm{valid}}}
\sum_{i=1}^{N_{\mathrm{valid}}}
\log p_i
$$

라면 perplexity는

$$
\operatorname{PPL}
=
\exp(\bar{\ell})
$$

이다. 식을 변형하면

$$
\begin{aligned}
\operatorname{PPL}
&=
\exp\left(
-
\frac{1}{N_{\mathrm{valid}}}
\sum_i\log p_i
\right)\\
&=
\left(
\prod_i p_i
\right)^{-1/N_{\mathrm{valid}}}.
\end{aligned}
$$

즉 정답 토큰 확률들의 **기하평균의 역수**다. 모든 정답에 확률 $0.5$를 줬다면 PPL은 $2$, 모두에 $0.25$를 줬다면 $4$다.

PPL을 “매번 정확히 몇 개 후보 사이에서 헷갈린다”라고 문자 그대로 해석하면 안 된다. 직관적인 유효 분기 수에 가깝지만 실제 분포는 위치마다 다르다.

더 중요한 주의점은 tokenizer다. 같은 문자열도 tokenizer A가 10개 토큰, tokenizer B가 20개 토큰으로 나누면 평균의 단위 자체가 다르다. 따라서 서로 다른 vocabulary·normalization·EOS 처리·평가 corpus·loss mask를 쓴 PPL은 직접 순위를 매기기 어렵다. byte당 bit 수인 BPB 같은 공통 단위를 쓰거나 동일 tokenizer로 통제해야 한다.

### 3.9 짧은 수치 예제: 문장 확률, 평균 NLL, PPL

이미 tokenization된 수열이

$$
\left[
\langle\mathrm{BOS}\rangle,
\text{AI},
\text{learns},
\text{patterns},
\langle\mathrm{EOS}\rangle
\right]
$$

이고, 네 target에 모델이 준 정답 확률이 차례로

$$
(0.5,\;0.25,\;0.8,\;0.4)
$$

라고 하자.

teacher forcing 아래 수열 likelihood는

$$
\begin{aligned}
p_{\theta}(\text{sequence})
&=
0.5\times0.25\times0.8\times0.4\\
&=
0.04.
\end{aligned}
$$

평균 NLL은

$$
\begin{aligned}
\bar{\ell}
&=
-
\frac{1}{4}
\log(0.04)\\
&\approx
0.8047.
\end{aligned}
$$

따라서

$$
\operatorname{PPL}
=
\exp(0.8047)
\approx
2.236
$$

이다. 수열이 길어지면 결합확률 자체는 정상적으로 매우 작아진다. 서로 다른 길이를 평가할 때 평균 NLL을 쓰는 이유다.

### 3.10 tokenizer가 계산량을 바꾸는 두 축

어휘 크기가 $V=|\mathcal{V}|$이고 model dimension이 $d$라 하자. 입력 embedding과 LM head를 따로 두면 두 행렬의 주요 파라미터 수는

$$
Vd+dV
=
2Vd
$$

이다. 두 가중치를 묶는 weight tying을 쓰면 대략

$$
Vd
$$

개의 고유 가중치만 필요하다. vocabulary가 커질수록 embedding·출력 head와 Softmax 비용이 커진다.

반대로 작은 조각을 많이 쓰면 같은 문장이 더 긴 $T$가 된다. dense self-attention의 score 원소 수는 head 수까지 포함해

$$
B h T^2
$$

이므로 tokenizer B가 tokenizer A보다 같은 원문을 $r$배 많은 토큰으로 만든다면 attention score 수는 같은 $B,h$에서

$$
\frac{(rT)^2}{T^2}
=
r^2
$$

배가 된다. 그러나 embedding·FFN·LM head처럼 token별 연산은 주로 $T$에 선형으로 늘고, sparse·linear attention에서는 관계가 다를 수 있다.

따라서 tokenizer 설계에는 다음 trade-off가 있다.

$$
\boxed{
\text{큰 vocabulary·짧은 sequence}
\quad\longleftrightarrow\quad
\text{작은 vocabulary·긴 sequence}
}
$$

“더 적은 토큰이 무조건 좋은 tokenizer”도 아니고 “byte면 완전히 공평하다”도 아니다. 압축률, 언어·도메인별 fertility, 문자 복원성, vocabulary 메모리, 학습 데이터 규모, 모델 구조를 함께 봐야 한다.

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

**causal loss는 완전히 새로운 종류의 손실 함수가 아니다.** 9월 7일에 배운 다중분류 cross-entropy/NLL에 조건부확률의 왼쪽→오른쪽 자기회귀 분해, 한 칸 이동한 label, causal attention, valid-token mask를 결합한 학습 계약이다.

### 4.1 원문에서 loss까지: 하나의 shape 계약

길이 $T+1$의 ID 수열을 batch로 모았다고 하자.

$$
\mathbf{S}
\in
\{0,\ldots,V-1\}^{B\times(T+1)}
$$

한 칸 이동해

$$
\mathbf{X}
=
\mathbf{S}_{:,0:T},
\qquad
\mathbf{Y}
=
\mathbf{S}_{:,1:T+1}
$$

를 만든다. 둘의 shape는

$$
\mathbf{X},\mathbf{Y}
\in
\{0,\ldots,V-1\}^{B\times T}
$$

이다.

입력 ID는 embedding lookup을 거쳐

$$
\mathbf{X}_{\mathrm{emb}}
\in
\mathbb{R}^{B\times T\times d}
$$

가 된다. causal Transformer 블록들을 통과한 마지막 표현은

$$
\mathbf{H}
\in
\mathbb{R}^{B\times T\times d}
$$

이고, LM head는

$$
\mathbf{Z}
=
\mathbf{H}\mathbf{W}_{\mathrm{U}}
+\mathbf{b}_{\mathrm{U}}
\in
\mathbb{R}^{B\times T\times V}
$$

를 만든다. 마지막 vocabulary 축으로 Softmax한 뒤 각 $(b,t)$에서 target $Y_{bt}$의 로그확률만 골라 평균하면 scalar loss가 된다.

$$
\boxed{
(B,T+1)
\rightarrow
(B,T)
\rightarrow
(B,T,d)
\rightarrow
(B,T,V)
\rightarrow
()
}
$$

마지막의 $()$는 축이 없는 scalar라는 뜻이다. 이 한 scalar의 gradient가 모든 유효 token 위치에서 합쳐져 모델 전체로 역전파된다.

### 4.2 tokenizer 학습과 tokenizer 적용은 다른 단계다

**Tokenizer 학습**은 corpus를 보고 vocabulary와 BPE merge 순서 또는 unigram 점수를 정하는 단계다. **Tokenizer 적용**은 이미 고정된 규칙으로 새 문자열을 ID 수열로 바꾸는 단계다.

모델 학습 중 tokenizer를 제멋대로 바꾸면 같은 ID가 다른 문자열을 뜻하게 될 수 있다. 예를 들어 어제 ID $314$가 “ing”였는데 오늘 “tion”이 되면 embedding table의 314번째 행이 가리키던 의미 계약이 깨진다.

따라서 재현 가능한 모델 artifact에는 최소한 다음이 함께 필요하다.

- vocabulary와 token ID 순서
- merge 규칙 또는 unigram model
- normalization과 pre-tokenization 설정
- BOS·EOS·PAD 등 special token ID
- chat template와 control token 규약
- tokenizer 구현 버전과 최대 길이·truncation 정책

ID 수열을 다시 문자열로 decode할 수 있다는 것과 원래 입력 byte를 완전히 복원할 수 있다는 것도 항상 같지 않다. normalization이 서로 다른 Unicode 표현이나 공백을 먼저 합쳤다면 그 차이는 이미 사라졌다.

### 4.3 부분단어가 희귀어와 다국어를 다루는 방법

단어 사전에 “초거대언어모델연구자”가 없어도 subword tokenizer는

$$
\text{초거대}
\;|\;
\text{언어}
\;|\;
\text{모델}
\;|\;
\text{연구}
\;|\;
\text{자}
$$

또는 더 작은 조각으로 표현할 수 있다. 정확한 분할은 tokenizer마다 다르다.

이 장점에는 비용이 있다.

- 학습 corpus에서 적게 등장한 언어는 같은 뜻을 더 많은 토큰으로 표현할 수 있다.
- 숫자·코드·공백 패턴이 비효율적으로 잘리면 context와 비용을 더 사용한다.
- 형태소 경계와 subword 경계가 일치하지 않을 수 있다.
- prompt 끝의 공백 하나가 다음 token 후보를 크게 바꿀 수 있다.
- special token 문자열을 일반 text처럼 encode하면 제어 구조가 달라질 수 있다.

그래서 tokenizer 평가는 전체 평균 tokens per character만 보지 않고, 언어·도메인별 fertility와 긴 꼬리 분포를 함께 봐야 한다.

$$
\operatorname{fertility}
=
\frac{\text{생성된 token 수}}
{\text{기준 word 수}}
$$

한국어처럼 띄어쓰기·교착 형태가 영어와 다른 언어에서는 “word”의 정의도 명시해야 공정하다.

### 4.4 한 칸 이동이 만드는 자기지도 정답

다음 tokenized sequence를 보자.

$$
\left[
\langle\mathrm{BOS}\rangle,
\text{나는},
\text{AI},
\text{를},
\text{배운다},
\langle\mathrm{EOS}\rangle
\right]
$$

입력과 target은 다음과 같다.

| 위치 $t$ | 모델 입력 $X_t$ | 예측 target $Y_t$ | 허용되는 입력 문맥 |
|---:|---|---|---|
| $0$ | $\langle\mathrm{BOS}\rangle$ | 나는 | $\langle\mathrm{BOS}\rangle$ |
| $1$ | 나는 | AI | $\langle\mathrm{BOS}\rangle$, 나는 |
| $2$ | AI | 를 | 앞의 세 input token |
| $3$ | 를 | 배운다 | 앞의 네 input token |
| $4$ | 배운다 | $\langle\mathrm{EOS}\rangle$ | 앞의 다섯 input token |

원문 자체가 한 칸 뒤의 정답을 제공하므로 사람이 “다음 단어는 이것”이라고 별도 annotation할 필요가 없다.

여기서 흔한 off-by-one 오류는 다음 두 가지다.

1. input과 target을 같은 수열로 두어 현재 token을 그대로 복사하게 만드는 오류
2. 라이브러리 모델이 내부에서 이미 shift하는데 외부에서 또 shift해 두 칸 뒤를 예측하게 만드는 오류

사용하는 framework의 loss API가 shift를 맡는지 반드시 문서와 작은 예제로 확인해야 한다.

### 4.5 causal mask가 병렬 teacher forcing을 안전하게 만든다

학습 입력에는 정답 수열의 여러 token이 한꺼번에 들어 있다. 위치 $i$가 attention에서 볼 수 있는 Key 위치 $j$를

$$
M_{ij}^{\mathrm{causal}}
=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i
\end{cases}
$$

로 제한한다. attention score에 이 mask를 더하고 Softmax하면 미래 위치는

$$
\exp(-\infty)=0
$$

의 weight를 받는다.

$$
A_{ij}
=
\operatorname{Softmax}_{j}
\left(
\frac{\mathbf{q}_i^{\top}\mathbf{k}_j}{\sqrt{d_h}}
+M_{ij}^{\mathrm{causal}}
\right)
$$

따라서 모든 위치를 같은 tensor에 넣어 병렬 계산해도 위치 $i$의 hidden state는 $j\le i$만 사용한다. 그 hidden state가 한 칸 뒤 target $x_{i+1}$을 예측한다.

$$
\text{병렬 계산}
\not\Rightarrow
\text{미래 정보 사용}
$$

병렬성은 GPU 실행 방식이고, 인과성은 mask가 정한 정보 흐름이다.

### 4.6 teacher forcing과 autoregressive 생성의 차이

학습에서는 다음처럼 정답 과거가 이미 있다.

$$
\begin{aligned}
\langle\mathrm{BOS}\rangle
&\rightarrow \text{나는},\\
\langle\mathrm{BOS}\rangle,\text{나는}
&\rightarrow \text{AI},\\
\langle\mathrm{BOS}\rangle,\text{나는},\text{AI}
&\rightarrow \text{를}.
\end{aligned}
$$

세 문맥을 실제로 따로 실행하지 않고, causal mask가 있는 한 번의 순전파로 모든 위치의 logits를 얻는다. **정답 prefix 전체를 제공하는 teacher forcing, causal mask, 시간축 recurrence가 없는 Transformer 연산**의 조합이 이 위치 병렬화를 만든다. teacher forcing만으로 모든 sequence model이 병렬화되는 것은 아니다.

추론에서는 정답이 없다.

$$
\begin{aligned}
\text{prompt}
&\rightarrow \hat{x}_{T+1},\\
\text{prompt},\hat{x}_{T+1}
&\rightarrow \hat{x}_{T+2},\\
\text{prompt},\hat{x}_{T+1},\hat{x}_{T+2}
&\rightarrow \hat{x}_{T+3}.
\end{aligned}
$$

여기서 $\hat{x}$는 greedy, sampling, beam search 등의 decoding 규칙으로 선택한 모델 출력이다. 첫 예측이 어색하면 이후 모델은 학습 중 드물게 본 문맥을 만나 오차가 누적될 수 있다.

그렇다고 teacher forcing이 잘못된 방법이라는 뜻은 아니다. 대규모 causal LM pretraining의 표준적이고 효율적인 최대가능도 방법이다. on-policy rollout이나 sequence-level objective는 학습·생성 분포의 mismatch를 직접 다루려 하지만 비용과 불안정성이 크다. preference optimization과 reinforcement learning은 생성된 응답의 행동 목표를, data augmentation은 문맥 변화에 대한 강건성을 개선할 수 있지만 exposure bias를 자동으로 없앤다고 보장되지는 않는다. 작은 sequence model에서 알려진 scheduled sampling도 모든 LLM의 당연한 해결책은 아니다.

### 4.7 attention mask와 loss mask는 서로 다른 질문에 답한다

길이가 다른 두 수열을 오른쪽 padding했다고 하자.

$$
\begin{aligned}
\mathbf{s}^{(1)}
&=
[a,b,c,\langle\mathrm{EOS}\rangle],\\
\mathbf{s}^{(2)}
&=
[d,e,\langle\mathrm{EOS}\rangle,\langle\mathrm{PAD}\rangle].
\end{aligned}
$$

attention mask는 “이 Query가 어느 Key 정보를 읽어도 되는가?”를 정한다. loss mask는 “이 위치의 target 예측을 학습 점수에 포함하는가?”를 정한다.

| 장치 | 묻는 질문 | 대표 shape | padding 처리 |
|---|---|---:|---|
| causal attention mask | 미래 Key인가? | $1\times1\times T\times T$ | 시간 방향 누출 차단 |
| padding attention mask | 실제 입력 Key인가? | $B\times1\times1\times T$ | PAD Key를 읽지 않게 함 |
| loss mask | 실제 학습 target인가? | $B\times T$ | PAD target의 NLL을 제외 |

세 mask를 같은 것으로 생각하면 안 된다. PAD target을 loss에서 뺐더라도 실제 token이 PAD Key를 읽는다면 hidden state가 오염될 수 있다. 반대로 attention에서 PAD를 막았어도 PAD target까지 정답으로 학습하면 모델이 padding 배치를 맞히는 데 용량을 쓴다.

모든 Key가 가려진 Query는 Softmax 분모가 $0$인 것과 같은 수치 문제를 만들 수 있다. framework가 padding Query를 어떻게 처리하는지, finite 최소값을 쓰는지, loss에서 완전히 제외하는지 확인해야 한다.

### 4.8 LM head와 weight tying

입력 embedding은 token ID를 $d$차원으로 올린다.

$$
\mathbf{E}
\in
\mathbb{R}^{V\times d}
$$

LM head는 반대로 hidden state를 vocabulary logits로 보낸다.

$$
\mathbf{W}_{\mathrm{U}}
\in
\mathbb{R}^{d\times V}
$$

두 공간의 의미가 밀접하므로

$$
\mathbf{W}_{\mathrm{U}}
=
\mathbf{E}^{\top}
$$

로 묶는 weight tying을 쓸 수 있다. 그러면 입력에서 어떤 token을 나타내는 방향과 출력에서 그 token을 점수화하는 방향이 같은 파라미터를 공유하고, 큰 $Vd$ 행렬 하나를 줄인다.

하지만 모든 모델이 완전히 같은 tying을 쓰는 것은 아니다. output bias, normalization, vocabulary 확장, multimodal token, tensor parallel 배치 때문에 구현이 달라질 수 있다. checkpoint config를 확인해야 한다.

### 4.9 최소 구현으로 보는 causal loss

개념적 의사 코드는 다음과 같다. 실제 library의 모델이 label shift를 내부에서 처리하는지는 별도로 확인해야 한다.

    # token_ids: [B, T + 1]
    # full_valid: bool [B, T + 1], True는 실제 token, False는 padding
    full_valid = build_validity_from_lengths(lengths, T + 1)
    inputs = token_ids[:, :-1]         # [B, T]
    targets = token_ids[:, 1:]         # [B, T]
    input_valid = full_valid[:, :-1]   # [B, T], token ID와 별도인 길이 정보
    target_valid = full_valid[:, 1:]   # [B, T]
    loss_targets = targets.masked_fill(~target_valid, -100)

    hidden = causal_transformer(
        inputs,
        key_keep_mask=input_valid,         # 이 의사 API에서는 True=읽기 허용
    )                                     # [B, T, d], causal mask도 내부 적용
    logits = lm_head(hidden)              # [B, T, V]

    loss = cross_entropy(
        logits.reshape(B * T, V),
        loss_targets.reshape(B * T),
        ignore_index=-100,
        reduction="mean",
    )

실무에서는 다음을 추가로 점검한다.

- packed sequence 경계 너머로 attention이 새어 가지 않는가?
- EOS를 target으로 학습하고 실제 padding 위치만 제외하는가?
- PAD와 EOS가 같은 ID인 모델에서도 token ID 비교가 아니라 별도 valid mask로 실제 EOS를 보존하는가?
- 문서 연결 시 앞 문서 끝과 다음 문서 시작 사이의 조건부확률을 의도했는가?
- label smoothing을 쓰면 one-hot NLL과 정확히 같은 식이 아님을 알고 있는가?
- mixed precision에서 fused cross-entropy가 안정적인 log-sum-exp를 쓰는가?
- 분산학습에서 각 장치의 valid-token 수가 다를 때 전역 token mean을 계산하는가?

마지막 항목은 특히 중요하다. 각 GPU의 local mean을 단순 평균하면 valid-token이 적은 GPU와 많은 GPU가 같은 가중치를 가져 전체 token mean과 달라질 수 있다.

### 4.10 학습 로그를 읽는 법

training loss가 내려간다는 것은 관측된 target token의 평균 log-probability가 올라간다는 뜻이다. 그러나 그것만으로 다음을 보장하지 않는다.

- 사실성
- 안전성
- 긴 추론 능력
- 지시 따르기
- 인간이 선호하는 문체
- 학습 corpus 밖의 일반화

validation NLL은 보지 않은 검증 token에 대한 확률 예측을 점검한다. training NLL만 계속 내려가고 validation NLL이 오르면 과적합 신호일 수 있다.

토큰 평균은 빈도가 높은 쉬운 token에 크게 지배될 수도 있다. 전체 loss 외에 언어·도메인·길이·희귀 token·코드·수학별 slice와 downstream 평가를 함께 봐야 한다.

### 4.11 수학과 구현의 1:1 연결표

| 3절의 수학 | 4절의 구현 부품 | 실패하면 생기는 일 |
|---|---|---|
| $\tau(s)=(x_1,\ldots,x_T)$ | tokenizer와 고정 vocabulary | checkpoint와 ID 의미가 불일치한다. |
| $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$ | 한 칸 shift + causal Transformer | 현재·미래 token을 베끼거나 엉뚱한 위치를 예측한다. |
| Softmax over $\mathcal{V}$ | LM head의 마지막 vocabulary 축 | 다른 축을 정규화하면 다음 token 분포가 아니다. |
| $-\log p(y_t)$ | token cross-entropy | 정답 확률을 키우는 학습 신호가 사라진다. |
| $M_{bt}\in\{0,1\}$ | ignore index·loss mask | PAD나 prompt-only 위치가 loss를 왜곡한다. |
| $N_{\mathrm{valid}}$ | 전역 valid-token reduction | batch 구성과 GPU별 padding에 따라 gradient scale이 흔들린다. |
| $\partial\ell/\partial z=p-y$ | autograd와 backpropagation | LM head부터 embedding까지 업데이트되지 않는다. |
| $\exp(\bar{\ell})$ | validation perplexity | tokenizer가 다른 수치를 부당하게 비교할 수 있다. |

### 4.12 초보자가 흔히 하는 오해와 주의할 점

| 오해 | 정확한 설명 |
|---|---|
| “토큰은 단어다.” | 한 토큰은 단어, 부분단어, 문장부호, 공백 포함 조각, byte 조각일 수 있다. |
| “token ID가 비슷하면 의미도 비슷하다.” | ID는 행 주소다. 의미 유사성은 학습된 embedding과 문맥 표현에서 생긴다. |
| “BPE는 언제나 byte 두 개를 합친다.” | 이름의 역사와 현대 구현을 구분해야 한다. 많은 구현은 byte-level 기반이지만 merge 대상은 현재 기호의 인접 쌍이다. |
| “teacher forcing은 모델의 답을 정답으로 강제로 바꾼다.” | 모델 출력은 그대로 loss를 받는다. 다음 위치의 입력 문맥에 실제 이전 token을 제공하는 것이다. |
| “학습도 token을 하나씩 순차 실행한다.” | causal mask 덕분에 한 수열의 모든 위치 logits를 병렬 계산할 수 있다. |
| “causal mask가 있으면 target shift는 필요 없다.” | mask는 미래 읽기를 막고, shift는 각 위치의 정답을 다음 token으로 맞춘다. 역할이 다르다. |
| “PAD를 attention에서 가리면 loss에서도 자동 제외된다.” | 두 mask는 별도일 수 있다. API 계약을 확인해야 한다. |
| “cross-entropy와 NLL은 언제나 같은 말이다.” | one-hot target의 다중분류에서는 같다. soft target·label smoothing에서는 cross-entropy가 여러 좌표를 포함한다. |
| “낮은 PPL 모델이 모든 면에서 더 좋은 LLM이다.” | 같은 tokenizer와 corpus에서 확률 예측을 비교하는 지표다. 사실성·안전성·도구 사용을 모두 측정하지 않는다. |
| “생성 때 가장 확률 높은 token만 고르면 학습 목표와 완전히 같다.” | 학습은 분포의 log-likelihood를 최적화한다. decoding은 그 분포에서 수열을 선택하는 별도 알고리즘이다. |

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

오늘은 2026-09-10~11에 공식 공개된 두 흐름을 고른다. 하나는 LLM이 먹는 token을 산업 규모에서 어떻게 고를지, 다른 하나는 text가 아닌 달 관측 자료도 어떻게 discrete token과 cross-entropy로 학습할지를 보여 준다.

### 5.1 Marin Datakit: 25.25조 Llama 3 token 원시 풀을 큐레이션한 데이터 정책

Open Athena의 Marin 팀은 2026-09-11, 최신 대규모 학습 run을 위해 사용한 **Datakit** 데이터 curation pipeline을 공개 설명했다.

출발점은 다음과 같다.

$$
152\text{개 open dataset}
\rightarrow
18.71\text{ billion raw documents}
\rightarrow
25.25\text{ trillion Llama 3 tokens}
$$

여기서 25.25조는 실제 한 model이 모두 학습한 양이 아니라 **큐레이션 전 raw pool을 Llama 3 tokenizer로 센 token 수**다. deduplication과 decontamination 뒤의 가용 pool은 약 23.11조 token이다. 오늘 배운 것처럼 tokenizer가 달라지면 같은 원문의 token 수와 계산 예산 해석도 달라진다.

#### 단계 1: provenance를 기록하고 corpus 전체에서 중복을 제거한다

Datakit은 각 source의 upstream repository, revision, license를 catalogue에 기록하고 공통 Parquet 표현으로 normalize한다. exact hash와 word 5-gram MinHash·LSH를 사용해 source 내부뿐 아니라 source 사이의 near-duplicate도 찾는다.

팀 보고에 따르면 global deduplication은

- 23.3억 documents, raw pool의 $12.5\%$
- 2.13조 tokens, raw token의 $8.4\%$

를 제거했다.

중복 제거는 저장 공간 청소만이 아니다. 같은 문서가 여러 dataset을 통해 반복되면 의도한 epoch 수보다 더 자주 보게 되고, memorization과 data mixture가 왜곡된다. 오늘의 likelihood 식에서 어떤 token 항을 몇 번 더하는지를 통제하는 일이다.

#### 단계 2: benchmark contamination을 보수적으로 거른다

평가 답이 pretraining corpus에 섞이면 generalization 대신 기억을 측정할 수 있다. Datakit은 Artificial Analysis Intelligence Index와 Marin 평가 세트에 대해 정확한 **13-word n-gram** 겹침을 찾고, 한 paragraph의 eligible 13-gram 중 최소 $50\%$가 일치하면 해당 document 전체를 flag한다. 현재 정책은 flag된 document를 모두 제거했고, 공식 글은 이 단계가 추가로 13.66 billion, 즉 136.6억 tokens를 제거했다고 보고한다.

이 규칙은 exact overlap에 높은 precision을 두므로 paraphrase나 일부 삽입형 contamination은 놓칠 수 있다. “decontaminated”가 모든 의미적 누출이 사라졌다는 뜻은 아니다.

#### 단계 3: 200개 topic-quality bucket을 만들고 mixture를 실험한다

모든 document를 Microsoft Harrier의 $1{,}024$차원 embedding으로 바꾼 뒤, spherical $k$-means로 $5{,}000$개 작은 cluster를 만들고 이를 40개 topic으로 합친다. 별도의 supervised quality scorer는 content type별 threshold를 사용해 다섯 quality band를 만든다.

$$
40\text{ topics}
\times
5\text{ quality bands}
=
200\text{ buckets}
$$

그다음 작은 MoE proxy model을 서로 다른 sampling weight로 반복 학습하고, bucket weight에서 benchmark 성능을 예측하는 regression을 fit한다. 2026-09-11 공식 글은 전체적으로 1,000회가 조금 넘는 small-scale experiment를 언급하고, 당시 공개된 smallest-MoE proxy snapshot을 872개 observation으로 설명한다. Hugging Face dataset은 이후 바뀔 수 있으므로 현재 row 수와 이 글의 snapshot 수를 같은 것으로 가정하면 안 된다.

흥미롭게도 공개 시각화는 872개 mixture의 초기 weight를 **PCA로 투영**한다. 9월 10일에 배운 PCA가 실제 대규모 실험 공간에서 “서로 비슷한 data mixture가 어디에 모였는가?”를 보는 도구로 돌아온 셈이다.

#### 엔지니어 인사이트 (Impact)

- **data curation은 loss의 숨은 가중치 설계다.** bucket $k$의 sampling weight를 $w_k$라 하면 기대 학습 목표는 대략

$$
\mathbb{E}_{k\sim\mathbf{w}}
\mathbb{E}_{x\sim D_k}
\left[
-\log p_{\theta}(x_t\mid x_{<t})
\right]
$$

이다. 모델 구조가 같아도 $\mathbf{w}$가 바뀌면 어떤 능력의 gradient를 얼마나 자주 보는지가 바뀐다.

- **중복 수와 epoch를 분리해야 실험이 해석된다.** crawler가 우연히 만든 반복과 연구자가 의도한 upsampling은 같은 token 반복처럼 보여도 원인이 다르다.
- **quality는 절대적인 filter가 아니다.** 작은 compute에서는 고품질 data 집중이 유리할 수 있지만, 큰 run에서는 같은 문서 반복에 따른 과적합 때문에 다양성이 더 중요해질 수 있다. Datakit이 점수를 바로 버리는 filter보다 bucket weight로 보존한 이유다.
- **proxy-to-hero transfer는 아직 가정이 남는다.** 작은 모델에서 fit한 regression의 최적 mixture가 535B-total/약 23B-active MoE hero run에서도 같은 순위를 유지하는지는 full-scale 결과로 검증해야 한다. 이 18조-token run의 pretraining 완료 예정일은 2026-12-01이므로 오늘 시점에는 최종 결과가 아니다. bucket weight 합이 $1$이므로 개별 상관도 독립적인 인과효과가 아니다.
- **open pipeline도 byte-for-byte 재현과 같지 않다.** 팀은 upstream dataset을 다시 host하지 않는다. commit을 pin해도 원 제작자의 opt-out 삭제가 반영되므로 나중에 ingestion한 corpus는 조금 달라질 수 있다. 이는 attribution·삭제권과 완전한 고정 재현성 사이의 의식적인 trade-off다.
- **license catalogue는 법률 보증서가 아니다.** source별 provenance와 허용 범위를 실제 배포 목적에 맞게 다시 검토해야 한다.

**1차 출처:** [Open Athena 공식 기술 글](https://openathena.ai/blog/marin-data-pipeline-overview/) · [535B-A23B run 공식 발표](https://openathena.ai/blog/huang-foundation-marin-535b-training-run/) · [Marin source registry](https://github.com/marin-community/marin/blob/64a9483cf1a928e6db97b83130d8a5d171a4aead/lib/marin/src/marin/datakit/sources.py) · [Datakit 공개 구현](https://github.com/marin-community/marin/tree/main/lib/marin/src/marin/datakit) · [공개 proxy-run dataset](https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm)

### 5.2 NASA–IBM Lunar Foundation Model: 달 관측 modality도 token으로 바꾸면 같은 CE를 쓸 수 있다

IBM과 NASA는 2026-09-10 **NASA–IBM Lunar Foundation Model**의 weights, tokenizer checkpoints, fine-tuning·inference code, SomBench 접근 경로를 공개했다. 이 모델은 대화형 LLM이 아니라 달 원격탐사 자료를 위한 multimodal·multi-resolution ViT encoder–decoder다.

공개 범위에는 경계가 있다. model weights와 공개 GitHub code는 Apache-2.0, SomBench corpus는 CC BY 4.0으로 서로 다르며 GitHub 저장소에는 pretraining code가 포함되지 않는다. Hugging Face dataset repository는 작은 sample과 전체 data 접근 안내를 제공하고, 전체 WAC 약 38 TB와 NAC 약 1.4 TB corpus는 public AWS S3에서 받는 구조다.

공식 model card의 핵심 규모는 다음과 같다.

| 항목 | 공식 공개 내용 |
|---|---|
| backbone | ViT-B encoder: width $768$, 12 layers, 12 heads |
| decoder | 같은 width의 12-layer Transformer |
| pretraining data | WAC $963{,}609$ + NAC $1{,}000{,}113$ tile bundles |
| modality | 11개, 그중 9개 dense image-like modality와 두 종류의 context |
| scale | NAC 약 $1$ m/pixel, WAC 약 $100$ m/pixel |
| objective | sampled target 위치의 discrete vocabulary에 대한 cross-entropy |
| tokenizers | 9개 modality-specific VQ-VAE, finite scalar quantization |
| training | 16 H100, 150k steps, global batch $1{,}536$, bf16, 약 $1.1$k GPU-hours |
| license | model weights·공개 code: Apache-2.0 / SomBench corpus: CC BY 4.0 |

각 pretraining sample은 NAC-centered 또는 WAC-centered이며 두 resolution family가 한 sample 안에서 섞이지 않는다. joint mixed-resolution은 두 family를 같은 mixed-batch loop와 하나의 backbone으로 학습한다는 뜻이다.

#### causal next-token LM과 무엇이 같고 무엇이 다른가?

이미지·고도·경사·반사율 같은 연속 값을 modality별 tokenizer가 discrete token으로 바꾸면 출력은 유한 vocabulary 위 분류가 된다. target token $y$에 대한 손실은 오늘과 똑같이

$$
\ell
=
-\sum_{v}y_v\log p_v
=
-\log p_y
$$

다.

하지만 **무엇을 조건으로 허용하는가**가 다르다.

$$
\begin{aligned}
\text{causal LM:}\quad
&p(x_t\mid x_{<t}),\\
\text{Lunar masked-token model:}\quad
&p(x_{\mathcal{M}}\mid x_{\mathrm{visible}},\text{geometry context}).
\end{aligned}
$$

causal LM은 미래를 가리고 왼쪽에서 오른쪽으로 생성한다. Lunar model은 여러 modality에서 일부 target token을 가리고, 보이는 modality·위치·획득 geometry를 조건으로 복원하는 TerraMind식 any-to-any objective를 쓴다. 같은 cross-entropy라도 **mask가 정의하는 조건부확률 문제**가 다르다.

특히 illumination angle, 태양 기준 좌표, tile footprint 같은 acquisition geometry를 명시적 sequence token으로 넣는다. 달 표면 영상은 물질 차이뿐 아니라 빛의 방향에 크게 달라지므로, 이미 sensor metadata에 있는 주요 confound를 모델이 영상만 보고 다시 추측하게 하지 않으려는 설계다.

#### 공식 보고 성능과 반드시 함께 읽을 경계

공식 model card는 polar ice prospectivity 회귀에서 full fine-tuning 모델의

$$
\operatorname{RMSE}
=
0.0293\pm0.0013
$$

을 보고했다. 비교한 SwinV2-B baseline은

$$
0.0377\pm0.0004
$$

였다. 보고된 두 평균값으로 직접 계산한 상대 RMSE 감소는

$$
\frac{0.0377-0.0293}{0.0377}
\times100
\approx
22.3\%
$$

다. IBM 공식 발표는 이를 “up to $22\%$”로 요약한다. $22.3\%$는 model card의 두 평균값에서 계산한 값이지 별도의 독립 측정치가 아니다.

그러나 target은 실제로 시추해 측정한 얼음이 아니라 지식 기반 fuzzy-overlay **prospectivity map**이다. 모델 card는 다음을 명확히 제한한다.

- 생성된 field는 calibrated scientific prediction이 아니다.
- absolute geodetic reference frame이 없어 생성 위도·경도가 수십 도 어긋날 수 있다.
- landing-site 인증이나 hazard clearance 같은 운영 결정에 검증되지 않았다.
- NAC pretraining site는 co-registered 3 m stereo DTM이 있는 1,095 frames로 제한된다.
- meter-scale crater와 일부 segmentation의 모델 간 차이는 seed spread보다 작아 사실상 comparable하게 읽어야 한다.
- Moon 밖이나 SomBench에 없는 sensor product에는 평가되지 않았다.

#### 엔지니어 인사이트 (Impact)

- **tokenization은 text 전용 기술이 아니다.** image patch나 물리 field를 finite codebook으로 양자화하면 Transformer와 cross-entropy라는 같은 학습 machinery를 재사용할 수 있다.
- **discretization은 정보 병목이다.** VQ tokenizer가 세부 물리량을 잃으면 Transformer가 뒤에서 복원할 수 없다. reconstruction fidelity와 downstream utility를 함께 평가해야 한다.
- **objective 이름보다 conditioning graph를 보자.** causal loss와 masked-token CE는 수식의 마지막 줄이 같아도 보이는 token 집합이 다르므로 학습되는 사용 방식이 다르다.
- **multimodal late fusion은 해상도 차이를 숨기지 않는다.** modality별 patch adapter와 sequence concatenation은 100배 scale gap을 한 backbone에서 다루게 하지만, 정렬 오차·coverage 차이·sensor bias가 사라지는 것은 아니다.
- **open artifact는 과학적 진실 판정기와 다르다.** weights·tokenizers·fine-tuning code·data 접근 경로의 공개는 재현과 adaptation의 출발점을 제공하지만 pretraining pipeline 전체가 공개된 것은 아니다. operational validation은 별도의 현장 증거와 uncertainty calibration이 필요하다.

**1차 출처:** [IBM–NASA 공식 발표](https://newsroom.ibm.com/2026-09-10-ibm-and-nasa-release-open-source-ai-model-to-support-lunar-exploration) · [공식 Hugging Face model card·weights](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model) · [NASA-IMPACT 공식 fine-tuning code](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) · [SomBench sample·전체 AWS 접근 안내](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-pretraining-data)

### 5.3 두 흐름을 함께 읽기: token은 중립적인 단위가 아니라 학습 정책이다

Marin은 원문을 Llama 3 token으로 바꾼 뒤 어떤 token을 몇 번 볼지 mixture weight로 정한다. Lunar FM은 서로 다른 sensor field를 modality별 discrete token으로 바꾼 뒤 어떤 token을 숨기고 무엇으로 복원할지 mask로 정한다.

$$
\boxed{
\text{tokenizer}
\rightarrow
\text{학습 가능한 사건의 단위}
\rightarrow
\text{conditioning·sampling policy}
\rightarrow
\text{gradient가 강조하는 능력}
}
$$

따라서 새 foundation model을 읽을 때 parameter 수와 benchmark만 보지 말고 다음을 확인해야 한다.

1. 무엇을 한 token으로 정의했는가?
2. tokenizer가 어떤 언어·도메인·modality의 정보를 더 길게 또는 더 거칠게 표현하는가?
3. 어느 token을 조건으로 보여 주고 어느 token을 target으로 삼는가?
4. token별 loss를 어떤 sampling weight와 mask로 평균하는가?
5. 공개된 weights뿐 아니라 tokenizer·data provenance·evaluation split도 재현 가능한가?

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

tokenizer가 다음 수열을 만들었다.

$$
\left[
\langle\mathrm{BOS}\rangle,
\text{AI},
\text{learns},
\text{patterns},
\langle\mathrm{EOS}\rangle
\right]
$$

모델이 teacher forcing으로 학습할 때 네 target 정답에 준 확률은 차례로

$$
(0.5,\;0.25,\;0.8,\;0.4)
$$

다.

**하나의 핵심 질문:** “문장 하나가 어떻게 조건부확률 네 개와 scalar loss 하나로 바뀌는가?”를 다음 항목으로 설명하라.

1. input과 target 수열을 각각 쓰고, 왜 한 칸 이동해야 하는지 설명하라.
2. 수열 likelihood를 조건부확률의 곱으로 쓰고 값을 계산하라.
3. 자연로그 기반 token-mean NLL과 perplexity를 계산하라.
4. teacher forcing을 쓰면서도 causal mask가 반드시 필요한 이유를 설명하라.
5. 세 번째 입력 위치의 vocabulary 예측 분포가 $\mathbf{p}$이고 target one-hot이 $\mathbf{y}$일 때, 그 위치의 per-token loss $\ell$에 대한 logit gradient를 쓰고 의미를 설명하라.
6. 같은 cross-entropy를 쓰는 Lunar masked-token model이 causal LM과 같은 생성 목표가 아닌 이유를 설명하라.

### 모범 답안

#### 1. 한 칸 이동한 input과 target

입력은 마지막 token을 빼고,

$$
\mathbf{X}
=
\left[
\langle\mathrm{BOS}\rangle,
\text{AI},
\text{learns},
\text{patterns}
\right]
$$

target은 첫 token을 뺀다.

$$
\mathbf{Y}
=
\left[
\text{AI},
\text{learns},
\text{patterns},
\langle\mathrm{EOS}\rangle
\right]
$$

전체 원본 수열을 $\mathbf{S}$라 하면 $X_t=S_t$, $Y_t=S_{t+1}$이다. 따라서 같은 위치 $t$의 hidden state가 현재 입력을 복사하는 것이 아니라 실제 다음 token을 맞힌다. 마지막 target에 EOS를 포함하므로 언제 끝날지도 학습한다.

#### 2. 조건부확률의 곱

수열 likelihood는

$$
\begin{aligned}
&p_{\theta}(
\text{AI},
\text{learns},
\text{patterns},
\langle\mathrm{EOS}\rangle
\mid
\langle\mathrm{BOS}\rangle
)\\
&=
p_{\theta}(\text{AI}\mid\langle\mathrm{BOS}\rangle)\\
&\quad\times
p_{\theta}(\text{learns}\mid\langle\mathrm{BOS}\rangle,\text{AI})\\
&\quad\times
p_{\theta}(\text{patterns}\mid\langle\mathrm{BOS}\rangle,\text{AI},\text{learns})\\
&\quad\times
p_{\theta}(\langle\mathrm{EOS}\rangle\mid
\langle\mathrm{BOS}\rangle,\text{AI},\text{learns},\text{patterns}).
\end{aligned}
$$

주어진 확률을 대입하면

$$
0.5\times0.25\times0.8\times0.4
=
0.04
$$

다.

#### 3. 평균 NLL과 perplexity

$$
\begin{aligned}
\bar{\ell}
&=
-
\frac{1}{4}
\left(
\log0.5+\log0.25+\log0.8+\log0.4
\right)\\
&=
-
\frac{1}{4}\log0.04\\
&\approx
0.8047.
\end{aligned}
$$

따라서

$$
\operatorname{PPL}
=
\exp(0.8047)
\approx
2.236
$$

이다. 이는 동일 tokenizer·corpus·mask 규약에서 해석해야 한다.

#### 4. teacher forcing과 causal mask

teacher forcing은 정답 과거 token들을 입력 tensor에 제공한다. 그 tensor에는 뒤 위치의 정답 token도 함께 들어 있으므로 mask가 없다면 위치 $i$가 미래 $j>i$를 attention해 target 정보를 미리 볼 수 있다.

causal mask는

$$
j>i
\quad\Longrightarrow\quad
A_{ij}=0
$$

이 되게 해 각 위치가 오직 과거와 현재 입력만 사용하도록 한다. 덕분에 미래 누출 없이 네 위치를 병렬 계산한다.

#### 5. logit gradient

Softmax와 one-hot cross-entropy를 합치면

$$
\frac{\partial\ell}
{\partial\mathbf{z}}
=
\mathbf{p}-\mathbf{y}
$$

이다. 이는 해당 위치의 per-token loss $\ell$에 대한 식이다. 앞에서 구한 네 token의 mean loss

$$
\bar{\ell}
=
\frac{1}{4}
\sum_{t=1}^{4}\ell_t
$$

에 대한 세 번째 위치 logit gradient는

$$
\frac{\partial\bar{\ell}}
{\partial\mathbf{z}_{\mathrm{third}}}
=
\frac{1}{4}
\left(
\mathbf{p}_{\mathrm{third}}
-
\mathbf{y}_{\mathrm{third}}
\right)
$$

이다. 정답 좌표의 per-token gradient는 $p_{\mathrm{correct}}-1<0$이므로 gradient descent가 정답 logit을 올린다. 각 오답 좌표는 $p_v>0$이므로 확률을 많이 잘못 준 오답일수록 logit을 더 강하게 내린다.

#### 6. 같은 CE, 다른 조건부확률

causal LM은

$$
p(x_t\mid x_{<t})
$$

를 학습해 왼쪽 문맥에서 다음 token을 생성한다. Lunar masked-token model은

$$
p(x_{\mathcal{M}}
\mid
x_{\mathrm{visible}},\text{geometry context})
$$

를 학습해 임의로 가린 modality token을 보이는 여러 modality로 복원한다. 둘 다 정답 discrete token의 cross-entropy를 최소화하지만 mask와 conditioning graph가 다르므로 학습되는 생성 절차도 다르다.

---

**다음 연결 고리:** 오늘은 Transformer가 next-token likelihood로 학습되는 과정을 완성했다. 다음에는 학습된 확률분포에서 실제 문장을 고르는 greedy decoding·temperature·top-$k$·top-$p$와, pretraining을 대화 행동으로 바꾸는 instruction fine-tuning을 연결한다.
