# [2026-09-29] 오늘 학습: 잠재계층·EM·변화점 우도비 & 반복 판정자 Gold Label·Detector Drift·Conformal Risk Control

> **오늘의 핵심 문장:** 여러 판정자가 같은 답을 냈다고 해서 그것이 곧 정답은 아니다. 숨은 진실과 판정자별 오류를 함께 추정하고, 그 오류 구조가 바뀌는 순간을 감시하며, 확정 감사 라벨로 위험 한도를 다시 검량해야 비로소 안전한 자동화가 된다.

지난 시간에는 진짜 안전 사건 $Y$와 자동 경보 $Z$를 분리하고, 고정된 민감도·특이도로 관측률을 역보정했다. 그러나 그때의 $Y$는 충분한 절차로 확정된 **gold label**이라고 가정했다.

현실의 LLM·VLM 운영에서는 이 가정이 자주 깨진다. 사람 검토자도 서로 다르게 판단하고, LLM judge도 위치·문체·언어에 따라 흔들리며, 정책이 바뀌면 어제 측정한 민감도와 특이도가 오늘에는 맞지 않을 수 있다. 오늘은 이 문제를 세 층으로 나눈다.

1. 여러 불완전한 판정으로부터 숨은 진실을 어떻게 추정하는가?
2. 판정자의 오류율이 달라지는 변화점을 어떻게 찾는가?
3. 어떤 사례를 자동 통과시키고 어떤 사례를 사람에게 보낼지, 위험 한도를 어떻게 검량하는가?

> **난이도 표지:** 출발점은 조건부확률과 베이즈 정리다. 이어서 잠재계층 모형, EM 알고리즘, 로그우도비, CUSUM, 변화점 mixture e-process, Conformal Risk Control까지 간다. 모든 수식은 “LLM 에이전트의 안전 위반을 여러 judge와 사람 감사로 감시한다”는 한 사례에 연결한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 다수결은 gold label이 아닐까?

세 명의 판정자 중 두 명이 “위험”이라고 답했다면 다수결은 위험으로 확정한다. 하지만 다음 세 상황을 구분하지 못한다.

- 두 판정자가 같은 기반 모델과 같은 프롬프트를 사용해 **같은 이유로 함께 틀렸다**.
- 한 판정자는 희귀한 실제 위험을 잘 잡지만, 다른 둘은 거의 모든 사례를 정상이라고 답한다.
- 위험 자체가 매우 드물어 양성 두 표가 나왔어도 사전확률까지 고려하면 불확실성이 남는다.

다수결은 각 표를 똑같이 한 장으로 센다. 반면 확률 모형은 “이 판정자가 실제 사건에서 양성을 낼 확률”과 “정상에서 음성을 낼 확률”을 따로 학습한다. 신뢰도 높은 판정자 한 명의 표가 신뢰도 낮은 판정자 두 명의 표보다 더 많은 정보를 줄 수 있다.

### 1.2 왜 숨은 변수를 도입해야 할까?

항목 $i$의 실제 안전 위반 여부를 $Y_i$라 하자. 우리가 직접 보는 것은 판정자 $r$의 답 $A_{ir}$뿐이고 $Y_i$는 보이지 않는다. 이렇게 직접 관측되지 않지만 관측값을 만들어 냈다고 가정하는 변수를 **잠재변수**라고 한다.

문제는 순환처럼 보인다.

- 진실 $Y_i$를 알아야 각 판정자의 민감도·특이도를 잴 수 있다.
- 판정자의 민감도·특이도를 알아야 숨은 진실 $Y_i$를 믿을 만하게 추정할 수 있다.

EM 알고리즘은 이 순환을 두 개의 계산으로 나눈다. 현재 판정자 성능으로 각 항목의 진실 확률을 계산하고, 그 확률을 “부드러운 개수”로 사용해 판정자 성능을 다시 계산한다. 이를 반복해 서로 일관된 해를 찾는다.

### 1.3 왜 한 번 학습한 오류율을 계속 믿을 수 없을까?

판정자의 성능은 고정된 기계 부품의 규격이 아니다. 다음 변화만으로도 달라질 수 있다.

- 한국어 트래픽 비중이 늘어난다.
- 에이전트의 도구 목록과 시스템 프롬프트가 바뀐다.
- 공격자가 detector가 놓치는 표현을 학습한다.
- LLM judge 제공자가 모델 버전을 조용히 교체한다.
- 사람 검토 지침이나 검토자 구성이 달라진다.

이처럼 데이터 생성 규칙이 시간에 따라 변하는 현상을 **drift**라고 한다. 따라서 “어제의 혼동행렬”을 저장하는 것만으로는 부족하다. 확정 감사 표본에서 오류가 누적되는지 순차적으로 감시하고, 변화가 의심되면 자동 통과를 멈춘 뒤 새 구간에서 다시 검량해야 한다.

### 1.4 왜 변화 탐지와 위험 보장은 서로 다른가?

변화점 검정은 “기준 상태와 달라졌다는 증거가 쌓였는가?”를 묻는다. Conformal Risk Control은 “이 자동화 정책의 평균 손실을 목표치 아래로 제한할 수 있는가?”를 묻는다.

- 변화점 경보가 없다고 해서 위험이 작다는 뜻은 아니다. 작은 변화는 아직 검출력이 부족할 수 있다.
- 위험 검량이 한 번 성공했다고 해서 영원히 유효한 것도 아니다. drift가 생기면 과거 검량 표본과 미래 요청의 교환가능성이 깨진다.

따라서 두 장치는 경쟁 관계가 아니라 직렬 안전장치다. **변화점 감시가 검량 가정의 파손을 알리고, 위험 제어가 정상 구간에서 자동화 범위를 정한다.**

### 1.5 역사적 흐름

1. **반복 판정과 잠재계층:** 의료 진단, 설문, 품질검사에서 정답이 없는 반복 판정을 모아 숨은 계층과 판정자 오류를 함께 추정했다.
2. **EM 알고리즘:** 잠재변수가 있어 직접 최대가능도 계산이 어려운 문제를 기대 단계와 최대화 단계로 나눴다.
3. **Dawid–Skene 모형:** 여러 판정자의 혼동행렬과 잠재 정답을 반복적으로 추정하는 대표 모형이 되었다.
4. **순차 변화 탐지:** 공정 불량률이나 신호 분포가 바뀌는 순간을 CUSUM과 우도비로 빠르게 감지했다.
5. **Conformal Prediction과 Risk Control:** 적은 분포 가정 아래 새 표본의 오류나 사용자 정의 손실을 검량하는 방법으로 확장됐다.
6. **AI 평가·운영:** 사람, LLM judge, 규칙 기반 detector를 모두 오류가 있는 측정기로 보고, 독립 감사와 drift 대응까지 포함한 시스템 설계가 필요해졌다.

## 2. 친절한 용어 사전

| 기호·용어 | 초보자를 위한 뜻 | 오늘의 역할 |
| --- | --- | --- |
| $i=1,\ldots,n$ | 판정할 항목의 번호 | 하나의 응답, 도구 호출, 에이전트 실행을 뜻한다. |
| $r=1,\ldots,R$ | 판정자의 번호 | 사람 검토자, LLM judge, 규칙 detector 등이 될 수 있다. |
| $Y_i\in\{0,1\}$ | 보이지 않는 실제 상태 | $1$은 실제 안전 위반, $0$은 정상이다. |
| $A_{ir}\in\{0,1\}$ | 판정자 $r$이 항목 $i$에 낸 답 | $1$은 위험 판정, $0$은 정상 판정이다. |
| $\mathcal O_i$ | 항목 $i$를 실제로 판정한 판정자 집합 | 누락된 판정을 곱과 합에서 제외한다. |
| $\pi=P(Y_i=1)$ | 실제 사건의 사전확률·유병률 | 판정을 보기 전 위험이 얼마나 드문지 표현한다. |
| $s_r$ | 판정자 $r$의 민감도 $P(A_{ir}=1\mid Y_i=1)$ | 실제 사건을 잡는 확률이다. |
| $c_r$ | 판정자 $r$의 특이도 $P(A_{ir}=0\mid Y_i=0)$ | 실제 정상을 정상으로 두는 확률이다. |
| 잠재변수 | 직접 관측되지 않지만 관측값을 설명하는 변수 | 오늘은 $Y_i$다. |
| 잠재계층 모형 | 각 항목이 보이지 않는 계층 중 하나에서 왔다고 보는 모형 | “실제 사건”과 “실제 정상”이라는 두 성분을 섞는다. |
| 조건부 독립 | $Y_i$를 안 뒤에는 판정자들의 남은 오류가 서로 독립이라는 가정 | 곱 형태의 우도를 가능하게 하지만 현실에서 자주 깨진다. |
| likelihood | 주어진 모수가 현재 관측 데이터를 얼마나 그럴듯하게 만드는지 나타낸 함수 | $\pi,s_r,c_r$를 데이터에 맞추는 목적함수다. |
| log-likelihood | 가능도에 로그를 취한 값 | 곱을 합으로 바꿔 미분과 수치 계산을 쉽게 한다. |
| MLE | maximum likelihood estimate, 가능도를 가장 크게 만드는 모수 | 판정자 성능과 사건률의 추정값이다. |
| $\gamma_i$ | responsibility, $P(Y_i=1\mid\mathbf A_i)$ | 항목 $i$가 실제 사건일 사후확률이며 soft label이다. |
| soft count | 한 항목을 $0$ 또는 $1$로 확정하지 않고 확률만큼 세는 방법 | $\gamma_i=0.8$이면 사건 쪽에 $0.8$개, 정상 쪽에 $0.2$개를 더한다. |
| EM | Expectation–Maximization | E-step에서 $\gamma_i$를 구하고 M-step에서 모수를 갱신한다. |
| E-step | 숨은 변수의 조건부기댓값을 계산하는 단계 | 현재 모수로 각 항목의 사건 확률을 구한다. |
| M-step | 기대 완전자료 로그우도를 최대화하는 단계 | soft count로 $\pi,s_r,c_r$를 다시 센다. |
| label switching | 두 잠재계층의 이름을 바꿔도 같은 관측분포가 되는 대칭 | anchor나 “판정자는 평균적으로 무작위보다 낫다”는 방향 가정이 필요하다. |
| identifiability | 무한한 데이터가 있어도 관측분포에서 모수를 하나로 정할 수 있는 성질 | 판정자 수만 늘린다고 자동으로 생기지 않는다. |
| anchor item | 실제 상태를 신뢰할 수 있게 확정한 항목 | 계층의 방향을 정하고 모형을 검증한다. |
| adjudication | 상충하는 판정을 근거와 절차로 재심해 확정하는 과정 | CRC와 drift 감시용 독립 gold label을 만든다. |
| pseudocount | 분자·분모에 작은 사전 개수를 더하는 안정화 | 표본이 적을 때 민감도나 특이도가 정확히 $0$ 또는 $1$로 붙는 것을 줄인다. |
| missingness | 일부 항목에 일부 판정이 없는 상태 | 누락 선택이 진실과 관련되면 별도 모형이 필요하다. |
| drift | 데이터나 오류 구조가 시간에 따라 달라지는 현상 | 과거의 $s_r,c_r$와 위험 검량을 낡게 만든다. |
| change point $\nu$ | 분포가 바뀌기 시작한 미지의 시점 | 여러 가능한 $\nu$를 혼합해 순차 검정한다. |
| $D_t$ | 시점 $t$의 확정 오류 지시자 | 판정이 gold label과 다르면 $1$, 같으면 $0$이다. |
| likelihood ratio | 변화 후 모형이 기준 모형보다 데이터를 얼마나 더 잘 설명하는지 나타내는 비 | 순차 증거를 곱해 간다. |
| Page CUSUM | 양의 로그우도비를 누적하고 음수가 되면 $0$으로 재시작하는 통계량 | 최근에 시작된 악화를 빠르게 찾는다. |
| e-process | 귀무가설 아래 기대값이 $1$ 이하인 비음수 증거 과정 | 언제 확인해도 유효한 변화점 경보를 만들 수 있다. |
| exchangeability | 표본 순서를 바꿔도 결합분포가 같은 성질 | calibration 표본과 미래 표본을 같은 안정 구간에서 뽑았다는 핵심 가정이다. |
| $\lambda$ | 자동화 정책의 보수성 임계값 | 클수록 더 많이 보류·사람 검토한다고 정한다. |
| $\ell_i(\lambda)$ | 정책 $\lambda$가 항목 $i$에서 낸 bounded loss | 예: 위험 응답을 자동 통과시키면 $1$, 아니면 $0$이다. |
| $R(\lambda)$ | 미래 평균 위험 $\mathbb E[\ell(\lambda)]$ | 목표 한도 $\rho$ 아래로 제어하려는 값이다. |
| abstention | 모델이 억지로 결정하지 않고 보류하는 행동 | 사람 검토로 보내 위험을 낮추는 대신 자동화율이 줄어든다. |
| calibration set | 정책 임계값을 정하는 독립 표본 | 반드시 충분히 확정된 결과를 포함해야 한다. |
| Conformal Risk Control | 교환가능한 검량 표본으로 사용자 정의 평균 손실을 제어하는 방법 | 자동 통과와 사람 검토의 경계를 정한다. |
| $\rho$ | 허용할 목표 위험 | 예를 들어 unsafe auto-pass 비율 목표 $5\%$다. |

오늘의 기본 모형은 항목들이 공통 사건률 $\pi$ 아래 독립적으로 관측되고, 판정자의 민감도·특이도가 항목마다 같으며, $Y_i$를 조건으로 판정자 오류가 독립이고, 누락된 판정이 무시 가능하다고 가정한다. 같은 사용자의 반복 요청, 같은 계열 LLM judge, 공유 프롬프트, 항목 난이도, 선택적 사람 검토가 있으면 이 가정을 별도로 점검해야 한다.

## 3. 수학의 해부학 (증명과 원리)

### 3.1 반복 판정자를 두 성분 혼합모형으로 쓰기

각 항목의 잠재 진실을

$$
Y_i\sim\operatorname{Bernoulli}(\pi)
$$

로 둔다. 판정자 $r$의 민감도와 특이도는

$$
s_r=P(A_{ir}=1\mid Y_i=1),
$$

$$
c_r=P(A_{ir}=0\mid Y_i=0)
$$

다. 따라서 실제 사건일 때 판정 벡터의 확률은

$$
B_{i1}
=
\prod_{r\in\mathcal O_i}
s_r^{A_{ir}}(1-s_r)^{1-A_{ir}},
$$

실제 정상일 때는

$$
B_{i0}
=
\prod_{r\in\mathcal O_i}
(1-c_r)^{A_{ir}}c_r^{1-A_{ir}}
$$

다. 지수 표기는 두 경우를 한 줄에 쓴 것이다. 예를 들어 $A_{ir}=1$이면 첫 식에서 $s_r$만 남고, $A_{ir}=0$이면 $1-s_r$만 남는다.

$Y_i$는 보이지 않으므로 전체확률법칙으로 주변화한다.

$$
P(\mathbf A_i)
=
\pi B_{i1}+(1-\pi)B_{i0}.
$$

전체 관측자료 가능도는

$$
L(\theta)
=
\prod_{i=1}^{n}
\left[
\pi B_{i1}+(1-\pi)B_{i0}
\right],
$$

여기서

$$
\theta=(\pi,s_1,c_1,\ldots,s_R,c_R)
$$

다. 각 항목의 대괄호 안에 합이 있어 로그를 취해도 모수별로 깔끔하게 분리되지 않는다. 이것이 EM을 사용하는 이유다.

### 3.2 완전자료라면 쉬운 문제다

만약 $Y_i$까지 보인다면 항목 $i$의 결합확률은

$$
P(Y_i,\mathbf A_i)
=
\left(\pi B_{i1}\right)^{Y_i}
\left((1-\pi)B_{i0}\right)^{1-Y_i}
$$

다. 로그를 취하면

$$
\begin{aligned}
\log P(Y_i,\mathbf A_i)
&=Y_i\log\pi+(1-Y_i)\log(1-\pi)\\
&\quad+Y_i\sum_{r\in\mathcal O_i}
\left[
A_{ir}\log s_r+(1-A_{ir})\log(1-s_r)
\right]\\
&\quad+(1-Y_i)\sum_{r\in\mathcal O_i}
\left[
A_{ir}\log(1-c_r)+(1-A_{ir})\log c_r
\right].
\end{aligned}
$$

숨은 $Y_i$만 알면 사건 개수, 실제 사건에서의 양성 개수, 실제 정상에서의 음성 개수를 세는 문제가 된다. EM은 보이지 않는 $Y_i$ 대신 그 조건부기댓값을 넣는다.

### 3.3 E-step: 베이즈 정리로 responsibility 구하기

현재 모수를 $\theta^{\text{old}}$라고 하자. 항목 $i$가 실제 사건일 사후확률은

$$
\begin{aligned}
\gamma_i
&=P(Y_i=1\mid\mathbf A_i,\theta^{\text{old}})\\
&=
\frac{
\pi B_{i1}
}{
\pi B_{i1}+(1-\pi)B_{i0}
}.
\end{aligned}
$$

이 식은 “사건 사전확률 $\times$ 사건일 때 이 판정 패턴이 나올 확률”을 두 가능한 계층의 합으로 나눈 베이즈 정리다.

로그오즈로 바꾸면 각 판정자의 기여가 더 잘 보인다.

$$
\begin{aligned}
\operatorname{logit}(\gamma_i)
&=\log\frac{\gamma_i}{1-\gamma_i}\\
&=\log\frac{\pi}{1-\pi}\\
&\quad+\sum_{r\in\mathcal O_i}
\left[
A_{ir}\log\frac{s_r}{1-c_r}
+(1-A_{ir})\log\frac{1-s_r}{c_r}
\right].
\end{aligned}
$$

- 양성 판정은 $\log\frac{s_r}{1-c_r}$만큼 증거를 더한다.
- 음성 판정은 $\log\frac{1-s_r}{c_r}$만큼 증거를 더한다.
- 사건이 희귀하면 $\log\frac{\pi}{1-\pi}$가 큰 음수에서 시작한다.

즉 단순 표 수가 아니라 **각 판정의 우도비**를 더하는 셈이다.

### 3.4 M-step: soft count로 혼동행렬 다시 세기

E-step에서 $Y_i$의 조건부기댓값은

$$
\mathbb E[Y_i\mid\mathbf A_i]=\gamma_i
$$

다. 완전자료 로그우도에서 $Y_i$를 $\gamma_i$로 바꾼 기대값을 $Q(\theta\mid\theta^{\text{old}})$라고 하자.

$\pi$와 관련된 항만 모으면

$$
Q_\pi
=
\sum_{i=1}^{n}
\left[
\gamma_i\log\pi+(1-\gamma_i)\log(1-\pi)
\right]
$$

다. 미분해 $0$으로 놓으면

$$
\frac{\partial Q_\pi}{\partial\pi}
=
\sum_i\frac{\gamma_i}{\pi}
-
\sum_i\frac{1-\gamma_i}{1-\pi}
=0,
$$

따라서

$$
\boxed{
\pi^{\text{new}}
=
\frac{1}{n}\sum_{i=1}^{n}\gamma_i
}
$$

다.

같은 방식으로 판정자 $r$이 실제로 판정한 항목 집합을 $\mathcal I_r$라 하면

$$
\boxed{
s_r^{\text{new}}
=
\frac{
\sum_{i\in\mathcal I_r}\gamma_i A_{ir}
}{
\sum_{i\in\mathcal I_r}\gamma_i
}
}
$$

이고,

$$
\boxed{
c_r^{\text{new}}
=
\frac{
\sum_{i\in\mathcal I_r}(1-\gamma_i)(1-A_{ir})
}{
\sum_{i\in\mathcal I_r}(1-\gamma_i)
}
}
$$

다. 분자는 각각 soft true positive와 soft true negative이고, 분모는 soft actual positive와 soft actual negative다.

E-step과 M-step을 반복하면 관측자료 로그우도는 감소하지 않는다. 그러나 이것이 참 모수를 찾았다는 뜻은 아니다. 초기값에 따라 다른 국소해에 도달할 수 있고, 모형 가정이 틀리면 매우 확신에 찬 오답을 만들 수도 있다.

### 3.5 손으로 계산하는 한 항목의 사후확률

실제 위험률을 $\pi=0.02$라 하고 세 판정자의 성능을 다음과 같이 두자.

$$
(s_1,s_2,s_3)=(0.9,0.8,0.7),
$$

$$
(c_1,c_2,c_3)=(0.98,0.95,0.9).
$$

세 판정이 $(1,1,0)$이었다. 다수결은 바로 위험으로 확정한다. 잠재계층 모형에서는

$$
B_{i1}
=0.9\times0.8\times(1-0.7)
=0.216
$$

이고,

$$
B_{i0}
=(1-0.98)\times(1-0.95)\times0.9
=0.0009
$$

다. 따라서

$$
\begin{aligned}
\gamma_i
&=
\frac{0.02\times0.216}
{0.02\times0.216+0.98\times0.0009}\\
&=\frac{0.00432}{0.005202}\\
&\approx0.8304.
\end{aligned}
$$

두 개의 양성 표는 강한 증거지만 결과는 $100\%$ 확정이 아니라 약 $83\%$다. 희귀 사건이라는 사전확률과 세 번째 판정자의 음성 표가 함께 반영되기 때문이다.

### 3.6 식별가능성: 표가 많아도 진실을 못 찾을 수 있다

#### 판정자가 두 명뿐인 경우

두 판정자의 이진 출력에는 네 패턴이 있지만 확률의 합이 $1$이므로 관측분포의 자유도는

$$
2^2-1=3
$$

이다. 반면 모수는

$$
1+2R=1+2\times2=5
$$

개다. 일반적으로 관측 정보보다 미지수가 많아 식별할 수 없다.

판정자가 세 명이면 관측 자유도와 모수 개수가 모두

$$
2^3-1=7,
\qquad
1+2\times3=7
$$

이 된다. 하지만 개수 일치는 필요조건일 뿐 충분조건이 아니다. 판정자들이 정보를 주고, 혼동행렬이 퇴화하지 않으며, 조건부 독립 같은 구조가 맞아야 일반적으로 식별할 수 있다.

#### label switching

다음 변환을 생각하자.

$$
\pi'=1-\pi,
\qquad
s_r'=1-c_r,
\qquad
c_r'=1-s_r.
$$

이는 “사건”과 “정상”이라는 잠재계층 이름만 바꾼 것이다. 관측분포는 변하지 않는다. 따라서 일부 anchor item의 진실을 알거나, 충분한 판정자에 대해

$$
s_r+c_r>1
$$

처럼 무작위보다 좋은 방향이라는 가정이 있어야 어느 계층을 사건이라고 부를지 정할 수 있다.

#### 상관된 오류

같은 기반 모델, 같은 프롬프트, 같은 검색 결과를 쓰는 LLM judge 셋은 $Y_i$를 알아도 함께 틀릴 수 있다. 이때 조건부 독립 모형은 같은 증거를 세 번 센다. 판정자 수는 $3$이지만 유효한 독립 정보는 $1$에 가까울 수 있으며, $\gamma_i$가 지나치게 $0$이나 $1$에 붙는다.

이를 완화하려면 판정자 계열별 random effect, 항목 난이도, 상관 오차, 서로 다른 prompt family를 모델링하거나 최소한 같은 계열을 하나의 군집으로 취급해야 한다.

### 3.7 drift를 로그우도비로 찾기

확정 감사 라벨이 성숙한 순서대로 detector의 오류를

$$
D_t=\mathbf 1\{Z_t\neq Y_t\}
$$

로 기록하자. 기준 오류율은 $e_0$, 감지하려는 악화 오류율은 $e_1$이라 하고

$$
0<e_0<e_1<1
$$

로 둔다. 단순한 Bernoulli 변화 모형에서는 과거 정보 $\mathcal F_{t-1}$이 주어졌을 때

$$
D_t\mid\mathcal F_{t-1}
\sim
\operatorname{Ber}(e_j),
\qquad j\in\{0,1\}
$$

라고 가정한다.

한 관측의 변화 후 대 변화 전 로그우도비는

$$
\ell_t
=
D_t\log\frac{e_1}{e_0}
+
(1-D_t)\log\frac{1-e_1}{1-e_0}.
$$

Page CUSUM은

$$
G_0=0,
\qquad
G_t=\max\left(0,G_{t-1}+\ell_t\right)
$$

로 정의한다. 기준 구간에서는

$$
\mathbb E_{e_0}[\ell_t]
=-\operatorname{KL}\!\left(
\operatorname{Ber}(e_0)
\,\|\,
\operatorname{Ber}(e_1)
\right)
<0,
$$

악화 후에는

$$
\mathbb E_{e_1}[\ell_t]
=\operatorname{KL}\!\left(
\operatorname{Ber}(e_1)
\,\|\,
\operatorname{Ber}(e_0)
\right)
>0
$$

다. 그래서 정상 시기에는 통계량이 $0$으로 돌아가고, 악화 후에는 위로 자란다.

예를 들어 $e_0=0.05$, $e_1=0.10$이면 오류 한 번의 증분은

$$
\log\frac{0.10}{0.05}=\log2\approx0.693
$$

이고, 정상 판정 한 번의 증분은

$$
\log\frac{0.90}{0.95}\approx-0.054
$$

다. 드문 오류가 연속되면 최근 변화의 증거가 빠르게 쌓인다.

그러나 전체 오류율은 detector의 민감도 $s$, 특이도 $c$, 실제 사건률 $\pi$가 함께 만든다.

$$
e=P(Z\neq Y)
=
\pi(1-s)+(1-\pi)(1-c).
$$

따라서 $e$의 변화만으로 detector 자체가 나빠졌다고 식별할 수 없다. $\pi$만 바뀌어도 전체 오류율은 움직인다. detector drift를 겨냥하려면 확정 양성 스트림의 FNR $1-s$와 확정 음성 스트림의 FPR $1-c$를 별도로 감시하고, 사건률 $\pi$도 따로 기록해야 한다.

중요한 주의점이 있다. CUSUM 임계값 $h$는 그 자체로 “언제 보아도 거짓 경보 확률이 $\alpha$ 이하”라는 뜻이 아니다. 보통 평균 경보 간격이나 탐지 지연 목표에 맞춰 설계·시뮬레이션한다.

### 3.8 미지의 변화시점을 mixture e-process로 감싸기

변화시점 후보를 $\nu=1,2,\ldots$라 하자. 시점 $u$의 과거 정보를 $\mathcal F_{u-1}$로 쓰고, 귀무·대립 조건부분포를 각각 $f_{0,u}(\cdot\mid\mathcal F_{u-1})$, $f_{1,u}(\cdot\mid\mathcal F_{u-1})$라 하자. 대립분포가 귀무분포에 대해 절대연속인

$$
f_{1,u}\ll f_{0,u}
$$

경우를 생각한다. $t<\nu$일 때 $L_{\nu,t}=1$로 두고, $t\ge\nu$이면

$$
L_{\nu,t}
=
\prod_{u=\nu}^{t}
\frac{
f_{1,u}(W_u\mid\mathcal F_{u-1})
}{
f_{0,u}(W_u\mid\mathcal F_{u-1})
}
$$

로 둔다. 여기서 $W_u$는 감사 결과다. 귀무가설 아래 분모가 실제 조건부분포이고 분자의 선택이 사전에 고정되거나 predictable하다면 각 밀도비의 조건부기댓값은 $1$이다.

데이터를 보기 전에 정한 가중치가

$$
w_\nu\ge0,
\qquad
\sum_{\nu=1}^{\infty}w_\nu=1
$$

을 만족하면

$$
\begin{aligned}
M_t
&=\sum_{\nu=1}^{\infty}w_\nu L_{\nu,t}\\
&=\sum_{\nu\le t}w_\nu L_{\nu,t}
+\sum_{\nu>t}w_\nu
\end{aligned}
$$

는 이 **정확히 지정된 단순 귀무가설** 아래 비음수 martingale이다. 따라서 Ville 부등식으로

$$
P_0\left(
\sup_tM_t\ge\frac{1}{\alpha_{\mathrm{chg}}}
\right)
\le\alpha_{\mathrm{chg}}
$$

를 얻는다. 이것은 여러 가능한 변화 시작점을 미리 정한 prior로 평균내면서도 anytime-valid 경보를 유지하는 한 방법이다.

단, $f_0,f_1$의 nuisance parameter를 같은 운영 데이터의 미래까지 보고 유리하게 고르면 이 증명은 깨진다. 독립 calibration에서 $\widehat f_0$를 추정해 고정하는 것은 데이터 누출을 막지만, $\widehat f_0$가 실제 귀무 조건부분포와 다르면 정확한 martingale을 자동으로 만들지 않는다. composite null에는 모든 허용 모수에서 유효한 e-process, 신뢰집합 위 최악값, 또는 nuisance를 안전하게 제거하는 별도 증명이 필요하다.

gold label이 즉시 없고 반복 판정만 있다면 관측값의 분포를 잠재 진실에 대해 주변화할 수 있다.

$$
f_j(z,\mathbf a)
=
\sum_{y\in\{0,1\}}
P(Y=y)
P_j(Z=z\mid Y=y)
\prod_{r\in\mathcal O}P(A_r=a_r\mid Y=y),
\qquad j\in\{0,1\}.
$$

이 식은 $P(Y)$와 반복 판정자의 $P(A_r\mid Y)$는 두 가설에서 같고, 감시 대상 detector의 $P_j(Z\mid Y)$만 달라진다는 detector-drift 대립을 쓴다. 모든 항에 $j$를 붙이면 사건률·반복 판정자·detector 중 무엇이 변했는지 구분하지 못하는 joint-distribution drift 검정이 된다.

어느 경우든 우도비의 유효성은 잠재모형이 맞고 모수 불확실성을 올바르게 처리했다는 조건부 보장이다. 독립적으로 확정된 감사 스트림을 완전히 대체하지는 않는다.

### 3.9 Conformal Risk Control로 자동화 임계값 정하기

$\lambda$가 클수록 더 많은 사례를 보류해 사람에게 보내고, 손실이 단조롭게 감소한다고 하자. 예를 들어

$$
\ell_i(\lambda)
=
\mathbf 1\{
Y_i=1\text{이고 정책이 auto-pass}
\}
$$

로 두면 위험은 **전체 요청 중 unsafe auto-pass 비율**이다. 이는 $P(\text{unsafe}\mid\text{auto-pass})$처럼 auto-pass만을 분모로 둔 조건부 비율과 다르다. 후자의 분모는 $\lambda$와 함께 바뀌므로 기본 CRC에 필요한 단조성이 자동으로 보장되지 않는다.

일반적인 손실은

$$
0\le\ell_i(\lambda)\le B
$$

이고, 미래 위험은

$$
R(\lambda)=\mathbb E[\ell(\lambda)]
$$

다. 독립적으로 확정한 calibration 표본 $n$개에서 경험위험은

$$
\widehat R_n(\lambda)
=
\frac{1}{n}\sum_{i=1}^{n}\ell_i(\lambda)
$$

다.

기본 CRC 정리를 쓰기 위해 다음을 가정한다.

1. calibration $n$개와 새 테스트 항목 하나가 만드는 손실함수 $\ell_1(\cdot),\ldots,\ell_{n+1}(\cdot)$는 교환가능하다.
2. 각 $\ell_i(\lambda)$는 $\lambda$에 대해 비증가하고 우연속이다.
3. 거의 확실하게 $\sup_\lambda\ell_i(\lambda)\le B$다.
4. 가장 보수적인 $\lambda_{\max}\in\Lambda$에서 $\ell_i(\lambda_{\max})\le\rho$다. 예를 들어 모든 요청을 review하면 unsafe auto-pass 손실은 $0$이다.
5. detector, score, 후보 정책 가족은 calibration 라벨을 보기 전에 고정한다.

이 조건 아래 다음 보정값을 사용한다.

$$
\widehat R_n^{+}(\lambda)
=
\frac{n}{n+1}\widehat R_n(\lambda)
+\frac{B}{n+1}.
$$

그리고 가능한 집합이 비어 있지 않으면

$$
\widehat\lambda
=
\inf\left\{
\lambda:
\widehat R_n^{+}(\lambda)\le\rho
\right\}
$$

를 선택한다. $\inf$는 조건을 만족하는 가장 작은 보수성 값을 뜻한다. 가능한 값이 하나도 없으면 $\widehat\lambda=\lambda_{\max}$로 둔다. 핵심 보장은 새 테스트 손실에 대한

$$
\boxed{
\mathbb E\left[\ell_{n+1}(\widehat\lambda)\right]
\le\rho
}
$$

다. calibration과 새 테스트 항목이 iid인 해석에서

$$
R(\lambda)
=
\mathbb E[\ell_{n+1}(\lambda)\mid\lambda]
$$

로 두면 이를

$$
\mathbb E_{\mathrm{cal}}
\left[R(\widehat\lambda)\right]
\le\rho
$$

라고도 쓸 수 있다. 즉 바깥 기대값은 calibration 표본의 무작위성까지 평균낸다.

$B=1$, $n=99$이고 어떤 정책에서 unsafe auto-pass가 $k=4$건이었다면

$$
\widehat R_n^{+}
=
\frac{99}{100}\frac{4}{99}
+\frac{1}{100}
=\frac{5}{100}
=0.05.
$$

따라서 목표가 $\rho=0.05$라면 경계에서 통과한다. 하지만 이것은 “$95\%$ 신뢰도로 실제 위험이 $5\%$ 이하”라는 고확률 진술이 아니다. 그런 보장이 필요하면 별도의 PAC형 상한이나 신뢰경계를 설계해야 한다.

또한 $4$건을 확정 라벨이 아니라 EM의 $\gamma_i$를 더한 soft error로 계산했다면 distribution-free CRC 보장이 자동으로 따라오지 않는다. 그것은 잠재모형이 맞다는 가정에 의존하는 모델 기반 위험이다.

### 3.10 오늘 수학의 한 줄 연결

| 수학 부품 | 계산하는 것 | AI 운영에서 맡는 역할 |
| --- | --- | --- |
| 전체확률법칙 | 숨은 $Y_i$를 합으로 제거 | 실제 진실 없이도 반복 판정의 관측 가능도를 쓴다. |
| 베이즈 정리 | $\gamma_i=P(Y_i=1\mid\mathbf A_i)$ | 항목별 모형 기반 soft label을 만든다. |
| 로그우도 최대화 | $\pi,s_r,c_r$ | 판정자별 오류 구조를 학습한다. |
| KL 발산 | 변화 전후 기대 로그우도비 | drift가 생기면 증거가 왜 양의 속도로 쌓이는지 설명한다. |
| martingale·Ville 부등식 | $M_t$의 anytime 임계값 | dashboard를 반복 확인해도 변화점 거짓 경보를 통제한다. |
| 교환가능성·CRC | $\widehat\lambda$의 기대 위험 | 자동 통과와 사람 검토 사이의 운영 경계를 정한다. |

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 전체 구조도: 추정과 보장을 분리하라

```text
새 응답·도구 실행 x
        │
        ├── 운영 detector Z ───────────────┐
        │                                  │
        ├── 서로 다른 반복 판정자 A₁...Aᴿ ├── 잠재계층·EM → 사건 posterior γ
        │                                  │              │
        └── 사전 지정 표본의 독립 adjudication Y ◀─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          변화점 e-process             CRC calibration
          오류 구조가 변했나?          자동 통과 위험이 목표 이내인가?
                 │                         │
                 └──── PASS / REVIEW / HOLD / ROLLBACK
```

핵심은 두 층을 섞지 않는 것이다.

- **모형 기반 추정 층:** 많은 값싼 판정으로 $\gamma_i$를 만들고 감사 우선순위를 정한다.
- **보장 층:** 독립 adjudication으로 실제 손실을 측정해 drift와 자동화 위험을 검증한다.

운영 detector $Z$를 판정자 패널에 넣은 뒤 같은 posterior로 $Z$의 성능을 평가하면 자기 답을 정답에 섞는 순환이 생긴다. 평가용 audit은 detector와 독립된 절차로 분리해야 한다.

### 4.2 실제로 EM은 어떻게 동작할까?

초기값을 정한 뒤 다음을 반복한다.

```text
1. E-step
   각 항목 i에 대해 현재 판정자 성능으로 γᵢ를 계산한다.

2. M-step
   γᵢ를 soft count로 사용해 사건률과 판정자별 민감도·특이도를 갱신한다.

3. 수렴 확인
   관측 로그우도의 증가가 매우 작아지면 멈춘다.

4. 별도 검증
   anchor/adjudicated holdout에서 calibration, 민감도, 특이도와 subgroup 오차를 본다.
```

실무에서는 여러 초기값으로 반복하고, $s_r,c_r$에 Beta prior나 pseudocount를 두며, 항목 난이도와 판정자 계열을 반영한 확장 모형도 비교해야 한다.

### 4.3 배포 파이프라인

1. **사건 정의를 고정한다.** “유해함”처럼 모호한 말 대신 실제 위반 조건과 증거 규칙을 문서화한다.
2. **대표성 있는 anchor set을 만든다.** EM 학습·진단을 위해 언어·도구·고객군·난이도를 층화하고 복수 전문가가 adjudication한다.
3. **판정자 다양성을 확보한다.** 같은 기반 모델의 temperature만 바꾼 복제본을 독립 판정자처럼 세지 않는다.
4. **EM은 학습용 구간에서만 적합한다.** 초기값을 여러 개 사용하고 label switching 방향을 anchor로 고정한다.
5. **확정 holdout으로 검증한다.** soft posterior가 아니라 실제 adjudication을 기준으로 reliability diagram과 subgroup 혼동행렬을 본다.
6. **CRC용 표본을 분리한다.** detector·score·정책 가족을 먼저 고정한 뒤, 미래 트래픽과 교환가능한 별도 무작위 calibration 표본을 adjudication한다. 불균등 층화 표본을 쓰려면 검증된 weighted·non-exchangeable CRC가 필요하다.
7. **CRC로 자동화 경계를 정한다.** 목표 위험 $\rho$뿐 아니라 review rate와 처리 지연도 함께 기록한다.
8. **운영 중 무작위 감사를 유지한다.** detector 양성만 검토하면 verification bias가 생기므로 음성·자동 통과 사례도 알려진 확률로 표집한다. 이 monitoring 표본이 곧 기본 CRC용 교환가능 표본이 되는 것은 아니다.
9. **성숙 라벨로 drift를 감시한다.** 결과가 늦게 오는 사례는 pending으로 두고 reveal time을 기록한다.
10. **경보 시 HOLD한다.** 새 regime에서 충분한 확정 라벨을 모으고 EM과 CRC를 다시 적합한 뒤 재개한다.

### 4.4 사건률 drift와 detector drift를 분리하기

관측 경보율이 올라간 이유는 최소 두 가지다.

- 실제 위험률 $\pi$가 증가했다.
- detector의 위양성률 $1-c_r$가 증가했다.

경보 $A_{ir}$만 보면 둘을 구별할 수 없다. 무작위 adjudication 표본에서 실제 $Y_i$를 확인해야 한다. 가능하면 다음 스트림을 따로 감시한다.

- 실제 양성 중 놓친 비율: $1-s_r$
- 실제 음성 중 잘못 울린 비율: $1-c_r$
- 실제 사건률: $\pi$
- 자동 통과 정책의 최종 손실: $R(\lambda)$

하나의 전체 accuracy로 합치면 희귀 사건과 subgroup 악화를 숨길 수 있다.

### 4.5 CRC는 안전과 자동화율의 교환이다

$\lambda$를 높여 모든 사례를 사람에게 보내면 자동 통과 손실은 쉽게 낮아진다. 하지만 비용과 지연이 폭증한다. 그래서 최소한 다음을 함께 보고해야 한다.

| 지표 | 질문 |
| --- | --- |
| risk | 전체 요청 중 unsafe인데 auto-pass된 비율은 얼마인가? |
| review rate | 전체 중 몇 퍼센트를 사람이 보는가? |
| coverage·automation rate | 모델이 스스로 처리한 비율은 얼마인가? |
| latency | 보류된 사례가 결론까지 걸린 시간은 얼마인가? |
| subgroup risk | 언어·도구·고객군별 위험 한도가 지켜지는가? |
| pending count | 아직 결과가 성숙하지 않은 사례가 얼마나 쌓였는가? |

위험만 낮고 거의 모든 요청을 보류하는 정책은 통계적으로 안전해 보여도 유용한 자동화라고 할 수 없다.

### 4.6 초보자가 흔히 하는 오해와 주의할 점

1. **“다수결이면 정답이다.”** 판정자 오류가 상관되면 같은 오답을 여러 번 센다.
2. **“EM이 진짜 라벨을 발견한다.”** EM은 선택한 잠재모형에 가장 잘 맞는 soft label을 만든다. 모형이 틀리면 진실 보장은 없다.
3. **“$\gamma_i=0.99$면 실제로 $99\%$ 확실하다.”** 모수 추정 불확실성과 모형 오지정은 이 숫자에 자동 반영되지 않는다.
4. **“판정자가 세 명이면 항상 식별된다.”** 세 명은 단순 차원 계산의 문턱일 뿐, 조건부 독립·비퇴화·방향 고정이 필요하다.
5. **“누락 판정은 그냥 빼면 된다.”** 어려운 사례만 추가 검토하거나 양성만 adjudication하면 누락이 정보적이다.
6. **“CUSUM 임계값 $h$는 곧 유의수준이다.”** 일반 Page CUSUM의 $h$는 보통 평균 경보 간격과 탐지 지연으로 설계한다.
7. **“변화점 경보가 없으니 drift가 없다.”** 비경보는 아직 증거가 충분하지 않다는 뜻이지 동일성의 증명이 아니다.
8. **“CRC 한 번이면 영구 보장이다.”** 배포 분포가 바뀌면 교환가능성이 깨져 재검량해야 한다.
9. **“soft label로 CRC를 해도 distribution-free다.”** 실제 $Y$가 아닌 EM posterior를 손실로 쓰면 잠재모형 의존 보장으로 바뀐다.
10. **“전체 평균만 맞으면 안전하다.”** 희귀한 고위험 세그먼트가 평균에 묻힐 수 있다. 여러 세그먼트를 동시에 보면 전날 배운 family 오류 예산도 계속 적용해야 한다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **조사 기준 시각:** 2026-09-29 02:33 KST. 발표사·저자의 공식 글, 모델·데이터 카드, 논문, 공식 저장소를 기준으로 확인했다. 아래 수치는 별도 표시가 없으면 발표사 또는 저자 보고값이며 독립 재현 결과가 아니다.

### 5.1 Holo4: GUI·코드·MCP/API를 한 정책으로 묶은 computer-use VLM

H Company는 2026년 9월 28일 [Holo4](https://hcompany.ai/newsroom/holo4)를 공개했다. 계열은 Qwen3.8 기반 **27B dense** 모델과 Qwen3.6 기반 **35B-A3B MoE** 모델로 구성된다. 후자는 총 35B 중 토큰당 약 3B가 활성화된다. 두 모델 모두 설정상 262,144-token context를 가지며, 화면 클릭·타이핑뿐 아니라 코드 실행과 MCP/API 호출을 같은 모델 인터페이스에서 수행하도록 설계됐다.

공식 설명에 따르면 약 $10{,}000$개의 검증 가능한 태스크를 Agentic Task Factory로 만들었고, 127B-token SFT 뒤 desktop/web 전문가와 terminal/MCP/API 전문가라는 두 LoRA를 비동기 online RL로 학습해 동일 비중으로 병합했으며 추가 학습은 하지 않았다.

#### 핵심 결과와 조건

| 평가 | Holo4-27B | Holo4-35B-A3B | 반드시 함께 읽을 조건 |
| --- | ---: | ---: | --- |
| OSWorld | $85.2\%$ | $80.8\%$ | 발표사 harness, 2–4회 실행 평균 |
| OSWorld 2.0 partial / success | $61.7\%$ / $41.5\%$ | $30.9\%$ / $12.3\%$ | 단일 실행이라 분산을 알 수 없음 |
| AutomationBench public 600 | $45.4\%$ | $34.5\%$ | 600개 중 480개는 학습 데이터 수집 split과 겹침 |
| AutomationBench held-out 120 | $49.3\%$ | $31.7\%$ | 대응 base model은 각각 $40.3\%$, $13.1\%$ |

발표사는 OSWorld 2.0 비용을 태스크당 각각 \$1.22와 \$0.61로 계산했지만, 모델별 harness·effort·benchmark release가 달라 폐쇄형 모델과의 점수·비용을 완전한 동조건 비교로 보면 안 된다. 특히 공식 글은 OSWorld 2.0과 ALE-CLI가 단일 실행임을 명시한다.

#### 공개 범위

- [Holo4-27B 모델 카드](https://huggingface.co/Hcompany/Holo4-27B): BF16·FP8·NVFP4·Q4 GGUF 가중치, **CC BY-NC 4.0**. 공개 가중치이지만 상업 이용 제한이 있어 “완전한 오픈소스”라고 뭉뚱그리면 안 된다.
- [Holo4-35B-A3B 모델 카드](https://huggingface.co/Hcompany/Holo4-35B-A3B): 주요 정밀도·양자화 가중치, **Apache 2.0**.
- [Holo4 trajectories](https://huggingface.co/datasets/Hcompany/trajectories): 7,366개 실행의 reasoning·action·tool result·screenshot을 담은 2.74GB 데이터, **Apache 2.0**. upstream task content는 원 라이선스를 유지한다. 자격증명·내부 host·개인정보 텍스트는 `<PII removed>`로 바꾸고 해당 screenshot은 placeholder로 교체했으며, 일부 태스크는 제외했다.

#### 엔지니어 인사이트 (Impact)

첫째, computer-use의 경쟁 단위가 “스크린만 보는 GUI 모델”에서 **GUI·코드·도구 호출을 상황에 따라 바꾸는 통합 정책**으로 이동하고 있다. Apache 2.0인 35B-A3B와 양자화 체크포인트는 로컬 agent 실험의 문턱을 낮춘다.

둘째, 이 점수는 순수한 가중치 점수가 아니다. 공식 글 자체가 memory compaction, desktop shell, step/time budget 확장이 성능에 영향을 줬다고 설명한다. 따라서 재현 단위는

$$
\text{agent system}
=
\text{model}
+\text{memory}
+\text{harness}
+\text{tools}
+\text{budget}
$$

이어야 한다.

셋째, 7,366개 trajectory 공개는 오늘 배운 반복 판정·drift 연구에 특히 가치가 있다. 성공 점수만 보는 대신 실패 단계, verifier 결과, token 사용량을 다시 분석할 수 있기 때문이다. 다만 원 학습 trajectory 전체, RL 코드·하이퍼파라미터, verifier와 평가 harness 전체가 재학습 가능한 형태로 공개된 것은 아니므로 “완전 재현 가능”과는 구분해야 한다.

### 5.2 DIAL: LLM judge의 위치 편향을 분리하고 제한된 사람 라벨로 보정하기

2026년 9월 25일 12:56 UTC, 즉 21:56 KST에 제출된 [DIAL 논문](https://arxiv.org/abs/2609.31215)은 오늘의 잠재 판정자 문제를 LLM 평가에 직접 적용한다. $i<j$를 canonical orientation으로 두고, $a=1$은 응답 $i$가 먼저 표시된 경우, $a=-1$은 $i$가 두 번째로 표시된 경우라 하자. 판정자 $k$의 비교를

$$
\operatorname{logit}
P_k(i\succ j\mid a)
=
s_{ki}-s_{kj}+ab_k
$$

로 모델링한다. $s_{ki}-s_{kj}$는 위치를 제거한 선호 차이다. $b_k>0$이면 첫 번째 표시 응답, $b_k<0$이면 두 번째 표시 응답을 선호하는 판정자별 위치 효과다.

DIAL은 세 단계를 잇는다.

1. 각 LLM judge의 위치 효과와 잠재 선호를 분리한다.
2. 여러 judge의 공통 선호 방향과 저랭크 불일치 방향을 학습한다.
3. 적은 사람 비교 라벨로 그 표현을 사람 선호 목표에 적응적으로 보정한다.

사람 자료와 LLM anchor 사이의 가중치를 $\lambda$라 하면 $\lambda=0$은 human-only endpoint, $\lambda\to\infty$는 LLM-anchored endpoint에 해당한다. DIAL-Ada는 leave-one-human-out loss를 근사하는 GACV로 $\lambda$를 선택해 둘 사이를 이동한다.

#### 데이터와 검증 범위

저자들은 Chatbot Arena, MT-Bench, PandaLM의 세 benchmark에서 22개 LLM judge를 양쪽 표시 순서로 질의했다. 모든 데이터셋에서 tie 비율이 $50\%$를 넘은 한 모델을 제외해 21개 judge를 분석했다. 이 21개 judge의 공개 출력은 tie 포함 $410{,}164$개이고, tie를 제외해 binary BTL에 사용한 판정은 $381{,}302$개다.

| 데이터셋 | 레코드 수 | 결정적 사람 라벨 | 공개 판정, tie 포함 | tie | 이진 분석 $n_l$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chatbot Arena | $7{,}738$ | $5{,}430$ | $321{,}668$ | $21{,}199$ | $300{,}469$ |
| MT-Bench | $1{,}199$ | $842$ | $49{,}172$ | $3{,}289$ | $45{,}883$ |
| PandaLM | $945$ | $842$ | $39{,}324$ | $4{,}374$ | $34{,}950$ |
| 합계 | $9{,}882$ | $7{,}114$ | $410{,}164$ | $28{,}862$ | $381{,}302$ |

주 실데이터 robustness 실험은 swapped copy를 $5\%$만 유지한 one-sided 조건과, 첫 표시 응답을 $0.9$ 확률로 고르는 pure-position judge를 추가한 조건을 사용했다. 이때 order-aware DIAL은 order-agnostic 기준보다 안정적이었다. LLM 판정 예산이 작거나 anti-consensus judge를 넣은 경우에는 adaptive variant가 사람 증거 쪽으로 이동했다. 실데이터 곡선은 최대 50개 record split의 평균과 $\pm1.96$ Monte Carlo standard error이며, 두 표시 순서는 항상 같은 split에 뒀다.

저자 보고 합성 실험의 정확 지정 조건에서는 사람 라벨이 $n_h\le200$일 때 DIAL-Anc가 Human-only보다 초과위험을 8–14배 줄이고 Kendall의 $\tau$를 약 $0.75$에서 $0.98$로 높였다. 이는 합성 설정에서의 결과이며 실서비스 효과 크기로 일반화할 수 없다.

#### 공개 범위와 한계

- [DIAL 공식 저장소](https://github.com/JinHongDu-Lab/DIAL)는 추정기, 실험 코드, 테스트, 준비된 benchmark, raw judge response, 수집 도구를 제공하며 저장소 라이선스는 **MIT**다. DIAL 자체 모델 가중치는 없다.
- 준비된 데이터는 Chatbot Arena·MT-Bench·PandaLM에서 왔고, open-weight judge 가중치와 상용 API 출력도 각 원천에서 왔다. 저장소의 MIT 표기가 이 모든 원천 자료를 자동으로 MIT로 재허가한다는 뜻은 아니므로 각각의 provenance와 이용 조건을 확인해야 한다.
- v1 사전논문이며, binary BTL 분석에서 LLM tie와 사람의 tie·both-bad 라벨을 제외했다. 비대칭 tie 자체가 위치 정보를 가질 수 있다는 점은 후속 과제로 남겼다.
- 여러 LLM judge가 기반 데이터·아키텍처·평가 습관을 공유할 수 있어 “판정자가 많다”와 “독립 증거가 많다”는 같지 않다.
- sandwich covariance에 기반한 이론적 불확실성 결과는 고정된 $\lambda$에 대한 것이다. GACV로 $\lambda$를 고른 뒤의 구간은 정식 post-selection 보장이 아니라 실험적 coverage 진단으로 읽어야 한다.
- 실데이터의 후보 응답 모델은 대부분 과거 세대다. 결과는 2026년 frontier 답변의 순위를 확정하기보다 평가·순위 추정 방법을 검증한다.
- 목표는 pooled human preference이지 철학적으로 유일한 객관적 진실이 아니다. 사람 집단과 언어가 달라지면 다시 보정해야 한다.

#### 엔지니어 인사이트 (Impact)

DIAL의 중요한 메시지는 “더 많은 LLM 판정”보다 **측정 설계**가 먼저라는 점이다. 응답 순서를 바꿔 같은 쌍을 다시 보여 주고, 판정자별 편향을 분리하며, 적은 사람 라벨을 anchor로 쓰면 자동 평가의 값이 달라진다.

Holo4의 공개 trajectory와 DIAL을 함께 보면 다음 원칙이 나온다.

- 에이전트 benchmark는 성공/실패 한 칸뿐 아니라 실행 trace와 verifier 근거를 남긴다.
- 하나의 LLM judge를 gold label로 삼지 말고 서로 다른 계열의 판정과 사람 anchor를 결합한다.
- 표시 순서, prompt template, 언어, harness version을 판정자의 오차를 바꾸는 공변량으로 기록한다.
- drift가 감지되면 과거 judge calibration과 CRC 임계값을 그대로 재사용하지 않는다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

교환가능한 확정 calibration 표본 **전체 $99$개**에서 정책 $\lambda$가 위험 사례를 자동 통과시킨 횟수가 $4$번이었다.

1. $B=1$인 CRC 보정 위험은 얼마인가?
2. 이것이 “$95\%$ 신뢰도로 실제 위험이 $5\%$ 이하”라는 뜻인가?
3. 이후 change-point 경보가 발생했거나, $4$건을 확정 라벨 대신 EM posterior의 soft count로 계산했다면 같은 보장을 유지할 수 있는가?

### 모범 답안

보정 위험은

$$
\frac{99}{100}\frac{4}{99}+\frac{1}{100}
=
\frac{5}{100}
=0.05
$$

다. 따라서 목표 $\rho=0.05$의 경계에 있다.

그러나 이 식의 기본 보장은 calibration과 새 테스트 항목에 대해

$$
\mathbb E
\left[
\ell_{100}(\widehat\lambda)
\right]
\le0.05
$$

라는 기대손실 제어다. iid fresh-test 해석에서는 $\mathbb E_{\mathrm{cal}}[R(\widehat\lambda)]\le0.05$라고도 쓸 수 있지만, $95\%$ 확률로 $R(\widehat\lambda)\le0.05$라는 고확률 신뢰구간은 아니다.

change-point 경보가 났다면 과거 calibration과 미래 요청의 교환가능성이 의심되므로 자동 확대를 `HOLD`하고 새 regime의 성숙 라벨로 다시 검량해야 한다. 또한 EM soft count는 선택한 잠재모형이 맞다는 가정에 의존한다. 확정 결과에 대한 distribution-free CRC 보장을 자동으로 대신하지 못한다.

> **내일을 위한 연결:** 다음 단계에서는 판정자들이 같은 계열이라 조건부 독립이 깨질 때의 계층적 random-effect 모형과, 모수 불확실성을 posterior predictive check로 드러내는 방법을 배운 뒤, covariate shift 아래에서 conformal 보장을 어떻게 수정하는지 살펴본다.
