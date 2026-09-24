# [2026-09-25] 오늘 학습: 마팅게일·e-process·희귀사건 상한 & Anytime-valid Canary·지연 결과 관측

> **오늘의 핵심 문장:** “사고가 아직 없었다”와 “안전하다는 증거가 충분하다”는 다르다. 비음수 마팅게일과 Ville 부등식은 결과를 계속 확인하면서도 거짓 경보 확률을 통제하게 해 주지만, 그 보장은 성숙한 결과·올바른 사건 정의·미리 정한 검정 규칙 위에서만 성립한다.

지난 시간에는 제약 정책 최적화와 Canary 배포를 연결하고, 여러 시점의 Hoeffding 구간에 오류 예산을 나누는 **alpha-spending** 방식의 confidence sequence를 만들었다. 그 방법은 이해하기 쉽고 유효하지만, 매 시점마다 별도의 오류 예산을 떼어 두기 때문에 보수적일 수 있다.

오늘은 “귀무가설이 맞다면 평균적으로 불어나지 않는 증거 계좌” 하나를 만든다. 이 계좌가 **비음수 supermartingale**이고 시작값이 $1$이면, **Ville 부등식**은 어느 시점에 계좌가 $1/\alpha$를 넘을 확률도 최대 $\alpha$라고 보장한다. 이 원리에서 **e-value**, **e-process**, **anytime-valid 검정**이 나온다.

이어 희귀 안전 사건에서 매우 위험한 오해인 “위반 $0$건이므로 위험률도 $0$”을 정확한 상한으로 교정한다. 마지막으로 아직 확정되지 않은 결과를 정상으로 세는 오류를 막기 위해 **노출 시간**과 **증거 시간**을 분리하고, 통계적 경보를 실제 Canary 확대·보류·자동 rollback 상태 기계에 연결한다.

> **난이도 표지:** 오늘 내용은 고급 확장이지만 출발점은 동전 던지기와 조건부평균이다. 9월 17일의 KL 발산, 9월 21~22일의 신뢰구간·LCB/UCB, 9월 23일의 optional stopping을 다시 사용한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 매시간 같은 $95\%$ 검정을 보면 문제가 생길까?

고정 표본 검정은 보통 “표본 수 $n$을 먼저 정하고 마지막에 한 번 판단한다”는 계약을 전제로 한다. 그런데 실제 운영자는 새 모델의 dashboard를 매시간 본다. 우연히 좋은 순간에는 확대하고, 우연히 나쁜 순간에는 rollback한다.

서로 독립인 $5\%$ 거짓 경보 기회를 $K$번 반복한다고 단순화하면, 적어도 한 번 거짓 경보가 날 확률은

$$
1-(1-0.05)^K
$$

이다. $K=24$이면

$$
1-0.95^{24}\approx0.708
$$

이다. 실제 중간 검정들은 서로 독립이 아니므로 이 숫자를 그대로 적용할 수는 없지만, “매번 $5\%$니까 하루 종일 봐도 전체가 $5\%$”라는 생각이 틀렸다는 점은 분명하다.

지난 시간의 alpha-spending은 각 시점에 $\alpha_t$를 배정하고

$$
\sum_{t=1}^{\infty}\alpha_t\le\alpha
$$

가 되게 해 이 문제를 해결했다. 오늘은 시점마다 쿠폰을 나누는 대신, 귀무가설 아래에서 공정하거나 불리한 **하나의 누적 증거 계좌**를 추적한다.

### 1.2 왜 마팅게일은 “공정한 도박”에서 통계적 증거가 되었을까?

카지노에서 현재 가진 돈이 $M_{t-1}$일 때, 다음 판을 치른 뒤의 돈 $M_t$의 조건부 기댓값이 그대로라면

$$
\mathbb E[M_t\mid\mathcal F_{t-1}]
=M_{t-1}
$$

이다. 여기서 $\mathcal F_{t-1}$은 $t-1$번째 판까지 알고 있는 모든 정보다. 이런 과정을 **martingale(마팅게일)**이라고 한다.

귀무가설이 맞을 때 증거 계좌가 마팅게일 또는 평균적으로 줄어드는 supermartingale이라면, “계좌가 엄청 커졌다”는 사건은 귀무가설 아래 드물어야 한다. Ville 부등식은 이 직관을 “언제 멈추더라도” 쓸 수 있는 확률 보장으로 바꾼다.

### 1.3 왜 “사고 $0$건”은 안전률 $100\%$가 아닐까?

위험률이 $p=1\%$인 시스템을 $100$번 시험해도 사고가 한 번도 없을 확률은

$$
(1-0.01)^{100}
=0.99^{100}
\approx0.366
$$

이다. 실제 위험률이 $1\%$인데도 약 $36.6\%$의 실험에서는 “사고 $0$건”을 보게 된다.

즉 관측 횟수 $0$은 모수 $p=0$을 뜻하지 않는다. 희귀 사건에서는 점추정치보다 **상한**이 중요하다. “사고가 없었다”가 아니라 “현재 데이터와 오류 예산 아래에서 위험률의 상한이 허용 기준보다 충분히 작은가?”를 물어야 한다.

### 1.4 왜 아직 결과가 안 나온 요청을 정상으로 세면 안 될까?

개인정보 노출은 즉시 탐지될 수 있지만, 환불·분쟁·사람 검토·후속 신고는 며칠 뒤 확정될 수 있다. 오늘 400명이 새 에이전트를 사용했고 300명의 결과만 확정되었는데, 100명의 미확정 결과를 정상으로 넣으면 분모만 커져 위험률이 인위적으로 작아진다.

더 나쁜 경우도 있다. 사고 가능성이 큰 사례일수록 조사에 오래 걸린다면 “빨리 끝난 사례만” 보는 분석은 구조적으로 안전해 보인다. 이것이 **delayed outcome(지연 결과)**과 **informative censoring(결과와 관련된 검열)** 문제다.

따라서 wall-clock 시간과 통계적 정보량을 분리해야 한다.

- **노출 수:** 후보 정책을 실제로 경험한 단위 수
- **성숙 결과 수:** 미리 정한 관찰 창을 지나 라벨이 확정된 단위 수
- **pending 수:** 노출되었지만 아직 증거 계산에 넣을 수 없는 단위 수

### 1.5 역사적 흐름

1. **조건부 기댓값과 마팅게일:** 과거를 모두 안 상태에서 다음 변화의 평균이 $0$인 공정한 과정이 정식화되었다.
2. **Ville 부등식:** 비음수 마팅게일이 어느 순간 큰 임계값을 넘을 확률을 한 번에 제한했다.
3. **순차확률비:** 새 표본마다 두 가설의 가능도 비를 곱해 누적 증거를 만들었다.
4. **e-value와 e-process:** 기댓값이 $1$ 이하인 비음수 증거를 이용해 optional stopping에 안전한 검정을 구성했다.
5. **Confidence sequence:** 모든 시점에서 동시에 참값을 포함하는 구간열을 순차 검정의 역집합으로 만들었다.
6. **Anytime-valid Canary:** 계속 들어오는 운영 결과를 보면서도 사전 오류율을 지키는 배포 게이트로 확장했다.

핵심은 “언제 볼 것인가”를 통제하는 대신, **언제 보아도 유효한 증거의 형태**를 설계하는 것이다.

## 2. 친절한 용어 사전

| 기호·용어 | 초보자를 위한 뜻 | 오늘의 역할 |
| --- | --- | --- |
| $t$ | 관측 순서를 세는 번호 | 달력 시간이 아니라 증거가 들어온 순서로 사용할 수 있다. |
| $X_t\in\{0,1\}$ | $t$번째 단위의 사건 표시자 | $1$은 안전 위반, $0$은 정상으로 둔다. |
| $S_t=\sum_{i=1}^{t}X_i$ | 지금까지의 위반 건수 | 희귀 사건의 누적 횟수다. |
| $p$ | 한 단위가 위반을 일으킬 확률 | 알고 싶은 실제 위험률이다. |
| $p_0$ | 제품이 정한 허용 위험률 또는 검정 경계 | 안전/위험 귀무가설을 나누는 기준이다. |
| $q$ | 비교하고 싶은 대안 위험률 | 반드시 빨리 탐지할 위험 크기 또는 기대하는 안전 수준이다. |
| Bernoulli 분포 | 결과가 $1$ 또는 $0$ 두 가지뿐인 확률 모형 | “위반/정상” 한 건을 표현한다. $\operatorname{Bern}(p)$는 $1$이 나올 확률이 $p$라는 뜻이다. |
| 귀무가설 $H_0$ | 일단 유지하다가 충분한 반증이 생기면 버리는 기준 주장 | harm 검정에서는 “위험률이 허용치 이하”라는 주장이다. |
| 대립가설 $H_1$ | 귀무가설과 맞서며 데이터가 지지하기를 기대하는 주장 | harm 검정에서는 “위험률이 허용치를 넘는다”는 주장이다. |
| $\mathcal F_t$ | 시점 $t$까지 관측한 모든 정보의 집합 | 미래 정보를 쓰지 않았는지 표현한다. |
| filtration | $\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots$처럼 정보가 누적되는 구조 | 순차 검정의 시간축을 정의한다. |
| adapted | $M_t$를 계산할 때 $\mathcal F_t$ 밖의 미래 정보를 쓰지 않는 성질 | 실시간 증거량이 지켜야 할 기본 조건이다. |
| $\mathbb E[Z\mid\mathcal F_t]$ | 현재 정보까지 안 상태에서 본 $Z$의 평균 예측 | 마팅게일 조건을 표현한다. |
| 적분 가능 | $\mathbb E[\lvert M_t\rvert]<\infty$, 즉 절댓값의 평균이 유한한 성질 | 조건부 기댓값과 마팅게일을 잘 정의하기 위한 조건이다. |
| martingale | 다음 값의 조건부 기댓값이 현재 값과 같은 과정 | 귀무가설 아래 공정한 증거 계좌다. |
| supermartingale | 다음 값의 조건부 기댓값이 현재 값 이하인 과정 | 귀무가설 아래 평균적으로 불어나지 않는 증거 계좌다. |
| nonnegative | 값이 항상 $0$ 이상인 성질 | Ville 부등식에서 임계 초과 확률을 제한하게 해 준다. |
| stopping time $\tau$ | 현재까지의 정보만으로 멈출지 결정할 수 있는 시점 | “처음 임계값을 넘은 시간”이 대표적이다. |
| optional stopping | 데이터를 보며 중단 시점을 선택하는 것 | 고정 시점 검정은 깨질 수 있지만 e-process는 이를 견딘다. |
| Ville 부등식 | 비음수 supermartingale의 최댓값을 제한하는 부등식 | 전체 시간에 대한 거짓 경보율을 한 번에 통제한다. |
| e-value | 귀무가설 아래 기댓값이 최대 $1$인 비음수 통계량 | 큰 값일수록 귀무가설에 불리한 증거다. |
| e-process | 모든 허용 시점에서 e-value로 쓸 수 있는 누적 과정 | dashboard를 반복 확인하는 근거가 된다. |
| $1/\alpha$ | e-process의 증거 임계값 | $\alpha=0.05$이면 $20$이다. |
| likelihood | 특정 모수가 현재 데이터를 만들어 낼 가능도 | 두 위험률을 비교하는 증거의 재료다. |
| likelihood ratio | 두 가설의 가능도 비 | 매 관측의 증거 배수를 곱해 e-process를 만든다. |
| $\ell_t$ | $t$번째 결과가 증거 계좌에 곱하는 배수 | 정상/위반 결과마다 다른 값을 갖는다. |
| $M_t=\prod_{i=1}^{t}\ell_i$ | 누적 증거 계좌 | 임계값을 넘으면 귀무가설을 기각한다. |
| $\alpha$ | 허용할 제1종 오류 확률 | 참인 귀무가설을 잘못 기각할 장기 예산이다. |
| anytime-valid $p$-value | 어느 시점에 읽어도 유효한 $p$값 | running maximum e-value에서 만들 수 있다. |
| confidence sequence | 모든 시점에서 동시에 참 모수를 덮는 구간열 | 위험률의 time-uniform LCB/UCB를 제공한다. |
| LCB / UCB | lower/upper confidence bound, 즉 신뢰하한/신뢰상한 | 참 위험률이 내려갈 수 있는 아래끝과 올라갈 수 있는 위끝이다. 안전 확대에는 주로 UCB를 본다. |
| test inversion | 기각되지 않은 모수값을 모아 신뢰집합을 만드는 방법 | e-process에서 confidence sequence로 가는 다리다. |
| Beta 함수 $B(a,b)$ | $\int_0^1q^{a-1}(1-q)^{b-1}dq$ | Bernoulli 대안들을 섞은 mixture e-process의 닫힌 형태에 쓰인다. |
| Beta 모양 모수 $a,b>0$ | $q$에 줄 사전 가중치 곡선의 모양을 정하는 두 양수 | $a$가 크면 큰 $q$, $b$가 크면 작은 $q$에 상대적으로 더 무게를 둔다. |
| $\pi(dq)$ | 데이터를 보기 전에 고정한 $q$들의 확률분포, 즉 총합이 $1$인 가중 규칙 | 여러 대안의 증거를 유효하게 평균내는 방법을 지정한다. |
| mixture | 여러 대안 $q$의 증거를 사전 가중치로 평균내는 것 | 하나의 $q$를 잘못 고르는 민감도를 낮춘다. |
| $\mathbf 1\{A\}$ | 조건 $A$가 참이면 $1$, 거짓이면 $0$인 indicator(지시함수) | 사건 발생 여부나 집합 포함 여부를 수식 한 줄로 표시한다. |
| $\inf C$, $\sup C$ | 집합 $C$의 원소보다 크지 않은 수 중 가장 큰 값, 원소보다 작지 않은 수 중 가장 작은 값 | 신뢰집합의 아래끝과 위끝을 정의한다. 각각 infimum(하한), supremum(상한)이다. |
| zero-count | 관측된 위반이 $0$건인 상황 | 위험률이 $0$이라는 뜻은 아니다. |
| rule of three | $95\%$ zero-count 상한을 약 $3/n$으로 보는 근사 | 필요한 무사고 표본 규모를 빠르게 계산한다. |
| exposed | 후보 정책에 노출된 상태 | 아직 결과가 확정되었다는 뜻은 아니다. |
| matured | 사전 정의한 추적 창을 지나 결과가 확정된 상태 | 증거 과정에 넣을 수 있는 단위다. |
| pending | 노출됐지만 아직 결과가 확정되지 않은 상태 | 정상 $0$으로 세면 안 된다. |
| censoring | 관찰 종료 전에 결과를 모르는 상태 | 결과 의존적이면 편향을 만든다. |
| reveal | 미확정 결과의 라벨이 관측자에게 공개되는 사건 | 공개되는 순서가 결과와 관련되면 순차 검정이 편향될 수 있다. |
| exposure index $J_m$ | $m$번째로 공개된 결과가 원래 몇 번째 노출이었는지 가리키는 번호 | 달력 순서와 결과 공개 순서를 연결한다. |
| missing-at-random | 관측 여부가 이미 아는 정보로 설명되면, 그 정보를 조건으로 아직 모르는 결과와 무관하다는 가정 | 지연·누락 보정 모형이 요구하는 대표적 가정이다. |
| predictable censoring weight | 다음 결과를 보기 전에 과거 정보만으로 정한 관측확률 보정 가중치 | 미래 라벨을 엿보지 않고 검열을 보정하려는 장치다. |
| IPW | inverse probability weighting, 즉 관측될 확률의 역수로 관측값을 가중하는 방법 | 누락을 보정할 수 있지만 그 자체만으로 anytime-valid를 보장하지 않는다. |
| information time | 확정된 독립 분석 단위가 늘어나는 시간축 | e-process를 갱신할 올바른 인덱스다. |
| cluster | 같은 사용자·조직·세션처럼 서로 상관된 관측 묶음 | 요청 수와 독립 표본 수가 다를 수 있음을 알려 준다. |
| hard stop | 단 한 건만으로도 즉시 중단하는 규칙 | 치명적·비가역 사건에는 통계적 유의성을 기다리지 않는다. |
| latched rollback | 한 번 rollback 조건이 충족되면 자동으로 해제하지 않는 상태 | 나중의 정상 결과가 과거 경보를 지워 버리는 것을 막는다. |
| power | 실제로 위험할 때 경보를 낼 확률 | 오류율 통제와 별개로 표본 수·대안 $q$가 결정한다. |

오늘의 기본 Bernoulli harm 검정은 분석 단위별 사건이 명확히 정의되고, 각 시점의 조건부 사건확률이 안전 귀무가설의 경계를 만족한다고 가정한다. 고정 표본 zero-count 상한, 3.10절의 scale 검정, Beta-mixture confidence sequence의 간단한 형태는 후보 버전과 분석 집단이 고정되고 공통 위험률을 갖는 조건부 독립 Bernoulli 모형을 사용한다. 사용자별 반복 요청, 정책의 중간 변경, 탐지기 오분류, 결과 의존적 지연이 있으면 분석 단위와 모형을 다시 설계해야 한다.

## 3. 수학의 해부학 (증명과 원리)

### 3.1 먼저 “지금까지 아는 것”을 수식으로 고정하기

$\mathcal F_t$를 $t$번째 확정 결과까지의 정책 버전, 배정, 입력, 실행 로그, 사건 라벨을 모두 포함한 정보라고 하자.

$$
\mathcal F_0
\subseteq
\mathcal F_1
\subseteq
\cdots
\subseteq
\mathcal F_t.
$$

$X_t$는 $\mathcal F_t$를 보면 알 수 있지만 $\mathcal F_{t-1}$에서는 아직 모른다. $t$번째 사건의 조건부확률을

$$
\mu_t
=
\mathbb E[X_t\mid\mathcal F_{t-1}]
=
P(X_t=1\mid\mathcal F_{t-1})
$$

라고 하자.

이 표기는 매 시점의 위험률이 완전히 같아야 한다고 강제하지 않는다. 사용자 구성과 입력이 시간에 따라 달라도, 안전 귀무가설을

$$
H_0^{\mathrm{safe}}:
\mu_t\le p_0
\quad\text{for every }t
$$

처럼 “과거 정보를 고려한 다음 사건확률이 항상 한도 이하”라고 쓸 수 있다.

### 3.2 Martingale과 supermartingale

과정 $(M_t)_{t\ge0}$가 $\mathcal F_t$에 adapted이고 적분 가능할 때,

$$
\mathbb E[M_t\mid\mathcal F_{t-1}]
=M_{t-1}
$$

이면 martingale이다. 반면

$$
\mathbb E[M_t\mid\mathcal F_{t-1}]
\le M_{t-1}
$$

이면 supermartingale이다.

이름 때문에 “값이 계속 일정하거나 감소한다”고 오해하기 쉽지만, 실제 경로는 크게 오르내릴 수 있다. 제한되는 것은 **다음 값의 조건부평균**이다.

예를 들어 공정한 동전의 앞면을 $Y_t=1$, 뒷면을 $Y_t=-1$로 놓고

$$
R_t=\sum_{i=1}^{t}Y_i
$$

라고 하면 $R_t$는 위아래로 움직인다. 그러나

$$
\mathbb E[R_t\mid\mathcal F_{t-1}]
=R_{t-1}
+\mathbb E[Y_t\mid\mathcal F_{t-1}]
=R_{t-1}
$$

이므로 martingale이다.

오늘의 최대 부등식에는 $M_t\ge0$이 중요하다. 음수까지 허용되는 임의의 martingale에는 같은 형태의 임계 초과 해석을 바로 적용할 수 없다.

### 3.3 Stopping time은 “미래를 보지 않는 중단 규칙”이다

다음과 같은 최초 임계 초과 시점을 생각하자.

$$
\tau
=
\inf\left\{
t\ge0:
M_t\ge\frac1\alpha
\right\}.
$$

$\tau$가 stopping time이라는 뜻은, “$\tau\le t$인가?”를 판단할 때 $\mathcal F_t$만 보면 충분하다는 뜻이다. 다음 주의 데이터가 나쁜지 미리 보고 오늘 멈추는 규칙은 stopping time이 아니다.

실무의 다음 규칙은 stopping time이 될 수 있다.

- 증거량이 처음 $20$을 넘은 시점
- 확정 사건이 처음 $3$건 누적된 시점
- 비용 UCB가 한도를 처음 넘은 시점

반면 “나중에 전체 그래프를 보고 가장 유리한 시점을 선택”하는 것은 같은 사전 규칙이 아니다.

### 3.4 Ville 부등식 단계별 증명

$M_0=1$, $M_t\ge0$인 supermartingale을 생각하자. 유한한 $T$에 대해 $\tau\wedge T=\min(\tau,T)$는 bounded stopping time이다. supermartingale의 bounded optional stopping 성질로

$$
\mathbb E[M_{\tau\wedge T}]
\le
\mathbb E[M_0]
=1.
$$

한편 $\tau\le T$인 사건에서는 정의상

$$
M_{\tau\wedge T}=M_\tau\ge\frac1\alpha.
$$

$\tau>T$인 사건에서도 $M_T\ge0$이므로

$$
M_{\tau\wedge T}
\ge
\frac1\alpha
\mathbf 1\{\tau\le T\}.
$$

양변의 기댓값을 취하면

$$
1
\ge
\mathbb E[M_{\tau\wedge T}]
\ge
\frac1\alpha P(\tau\le T).
$$

따라서

$$
P(\tau\le T)\le\alpha.
$$

$T\to\infty$로 보내면 사건 $\{\tau\le T\}$가 $\{\tau<\infty\}$로 증가하므로 먼저

$$
P\left(
\exists t\ge0:
M_t\ge\frac1\alpha
\right)
\le\alpha
$$

를 얻는다. 임계값을 아래에서만 가까이 가고 유한 시점에는 닿지 않는 경로까지 포함한 표준 supremum 형태에는 한 단계가 더 필요하다. 임의의 $\varepsilon\in(0,1)$에 대해 더 낮은 임계값 $(1-\varepsilon)/\alpha$에 같은 논증을 적용하면

$$
P\left(
\sup_{t\ge0}M_t
\ge\frac1\alpha
\right)
\le
\frac{\alpha}{1-\varepsilon}.
$$

$\varepsilon\downarrow0$으로 보내면

$$
\boxed{
P\left(
\sup_{t\ge0}M_t
\ge\frac1\alpha
\right)
\le\alpha
}
$$

를 얻는다. 이것이 Ville 부등식이다.

중요한 차이는 “각 시점에서 $P(M_t\ge1/\alpha)\le\alpha$”만 말하는 것이 아니라, **전 시간 중 어느 한 번이라도** 넘을 확률을 $\alpha$로 제한한다는 점이다.

### 3.5 e-value, e-process, anytime-valid $p$-value

복합 귀무가설 $H_0$의 모든 분포 $P$ 아래에서 비음수 확률변수 $E$가

$$
\sup_{P\in H_0}
\mathbb E_P[E]
\le1
$$

을 만족하면 $E$를 e-value라고 한다. $E=20$이면 $\alpha=0.05$ 임계값 $1/\alpha=20$에 도달한 것이다.

e-value $20$은 다음을 뜻하지 않는다.

- 귀무가설의 사후확률이 $5\%$라는 뜻이 아니다.
- 대립가설이 귀무가설보다 정확히 $20$배 가능하다는 보편적 뜻도 아니다.

정확한 빈도주의 해석은 “귀무가설 아래에서 유효하게 설계된 비음수 증거가 $20$ 이상일 확률이 최대 $5\%$”라는 것이다.

$(E_t)$가 모든 허용 stopping time에서 e-value 성질을 유지하면 e-process라고 한다. 비음수 test supermartingale은 e-process를 만드는 대표적 방법이다.

running maximum

$$
E_t^*
=
\max_{0\le s\le t}E_s
$$

를 사용하면 anytime-valid $p$-value를

$$
p_t^{\mathrm{av}}
=
\min\left(
1,
\frac1{E_t^*}
\right)
$$

로 정의할 수 있다. 임계값을 한 번 넘은 사실이 나중에 증거량이 내려갔다고 사라지지 않도록 maximum을 쓰는 점이 핵심이다.

### 3.6 안전한 상태가 귀무가설일 때의 Bernoulli e-process

$X_t=1$을 안전 위반이라고 하자. 허용 기준 $p_0$보다 위험률이 크다는 사실을 빠르게 탐지하려면

$$
H_0^{\mathrm{safe}}:
\mu_t\le p_0
$$

를 귀무가설로 두고, 반드시 탐지하고 싶은 위험률 $q>p_0$를 대안으로 정한다.

$t$번째 증거 배수를

$$
\ell_t(q,p_0)
=
\left(\frac q{p_0}\right)^{X_t}
\left(\frac{1-q}{1-p_0}\right)^{1-X_t}
$$

로 놓고

$$
M_t
=
\prod_{i=1}^{t}\ell_i(q,p_0),
\qquad M_0=1
$$

로 정의한다.

$X_t=1$이면 $q/p_0>1$을 곱하므로 위험 증거가 커진다. $X_t=0$이면

$$
\frac{1-q}{1-p_0}<1
$$

을 곱하므로 안전 귀무가설에 불리한 증거가 줄어든다.

조건부 기댓값을 계산하면

$$
\begin{aligned}
\mathbb E[
\ell_t(q,p_0)
\mid\mathcal F_{t-1}]
&=
\mu_t\frac q{p_0}
+(1-\mu_t)\frac{1-q}{1-p_0}.
\end{aligned}
$$

오른쪽은 $\mu_t$에 대한 일차함수이고 그 기울기는

$$
\frac q{p_0}
-\frac{1-q}{1-p_0}
=
\frac{q-p_0}{p_0(1-p_0)}
>0
$$

이다. $\mu_t=p_0$일 때 값이 정확히 $1$이므로, $\mu_t\le p_0$에서는

$$
\mathbb E[
\ell_t(q,p_0)
\mid\mathcal F_{t-1}]
\le1.
$$

따라서

$$
\begin{aligned}
\mathbb E[M_t\mid\mathcal F_{t-1}]
&=
M_{t-1}
\mathbb E[\ell_t\mid\mathcal F_{t-1}]\\
&\le M_{t-1}.
\end{aligned}
$$

$M_t$는 $H_0^{\mathrm{safe}}$ 아래 비음수 supermartingale이다. 그러므로

$$
\sup_{P\in H_0^{\mathrm{safe}}}
\mathbb P_P\left(
\exists t:
M_t\ge\frac1\alpha
\right)
\le\alpha.
$$

### 3.7 왜 기대 로그 증거가 KL 발산이 될까?

실제 사건확률이 대안으로 정한 $q$와 정확히 같다고 하자. 한 관측의 기대 로그 증거는

$$
\begin{aligned}
\mathbb E_q[\log\ell_t]
&=
q\log\frac q{p_0}
+(1-q)\log\frac{1-q}{1-p_0}\\
&=
D_{\mathrm{KL}}
\left(
\operatorname{Bern}(q)
\Vert
\operatorname{Bern}(p_0)
\right)
\ge0.
\end{aligned}
$$

따라서 진짜 위험률이 $q$라면 로그 증거는 평균적으로 KL 발산만큼 증가한다. 9월 17일에 배운 KL이 여기서는 “두 위험률을 구분하는 관측당 평균 정보량”이 된다.

$q$가 $p_0$와 너무 가까우면 탐지하려는 변화가 작아 증거가 느리게 쌓인다. $q$가 실제 위험률과 너무 멀면 역시 효율이 나빠질 수 있다. 제1종 오류 보장은 Ville 부등식이 담당하지만, 얼마나 빨리 경보가 나는지는 $q$와 실제 분포가 결정한다.

### 3.8 대안 $q$를 모르면 mixture를 쓴다

데이터를 모두 본 뒤 가장 크게 나온 $M_t(q)$만 골라 보고하면 선택 편향이 생긴다. harm 검정에서는 데이터를 보기 전에 $q\in(p_0,1]$ 위의 고정된 확률분포 $\pi(dq)$, 즉 전체 가중치가 $1$인 규칙을 정하고

$$
M_t^{\mathrm{mix}}
=
\int_{(p_0,1]} M_t(q)\,\pi(dq)
$$

로 섞을 수 있다.

적분과 조건부 기댓값의 선형성 때문에 각 $M_t(q)$가 supermartingale이면 mixture도 supermartingale이다.

$$
\begin{aligned}
\mathbb E[
M_t^{\mathrm{mix}}
\mid\mathcal F_{t-1}]
&=
\int_{(p_0,1]}
\mathbb E[M_t(q)\mid\mathcal F_{t-1}]
\pi(dq)\\
&\le
\int_{(p_0,1]} M_{t-1}(q)\pi(dq)\\
&=
M_{t-1}^{\mathrm{mix}}.
\end{aligned}
$$

$q_t$를 시간에 따라 바꾸는 것도 가능하지만, harm 검정에서는 항상 $q_t\in(p_0,1]$여야 하며 $X_t$를 보기 전의 $\mathcal F_{t-1}$만으로 정한 **predictable choice**여야 한다. 현재 결과를 본 뒤 그 결과에 맞는 $q_t$를 선택하면 공정한 증거 계좌가 아니다.

### 3.9 Rollback 장난감 예제

허용 위험률을

$$
p_0=0.001
\quad(0.1\%)
$$

로 두고, 빠르게 탐지할 위험률을

$$
q=0.01
\quad(1\%)
$$

로 두자. $\alpha=0.05$이면 rollback 증거 임계값은 $20$이다.

정상 결과의 배수는

$$
\ell(0)
=
\frac{1-q}{1-p_0}
=
\frac{0.99}{0.999}
\approx0.990991
$$

이고, 위반 결과의 배수는

$$
\ell(1)
=
\frac q{p_0}
=10
$$

이다.

98건이 연속으로 정상이면

$$
M_{98}
=
\left(\frac{0.99}{0.999}\right)^{98}
\approx0.41194.
$$

그 뒤 위반 한 건이 나오면

$$
M_{99}
\approx0.41194\times10
=4.1194
$$

이고, 다음 확정 단위에서도 위반이 나오면

$$
M_{100}
\approx4.1194\times10
=41.194
>20.
$$

따라서 미리 정한 규칙대로 rollback한다. 이후 정상 결과가 많이 나와 $M_t$ 자체가 내려가더라도, running maximum은 이미 $20$을 넘었다. 자동 운영에서는 경보를 **latched** 상태로 유지하고 원인 분석·수정·새 실험 승인을 거쳐야 한다.

### 3.10 안전함을 보이려면 반대 방향의 검정이 필요하다

방금 $M_t<20$이라는 사실은 “위험하다는 충분한 증거가 아직 없다”는 뜻일 뿐 “안전하다는 충분한 증거가 있다”는 뜻이 아니다.

확대 근거를 만들 때는 해석을 더 엄격히 해야 한다. 후보 버전과 분석 집단을 고정하고, 성숙한 단위가

$$
P(X_t=1\mid\mathcal F_{t-1})=p
$$

인 공통 위험률 $p$의 조건부 독립 Bernoulli라고 정당화할 수 있다고 하자. 그러면

$$
H_0^{\mathrm{unsafe}}:
p\ge p_0
$$

를 귀무가설로 두고 더 안전한 대안 $q<p_0$를 비교하는 별도 과정이 필요하다. 같은 배수

$$
\ell_t(q,p_0)
=
\left(\frac q{p_0}\right)^{X_t}
\left(\frac{1-q}{1-p_0}\right)^{1-X_t}
$$

를 사용하면 $p\ge p_0$에서 한 단계의 기대 배수는 $1$ 이하이므로 unsafe 귀무가설 아래 supermartingale이 된다.

시간마다 다른 조건부위험률 $\mu_t$만 허용한 채 $H_0:\mu_t\ge p_0$ for every $t$를 기각하는 구성도 수학적으로 가능하다. 그러나 그 기각은 “모든 시점이 한도 이상”이라는 강한 명제가 거짓이라는 뜻일 뿐, 평균·현재·미래 위험률이 한도 아래라는 보장은 아니다. 이질성이나 drift가 크면 공통 $p$ 모형 대신 사전 정의한 시간평균, 층별 최악 상한, 또는 그 목표에 맞춘 별도 confidence sequence가 필요하다.

가장 단순한 zero-count 대안 $q=0$을 쓰면

$$
\ell_t
=
\begin{cases}
0, & X_t=1,\\[4pt]
\dfrac1{1-p_0}, & X_t=0.
\end{cases}
$$

위반이 한 번이라도 나오면 이 과정은 $0$이 되어 복구되지 않는다. 교육적으로는 명확하지만 실제 운영에는 데이터를 보기 전에 $\pi([0,p_0))=1$이 되도록 정한 여러 작은 $q$의 mixture가 더 유연하다.

$p_0=0.001$, $\alpha=0.05$에서 위반 없이 $n$건이 성숙했다면

$$
M_n
=
\left(\frac1{0.999}\right)^n.
$$

$M_n\ge20$이 되려면

$$
n
\ge
\frac{\log20}{-\log0.999}
\approx2994.23
$$

이므로 최소 $2{,}995$건의 완전한 무위반 결과가 필요하다. $300$건만으로는

$$
M_{300}
\approx1.350
<20
$$

이다.

### 3.11 Zero-count의 고정 표본 정확 상한과 rule of three

서로 독립이고 공통 위험률이 $p$인 Bernoulli 표본 $n$개에서 위반이 $0$건일 확률은

$$
P_p(S_n=0)
=(1-p)^n.
$$

단측 $1-\alpha$ 정확 상한 $U_n$은

$$
(1-U_n)^n=\alpha
$$

를 만족하도록 정한다. 풀면

$$
\boxed{
U_n
=
1-\alpha^{1/n}
}
$$

이다.

$n$이 크면

$$
\alpha^{1/n}
=
\exp\left(\frac{\log\alpha}{n}\right)
\approx
1+\frac{\log\alpha}{n}
$$

이므로

$$
U_n
\approx
\frac{-\log\alpha}{n}.
$$

$\alpha=0.05$에서는 $-\log0.05\approx2.996$이어서

$$
U_n\approx\frac3n.
$$

이것이 rule of three다.

예를 들어 성숙한 $300$건에서 위반이 $0$건이면

$$
U_{300}
=
1-0.05^{1/300}
\approx0.009936
=0.9936\%.
$$

점추정치는 $0\%$지만 $95\%$ 상한은 약 $0.994\%$다. 허용 기준이 $0.1\%$라면 거의 열 배 높으므로 확대 근거가 부족하다.

이 상한은 **고정된 $n$에서 한 번 계산하는 구간**이다. 매시간 다시 계산해 처음 기준을 통과한 시점에 멈추면 원래 coverage를 주장할 수 없다. 앞 절의 $q=0$ e-process는 같은 zero-count 아이디어를 anytime-valid 방식으로 재구성한 것이다.

### 3.12 Mixture e-process를 뒤집어 confidence sequence 만들기

이번에는 $X_i$가 공통 위험률 $p\in(0,1)$를 갖는 조건부 독립 Bernoulli라고 하자. 후보 모수 $p$를 귀무가설로 놓고, $a,b>0$인 대안 $q\sim\operatorname{Beta}(a,b)$를 섞는다. 이는 앞 절에서 여러 모수값을 함께 포함하는 **복합 귀무가설**을 한 방향으로 검정하려고 지지집합을 제한한 mixture와 달리, 하나의 모수값만 지정하는 **점귀무가설** $p$를 뒤집어 양쪽 confidence sequence를 만드는 전체 구간 mixture다.

고정 $q$ 대 $p$의 가능도비는

$$
L_t(q,p)
=
\left(\frac q p\right)^{S_t}
\left(\frac{1-q}{1-p}\right)^{t-S_t}.
$$

이를 Beta 밀도로 평균내면

$$
\begin{aligned}
M_t(p)
&=
\int_0^1
L_t(q,p)
\frac{
q^{a-1}(1-q)^{b-1}
}{
B(a,b)
}
dq\\
&=
\frac{
B(S_t+a,t-S_t+b)
}{
B(a,b)
p^{S_t}(1-p)^{t-S_t}
}.
\end{aligned}
$$

진짜 위험률이 후보 $p$와 같다면 각 $L_t(q,p)$가 martingale이므로 $M_t(p)$도 martingale이다. 경계에서는 연속극한으로

$$
M_t(0)
=
\begin{cases}
\dfrac{B(a,t+b)}{B(a,b)}, & S_t=0,\\[6pt]
+\infty, & S_t>0,
\end{cases}
$$

$$
M_t(1)
=
\begin{cases}
\dfrac{B(t+a,b)}{B(a,b)}, & S_t=t,\\[6pt]
+\infty, & S_t<t
\end{cases}
$$

로 정의한다. 경계에서는 귀무가설이 불가능하다고 보는 경로에 대안이 양의 확률을 줄 수 있어 이 극한 과정의 기댓값이 $1$보다 작을 수 있다. 그래도 비음수 supermartingale 성질이면 Ville 부등식과 coverage에는 충분하다.

시점 $t$에서 기각되지 않은 후보를 모아

$$
C_t
=
\left\{
p\in[0,1]:
M_t(p)<\frac1\alpha
\right\}
$$

라고 하자. Ville 부등식 때문에 진짜 $p$에 대해

$$
P_p
\left(
\forall t\ge0:
p\in C_t
\right)
\ge1-\alpha.
$$

즉 $(C_t)$는 $[0,1]$ 위의 confidence sequence다. $C_t$가 비어 있지 않을 때 $C_t$를 포함하는 가장 작은 구간의 끝점을

$$
L_t^{\mathrm{risk}}
=
\inf C_t,
\qquad
U_t^{\mathrm{risk}}
=
\sup C_t
$$

라고 하면, 확대에는 보통

$$
U_t^{\mathrm{risk}}\le p_0
$$

처럼 위험률 상한이 허용 기준 아래에 들어오는지를 사용한다.

### 3.13 지연 결과에서는 달력 시간과 증거 시간을 분리한다

달력 시점 $c$까지 후보 정책에 노출된 수를 $N_{\mathrm{exp}}(c)$, 미리 정한 추적 창을 지나 라벨이 확정된 수를 $N_{\mathrm{mat}}(c)$라고 하자.

$$
N_{\mathrm{pending}}(c)
=
N_{\mathrm{exp}}(c)
-N_{\mathrm{mat}}(c).
$$

e-process의 인덱스는 달력 시점 $c$가 아니라 성숙 순서

$$
m
=
N_{\mathrm{mat}}(c)
$$

로 두는 것이 자연스럽다. pending 단위에는 아직 $X_i$가 없으므로 증거 배수도 곱하지 않는다.

그러나 순서만 $m$으로 바꾼다고 유효성이 자동으로 생기지는 않는다. $J_m$을 $m$번째로 공개된 단위의 원래 exposure index라고 하자. $\mathcal G_{m-1}^{\mathrm{pre}}$는 그 라벨을 보기 직전까지의 전체 exposure 기록, pending 상태·나이, 이미 공개된 라벨 $X_{J_1},\ldots,X_{J_{m-1}}$과 각각의 reveal 시점, 누적 증거량 $M_{m-1}$, 현재 결과를 보기 전에 정한 예측 가능한 증거 배수 선택, reveal 과정, 그리고 공개될 단위 $J_m$의 신원을 모두 포함하는 정보다. 실제로 곱하는 배수는

$$
\ell_m
=
\ell(X_{J_m})
$$

이고, 필요한 조건은

$$
\mathbb E[
\ell(X_{J_m})
\mid
\mathcal G_{m-1}^{\mathrm{pre}}
]
\le1
$$

이다. reveal 순서가 아직 보지 않은 결과와 관련되면 이 조건이 깨질 수 있다. 고정된 추적 창 뒤 모든 단위를 같은 규칙으로 확정하는 cohort 설계가 가장 단순하다.

400건이 노출되었지만 300건만 성숙했고, 성숙 결과에서 위반이 $0$건이라고 하자. pending을 정상으로 잘못 세면

$$
1-0.05^{1/400}
\approx0.007461
=0.7461\%
$$

이라는 상한을 얻는다. 올바르게 성숙한 300건만 쓰면

$$
1-0.05^{1/300}
\approx0.009936
=0.9936\%
$$

이다. 확대 기준이 $0.8\%$라면 잘못된 계산은 통과하고 올바른 계산은 보류한다.

단순히 “결과가 도착하는 즉시 넣는다”는 것도 지연이 결과와 무관할 때만 안전하다. 위험한 사례가 조사에 더 오래 걸린다면 먼저 도착한 결과의 조건부위험률이 전체와 다르다. 실무에서는 다음 중 하나가 필요하다.

1. 모든 단위에 같은 사전 추적 창 $W$를 적용하고 $W$가 지난 cohort만 갱신한다.
2. $W$ 안의 미확정을 실패로 세려면 사건 자체를 처음부터 “위반 또는 $W$ 안에 미확정”으로 정의한다. 원래 사건 정의를 유지한다면 통계량에 임의 대입하지 말고 deterministic HOLD·rollback gate로 분리한다.
3. 관측확률 모형을 쓴다면 missing-at-random, predictable censoring weight, 양의 관측확률을 명시한다. 일반 IPW 추정치를 넣는 것만으로는 anytime-valid가 아니며, 보정된 증거 배수의 조건부 기댓값이 $1$ 이하임을 따로 증명해야 한다.
4. 결과 의존적 지연 자체를 공동 모형화하되, 그 모형이 틀릴 위험을 별도 감사한다.

### 3.14 수학적 보장의 경계

Ville 부등식은 강력하지만 다음 문제를 자동 해결하지 않는다.

- 사건 탐지기가 실제 위반을 놓치는 false negative
- 같은 사용자의 요청을 독립 표본처럼 세는 상관
- 실험 중 정책·프롬프트·권한을 바꾸고 같은 과정을 계속 쓰는 것
- dashboard를 보고 지표·세그먼트·귀무가설을 바꾸는 것
- 여러 안전 지표마다 $\alpha=0.05$를 별도로 써 전체 오류율을 키우는 것
- 치명적 사건이 한 건 발생했는데 “통계적으로 유의하지 않다”며 계속 노출하는 것

e-process는 **올바르게 정의된 관측 과정에서 반복 확인 문제**를 해결한다. 잘못된 라벨과 잘못된 운영 계약을 수학이 고쳐 주지는 않는다.

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Anytime-valid Canary의 전체 구조

AI 배포를 다음 상태 기계로 생각할 수 있다.

~~~text
OFFLINE EVAL
    |
    v
SHADOW  ---> telemetry invalid ------------------+
    |                                             |
    v                                             v
LIMITED CANARY ---> HOLD_PENDING ---> SCALE    ROLLBACK_LATCHED
    |                 ^              |             ^
    |                 |              v             |
    +-- insufficient mature evidence +-- harm -----+
    +-- critical event ----------------------------+
~~~

각 화살표에는 숫자로 된 사전 규칙이 있어야 한다. “팀이 보기 좋아서 확대”가 아니라 품질 LCB, 위험 UCB, e-process, 성숙 표본 수, pending 비율, 최소 관찰 기간이 상태 전이를 결정한다.

### 4.2 하나의 숫자로 확대와 rollback을 동시에 결정하지 않는다

희귀 사건에는 서로 다른 두 질문이 있다.

1. **Rollback 질문:** 현재 정책이 허용 위험률 이하라는 귀무가설을 기각할 만큼 위험 증거가 쌓였는가?
2. **Scale 질문:** 현재 정책이 허용 위험률 이상이라는 귀무가설을 기각할 만큼 안전 증거가 쌓였는가?

이를 각각

$$
M_t^{\mathrm{harm}}
\quad\text{and}\quad
E_t^{\mathrm{safe}}
$$

로 분리한다.

Rollback 조건의 예는

$$
\text{critical event}
\quad\text{or}\quad
M_t^{\mathrm{harm}}
\ge\frac1{\alpha_{\mathrm{harm}}}
\quad\text{or}\quad
\text{telemetry invalid}
$$

이다.

Scale 조건은 더 보수적으로

$$
\begin{aligned}
E_t^{\mathrm{safe}}
&\ge\frac1{\alpha_{\mathrm{safe}}},\\
L_t^{\mathrm{quality}}
&\ge\delta_{\min},\\
N_{\mathrm{mat}}
&\ge n_{\min},\\
\frac{N_{\mathrm{pending}}}{N_{\mathrm{exp}}}
&\le r_{\max}
\end{aligned}
$$

를 모두 요구할 수 있다.

$M_t^{\mathrm{harm}}$가 임계값 아래라는 사실만으로 scale해서는 안 된다. 이는 “화재경보가 안 울렸다”와 “건물이 안전검사를 통과했다”의 차이다.

여기서 $E_t^{\mathrm{safe}}$와 $U_t^{\mathrm{risk}}$의 scale 해석은 후보 정책·권한·traffic composition이 고정되고 사전 정의한 위험률 estimand에 맞는 모형이 유지될 때만 유효하다. 정책이나 노출 집단이 바뀌면 새 과정을 시작하거나, drift·층화를 명시적으로 허용하는 별도 time-uniform 구성을 사용한다.

### 4.3 사건 계약을 먼저 정의한다

$X_t$가 무엇인지 모호하면 수식도 모호하다. 예를 들어 도구 사용 LLM에서 다음을 분리할 수 있다.

| 사건 | 분석 단위 | 확정 시점 | 운영 반응 |
| --- | --- | --- | --- |
| 승인 없는 결제·삭제 | 사용자 작업 1건 | side effect audit 완료 시 | 한 건도 즉시 hard stop |
| 개인정보 외부 전송 | 조직 또는 세션 | DLP와 사람 검토 완료 시 | 심각도에 따라 hard stop 또는 e-process |
| 금지 도구 호출 시도 | agent trajectory | 구조화 로그 파싱 직후 | 권한 gateway에서 차단하고 비율 감시 |
| 유해 응답 | 독립 프롬프트 또는 사용자 | 사람/검증기 라벨 확정 시 | 위험률 UCB와 세그먼트 분석 |
| 장기 환불·분쟁 | 주문·작업 | 추적 창 $W$ 종료 시 | mature cohort만 통계 갱신 |
| VLM 좌표 파싱 실패 | 평가 예제 | raw output audit 후 | 인식 오류와 readout 오류를 분리 |

사건 정의에는 최소한 다음이 포함되어야 한다.

- 분자와 분모
- 중복 제거 규칙
- 사용자·세션·조직 중 분석 단위
- 라벨러·탐지기의 버전
- 확정에 필요한 추적 창
- 미확정·재시도·취소 처리
- 심각도와 hard-stop 여부

### 4.4 요청 수가 곧 표본 수는 아니다

한 사용자가 같은 에이전트에 $100$번 요청하면 그 $100$건은 환경·권한·행동 패턴을 공유한다. 독립 Bernoulli $100$개로 세면 불확실성을 과소평가할 수 있다.

가능한 대안은 다음과 같다.

- 사용자 단위로 “한 번이라도 위반”을 $X_i$로 정의한다.
- 조직·세션 단위 block을 하나의 분석 단위로 묶는다.
- 무작위 배정 단위와 분석 단위를 일치시킨다.
- 시간대·고객 등급·도구 권한별 층화 과정을 사전 정의한다.
- 의존성을 허용하는 martingale construction이나 cluster-robust 순차 추론을 사용한다.

단순 요청 단위 공식을 쓰고 싶다면 조건부확률 한도 $\mu_t\le p_0$가 실제로 타당한지 감사해야 한다.

### 4.5 지연 결과를 운영 상태로 드러내기

dashboard에는 성공률 하나가 아니라 최소한 다음을 함께 표시한다.

| 지표 | 의미 | 위험 신호 |
| --- | --- | --- |
| $N_{\mathrm{exp}}$ | 총 노출 단위 | 빠르게 증가하는데 다른 수치가 멈춤 |
| $N_{\mathrm{mat}}$ | 확정 결과 단위 | 증거 정보량 |
| $N_{\mathrm{pending}}$ | 미확정 단위 | backlog와 잠재 위험 |
| age distribution | pending이 얼마나 오래됐는지 | 긴 꼬리·조사 지연 |
| label completion rate | 추적 창 안에 확정된 비율 | 데이터 파이프라인 이상 |
| $M_t^{\mathrm{harm}}$ | 위험 증거량 | $1/\alpha_{\mathrm{harm}}$ 접근 |
| $U_t^{\mathrm{risk}}$ | 위험률의 time-uniform 상한 | 허용 기준 초과 |
| $L_t^{\mathrm{quality}}$ | 품질 개선의 time-uniform 하한 | 최소 개선폭 미달 |

미확정 결과가 많아지면 “아직 위험 증거가 없다”가 아니라 “관측 시스템이 결정을 내릴 준비가 안 됐다”로 해석한다.

### 4.6 자동 게이트 의사코드

~~~text
if critical_event_seen:
    state = ROLLBACK_LATCHED
elif telemetry_invalid:
    state = ROLLBACK_LATCHED
elif harm_e_process >= 1 / alpha_harm:
    state = ROLLBACK_LATCHED
elif matured_units < minimum_matured:
    state = HOLD_PENDING
elif pending_rate > maximum_pending_rate:
    state = HOLD_PENDING
elif safe_e_process >= 1 / alpha_safe
     and quality_lcb >= minimum_quality_gain
     and minimum_calendar_window_complete:
    state = SCALE_ONE_STEP
else:
    state = LIMITED_CANARY
~~~

실제 구현에서는 상태 변경이 원자적으로 기록되어야 하며, rollback 명령이 실패했을 때의 독립 kill switch가 필요하다. 통계 경보는 rollback 요청을 생성할 뿐, 네트워크·캐시·worker에 남은 후보 정책이 실제로 사라졌는지까지 보장하지 않는다.

### 4.7 기준군과 후보군의 차이를 보고 싶다면

오늘 유도한 과정은 후보의 절대 위험률 $p_C$를 한도 $p_0$와 비교한다. 기준군 대비 증가량

$$
\Delta
=
p_C-p_B
$$

를 검정하려면 무작위 배정과 두 표본 또는 paired sequential construction이 필요하다.

다음은 유효하지 않다.

- 후보 e-value에서 기준 e-value를 빼기
- 두 개의 $p$-value를 단순히 빼기
- 기준군의 고정 시점 상한과 후보군의 anytime-valid 상한을 직접 비교하기

사용자 단위로 후보/기준을 무작위 배정하고 같은 추적 창을 적용한 뒤, $\Delta\le\eta$ 같은 사전 가설에 맞는 순차 검정을 설계해야 한다. 절대 hard limit와 상대 증가 limit는 둘 다 통과하게 하는 편이 안전하다.

### 4.8 여러 안전 지표와 세그먼트의 오류 예산

권한 위반, 개인정보 노출, 유해 응답, 비용 폭주에 각각 $\alpha=0.05$를 쓰면 전체 시스템의 오경보 확률은 $5\%$가 아니다. 최소한

$$
\sum_{k=1}^{K}\alpha_k
\le
\alpha_{\mathrm{family}}
$$

처럼 지표별 예산을 사전 배분할 수 있다.

지역·기기·고객 등급별 dashboard를 모두 훑고 가장 나쁜 세그먼트만 보고하면 그것도 다중 탐색이다. 세그먼트는 사전 등록하거나 계층적 검정·e-value 결합 규칙을 사용한다. 심각도가 다른 사건에 같은 $\alpha$를 기계적으로 나누기보다, 피해 비용과 필요한 탐지 속도를 반영해 예산과 hard stop을 설계한다.

### 4.9 수학 부품과 AI 시스템의 1:1 연결

| 수학 부품 | AI 엔지니어링 부품 | 반드시 확인할 질문 |
| --- | --- | --- |
| $\mathcal F_t$ | 감사 가능한 실행·라벨 로그 | 미래 정보나 사후 수정이 섞이지 않았는가? |
| $X_t$ | 사전 정의된 안전 사건 | 탐지기 오분류와 중복을 어떻게 처리하는가? |
| $p_0$ | 제품·보안·법무의 위험 한도 | 평균 한도인지 hard ban인지 구분했는가? |
| $q$ | 빨리 탐지할 최소 위험 크기 | 운영상 의미 있는 effect size인가? |
| $M_t^{\mathrm{harm}}$ | rollback 증거 계좌 | 임계 초과 시 자동 동작이 실제로 실행되는가? |
| $E_t^{\mathrm{safe}}$ | scale-up을 위한 안전 증거 | rollback 비발생을 안전 증거로 오해하지 않는가? |
| Ville 부등식 | 반복 dashboard 확인의 오류율 통제 | 검정 규칙을 사후 변경하지 않았는가? |
| running maximum | latched alert | 이후 정상 결과가 경보를 지우지 않는가? |
| confidence sequence | 시간에 따라 갱신되는 위험 LCB/UCB | 고정 시점 구간을 반복 사용하지 않는가? |
| $N_{\mathrm{mat}}$ | 확정된 독립 분석 단위 | pending을 정상으로 세지 않는가? |
| censoring 가정 | 지연 라벨 파이프라인 | 위험한 사례일수록 늦어지는지 감사했는가? |
| mixture | 여러 의미 있는 대안 위험률 | 사후에 가장 유리한 $q$를 고르지 않는가? |
| $\alpha_k$ | 지표·세그먼트별 오류 예산 | 전체 family-wise 위험을 관리하는가? |
| hard stop | 비가역 피해 방지 규칙 | 통계적 유의성을 기다리면 안 되는 사건은 무엇인가? |

### 4.10 초보자가 흔히 하는 오해

| 오해 | 바로잡기 |
| --- | --- |
| “Martingale은 그래프가 평평하다.” | 실제 경로는 크게 변한다. 조건부 기댓값만 현재 값과 같다. |
| “e-value $20$이면 귀무가설 확률이 $5\%$다.” | 사후확률이 아니다. 귀무가설 아래 임계 초과의 장기 빈도 보장이다. |
| “e-process가 $20$ 아래면 안전하다.” | 위험 기각 증거가 부족할 뿐이다. 안전 확대에는 반대 방향 증거가 필요하다. |
| “사고가 $0$건이면 위험률도 $0$이다.” | zero-count 상한은 양수다. $95\%$ 근사는 약 $3/n$이다. |
| “Anytime-valid면 지표를 나중에 골라도 된다.” | stopping 자유를 주지만 사후 가설·세그먼트 선택을 공짜로 허용하지 않는다. |
| “불리한 결과 뒤 $M_t=1$로 재시작하면 된다.” | 반복 재시작은 오류 예산을 다시 쓰는 행위다. 새 실험과 새 $\alpha$ 계약이 필요하다. |
| “Pending은 아직 사고가 아니므로 정상이다.” | 라벨이 없을 뿐이다. $X=0$으로 넣으면 위험을 낮게 편향시킨다. |
| “완료된 사례만 보면 지연 문제는 사라진다.” | 완료 속도가 결과와 관련되면 완료 사례 자체가 편향된 표본이다. |
| “요청 백만 건이면 독립 표본 백만 개다.” | 사용자·세션·조직 내 상관 때문에 유효 정보량이 훨씬 작을 수 있다. |
| “통계적 rollback이면 치명적 사건도 몇 건 모아야 한다.” | 비가역·고위험 사건은 한 건 hard stop이 우선이다. |
| “모델 점수만 맞으면 사건 라벨도 맞다.” | parser·judge·좌표 규약이 능력과 다른 readout 오류를 만들 수 있다. |
| “Rollback은 이미 발생한 피해를 되돌린다.” | 향후 노출을 줄일 뿐이다. 권한 제한과 사전 방어가 먼저다. |

### 4.11 엔지니어의 최소 감사 체크리스트

1. 위험 사건의 분자·분모·분석 단위·심각도를 사전 등록한다.
2. hard-ban 사건과 확률 한도 사건을 분리한다.
3. 후보·기준 배정 단위와 분석 단위를 일치시킨다.
4. $p_0$, $q$, $\alpha$, 최소 표본, 최소 기간을 결과 보기 전에 고정한다.
5. rollback용 harm e-process와 scale용 safe evidence를 분리한다.
6. 여러 지표·후보·세그먼트의 전체 오류 예산을 관리한다.
7. raw output, parser, judge, 사람 라벨의 버전을 함께 저장한다.
8. exposed·matured·pending과 pending age를 별도 기록한다.
9. 지연이 결과와 관련되는지 정기적으로 감사한다.
10. 정책·프롬프트·권한·sampler가 바뀌면 같은 실험을 계속 써도 되는지 재검토한다.
11. 임계 초과 경보를 latched 상태로 기록하고 사람 승인 없이 해제하지 않는다.
12. rollback 명령의 실제 전파 시간과 실패 복구를 훈련한다.
13. 고위험 도구는 Canary 비율과 무관하게 permission gateway·rate limit·hardware stop을 유지한다.
14. “통과” 뒤에도 한 단계만 확대하고 새 부하·새 세그먼트에서 다시 관측한다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

### 5.1 FLUX 3 Action: 비디오 생성 backbone이 로봇 행동 정책으로 이동하다

Black Forest Labs는 2026년 9월 23일 [FLUX 3 Action 소개](https://huggingface.co/blog/black-forest-labs/flux-3-action)와 [공식 연구 보고서](https://bfl.ai/models/flux-3-action)를 공개했다. 이 모델은 **7B open-weight world action model(WAM)**이다. 카메라 프레임, 로봇 상태, 텍스트 명령을 입력받아 미래 영상 토큰과 행동 토큰을 하나의 diffusion sequence에서 함께 denoise한다.

[공식 DROID 모델 카드](https://huggingface.co/black-forest-labs/flux-3-action-droid)에 따르면 한 번에 $15\,\text{Hz}$의 행동 $32$개를 출력하므로 약

$$
\frac{32}{15}
\approx2.13\text{초}
$$

의 행동 chunk를 예측한다. 실제 제어에서는 전체 chunk를 맹목적으로 끝까지 실행하기보다 앞부분을 실행하고 새 관측으로 다시 계획한다.

공식 [RoboLab-120 leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)는 Isaac Sim의 탁상 과제 $120$개를 각 $10$회 실행한 $1{,}200$ rollout에서 FLUX 3 Action이

$$
\frac{515}{1200}
=42.92\%
$$

의 성공률로 현재 표의 1위를 기록했다고 보고한다. 그러나 난이도별로 보면 complex subset은

$$
\frac{48}{170}
=28.2\%
$$

다. 전체에서도 $57.08\%$, complex subset에서는 $71.8\%$가 실패했다는 뜻이다. 이는 공식 benchmark 조건에서 계산한 실패율이지 실제 로봇의 안전 사건률은 아니다.

공개 범위는 세 층으로 구분해야 한다.

- [공식 코드 저장소](https://github.com/black-forest-labs/flux-action)는 data preparation, distributed full fine-tuning, checkpoint/resume, export, BF16·FP8 inference를 포함하며 Apache-2.0이다.
- base·DROID·SO-101 체크포인트가 공개되며 LeRobot 통합도 제공된다. 이 중 DROID 저장소는 base·guidance-distilled·step-distilled 각각의 BF16·FP8r 변형을 제공한다.
- 가중치는 OSI 승인 오픈소스 라이선스가 아니라 [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/main/LICENSE.md)으로 공개된다. 일반적으로 비상업·비프로덕션 사용을 허용한다. 사용자와 계열사의 합산 연환산 총매출이 미화 $5{,}000{,}000$ 미만인 Qualifying User에게는 content filtering 또는 output review와 법이 요구하는 AI 표시 등의 조건 아래, 프로덕션에서 모델·파생물을 output 생성 목적으로 사용하는 제한적 상업 권리가 주어진다. 그 밖의 상업 사용에는 별도 라이선스가 필요하다.

따라서 “코드는 오픈소스이고 가중치는 공개되지만 제한적 라이선스가 적용된다”라고 구분하는 것이 정확하다.

하드웨어 장벽도 남아 있다. 모델 카드는 BF16에서 약 $32$GB GPU memory를 사용하고, FP8 양자화와 text encoder offload로 $24$GB 카드에 맞출 수 있다고 설명한다. 무엇보다 모델 카드에는 joint velocity, force, workspace를 모델 자체가 제한하지 않는다고 명시되어 있다. simulator 검증, 로봇의 저수준 limit, 사람 감독, 물리적 hardware stop이 필요하다.

**엔지니어 인사이트 (Impact):** 생성 모델의 세계 예측 표현을 행동 정책으로 재사용하고 distillation으로 제어 지연을 낮추는 흐름은 로봇과 향후 computer-use agent 모두에 중요하다. 그러나 leaderboard 1위와 안전한 물리 배포는 전혀 다른 주장이다. $1{,}200$개 rollout은 $120$개 task마다 $10$회씩 중첩된 구조이므로 모두를 독립·동일분포 표본으로 간주하기보다 task 내 의존 가능성을 점검해야 한다. 충돌·힘 초과·사람 접근 같은 희귀 안전 사건에는 별도 사건 계약도 필요하다. 오늘 배운 관점에서는 simulator에서 $M_t^{\mathrm{harm}}$와 위험 UCB를 먼저 만들고, 실제 장비에서는 작은 단계별 Canary, 힘·속도 hard limit, 즉시 latched stop을 함께 사용해야 한다. “최고 성공률”은 안전 e-process를 대체하지 않는다.

### 5.2 VLM Readout Limit: 같은 시각 능력도 답변 좌표계가 점수를 뒤집는다

[What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit](https://arxiv.org/abs/2609.27408)은 2026년 9월 23일 UTC에 제출된 사전논문이다. 연구는 같은 이미지·같은 객체·같은 $3\times3$ 위치 선택 문제에서 답 후보를 표현하는 규약만 바꾸었다.

COCO 사진 $200$장에서 Qwen3-VL-4B의 정확도는 다음과 같았다.

| 답 후보 표현 | 정확도 |
| --- | ---: |
| 영어 셀 이름 | $68.5\%$ |
| 정규화 좌표 $0$–$999$ | $59.5\%$ |
| 절대 픽셀 좌표 | $20.0\%$ |
| 무작위 기회 수준 | $11.1\%$ |

영어 이름과 픽셀 좌표의 차이는 $48.5\,\mathrm{pp}$이고, 논문이 보고한 exact McNemar 결과는 $p=1.2\times10^{-21}$이다. 반대로 좌표를 질문에 주고 “그 위치의 객체가 무엇인가?”라고 물었을 때의 픽셀 penalty는 $3.5\,\mathrm{pp}$로 유의하지 않았다. 즉 시각 위치를 전혀 모른다기보다 **답을 특정 좌표 문자열로 읽어 내는 인터페이스**가 병목일 수 있다.

모델 순위도 바뀌었다. Qwen2.5-VL-7B와 Qwen3-VL-4B는 영어 셀 이름에서 각각 $68.0\%$와 $68.5\%$로 거의 같지만, 절대 픽셀에서는 구형 모델이 $39\,\mathrm{pp}$ 앞서고 정규화 좌표에서는 신형 모델이 $54.5\,\mathrm{pp}$ 앞섰다. 하나의 좌표 규약만 쓴 leaderboard는 같은 시각 문제에서도 반대 순위를 만들 수 있다.

논문은 자기 평가 파이프라인에서도 scorer·target·parser·answer interface가 모델 출력과 어긋나 유능한 모델을 무능하다고 기록한 사례를 다섯 번 발견했다고 보고한다. 여기에는 token surface 불일치뿐 아니라 first-token readout 실패, target/key 오류, index-space 혼동, reasoning token이 answer budget을 소진한 경우가 포함된다. 그래서 raw output을 다시 파싱하고, trial–answer 연결을 섞었을 때 chance로 내려가는지 확인하며, 모델이 실제로 내지 않는 answer vocabulary는 scoring하지 않는 audit를 제안한다.

한계도 크다.

- 실제 이미지 결과는 COCO 한 데이터셋과 coarse grid selection 한 과제에 한정된다.
- free-form bounding-box regression의 결론으로 확장할 수 없다.
- 두 frontier model 중 하나에서만 효과가 확인됐다.
- 주 실험은 4-bit이고 8-bit 확인은 했지만 full precision은 실행하지 않았다.
- greedy decoding과 각 규약의 한 문구만 시험했다.
- 제안한 readership 지표는 사전 등록된 두 새 규약의 정확도 순서를 예측하는 데 실패했다.
- arXiv 논문 텍스트는 CC BY 4.0이지만, 코드·로그·프롬프트·분석 계획·audit script는 저자 요청으로 제공되며 추후 공개될 예정이라고 쓰여 있다. v1 시점에는 이 연구 artifact의 공개 저장소 URL이나 별도 라이선스가 없어, 논문의 공개 라이선스를 코드 라이선스로 해석하거나 즉시 완전 재현 가능한 release로 보아서는 안 된다.

**엔지니어 인사이트 (Impact):** 운영 지표의 $X_t$는 자연이 직접 주는 숫자가 아니라 parser·좌표계·judge·라벨러가 만든 관측이다. readout 규약이 바뀌면 “모델 사고”로 분류된 사건 중 일부가 측정 실패일 수 있고, 반대로 parser가 조용히 틀리면 실제 위험을 놓칠 수 있다. Canary에서는 정답률 하나 대신 인식 오류, parse failure, 실행 변환 오류, 실제 side effect를 분리하고 raw output 표본을 감사해야 한다. telemetry가 깨졌다면 e-process를 계속 업데이트하지 말고 배포를 HOLD 또는 rollback해야 한다.

두 소식은 같은 교훈을 준다. FLUX 3 Action은 모델이 시뮬레이션 점수에서 앞서도 실제 제어 안전장치가 필요함을 보여 주고, Readout Limit은 점수 자체가 출력 규약에 민감함을 보여 준다. 따라서 다음 세대 AI 엔지니어의 핵심 역량은 모델을 고르는 것뿐 아니라 **측정 계약, 시간에 따라 유효한 증거, 지연 라벨, 권한·물리 방어선**을 한 시스템으로 조립하는 것이다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

**질문:** 허용 안전 위반률이 $0.1\%$인 VLM 도구 에이전트 Canary가 있다. 현재 $350$개 사용자 단위가 노출되었고, 그중 $300$개는 추적 창을 지나 위반 $0$건으로 확정되었으며 $50$개는 사람 검토 중이다. 팀은 매시간 일반 $95\%$ 구간을 확인했고, “사고가 없고 새 좌표 출력 형식의 benchmark 점수도 높아졌다”며 전체 배포를 제안한다. 무엇이 잘못되었고 어떤 게이트를 써야 하는가?

**모범 답안:** 전체 배포하면 안 된다. 첫째, 확정된 $300$건만 사용한 고정 표본 zero-count $95\%$ 상한도

$$
1-0.05^{1/300}
\approx0.9936\%
$$

로 허용률 $0.1\%$의 거의 열 배다. 위반 점추정치가 $0\%$라는 사실은 안전 증거가 아니다. 둘째, 매시간 고정 시점 구간을 반복 확인했다면 명목 $95\%$ coverage를 그대로 주장할 수 없다. 사전에 정한 safe e-process 또는 time-uniform confidence sequence로 $U_t^{\mathrm{risk}}\le0.1\%$를 확인해야 한다.

셋째, 검토 중인 $50$건은 정상으로 셀 수 없다. 같은 추적 창을 지난 mature cohort만 증거에 넣고, pending 비율·나이·결과 의존적 지연을 감사해야 한다. 넷째, 좌표 형식이 달라져 benchmark 점수가 오른 것은 실제 시각 능력 향상과 readout 호환성 향상을 섞을 수 있다. 좌표 규약 swap, parse failure, raw output, 실제 tool side effect를 분리해 측정해야 한다.

운영 규칙은 critical event 또는 telemetry 이상 또는 $M_t^{\mathrm{harm}}\ge1/\alpha_{\mathrm{harm}}$에서 즉시 latched rollback하고, 반대로 safe evidence 임계값, 품질의 time-uniform LCB, 최소 mature 표본·기간, pending 한도를 모두 통과할 때만 한 단계 확대해야 한다. 위험 경보가 아직 없다는 사실과 안전 확대 근거가 있다는 사실은 서로 다른 검정이다.

> **다음 연결:** 다음에는 사건 탐지기의 민감도·특이도와 라벨 노이즈를 확률 모형에 포함하고, 여러 e-process·세그먼트를 동시에 운영할 때의 e-value 결합과 온라인 오류율 제어로 확장할 수 있다.
