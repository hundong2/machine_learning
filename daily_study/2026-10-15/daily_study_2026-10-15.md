# [2026-10-15] 오늘 학습: 교차분류·시간가변 MRP & Contextual Bandit 적응 감사

> **오늘의 핵심 문장:** 어제의 희소 집단을 부분 풀링으로 안정화하는 것만으로는 충분하지 않다. 언어·모달리티·기기처럼 서로 교차하는 집단과 시간에 따라 변하는 위험을 MRP로 복원하고, contextual bandit이 감사를 효율적으로 배분하더라도 모든 선택확률을 양수로 남겨 기록해야 측정과 발견을 동시에 지킬 수 있다.

지난 시간에는 모집단을 층으로 나누고, 층화 추정·Neyman 배분·부분 풀링으로 제한된 감사 예산을 효율적이고 공정하게 나누는 방법을 배웠다. 하지만 실제 AI 서비스의 집단은 한 줄짜리 목록이 아니다.

- 한국어이면서 이미지 입력이고 모바일인 요청처럼 특성이 **교차**한다.
- 오늘 안전했던 조합이 새 모델·새 프롬프트·새 사용자 유입 뒤에는 달라질 수 있다.
- 실패를 많이 찾으려는 적응형 정책은 위험해 보이는 곳만 골라 보므로, 관측 데이터 자체를 편향시킨다.
- 여러 집단과 여러 날짜를 계속 확인하면 우연히 나쁜 것처럼 보이는 경보가 늘어난다.

오늘은 이 네 문제를 하나의 흐름으로 묶는다.

- **수학:** 교차분류 계층모형, 로짓 링크, 시간가변 상태모형, poststratification, 중요도 가중을 유도한다.
- **AI 기초:** contextual bandit의 탐색–활용, LinUCB, propensity logging, 오프정책 평가를 감사 배분에 연결한다.
- **운영:** MRP의 모델 기반 추정과 무작위 감사의 설계 기반 추정을 분리해 서로 감시하게 한다.
- **트렌드:** 이미지별 시각 토큰 예산을 배분하는 mmVB와 추론 토큰을 줄이는 DoubtLess를 살펴본다.

> **난이도 표지:** 오늘은 “예측”, “선택”, “추론”을 구분해야 한다. MRP는 빈 칸의 위험을 **예측**한다. Bandit은 다음에 무엇을 감사할지 **선택**한다. IPS·DR·동시구간은 그렇게 모인 데이터로 어떤 결론을 낼 수 있는지 **추론**한다. 한 구성요소가 나머지 둘을 자동으로 해결하지 않는다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 한 가지 기준으로 만든 층만으로는 부족할까?

어제는 전체 트래픽을 서로 겹치지 않는 층으로 나눴다. 예를 들어 언어만 보면 한국어·영어·일본어라는 세 층이 생긴다. 그런데 이미지 입력의 위험이 언어마다 다르고, 모바일에서만 OCR이 무너진다면 언어만으로는 원인을 찾을 수 없다.

다음 세 축을 생각해 보자.

- 언어: 한국어, 영어, 일본어
- 모달리티: 텍스트, 이미지
- 기기: 모바일, 데스크톱

모든 조합을 만들면 $3\times 2\times 2=12$개 셀이 된다. 여기에 지역 10개, 모델 버전 4개, 날짜 30개를 더하면

$$
3\times 2\times 2\times 10\times 4\times 30=14{,}400
$$

개의 셀이 생긴다. 대부분의 셀에는 라벨이 매우 적거나 아예 없다. 이를 **희소 셀 문제**라고 한다.

셀을 합치면 중요한 상호작용을 숨기고, 모두 따로 추정하면 작은 셀의 비율이 $0\%$와 $100\%$ 사이에서 요동한다. 교차분류 다층모형은 이 두 극단 사이에서 정보를 빌리는 장치다.

### 1.2 왜 시간은 단순한 또 하나의 범주가 아닐까?

월요일과 화요일을 서로 아무 관계 없는 집단처럼 다루면 화요일의 데이터가 적을 때 월요일 정보를 전혀 쓰지 못한다. 반대로 한 달을 한꺼번에 합치면 어제 발생한 급격한 장애를 과거의 많은 정상 데이터가 덮어 버린다.

시간에는 순서가 있다.

- 오늘은 대체로 어제와 비슷하다.
- 배포·정책 변경·트래픽 이동이 있으면 갑자기 달라질 수 있다.
- 오래된 정보는 도움이 되지만 무한히 같은 무게로 믿어서는 안 된다.

그래서 시간가변 모형은 오늘의 상태가 어제 상태에서 조금 움직였다고 표현한다. 이는 “과거를 버리기”와 “과거에 묶이기” 사이의 수학적 절충이다.

### 1.3 왜 실패를 잘 찾는 정책이 위험률을 잘 재는 정책은 아닐까?

감사자가 위험 점수가 높은 요청만 보면 실제 위험 사례를 많이 발견한다. 운영팀 입장에서는 효율적이다. 그러나 그 감사 표본에서 단순 평균을 내면 전체 위험률을 크게 과대평가한다.

반대 문제도 있다. bandit이 지금까지 안전해 보였던 작은 집단을 거의 고르지 않으면, 그 집단에서 새로 시작된 실패를 영원히 배우지 못한다. 이를 **탐색의 소멸**이라고 생각할 수 있다.

따라서 두 질문은 분리해야 한다.

1. 같은 예산으로 위험 사례를 얼마나 많이 **발견**할 것인가?
2. 전체와 각 집단의 위험률을 얼마나 정직하게 **측정**할 것인가?

Contextual bandit은 첫 질문을 잘 풀 수 있다. 두 번째 질문까지 풀려면 선택확률, 즉 propensity를 선택 전에 계산하고 그대로 기록해야 한다.

### 1.4 왜 “그때의 확률”을 기록해야 할까?

적응형 정책은 매 시점 달라진다. 어제의 발견 결과가 오늘의 선택확률을 바꾸기 때문이다. 나중에 현재 모델만 다시 실행해서 과거 선택확률을 복원하려 해도 다음이 달라질 수 있다.

- 모델 파라미터와 feature pipeline
- 후보 집합과 quota
- 난수 시드와 동률 처리
- 이미 도착했던 지연 라벨
- 사람이 개입한 예외 규칙

선택확률은 영수증과 같다. 물건을 산 뒤 가격표가 바뀌어도 결제 당시 영수증이 있어야 비용을 증명할 수 있다. 감사 데이터도 선택 순간의 확률을 로그로 남겨야 역확률 가중이 가능하다.

### 1.5 왜 여러 집단을 동시에 보면 오류가 커질까?

각 집단마다 $95\%$ 구간을 따로 만들었다고 하자. 집단이 하나일 때 참값을 놓칠 확률은 $5\%$다. 하지만 100개 집단을 모두 훑으며 가장 나빠 보이는 하나를 고르면, 우연한 흔들림을 진짜 문제로 오해할 가능성이 훨씬 커진다.

개별 구간과 동시 구간은 질문이 다르다.

- 개별 구간: 미리 지정한 **이 한 집단**의 참값을 포함하는가?
- 동시 구간: 보고서에 나온 **모든 집단**의 참값을 한꺼번에 포함하는가?

운영 대시보드에서 “가장 나쁜 셀”을 고르는 순간 동시 추론이 필요하다. 시간을 반복해서 들여다보면 순차 추론까지 필요하다.

### 1.6 역사적 흐름

1. **회귀와 표본조사:** 관측된 표본의 결과를 특징과 연결하고, 실제 모집단 구성으로 다시 가중하는 poststratification이 발전했다.
2. **다층모형과 소지역 추정:** 표본이 작은 지역·집단은 관련 집단의 정보를 부분적으로 빌리도록 계층모형이 사용되었다.
3. **MRP:** Multilevel Regression and Poststratification은 다층 회귀로 셀별 값을 예측한 뒤 모집단 셀 수로 다시 합친다.
4. **동적 상태모형:** 경제·신호처리·제어 분야에서 시간에 따라 변하는 숨은 상태를 이전 상태와 새 관측으로 갱신해 왔다.
5. **Multi-Armed Bandit:** 임상시험·광고·추천에서 탐색과 활용을 조절하며 순차적으로 행동을 선택하는 방법이 발전했다.
6. **오프정책 평가:** 이미 다른 정책이 만든 로그로 새 정책의 가치를 추정하기 위해 IPS와 DR이 사용되었다.
7. **AI 감사:** 희소 교차집단, 모델 drift, 비싼 사람 라벨, 적응 선택, 동시 안전 게이트가 한 시스템 안에서 만나게 되었다.

### 1.7 오늘의 전체 지도

| 단계 | 묻는 질문 | 핵심 도구 | 실패하면 생기는 일 |
| --- | --- | --- | --- |
| 셀 정의 | 누구의 위험을 구분할까? | 교차분류 atomic cell | 취약한 조합이 평균에 숨는다. |
| 셀 위험 예측 | 희소·빈 셀을 어떻게 안정화할까? | 다층 로지스틱 회귀 | 작은 셀 추정치가 폭주한다. |
| 시간 갱신 | 최근 변화에 어떻게 반응할까? | random walk·AR(1) | 오래된 평균이 장애를 덮는다. |
| 모집단 복원 | 실제 트래픽 위험은 얼마인가? | poststratification | 감사 표본 구성이 전체로 오인된다. |
| 감사 선택 | 다음 라벨을 어디서 살까? | contextual bandit | 예산을 낮은 가치에 쓴다. |
| 식별 보존 | 선택 편향을 어떻게 되돌릴까? | 탐색 하한·propensity | 일부 집단은 평가 불가능해진다. |
| 정책 평가 | 새 배분 정책이 나은가? | IPS·DR·ESS | 발견률 향상을 과장한다. |
| 배포 판정 | 여러 셀을 안전하게 볼까? | 동시구간·순차 오류 제어 | 우연한 경보를 선택한다. |

## 2. 친절한 용어 사전

| 기호·용어 | 초보자를 위한 뜻 | 오늘의 역할 |
| --- | --- | --- |
| 관측 단위 | 한 번의 사용자 요청 또는 한 감사 후보 | 가장 작은 데이터 한 건이다. |
| $Y_i\in\{0,1\}$ | 요청 $i$의 확정 위험 라벨 | 위험이면 $1$, 아니면 $0$이다. |
| factor | 언어·모달리티처럼 셀을 만드는 범주형 축 | 집단 차이의 원인을 나눈다. |
| level | factor 안의 구체적 값 | 한국어는 언어 factor의 한 level이다. |
| crossed factors | 한 축이 다른 축 안에 포함되지 않고 조합되는 관계 | 한국어와 이미지는 서로 교차한다. |
| nested factors | 한 집단이 다른 집단 안에만 속하는 관계 | 학교 안의 학급처럼 중첩된다. |
| atomic cell $c$ | 여러 factor의 값을 모두 지정한 최소 조합 | 한국어×이미지×모바일 같은 셀이다. |
| structural zero | 현실적으로 존재할 수 없는 셀 | 표본이 우연히 0건인 셀과 다르다. |
| sparse cell | 존재하지만 라벨이 거의 없는 셀 | 직접 비율이 불안정하다. |
| $N_{c,t}$ | 시점 $t$의 셀 $c$ 전체 요청 수 | poststratification 가중치를 만든다. |
| $W_{c,t}$ | 시점 $t$의 셀 $c$ 트래픽 비중 | $N_{c,t}/\sum_jN_{j,t}$다. |
| $p_{c,t}$ | 시점 $t$, 셀 $c$의 참 위험확률 | 우리가 알고 싶은 숨은 값이다. |
| Bernoulli 분포 | 결과가 $0$ 또는 $1$인 한 번의 시행 분포 | 위험 라벨의 기본 모형이다. |
| likelihood | 모형이 관측 데이터를 얼마나 그럴듯하게 만드는지 나타내는 식 | 회귀계수를 데이터로 학습한다. |
| logit | 확률 $p$를 실수로 바꾸는 $\log\{p/(1-p)\}$ | 선형 예측자와 확률을 연결한다. |
| sigmoid $\sigma(z)$ | 실수 $z$를 $0$과 $1$ 사이로 바꾸는 함수 | $\sigma(z)=1/(1+e^{-z})$다. |
| fixed effect | 모든 level의 차이를 별도 상수로 직접 추정하는 효과 | 데이터가 충분할 때 해석이 쉽다. |
| random effect | level 효과가 공통 분포에서 왔다고 보는 효과 | 희소 level끼리 정보를 빌린다. |
| hyperparameter | random effect 분포의 크기 등을 정하는 상위 매개변수 | 수축 강도를 데이터와 연결한다. |
| partial pooling | 각 셀 자료와 공통 경향을 함께 쓰는 것 | 작은 셀을 안정화한다. |
| interaction | 두 factor의 결합이 각 효과의 단순 합과 다른 부분 | 이미지 위험이 특정 언어에서만 큰 현상을 담는다. |
| identifiability | 서로 다른 매개변수가 같은 예측을 만드는 혼동을 제거하는 성질 | 절편과 집단효과를 구분한다. |
| sum-to-zero | 한 factor 효과들의 합을 $0$으로 두는 제약 | 절편의 의미를 고정한다. |
| MRP | Multilevel Regression and Poststratification | 셀 위험을 예측하고 실제 셀 크기로 합친다. |
| poststratification | 예측된 셀 값을 실제 모집단 구성으로 다시 합치는 절차 | 표본과 트래픽 구성 차이를 보정한다. |
| estimand | 정확히 무엇을 추정하려는지 적은 대상 | 오늘 위험률인지 7일 평균인지 구분한다. |
| latent state | 직접 보이지 않지만 관측을 만드는 숨은 상태 | 날짜별 공통 위험 수준이다. |
| random walk | 오늘 상태가 어제 상태에 작은 충격을 더해 만들어지는 모형 | 변화는 허용하되 인접 날짜를 연결한다. |
| AR(1) | 오늘 상태가 어제 상태의 $\rho$배와 새 충격의 합인 모형 | $\lvert\rho\rvert<1$이면 장기 평균으로 되돌아간다. |
| process noise | 실제 상태가 움직이는 정도 | drift에 얼마나 빨리 반응할지 정한다. |
| observation noise | 라벨 표본 때문에 관측치가 흔들리는 정도 | 작은 표본을 덜 믿게 한다. |
| filtering | 과거와 현재 관측만으로 현재 상태를 갱신하는 것 | 온라인 대시보드에 적합하다. |
| smoothing | 미래 관측까지 사용해 과거 상태를 다시 추정하는 것 | 사후 분석에는 좋지만 당시 의사결정 재현과 다르다. |
| context $X_t$ | 행동을 고르기 전에 보이는 정보 | 언어·모달리티·위험 점수·시간대 등이다. |
| arm·action $A_t$ | bandit이 고르는 선택지 | 다음에 감사할 queue 또는 정책이다. |
| reward $U_t$ | 선택 뒤 관측하는 효용 | 발견한 위험의 심각도·정보가치 등을 담는다. |
| potential reward $U_t(a)$ | 행동 $a$를 골랐다면 얻었을 가상 효용 | 실제로는 고른 행동 하나만 관측한다. |
| contextual bandit | context를 보고 행동 하나를 골라 즉시 reward를 배우는 문제 | 감사 예산을 순차 배분한다. |
| exploration | 아직 잘 모르는 선택지도 시험하는 것 | 새 실패와 작은 집단을 보이게 한다. |
| exploitation | 현재 가장 좋아 보이는 선택을 하는 것 | 발견 효율을 높인다. |
| regret | 매번 최선 행동을 알았을 때와 실제 reward의 누적 차이 | bandit 학습 효율의 대표 척도다. |
| feature $\phi(X_t,a)$ | context와 행동을 숫자 벡터로 바꾼 것 | 선형 reward 모형의 입력이다. |
| ridge regression | 제곱오차에 계수 크기 벌점을 더한 회귀 | 희소·상관 feature를 안정화한다. |
| UCB | 평균 예측에 불확실성 보너스를 더한 상한 | 낙관적으로 탐색한다. |
| policy $b_t(a\mid X_t,\mathcal F_{t-1})$ | 시점 $t$에 과거 이력과 현재 context를 보고 행동 $a$를 고를 확률 규칙 | 실제 로그를 만든 behavior policy다. |
| target policy $\pi(a\mid x)$ | 평가하고 싶은 새 정책 | 오프정책 평가의 대상이다. |
| propensity | 실제 선택 행동에 부여됐던 확률 | 중요도 가중의 분모다. |
| positivity·overlap | target이 고를 행동을 behavior도 양의 확률로 고르는 조건 | 확률 $0$인 행동은 로그로 평가할 수 없다. |
| exploration floor $\varepsilon$ | 어떤 행동도 완전히 사라지지 않게 하는 최소 무작위화 | overlap과 새 실패 탐지를 지킨다. |
| predictable | 선택확률이 과거와 선택 전 정보만으로 정해지는 성질 | 미래 결과를 미리 사용한 편향을 막는다. |
| IPS | Inverse Propensity Scoring | $\pi/b$로 behavior 로그를 target 분포로 바꾼다. |
| DR | Doubly Robust estimation | outcome 모형과 IPS 보정을 함께 쓴다. |
| outcome model $\widehat q(x,a)$ | context와 행동에서 기대 reward를 예측하는 모형 | DR의 낮은 분산 부분이다. |
| importance weight $w_t$ | $\pi(A_t\mid X_t)/b_t(A_t\mid X_t,\mathcal F_{t-1})$ | 정책 분포 차이를 보정한다. |
| weight clipping | 너무 큰 가중치를 상한에서 자르는 것 | 분산은 줄지만 편향이 생긴다. |
| ESS | Effective Sample Size | 불균등 가중 데이터가 균등 표본 몇 건과 비슷한지 나타낸다. |
| simultaneous inference | 여러 셀의 오류확률을 가족 전체에서 제어하는 추론 | 최악 셀 선택의 우연성을 다룬다. |
| Bonferroni | 전체 $\alpha$를 여러 비교에 나누는 보수적 방법 | 의존 구조를 몰라도 동시 오류를 제어한다. |
| credible interval | 주어진 Bayesian 모형 아래 사후확률을 해석하는 구간 | MRP의 모델 의존 불확실성이다. |
| confidence interval | 반복 표본설계에서 장기 포함률을 해석하는 구간 | 설계 기반 품질 게이트에 쓴다. |
| policy version | 선택 로직의 불변 식별자 | 과거 propensity를 재현하게 한다. |
| label delay | 선택 뒤 확정 라벨이 늦게 도착하는 현상 | pending을 음성 라벨로 오인하면 안 된다. |
| drift | 데이터·위험·정책 관계가 시간에 따라 변하는 현상 | 시간가변 MRP와 할인 bandit이 다룬다. |

## 3. 수학의 해부학 (증명과 원리)

### 3.1 먼저 추정 대상을 고정한다

시점 $t$에 존재하는 atomic cell 집합을 $\mathcal C_t$라 하자. 셀 $c$의 전체 요청 수는 $N_{c,t}$이고 총 요청 수는

$$
N_t=\sum_{c\in\mathcal C_t}N_{c,t}
$$

이다. 셀 비중은

$$
W_{c,t}=\frac{N_{c,t}}{N_t},
\qquad
\sum_{c\in\mathcal C_t}W_{c,t}=1
$$

이다. 셀 위험확률을 $p_{c,t}$라 하면 **그날 전체 요청 하나의 위험확률**은

$$
P_t=\sum_{c\in\mathcal C_t}W_{c,t}p_{c,t}
$$

다.

7일 전체 요청의 위험률은 날짜별 위험률을 단순 평균한 값이 아니다. 날짜마다 트래픽이 다르면

$$
P_{1:7}
=
\frac{\sum_{t=1}^{7}\sum_cN_{c,t}p_{c,t}}
{\sum_{t=1}^{7}\sum_cN_{c,t}}
$$

로 계산해야 한다. **사람을 똑같이 셀지, 날짜를 똑같이 셀지**가 estimand를 바꾼다.

### 3.2 직접 셀 평균은 왜 흔들릴까?

감사된 셀 $c,t$의 라벨 수가 $n_{c,t}$이고 위험 건수가 $y_{c,t}$라면 직접 비율은

$$
\widehat p_{c,t}^{\mathrm{direct}}
=
\frac{y_{c,t}}{n_{c,t}}
$$

이다. 독립 Bernoulli 근사 아래 분산은

$$
\operatorname{Var}\left(
\widehat p_{c,t}^{\mathrm{direct}}
\right)
\approx
\frac{p_{c,t}(1-p_{c,t})}{n_{c,t}}
$$

이다. 분모 $n_{c,t}$가 작으면 크게 흔들린다. $n_{c,t}=0$이면 아예 계산할 수 없다.

특히 $y_{c,t}=0$이라고 해서 $p_{c,t}=0$은 아니다. $n$건에서 위험을 하나도 못 볼 확률은

$$
\Pr(Y=0\mid p)=(1-p)^n
$$

이다. 예를 들어 $n=5$, $p=0.2$이면

$$
(1-0.2)^5=0.8^5=0.32768
$$

이다. 실제 위험률이 $20\%$여도 약 $32.8\%$ 확률로 위험 $0$건을 볼 수 있다.

### 3.3 로짓 링크는 왜 필요한가?

확률을 선형식으로 직접 예측해

$$
p=\beta_0+\beta_1x
$$

라고 쓰면 어떤 $x$에서는 $p<0$ 또는 $p>1$이 될 수 있다. 로짓은 확률을 실수 전체로 옮긴다.

$$
\operatorname{logit}(p)
=
\log\frac{p}{1-p}
$$

역함수는 sigmoid다.

$$
p
=
\sigma(\eta)
=
\frac{1}{1+e^{-\eta}}
$$

$\eta=0$이면 $p=0.5$이고, $\eta$가 커질수록 $p$는 $1$에 가까워지며 작아질수록 $0$에 가까워진다.

두 집단의 선형 예측자가 $\Delta$만큼 다르면 오즈 비는

$$
\frac{p_1/(1-p_1)}{p_0/(1-p_0)}
=
e^\Delta
$$

다. 예를 들어 $\Delta=\log 2$이면 오즈가 두 배다. **오즈가 두 배**는 확률이 항상 두 배라는 뜻이 아니다.

### 3.4 Bernoulli likelihood에서 로지스틱 손실이 나온다

요청 $i$의 위험확률을 $p_i$라 하자.

$$
\Pr(Y_i=y_i)
=
p_i^{y_i}(1-p_i)^{1-y_i}
$$

독립 관측의 likelihood는 곱이다.

$$
L
=
\prod_{i=1}^{n}
p_i^{y_i}(1-p_i)^{1-y_i}
$$

로그를 취하면 곱이 합이 된다.

$$
\log L
=
\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right]
$$

따라서 음의 로그가능도를 최소화하는 것은 binary cross-entropy를 최소화하는 것과 같다.

$$
\mathcal L_{\mathrm{BCE}}
=
-\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right]
$$

이전의 손실 함수 학습과 오늘의 위험률 모형이 같은 수학 부품을 쓴다.

### 3.5 교차분류 다층 로지스틱 모형

셀 $c$가 언어 $g(c)$, 모달리티 $m(c)$, 기기 $d(c)$의 조합이라고 하자. 날짜 $t$의 선형 예측자를

$$
\eta_{c,t}
=
\beta_0
+
a_{g(c)}
+
b_{m(c)}
+
d_{d(c)}
+
h_{g(c),m(c)}
+
u_t
+
v_{g(c),t}
$$

로 둔다. 각 항은 다음 뜻이다.

- $\beta_0$: 전체 기준 위험
- $a_g$: 언어별 차이
- $b_m$: 모달리티별 차이
- $d_d$: 기기별 차이
- $h_{g,m}$: 언어×모달리티 상호작용
- $u_t$: 그날 모든 셀에 공통인 시간 효과
- $v_{g,t}$: 특정 언어에서만 변하는 시간 효과

셀 위험확률은

$$
p_{c,t}=\sigma(\eta_{c,t})
$$

다.

여기서 언어와 모달리티는 서로 중첩되지 않는다. 각 언어에 텍스트와 이미지가 모두 나타날 수 있으므로 **crossed**다. 상호작용 $h_{g,m}$이 없다면 한국어 이미지의 효과는 한국어 효과와 이미지 효과의 합으로만 제한된다. 특정 조합에서만 생기는 실패를 표현하려면 상호작용이 필요하다.

### 3.6 부분 풀링은 수학적으로 어디서 오는가?

직관을 위해 로짓이 아니라 정규근사로 한 집단 효과를 보자. 직접 추정치가

$$
\widehat\theta_g\mid\theta_g
\sim
\mathcal N(\theta_g,s_g^2)
$$

이고 집단 효과의 사전분포가

$$
\theta_g
\sim
\mathcal N(\mu,\tau^2)
$$

라고 하자. 정규–정규 결합의 사후평균은

$$
\mathbb E[\theta_g\mid\widehat\theta_g]
=
\frac{\tau^2}{\tau^2+s_g^2}\widehat\theta_g
+
\frac{s_g^2}{\tau^2+s_g^2}\mu
$$

다. 이를

$$
B_g=\frac{\tau^2}{\tau^2+s_g^2}
$$

라 쓰면

$$
\mathbb E[\theta_g\mid\widehat\theta_g]
=
B_g\widehat\theta_g+(1-B_g)\mu
$$

다.

- 표본이 많아 $s_g^2$가 작으면 $B_g\approx 1$이므로 자기 데이터를 주로 믿는다.
- 표본이 적어 $s_g^2$가 크면 $B_g\approx 0$이므로 공통 평균 $\mu$ 쪽으로 수축한다.
- 집단 간 실제 차이 $\tau^2$가 크면 수축이 약해진다.

로지스틱 다층모형에서는 정확히 같은 닫힌식이 나오지 않지만, 부분 풀링의 방향은 같다.

### 3.7 random effect와 identifiability

집단효과에 다음과 같은 분포를 둔다.

$$
a_g\sim\mathcal N(0,\tau_{\mathrm{lang}}^2),
\qquad
b_m\sim\mathcal N(0,\tau_{\mathrm{mod}}^2),
\qquad
h_{g,m}\sim\mathcal N(0,\tau_{\mathrm{int}}^2)
$$

평균이 $0$인 proper prior는 Bayesian 모형을 규제한다. fixed-effect 방식이라면

$$
\sum_ga_g=0,
\qquad
\sum_mb_m=0
$$

같은 제약 또는 기준 level 하나를 $0$으로 고정하는 규칙이 필요하다. 그렇지 않으면 $\beta_0$에 상수를 더하고 모든 $a_g$에서 같은 상수를 빼도 예측이 같아 매개변수가 고유하지 않다.

상호작용에도 행·열 합 제약을 둘 수 있다.

$$
\sum_gh_{g,m}=0,
\qquad
\sum_mh_{g,m}=0
$$

이는 “상호작용”이 주효과로 설명되지 않는 나머지라는 뜻을 분명하게 만든다.

### 3.8 상호작용을 무한히 넣으면 왜 안 될까?

교차축이 많을수록 가능한 상호작용 수가 폭발한다. 언어 20개와 모달리티 5개의 2차 상호작용만 해도 $100$개다. 기기 6개까지 3차 상호작용을 만들면 $600$개다.

모든 상호작용을 무규제로 넣으면 희소 셀의 잡음을 외운다. 실무에서는 다음 순서를 쓴다.

1. 도메인상 중요한 저차 상호작용을 먼저 정한다.
2. random effect로 강하게 수축한다.
3. held-out 기간과 posterior predictive check로 확인한다.
4. 빈 셀 예측은 “데이터가 없는 직접 관측”이 아니라 “모형 기반 외삽”임을 표시한다.

**구조적으로 존재하지 않는 셀**은 poststratification 표에서 $N_{c,t}=0$으로 제외한다. 존재하지만 감사 표본만 없는 셀은 $N_{c,t}>0$, $n_{c,t}=0$이며 모형 예측이 필요하다. 두 종류의 0을 섞으면 안 된다.

### 3.9 시간가변 상태: random walk와 AR(1)

가장 단순한 공통 시간 효과는 random walk다.

$$
u_t=u_{t-1}+\xi_t,
\qquad
\xi_t\sim\mathcal N(0,q)
$$

$q$는 process variance다.

- $q$가 작으면 상태가 천천히 움직여 과거를 오래 믿는다.
- $q$가 크면 새 관측에 빨리 반응하지만 잡음도 drift로 오해할 수 있다.

장기 평균으로 되돌아오게 하려면 AR(1)을 쓴다.

$$
u_t=\rho u_{t-1}+\xi_t,
\qquad
|\rho|<1
$$

$\rho$가 $1$에 가까우면 변화가 오래 지속된다. $\rho=0$이면 인접 날짜 연결이 사라진다. 정상 AR(1)은 proper initial prior, 예를 들어

$$
u_1\sim
\mathcal N\left(
0,\frac{q}{1-\rho^2}
\right)
$$

를 둘 수 있다. random walk는 $\rho=1$인 비정상 경계 사례이므로 초기 상태 $u_1$에 proper prior를 두거나 한 시점을 기준으로 고정하는 anchor가 필요하다.

특정 언어만 변할 수 있게

$$
v_{g,t}=\rho_gv_{g,t-1}+\zeta_{g,t},
\qquad
\zeta_{g,t}\sim\mathcal N(0,q_g)
$$

를 추가할 수 있다. stationary AR(1)으로 해석하려면 $\lvert\rho_g\rvert<1$과 proper initial prior도 필요하다. 데이터가 적으면 $u_t$와 $v_{g,t}$가 서로 같은 변화를 설명하려 할 수 있으므로, 예를 들어 매 시점

$$
\sum_gW_{g,t}v_{g,t}=0
$$

을 두거나 기준 언어의 $v_{g,t}$를 $0$으로 고정하는 식별 제약이 필요하다.

### 3.10 filtering의 핵심: 불확실성에 따른 가중 평균

비선형 로지스틱 모형의 정확한 필터링은 복잡하므로, 핵심 직관을 Gaussian 상태 하나로 보자. 관측 전 현재 상태 예측이

$$
\theta_t\mid\mathcal F_{t-1}
\sim
\mathcal N(m_t^{-},P_t^{-})
$$

이고 새 관측 요약 $z_t$가

$$
z_t\mid\theta_t
\sim
\mathcal N(\theta_t,V_t)
$$

라고 하자. Kalman gain은

$$
K_t
=
\frac{P_t^{-}}{P_t^{-}+V_t}
$$

이고 갱신 평균은

$$
m_t
=
m_t^{-}
+
K_t(z_t-m_t^{-})
$$

이다. 갱신 분산은

$$
P_t=(1-K_t)P_t^{-}
$$

다.

- 새 관측이 정밀해 $V_t$가 작으면 $K_t$가 커져 새 데이터 쪽으로 많이 움직인다.
- 과거 상태가 확실해 $P_t^{-}$가 작으면 $K_t$가 작아져 갑작스러운 작은 표본에 덜 흔들린다.

실제 동적 로지스틱 MRP는 Laplace 근사, 변분추론, 확장 Kalman filter, particle filter, MCMC 같은 방법을 쓸 수 있다. 위 식은 구현을 그대로 뜻하기보다 “두 불확실성의 비율로 신·구 정보를 섞는다”는 원리를 보여 준다.

### 3.11 poststratification은 링크를 통과한 뒤 한다

각 셀의 posterior draw 또는 점추정 $\widehat\eta_{c,t}$를 확률로 바꾼다.

$$
\widehat p_{c,t}
=
\sigma(\widehat\eta_{c,t})
$$

그다음 실제 트래픽 비중으로 합친다.

$$
\widehat P_t^{\mathrm{MRP}}
=
\sum_cW_{c,t}\widehat p_{c,t}
$$

일반적으로

$$
\sigma\left(\sum_cW_{c,t}\widehat\eta_{c,t}\right)
\neq
\sum_cW_{c,t}\sigma(\widehat\eta_{c,t})
$$

이다. sigmoid는 비선형이므로 **로짓을 먼저 평균한 뒤 sigmoid**를 취하면 다른 estimand가 된다.

Bayesian MRP에서는 posterior draw $s=1,\ldots,S$마다

$$
P_t^{(s)}
=
\sum_cW_{c,t}\sigma\left(\eta_{c,t}^{(s)}\right)
$$

를 계산한다. 이 draw들의 평균과 분위수로 전체 위험의 사후평균과 credible interval을 만든다. 셀 예측의 공분산이 이 과정에 자연스럽게 포함된다.

### 3.12 숫자로 보는 MRP

오늘 트래픽이 네 셀로 구성됐다고 하자.

| 셀 | $W_{c,t}$ | MRP 위험 예측 $\widehat p_{c,t}$ | 전체 위험 기여 |
| --- | ---: | ---: | ---: |
| 한국어×텍스트 | $0.45$ | $0.02$ | $0.009$ |
| 한국어×이미지 | $0.05$ | $0.12$ | $0.006$ |
| 영어×텍스트 | $0.40$ | $0.01$ | $0.004$ |
| 영어×이미지 | $0.10$ | $0.08$ | $0.008$ |

따라서

$$
\widehat P_t
=
0.45(0.02)+0.05(0.12)+0.40(0.01)+0.10(0.08)
=
0.027
$$

즉 전체 위험률은 $2.7\%$다.

네 셀을 똑같이 평균하면

$$
\frac{0.02+0.12+0.01+0.08}{4}
=
0.0575
$$

로 $5.75\%$가 된다. 이는 “셀 하나를 균등하게 뽑았을 때”의 평균이지 “요청 하나를 뽑았을 때”의 위험률이 아니다.

### 3.13 전체 위험 변화 = 셀 내부 변화 + 트래픽 구성 변화

두 날짜의 cell set이 다르면 먼저 두 집합의 합집합을 공통 master cell 집합으로 만든다. 해당 날짜에 존재하지 않는 셀은 $W_{c,t}=0$으로 두고, 아래 곱이 정의되도록 비교 기준의 잠재 셀 위험 $p_{c,t}$를 모형 안에서 정의해야 한다.

두 날짜의 전체 위험 차이는 다음처럼 정확히 분해할 수 있다.

$$
\begin{aligned}
P_t-P_{t-1}
&=
\sum_cW_{c,t}p_{c,t}
-
\sum_cW_{c,t-1}p_{c,t-1}
\\
&=
\sum_cW_{c,t}
\left(p_{c,t}-p_{c,t-1}\right)
+
\sum_c
\left(W_{c,t}-W_{c,t-1}\right)
p_{c,t-1}.
\end{aligned}
$$

첫 항은 오늘 트래픽 구성을 기준으로 한 **셀 내부 위험 변화**다. 둘째 항은 어제 셀 위험을 기준으로 한 **트래픽 구성 변화**다. 중간에 어떤 기준을 더하고 빼느냐에 따라 두 항의 귀속은 달라질 수 있으므로 기준시점을 반드시 적어야 한다. 이는 순서에 의존하는 Oaxaca형 산술 항등식이지 “트래픽 구성이 위험을 인과적으로 만들었다”는 인과 분해가 아니다. 하지만 두 항의 합은 정확히 전체 변화다.

네 셀의 Day 1과 Day 2가 다음과 같다고 하자.

$$
\begin{aligned}
\mathbf W_1&=(0.60,0.10,0.25,0.05),\\
\mathbf p_1&=(0.01,0.08,0.02,0.20),\\
\mathbf W_2&=(0.40,0.20,0.30,0.10),\\
\mathbf p_2&=(0.01,0.10,0.02,0.25).
\end{aligned}
$$

전체 위험은

$$
P_1
=
0.60(0.01)+0.10(0.08)+0.25(0.02)+0.05(0.20)
=
0.029
$$

와

$$
P_2
=
0.40(0.01)+0.20(0.10)+0.30(0.02)+0.10(0.25)
=
0.055
$$

다. 총 증가는 $2.6\%p$다. Day 2 가중치를 쓴 셀 내부 변화는

$$
\sum_cW_{c,2}(p_{c,2}-p_{c,1})
=
0.009
$$

즉 $0.9\%p$다. Day 1 위험을 쓴 구성 변화는

$$
\sum_c(W_{c,2}-W_{c,1})p_{c,1}
=
0.017
$$

즉 $1.7\%p$다. 위험 증가의 더 큰 부분은 각 셀이 나빠져서가 아니라 원래 위험이 큰 셀의 트래픽 비중이 커져서 생겼다.

### 3.14 감사 선택이 outcome 모형에 미치는 영향

감사 여부를 $R_i\in\{0,1\}$라 하자. 적응 감사에서는 선택 전 이력 $\mathcal F_{i-1}$까지 조건에 넣은

$$
R_i\perp Y_i\mid \left(\mathcal F_{i-1},X_i\right)
$$

즉 선택 시점의 관측 정보와 과거를 알고 나면 감사 여부가 아직 보지 못한 라벨과 독립인 **sequential MAR** 상황이다. $X_i$가 선택 이력의 충분통계를 모두 포함하는 특별한 경우에만 이를 $R_i\perp Y_i\mid X_i$로 줄여 쓸 수 있다. 이때 올바르게 명시된 조건부 outcome 모형 $p(Y_i\mid X_i,\mathcal F_{i-1})$은 선택모형을 likelihood에서 생략해도 일관될 수 있다.

하지만 다음 중 하나면 안심할 수 없다.

- 감사자가 $X$에 기록되지 않은 단서를 보고 선택했다.
- 위험을 암시하는 사후 정보가 선택에 들어갔다.
- outcome 모형이 중요한 상호작용을 빠뜨렸다.
- 어떤 영역의 감사확률이 $0$이다.

population-weighted pseudo-target에 더 가깝게 맞추는 보조책으로, 알려진 감사확률 $\pi_i$를 이용한 가중 pseudo-likelihood를 생각할 수 있다.

$$
\ell_w(\theta)
=
\sum_{i:R_i=1}
\frac{1}{\pi_i}
\log p_\theta(Y_i\mid X_i,\mathcal F_{i-1})
$$

다만 큰 $1/\pi_i$는 분산과 수치 불안정을 키운다. 이 가중 pseudo-likelihood 자체가 misspecified outcome 모형을 자동으로 고치는 double-robust 방법은 아니다. 다층모형은 분산을 줄이는 데 도움을 주지만 잘못된 선택확률이나 positivity 위반을 마법처럼 고치지 않는다.

### 3.15 contextual bandit의 수학적 틀

시점 $t$마다 다음 순서가 반복된다.

1. 선택 전에 context $X_t$를 본다.
2. 정책 $b_t(a\mid X_t,\mathcal F_{t-1})$가 행동 $A_t$를 뽑는다.
3. 선택한 행동의 reward $U_t(A_t)$만 관측한다.
4. 이력을 $\mathcal F_t$로 갱신한다.

감사 배분에서는 행동이 “어느 queue에서 다음 요청을 감사할까?”가 될 수 있다. reward의 예시는 다음과 같다.

$$
U_t
=
\frac{s_tY_t}{c_t}
+
\lambda I_t
$$

- $s_t$: 위험 심각도
- $Y_t$: 위험 발견 여부
- $c_t$: 감사 비용
- $I_t$: 불확실성 감소 또는 새로움의 근사치
- $\lambda$: 발견과 정보가치의 균형

reward 정의가 목표를 만든다. 단순히 $Y_t$만 최대화하면 흔한 경미 오류만 쫓을 수 있고, $s_tY_t/c_t$만 쓰면 비싼 희귀 집단을 버릴 수 있다. 공정성 하한은 reward에 슬쩍 섞기보다 별도 제약으로 두는 편이 감사 가능하다.

누적 regret는

$$
\operatorname{Regret}(T)
=
\sum_{t=1}^{T}
\left[
\mu(X_t,a_t^\star)-\mu(X_t,A_t)
\right]
$$

이며

$$
a_t^\star
\in
\arg\max_a\mu(X_t,a),
\qquad
\mu(x,a)=\mathbb E[U_t(a)\mid X_t=x]
$$

다. regret가 작다는 것은 발견 효율이 좋다는 뜻이지 위험률 추정이 자동으로 불편하다는 뜻은 아니다.

### 3.16 LinUCB의 예측과 불확실성 보너스

context와 행동을 feature 벡터

$$
\phi_t(a)=\phi(X_t,a)\in\mathbb R^d
$$

로 만들고 기대 reward를 선형으로 근사한다.

$$
\mathbb E[U_t(a)\mid X_t]
\approx
\phi_t(a)^\top\theta
$$

과거에 실제 선택한 feature를 $\phi_s=\phi_s(A_s)$라 하자. ridge 목적함수는

$$
\widehat\theta_t
=
\arg\min_\theta
\left\{
\sum_{s<t}
\left(U_s-\phi_s^\top\theta\right)^2
+
\lambda\|\theta\|_2^2
\right\}
$$

이다. 미분을 $0$으로 놓으면

$$
\left(
\lambda I+\sum_{s<t}\phi_s\phi_s^\top
\right)\widehat\theta_t
=
\sum_{s<t}\phi_sU_s
$$

이다. 다음을 정의하면

$$
V_t
=
\lambda I+\sum_{s<t}\phi_s\phi_s^\top,
\qquad
g_t
=
\sum_{s<t}\phi_sU_s
$$

해는

$$
\widehat\theta_t=V_t^{-1}g_t
$$

다.

행동 $a$의 예측 평균은

$$
\widehat\mu_t(a)=\phi_t(a)^\top\widehat\theta_t
$$

이고, 데이터가 덜 쌓인 방향의 불확실성 크기는

$$
s_t(a)
=
\sqrt{
\phi_t(a)^\top V_t^{-1}\phi_t(a)
}
$$

로 근사한다. LinUCB 점수는

$$
\operatorname{UCB}_t(a)
=
\widehat\mu_t(a)
+
\beta_t s_t(a)
$$

다. 첫 항은 활용, 두 번째 항은 탐색이다.

### 3.17 UCB의 argmax만 쓰면 propensity가 사라진다

순수 LinUCB는

$$
A_t
\in
\arg\max_a\operatorname{UCB}_t(a)
$$

로 결정적으로 선택할 수 있다. 그러면 선택되지 않은 행동의 확률은 $0$이 된다. 이 로그로 그 행동을 고르는 target policy를 평가할 수 없다.

가장 단순한 해결은 $\varepsilon$-mixture다. $a_t^{\mathrm{ucb}}$를 UCB 최선 행동, $q_t(a\mid X_t,\mathcal F_{t-1})$를 탐색 분포라 하면

$$
b_t(a\mid X_t,\mathcal F_{t-1})
=
(1-\varepsilon_t)
\mathbf 1\{a=a_t^{\mathrm{ucb}}\}
+
\varepsilon_tq_t(a\mid X_t,\mathcal F_{t-1})
$$

이다. $K$개 행동에 균등 탐색하면

$$
q_t(a\mid X_t,\mathcal F_{t-1})=\frac{1}{K}
$$

이므로 모든 행동에

$$
b_t(a\mid X_t,\mathcal F_{t-1})\geq\frac{\varepsilon_t}{K}>0
$$

가 보장된다.

행동이 3개이고 $\varepsilon=0.15$라면 UCB 최선 행동의 확률은

$$
0.85+\frac{0.15}{3}=0.90
$$

이고 나머지 행동은 각각

$$
\frac{0.15}{3}=0.05
$$

다. 탐색 하한이 작을수록 발견 효율은 좋아 보일 수 있지만 역가중치 상한은 커진다.

개별 요청을 독립 Bernoulli 방식으로 감사할 때는 예산을 기대값으로 맞출 수도 있다. 요청 $i$의 위험·불확실성·피해와 비용을 합친 점수를

$$
s_i
=
\frac{L_i\left(\widehat p_i+\kappa\,\widehat{\operatorname{sd}}_i\right)}
{c_i}
$$

라 하고, 선택확률을

$$
e_i
=
\varepsilon
+
(1-\varepsilon)
\sigma\left(\frac{s_i-\lambda}{\tau}\right)
$$

로 둔다. $\lambda$를 조정해

$$
\sum_ic_ie_i=B
$$

가 되게 하면 기대 감사비용이 $B$가 된다. 여기서는 $\tau>0$, $c_i>0$, $0\leq\varepsilon<1$을 가정한다. 가능한 예산 범위는

$$
\varepsilon\sum_ic_i
\leq
B
\leq
\sum_ic_i
$$

이다. 유한한 $\lambda$에 대한 유일해는 strict interior

$$
\varepsilon\sum_ic_i
<
B
<
\sum_ic_i
$$

에서 존재한다. 양 끝점은 $\lambda\to+\infty$ 또는 $\lambda\to-\infty$인 극한으로 접근한다. 내부 예산에서는 왼쪽 합이 $\lambda$에 대해 단조 감소하므로 이분법으로 $\lambda$를 찾을 수 있다.

이 식은 **안전한 무작위 정책의 한 예**이지 보편적 최적 bandit 정리가 아니다. 정확히 $k$건을 뽑는 strict quota, 중복 제거, queue 고갈을 적용하면 선택들이 의존하고 nominal $e_i$가 최종 포함확률과 달라질 수 있다. 최종 randomizer의 실제 포함확률을 계산해 로그해야 한다.

### 3.18 IPS가 오프정책 가치를 되살리는 이유

behavior policy $b_t$가 만든 로그로 target policy $\pi$의 기간 평균 reward를 추정하고 싶다. 적응 환경에서는 기대 reward가 과거에 따라 달라질 수 있으므로

$$
\mu_t(x,a,\mathcal F_{t-1})
=
\mathbb E
\left[
U_t(a)
\mid
X_t=x,\mathcal F_{t-1}
\right]
$$

로 정의하고 target을

$$
V_T(\pi)
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb E
\left[
\sum_a
\pi(a\mid X_t)
\mu_t(X_t,a,\mathcal F_{t-1})
\right]
$$

로 둔다. iid·stationary 환경이면 익숙한 $V(\pi)=\mathbb E_X[\sum_a\pi(a\mid X)\mu(X,a)]$로 단순화된다. IPS 추정량은

$$
\widehat V_{\mathrm{IPS},T}(\pi)
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{\pi(A_t\mid X_t)}
{b_t(A_t\mid X_t,\mathcal F_{t-1})}
U_t
$$

다.

왜 되는지 context와 과거 이력을 고정하고 조건부기댓값을 계산해 보자.

$$
\begin{aligned}
&\mathbb E_b
\left[
\frac{\pi(A_t\mid X_t)}
{b_t(A_t\mid X_t,\mathcal F_{t-1})}
U_t
\;\middle|\;
X_t,\mathcal F_{t-1}
\right]
\\
&=
\sum_a
b_t(a\mid X_t,\mathcal F_{t-1})
\frac{\pi(a\mid X_t)}
{b_t(a\mid X_t,\mathcal F_{t-1})}
\mu_t(X_t,a,\mathcal F_{t-1})
\\
&=
\sum_a
\pi(a\mid X_t)
\mu_t(X_t,a,\mathcal F_{t-1}).
\end{aligned}
$$

분자와 분모의 behavior 확률이 소거된다. 이 증명에는 다음 조건이 숨어 있다.

1. $b_t$가 선택 전에 과거와 context로 정해지는 predictable 정책이다.
2. 실제 reward는 선택한 행동의 potential reward라는 consistency가 성립한다.
3. 과거와 context를 조건으로 행동 추첨이 potential rewards와 독립인 sequential randomization이 성립한다.
4. $\pi(a\mid x)>0$인 곳에서 $b_t(a\mid x,\mathcal F_{t-1})>0$이다.
5. 로그의 propensity가 실제 추첨 확률과 정확히 같다.

적응형 정책이어도 이 조건을 지키면 조건부 증명이 가능하다. 하지만 “현재 정책을 과거 데이터에 다시 실행한 확률”은 실제 역사적 propensity가 아니다.

### 3.19 큰 가중치와 ESS

importance weight를

$$
w_t
=
\frac{\pi(A_t\mid X_t)}
{b_t(A_t\mid X_t,\mathcal F_{t-1})}
$$

라 하자. $\pi(A_t\mid X_t)=1/3$, $b_t=0.05$인 탐색 행동이 reward $1$을 얻으면 그 한 건의 가중 기여는

$$
w_tU_t
=
\frac{1/3}{0.05}
=
6.\overline{6}
$$

이다. 평균적으로는 맞지만 몇 건이 추정량을 지배할 수 있다.

가중치의 유효표본크기는

$$
\operatorname{ESS}
=
\frac{\left(\sum_tw_t\right)^2}
{\sum_tw_t^2}
$$

로 진단한다. $T$가 커도 가중치가 매우 불균등하면 ESS는 작다.

가중치를

$$
\widetilde w_t=\min(w_t,c)
$$

로 자르면 분산은 줄지만 일반적으로 불편성이 깨진다. 따라서 clipping 임계값, clipped 비율, unclipped 민감도 분석을 함께 보고해야 한다.

### 3.20 DR 추정량은 예측과 무작위화를 결합한다

선택 직전 정보로 학습한 기대 reward 예측모형을 $\widehat q_t(x,a,\mathcal F_{t-1})$라 하고 target policy 아래 예측 평균을

$$
\widehat q_t(x,\pi,\mathcal F_{t-1})
=
\sum_a\pi(a\mid x)
\widehat q_t(x,a,\mathcal F_{t-1})
$$

라 하자. DR 추정량은

$$
\widehat V_{\mathrm{DR},T}(\pi)
=
\frac{1}{T}
\sum_{t=1}^{T}
\left[
\widehat q_t(X_t,\pi,\mathcal F_{t-1})
+
\frac{\pi(A_t\mid X_t)}
{b_t(A_t\mid X_t,\mathcal F_{t-1})}
\left\{
U_t-\widehat q_t(X_t,A_t,\mathcal F_{t-1})
\right\}
\right]
$$

이다.

첫 항은 모든 행동의 reward를 모형으로 채운 값이고, 두 번째 항은 실제 관측 잔차를 중요도 가중해 교정한다.

- propensity가 정확하면 outcome 모형이 틀려도 잔차 교정의 기대값이 편향을 제거한다.
- outcome 모형이 정확하면 propensity 모형이 맞지 않아도 특정 정규성 조건 아래 일관될 수 있다.
- 둘 다 틀리면 보호막은 없다.

두 번째 문장은 일관성, sequential randomization, positivity와 적절한 정규성 조건을 둔 superpopulation 모형 아래의 일반적인 double-robust 성질이다. 오늘 뒤에서 보일 “라벨을 고정한 조건부 설계 불편성”은 정확한 propensity에만 기대는 별도 주장이다. 감사 시스템에서는 propensity가 정책이 직접 출력한 알려진 확률이어야 한다. 이를 나중에 추정하는 문제로 만들지 않는 것이 가장 좋다. outcome 모형은 시점 $t$의 reward를 본 뒤 같은 관측에 맞춰서는 안 되며, 과적합 편향을 줄이기 위해 시간순 cross-fitting 또는 독립 학습 구간을 사용할 수 있다.

### 3.21 요청 위험률을 적응 감사로 추정하기

각 요청 $i$를 감사할 확률을 $\pi_i$라 하고 실제 감사 여부를 $R_i$라 하자. 유한한 $N$개 요청의 위험률

$$
P_N=\frac{1}{N}\sum_{i=1}^{N}Y_i
$$

에 대한 Horvitz–Thompson형 추정량은

$$
\widehat P_{\mathrm{HT}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{R_iY_i}{\pi_i}
$$

다. 선택 전에 $\pi_i$가 정해지고

$$
\mathbb E[R_i\mid\mathcal F_{i-1},X_i,Y_i]=\pi_i
$$

라면

$$
\mathbb E
\left[
\frac{R_iY_i}{\pi_i}
\;\middle|\;
\mathcal F_{i-1},X_i,Y_i
\right]
=
Y_i
$$

이므로 평균을 복원한다.

단, 고정 quota로 여러 요청을 비복원 추출하면 포함지시자들이 서로 독립이 아니다. 정확한 분산에는 2차 포함확률 $\pi_{ij}$가 필요할 수 있다. 독립 Bernoulli sampling 공식을 quota sampling에 그대로 쓰면 안 된다.

### 3.22 동적 MRP를 baseline으로 쓴 AIPW

전체 요청 $i=1,\ldots,N$ 각각에 대해, 현재 라벨을 보기 전에 동적 MRP가 만든 위험 예측을 $m_i$라 하자. 감사확률은 $e_i$, 감사 여부는 $R_i$, 확정 라벨은 $Y_i$다. 다음 pseudo-outcome을 만든다.

$$
\psi_i
=
m_i
+
\frac{R_i}{e_i}(Y_i-m_i)
$$

전체 위험의 AIPW 추정량은

$$
\widehat P_{\mathrm{AIPW}}
=
\frac{1}{N}\sum_{i=1}^{N}\psi_i
$$

다. $m_i$와 $e_i$가 선택 전에 고정되고 실제 randomizer가 정확히 기록됐다면

$$
\begin{aligned}
\mathbb E_R[\psi_i\mid\mathcal F_{i-1},X_i,Y_i,m_i]
&=
m_i
+
\frac{\mathbb E_R[R_i\mid\cdot]}{e_i}(Y_i-m_i)
\\
&=
m_i+(Y_i-m_i)
\\
&=
Y_i.
\end{aligned}
$$

즉 propensity가 편향을 막는다. 조건부 분산은

$$
\operatorname{Var}_R(\psi_i-Y_i\mid\cdot)
=
\frac{1-e_i}{e_i}(Y_i-m_i)^2
$$

이다. 좋은 MRP가 $Y_i$에 가까우면 잔차가 작아져 분산도 작아진다. 이것이 두 도구의 정확한 역할 분담이다.

같은 블록의 현재 라벨로 $m_i$를 과적합한 뒤 바로 같은 식에 넣으면 “선택 전에 고정” 조건이 깨진다. 이전 mature block에서 모형을 고정하거나, 시간 순서를 지키는 cross-fitting을 사용해야 한다.

Day 2에 총 $N=10{,}000$개 요청, 셀 크기

$$
(N_1,N_2,N_3,N_4)
=
(4{,}000,2{,}000,3{,}000,1{,}000)
$$

가 있다고 하자. 셀별 Bernoulli 감사확률과 관측 위험 건수는

$$
\mathbf e=(0.05,0.15,0.05,0.35),
\qquad
\mathbf y=(2,30,3,88)
$$

다. 기대 감사 수는 $(200,300,150,350)$으로 총 $1{,}000$건이며, 계산을 투명하게 만들기 위해 이 구성 예제에서는 실제 감사 수도 우연히 이 기대수와 같았다고 둔다. 감사 표본 단순 비율은

$$
\frac{2+30+3+88}{1{,}000}
=
0.123
$$

즉 $12.3\%$지만, HT는

$$
\widehat P_{\mathrm{HT}}
=
\frac{
2/0.05+30/0.15+3/0.05+88/0.35
}
{10{,}000}
=
0.055142\ldots
$$

즉 약 $5.514\%$다. 위험한 셀을 많이 본 선택을 되돌린 결과다.

MRP baseline이 앞에서 구한 $5.5\%$이고 처음 세 셀의 관측 위험이 MRP 기대와 정확히 같으며, 네 번째 셀에서만 기대 $350(0.25)=87.5$건 대신 $88$건을 봤다고 단순화하자. AIPW 잔차 교정은

$$
\frac{88-350(0.25)}{0.35(10{,}000)}
=
0.000142857\ldots
$$

즉 약 $0.0143\%p$다. 따라서 AIPW는

$$
5.5\%+0.0143\%p
\approx
5.5143\%
$$

로 HT와 맞는다. 실제 데이터에서는 모든 요청의 잔차가 들어가며 이렇게 정확히 일치할 필요는 없다.

이 예제에서 정확히 맞는 이유는 각 셀의 실제 감사 수가 $N_he_h$와 같고, 셀 안에서 $m_i$가 상수이기 때문이다. MRP가 참값을 정확히 알아서가 아니다. 실제 Bernoulli 추첨에서는 셀별 감사 수가 기대수와 달라지므로 HT와 AIPW는 일반적으로 서로 다른 값을 낸다.

### 3.23 MRP와 HT는 경쟁자가 아니라 서로 다른 안전망이다

| 속성 | 동적 MRP | HT·IPS·DR |
| --- | --- | --- |
| 주된 힘 | 셀 간·시간 간 구조를 빌려 분산 감소 | 알려진 무작위화로 선택 편향 보정 |
| 빈 셀 | 모형 가정 아래 예측 가능 | 양의 포함확률이어도 관측 0건이면 직접 정보 없음 |
| 취약점 | 모형 누락·외삽·prior 민감도 | 작은 propensity의 큰 분산 |
| 불확실성 | posterior·모형 기반 | 설계 기반·martingale 기반 |
| 좋은 용도 | 진단, ranking, 희소 셀 안정화 | 공식 KPI, 정책 비교, 안전 gate |

가장 강한 설계는 둘을 결합한다.

1. MRP로 어디가 위험하고 불확실한지 예측한다.
2. bandit은 이 정보를 이용해 발견 예산을 배분한다.
3. 별도의 무작위 측정 스트림 또는 명시적 탐색 하한을 유지한다.
4. 공식 gate는 설계 기반 추정과 동시 오류 제어로 확인한다.
5. 두 추정이 크게 어긋나면 모형 누락·propensity 오류·drift를 조사한다.

### 3.24 동시 추론: 여러 셀을 고른 뒤 놀라지 않기

$M$개 셀에 각각 오류확률 $\alpha_m$인 구간을 만들고

$$
\sum_{m=1}^{M}\alpha_m\leq\alpha
$$

로 두자. union bound에 의해

$$
\Pr(\text{적어도 한 구간이 실패})
\leq
\sum_{m=1}^{M}\alpha_m
\leq
\alpha
$$

이다. 균등 Bonferroni는

$$
\alpha_m=\frac{\alpha}{M}
$$

를 쓴다. 예를 들어 $M=4$, $\alpha=0.05$이면 각 양측 구간은 $\alpha_m=0.0125$를 쓴다.

이 보장은 미리 정의한 한 보고서의 비교 가족에 대한 것이다. 매일 같은 구간을 새로 만들어 무기한 확인하면 시간축 오류가 다시 누적된다. 그 경우 이전에 배운 alpha-spending, confidence sequence, e-process 같은 anytime-valid 도구가 필요하다.

MRP posterior에서 셀별 $95\%$ credible interval을 만든 것만으로는 자동으로 frequentist family-wise $95\%$ coverage가 생기지 않는다. 진단용 posterior와 배포용 동시 gate의 해석을 구분해야 한다.

고정된 horizon에서 교육용으로 보수적인 폭을 만들 수 있다. 과거에 대해 predictable한 조건부 Bernoulli 감사이고 선택 전에 정한 확률이

$$
e_i\geq\varepsilon>0
$$

를 만족한다고 하자. 셀 $h$의 HT 평균에 대해

$$
Z_i=\frac{R_iY_i}{e_i},
\qquad
D_i=Z_i-Y_i
$$

는 유한모집단 라벨을 고정하고 selection randomness만 보거나, $R_i\perp Y_i\mid(\mathcal F_{i-1},X_i)$인 sequential MAR와 정확한 $e_i$를 가정하면 선택 이력에 대한 martingale difference다. 각 항의 조건부 범위 폭은 최대 $1/\varepsilon$다. 셀 크기가 $N_h$일 때 Azuma–Hoeffding형 폭

$$
w_h
=
\frac{1}{\varepsilon}
\sqrt{
\frac{1}{2N_h}
\log\frac{2}{\alpha_h}
}
$$

를 쓰면 매우 보수적인 fixed-horizon 구간

$$
\widehat P_h\pm w_h
$$

를 얻을 수 있다. $\sum_h\alpha_h\leq\alpha$로 두면 union bound로 가족 오류를 제어한다. $\varepsilon$가 작을 때 구간이 얼마나 폭발하는지 식에서 바로 보인다.

적응형 $e_i$의 사후 실현값을 단순히 $\sum e_i^{-2}$에 넣어 더 날카로운 Hoeffding 폭을 만드는 것은 일반적으로 정당화되지 않는다. 위 식은 deterministic floor만 사용한 단순하고 보수적인 교육용 경계다. 실제 운영에서는 predictable variance를 쓰는 정식 mixture·stitching confidence sequence나 전용 survey variance를 검토한다.

매 시점 반복해서 볼 때는 각 셀의 미리 명시한 귀무가설 $H_h$ 아래에서 유효한 anytime-valid confidence sequence 또는 e-process $E_{h,t}$를 두고

$$
E_{h,t}\geq\frac{1}{\alpha_h}
$$

를 경보 기준으로 삼을 수 있다. 실제로 참인 귀무가설의 집합을 $\mathcal H_0$라 하자. $\sum_h\alpha_h\leq\alpha$이면

$$
\Pr\left(
\exists h\in\mathcal H_0,t:
E_{h,t}\geq\frac{1}{\alpha_h}
\right)
\leq
\sum_{h\in\mathcal H_0}\alpha_h
\leq
\alpha
$$

로 집단과 시간 양쪽의 선택을 제어할 수 있다.

### 3.25 오늘 수학과 AI 운영의 1:1 매핑

| 수학 부품 | AI 감사에서 맡는 부품 |
| --- | --- |
| atomic cell | 언어×모달리티×기기×모델 버전 조합 |
| random effect prior | 희소 집단이 관련 집단에서 정보를 빌리는 규칙 |
| interaction | 특정 조합에서만 나타나는 failure mode |
| AR(1)·random walk | 배포 뒤 시간에 따른 위험 drift |
| sigmoid | 실수 선형 예측자를 위험확률로 변환 |
| poststratification | 감사 표본을 실제 트래픽 구성으로 복원 |
| ridge normal equation | LinUCB의 reward 계수 추정 |
| confidence bonus | 덜 본 queue를 시험하게 하는 탐색 압력 |
| $\varepsilon$-mixture | 모든 중요 행동의 overlap 보장 |
| propensity ratio | 적응 로그를 target policy 분포로 변환 |
| DR residual correction | 위험 예측의 편향을 실제 라벨로 교정 |
| ESS | policy 비교에 실제 정보가 얼마나 남았는지 진단 |
| union bound | 여러 집단 gate의 가족 오류 예산 |

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Contextual bandit은 강화학습의 가장 짧은 형태다

강화학습에서는 행동이 미래 상태와 이후 reward까지 바꾼다. Contextual bandit에서는 한 시점의 구조를 다음처럼 단순화한다.

$$
X_t
\longrightarrow
A_t
\longrightarrow
U_t
$$

행동 뒤 긴 상태 전이가 없고 즉시 reward 하나를 받는다. 그래서 추천할 항목, 보여 줄 광고, 실행할 모델, 감사할 queue를 고르는 문제에 잘 맞는다.

하지만 감사 문제에는 일반 추천과 다른 목적이 있다.

- 추천은 사용자의 만족 reward를 높이면 된다.
- 감사는 위험을 많이 찾는 동시에 전체 위험을 정직하게 측정해야 한다.
- 감사하지 않은 요청의 확정 라벨은 대개 보이지 않는다.
- 발견한 실패를 수정하면 다음 시점의 데이터 생성과정도 바뀐다.

따라서 “bandit reward가 높다”와 “감사 시스템이 통계적으로 안전하다”는 별도 요구사항이다.

### 4.2 전체 시스템 구조

~~~text
[전체 운영 트래픽]
        |
        +--> 선택 전 feature + 시점별 cell count --------+
        |                                                |
        |                                     [동적 교차분류 MRP]
        |                                                |
        |                                   risk + uncertainty 예측
        |                                                |
        +--> [측정용 무작위 스트림] ----+                 |
        |                               |                 v
        +--> [발견용 contextual bandit] +----> [최종 randomizer]
                                                |
                                  propensity + policy version 기록
                                                |
                                          [사람 감사]
                                                |
                              event time / reveal time / label 기록
                                                |
                      +-------------------------+----------------------+
                      |                                                |
            [MRP + poststratification]                      [HT/AIPW + CS]
                      |                                                |
             진단·희소 셀 안정화                         설계 기반 KPI·gate
                      +-------------------------+----------------------+
                                                |
                                  차이 진단 후 확대·HOLD·rollback
~~~

한 모델이 모든 일을 하지 않는다. 각 층의 책임을 분리해야 어느 가정이 깨졌는지 찾을 수 있다.

### 4.3 운영 블록의 안전한 순서

하루 또는 한 시간 같은 고정 블록을 정했다고 하자.

1. **Estimand를 고정한다.** 오늘 도착한 전체 요청의 위험률인지, 완료된 감사의 위험률인지 적는다.
2. **모집단 frame을 저장한다.** 모든 운영 요청의 셀 ID와 $N_{c,t}$를 label과 무관하게 집계한다.
3. **mature label만으로 MRP를 학습한다.** 아직 결과가 안 나온 요청을 안전 $0$으로 넣지 않는다.
4. **현재 블록 예측을 freeze한다.** 각 후보의 $m_i$, 불확실성, bandit feature를 선택 전에 저장한다.
5. **무작위 정책을 실행한다.** 탐색 하한, 비용, quota, 중복 제거를 모두 반영한 최종 선택을 한다.
6. **실제 propensity를 기록한다.** 점수나 정규화 전 확률이 아니라 최종 포함확률을 남긴다.
7. **라벨을 기다린다.** pending·실패·타임아웃을 별도 상태로 유지한다.
8. **두 트랙으로 추정한다.** MRP–poststratification과 HT/AIPW를 둘 다 계산한다.
9. **사전 정의한 gate를 적용한다.** 여러 셀과 반복 확인은 동시·순차 오류를 제어한다.
10. **다음 블록에서 갱신한다.** 현재 블록 결과로 현재 블록의 확률을 소급해 바꾸지 않는다.

이 순서가 깨지면 leakage가 생긴다. 특히 4번의 freeze와 6번의 실제 확률 로그가 핵심이다.

### 4.4 event time과 reveal time

요청이 발생한 시각을 $T_i^{\mathrm{event}}$, 확정 라벨이 도착한 시각을 $T_i^{\mathrm{reveal}}$이라 하자.

10월 14일 요청이 10월 16일에 판정됐더라도 그 라벨은 10월 14일 트래픽의 결과다. 10월 16일 위험률 분모에 넣으면 두 가지가 뒤섞인다.

- 실제 서비스가 위험을 만든 시점
- 사람이 그 위험을 알게 된 시점

동적 MRP의 셀과 poststratification 가중치는 원칙적으로 event time에 붙인다. 운영 대기열과 라벨 지연 분석은 reveal time으로 본다. 두 시각을 모두 저장하지 않으면 drift와 처리 지연을 구분할 수 없다.

### 4.5 poststratification frame은 감사 데이터가 아니다

$N_{c,t}$는 감사 완료 건수가 아니라 **전체 운영 로그에서 센 요청 수**다. 감사 데이터로 셀 비중을 추정하면 bandit이 많이 본 셀이 모집단에서도 큰 것처럼 보인다.

필수 점검은 다음과 같다.

- 모든 대상 요청이 정확히 한 atomic cell에 속하는가?
- 셀 정의 버전이 기간 중 바뀌지 않았는가?
- 누락된 feature를 독립 level로 둘지 제외할지 사전에 정했는가?
- bot·테스트 트래픽·재시도 요청을 estimand에 포함하는가?
- 시간대와 모델 버전이 event time 기준으로 기록됐는가?
- structural zero와 missing cell을 구분했는가?

MRP가 훌륭해도 $W_{c,t}$가 틀리면 전체 위험률은 틀린다.

### 4.6 reward를 잘못 만들면 bandit은 잘못된 일을 매우 잘한다

감사 reward는 최소 세 목표가 충돌한다.

| 목표 | 가능한 reward 신호 | 그대로 최적화할 때의 함정 |
| --- | --- | --- |
| 위험 발견 | $Y_i$ 또는 심각도 $s_iY_i$ | 흔한 집단만 쫓는다. |
| 정보 획득 | posterior variance 감소 | 실제 피해가 작은 불확실성만 쫓을 수 있다. |
| 비용 효율 | reward를 비용 $c_i$로 나눔 | 비싼 취약 집단을 버릴 수 있다. |

실무에서는 다음처럼 분리하는 편이 낫다.

- 목적함수: 발견할 위험의 기대 심각도 또는 정보가치
- 제약: 총 비용, 집단별 최소 확률, 최대 처리량
- 공식 측정: 별도 무작위 스트림 또는 정확한 propensity 보정

공정성을 reward의 작은 가중치 하나로만 넣으면 비용 압박이 커질 때 보호가 사라질 수 있다. 최소 확률·최소 quota처럼 감사 가능한 제약으로 표현해야 한다.

### 4.7 측정 스트림과 발견 스트림

가장 이해하기 쉬운 설계는 예산을 두 스트림으로 나누는 것이다.

1. **측정 스트림:** 층화 무작위 표집과 고정 탐색 하한으로 전체·집단 위험을 잰다.
2. **발견 스트림:** contextual bandit으로 위험·불확실성이 큰 사례를 더 본다.

두 스트림이 조건부 독립 Bernoulli로 요청 $i$를 각각 확률 $p_i^{(M)}$, $p_i^{(D)}$로 선택하고 둘 중 하나라도 선택하면 최종 감사한다고 하자. 최종 포함확률은

$$
\pi_i
=
1-
\left(1-p_i^{(M)}\right)
\left(1-p_i^{(D)}\right)
$$

이다. 단순히 두 확률을 더하면 중복 선택을 두 번 센다.

실제 시스템에서 discovery가 measurement 결과를 보고 같은 배치 안에서 바뀌거나, quota가 먼저 찬 스트림이 다른 스트림을 막으면 조건부 독립이 깨진다. 그때는 위 식을 쓰지 말고 실제 결합 randomizer의 포함확률을 계산해야 한다.

### 4.8 탐색 하한을 시간에 따라 줄여도 될까?

온라인 학습 문헌에서는 탐색확률 $\varepsilon_t$를 줄이기도 한다. $\varepsilon_t$-mixture에서 행동이 계속 available하고 탐색분포가 각 행동에 고정된 양의 질량을 주는 경우, 각 행동을 무한히 자주 볼 최소 조건으로 흔히

$$
\sum_{t=1}^{\infty}\varepsilon_t=\infty
$$

같은 조건을 생각한다. 더 일반적으로는 관심 행동 $a$마다

$$
\sum_{t=1}^{\infty}
b_t(a\mid X_t,\mathcal F_{t-1})
=
\infty
\qquad\text{a.s.}
$$

가 필요하다. 그러나 이것은 “언젠가 계속 본다”는 탐색 성질에 가깝다.

안정적인 역가중 추론은 더 강한 요구를 가진다. 실제 최소 propensity를 $p_{\min,t}=\inf_a b_t(a\mid X_t,\mathcal F_{t-1})$라 하면 $p_{\min,t}\to 0$일 때 최대 역가중치는 $1/p_{\min,t}$로 발산한다. 앞의 균등 $K$-행동 $\varepsilon_t$-mixture에서는 $p_{\min,t}=\varepsilon_t/K$이므로 상한은 $K/\varepsilon_t$다. bounded-weight confidence sequence를 운영하려면 보통

$$
p_{\min,t}\geq p_{\min}>0
$$

같은 고정 하한이 필요하다.

탐색 비용을 줄이고 싶다면 무작정 $0$으로 보내기보다 다음을 함께 본다.

- 정책 비교의 ESS
- 셀별 마지막 관측 시각
- 최대 중요도 가중치
- 동시구간 또는 confidence sequence 폭
- 작은 집단 최소 예상 감사 수

### 4.9 UCB와 Thompson sampling도 자동으로 추론 가능하지 않다

UCB는 점수이고, 결정적 argmax만 쓰면 propensity가 $0$ 또는 $1$이다. Thompson sampling은 posterior에서 표본을 뽑으므로 무작위처럼 보이지만, 실제 행동확률을 계산·저장하지 않으면 나중에 정확한 IPS 분모를 알기 어렵다.

또한 다음 후처리가 최종 확률을 바꾼다.

- 고정 quota
- 이미 감사된 요청의 중복 제거
- 감사자 전문성에 따른 routing
- queue의 일시적 고갈
- 사람의 수동 override
- 비용 한도 도달 뒤의 차단

따라서 “알고리즘 이름”이 아니라 **최종 randomizer가 그 후보를 선택할 조건부 확률**이 propensity다.

### 4.10 nonstationarity를 다루는 세 가지 창

시간 변화에 대응하는 방법마다 다른 편향–분산 절충이 있다.

| 방법 | 장점 | 위험 |
| --- | --- | --- |
| sliding window | 최근 데이터만 써 급격한 변화를 빨리 반영 | 창 밖의 희귀 집단 정보를 버린다. |
| exponential discount | 오래된 정보를 부드럽게 약화 | 할인율 해석과 유효표본크기가 어렵다. |
| 동적 상태모형 | process noise로 변화 정도를 명시 | 모형이 복잡하고 change point에서 지연될 수 있다. |

Bandit의 reward 모형과 MRP의 위험 모형이 같은 forgetting 규칙을 써야 하는 것은 아니다. Bandit은 발견 효율을 빠르게 따라가고, MRP는 위험 estimand를 안정적으로 추정해야 한다.

배포 시각처럼 알려진 change point가 있다면 단순 random walk에만 맡기지 말고 다음을 추가할 수 있다.

$$
\eta_{c,t}
=
\cdots
+
\gamma\,\mathbf 1\{t\geq t_{\mathrm{deploy}}\}
+
\delta_{c,t}
$$

배포 지시자 $\gamma$는 공통 수준 변화를 분리하고, $\delta_{c,t}$는 나머지 drift를 담는다.

### 4.11 지연 라벨을 $0$으로 넣으면 왜 위험할까?

라벨이 아직 도착하지 않은 요청을 안전으로 기록하면 최근 위험률이 인위적으로 낮아진다. 특히 위험 사례일수록 판정이 오래 걸리면 편향이 더 심하다.

각 요청의 상태를 최소한 다음처럼 구분한다.

- eligible: 감사 후보가 됨
- selected: randomizer가 선택함
- pending: 선택됐지만 확정 라벨이 없음
- revealed-safe: 안전 $0$ 확정
- revealed-risk: 위험 $1$ 확정
- censored: 운영상 영원히 관측되지 않음

Bandit 업데이트는 mature reward만 사용하거나 지연 bandit 전용 방법을 써야 한다. 대시보드에는 event-time cohort별 completion rate도 함께 보여야 한다.

### 4.12 동시 gate는 진단 모델과 별도로 설계한다

다음 세 출력은 역할이 다르다.

1. **MRP posterior:** 어디가 위험할 가능성이 큰지 진단한다.
2. **설계 기반 추정:** 전체와 사전 지정 집단의 위험을 보정한다.
3. **동시·상시유효 gate:** 확대·HOLD·rollback 같은 행동의 오류를 제어한다.

예를 들어 네 보호집단에 전체 $\alpha=0.05$를 균등 배분하면

$$
\alpha_h=\frac{0.05}{4}=0.0125
$$

다. 그러나 집단의 피해와 중요도가 다르면 사전에 가중 배분할 수 있다.

$$
\alpha_h=\alpha\omega_h,
\qquad
\omega_h\geq 0,
\qquad
\sum_h\omega_h=1
$$

데이터를 본 뒤 가장 나빠 보이는 셀에 큰 $\alpha_h$를 몰아 주면 오류 제어가 깨진다. 가중치는 데이터를 보기 전에 정하거나, 유효한 온라인 다중검정 규칙으로 갱신해야 한다.

새로운 slice를 탐색적으로 발견했다면 같은 블록에서 공식 gate로 확증하지 않는다. 다음 독립 블록에서 slice 정의와 임계값을 사전 고정해 확인한다.

### 4.13 최소 로그 스키마

| 필드 | 왜 필요한가 |
| --- | --- |
| request_id | 중복 제거와 요청 단위 결합 |
| event_time | 위험이 발생한 시점의 셀·분모 결정 |
| eligibility_time | 언제 후보가 됐는지 확인 |
| cell_definition_version | atomic cell 규칙 재현 |
| context_snapshot | 선택 전에 실제로 본 feature |
| candidate_set | 그 순간 선택 가능했던 행동 |
| score_vector | bandit 진단용 값 |
| final_propensity | HT·IPS·AIPW의 실제 분모 |
| selected_action | 실제 감사 queue 또는 요청 |
| inclusion_indicator | 최종 감사 포함 여부 |
| policy_version | 코드·계수·exploration 규칙 식별 |
| model_version | 평가 대상 AI 버전 식별 |
| RNG_seed_or_draw_id | 추첨 감사와 재현 |
| override_reason | 사람 개입과 확률 변화 기록 |
| label_status | pending·revealed·censored 구분 |
| reveal_time | 지연과 censoring 분석 |
| label_version | 판정 규칙 변경 추적 |
| cost | 비용 제약과 reward 재계산 |
| severity | 단순 오류와 중대 피해 구분 |

score는 propensity가 아니다. 모델이 위험 점수 $0.82$를 냈다고 해서 선택확률이 $0.82$인 것은 아니다.

### 4.14 작은 구현 청사진

~~~text
for each time block t:
    freeze population frame N_cell_t
    train dynamic MRP only on labels mature before block t
    freeze risk prediction m_i and uncertainty for every eligible request

    for each eligible request i:
        compute bandit utility score from pre-selection context
        compute randomized inclusion probability e_i with a fixed floor
        apply the documented final randomizer
        persist e_i, candidate set, policy version, and draw id

    collect labels without turning pending into safe

    when the analysis cutoff arrives:
        poststratify MRP predictions with event-time N_cell_t
        compute HT and pre-block/cross-fitted AIPW
        report weight distribution, ESS, and completion by cell
        update simultaneous confidence sequences

        if any hard-stop event:
            rollback
        else if uncertainty or logging checks fail:
            HOLD
        else if all pre-registered gates pass:
            expand
~~~

이 청사진에서 HOLD는 실패와 성공 사이의 중요한 세 번째 상태다. 데이터가 부족하거나 propensity가 깨졌을 때 “안전”으로 통과시키지 않는다.

### 4.15 필수 단위 테스트와 모니터링

1. **확률 합 테스트:** 상호배타 arm의 $\sum_ab_t(a\mid x)=1$인지 확인한다.
2. **floor 테스트:** 모든 보호 arm에서 $b_t(a\mid x)\geq\varepsilon_a$인지 확인한다.
3. **Monte Carlo 포함확률 테스트:** 고정 후보 집합을 반복 추첨해 로그 확률과 경험 빈도를 비교한다.
4. **quota 경계 테스트:** 예산 소진 직전·직후의 실제 포함확률을 확인한다.
5. **중복 제거 테스트:** 여러 스트림의 합집합 확률이 정확한지 확인한다.
6. **시간 누수 테스트:** 선택 시점 이후 feature나 label이 score에 들어가지 않는지 확인한다.
7. **frame 합계 테스트:** $\sum_cN_{c,t}=N_t$인지 확인한다.
8. **빈 셀 테스트:** structural zero와 표본 0건을 다르게 처리하는지 확인한다.
9. **재현 테스트:** policy version과 draw ID로 선택 과정을 감사할 수 있는지 확인한다.
10. **추론 테스트:** 시뮬레이션에서 HT/AIPW의 coverage와 type-I error를 확인한다.

### 4.16 초보자가 흔히 하는 오해

| 오해 | 왜 틀렸는가 | 바른 생각 |
| --- | --- | --- |
| 교차분류 모형은 모든 셀을 따로 추정한다. | 주효과·상호작용을 공유하는 것이 핵심이다. | 셀은 다르지만 계층구조로 정보를 빌린다. |
| 빈 셀의 MRP 값은 관측된 증거다. | 해당 셀 직접 라벨이 없을 수 있다. | 모형 기반 외삽이며 별도 표시한다. |
| random walk면 change point를 즉시 잡는다. | smoothing prior 때문에 급변을 늦게 따라갈 수 있다. | 배포 지시자·변화점 모형·hard alert를 병행한다. |
| 어제 $W_c$로 오늘 위험을 계산해도 된다. | 트래픽 구성 변화가 빠진다. | event-time의 오늘 population frame을 쓴다. |
| bandit의 높은 발견률이 전체 위험률이다. | 위험해 보이는 사례를 의도적으로 많이 봤다. | HT/AIPW 또는 독립 측정 스트림으로 복원한다. |
| UCB score가 propensity다. | score는 순위이고 확률이 아닐 수 있다. | 최종 randomizer의 조건부 확률을 기록한다. |
| 결정적 top-$k$ 뒤에 propensity를 ML로 추정하면 된다. | 선택되지 않은 영역의 support $0$은 회복되지 않는다. | 처음부터 양의 확률로 무작위화한다. |
| 아주 작은 탐색확률도 양수이므로 충분하다. | 실제 최소 propensity $p_{\min}$가 작으면 최대 역가중치가 $1/p_{\min}$로 폭발한다. | 탐색 비용과 추론 정밀도를 함께 설계한다. |
| Thompson sampling이면 logging이 자동이다. | quota·dedupe·override가 실제 확률을 바꾼다. | 최종 행동확률과 후보 집합을 저장한다. |
| MRP $95\%$ 구간 100개가 $95\%$ 동시보장이다. | 각 구간과 가족 전체 coverage는 다르다. | 사전 $\alpha$ 배분 또는 동시방법을 쓴다. |
| 매 시간 일반 $p<0.05$를 보면 된다. | 반복 peeking으로 오류가 누적된다. | confidence sequence·e-process를 쓴다. |
| 감사가 사용자의 결과를 바꿔도 같은 AIPW면 된다. | 감사 자체가 개입이면 missing-label 문제가 아니다. | 잠재결과와 인과적 bandit 설계가 필요하다. |

### 4.17 오늘의 원리가 모델 학습과 맞물리는 지점

오늘의 내용은 운영 통계에만 머물지 않는다.

- **임베딩:** 언어·모달리티·기기 context를 feature $\phi(x,a)$로 표현한다.
- **손실 함수:** Bernoulli likelihood는 binary cross-entropy가 된다.
- **정규화:** random-effect prior와 ridge 벌점은 과적합을 줄인다.
- **경사하강법:** MRP와 bandit reward 모형의 파라미터를 최적화한다.
- **어텐션:** 여러 feature·시점의 관련 정보를 조건부 예측에 모을 수 있다.
- **강화학습:** contextual bandit은 한 단계짜리 정책 학습이다.
- **오프정책 평가:** 새 선택 정책을 실제 배포 전에 과거 로그로 비교한다.

중요한 차이도 있다. 신경망의 예측력이 높다고 해서 인과적 식별이나 표본설계의 보장이 생기지 않는다. 예측모형은 잔차를 줄이고, 무작위화와 propensity는 어떤 평균을 복원할 수 있는지 결정한다.

### 4.18 실무 체크리스트

- [ ] estimand와 분석 기간을 배포 전에 적었다.
- [ ] atomic cell 정의와 population frame 버전을 저장했다.
- [ ] event time과 reveal time을 모두 기록했다.
- [ ] 현재 블록 label을 보기 전에 MRP 예측을 freeze했다.
- [ ] 중요한 모든 행동에 실질적인 탐색 하한이 있다.
- [ ] 최종 quota·dedupe·override 뒤의 propensity를 기록한다.
- [ ] pending을 안전 라벨로 넣지 않는다.
- [ ] weight 분포·최대값·ESS를 보고한다.
- [ ] MRP와 HT/AIPW의 차이를 진단한다.
- [ ] 여러 셀과 반복 관측을 위한 오류 예산이 있다.
- [ ] exploratory slice는 독립 블록에서 확인한다.
- [ ] hard-stop 사건은 평균·구간과 무관하게 즉시 처리한다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **정보 기준 시각:** 2026-10-15 02:41 KST. 공개 시각·수치·라이선스는 논문 원문과 공식 저장소를 기준으로 확인했다. 아래 성능은 별도 독립 재현이 아니라 각 논문 저자의 보고다.

오늘의 두 연구는 겉으로 다른 문제를 다룬다.

- mmVB는 여러 이미지에 **시각 토큰 예산**을 어떻게 나눌지 묻는다.
- DoubtLess는 추론 중 **사고 토큰 예산**을 어떻게 줄일지 묻는다.

공통 질문은 오늘 배운 것과 같다.

> 모든 입력에 같은 자원을 쓰지 않을 때, 어디에 얼마를 배분하고 품질 손실을 어떻게 감시할 것인가?

### 5.1 먼저 알아둘 트렌드 용어

| 용어 | 뜻 | 주의점 |
| --- | --- | --- |
| visual token | 이미지를 vision encoder와 projector가 언어 모델이 처리할 벡터 열로 바꾼 단위 | 픽셀 수와 토큰 수는 같지 않다. |
| visual budget | 이미지마다 허용할 해상도·시각 토큰 양 | 작게 잡으면 작은 글자·미세 특징을 잃는다. |
| pre-encoding | 비전 인코더가 이미지를 처리하기 전 단계 | 여기서 해상도를 줄이면 encoder 계산도 절약한다. |
| prefill | 입력 토큰 전체의 KV 상태를 만드는 첫 계산 단계 | 입력이 길수록 비용과 TTFT가 커진다. |
| TTFT | Time To First Token | 사용자가 첫 출력 토큰을 보기까지의 시간이다. |
| FLOPs | 부동소수점 연산 횟수의 근사 | 실제 지연시간과 완전히 같지는 않다. |
| ordinal prediction | 순서가 있는 level을 예측하는 문제 | level 3과 2의 차이를 3과 0보다 가깝게 다룬다. |
| black-box transfer | 대상 모델 내부를 바꾸거나 읽지 않고 입력만 바꿔 적용 | 모델별 최적화가 아니라 전이 가능성의 증거다. |
| logit | softmax 전의 점수 | 차이를 더하면 다음 토큰 확률비가 바뀐다. |
| contrastive decoding | 두 조건의 logit 차이로 기본 모델의 출력을 유도 | 대조 방향과 강도가 품질을 좌우한다. |
| CoT | Chain of Thought, 답을 내기 전의 추론 토큰 열 | 길수록 항상 정확한 것은 아니다. |
| thinking tag | 추론 구간을 표시하는 특별 토큰 | 모델 가족마다 형식이 다를 수 있다. |
| top-$p$ | 누적확률이 $p$에 도달하는 후보 집합에서 샘플링 | 후보 제한이 logit 조정 범위를 바꾼다. |
| training-free | 대상 큰 모델 가중치를 다시 학습하지 않음 | 추가 모델 실행이나 튜닝 비용이 없다는 뜻은 아니다. |

### 5.2 VLM: Visual Tokens on Demand와 mmVB

Amazon 연구진의 **Visual Tokens on Demand: Visual Budgeting for Scalable and Efficient Multi-Image Reasoning in Multimodal LLMs**는 arXiv v1이 2026-10-13 17:59 UTC, 즉 2026-10-14 02:59 KST에 제출됐다.

#### 왜 이 문제가 중요한가?

여러 이미지를 한꺼번에 보는 MLLM은 이미지마다 많은 시각 토큰을 만든다. 모든 이미지를 최고 해상도로 처리하면

- vision encoder 계산량이 커지고,
- LLM prefill 토큰이 길어지며,
- 첫 토큰 지연시간과 메모리가 증가한다.

그러나 모든 이미지를 같은 비율로 축소하면 질문에 결정적인 작은 글자나 비교 대상을 잃을 수 있다. 핵심은 **총 토큰 수뿐 아니라 이미지 사이에 어떻게 나누는가**다.

#### mmVB는 어떻게 동작하는가?

이미지 $N$개와 질문 $Q$가 있을 때 각 이미지에

$$
\ell_i\in\{0,1,2,3,4\}
$$

의 해상도 level을 배정한다.

구조는 다음과 같다.

1. 동결된 EfficientNet-B0가 각 이미지의 저비용 전역 표현을 만든다.
2. 동결된 MLLM embedding layer가 질문을 표현하고 평균 pooling한다.
3. 이미지 토큰 $N$개와 질문 토큰 하나를 2층·4-head Transformer가 함께 본다.
4. 공유 MLP가 이미지마다 네 개의 순서형 logit을 만든다.

각 임계확률은

$$
\Pr(\ell_i\geq k)
=
\sigma(z_{i,k}),
\qquad
k\in\{1,2,3,4\}
$$

로 예측하고

$$
\widehat\ell_i
=
\sum_{k=1}^{4}
\mathbf 1\{\sigma(z_{i,k})>0.5\}
$$

로 level을 정한다. 학습 가능한 파라미터는 약 $1.6$M이고 저자 보고 추가 지연은 약 $2$ms다.

기반 MLLM을 다시 학습하지 않고, 추론 때 dense attention matrix를 읽지 않는다. mmVB의 경량 EfficientNet-B0 예측은 먼저 실행되지만, 이미지를 **기반 MLLM의 주 vision encoder**에 넣기 전에 예측 level로 축소하므로 그 encoder와 LLM 계산을 둘 다 줄일 수 있다.

#### 저자 보고 결과

Qwen3.5-4B를 기반 모델로 사용하고 greedy decoding·thinking off 조건에서 8개 다중 이미지 벤치마크를 평가했다. 벤치마크별 비율의 기하평균 기준으로 mmVB는

- full-resolution 정확도의 $94.4\%$ 유지
- 시각 토큰 $9.2\times$ 감소
- prefill FLOPs $7.3\times$ 감소
- TTFT 약 $5\times$ 개선

을 보고했다.

Qwen3.5-4B로 학습한 budgeter를 미학습 모델에 입력 전처리로 적용한 black-box 전이에서는 8개 벤치마크 비율의 기하평균으로 다음을 보고했다.

| 대상 모델 | full-resolution 대비 정확도 유지 | 입력·시각 토큰 감소 |
| --- | ---: | ---: |
| Qwen3.5-27B | $92.9\%$ | $9.1\times$ |
| Claude Opus 4.6 | $89.8\%$ | $3.9\times$ |
| Pixtral Large | $89.9\%$ | $4.8\times$ |

“정확도 유지 $94.4\%$”는 정확도가 $94.4\%p$라는 뜻이 아니다. 예를 들어 full-resolution 정확도 $50\%$에서 $94.4\%$를 유지하면 새 정확도는 약 $47.2\%$다.

#### 라벨은 진짜 최소 예산인가?

논문은 먼저 각 이미지 level을 올려 보며 정답이 유지되는 level을 찾고, 여러 이미지를 함께 축소한 뒤 다시 검증한다. 실패하면 일부 level을 완화하며, 마지막에는 full resolution으로 되돌린다.

따라서 생성된 level은 **정답을 유지한다고 확인된 feasible allocation**이다. 모든 level 조합을 완전탐색한 전역 최소 예산은 아니다. 저자들도 greedy relaxation이 두 이미지를 동시에 올려야 하는 더 싼 조합을 놓쳐 보수적으로 과할당할 수 있다고 설명한다.

이 구분은 오늘 bandit의 “높은 점수”와 “최적 정책 증명”을 구분한 것과 같다.

#### 한계와 재현성

- 학습 데이터는 VQAv2·TextVQA·GQA 등 자연 이미지 QA와 Mantis-Instruct의 다중 이미지 표본에 집중된다.
- 의료 영상의 미세 병변처럼 몇 픽셀이 결정적인 영역에서는 토큰을 과소 배정할 수 있다고 저자들이 명시한다.
- 핵심 결과는 특정 기반 모델·decoding·벤치마크 조건의 저자 보고다.
- 논문은 TTFT 측정의 GPU·batch·warm-up 같은 시스템 조건을 상세히 공개하지 않으므로 $5\times$ 수치를 다른 서빙 환경에 그대로 옮길 수 없다.
- 논문 원고는 CC BY 4.0이지만, 기준 시각에 arXiv 문서에는 공식 코드 저장소와 구현 라이선스가 연결돼 있지 않았다. 논문 공개와 구현 오픈소스를 같은 것으로 보면 안 된다.

#### 엔지니어 인사이트 (Impact)

첫째, 압축 지점을 LLM 내부가 아니라 **입력 해상도 결정**으로 당기면 FlashAttention을 유지하고, attention map을 주지 않는 API 모델에도 입력 전처리 형태로 적용할 여지가 있다.

둘째, 평균 토큰 절감만 보면 위험하다. OCR·도면·의료처럼 작은 픽셀이 중요한 slice에 최소 해상도 floor를 두고, budgeter의 불확실성이 크면 full resolution으로 fallback해야 한다.

셋째, 이는 오늘의 적응 감사와 같은 자원 배분 문제다. 이미지별 “필요 토큰” 예측이 틀릴 수 있으므로 도메인별 회귀표본을 무작위로 full resolution에 남기고, 정확도 손실을 propensity-aware하게 감시하는 설계가 가능하다.

**1차 출처**

- [arXiv 초록·제출 기록](https://arxiv.org/abs/2610.18858)
- [논문 HTML 원문](https://arxiv.org/html/2610.18858)

### 5.3 LLM: DoubtLess의 training-free contrastive decoding

**DoubtLess: Training-Free Contrastive Decoding for Efficient Reasoning**도 arXiv v1이 2026-10-13 17:59 UTC, 즉 2026-10-14 02:59 KST에 제출됐다.

#### 왜 이 문제가 중요한가?

추론 모델은 답을 이미 찾은 뒤에도 다시 계산하고 검증하는 토큰을 길게 생성할 수 있다. 무조건 일찍 끊으면 어려운 문제의 필요한 추론까지 잃는다. 모델마다 다시 학습하면 비용이 크다.

DoubtLess는 큰 모델을 재학습하지 않고 작은 모델이 매 토큰에서 “간결함 방향”을 알려 주도록 한다.

#### logit 차이가 어떻게 작동하는가?

큰 base model의 다음 토큰 logit을 $\mathbf z_{\mathrm{base}}$라 하자. 같은 작은 모델이 현재까지의 추론을 두 번 읽는다.

- brief instruction 아래 logit: $\mathbf z_{\mathrm{brief}}$
- verbose instruction 아래 logit: $\mathbf z_{\mathrm{verbose}}$

새 logit은

$$
\mathbf z
=
\mathbf z_{\mathrm{base}}
+
\alpha
\left(
\mathbf z_{\mathrm{brief}}
-
\mathbf z_{\mathrm{verbose}}
\right)
$$

다. $\alpha$는 steering strength다.

- $\alpha=0$이면 기본 모델과 같다.
- 양의 $\alpha$는 작은 모델이 “간결함에 더 어울린다”고 보는 토큰을 올린다.
- 너무 큰 $\alpha$는 필요한 검증까지 억제할 수 있다.

논문 구현은 큰 모델의 top-$p$ 후보 안에서만 이 조정을 적용하고, 추론 종료를 막지 않도록 EOS와 &lt;/think&gt;의 base logit은 그대로 둔다. 조정은 &lt;think&gt;...&lt;/think&gt; 구간에서만 작동한다.

#### 저자 보고 결과

Qwen3-32B를 Qwen3-0.6B로 steer한 기본 설정에서, 경쟁 수학·대학원 수준 과학·경쟁 프로그래밍의 7개 벤치마크를 합쳐

- $95\%$ bootstrap interval 기준 유의한 정확도 저하 없이 평균 사고 토큰 $31\%$ 감소
- self-debugging HumanEval+ agent에서 유의한 성능 손실 없이 관측 pass rate $2.0\%p$ 상승, turn당 사고 토큰 $20\%$ 감소

를 보고했다.

핵심 벤치마크 설정은 $\alpha=5$, temperature $0.6$, top-$p=0.95$, top-$k=20$이며, 사고 토큰 예산은 대부분 $32{,}768$, GPQA는 $24{,}576$이다. 총 528문제를 문제당 3개 seed로 평가했으며, $\alpha=5$는 AIME 2026에서 선택됐다. 따라서 “유의한 저하 없음”은 이 실험 범위의 주장이다.

지연시간 측정은 B200 GPU 한 장에서 7개 벤치마크당 2문제, 총 14문제를 한 번에 하나씩 decoding한 소규모 조건이다.

| 방법 | 문제당 평균 지연 |
| --- | ---: |
| 기본 decoding | $156.2$s |
| 간결 prompt | $137.5$s |
| DoubtLess | $119.4$s |

저자 보고로 기본 decoding 대비 약 $24\%$, 간결 prompt 대비 약 $13\%$ 줄었다. 작은 $0.6$B 모델을 두 번 실행하는 추가 FLOPs는 $32$B 모델 대비 약 $4\%$로 계산됐다.

#### 강도 $\alpha$는 무료 손잡이가 아니다

AIME 2025 ablation에서 기본값 $\alpha=5$까지는 유의한 정확도 저하가 보고되지 않았지만

$$
\alpha=10
$$

에서는 정확도가 $7.8\%p$ 하락했다. 더 큰 보조 모델도 같은 $\alpha$에서 더 강한 대조 방향을 만들어 과도하게 줄일 수 있었다.

이는 오늘의 bandit과 같다. 더 강한 활용은 단기 효율을 높일 수 있지만, 탐색·품질 floor를 없애면 blind spot이 생긴다. $\alpha$는 모델 이름만 보고 복사할 상수가 아니라 업무별로 검증할 정책 매개변수다.

#### 오픈소스 상태와 한계

- 공식 코드는 Apache-2.0 라이선스로 공개됐다.
- 현재 README는 Linux, NVIDIA GPU, vLLM 0.29.0을 요구한다.
- 예제의 Qwen3-32B와 Qwen3-0.6B 구성은 80GB GPU 한 장을 사용한다.
- 큰 모델과 작은 모델은 token ID 대응과 thinking 종료 형식이 맞아야 한다.
- logits를 제공하지 않는 폐쇄형 API에는 그대로 적용하기 어렵다.
- 평가는 수학·과학·코드에 집중되어 일반 대화·장문·다국어 품질은 별도 검증이 필요하다.
- latency 수치는 14문제의 특정 B200 환경이므로 서비스 p95·동시성·KV-cache 조건으로 일반화하면 안 된다.

#### 엔지니어 인사이트 (Impact)

첫째, 추론 길이를 가중치 재학습 문제가 아니라 배포 시 조절 가능한 decoding policy로 바꾼 점이 실용적이다.

둘째, 평균 토큰만 줄었다고 성공은 아니다. 정확도, calibration, 사고 토큰, 전체 출력 토큰, p50·p95 지연, GPU 메모리, 실패 유형을 함께 A/B 테스트해야 한다.

셋째, 모든 요청에 하나의 $\alpha$를 쓰기보다 난이도·도메인 context에 따라 여러 안전한 $\alpha$ 중 하나를 고르는 contextual bandit을 생각할 수 있다. 이때 reward를 비용 절감만으로 만들면 어려운 문제의 정확도를 희생하므로 품질 constraint와 탐색 하한이 필요하다.

**1차 출처**

- [arXiv 초록·제출 기록](https://arxiv.org/abs/2610.18862)
- [논문 HTML 원문](https://arxiv.org/html/2610.18862)
- [공식 GitHub 저장소](https://github.com/leoluolol/DoubtLess)

### 5.4 두 트렌드가 오늘 수학에 주는 공통 교훈

| 오늘의 원리 | mmVB | DoubtLess |
| --- | --- | --- |
| context | 질문과 여러 이미지 | prompt와 지금까지의 CoT |
| action | 이미지별 해상도 level | logit steering strength·방향 |
| 제한 자원 | 시각 토큰·prefill FLOPs | 사고 토큰·decoding 시간 |
| 예측 | 필요한 시각 상세 수준 | brief와 verbose의 다음 토큰 차이 |
| floor·fallback | 불확실하면 full resolution | EOS 보존·안전한 $\alpha$ |
| 평가 | 정확도 유지와 토큰·TTFT | 정확도와 토큰·전체 지연 |
| drift 위험 | 도메인·모델이 바뀜 | 문제 유형·모델 가족이 바뀜 |
| 필요한 감사 | 도메인별 full-resolution 대조군 | 요청별 기본 decoding 대조군 |

두 연구 모두 “균등 자원 배분”을 넘어선다. 그러나 적응적 배분의 진짜 품질은 평균 효율 수치 하나가 아니라, 어떤 slice가 자원을 덜 받았고 그 선택확률과 품질 손실을 다시 측정할 수 있는지에 달려 있다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

AI 서비스가 네 atomic cell을 추적한다.

1. 한국어×텍스트
2. 한국어×이미지
3. 영어×텍스트
4. 영어×이미지

Day 1과 Day 2의 트래픽 비중과 동적 MRP 셀 위험 예측은 다음과 같다. 따라서 아래 $P_1$, $P_2$도 참 위험률 자체가 아니라 MRP poststratified 추정치다.

$$
\begin{aligned}
\mathbf W_1&=(0.60,0.10,0.25,0.05),&
\mathbf p_1&=(0.01,0.08,0.02,0.20),\\
\mathbf W_2&=(0.40,0.20,0.30,0.10),&
\mathbf p_2&=(0.01,0.10,0.02,0.25).
\end{aligned}
$$

Day 2의 요청 수는 $N=10{,}000$, 셀 크기는 $(4{,}000,2{,}000,3{,}000,1{,}000)$이다. 적응 감사의 셀별 포함확률은

$$
\mathbf e=(0.05,0.15,0.05,0.35)
$$

이고, 실제 감사 수도 기대수와 같은 $(200,300,150,350)$건이었다. 관측한 위험 건수는

$$
\mathbf y=(2,30,3,88)
$$

이다.

다음을 스스로 설명하라.

1. Day 1과 Day 2의 전체 위험률은 얼마인가?
2. 증가분을 셀 내부 위험 변화와 트래픽 구성 변화로 나누면 각각 얼마인가?
3. 감사 표본의 단순 위험률과 HT 위험률은 왜 다르며 각각 얼마인가?
4. Day 2 MRP 예측을 사전 고정한 AIPW는 이 구성 예제에서 얼마인가?
5. 네 셀을 매시간 반복 감시하려면 어떤 오류 제어와 로그가 필요한가?

### 모범 답안

#### 1단계: 두 날짜의 전체 위험률

Day 1은

$$
\begin{aligned}
P_1
&=
0.60(0.01)
+
0.10(0.08)
+
0.25(0.02)
+
0.05(0.20)
\\
&=
0.006+0.008+0.005+0.010
\\
&=
0.029.
\end{aligned}
$$

따라서 Day 1의 MRP poststratified 위험 추정치는 $2.9\%$다.

Day 2는

$$
\begin{aligned}
P_2
&=
0.40(0.01)
+
0.20(0.10)
+
0.30(0.02)
+
0.10(0.25)
\\
&=
0.004+0.020+0.006+0.025
\\
&=
0.055.
\end{aligned}
$$

따라서 Day 2의 MRP poststratified 위험 추정치는 $5.5\%$다. 추정치의 증가는

$$
0.055-0.029=0.026
$$

즉 $2.6\%p$다.

#### 2단계: 위험 변화와 구성 변화

Day 2 가중치를 기준으로 한 셀 내부 위험 변화는

$$
\begin{aligned}
\Delta_{\mathrm{rate}}
&=
\sum_cW_{c,2}(p_{c,2}-p_{c,1})
\\
&=
0.40(0)
+
0.20(0.02)
+
0.30(0)
+
0.10(0.05)
\\
&=
0.004+0.005
\\
&=
0.009.
\end{aligned}
$$

즉 $0.9\%p$다.

Day 1 위험을 기준으로 한 구성 변화는

$$
\begin{aligned}
\Delta_{\mathrm{mix}}
&=
\sum_c(W_{c,2}-W_{c,1})p_{c,1}
\\
&=
(-0.20)(0.01)
+
(0.10)(0.08)
+
(0.05)(0.02)
+
(0.05)(0.20)
\\
&=
-0.002+0.008+0.001+0.010
\\
&=
0.017.
\end{aligned}
$$

즉 $1.7\%p$다.

검산하면

$$
0.9\%p+1.7\%p=2.6\%p
$$

다. 이는 기준시점을 정한 산술 분해이며 인과효과 주장이 아니다.

#### 3단계: 단순 평균과 HT

감사 표본의 단순 위험률은

$$
\widehat P_{\mathrm{naive}}
=
\frac{2+30+3+88}{200+300+150+350}
=
\frac{123}{1{,}000}
=
0.123
$$

즉 $12.3\%$다.

하지만 네 번째 위험 셀은 $35\%$를 감사했고 첫째·셋째 셀은 $5\%$만 감사했다. 선택확률을 되돌리는 HT는

$$
\begin{aligned}
\widehat P_{\mathrm{HT}}
&=
\frac{1}{10{,}000}
\left(
\frac{2}{0.05}
+
\frac{30}{0.15}
+
\frac{3}{0.05}
+
\frac{88}{0.35}
\right)
\\
&=
\frac{
40+200+60+251.428571\ldots
}
{10{,}000}
\\
&=
0.055142857\ldots.
\end{aligned}
$$

즉 약 $5.514\%$다. 단순 비율이 높았던 이유는 위험 셀을 의도적으로 많이 감사했기 때문이다.

#### 4단계: MRP–AIPW

MRP baseline은

$$
\frac{1}{N}\sum_im_i=P_2=0.055
$$

다. 처음 세 셀은 관측 위험 건수가 각각

$$
200(0.01)=2,
\qquad
300(0.10)=30,
\qquad
150(0.02)=3
$$

으로 MRP 기대와 같아 잔차 합이 $0$이다. 네 번째 셀의 잔차 보정은

$$
\frac{88-350(0.25)}
{0.35(10{,}000)}
=
\frac{0.5}{3{,}500}
=
0.000142857\ldots
$$

다. 따라서

$$
\widehat P_{\mathrm{AIPW}}
=
0.055+0.000142857\ldots
=
0.055142857\ldots
$$

즉 약 $5.514\%$다.

이 예제에서 HT와 정확히 맞는 것은 셀별 실제 감사 수가 $N_he_h$와 같고 셀 안의 $m_i$가 상수인 특수 구성 때문이다. 일반 데이터에서는 두 추정치가 다를 수 있다.

#### 5단계: 동시·순차 gate와 로그

한 번의 고정 보고서라면 네 셀에

$$
\alpha_h=\frac{0.05}{4}=0.0125
$$

를 배분하는 Bonferroni 방식으로 가족 오류를 제어할 수 있다. 그러나 매시간 반복해서 확인한다면 고정시점 구간을 재사용하면 안 된다. 각 셀에 anytime-valid confidence sequence 또는 e-process를 두고

$$
\sum_h\alpha_h\leq 0.05
$$

를 유지해야 한다.

최소 로그는 다음과 같다.

- 선택 전에 본 context와 전체 candidate set
- 최종 randomizer가 준 실제 포함확률 $e_i$
- 선택 여부·행동·policy version·model version
- quota·dedupe·override와 draw ID
- event time·reveal time·pending/censored 상태
- cell-definition version과 전체 population count $N_{c,t}$

가장 중요한 한 줄은 다음이다.

> MRP는 잔차를 작게 만들 수 있지만, 정확한 propensity만이 적응 선택을 설계 기반으로 되돌린다.

---

**다음 학습 예고:** 오늘은 주어진 정책의 위험과 발견 효율을 추정했다. 다음에는 로그에서 새 감사 정책을 직접 학습할 때 과대평가를 막는 **pessimistic off-policy learning**, 안전 제약을 둔 정책 개선, 그리고 배포 전 lower confidence bound를 다룬다.
