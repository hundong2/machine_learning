# [2026-10-16] 오늘 학습: 최댓값 선택 편향·균등 신뢰경계 & 비관적 오프정책 Contextual Bandit 학습

> **오늘의 핵심 문장:** 한 후보 정책의 오프정책 가치를 추정하는 것과, 같은 로그를 뒤져 가장 좋아 보이는 정책을 고르는 것은 전혀 다른 통계 문제다. 정책을 고르는 과정까지 포함해 동시에 유효한 신뢰경계를 만들고, 불확실성이 큰 후보의 점수를 깎는 비관적 학습을 해야 “로그에서 우연히 이긴 정책”을 “배포해도 되는 정책”으로 착각하지 않는다.

지난 시간에는 교차분류·시간가변 MRP로 실제 트래픽의 위험률을 복원하고, contextual bandit이 감사 대상을 적응적으로 골라도 당시의 propensity를 기록하면 IPS·DR·AIPW로 정책 가치를 평가할 수 있음을 배웠다.

오늘은 한 단계 더 나아간다.

- 어제는 **주어진 정책을 평가**했다.
- 오늘은 같은 로그에서 **새 정책을 직접 학습하고 선택**한다.
- 후보가 많아질수록 가장 높은 추정값에는 좋은 성능뿐 아니라 운 좋은 잡음도 섞인다.
- 로그가 거의 없는 행동은 추정 불확실성이 크지만, 단순 최대화기는 바로 그 빈틈으로 달려간다.
- 따라서 평균 추정값에서 불확실성 벌점을 뺀 **비관적 목적함수**가 필요하다.
- 품질뿐 아니라 안전·비용 제약에도 후보 전체에 동시에 유효한 상한을 적용해야 한다.
- 마지막에는 정책 학습에 한 번도 쓰지 않은 독립 로그로 고정된 후보를 검증해야 한다.

> **난이도 표지:** 오늘 내용은 고급 확장이지만 출발점은 “시험 문제를 여러 번 풀어 가장 잘 나온 점수만 기록하면 실력이 부풀려진다”는 직관이다. 9월 21~23일의 IPS·DR·LCB·제약 최적화와 10월 15일의 적응 감사 로그를 재사용하되, 오늘의 새 핵심은 **정책 선택 이후에도 유효한 균등 경계**다.

9월 22일에는 “후보가 여러 개면 선택 편향을 보정해야 한다”고 경고했다. 오늘은 그 경고를 유한 정책군의 uniform bound로 실제 유도한다.

10월 15일에는 union bound로 여러 셀을 동시에 보호했다. 오늘은 같은 논리를 여러 정책 후보에 적용해, 선택된 정책에도 남는 LCB를 만든다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 고정된 정책 평가만으로는 부족할까?

과거 로그로 후보 정책 하나의 평균 보상을 추정했다고 하자.

$$
\widehat V(\pi)
$$

여기서 $\pi$는 context를 보고 행동 확률을 정하는 정책이고, $V(\pi)$는 그 정책을 실제로 실행했을 때의 평균 보상이다.

후보가 사전에 딱 하나로 고정돼 있었다면 핵심 질문은 단순하다.

> 추정 오차를 고려해도 이 후보가 기준 정책보다 나은가?

하지만 실제 개발 과정은 보통 이렇다.

1. 수십 개의 feature 조합을 만든다.
2. 학습률·온도·정규화 강도를 바꾼다.
3. 여러 reward 정의를 시험한다.
4. 같은 로그에서 모든 후보를 평가한다.
5. 가장 높은 점수를 보인 후보만 보고한다.

이제 선택된 후보는 더 이상 “사전에 고정된 후보 하나”가 아니다. 데이터의 우연한 잡음을 보고 선택된 **데이터 의존적 후보**다.

### 1.2 왜 가장 높은 점수에는 행운이 섞일까?

후보 $k$의 추정값을

$$
\widehat V_k=V_k+\varepsilon_k
$$

라고 쓰자.

- $V_k$: 실제 가치
- $\varepsilon_k$: 유한한 로그 때문에 생긴 추정 잡음

우리는 보통

$$
\widehat k
=
\arg\max_{1\leq k\leq K}\widehat V_k
$$

를 고른다.

모든 후보의 실제 가치가 같더라도, $\widehat k$는 가장 큰 양의 잡음을 얻은 후보가 되기 쉽다. 즉

$$
\widehat V_{\widehat k}
-
V_{\widehat k}
$$

의 기댓값은 $0$ 이상이고, 후보별 잡음이 완전히 같은 값으로 움직이지 않는 비퇴화 상황에서는 보통 양수다.

이를 **최댓값 선택 편향**, **optimizer’s curse**, 또는 **winner’s curse**라고 부른다. 최적화기가 성능뿐 아니라 오차까지 최적화해 버린 것이다.

### 1.3 왜 pointwise 95% 구간 100개는 전체 95%가 아닐까?

각 후보마다 따로 만든 $95\%$ 신뢰구간은 그 후보가 미리 고정됐을 때의 장기 포함률을 말한다.

후보가 $100$개이고 각 구간의 실패 사건이 독립이라고 단순화하면, 적어도 하나가 실패할 확률은

$$
1-0.95^{100}
\approx
0.994
$$

다.

독립성은 실제 정책 후보 사이에서 성립하지 않을 수 있지만, 핵심은 변하지 않는다.

> 여러 구간 중 가장 유리한 하나를 고르면, 각 구간의 개별 보장만으로는 선택된 후보를 보호할 수 없다.

우리가 원하는 것은

$$
\Pr\left(
\forall \pi\in\Pi:
V(\pi)\geq \operatorname{LCB}(\pi)
\right)
\geq
1-\delta
$$

처럼 정책군 $\Pi$ 전체에 동시에 성립하는 **균등 보장**이다.

### 1.4 왜 로그가 적은 행동을 낙관하면 위험할까?

행동 $a$가 context $x$에서 과거 정책에 의해 거의 선택되지 않았다면

$$
b(a\mid x)\approx 0
$$

이다.

중요도 가중치

$$
w
=
\frac{\pi(a\mid x)}{b(a\mid x)}
$$

는 커지고, 추정 분산도 커진다.

단순 최대화

$$
\max_{\pi}\widehat V(\pi)
$$

는 “좋다는 증거가 충분한 행동”과 “자료가 없어 우연히 좋아 보이는 행동”을 구분하지 못한다.

비관적 학습은

$$
\max_{\pi}
\left[
\widehat V(\pi)-\beta(\pi)
\right]
$$

처럼 정책별 불확실성 반경 $\beta(\pi)$를 뺀다.

증거가 부족한 후보는 큰 벌점을 받고, 충분히 관측된 후보는 작은 벌점을 받는다.

### 1.5 왜 안전 비용에는 하한이 아니라 상한이 필요할까?

품질 보상 $R$은 클수록 좋다. 따라서 품질 개선에는 보수적인 하한을 본다.

$$
\operatorname{LCB}_{R}(\pi)
$$

반면 권한 위반·유해 응답·지연·비용 같은 cost $C_j$는 작을수록 좋다. 안전 제약에는 보수적인 상한을 본다.

$$
\operatorname{UCB}_{C_j}(\pi)
$$

안전한 정책은 평균 품질이 높아 보이는 정책이 아니라, 다음을 함께 만족하는 정책이다.

$$
\operatorname{LCB}_{\Delta R}(\pi)\geq m
$$

$$
\operatorname{UCB}_{\Delta C_j}(\pi)\leq d_j
\qquad
\text{for every }j
$$

여기서 $m$은 요구되는 최소 품질 개선폭이고, $d_j$는 허용 가능한 cost 증가 한도다.

### 1.6 왜 학습·선택·최종 검증 데이터를 나눌까?

같은 데이터를 반복해서 보면 마지막 숫자는 그 반복을 기억한다.

- **학습 로그:** 정책 파라미터와 후보를 만든다.
- **선택 로그:** 여러 후보와 하이퍼파라미터 중 하나를 고른다.
- **최종 게이트 로그:** 잠근 후보 하나를 마지막으로 검증한다.

최종 게이트 결과를 본 뒤 정책을 다시 바꾸면 그 데이터도 학습 데이터가 된다. 그러면 새 최종 게이트가 필요하다.

이 분리는 데이터를 낭비하는 의식이 아니라, “정책을 만든 증거”와 “정책이 실제로 좋은지 확인하는 증거”를 분리하는 장치다.

### 1.7 역사적 흐름

오늘의 개념은 여러 분야의 오래된 문제를 하나로 묶는다.

1. 통계학은 여러 가설을 동시에 보면 거짓 발견이 늘어남을 다중검정으로 다뤘다.
2. 학습이론은 한 모델이 아니라 함수군 전체에서 경험값과 참값의 차이를 제어하는 균등수렴을 발전시켰다.
3. 강화학습은 로그 밖 행동을 가치함수가 과대평가하는 extrapolation error를 겪었다.
4. 오프라인 RL·bandit은 불확실한 영역의 가치를 낮춰 보는 pessimism을 안전한 학습 원리로 사용했다.
5. 현대 AI 운영은 품질·안전·비용을 동시에 보고, 후보 생성과 독립 배포 게이트를 분리한다.

### 1.8 오늘의 전체 지도

오늘 배울 연결은 다음과 같다.

$$
\text{적응적으로 수집된 로그}
\longrightarrow
\text{DR 정책가치 추정}
\longrightarrow
\text{최댓값 선택 편향}
\longrightarrow
\text{정책군 전체의 균등 경계}
$$

$$
\longrightarrow
\text{비관적 목적함수}
\longrightarrow
\text{품질 LCB·안전 UCB 제약}
\longrightarrow
\text{독립 final gate}
\longrightarrow
\text{shadow·canary}
$$

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| context $X$ | 행동을 고르기 전에 보이는 정보 | 언어, 모달리티, 위험 점수, 시간대 등을 담는다. |
| action $A$ | 정책이 고르는 선택지 | 감사 queue, 모델, 해상도, 도구, 응답 전략 등이 될 수 있다. |
| reward $R$ | 클수록 좋은 결과 | 정확도, 해결 성공, 유용성 등을 수치화한다. |
| cost $C_j$ | 작을수록 좋은 위험·비용 | 유해 응답, 권한 위반, 지연, GPU 비용 등을 따로 기록한다. |
| policy $\pi(a\mid x)$ | context $x$에서 행동 $a$를 고를 확률 규칙 | 오늘 학습할 대상이다. |
| behavior policy $b_i$ | 로그의 $i$번째 행동을 실제로 고른 과거 정책 | 중요도 가중치의 분모다. |
| baseline policy $\pi_0$ | 현재 운영 중인 기준 정책 | 후보 개선량을 비교하는 기준이다. |
| target policy | 로그로 평가하거나 배포하려는 정책 | behavior policy와 같을 필요는 없다. |
| policy class $\Pi$ | 고려 가능한 정책들의 집합 | 후보 수와 복잡도가 선택 편향을 결정한다. |
| deterministic policy | 각 context에서 한 행동을 확률 $1$로 고르는 정책 | 단순하지만 overlap을 잃기 쉽다. |
| stochastic policy | 여러 행동에 양의 확률을 남기는 정책 | 탐색과 오프정책 평가를 가능하게 한다. |
| propensity | 실제 선택 행동에 부여된 확률 | 반드시 당시 최종 확률을 로그에 남겨야 한다. |
| overlap·positivity | target이 고를 행동을 behavior도 양의 확률로 고르는 조건 | 관측되지 않은 행동은 로그만으로 식별할 수 없다. |
| importance weight $w_i$ | $\pi(A_i\mid X_i)/b_i(A_i\mid X_i)$ | behavior 로그를 target 정책 분포로 바꾼다. |
| IPS | 중요도 가중 평균으로 정책 가치를 추정하는 방법 | 단순하지만 큰 가중치에 민감하다. |
| outcome model $\widehat q(x,a)$ | context와 행동에서 기대 결과를 예측하는 모델 | DR의 낮은 분산 기준값이다. |
| DR | outcome model과 중요도 보정을 결합한 이중 강건 추정 | 오늘 정책가치 점수의 기본 부품이다. |
| nuisance model | 최종 관심 대상은 아니지만 추정에 필요한 보조 모델 | $\widehat q$와 추정 propensity가 여기에 속한다. |
| cross-fitting | 한 fold에서 보조 모델을 학습하고 다른 fold의 점수를 계산하는 절차 | 같은 표본을 외워 생기는 편향을 줄인다. |
| off-policy evaluation | 새 정책을 실행하지 않고 과거 정책 로그로 가치를 추정하는 일 | OPE라고 줄여 부른다. |
| off-policy learning | 과거 정책 로그로 새 정책 자체를 학습하는 일 | OPL이라고 줄여 부른다. |
| empirical value $\widehat V$ | 유한한 로그에서 계산한 정책 가치 | 참값 $V$와 추정 오차가 있다. |
| improvement $\Delta(\pi)$ | $V(\pi)-V(\pi_0)$ | 기준보다 얼마나 좋아지는지를 직접 본다. |
| standard error | 추정량이 표본마다 흔들리는 크기의 추정치 | 신뢰경계 폭을 정한다. |
| confidence interval | 반복 표본에서 정해진 비율로 참값을 덮는 절차 | 데이터가 고정된 뒤 참값이 무작위라는 뜻은 아니다. |
| LCB | lower confidence bound, 신뢰하한 | 품질이 보수적으로 얼마나 좋은지 본다. |
| UCB | upper confidence bound, 신뢰상한 | 위험이 보수적으로 얼마나 나쁠 수 있는지 본다. |
| pointwise bound | 미리 고정된 한 후보에 대한 경계 | 후보를 보고 고른 뒤에는 그대로 쓰면 안 된다. |
| uniform bound | 정책군의 모든 후보에 동시에 성립하는 경계 | 선택된 후보에도 보장이 남는다. |
| union bound | 여러 실패 사건의 합집합 확률을 각 확률의 합으로 누르는 부등식 | 유한 후보군의 균등 경계를 만든다. |
| Hoeffding inequality | 독립·유계 확률변수 평균의 꼬리를 제어하는 부등식 | 가장 단순한 균등 경계를 유도한다. |
| empirical Bernstein bound | 관측 분산이 작을 때 더 좁아질 수 있는 유계 평균 경계 | 후보별 불확실성 차이를 반영한다. |
| maximum selection bias | 여러 잡음 섞인 값 중 최댓값을 고를 때 생기는 낙관 편향 | 오늘의 핵심 문제다. |
| optimizer’s curse | 최적화기가 참성능과 추정오차를 함께 최대화하는 현상 | 후보가 많고 로그가 약할수록 심해진다. |
| multiple-policy selection | 같은 로그에서 여러 정책을 비교해 하나를 고르는 절차 | 다중 비교 보정이 필요하다. |
| pessimism | 불확실한 후보의 가치를 낮춰 보는 원리 | 평균 추정값에서 uncertainty penalty를 뺀다. |
| uncertainty radius $\beta(\pi)$ | 정책 $\pi$의 추정 오차를 덮도록 만든 반경 | 비관적 목적함수의 벌점이다. |
| support penalty | behavior가 거의 선택하지 않은 행동에 주는 벌점 | 로그 밖으로 뛰는 정책을 막는다. |
| effective sample size | 불균등 가중 표본이 균등 표본 몇 개와 비슷한지 나타내는 진단값 | 경계의 대체물이 아니라 불안정성 경보다. |
| clipping | 지나치게 큰 가중치를 잘라내는 것 | 분산을 줄이지만 편향을 만든다. |
| sample splitting | 데이터를 학습·선택·최종 검증 용도로 분리하는 것 | 선택 편향을 구조적으로 차단한다. |
| final gate | 잠근 정책 하나를 독립 데이터에서 검증하는 마지막 관문 | 통과 후에도 온라인 canary가 필요하다. |
| pre-registration | 평가 지표·후보·중단 규칙을 결과 보기 전에 고정하는 것 | 사후 규칙 변경을 막는다. |
| constraint | 반드시 지켜야 하는 제한식 | 안전·비용·지연 한도를 표현한다. |
| Lagrange multiplier $\lambda_j$ | 제약 위반에 가격을 붙이는 음이 아닌 변수 | 품질과 여러 cost를 함께 최적화한다. |
| dual ascent | 위반된 제약의 multiplier를 키우는 갱신법 | 위험이 큰 후보에 더 큰 벌점을 준다. |
| stochastic mixture | 기준 정책과 후보 정책을 확률적으로 섞은 정책 | 작은 변화부터 안전하게 시험한다. |
| fallback | 불확실하거나 지원이 약할 때 기준 정책으로 되돌리는 규칙 | 모델이 모르는 영역에서의 기본 안전장치다. |
| shadow test | 사용자에게 영향을 주지 않고 후보 결정을 병렬 계산하는 시험 | parser·router·latency 문제를 먼저 찾는다. |
| canary | 작은 무작위 트래픽에서 후보를 실제 실행하는 단계 | 오프라인에서 보지 못한 배포 위험을 확인한다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 관측 로그와 목표를 정확히 적는다

$n$개의 logged interaction을 생각하자.

$$
\mathcal D
=
\left\{
(X_i,A_i,R_i,C_{i1},\ldots,C_{iJ},b_i)
\right\}_{i=1}^{n}
$$

각 기록은 다음 순서로 생긴다.

**1단계 — context:** $X_i$가 관측된다.

**2단계 — behavior policy:** 과거와 현재 context만 사용해 $b_i$가 정해진다.

**3단계 — action:** 행동은

$$
A_i\sim b_i(\cdot\mid X_i,\mathcal H_{i-1})
$$

로 선택된다.

**4단계 — outcome:** 보상 $R_i$와 cost $C_{ij}$가 나중에 관측된다.

$\mathcal H_{i-1}$은 $i$번째 결정 전까지의 이력이고, 행동 직전 정보장은

$$
\mathcal G_i
=
\sigma(\mathcal H_{i-1},X_i)
$$

다. 미래 결과를 사용하지 않고 선택확률을 정했다는 성질을 **predictability**라고 한다.

적응 로그를 인과적으로 해석하려면 행동 $a$를 골랐을 때의 잠재 결과 $R_i(a)$와 $C_{ij}(a)$를 생각하고 다음 조건을 둔다.

**Consistency**

$$
R_i=R_i(A_i),
\qquad
C_{ij}=C_{ij}(A_i).
$$

**Sequential randomization**

$$
A_i
\perp
\left\{
R_i(a),C_{ij}(a)
\right\}_{a,j}
\mid
\mathcal G_i.
$$

즉, 기록된 과거와 현재 context를 조건으로 하면 행동 추첨이 아직 보지 못한 잠재 결과를 몰래 사용하지 않는다. 여기에 정확한 logged propensity와 overlap이 필요하다.

적응 환경의 시점별 outcome mean을

$$
q_{R,i}(h,x,a)
=
\mathbb E
\left[
R_i(a)
\mid
\mathcal H_{i-1}=h,X_i=x
\right]
$$

라 하고, 이력까지 사용할 수 있는 target policy의 시점별 조건부 가치를

$$
v_{R,i}(\pi)
=
\sum_a
\pi(a\mid X_i,\mathcal H_{i-1})
q_{R,i}(\mathcal H_{i-1},X_i,a)
$$

라 하자. 유한 기간 평균 estimand는

$$
V_{R,n}(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb E
\left[
v_{R,i}(\pi)
\right]
$$

다. cost도 $R$을 $C_j$로 바꿔 같은 방식으로 정의한다.

이 값은 로그가 보여 준 context·history 분포에서 **현재 bandit 행동만** target policy로 바꾸는 평균이다. 감사 행동이 미래 요청의 도착이나 상태 자체를 바꾸지 않는다는 no-carryover 해석이 필요하다. 현재 행동이 다음 상태를 바꾸는 에이전트·MDP 문제라면 trajectory 전체의 target-induced state distribution과 누적 importance ratio를 다뤄야 한다.

이제 3.2절부터 3.24절까지는 균등 경계의 핵심을 선명하게 보려고 다음 교육용 단순화를 사용한다.

- 환경은 stationary다.
- 이 교육용 구간에서는 context와 outcome score의 평가 행이 독립이며, outcome-adaptive behavior는 3.25절로 미룬다.
- target policy는 현재 $X_i$만 쓰는 memoryless policy다.
- candidate class는 평가 결과를 보기 전에 고정된다.

이때

$$
q_{R,i}(h,x,a)=q_R(x,a),
\qquad
V_{R,n}(\pi)=V_R(\pi)
$$

로 단순화된다. 3.25절에서 적응 로그의 martingale 형태로 다시 돌아간다.

또한 보상과 cost는

$$
0\leq R_i\leq 1,
\qquad
0\leq C_{ij}\leq 1
$$

로 정규화했다고 하자.

### 3.2 정책 가치는 무엇인가?

행동별 조건부 기대 보상을

$$
q_R(x,a)
=
\mathbb E[R\mid X=x,A=a]
$$

라 하자.

정책 $\pi$의 가치는

$$
V_R(\pi)
=
\mathbb E_X
\left[
\sum_{a\in\mathcal A}
\pi(a\mid X)q_R(X,a)
\right]
$$

이다.

cost $j$의 정책 값도 같은 모양이다.

$$
V_{C_j}(\pi)
=
\mathbb E_X
\left[
\sum_{a\in\mathcal A}
\pi(a\mid X)q_{C_j}(X,a)
\right]
$$

오늘의 품질 개선량은

$$
\Delta_R(\pi)
=
V_R(\pi)-V_R(\pi_0)
$$

이고 cost 변화량은

$$
\Delta_{C_j}(\pi)
=
V_{C_j}(\pi)-V_{C_j}(\pi_0)
$$

이다.

같은 표본에서 후보와 기준의 score를 먼저 빼는 것과 두 표본평균을 나중에 빼는 것은 점추정으로는 대수적으로 같다. 핵심은 paired score로 표준오차를 계산해야 두 추정량 사이 covariance가 보존되고, 그 covariance가 양수일 때 차이의 분산이 작아질 수 있다는 점이다.

### 3.3 overlap이 없으면 왜 식별할 수 없을까?

target policy가 어떤 $(x,a)$에 양의 확률을 주는데 behavior policy가 그 행동을 절대 선택하지 않았다면

$$
\pi(a\mid x)>0,
\qquad
b_i(a\mid x,\mathcal H_{i-1})=0
$$

이다.

그러면 중요도 비율

$$
\frac{\pi(a\mid x)}{b_i(a\mid x,\mathcal H_{i-1})}
$$

은 정의되지 않는다.

더 본질적으로는 그 행동의 결과가 로그에 없으므로, 서로 다른 두 세계가 관측 로그는 똑같지만 그 행동의 보상만 다를 수 있다. 로그만 보고 어느 세계인지 구분할 수 없다.

따라서

$$
\pi(a\mid x)>0
\Longrightarrow
b_i(a\mid x,\mathcal H_{i-1})>0
$$

가 필요하다.

실무에서는 더 강하게

$$
b_i(a\mid x,\mathcal H_{i-1})
\geq
\varepsilon
$$

같은 exploration floor를 둔다.

### 3.4 IPS가 target 정책의 평균을 복원하는 이유

중요도 가중치를

$$
w_i(\pi)
=
\frac{\pi(A_i\mid X_i)}
{b_i(A_i\mid X_i,\mathcal H_{i-1})}
$$

라 하자.

IPS 추정량은

$$
\widehat V_{\mathrm{IPS}}(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}
w_i(\pi)R_i
$$

이다.

$X_i=x$와 과거 이력 $\mathcal H_{i-1}=h$를 고정하면

$$
\begin{aligned}
\mathbb E_{A_i\sim b_i}
\left[
\frac{\pi(A_i\mid x)}{b_i(A_i\mid x,h)}R_i
\middle|
\mathcal G_i
\right]
&=
\sum_a
b_i(a\mid x,h)
\frac{\pi(a\mid x)}{b_i(a\mid x,h)}
\mathbb E[R_i(a)\mid\mathcal G_i]
\\
&=
v_{R,i}(\pi).
\end{aligned}
$$

behavior 확률이 약분되어 target 정책의 행동 혼합이 남는다. stationary·memoryless 단순화에서는

$$
v_{R,i}(\pi)
=
\sum_a\pi(a\mid X_i)q_R(X_i,a)
$$

가 된다.

### 3.5 DR 점수는 예측과 무작위화를 결합한다

교차 적합으로 만든 outcome model을

$$
\widehat q_R^{(-k(i))}(x,a)
$$

라고 쓰자. $i$번째 표본이 속한 fold는 이 모델 학습에 사용하지 않았다는 뜻이다.

정책 $\pi$의 DR 점수는

$$
\psi_i^R(\pi)
=
\sum_a
\pi(a\mid X_i)
\widehat q_R^{(-k(i))}(X_i,a)
+
w_i(\pi)
\left[
R_i-
\widehat q_R^{(-k(i))}(X_i,A_i)
\right]
$$

이다.

첫 항은 모든 행동의 예측 평균이고, 둘째 항은 실제로 선택된 행동의 예측 오차를 propensity로 보정한다.

정확한 logged propensity를 사용하고 $\widehat q_R$가 해당 관측의 결과를 보지 않고 만들어졌다면, outcome model이 틀려도 조건부 기댓값에서 오차가 상쇄된다.

$$
\begin{aligned}
\mathbb E_b
\left[
\psi_i^R(\pi)
\middle|
\mathcal G_i,
\widehat q_R^{(-k(i))}
\right]
&=
\sum_a
\pi(a\mid X_i)
\widehat q_R^{(-k(i))}(X_i,a)
\\
&\quad+
\sum_a\pi(a\mid X_i)
\left[
q_R(X_i,a)
-
\widehat q_R^{(-k(i))}(X_i,a)
\right]
\\
&=
\sum_a\pi(a\mid X_i)q_R(X_i,a).
\end{aligned}
$$

위 식은 정확한 known propensity를 사용한 경우다. propensity도 $\widehat b$로 추정한다면 “outcome model 또는 propensity model 중 하나가 일관되면 된다”는 double robustness를 말할 수 있지만, positivity, nuisance consistency와 수렴률, 적절한 cross-fitting 같은 추가 조건이 필요하다. 또한 outcome model만 cross-fit했다고 해서 같은 평가 결과를 보고 학습한 정책 $\pi$가 자동으로 고정되는 것은 아니다.

따라서

$$
\widehat V_{\mathrm{DR}}(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}
\psi_i^R(\pi)
$$

로 가치를 추정한다.

### 3.6 개선량은 paired score로 계산한다

기준 정책 $\pi_0$와 후보 $\pi$의 표본별 차이를

$$
D_i^R(\pi)
=
\psi_i^R(\pi)-\psi_i^R(\pi_0)
$$

라고 하자.

그러면

$$
\widehat\Delta_R(\pi)
=
\frac{1}{n}
\sum_{i=1}^{n}
D_i^R(\pi)
$$

이다.

표본분산은

$$
s_R^2(\pi)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
\left[
D_i^R(\pi)-\widehat\Delta_R(\pi)
\right]^2
$$

이고, 독립 표본이라는 단순한 경우 표준오차는

$$
\widehat{\operatorname{SE}}_R(\pi)
=
\frac{s_R(\pi)}{\sqrt n}
$$

이다.

사용자별로 여러 요청이 묶이거나, 날짜별 공통 충격이 있으면 요청 단위 독립 표준오차를 쓰면 안 된다. 사용자·세션·날짜 같은 독립에 가까운 단위로 cluster-robust variance 또는 block bootstrap을 사용해야 한다.

### 3.7 고정된 후보 하나의 LCB

정규근사가 적절하고 후보 $\pi$가 데이터 보기 전에 고정됐다고 하자.

한쪽 $1-\alpha$ 신뢰하한은

$$
\operatorname{LCB}_{1-\alpha}^{\mathrm{point}}(\pi)
=
\widehat\Delta_R(\pi)
-
z_{1-\alpha}
\widehat{\operatorname{SE}}_R(\pi)
$$

이다.

예를 들어 $\alpha=0.05$이면

$$
z_{0.95}\approx 1.645
$$

다.

하지만 이 식을 같은 데이터에서 $100$개 후보를 비교한 뒤 승자에게만 적용하면, “후보가 고정돼 있다”는 전제가 깨진다.

### 3.8 최댓값 선택 편향을 수식으로 본다

후보들의 실제 가치가 모두 $V$이고 추정오차가 평균 $0$이라고 하자.

$$
\widehat V_k=V+\varepsilon_k
$$

선택한 최댓값은

$$
\max_k\widehat V_k
=
V+\max_k\varepsilon_k
$$

이므로

$$
\mathbb E\left[\max_k\widehat V_k\right]
=
V+
\mathbb E\left[\max_k\varepsilon_k\right]
\geq
V.
$$

오차가 독립인 정규분포

$$
\varepsilon_k\sim\mathcal N(0,\sigma^2)
$$

에 가깝다면 큰 $K$에서 최대 오차의 선도적 규모이자 간단한 상계는

$$
\sigma\sqrt{2\log K}
$$

수준으로 자란다.

$K=100$, $\sigma=0.01$이면

$$
0.01\sqrt{2\log 100}
\approx
0.030
$$

이다.

이다. 이 선도식은 약 $3$%p 규모를 보수적으로 보여 준다. 독립 표준정규 후보 $100$개의 실제 기대 최댓값은 약 $2.51$이므로 이 예에서는 약 $2.5$%p다. 어느 쪽도 실제 후보군의 정확한 보정식은 아니며, 후보 사이 상관과 분산 차이를 반영해야 한다.

### 3.9 union bound는 왜 강력한가?

사건 $E_1,\ldots,E_K$에 대해

$$
\Pr\left(
\bigcup_{k=1}^{K}E_k
\right)
\leq
\sum_{k=1}^{K}\Pr(E_k)
$$

가 항상 성립한다.

후보 사이가 독립일 필요도 없다.

각 후보의 경계 실패확률을

$$
\frac{\delta}{K}
$$

이하로 만들면

$$
\Pr(\text{후보 중 하나라도 실패})
\leq
K\frac{\delta}{K}
=
\delta
$$

다.

이것이 Bonferroni 보정과 유한 정책군 균등 경계의 가장 단순한 원리다.

### 3.10 Hoeffding 부등식에서 균등 경계를 유도한다

유한 후보군 $\Pi$는 평가 block의 결과를 보기 전에 고정됐다고 하자. 독립 $\mathcal D_{\mathrm{train}}$만으로 후보를 만들었다면 $\mathcal D_{\mathrm{train}}$에 조건부로 고정된 집합으로 취급할 수 있다.

먼저 설명을 위해 score $Z_i(\pi)$가 서로 독립이고

$$
\ell\leq Z_i(\pi)\leq u
$$

를 만족한다고 하자. 범위 폭을

$$
B=u-\ell
$$

라 쓰자.

Hoeffding 부등식은 고정된 $\pi$에 대해

$$
\Pr\left(
\left|
\widehat\mu(\pi)-\mu(\pi)
\right|>t
\right)
\leq
2\exp\left(
-\frac{2nt^2}{B^2}
\right)
$$

를 준다.

$K=|\Pi|$개의 정책에 union bound를 적용하면

$$
\Pr\left(
\exists\pi\in\Pi:
\left|
\widehat\mu(\pi)-\mu(\pi)
\right|>t
\right)
\leq
2K\exp\left(
-\frac{2nt^2}{B^2}
\right).
$$

오른쪽을 $\delta$로 두고 $t$를 풀면

$$
t
=
B
\sqrt{
\frac{\log(2K/\delta)}{2n}
}.
$$

따라서 확률 적어도 $1-\delta$로 모든 정책에 동시에

$$
\mu(\pi)
\geq
\widehat\mu(\pi)
-
B
\sqrt{
\frac{\log(2K/\delta)}{2n}
}
$$

가 성립한다.

균등 LCB는

$$
\operatorname{LCB}_{\mathrm{unif}}(\pi)
=
\widehat\mu(\pi)
-
B
\sqrt{
\frac{\log(2K/\delta)}{2n}
}
$$

이다.

### 3.11 왜 선택된 정책에도 보장이 남는가?

균등 사건을

$$
\mathcal E
=
\left\{
\forall\pi\in\Pi:
\mu(\pi)\geq\operatorname{LCB}_{\mathrm{unif}}(\pi)
\right\}
$$

라 하자.

데이터를 보고 고른 정책

$$
\widehat\pi
=
\arg\max_{\pi\in\Pi}
\operatorname{LCB}_{\mathrm{unif}}(\pi)
$$

도 $\Pi$ 안에 있다.

$\mathcal E$가 성립하는 모든 데이터셋에서는 자동으로

$$
\mu(\widehat\pi)
\geq
\operatorname{LCB}_{\mathrm{unif}}(\widehat\pi)
$$

다.

여기서 데이터에 의존해도 되는 것은 이미 고정된 $\Pi$ 안에서의 선택 index $\widehat\pi$다. 후보 자체를 같은 평가 데이터의 결과를 보며 생성·수정했다면, 관측된 $K$개에 대한 union bound만으로는 충분하지 않다. 그 경우 전체 가능한 함수군을 덮는 복잡도 경계나 독립 selection block이 필요하다.

### 3.12 IPS에서 범위가 왜 커질 수 있을까?

보상이 $[0,1]$에 있고

$$
b_i(a\mid x)\geq\varepsilon
$$

이면

$$
0
\leq
\frac{\pi(A_i\mid X_i)}{b_i(A_i\mid X_i)}R_i
\leq
\frac{1}{\varepsilon}.
$$

따라서 IPS 점수의 범위 폭은 최대

$$
B_{\mathrm{IPS}}
=
\frac{1}{\varepsilon}
$$

이다.

$\varepsilon=0.01$이면 $B_{\mathrm{IPS}}=100$이다. Hoeffding 경계는 매우 넓어질 수 있다.

이것은 수학의 실패가 아니라 데이터의 경고다.

> 과거 정책이 어떤 행동을 거의 하지 않았다면, 그 행동을 많이 하는 새 정책의 가치를 로그만으로 정밀하게 알 수 없다.

### 3.13 DR 점수도 원래 보상 범위를 벗어날 수 있다

$\widehat q_R(x,a)\in[0,1]$이고 중요도 가중치가 $0\leq w_i\leq W$라고 하자.

DR 점수는

$$
\psi_i^R(\pi)
=
\underbrace{
\sum_a\pi(a\mid X_i)\widehat q_R(X_i,a)
}_{[0,1]}
+
\underbrace{
w_i
\left(
R_i-\widehat q_R(X_i,A_i)
\right)
}_{[-W,W]}
$$

이므로 거칠게

$$
-W
\leq
\psi_i^R(\pi)
\leq
1+W
$$

다.

범위 폭은

$$
B_{\mathrm{DR}}
\leq
1+2W
$$

다. 이는 단일 $\psi_i^R(\pi)$의 범위 폭이다. 후보와 기준의 차이

$$
D_i^R(\pi)
=
\psi_i^R(\pi)-\psi_i^R(\pi_0)
$$

를 쓸 때 두 정책 모두 $w\leq W$라면, 별도 구조를 이용하지 않은 거친 범위 폭 상계는

$$
B_D
\leq
2+4W
$$

다.

따라서 DR이 평균적으로 분산을 낮출 수 있어도, 아무 가정 없이 좁은 유계 경계가 자동으로 생기지는 않는다. 실제 구현은 overlap 제한, cross-fitting, cluster 구조, clipping 편향을 함께 보고 경계를 선택해야 한다.

### 3.14 분산을 반영하는 empirical Bernstein 반경

모든 후보에 대해 개선량 score $D_i(\pi)$가 **사전에 정한** 길이 $B$의 구간 안에 있고, $n\geq2$인 i.i.d. 평가 표본과 평가 결과를 보기 전에 고정한 후보군이라는 교육용 설정을 유지하자. 관측된 표본 최솟값·최댓값을 보고 $B$를 사후 결정하면 이 보장은 성립하지 않는다.

후보별 표본분산을

$$
s^2(\pi)
=
\frac{1}{n-1}
\sum_{i=1}^{n}
\left[
D_i(\pi)-\widehat\Delta(\pi)
\right]^2
$$

라 하면, 널리 쓰이는 empirical Bernstein 형태의 균등 반경은

$$
\beta_{\mathrm{EB}}(\pi)
=
\sqrt{
\frac{
2s^2(\pi)\log(4K/\delta)
}{n}
}
+
\frac{
7B\log(4K/\delta)
}{
3(n-1)
}
$$

처럼 쓸 수 있다. 여기서는 모든 후보의 양쪽 편차

$$
\left|
\widehat\Delta(\pi)-\Delta(\pi)
\right|
$$

를 동시에 덮도록 실패확률을 배분했다. LCB만 필요한 한쪽 정리를 직접 쓰면 약간 더 좁은 상수를 사용할 수 있다.

정확한 상수는 사용하는 정리와 분산 정의에 따라 달라질 수 있으므로 구현에서는 선택한 정리의 조건·상수를 그대로 고정해야 한다.

핵심 구조는 다음 두 항이다.

$$
\text{반경}
\approx
\sqrt{
\frac{
\text{분산}\times\log(K/\delta)
}{n}
}
+
\frac{
\text{범위}\times\log(K/\delta)
}{n}.
$$

분산이 작은 후보는 더 좁은 경계를 얻지만, 극단값 가능성을 나타내는 범위 항은 사라지지 않는다. 이 식은 임의의 적응 로그나 서로 의존하는 cross-fit score에 자동으로 유효하지 않으며, 그 경우 로그 생성과 fold 구조에 맞는 martingale·cluster 경계가 필요하다.

### 3.15 비관적 정책 학습의 목적함수

정책별 균등 반경 $\beta_R(\pi)$를 만들었다면

$$
\widehat\pi_{\mathrm{pess}}
=
\arg\max_{\pi\in\Pi}
\left[
\widehat\Delta_R(\pi)
-
\beta_R(\pi)
\right]
$$

로 정책을 고를 수 있다.

대괄호 안은 바로 품질 개선량의 균등 LCB다.

$$
\widehat\Delta_R(\pi)
-
\beta_R(\pi)
=
\operatorname{LCB}_{\Delta R}(\pi).
$$

평균 추정값이 조금 더 높더라도 불확실성이 큰 후보보다, 평균은 약간 낮지만 증거가 단단한 후보를 선택할 수 있다.

### 3.16 비관적 선택의 regret 직관

균등 사건에서 모든 $\pi$에 대해

$$
\left|
\widehat V(\pi)-V(\pi)
\right|
\leq
\beta(\pi)
$$

라고 하자.

최적 정책을

$$
\pi^\star
=
\arg\max_{\pi\in\Pi}V(\pi)
$$

라 하자. 비관적 선택은

$$
\widehat\pi
=
\arg\max_{\pi\in\Pi}
\left[
\widehat V(\pi)-\beta(\pi)
\right]
$$

이고, 균등 사건 덕분에 선택 후 보장이 남는다.

$$
\begin{aligned}
V(\pi^\star)-V(\widehat\pi)
&\leq
\left[
\widehat V(\pi^\star)-\beta(\pi^\star)
\right]
+
2\beta(\pi^\star)
-
V(\widehat\pi)
\\
&\leq
2\beta(\pi^\star).
\end{aligned}
$$

첫 부등식은 $V(\pi^\star)\leq\widehat V(\pi^\star)+\beta(\pi^\star)$를 썼고, 둘째 부등식은 비관적 목적의 최적성과

$$
\widehat V(\widehat\pi)-\beta(\widehat\pi)
\leq
V(\widehat\pi)
$$

를 썼다. 최적 정책 자체가 충분히 잘 관측되어 $\beta(\pi^\star)$가 작으면, 비관적 선택의 regret도 작아진다.

### 3.17 안전 제약을 함께 넣는다

품질 최소 개선폭을 $m$이라 하고, cost $j$의 허용 증가량을 $d_j$라 하자.

후보군 전체와 모든 cost에 동시에 유효한 경계를 만들고 다음 feasible set을 정의한다.

$$
\widehat{\mathcal F}
=
\left\{
\pi\in\Pi:
\operatorname{LCB}_{\Delta R}(\pi)\geq m,
\quad
\operatorname{UCB}_{\Delta C_j}(\pi)\leq d_j
\ \forall j
\right\}.
$$

그 안에서 가장 보수적인 품질이 좋은 정책을 고른다.

$$
\widehat\pi
=
\arg\max_{\pi\in\widehat{\mathcal F}}
\operatorname{LCB}_{\Delta R}(\pi).
$$

단, $\widehat{\mathcal F}=\varnothing$이면 억지로 후보 하나를 고르지 않는다. 기준 정책 $\pi_0$을 유지하거나 사전에 정한 안전 fallback으로 돌아가고, 더 많은 탐색 로그를 모은 뒤 다시 평가한다.

후보 $K$개, 품질 지표 하나, cost $J$개를 모두 검사한다면 가장 단순한 Bonferroni 배분은 각 한쪽 경계에

$$
\alpha_{\mathrm{cell}}
=
\frac{\delta}{K(J+1)}
$$

를 주는 것이다.

보수적이지만 의존 구조를 몰라도 가족 전체 실패확률을 $\delta$ 이하로 제어한다.

기준 정책 자체가 안전하지 않다면 “cost가 더 늘지 않는다”는 증가량 제약만으로는 충분하지 않다. 이때는 절대 수준에도

$$
\operatorname{UCB}_{C_j}(\pi)\leq c_j
$$

같은 한도를 둔다. 평균 cost UCB는 희귀 치명 사건, 금지 행동, 특정 slice의 최악 위험을 자동으로 막지 않으므로 hard execution block, tail·chance constraint, slice별 gate를 병행한다.

이때 절대 cost, slice, tail 지표까지 통계적 gate로 추가하면 검사 셀 수가 늘어난다. 따라서 $\delta/[K(J+1)]$을 그대로 재사용하지 말고, 새 경계들까지 포함해 가족 전체 error budget을 다시 배분해야 한다.

### 3.18 라그랑지안으로 미분 가능한 학습 문제를 만든다

정책을 연속 파라미터 $\theta$로 표현하면 모든 후보를 열거하기 어렵다.

보수적 reward 목적과 cost 제약을 라그랑지안으로 결합할 수 있다.

$$
\mathcal L(\theta,\lambda)
=
\operatorname{LCB}_{\Delta R}(\pi_\theta)
-
\sum_{j=1}^{J}
\lambda_j
\left[
\operatorname{UCB}_{\Delta C_j}(\pi_\theta)-d_j
\right],
\qquad
\lambda_j\geq 0.
$$

정책 파라미터는 $\mathcal L$을 키우고, multiplier는 위반된 제약의 가격을 높인다.

$$
\theta
\leftarrow
\theta
+
\eta_\theta
\nabla_\theta\mathcal L
$$

$$
\lambda_j
\leftarrow
\left[
\lambda_j
+
\eta_\lambda
\left(
\operatorname{UCB}_{\Delta C_j}(\pi_\theta)-d_j
\right)
\right]_+.
$$

여기서

$$
[z]_+=\max(z,0)
$$

이다.

단, 최적화 중 매 step마다 같은 데이터의 신뢰반경을 다시 이용하면 최종 정책은 연속적인 데이터 의존 함수가 된다. 유한 후보 union bound를 그대로 붙일 수 없으며, 함수군 복잡도 경계나 독립 selection/final-gate 데이터가 필요하다.

### 3.19 무한 정책군에서는 $K$ 대신 복잡도가 들어간다

신경망 정책처럼 가능한 정책이 사실상 무한하면

$$
\log K
$$

를 쓸 수 없다.

대신 다음과 같은 복잡도 개념이 필요하다.

- VC dimension
- covering number
- Rademacher complexity
- PAC-Bayes complexity
- 안정성 기반 일반화 경계

직관은 같다.

$$
\text{일반화 오차}
\approx
\sqrt{
\frac{
\text{정책군의 유효 복잡도}
+
\log(1/\delta)
}{n}
}.
$$

실무에서는 이론적 복잡도 경계를 직접 계산하기보다

1. 학습 데이터에서 연속 최적화,
2. selection 데이터에서 유한한 checkpoint·seed·하이퍼파라미터 후보 비교,
3. final-gate 데이터에서 잠근 후보 하나 검증

으로 문제를 다시 유한하고 감사 가능한 형태로 바꾸는 경우가 많다.

### 3.20 Softmax 정책의 그래디언트

행동 점수를 $s_\theta(x,a)$라 하고 온도 $\tau>0$인 Softmax 정책을 쓰자.

$$
\pi_\theta(a\mid x)
=
\frac{
\exp(s_\theta(x,a)/\tau)
}{
\sum_{a'}
\exp(s_\theta(x,a')/\tau)
}.
$$

로그 정책의 그래디언트는

$$
\nabla_\theta
\log\pi_\theta(a\mid x)
=
\frac{1}{\tau}
\left[
\nabla_\theta s_\theta(x,a)
-
\sum_{a'}
\pi_\theta(a'\mid x)
\nabla_\theta s_\theta(x,a')
\right].
$$

IPS 목적함수의 그래디언트는

$$
\nabla_\theta
\widehat V_{\mathrm{IPS}}(\pi_\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{
\pi_\theta(A_i\mid X_i)
}{
b_i(A_i\mid X_i)
}
R_i
\nabla_\theta
\log\pi_\theta(A_i\mid X_i).
$$

큰 중요도 가중치는 값뿐 아니라 그래디언트도 폭발시킨다. overlap과 비관적 벌점은 통계적 안전장치이면서 최적화 안정장치다.

### 3.21 상태·행동별 비관적 가치

정책 전체의 LCB 대신 행동별 outcome 추정에 불확실성 벌점을 줄 수도 있다.

$$
\widetilde q_R(x,a)
=
\widehat q_R(x,a)-u_R(x,a)
$$

$$
\widetilde q_{C_j}(x,a)
=
\widehat q_{C_j}(x,a)+u_{C_j}(x,a).
$$

- reward는 불확실할수록 낮게 본다.
- cost는 불확실할수록 높게 본다.

정책은

$$
\max_\pi
\mathbb E_X
\left[
\sum_a
\pi(a\mid X)
\widetilde q_R(X,a)
\right]
$$

를 풀되

$$
\mathbb E_X
\left[
\sum_a
\pi(a\mid X)
\widetilde q_{C_j}(X,a)
\right]
\leq d_j
$$

를 지킨다.

하지만 $u(x,a)$가 단순한 신경망 ensemble disagreement라면 자동으로 유효한 신뢰반경이 되는 것은 아니다. calibration과 coverage 실험이 필요하다.

### 3.22 기준 정책 혼합은 변화 폭을 줄인다

후보 정책 $\widetilde\pi$와 기준 정책 $\pi_0$를

$$
\pi_\lambda
=
(1-\lambda)\pi_0
+
\lambda\widetilde\pi,
\qquad
0\leq\lambda\leq 1
$$

로 섞자.

정책 가치는 선형이다.

$$
V(\pi_\lambda)
=
(1-\lambda)V(\pi_0)
+
\lambda V(\widetilde\pi).
$$

behavior가 기준 정책과 같다면 중요도 비율은

$$
\frac{\pi_\lambda(a\mid x)}{\pi_0(a\mid x)}
=
1-\lambda
+
\lambda
\frac{\widetilde\pi(a\mid x)}{\pi_0(a\mid x)}
$$

이다.

작은 $\lambda$는 극단적 비율을 완화하고 canary의 변화 폭을 줄인다. 그러나

$$
\pi_0(a\mid x)=0
$$

인 행동에 support를 새로 만들어 주지는 못한다.

### 3.23 ESS는 유용하지만 신뢰구간이 아니다

가중치 $w_i$의 유효표본크기는

$$
\operatorname{ESS}
=
\frac{
\left(\sum_iw_i\right)^2
}{
\sum_iw_i^2
}
$$

이다.

모든 가중치가 같으면 ESS는 $n$이고, 몇 개의 가중치가 대부분을 차지하면 ESS는 작아진다.

하지만 다음은 성립하지 않는다.

$$
\text{ESS가 크다}
\Longrightarrow
\text{추정이 무편향이고 경계가 유효하다}.
$$

propensity가 잘못 기록됐거나 outcome이 지연 누락됐거나 사용자 군집을 무시하면 ESS가 커도 추론은 틀릴 수 있다.

ESS는 overlap 진단이고, 신뢰경계의 가정 검증을 대신하지 않는다.

### 3.24 clipping은 분산과 편향을 교환한다

가중치를

$$
\widetilde w_i
=
\min(w_i,c)
$$

로 자르면 극단값 영향이 줄어든다.

하지만

$$
\mathbb E[\widetilde w_iR_i]
\neq
V(\pi)
$$

일 수 있다.

따라서 clipping을 사용하면 최소한 다음을 함께 보고해야 한다.

- clip threshold $c$
- clip된 표본 비율
- clip 전후 최대 가중치
- clip 전후 ESS
- 여러 $c$에 대한 민감도
- 편향을 고려한 보수적 경계 또는 별도 검증

“분산이 작아졌으니 더 정확하다”는 결론은 성립하지 않는다.

### 3.25 적응 로그에서는 martingale 구조를 보존한다

어제의 contextual bandit이 시간에 따라 behavior policy를 바꿨다면 $b_i$는 고정 분포가 아니다.

행동 직전 정보장 $\mathcal G_i=\sigma(\mathcal H_{i-1},X_i)$에 대해

$$
w_i(\pi)
=
\frac{
\pi(A_i\mid X_i,\mathcal H_{i-1})
}{
b_i(A_i\mid X_i,\mathcal H_{i-1})
}
$$

라 하자. 3.1절의 consistency·sequential randomization, 정확한 propensity, overlap 아래

$$
Z_i(\pi)
=
w_i(\pi)R_i-v_{R,i}(\pi)
$$

는

$$
\mathbb E
\left[
Z_i(\pi)
\mid
\mathcal G_i
\right]
=
0
$$

을 만족한다. 따라서

$$
M_t
=
\sum_{i=1}^{t}
\left[
w_i(\pi)R_i-v_{R,i}(\pi)
\right]
$$

는 적절한 filtration에 대한 martingale이다. 이 식은 단순히 임의의 변수를 자기 조건부기댓값으로 중심화한 것이 아니라, 정확한 propensity가 target policy의 시점별 가치 $v_{R,i}(\pi)$를 복원한다는 식별 관계를 명시한다.

따라서 martingale 경계가 직접 제어하는 것은

$$
\frac{1}{n}
\sum_{i=1}^{n}
w_i(\pi)R_i
-
\frac{1}{n}
\sum_{i=1}^{n}
v_{R,i}(\pi)
$$

이다. 3.1절의 deterministic 평균

$$
V_{R,n}(\pi)
=
\frac{1}{n}
\sum_i
\mathbb E[v_{R,i}(\pi)]
$$

까지 제어하려면

$$
\frac{1}{n}
\sum_i
\left(
v_{R,i}(\pi)
-
\mathbb E[v_{R,i}(\pi)]
\right)
$$

에 대한 별도 concentration·ergodicity 조건이 필요하다. 또는 realized predictable average를 조건부 estimand로 명시해야 한다.

이 경우 독립표본 Hoeffding 공식을 무비판적으로 쓰지 말고 다음 중 설계에 맞는 도구를 선택한다.

- bounded increment와 predictable quadratic variation 조건을 확인한 martingale Bernstein/Freedman 경계
- 약의존·stationary block 조건이 정당화될 때의 cluster 또는 block bootstrap
- 사전 고정된 평가 블록
- anytime-valid confidence sequence

오늘의 유한 정책군 union-bound 논리는 그대로 적용할 수 있지만, 각 정책의 기초 경계가 로그 생성 과정에 맞아야 한다.

또한 미래 시점 데이터를 사용해 만든 일반적인 random-fold nuisance model은 행동 직전 정보 $\mathcal G_i$에 measurable하지 않을 수 있다. 적응 로그에서는 frozen pre-period model, 과거만 쓰는 prequential fit, 또는 이론적으로 정당화된 time-block cross-fitting을 사용해야 한다.

### 3.26 학습·선택·최종 게이트의 확률적 역할

데이터를

$$
\mathcal D
=
\mathcal D_{\mathrm{train}}
\cup
\mathcal D_{\mathrm{select}}
\cup
\mathcal D_{\mathrm{gate}}
$$

로 나누자.

#### 학습 단계

$$
\Pi_{\mathrm{cand}}
=
\mathcal A(\mathcal D_{\mathrm{train}})
$$

알고리즘 $\mathcal A$가 후보 정책들을 만든다.

$\mathcal D_{\mathrm{train}}$에 조건부로 보면 이 후보군은 독립 selection block을 열기 전에 고정돼 있으므로, 유한 후보 union bound를 적용할 수 있다.

#### 선택 단계

$$
\widehat\pi
=
\mathcal S(
\Pi_{\mathrm{cand}},
\mathcal D_{\mathrm{select}}
)
$$

선택 규칙 $\mathcal S$가 균등 LCB·UCB와 운영 조건으로 후보 하나를 고른다.

#### 최종 게이트

$\mathcal D_{\mathrm{gate}}$는 $\widehat\pi$를 만드는 데 사용되지 않았으므로, **정책 차원에서는** 잠근 후보 하나의 pointwise 경계를 계산할 수 있다.

$$
\operatorname{LCB}_{\mathrm{gate}}(\widehat\pi)
$$

그러나 품질 하나와 cost $J$개, 여러 slice를 동시에 gate한다면 metric·slice 차원의 오류 예산 배분은 여전히 필요하다. 또한 gate를 본 뒤 threshold·reward·후보를 바꾸면 독립성이 깨진다.

### 3.27 무작위 분할이 언제 위험할까?

같은 사용자의 인접 요청을 train과 gate에 무작위로 흩뿌리면 사용자 습관이 누수될 수 있다.

시간 drift가 목표라면 과거로 학습하고 미래 블록으로 검증해야 한다.

따라서 분할 단위는 estimand에 맞춘다.

| 목표 | 권장 분할 |
| --- | --- |
| 새 사용자 일반화 | 사용자 단위 group split |
| 미래 배포 일반화 | 시간 순서 split |
| 세션 내 반복 의존 | 세션 단위 split |
| 지역·고객사 이전 | 지역·고객사 holdout |
| 후보 선택 후 고정 검증 | 완전히 잠근 final-gate block |

### 3.28 작은 숫자 예제

기준 정책 대비 후보 세 개의 품질 개선량과 안전 cost 증가량을 평가했다고 하자.

후보 $K=3$, cost $J=1$, 가족 실패확률 $\delta=0.05$를 Bonferroni로 나누면 각 한쪽 경계의 유의수준은

$$
\alpha_{\mathrm{cell}}
=
\frac{0.05}{3(1+1)}
\approx
0.00833
$$

이고 정규 임계값은 대략

$$
z_{1-\alpha_{\mathrm{cell}}}
\approx
2.40
$$

이다.

이는 후보군이 평가 전에 고정되고, 각 후보의 cluster·martingale 구조를 반영한 유효한 표준오차가 있으며, 정규근사가 적절하다고 보는 **한쪽 Bonferroni–Wald 교육용 예제**다. 작은 표본에서의 정확한 finite-sample 보장을 뜻하지 않는다.

요구 조건은

$$
\operatorname{LCB}_{\Delta R}>0
$$

과

$$
\operatorname{UCB}_{\Delta C}\leq 0.002
$$

라고 하자.

| 후보 | $\widehat\Delta_R$ | $\operatorname{SE}_R$ | 품질 LCB | $\widehat\Delta_C$ | $\operatorname{SE}_C$ | cost UCB | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | $0.024$ | $0.006$ | $0.0096$ | $0.0010$ | $0.0008$ | $0.00292$ | cost 실패 |
| B | $0.018$ | $0.0045$ | $0.0072$ | $0.0002$ | $0.0006$ | $0.00164$ | 통과 |
| C | $0.030$ | $0.014$ | $-0.0036$ | $-0.0004$ | $0.0005$ | $0.00080$ | 품질 실패 |

계산은 다음과 같다.

$$
\operatorname{LCB}_{\Delta R,B}
=
0.018-2.40(0.0045)
=
0.0072
$$

$$
\operatorname{UCB}_{\Delta C,B}
=
0.0002+2.40(0.0006)
=
0.00164.
$$

단순 평균 개선이 가장 큰 C가 아니라, 품질 하한과 cost 상한을 모두 통과한 B가 선택된다.

### 3.29 오늘 수학과 AI 운영의 1:1 매핑

| 수학 부품 | AI 시스템 부품 | 없으면 생기는 문제 |
| --- | --- | --- |
| $b_i(a\mid x)$ | 실제 router·sampler의 최종 선택확률 | IPS·DR을 재현할 수 없다. |
| overlap | 각 행동의 exploration floor | 로그 밖 정책을 평가하게 된다. |
| paired DR score | 기준·후보의 공통 요청 비교 | 분산이 불필요하게 커진다. |
| $\log K$ 또는 정책군 복잡도 | 시도한 후보·seed·checkpoint 수 | 숨은 실험 횟수 때문에 승자 편향이 생긴다. |
| uniform LCB | 후보 선택 이후에도 유효한 품질 하한 | 선택된 승자의 구간이 낙관적이 된다. |
| cost UCB | 안전·비용의 보수적 최악값 | 평균 품질만 보고 위험한 후보를 배포한다. |
| pessimistic objective | 불확실한 정책의 점수 감점 | low-support 행동으로 정책이 뛰어간다. |
| sample splitting | train·selection·gate 로그 분리 | 평가 데이터까지 학습해 버린다. |
| cluster/block variance | 사용자·세션·날짜 의존성 | 표준오차를 과소평가한다. |
| mixture $\pi_\lambda$ | 기준 정책 fallback·단계적 rollout | 한 번에 큰 분포 이동이 생긴다. |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Contextual Bandit을 다시 가장 작은 말로 설명한다

Contextual bandit은 매 시점 다음 세 단계만 가진다.

1. 상황을 본다.
2. 행동 하나를 고른다.
3. 그 행동의 결과만 본다.

예를 들어 AI 안전 감사 시스템에서

- context: 언어, 요청 유형, 모델 버전, 위험 점수
- action: 어떤 queue에 사람 감사를 배정할지
- reward: 중요한 실패를 찾아낸 정보가치
- cost: 감사 비용, 지연, 특정 집단의 과소감사

가 된다.

선택하지 않은 행동의 결과는 보이지 않는다는 점이 지도학습과 다르다.

### 4.2 평가와 학습을 분리한다

오프정책 평가는 묻는다.

> 이 정책 $\pi$가 주어졌을 때 가치는 얼마인가?

오프정책 학습은 묻는다.

> 로그에서 어떤 정책 $\pi$를 골라야 하는가?

학습은 평가기를 내부에서 수천 번 호출할 수 있다. 따라서 평가기의 작은 낙관 오차가 정책에 의해 적극적으로 악용된다.

이 현상을 reward hacking의 통계 버전으로 볼 수 있다.

### 4.3 전체 시스템 구조

안전한 오프정책 학습 파이프라인은 다음처럼 나눈다.

    원시 이벤트 로그
        ↓
    propensity·policy version·결과 성숙도 검증
        ↓
    사용자·시간 기준 train / selection / gate 분리
        ↓
    train에서 outcome model과 후보 정책 학습
        ↓
    selection에서 uniform LCB·cost UCB·ESS·slice gate
        ↓
    정책·threshold·reward 정의 잠금
        ↓
    독립 gate에서 고정 후보 재평가
        ↓
    shadow → 작은 무작위 canary → 단계 확대
        ↓
    drift·pending·위험 e-process 감시와 rollback

각 화살표 사이에 버전과 시간 범위를 남겨야 결과를 재현할 수 있다.

### 4.4 로그 계약이 학습 알고리즘보다 먼저다

최소한 다음 필드를 원자적으로 기록한다.

| 필드 | 이유 |
| --- | --- |
| event ID | 중복·누락을 찾는다. |
| user/session cluster ID | 의존성을 반영한 분산을 계산한다. |
| event time | 행동을 고른 시각을 안다. |
| reveal time | 결과가 성숙한 시각을 안다. |
| context snapshot | 당시 실제로 본 입력만 재현한다. |
| eligible action set | 선택 가능한 행동 집합을 복원한다. |
| sampled action | 실제 실행 행동이다. |
| final propensity | 모든 mask·fallback 뒤 실제 확률이다. |
| policy version | 어느 코드와 파라미터가 선택했는지 안다. |
| RNG seed 또는 draw ID | sampler 재현과 감사에 유용하다. |
| reward components | 나중에 보상 정의를 감사한다. |
| cost components | 품질에 숨겨진 안전 악화를 찾는다. |
| label status | pending을 $0$으로 오인하지 않는다. |
| model/tool versions | 환경 변화와 정책 효과를 구분한다. |

### 4.5 reward를 한 숫자로 너무 일찍 합치지 않는다

다음 식처럼 모든 것을 하나로 합치고 싶을 수 있다.

$$
U
=
R
-
\lambda_1C_1
-
\lambda_2C_2.
$$

하지만 하나의 scalar reward만 저장하면 나중에

- 품질이 올랐는지,
- 유해율이 늘었는지,
- 지연이 커졌는지,
- 특정 집단이 희생됐는지

구분하기 어렵다.

원시 component를 따로 저장하고, 배포 gate에서는 안전 cost를 명시적 제약으로 다루는 편이 감사 가능하다.

### 4.6 후보 정책을 어떻게 표현할까?

초보자에게 가장 관리하기 쉬운 후보는 다음 세 종류다.

#### 규칙 기반 정책

예:

$$
\pi(a=\text{human audit}\mid x)
=
\begin{cases}
p_{\mathrm{high}}, & s(x)\geq t,\\
p_{\mathrm{low}}, & s(x)<t.
\end{cases}
$$

threshold 후보 수가 유한해 균등 보정이 쉽다.

#### Softmax 선형 정책

$$
s_\theta(x,a)=\theta_a^\top\phi(x)
$$

$$
\pi_\theta(a\mid x)
=
\operatorname{softmax}_a
\left(
\frac{s_\theta(x,a)}{\tau}
\right).
$$

해석 가능하고 exploration floor를 넣기 쉽다.

#### 기준 정책 혼합

$$
\pi_{\theta,\lambda}
=
(1-\lambda)\pi_0+\lambda\pi_\theta.
$$

$\lambda$를 제한하면 변화 폭과 중요도 비율을 함께 제어할 수 있다.

### 4.7 exploration floor를 정책 정의에 넣는다

context $x$에서 실제로 허용된 행동 집합을 $\mathcal A(x)$라 하고, 그 집합 안에서만 양의 확률을 갖는 탐색 분포를 $\nu(a\mid x)$라 하자. 원시 Softmax 정책은 eligibility mask를 적용한 뒤 $\mathcal A(x)$ 위에서 다시 정규화해 $\widetilde\pi_\theta(a\mid x)=0$ for $a\notin\mathcal A(x)$가 되게 한다. 이 정책과 $\nu$를 섞으면

$$
\pi_{\theta,\gamma}(a\mid x)
=
(1-\gamma)
\widetilde\pi_\theta(a\mid x)
+
\gamma\nu(a\mid x),
\qquad
a\in\mathcal A(x)
$$

가 된다.

균등 탐색 분포

$$
\nu(a\mid x)
=
\frac{1}{|\mathcal A(x)|}
$$

를 쓰면 eligible action의 floor는

$$
\frac{\gamma}{|\mathcal A(x)|}
$$

이다. 이 floor는 새 정책을 배포해 미래 behavior policy로 사용할 때 탐색을 보존한다.

현재 OPE의 overlap은 과거 로그의 $b_i$가 보장해야 한다. target policy에 floor를 넣는 것만으로 과거의

$$
b_i(a\mid x,\mathcal H_{i-1})=0
$$

을 복구할 수 없고, 오히려 target support를 넓혀 중요도 비율을 키울 수도 있다. 현재 로그에서 가중치를 직접 제한하려면

$$
\pi(a\mid x)
\leq
W
b_i(a\mid x,\mathcal H_{i-1})
$$

같은 density-ratio 제약이 필요하다.

실제 배포에서는 eligibility mask와 fallback이 적용된 뒤의 최종 확률이 설계한 floor와 ratio 제한을 만족하는지 확인한다.

### 4.8 outcome model은 정책과 별도로 검증한다

보상 모델이 잘 맞는지 전체 RMSE 하나만 보면 부족하다.

다음을 action과 주요 slice별로 본다.

- calibration
- residual 평균
- tail error
- coverage가 낮은 구간의 오차
- 시간 drift
- label delay별 오차
- 모델 버전 전환점의 오차

outcome model이 틀려도 정확한 propensity가 DR의 평균을 보정할 수 있지만, 심하게 틀린 모델은 분산과 정책 그래디언트를 악화시킨다.

### 4.9 cross-fitting의 구현 순서

$K_{\mathrm{fold}}$개의 fold를 만든다고 하자.

각 fold $k$에 대해

1. 나머지 fold에서 outcome model을 학습한다.
2. fold $k$의 모든 행동별 예측을 저장한다.
3. fold $k$에서 DR score를 계산한다.
4. 모든 fold의 out-of-fold score를 합친다.

동일 사용자의 요청이 여러 fold에 섞이지 않도록 group fold를 사용해야 한다.

이 random group-fold 절차는 stationary 평가를 위한 기본형이다. 적응 로그에서는 미래 시점으로 학습한 nuisance model이 과거 score에 들어가지 않도록 frozen pre-period, prequential, 또는 정당화된 time-block cross-fitting을 사용한다.

cross-fitting은 train 데이터 재사용을 효율화하지만, final-gate 독립성을 대신하지 않는다.

### 4.10 비관적 학습의 실용 목적함수

한 가지 실용적인 형태는

$$
\widehat J(\theta)
=
\widehat V_{\mathrm{DR}}(\pi_\theta)
-
\beta_{\mathrm{global}}(\pi_\theta)
-
\lambda_{\mathrm{KL}}
\mathbb E_X
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid X)
\|
b_{\mathrm{ref}}(\cdot\mid X)
\right)
\right]
$$

이다.

- 첫 항: 추정 보상
- 둘째 항: 통계적 불확실성 벌점
- 셋째 항: 로그 정책에서 너무 멀리 가지 않게 하는 trust penalty

KL 벌점은 overlap을 돕지만 support가 없는 행동을 식별 가능하게 만들지는 않는다.

### 4.11 후보 수는 숨기지 않는다

보고서에는 최종 후보 하나만이 아니라 다음을 남긴다.

- 생성한 정책 수
- 학습 seed 수
- checkpoint 수
- threshold grid 크기
- reward 정의 변경 횟수
- 중간 결과를 본 사람과 시각
- 제외한 후보의 사유
- 선택 규칙의 버전

사람이 수동으로 대시보드를 보고 후보를 버린 것도 selection 과정이다.

숨은 후보 수는 숨은 다중검정이다.

### 4.12 selection 데이터에서 해야 할 일

후보마다 다음 표를 만든다.

| 진단 | 질문 |
| --- | --- |
| 품질 uniform LCB | 선택 이후에도 최소 개선폭을 넘는가? |
| 각 cost uniform UCB | 안전·비용 한도를 넘을 가능성이 남는가? |
| ESS | 일부 요청 몇 개가 결과를 지배하는가? |
| 최대 가중치 | overlap이 사실상 깨진 slice가 있는가? |
| clip 민감도 | 결론이 threshold 하나에 의존하는가? |
| slice LCB/UCB | 평균 개선 뒤에 특정 집단 악화가 숨는가? |
| pending 비율 | 미성숙 결과를 음성으로 취급했는가? |
| drift 비교 | 오래된 로그와 최근 로그의 결론이 같은가? |
| placebo test | 결과 전 feature만 사용했다는 증거가 있는가? |

### 4.13 final gate에서 정책을 잠근다는 뜻

다음을 해시 가능한 artifact로 고정한다.

- 모델 가중치 또는 규칙 파일
- feature schema
- action eligibility
- temperature와 exploration floor
- fallback 순서
- reward·cost 정의
- propensity 계산 코드
- gate 표본의 시작·종료 규칙
- 통계 검정과 실패확률 배분
- 최소 표본·최소 기간

gate 결과를 본 뒤 하나라도 바꾸면 새 정책이다.

### 4.14 final gate가 통과해도 바로 전체 배포하지 않는다

오프정책 로그는 과거 환경에서 만들어졌다. 새 정책은 다음을 바꿀 수 있다.

- 사용자의 후속 행동
- queue 혼잡
- 사람 감사자의 피로
- 도구 latency
- downstream 모델 입력분포
- 공격자의 적응

따라서 순서는 보통 다음과 같다.

1. shadow
2. 작은 무작위 canary
3. 사전 고정된 최소 기간 유지
4. time-uniform 품질 LCB·위험 UCB 확인
5. 단계적 확대
6. rollback 기준 상시 감시

### 4.15 데이터가 부족하면 abstain한다

정책은 반드시 모든 context에서 새 행동을 강하게 선택할 필요가 없다.

지원이 약한 영역 $\mathcal U$에서는

$$
\pi(a\mid x)
=
\pi_0(a\mid x),
\qquad
x\in\mathcal U
$$

로 fallback할 수 있다.

예를 들어

$$
\mathcal U
=
\left\{
x:
\min_{a:\pi(a\mid x)>\eta}
\widehat n_{\mathrm{eff}}(x,a)
<
n_{\min}
\right\}
$$

처럼 유효 표본이 부족한 context를 정한다.

“모르겠다”를 정책의 합법적 행동으로 만드는 것이 비관적 학습의 중요한 실무 해석이다.

### 4.16 drift가 있으면 오래된 큰 표본이 항상 낫지 않다

오래된 로그를 많이 넣으면 표준오차는 작아질 수 있다. 그러나 정책 가치 자체가 바뀌었다면

$$
V_t(\pi)\neq V_{t-h}(\pi)
$$

이다.

오늘 배포 가치가 estimand라면 오래된 로그의 낮은 분산보다 최근 분포의 관련성이 중요할 수 있다.

다음을 함께 보고한다.

- 전체 기간 추정
- 최근 고정 window 추정
- 시간 block별 추정
- context mix 표준화 결과
- 후보 순위 안정성

### 4.17 초보자가 흔히 하는 오해

| 오해 | 왜 틀렸는가 |
| --- | --- |
| “후보마다 95% LCB를 계산했으니 가장 큰 것을 고르면 된다.” | pointwise 경계는 선택된 승자를 보호하지 않는다. |
| “DR이면 모델과 propensity가 둘 다 틀려도 안전하다.” | 둘 중 하나가 맞아야 한다는 구조이며, 지연 누락·support 부재까지 해결하지 않는다. |
| “ESS가 1,000이면 표본 1,000개짜리 정규구간을 쓰면 된다.” | ESS는 가중치 집중 진단이지 완전한 표본분포 정리가 아니다. |
| “clipping은 큰 가중치를 없애니 항상 더 정확하다.” | 분산 대신 편향을 산다. |
| “KL 제약이 있으면 overlap이 보장된다.” | 기준 정책이 확률 $0$인 행동에는 KL도 정보를 만들지 못한다. |
| “validation에서 고른 뒤 같은 validation의 LCB를 보고하면 된다.” | 선택 데이터의 우연한 승리를 다시 보고하는 셈이다. |
| “final gate를 본 뒤 threshold만 조금 바꾸는 건 괜찮다.” | 데이터에 맞춘 새 정책이므로 새 gate가 필요하다. |
| “평균 cost가 통과하면 모든 집단이 안전하다.” | 희소 집단 악화가 평균에 숨을 수 있다. |
| “오프라인 LCB가 양수면 전체 배포해도 된다.” | 환경 반응과 시스템 오류는 온라인 canary에서만 드러날 수 있다. |
| “후보 수는 저장된 checkpoint 수다.” | 수동 실험, seed, reward 변경, 제외 규칙도 선택 복잡도다. |

### 4.18 최소 단위 테스트

1. propensity 합이 action set에서 $1$인지 검사한다.
2. eligibility mask 뒤 선택된 행동의 propensity가 양수인지 검사한다.
3. 같은 policy version으로 로그를 replay했을 때 확률이 재현되는지 확인한다.
4. 모든 action이 같은 reward를 가질 때 IPS·DR 정책 가치가 같아지는지 확인한다.
5. $\pi=b$일 때 중요도 가중치가 $1$인지 확인한다.
6. outcome model이 참값일 때 DR residual 평균이 $0$에 가까운지 확인한다.
7. 후보 순서를 바꿔도 uniform correction 결과가 같은지 확인한다.
8. gate 데이터를 후보 생성 코드가 읽지 못하도록 권한을 분리한다.
9. pending label이 reward $0$으로 변환되지 않는지 확인한다.
10. 한 사용자의 이벤트가 split 경계를 넘지 않는지 확인한다.

### 4.19 운영 체크리스트

- [ ] estimand가 현재 트래픽 평균인지 특정 집단 평균인지 고정했는가?
- [ ] reward와 각 cost를 별도 component로 저장했는가?
- [ ] 실제 최종 propensity를 로그에 남겼는가?
- [ ] candidate policy의 support가 behavior support 안에 있는가?
- [ ] cross-fitting이 사용자·세션 단위를 존중하는가?
- [ ] 시도한 모든 후보·seed·checkpoint 수를 기록했는가?
- [ ] pointwise가 아니라 selection-valid 경계를 사용했는가?
- [ ] 품질은 LCB, 위험은 UCB로 판단했는가?
- [ ] 가족 전체 실패확률을 후보와 지표에 배분했는가?
- [ ] clipping·ESS·최대 가중치 민감도를 보고했는가?
- [ ] train·selection·final-gate가 실제로 분리됐는가?
- [ ] gate 이후 정책과 규칙을 잠갔는가?
- [ ] low-support context의 baseline fallback이 있는가?
- [ ] shadow와 무작위 canary를 거치는가?
- [ ] drift·label delay·rollback을 계속 감시하는가?

### 4.20 오늘 원리가 신경망 학습과 만나는 지점

일반 지도학습에서도 training loss를 가장 작게 만드는 모델을 선택한다.

$$
\widehat f
=
\arg\min_{f\in\mathcal F}
\widehat L(f)
$$

모델군이 복잡할수록 training loss와 true risk의 간격이 커질 수 있어 정규화·validation·일반화 경계가 필요하다.

오프정책 학습도 같은 구조지만 더 어렵다.

$$
\widehat\pi
=
\arg\max_{\pi\in\Pi}
\widehat V_{\mathrm{OPE}}(\pi)
$$

여기서는 선택되지 않은 행동의 label이 없고, propensity 비율이 잡음을 확대한다.

즉 오늘의 pessimism은 단순한 겁이 아니라

$$
\text{오프정책 추정 오차를 반영한 정규화}
$$

라고 이해할 수 있다.

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **정보 기준 시각:** 2026-10-16 02:35 KST. 발표일·수치·라이선스는 공식 모델 카드, 기술보고서, 논문 원문을 기준으로 확인했다. 아래 성능 수치는 별도 독립 재현이 아니라 각 연구팀의 보고다.

오늘의 두 트렌드는 서로 다른 층에서 같은 교훈을 준다.

- Apertus 1.5는 텍스트·이미지·오디오를 하나의 공개된 token pipeline으로 합친다.
- Weighted Ensemble은 희귀한 LLM 실패를 더 자주 찾으면서도 원래 모델 아래의 확률을 가중치로 복원한다.

둘 다 “많이 실행했다”보다 **어떤 경로로 데이터를 만들었고 그 선택을 어떻게 보정하는가**가 중요하다.

### 5.1 먼저 알아둘 트렌드 용어

| 용어 | 뜻 | 주의점 |
| --- | --- | --- |
| fully open | 가중치뿐 아니라 데이터·레시피·평가·코드까지 공개 범위를 넓힌 접근 | 각 artifact의 실제 라이선스와 접근 조건은 따로 확인한다. |
| open-weight | 가중치는 내려받을 수 있지만 데이터·학습 과정이 모두 공개됐다는 뜻은 아님 | fully open과 구분한다. |
| multimodal | 텍스트 외 이미지·오디오 같은 여러 양식을 처리 | 입력 지원과 출력 생성 지원은 다르다. |
| discrete tokenizer | 연속 이미지·오디오를 정수 token ID 열로 바꾸는 장치 | 압축 과정에서 정보가 손실될 수 있다. |
| early fusion | 서로 다른 modality token을 초기에 같은 sequence로 섞는 방식 | 별도 encoder 방식과 비용 구조가 다르다. |
| context window | 한 번에 처리할 수 있는 입력·출력 token 범위 | nominal 길이와 실제 품질은 다르다. |
| thinking mode | 답 전에 별도 reasoning token 구간을 생성하는 모드 | 더 긴 출력과 지연을 쓸 수 있다. |
| rare event | 확률은 매우 낮지만 피해가 큰 사건 | 직접 Monte Carlo로는 관측 자체가 어렵다. |
| Monte Carlo | 같은 확률과정을 여러 번 실행해 빈도로 기대값·확률을 추정 | 사건 확률이 $10^{-5}$이면 막대한 표본이 필요하다. |
| prefix | 아직 끝나지 않은 부분 생성문 | 최종 실패 가능성에 대한 중간 신호가 될 수 있다. |
| walker | Weighted Ensemble이 유지하는 가중 prefix 하나 | 복제·병합돼도 확률 질량을 보존한다. |
| reaction coordinate | prefix가 희귀사건으로 향하는 정도를 나타내는 feature | 나쁘면 편향보다 주로 분산이 커진다. |
| resampling | 일부 prefix를 복제하고 일부를 줄여 계산 예산을 재배분 | 각 child의 가중치를 바르게 나눠야 한다. |
| hit rate | 생성 표본 중 rare event에 도달한 비율 | 원래 rare-event 확률과 같은 값이 아니다. |
| unbiasedness | 반복 실행한 추정량 평균이 참값과 맞는 성질 | 한 번의 실행 오차가 작다는 뜻은 아니다. |

### 5.2 오픈 멀티모달: Apertus 1.5

Project Apertus는 **Apertus 1.5: From Open Text Models to Open Multimodal Reasoners** 기술보고서를 2026-10-11 공개했고, 2026-10-15 릴리스 노트에서 내용을 정리했다.

중요한 날짜 구분이 있다.

- 기술보고서: 2026-10-11
- 릴리스 노트: 2026-10-15
- 8B·70B 가중치: 2026-07-24 공개

따라서 “어제 새 가중치가 출시됐다”고 쓰면 부정확하다. 어제 공개된 것은 상세 릴리스 노트다.

#### 무엇이 달라졌나?

Apertus 1.0의 8B·70B text model을 계속 사전학습했다.

| 모델 | 추가 학습 token | 입력 modality | 출력 |
| --- | ---: | --- | --- |
| Apertus 1.5 8B | 약 $4$T | text·image·audio | text |
| Apertus 1.5 70B | 약 $2$T | text·image·audio | text |

context window는

$$
262{,}144
$$

token으로, 이전 세대의 네 배라고 보고했다.

모델의 decoder-only Transformer 본체에 별도 CLIP류 semantic vision tower를 붙이는 대신, 동결된 Emu3.5 neural vision tokenizer와 WavTokenizer가 이미지·오디오를 이산 code로 바꾸고 이를 text와 같은 sequence에 넣는 **discrete-token early fusion**을 사용한다. 따라서 “시각·오디오 encoder가 전혀 없다”는 뜻은 아니다.

#### 이미지와 오디오는 몇 token이 되는가?

이미지는 $16\times16$ pixel block마다 code 하나를 만든다.

$512\times512$ 이미지는

$$
\frac{512}{16}
\times
\frac{512}{16}
=
32\times32
=
1{,}024
$$

개의 visual code token이 된다. 실제 sequence에는 image start, 크기, row separator, end 같은 구조 token이 추가로 붙으므로 $1{,}024$는 구조 token을 제외한 code 수다.

오디오는 $24$ kHz mono를 기준으로 초당 $40$ token이다.

1분이면

$$
60\times40
=
2{,}400
$$

token이다.

따라서 multimodal 입력은 “무료 부가 기능”이 아니다. 이미지 수·해상도·오디오 길이가 context와 prefill 비용을 빠르게 사용한다.

#### thinking은 별도 모델인가?

아니다. 모델 카드에서는 chat template의

$$
\text{enable\_thinking=True}
$$

같은 flag로 reasoning 구간을 켠다. 기술보고서는 8B에서 수학 reasoning이 최대 $20$점 개선됐다고 보고하지만, 이는 저자 평가이고 task별 token 비용·정확도 trade-off를 별도로 확인해야 한다.

#### 무엇이 공개됐나?

프로젝트는 다음을 공개한다고 설명한다.

- 8B·70B model weights
- 데이터 준비와 provenance 자료
- 학습 recipe와 launch script
- 중간 post-training checkpoint
- evaluation suite
- alignment charter

모델 가중치는 Apache-2.0으로 표시돼 있다. 다만 Hugging Face에서 파일 접근 시 usage policy 동의와 연락처 공유가 요구되며, 연결된 데이터·코드 artifact마다 개별 라이선스를 확인해야 한다.

#### 한계

- 입력은 text·image·audio지만 출력은 text뿐이다.
- native video 입력은 지원하지 않아 frame 추출 같은 전처리가 필요하다.
- model card는 중요한 답을 검증해야 하며 hallucination과 bias가 남는다고 명시한다.
- 기술보고서는 전문 vision·audio 모델이 일부 perception과 고자원 언어 transcription에서 앞선다고 보고한다.
- 공식 Transformers·vLLM upstream 통합이 기준 시각에 진행 중이어서 프로젝트 branch·container가 필요할 수 있다.
- 긴 nominal context가 모든 위치에서 같은 retrieval·reasoning 품질을 보장하지 않는다.

#### 엔지니어 인사이트 (Impact)

첫째, early fusion은 modality마다 완전히 다른 serving graph를 두는 대신 공통 token sequence와 decoder stack을 활용하게 한다. 이는 학습·서빙 인프라를 단순화할 수 있지만, 이미지·오디오 token이 text와 같은 context 예산을 경쟁한다.

둘째, “완전 공개”의 실질적 가치는 benchmark 숫자 하나보다 실패를 추적할 수 있는 provenance에 있다. 어떤 데이터와 stage에서 능력이 생겼는지 조사할 수 있어야 모델 업데이트를 감사할 수 있다.

셋째, 오늘의 정책 선택 원리를 그대로 적용해야 한다. text·document·speech·tool-use benchmark 중 잘 나온 것만 골라 모델을 선택하면 최댓값 선택 편향이 생긴다. 실제 한국어 문서·OCR·오디오·도구호출 workload를 사전 고정하고, 후보 모델 전체에 동시 유효한 품질·비용 경계를 적용하는 편이 안전하다.

**1차 출처**

- [Apertus 1.5 릴리스 노트](https://huggingface.co/blog/swiss-ai/apertus-v15-release-notes)
- [Apertus 1.5 기술보고서](https://arxiv.org/abs/2610.15542)
- [Apertus 1.5 8B 공식 모델 카드](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)

### 5.3 LLM 안전 평가: Weighted Ensemble Sampling

**Weighted Ensemble Sampling for Rare-Event Estimation in Language Models**는 arXiv v1이 2026-10-13 21:11 UTC에 제출됐고 2026-10-15 신규 목록에 공개됐다.

#### 왜 이 문제가 중요한가?

희귀 실패 확률을

$$
p_\star
=
\Pr_{X\sim p_0}
\left(
R(X)\geq\gamma_\star
\right)
$$

라고 하자.

직접 Monte Carlo에서는 $N$개의 완전한 응답을 독립 생성해

$$
\widehat p_{\mathrm{MC}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1
\left\{
R(X_i)\geq\gamma_\star
\right\}
$$

로 추정한다.

사건 확률이 $10^{-5}$이면 평균적으로 사건 하나를 보기 위해 약 $100{,}000$개 생성이 필요하다. 안정적인 추정을 위해서는 그보다 훨씬 많은 표본이 필요할 수 있다.

#### Weighted Ensemble의 핵심

논문은 완전한 응답을 끝까지 독립 생성하는 대신, 생성 중간 checkpoint에서 prefix들을 bin으로 나눈다.

각 prefix는 확률 질량을 나타내는 가중치 $w_i$를 가진다.

bin $b$의 총 질량은

$$
W_b
=
\sum_{i\in I_b}w_i
$$

이다.

그 bin에 다음 단계 walker $n_b$개를 배정한다. bin 안에서는 parent $i$를

$$
\Pr(J=i\mid b)
=
\frac{w_i}{W_b}
$$

의 확률로 복원추출하고, 각 child의 가중치는

$$
\frac{W_b}{n_b}
$$

가 된다.

논문의 실험은 occupied bin마다 거의 같은 walker 수를 주는 equal-population allocation을 사용한다. 따라서 원래 확률 질량이 작은 bin은 상대적으로 oversample되고, reaction coordinate가 유익하면 희귀사건으로 이어지는 경로가 더 오래 보존된다. 이는 “위험 점수에 정비례해 walker를 배정한다”는 뜻이 아니다. 모든 다음 token은 여전히 원래 모델

$$
p_0(x_t\mid x_{<t},c)
$$

에서 뽑는다.

최종 추정량은

$$
\widehat p_{\mathrm{WE}}
=
\sum_{i=1}^{N}
w_i
\mathbf 1
\left\{
R(X_i)\geq\gamma_\star
\right\}
$$

이다.

hit rate는 크게 올릴 수 있지만, 각 표본의 가중치가 다르므로 단순 사건 비율을 원래 확률로 해석하면 안 된다.

#### reaction coordinate는 무엇인가?

prefix가 희귀사건에 얼마나 가까운지 나타내는 feature다.

논문은 다음을 비교한다.

- 현재 rare-event reward만 사용
- 무작위 feature
- 초기에는 LLM hidden representation으로 clustering
- 후반에는 rare-event reward로 전환

좋은 coordinate는 미래 rare-event 확률이 비슷한 prefix를 같은 bin에 모아 분산을 낮춘다.

#### 저자 보고 결과

TinyStories-33M toxic-story 설정의 한 prompt에서 직접 Monte Carlo 기준 확률은

$$
1.24\times10^{-5}
$$

였다.

같은 총 sampling budget의 비교에서 hidden representation을 초기 coordinate로 쓰고 후반에 reward로 전환한 한 설정은

- 추정값 $1.29\times10^{-5}\pm2.00\times10^{-5}$
- 상대오차 $4.2\%$
- zero estimate 비율 $2.9\%$
- rare-event hit rate $76.7\%$
- direct Monte Carlo 대비 저자 계산 sample-efficiency gain $2.16\times$

를 보고했다. 이 표는 $40$개 seed의 결과를 요약하며, 상대오차 $4.2\%$는 $1{,}048{,}576$개 direct Monte Carlo 표본으로 만든 **large-sample reference**에 대한 seed 평균의 오차다. 그 reference 자체가 알려진 ground truth는 아니다. 또한 $2.16\times$는 empirical variance로 예측한 점근적 sample-efficiency gain이지 wall-clock 속도 향상이 아니다.

그러나 reward-only 설정은 hit rate가 $97.3\%$로 더 높았어도 상대오차가 $49.7\%$이고 gain이 $0.38\times$였다.

> rare event를 자주 찾는 것과 원래 확률을 정확히 추정하는 것은 같은 목표가 아니다.

이는 어제 학습한 “발견 정책”과 “측정 정책”의 구분과 정확히 맞닿는다.

#### 현대 LLM 실험

논문은 StrongREJECT의 여섯 harmful category에서

- Llama-3.1-8B
- Qwen2.5-7B-Instruct

를 평가했다.

각 WE run은 walker $1{,}024$개와 $5$개 seed를 사용했고, 직접 Monte Carlo 기준은 prompt별 $10{,}000$개 표본을 사용했다. hidden representation을 쓰는 hybrid coordinate가 여러 설정에서 reward-only보다 낮은 상대오차를 보였지만, 결과는 category와 모델에 따라 달랐다.

#### rare-event 정의가 바뀌면?

WE는 원래 모델에서 계속 token을 생성하고 weight만 재배분한다. 논문은 희귀사건 threshold를 나중에 낮춰 사건 집합을 넓혀도, 저장된 weighted sample로 새 확률을 무편향 추정할 수 있는 조건을 제시한다.

특정 원래 사건만 겨냥해 학습한 proposal distribution은 새로 추가된 사건 영역에 확률 $0$을 줄 수 있어 재사용이 깨질 수 있다.

#### 한계와 오픈소스 상태

- 핵심 결과는 arXiv v1의 저자 보고다.
- hybrid coordinate는 평가 모델의 hidden state에 접근해야 하므로 폐쇄형 API에는 그대로 쓸 수 없다.
- reward-only WE는 black-box에서도 가능하지만 분산이 더 클 수 있다.
- rare-event score 자체가 잘못되면 “측정하고 싶은 실패”와 “점수가 잡는 실패”가 달라진다.
- adaptive resampling이 독립 표본을 만들지 않으므로 walker 수를 그대로 일반 표본 수로 해석하면 안 된다.
- 논문 표 1은 $40$개 seed라고 쓰면서 zero rate $2.9\%$, $5.9\%$, $8.8\%$를 보고한다. 이는 $1/40=2.5\%$의 정수배와 맞지 않는 원문 내부 불일치이므로 해당 비율은 저자 표의 수치를 옮긴 것으로만 해석해야 한다.
- 논문 원고는 CC BY 4.0이지만, 기준 시각에 arXiv 페이지에는 공식 구현 저장소가 연결돼 있지 않았다. 논문 공개와 재현 가능한 오픈소스 구현을 같은 것으로 보면 안 된다.

#### 엔지니어 인사이트 (Impact)

첫째, 안전 평가에서 단순 hit count를 최적화하면 실제 배포 확률 추정이 무너질 수 있다. 발견 효율·확률 추정 오차·ESS·zero rate를 함께 봐야 한다.

둘째, prefix hidden state를 이용한 계산 예산 배분은 LLM 자체가 가진 중간 위험 신호를 평가 인프라에 활용하는 패턴이다. 다만 그 신호를 선택에 썼다면 최종 확률에는 split·merge weight가 정확히 반영돼야 한다.

셋째, 오늘의 비관적 정책 학습과 대칭이다. OPL은 불확실한 행동의 낙관값을 깎고, WE는 희귀 prefix에 계산을 더 주되 확률 질량을 보존한다. 둘 다 적응적 자원 배분을 하면서 선택 편향을 명시적으로 교정한다.

**1차 출처**

- [arXiv 초록·제출 기록](https://arxiv.org/abs/2610.19202)
- [논문 HTML 원문](https://arxiv.org/html/2610.19202)

### 5.4 두 트렌드가 오늘 수학에 주는 공통 교훈

| 오늘의 원리 | Apertus 1.5 평가 | Weighted Ensemble |
| --- | --- | --- |
| 후보군 | modality·thinking·checkpoint·모델 크기 | prefix bin·reaction coordinate |
| 선택 신호 | workload benchmark | prefix hidden state·rare-event score |
| 선택 편향 위험 | 잘 나온 benchmark만 보고 모델 선택 | hit rate가 높은 설정을 정확한 추정기로 착각 |
| 보정 대상 | 모델·설정 전체의 품질·비용 불확실성 | split·merge 뒤의 probability weight |
| coverage 문제 | 실제 workload에 없는 언어·modality 조합 | 유한 walker가 희귀 prefix 영역을 충분히 덮지 못해 커지는 분산 |
| 보수적 운영 | 독립 workload gate·fallback | ESS·상대오차·zero rate 동시 보고 |
| 재현성 | 데이터·recipe·checkpoint 공개 | 논문은 공개됐으나 공식 구현 연결은 미확인 |

공통 결론은 간단하다.

> 적응적으로 더 유망한 곳에 계산을 쓰는 것은 강력하다. 그러나 선택 규칙과 가중치, 실패확률, 최종 독립 검증을 기록하지 않으면 효율 향상이 곧 측정 왜곡이 된다.

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

한 팀이 같은 적응 감사 로그에서 $50$개의 후보 정책을 학습했다. 각 후보는 기준 정책 대비 DR 품질 개선량과 안전 cost 변화량을 가진다.

팀은 다음 절차를 썼다.

1. 같은 로그 전체로 outcome model을 학습했다.
2. 같은 로그 전체에서 $50$개 후보를 학습했다.
3. 각 후보의 pointwise $95\%$ LCB를 계산했다.
4. 품질 LCB가 가장 큰 후보를 골랐다.
5. 그 후보의 cost 평균이 한도 아래라 바로 전체 배포하려 한다.

이 절차가 왜 위험한가? 최소한 어떻게 고쳐야 하는가?

### 모범 답안

첫째, pointwise LCB는 후보가 미리 고정됐을 때의 경계다. 같은 로그에서 $50$개를 비교해 가장 큰 LCB를 고르면 선택된 후보에는 최댓값 선택 편향이 남는다.

후보 $k$의 추정값을

$$
\widehat\Delta_k
=
\Delta_k+\varepsilon_k
$$

라고 하면 선택된 후보는

$$
\widehat k
=
\arg\max_k
\left[
\widehat\Delta_k
-
1.645\widehat{\operatorname{SE}}_k
\right]
$$

이다. 이 후보는 실제 개선뿐 아니라 유리한 $\varepsilon_k$까지 큰 정책일 가능성이 높다.

둘째, outcome model을 같은 표본에서 학습하고 같은 표본의 residual을 계산하면 overfit된 예측오차가 DR score를 낙관적으로 만들 수 있다. 사용자·세션 단위 cross-fitting으로 out-of-fold prediction을 만들어야 한다.

셋째, cost의 평균만 한도 아래인 것은 충분하지 않다. cost는 작을수록 좋은 양이므로 후보 선택까지 반영한 동시 UCB가 한도 아래여야 한다.

$$
\operatorname{UCB}_{\Delta C_j}(\pi)
\leq
d_j
$$

를 모든 중요 cost와 주요 slice에서 확인해야 한다.

넷째, propensity와 overlap을 검사해야 한다.

$$
w_i(\pi)
=
\frac{\pi(A_i\mid X_i)}
{b_i(A_i\mid X_i,\mathcal H_{i-1})}
$$

의 최댓값·분포·ESS를 보고, behavior가 선택하지 않은 행동을 후보가 선택한다면 로그만으로 그 후보를 검증할 수 없다.

다섯째, 최소 수정 절차는 다음과 같다.

1. 사용자·시간 구조를 고려해 로그를 train·selection·final-gate로 분리한다.
2. train 안에서 group cross-fitting으로 outcome model과 DR score를 만든다.
3. train에서 후보를 학습하되 support floor·KL 또는 baseline mixture로 분포 이동을 제한한다.
4. selection에서 $50$개 후보와 모든 cost에 대해 family-wise 실패확률을 배분한 uniform LCB·UCB를 계산한다.
5. 품질 LCB가 최소 개선폭을 넘고 모든 cost UCB가 한도 아래이며 ESS·slice gate를 통과한 후보만 남긴다.
6. 후보 하나와 reward·cost·threshold·fallback을 잠근다.
7. 한 번도 보지 않은 final-gate 로그에서 그 고정 후보의 paired DR 개선량과 cost를 다시 계산한다.
8. gate를 통과해도 shadow와 작은 무작위 canary를 거쳐 time-uniform 품질 LCB·위험 UCB를 확인한다.

여섯째, final-gate 결과가 마음에 들지 않아 threshold를 바꾸면 그 gate는 더 이상 독립 검증 데이터가 아니다. 바꾼 정책은 새 데이터에서 다시 검증해야 한다.

마지막으로 가장 중요한 한 문장으로 요약하면 다음과 같다.

> 정책 학습은 추정 오차를 이용할 수 있으므로, 선택 과정 전체를 덮는 균등 경계와 독립 검증이 없으면 “가장 좋아 보이는 정책”은 단지 “잡음을 가장 잘 맞춘 정책”일 수 있다.

---

**다음 학습 예고:** 오늘은 같은 로그에서 정책을 학습·선택할 때의 optimizer’s curse를 균등 경계와 pessimism으로 다뤘다. 다음에는 제한된 정책군을 넘어 함수 근사기가 만드는 epistemic uncertainty를 ensemble·bootstrap·conformal 방법으로 calibration하고, distributionally robust optimization으로 context mix drift까지 견디는 정책을 설계한다.
