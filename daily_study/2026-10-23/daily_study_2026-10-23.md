# [2026-10-23] 오늘 학습: 파티클 필터·순차 몬테카를로 & Replay 영향 추정·Machine Unlearning

> **오늘의 핵심 문장:** 복잡한 현실에서는 숨은 상태의 posterior를 하나의 평균과 분산만으로 표현할 수 없고, 학습 데이터 한 건의 흔적도 파라미터 한 곳에만 저장되지 않는다. 파티클 필터는 여러 가능한 세계와 그 믿음의 무게를 함께 운반하며, influence function과 Shapley 근사는 어떤 기억이 모델에 기여했는지를 국소적 또는 협력게임 관점에서 추적한다. 그러나 영향 추정은 삭제 자체가 아니며, “잊었다”는 주장은 재학습 기준선·보존 성능·공격·재등장 검사를 포함한 독립 검증을 통과해야 한다.

지난 시간에는 HMM·HSMM·BOCPD로 숨은 상태, 체류시간, 변화 시점을 추론하고 제한된 replay buffer를 설계했다. 그 과정에는 두 가지 강한 가정이 숨어 있었다.

1. HMM의 상태가 유한 개이고 forward–backward 합을 정확히 계산할 수 있다.
2. replay buffer에 무엇을 넣을지는 정했지만, 각 사례가 실제 업데이트에 얼마나 영향을 주었는지는 아직 모른다.

오늘은 이 두 한계를 같은 질문으로 연결한다.

$$
\text{복잡한 분포}
\xrightarrow{\text{가중 표본}}
\text{계산 가능한 근사}
\qquad\text{그리고}\qquad
\text{복잡한 학습 이력}
\xrightarrow{\text{영향 근사}}
\text{감사 가능한 삭제 후보}
$$

첫 번째 화살표가 particle filter와 Sequential Monte Carlo(SMC)이고, 두 번째 화살표가 influence function·Shapley 근사와 machine unlearning이다. 둘 다 “작은 대표 집합”을 쓰지만 목적은 다르다.

- particle은 현재 숨은 상태분포를 근사하는 **확률 표본**이다.
- replay 사례는 과거 능력을 보존하려고 저장한 **학습 데이터**다.
- influence·Shapley 점수는 학습 결과에 대한 사례의 **추정 기여도**다.
- unlearning은 지정된 영향의 제거를 목표로 하는 **모델 수명주기 작업**이다.
- deletion verification은 제거 주장이 허용되는 범위를 밝히는 **증거 절차**다.

오늘의 전체 흐름은 다음과 같다.

$$
\text{비선형·비Gaussian 상태공간모형}
\rightarrow
\text{particle 예측}
\rightarrow
\text{관측으로 재가중}
\rightarrow
\text{ESS 진단·재표집}
\rightarrow
\text{불확실성 있는 상태 추정}
$$

$$
\text{replay·학습 이력}
\rightarrow
\text{influence·Shapley 근사}
\rightarrow
\text{삭제 범위 결정}
\rightarrow
\text{exact 또는 approximate unlearning}
\rightarrow
\text{forget·retain·privacy·relearning 검증}
$$

> **난이도 표지:** 파티클 필터는 안개 속 자동차 위치를 추적하는 여러 정찰대다. 관측과 잘 맞는 정찰대의 표를 더 많이 복제하고, 맞지 않는 표는 줄인다. Influence function은 “이 영수증 한 장을 장부에서 빼면 최종 합계가 어느 방향으로 움직일까?”를 현재 해 주변에서 미분해 보는 도구다. Shapley 값은 여러 팀 조합에 한 사람을 넣었을 때 성과가 얼마나 늘었는지를 모든 입장 순서에 걸쳐 평균낸 공로 점수다. 어느 것도 그 자체로 삭제 완료 증명서는 아니다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 Kalman filter나 HMM만으로 모든 상태를 추적할 수 없을까?

상태공간모형은 보이지 않는 상태 $X_t$가 시간에 따라 움직이고, 관측 $Y_t$가 그 상태의 불완전한 흔적이라고 본다.

$$
X_t\sim f_t(x_t\mid x_{t-1}),
\qquad
Y_t\sim g_t(y_t\mid x_t).
$$

Kalman filter는 선형 변환과 Gaussian 잡음이라는 특별한 조건에서 posterior를 평균과 공분산만으로 정확히 표현한다. HMM은 상태가 유한 개일 때 각 상태의 확률을 정확히 합산한다. 그러나 현실에는 다음과 같은 경우가 흔하다.

- 로봇 동역학이 회전·접촉 때문에 비선형이다.
- 카메라 관측이 가려짐 때문에 여러 위치를 동시에 그럴듯하게 만든다.
- 이상치가 많아 잡음분포의 꼬리가 두껍다.
- posterior가 “왼쪽 통로”와 “오른쪽 통로”라는 두 봉우리를 갖는다.

두 봉우리를 평균 하나로 압축하면 평균 위치가 벽 한가운데처럼 실제로 불가능한 곳에 놓일 수도 있다. 모든 가능한 상태를 촘촘한 격자로 적분하면 차원이 커질수록 계산량이 폭발한다. 그래서 분포 전체를 유한한 가중 표본으로 근사하는 SMC가 등장했다.

### 1.2 왜 그냥 상태를 여러 개 무작위로 뽑기만 하면 안 될까?

목표 posterior $p(x_t\mid y_{1:t})$에서 직접 표본을 뽑기 어려워 다른 proposal $q(x_t\mid\cdots)$에서 뽑는다고 하자. 그러면 proposal이 자주 보여 준 영역과 목표분포가 중요하게 보는 영역이 다를 수 있다. 단순 평균은 이 차이를 무시한다.

Importance sampling은 각 표본에

$$
w(x)\propto\frac{\text{목표밀도}}{\text{proposal 밀도}}
$$

라는 영수증을 붙인다. 목표에는 중요한데 proposal에서는 드물게 나온 표본은 큰 무게를 받고, proposal이 과하게 보여 준 표본은 작은 무게를 받는다. SMC는 이 아이디어를 시간축에서 재귀적으로 수행한다.

### 1.3 왜 재가중만 계속하면 실패할까?

확률을 여러 번 곱하면 대부분 particle의 가중치는 거의 $0$이 되고 소수만 살아남는다. $N=10{,}000$개를 계산했어도 사실상 한 particle만 결정한다면 계산 예산 대부분이 낭비된다. 이를 **weight degeneracy**라고 한다.

재표집(resampling)은 큰 가중치의 particle을 여러 번 복제하고 작은 가중치의 particle을 버려 다시 균등 가중치로 출발한다. 그러나 너무 자주 복제하면 서로 다른 조상이 사라지고 같은 particle만 남는다. 이후 전이잡음이 작으면 복제본들이 거의 같은 길을 걷는다. 이것이 **particle impoverishment**이며, 과거 경로까지 보면 **path degeneracy**가 생긴다.

따라서 정답은 “항상 재표집”도 “절대 재표집하지 않음”도 아니다. ESS 같은 진단값과 목적에 따라 시점을 정해야 한다.

### 1.4 왜 replay에 들어간 사례의 수만 세어서는 부족할까?

Replay buffer 안에 있다는 사실은 그 사례가 중요하다는 뜻이 아니다.

- 이미 수천 개의 비슷한 사례가 있다면 하나를 빼도 모델은 거의 변하지 않을 수 있다.
- 희귀 안전 사례 하나가 결정경계를 크게 움직일 수 있다.
- 학습 초기에 본 사례와 마지막에 본 사례의 영향이 optimizer 경로 때문에 다를 수 있다.
- 같은 사례가 중복·증강·파생 데이터로 여러 번 나타날 수 있다.

그래서 “포함 여부”보다 “이 사례를 빼면 파라미터·검증 손실·특정 행동이 어떻게 달라지는가?”를 묻는 영향 추정이 필요하다.

### 1.5 왜 influence function이 등장했을까?

데이터 한 건을 뺄 때마다 모델을 처음부터 다시 학습하면 가장 직접적이지만, 대형 모델에서는 비용이 너무 크다. Influence function은 최적점에서 학습 사례의 가중치를 아주 조금 바꾸고, 최적점이 움직이는 방향을 implicit differentiation으로 구한다.

이는 다음 질문에 대한 **국소 1차 근사**다.

> 사례 $z_i$의 학습 비중을 아주 조금 줄이면 모델 파라미터와 관심 평가량이 어느 방향으로 얼마나 변할까?

비선형 신경망의 전체 재학습 경로를 정확히 재현하는 마법은 아니다. Hessian의 역작용, 최적점 가정, 미분 가능성, 작은 교란이라는 조건이 필요하다.

### 1.6 왜 Shapley 값은 influence와 다른가?

데이터의 가치는 상호작용한다. 예를 들어 같은 희귀 패턴을 설명하는 두 사례는 각각 혼자 있을 때 중요하지만 둘이 함께 있으면 서로 대체할 수 있다. Influence function은 현재 최적점 주변의 미소 변화에 강하고, 이런 큰 조합 효과를 완전히 표현하지 못한다.

Shapley 값은 사례 $i$가 어떤 부분집합 $S$ 뒤에 합류했을 때 성능이 얼마나 변하는지 모든 가능한 순서에 걸쳐 평균낸다. 공정한 기여 배분 공리를 만족하지만, 부분집합마다 재학습해야 하므로 정확 계산은 지수적으로 비싸다. 실무에서는 permutation sampling, group Shapley, surrogate model 같은 근사를 쓴다.

### 1.7 왜 “영향이 작다”와 “삭제됐다”는 다른 말일까?

Influence 점수가 작아도 다음 흔적이 남을 수 있다.

- 모델 가중치 속 memorized association
- optimizer moment와 checkpoint
- embedding index·vector store·cache
- fine-tuning adapter와 quantized 사본
- synthetic derivative와 증강 표본
- 로그·평가셋·백업
- 이미 배포된 여러 replica

반대로 특정 프롬프트에 답하지 못하게 만드는 것만으로는 학습 영향이 제거됐다고 말할 수 없다. 단순 refusal, prompt filter, 출력 억제는 내부 표현이나 우회 질의에서 정보가 남아 있을 수 있다.

Machine unlearning은 보통 삭제 대상 데이터로 학습하지 않은 reference model과 충분히 가까운 결과를 효율적으로 만들려 한다. 그러나 “가깝다”의 기준은 파라미터 거리, 출력분포, 공격 성공률, downstream utility마다 다르다. 따라서 주장은 반드시 범위를 가져야 한다.

### 1.8 오늘의 본질

오늘의 두 트랙은 모두 근사에 관한 수업이다.

$$
\widehat p_t^N
\approx
p(x_t\mid y_{1:t})
\qquad\text{와}\qquad
\widehat{\operatorname{Influence}}_i
\approx
\text{실제 leave-one-out 재학습 효과}.
$$

근사는 틀릴 수 있다는 사실이 약점만은 아니다. 어떤 가정 아래 무엇을 근사하며, 언제 진단이 무너지고, 어떤 기준선으로 확인할지를 명시하면 계산 가능한 과학이 된다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| latent state $X_t$ | 시점 $t$에 실제로 존재하지만 직접 보이지 않는 상태 | 로봇 위치·사용자 의도·서비스 건강도처럼 추적할 대상이다. |
| observation $Y_t$ | 센서나 로그로 실제 관찰한 값 | 숨은 상태에 대한 불완전한 단서다. |
| transition $f_t(x_t\mid x_{t-1})$ | 이전 상태에서 다음 상태가 생기는 조건부 분포 | particle을 미래로 이동시킨다. |
| likelihood $g_t(y_t\mid x_t)$ | 상태가 $x_t$일 때 실제 관측 $y_t$가 나올 그럴듯함 | 관측과 맞는 particle의 무게를 키운다. |
| filtering posterior $p(x_t\mid y_{1:t})$ | 지금까지의 관측을 모두 본 뒤 현재 상태에 대한 믿음 | particle filter가 매 시점 근사한다. |
| predictive distribution | 새 관측을 보기 직전 다음 상태에 대한 믿음 | 전이모형으로 이전 posterior를 밀어 보낸 결과다. |
| Gaussian | 종 모양의 연속 확률분포 | Kalman filter가 정확해지는 핵심 조건의 일부다. |
| non-Gaussian | 종 모양 하나로 충분히 표현되지 않는 분포 | 여러 봉우리·두꺼운 꼬리를 가질 수 있다. |
| Monte Carlo | 무작위 표본 평균으로 적분을 근사하는 방법 | 고차원 합·적분을 계산 가능한 평균으로 바꾼다. |
| SMC | 표본과 가중치를 시간 순서대로 갱신하는 Sequential Monte Carlo | particle filter를 포함하는 넓은 방법군이다. |
| particle $X_t^{(i)}$ | 가능한 상태 한 개를 나타내는 표본 | $i=1,\ldots,N$개의 가설 중 하나다. |
| unnormalized weight $\widetilde w_t^{(i)}$ | 아직 합이 $1$이 되도록 나누지 않은 중요도 | 곱셈 갱신에 편리하다. |
| normalized weight $W_t^{(i)}$ | 모든 particle 가중치 합이 $1$이 되도록 정규화한 값 | posterior 질량의 몫처럼 해석한다. |
| proposal $q_t$ | 실제로 particle을 뽑는 분포 | 목표 posterior와 다를 수 있어 가중치 보정이 필요하다. |
| importance ratio | 목표밀도와 proposal 밀도의 비 | 표본 추출 편향을 보정한다. |
| bootstrap particle filter | 전이분포 자체를 proposal로 쓰는 가장 기본적인 particle filter | 증분 중요도 인자가 likelihood로 단순화되며, 재표집하지 않았다면 전체 가중치에는 이전 가중치도 곱한다. |
| empirical measure $\widehat p_t^N$ | particle 위치에 가중 질량을 놓은 이산분포 | 연속 posterior의 계산 가능한 근사다. |
| estimator | 모르는 양을 데이터로 계산한 값 | $\sum_iW_i\varphi(X_i)$처럼 기대값을 근사한다. |
| ESS | 가중 표본이 균등 표본 몇 개와 비슷한지 나타내는 effective sample size | 재표집 여부를 정하는 진단값이다. |
| weight degeneracy | 가중치 대부분이 거의 $0$이 되는 현상 | 계산한 particle 수보다 실제 정보량을 작게 만든다. |
| resampling | 가중치에 비례해 particle의 조상 인덱스를 다시 뽑는 단계 | 낮은 가중치 입자를 제거하고 계산을 집중한다. |
| multinomial resampling | 각 자식을 독립적으로 범주분포에서 뽑는 기본 방식 | 단순하지만 resampling 분산이 비교적 크다. |
| stratified resampling | $[0,1]$을 층으로 나누어 층마다 하나씩 뽑는 방식 | multinomial보다 분산을 줄이는 대표 방법이다. |
| systematic resampling | 하나의 무작위 시작점과 등간격 점을 쓰는 방식 | 빠르고 흔하지만 모든 함수에 대해 항상 최소분산이라는 뜻은 아니다. |
| particle impoverishment | 재표집 뒤 복제본만 많아져 다양성이 부족해지는 현상 | 전이잡음이 작을 때 특히 심하다. |
| path degeneracy | 현재 particle은 여럿이어도 과거 조상이 몇 개로 합쳐지는 현상 | 긴 경로 smoothing과 계보 추적을 어렵게 한다. |
| rejuvenation | Target posterior를 보존하는 MCMC move 등으로 particle 다양성을 회복하는 조치 | 재표집 후 빈곤화를 완화한다. 단순 jitter인 roughening은 다양성 대신 편향을 만들 수 있다. |
| log-sum-exp | 작은 확률 곱을 로그공간에서 안정적으로 합치는 기법 | weight underflow를 막는다. |
| replay buffer $\mathcal B$ | 과거 학습 사례 일부를 보관한 메모리 | 지속학습과 삭제 요청의 데이터 계보가 된다. |
| training example $z_i$ | 입력·정답·메타데이터를 포함한 학습 단위 | 영향과 삭제의 기본 대상이지만 실제 요청 단위와 다를 수 있다. |
| empirical risk $R(\theta)$ | 학습 사례 손실의 평균 | 학습이 최소화하려는 목적함수다. |
| parameter $\theta$ | 모델이 학습하는 모든 수치 | 가중치·편향·adapter 등이 포함된다. |
| gradient $g_i$ | 사례 $i$의 손실이 가장 빠르게 증가하는 1차 방향 | Gradient descent는 반대 방향 $-g_i$로 파라미터를 움직이며, $g_i$는 influence·TracIn의 핵심 재료다. |
| Hessian $H$ | 목적함수의 2차 미분 행렬 | 최적점 주변 곡률과 방향별 민감도를 나타낸다. |
| inverse-Hessian vector product | $H^{-1}$를 직접 만들지 않고 벡터에 작용시킨 결과 | 대형 모델 influence 계산의 현실적 목표다. |
| damping $\lambda$ | $H+\lambda I$처럼 곡률행렬에 더하는 안정화 상수 | 역문제가 불안정하거나 특이할 때 쓴다. |
| influence function | 사례 가중치의 미소 변화가 최적점·평가량에 미치는 1차 근사 | leave-one-out 효과를 싸게 추정한다. |
| leave-one-out | 사례 하나를 빼고 다시 학습해 차이를 보는 기준 | 비싸지만 영향 근사의 소규모 검증 기준선이다. |
| TracInCP | 여러 checkpoint에서 train/test gradient 내적을 누적하는 TracIn의 실용 근사 | Hessian 역을 피하지만 checkpoint와 학습률 선택에 의존한다. |
| value function $v(S)$ | 데이터 부분집합 $S$로 학습했을 때의 효용 | Shapley 값이 나눠 줄 전체 성과다. |
| Shapley value $\phi_i$ | 모든 합류 순서에서 사례 $i$의 한계기여를 평균낸 값 | 상호작용을 반영하는 데이터 가치 정의다. |
| marginal contribution | $v(S\cup\{i\})-v(S)$ | 사례 $i$를 조합에 추가했을 때의 성과 변화다. |
| permutation sampling | 무작위 합류 순서를 몇 개 뽑아 Shapley 값을 근사하는 방법 | $2^n$ 조합 비용을 줄인다. |
| exact unlearning | unlearning 결과의 확률분포가 삭제 대상 없이 재학습한 결과분포와 같은 절차 | 결정론적 학습에서는 같은 결과로 줄지만 확률적 학습에서는 분포를 비교한다. |
| approximate unlearning | 제한된 업데이트로 reference retraining 결과에 가까워지려는 절차 | 빠르지만 근사 범위와 오차를 검증해야 한다. |
| deletion request | 어떤 주체·레코드·파생물을 어느 시스템에서 제거할지 정한 요청 | 단순한 row ID보다 넓은 계보 정의가 필요하다. |
| forget set $\mathcal D_f$ | 제거해야 할 데이터 집합 | unlearning이 줄여야 할 대상 영향이다. |
| retain set $\mathcal D_r$ | 계속 학습돼 있어야 할 데이터 집합 | 삭제 중 보존해야 할 일반 능력을 평가한다. |
| retrain reference $\theta_{-f}$ | forget set 없이 처음부터 같은 절차로 학습한 기준 모델 | unlearning 품질의 가장 직접적인 기준이다. |
| utility | 남겨야 할 과제에서의 유용성 | 무작정 성능을 파괴해 잊은 척하는 것을 막는다. |
| membership inference | 특정 사례가 학습에 들어갔는지 출력으로 추정하는 공격 | privacy 진단 중 하나이지 삭제의 충분조건은 아니다. |
| extraction attack | 우회 프롬프트·다양한 조건으로 기억된 내용을 끌어내려는 공격 | 단일 정답 질의보다 강한 forget 검증이다. |
| relearning test | 소량의 재노출로 대상 지식이 얼마나 빨리 학습되는지 보는 스트레스 검사 | Candidate만 빠르다는 것으로는 부족하고 retrain reference·일반 plasticity control과 같은 protocol로 비교한다. |
| model lineage | base model부터 adapter·quantization·index·배포 replica까지의 계보 | 삭제 범위를 누락하지 않게 한다. |
| claim scope | 삭제 증거가 실제로 지지하는 정확한 범위 | “모든 흔적 삭제” 같은 과장을 막는다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 정확한 Bayesian filtering 재귀식

관측 전에는 이전 posterior와 전이분포를 적분해 예측분포를 만든다.

$$
p(x_t\mid y_{1:t-1})
=
\int
f_t(x_t\mid x_{t-1})
p(x_{t-1}\mid y_{1:t-1})
\,dx_{t-1}.
$$

새 관측 $y_t$를 보면 Bayes 정리로 갱신한다.

$$
p(x_t\mid y_{1:t})
=
\frac{
g_t(y_t\mid x_t)
p(x_t\mid y_{1:t-1})
}{
\int
g_t(y_t\mid x_t')
p(x_t'\mid y_{1:t-1})
\,dx_t'
}.
$$

분모는 모든 가능한 현재 상태에 대해 likelihood를 평균낸 evidence다. 비선형·비Gaussian 모형에서는 이 적분을 닫힌형으로 계산하기 어렵다.

### 3.2 분포를 가중 원자들의 합으로 바꾸기

시점 $t-1$ posterior를 다음 경험분포로 근사한다고 하자.

$$
\widehat p_{t-1}^N(dx)
=
\sum_{i=1}^N
W_{t-1}^{(i)}
\delta_{X_{t-1}^{(i)}}(dx).
$$

$\delta_a$는 질량 전부를 점 $a$에 놓는 Dirac measure다. 어려운 말처럼 보이지만 “particle 위치에 해당 가중치만큼 점을 찍은 히스토그램”이다.

어떤 함수 $\varphi$의 posterior 기대값은

$$
\mathbb E[\varphi(X_t)\mid y_{1:t}]
\approx
\sum_{i=1}^N
W_t^{(i)}\varphi(X_t^{(i)})
$$

로 계산한다. $\varphi(x)=x$이면 posterior mean, $\varphi(x)=xx^\top$이면 2차 moment를 근사한다. 지도 위 특정 안전영역 $A$에 있을 확률은 $\varphi(x)=\mathbf 1\{x\in A\}$로 두면 된다.

### 3.3 일반 proposal에서 순차 중요도 가중치 유도

전체 경로 $x_{0:t}$의 목표분포는 상수에 비례해

$$
\pi_t(x_{0:t})
\propto
p(x_0)
\prod_{s=1}^t f_s(x_s\mid x_{s-1})
\prod_{s=1}^t g_s(y_s\mid x_s)
$$

로 쓸 수 있다. proposal을 재귀적으로

$$
q_t(x_{0:t}\mid y_{1:t})
=
q_{t-1}(x_{0:t-1}\mid y_{1:t-1})
q_t(x_t\mid x_{0:t-1},y_{1:t})
$$

로 잡는다. 목표와 proposal의 비를 시점 $t-1$까지와 새 항으로 나누면

$$
\widetilde w_t^{(i)}
=
\widetilde w_{t-1}^{(i)}
\frac{
g_t(y_t\mid X_t^{(i)})
f_t(X_t^{(i)}\mid X_{t-1}^{(i)})
}{
q_t(X_t^{(i)}\mid X_{0:t-1}^{(i)},y_{1:t})
}.
$$

이것이 sequential importance sampling의 핵심이다. 분자는 목표모형이 새 상태·관측 쌍에 주는 질량이고, 분모는 우리가 그 상태를 얼마나 자주 뽑았는지다.

정규화는

$$
W_t^{(i)}
=
\frac{\widetilde w_t^{(i)}}{\sum_{j=1}^N\widetilde w_t^{(j)}}
$$

로 한다.

### 3.4 Bootstrap particle filter가 단순해지는 이유

가장 단순한 선택은 proposal을 전이분포로 두는 것이다.

$$
q_t(x_t\mid x_{0:t-1},y_{1:t})
=
f_t(x_t\mid x_{t-1}).
$$

그러면 전이항이 약분되어

$$
\widetilde w_t^{(i)}
=
\widetilde w_{t-1}^{(i)}
g_t(y_t\mid X_t^{(i)})
$$

만 남는다. 재표집 직후 모든 조상 가중치를 $1/N$으로 초기화했다면 상수는 정규화에서 사라져

$$
\widetilde w_t^{(i)}
\propto
g_t(y_t\mid X_t^{(i)})
$$

이다.

장점은 구현이 쉽다는 것이다. 단점은 최신 관측 $y_t$를 보지 않고 전이분포에서 먼저 뽑는다는 점이다. 관측이 매우 날카롭거나 전이분포와 멀리 떨어져 있으면 대부분 particle이 관측과 맞지 않아 즉시 퇴화한다. 관측을 반영한 proposal은 가중치 분산을 줄일 수 있지만 sampling 비용이 커진다.

### 3.5 로그 가중치와 수치 안정성

긴 시계열에서 likelihood를 곱하면 부동소수점이 $0$으로 내려앉는다. 로그공간에서는 곱이 합이 된다.

$$
\ell_t^{(i)}
=
\log\widetilde w_{t-1}^{(i)}
+\log g_t(y_t\mid X_t^{(i)})
+\log f_t(X_t^{(i)}\mid X_{t-1}^{(i)})
-\log q_t(X_t^{(i)}\mid\cdots).
$$

$m_t=\max_i\ell_t^{(i)}$라 두면

$$
W_t^{(i)}
=
\frac{\exp(\ell_t^{(i)}-m_t)}
{\sum_j\exp(\ell_t^{(j)}-m_t)}
$$

로 계산할 수 있다. 모든 log weight에서 같은 $m_t$를 빼도 정규화 결과는 바뀌지 않지만 overflow·underflow가 줄어든다.

### 3.6 ESS는 무엇을 측정하는가?

정규화된 가중치의 합은 $1$이다. 널리 쓰는 경험적 ESS는

$$
\widehat{\operatorname{ESS}}_t
=
\frac{1}{\sum_{i=1}^N(W_t^{(i)})^2}.
$$

두 극단을 확인하자.

- 모두 균등하면 $W_i=1/N$이므로 $\sum_iW_i^2=1/N$, 따라서 ESS는 $N$이다.
- 한 particle의 가중치만 $1$이면 $\sum_iW_i^2=1$, 따라서 ESS는 $1$이다.

ESS는 실제 독립표본 수를 정확히 세는 정리가 아니라 가중치 집중도의 휴리스틱 진단이다. particle 사이의 상관, proposal 품질, 추정하려는 함수 $\varphi$에 따른 차이를 모두 담지는 못한다. 흔한 규칙은

$$
\widehat{\operatorname{ESS}}_t<\tau N
$$

일 때 재표집하는 것이며 $\tau$는 예를 들어 $0.5$일 수 있지만, 비용·동역학·위험도에 맞춰 검량해야 한다.

### 3.7 재표집 추정량의 통계적 불편성(unbiasedness)

재표집 전 particle $X_t^{(j)}$의 자식 수를 $N_t^{(j)}$라 하자. 좋은 재표집은 적어도 조건부 기댓값

$$
\mathbb E[N_t^{(j)}\mid W_t^{(1:N)}]
=
NW_t^{(j)}
$$

을 만족한다. 그러면 재표집 후 균등 평균의 조건부 기댓값은

$$
\mathbb E\left[
\frac{1}{N}
\sum_{j=1}^N
N_t^{(j)}\varphi(X_t^{(j)})
\;\middle|\;
X_t^{(1:N)},W_t^{(1:N)}
\right]
=
\sum_{j=1}^N
W_t^{(j)}\varphi(X_t^{(j)}).
$$

즉 재표집은 조건부 평균을 보존하지만 Monte Carlo 잡음을 추가한다. Multinomial, stratified, systematic resampling은 같은 평균을 목표로 해도 분산과 상관구조가 다르다.

### 3.8 Bootstrap particle filter 알고리즘

초기화:

$$
X_0^{(i)}\sim p(x_0),
\qquad
W_0^{(i)}=\frac1N.
$$

각 시점 $t=1,2,\ldots$에서 다음을 반복한다.

1. 재표집한다면 가중치에 따라 조상 $A_{t-1}^{(i)}$를 뽑고 기본 가중치를 $\overline W_{t-1}^{(i)}=1/N$으로 둔다. 재표집하지 않는다면 $A_{t-1}^{(i)}=i$, $\overline W_{t-1}^{(i)}=W_{t-1}^{(i)}$로 둔다.
2. 선택한 조상에서 다음 상태를 전파한다.

$$
X_t^{(i)}
\sim
f_t\left(x_t\mid X_{t-1}^{(A_{t-1}^{(i)})}\right).
$$

3. 기본 가중치에 새 관측의 likelihood를 곱한다.

$$
\widetilde w_t^{(i)}
=
\overline W_{t-1}^{(i)}
g_t(y_t\mid X_t^{(i)}).
$$

4. 로그공간에서 정규화해 $W_t^{(i)}$를 얻는다.
5. 관심 상태·안전확률·예측량을 가중 평균으로 계산한다.
6. ESS, 최대 가중치, 고유 조상 수, 공간 coverage를 기록한다.

조상을 먼저 뽑는 resample–propagate(selection–mutation) convention과 상태를 전파·가중한 뒤 재표집하는 propagate–weight–resample convention이 모두 쓰인다. 구현을 비교할 때는 순서와 가중치 초기화가 일치하는지 확인해야 한다. 별도의 **auxiliary particle filter**는 최신 관측을 반영한 look-ahead 또는 first-stage weight로 조상을 고르는 방법이므로 단순히 “먼저 재표집한다”는 뜻과 구분한다.

### 3.9 손계산 예제: 관측 하나가 particle 믿음을 바꾸는 과정

1차원 위치 particle 세 개가

$$
X_t^{(1)}=-1,\qquad X_t^{(2)}=0,\qquad X_t^{(3)}=2
$$

이고 재표집 직후라 이전 가중치가 모두 $1/3$이라고 하자. 센서 관측 $y_t=0.2$이고

$$
Y_t\mid X_t=x
\sim
\mathcal N(x,1)
$$

이라면 공통상수를 제외한 likelihood는

$$
g_t(0.2\mid x)
\propto
\exp\left(-\frac{(0.2-x)^2}{2}\right).
$$

각 값은 대략

$$
(0.487,\;0.980,\;0.198)
$$

이므로 정규화 가중치는

$$
W_t
\approx
(0.292,\;0.589,\;0.119).
$$

posterior mean 근사는

$$
\widehat{\mathbb E}[X_t\mid y_{1:t}]
=
(-1)(0.292)+(0)(0.589)+(2)(0.119)
\approx
-0.054
$$

이다. 가장 가까운 particle 하나만 고르는 것이 아니라 모든 가설을 가중 평균한다.

ESS는

$$
\widehat{\operatorname{ESS}}
=
\frac{1}{0.292^2+0.589^2+0.119^2}
\approx
2.24.
$$

$N=3$에서 임계값 $\tau N=1.5$라면 아직 재표집하지 않을 수 있다. 임계값이 $2.4$라면 재표집한다. 임계값은 수학적으로 자동 결정된 진리가 아니라 운영 설계다.

### 3.10 Weight degeneracy와 particle impoverishment를 구분하기

| 문제 | 무엇이 무너졌는가? | 대표 진단 | 대표 대응 |
| --- | --- | --- | --- |
| weight degeneracy | 가중치 질량이 소수 particle에 몰림 | ESS, 최대 $W_i$, weight entropy | 더 나은 proposal, 적응 재표집, particle 증가 |
| particle impoverishment | 재표집 후 위치 다양성이 사라짐 | 고유 particle 수, 공분산·coverage | Target-invariant MCMC move, particle 증가, 전이모형 재검토; heuristic roughening은 bias를 따로 평가 |
| path degeneracy | 현재 particle의 먼 과거 조상이 소수로 합쳐짐 | 시점별 고유 ancestor 수 | fixed-lag smoothing, backward simulation, particle MCMC |
| model misspecification | 실제 상태가 transition·likelihood가 허용하는 범위 밖 | predictive residual, posterior predictive check | 모형 자체 수정, robust likelihood, 새 상태 가설 |

Particle 수를 늘리는 것만으로 잘못된 transition이나 likelihood를 고칠 수는 없다. 틀린 지도를 들고 정찰대를 늘리면 더 정밀하게 틀릴 수 있다.

### 3.11 Influence function의 파라미터 영향 유도

학습 데이터 $\mathcal D=\{z_1,\ldots,z_n\}$의 경험위험을

$$
R(\theta)
=
\frac1n\sum_{j=1}^n\ell(z_j,\theta)
$$

라 하고, 국소 최적점을

$$
\widehat\theta
=
\arg\min_\theta R(\theta)
$$

라 하자. 사례 $z_i$를 $\varepsilon$만큼 추가 가중한 목적은

$$
R_{\varepsilon,i}(\theta)
=
R(\theta)+\varepsilon\ell(z_i,\theta)
$$

이다. 그 최적점을 $\widehat\theta_{\varepsilon,i}$라 하면 1차 조건은

$$
\nabla_\theta R(\widehat\theta_{\varepsilon,i})
+\varepsilon\nabla_\theta\ell(z_i,\widehat\theta_{\varepsilon,i})
=0.
$$

$\varepsilon$로 미분하고 $\varepsilon=0$을 대입하면

$$
H_{\widehat\theta}
\left.
\frac{d\widehat\theta_{\varepsilon,i}}{d\varepsilon}
\right|_{\varepsilon=0}
+
\nabla_\theta\ell(z_i,\widehat\theta)
=0,
$$

여기서

$$
H_{\widehat\theta}
=
\nabla_\theta^2R(\widehat\theta).
$$

$H_{\widehat\theta}$가 가역이라고 가정하면

$$
\left.
\frac{d\widehat\theta_{\varepsilon,i}}{d\varepsilon}
\right|_{\varepsilon=0}
=
-H_{\widehat\theta}^{-1}
\nabla_\theta\ell(z_i,\widehat\theta).
$$

사례 하나 제거는 1차적으로 $\varepsilon=-1/n$의 downweighting에 대응하므로

$$
\widehat\theta_{-i}-\widehat\theta
\approx
\frac1n
H_{\widehat\theta}^{-1}
\nabla_\theta\ell(z_i,\widehat\theta).
$$

정규화 항이 없는 순수 경험위험에서는 $n^{-1}\sum_{j\ne i}\ell_j$와 $(n-1)^{-1}\sum_{j\ne i}\ell_j$가 양의 상수배 관계라 최적점이 같다. 따라서 $\varepsilon=-1/n$은 leave-one-out 목적에 정확히 대응하고, 근사 오차는 $\widehat\theta_{\varepsilon,i}$를 $\varepsilon=0$ 주변에서 1차 Taylor 전개한 데서 생긴다. 고정 계수 regularizer가 있다면 데이터 평균의 재정규화가 regularizer와의 상대 비중을 바꾸므로 어떤 convention을 유지하는지 따로 명시해야 한다.

### 3.12 학습 사례가 검증 손실에 미치는 영향

관심 평가점 $z_{\mathrm{test}}$의 손실 변화는 Taylor 전개로

$$
\ell(z_{\mathrm{test}},\widehat\theta_{-i})
-\ell(z_{\mathrm{test}},\widehat\theta)
\approx
\nabla_\theta\ell(z_{\mathrm{test}},\widehat\theta)^\top
(\widehat\theta_{-i}-\widehat\theta).
$$

따라서

$$
\Delta_{-i}\ell_{\mathrm{test}}
\approx
\frac1n
\nabla_\theta\ell(z_{\mathrm{test}},\widehat\theta)^\top
H_{\widehat\theta}^{-1}
\nabla_\theta\ell(z_i,\widehat\theta).
$$

양수면 사례 $i$를 제거했을 때 test loss가 늘어날 것으로 예측하므로 그 사례가 해당 test point에 도움을 줬다는 뜻이다. 부호 해석은 “upweight 영향”을 보고하는지 “removal 영향”을 보고하는지에 따라 뒤집힐 수 있으므로 구현 문서에 convention을 고정해야 한다.

대형 모델에서는 $H^{-1}$를 직접 만들지 않는다. 선형계

$$
(H+\lambda I)s
=
\nabla_\theta\ell(z_{\mathrm{test}},\widehat\theta)
$$

를 풀어 $s$를 구하면 single-example removal에 따른 test-loss 변화는

$$
\Delta_{-i}\ell_{\mathrm{test}}
\approx
\frac1n
s^\top\nabla_\theta\ell(z_i,\widehat\theta)
$$

이다. 반면 unit upweight 영향은 $-s^\top\nabla_\theta\ell(z_i,\widehat\theta)$이므로 부호와 $1/n$을 생략하면 다른 양이 된다. **Hessian-vector product**는 거대한 $H$ 전체를 저장하지 않고 주어진 벡터 $v$에 대한 $Hv$만 계산하는 연산이고, **conjugate gradient**는 이런 곱셈을 반복해 선형계의 해를 찾는 알고리즘이다. 표준 conjugate gradient는 대칭 양의 정부호 행렬을 가정한다. 비볼록 신경망에서는 $H+\lambda I$도 $\lambda$가 모든 음의 곡률을 누르지 못하면 그 조건을 만족하지 않을 수 있으므로, 충분한 damping·양의 준정부호 Gauss–Newton/Fisher 근사 또는 부정부호 행렬을 다루는 다른 solver를 명시해야 한다. Damping과 곡률 근사는 수치 안정성을 높이지만 원래 influence estimand를 바꾼다.

### 3.13 왜 deep network에서 influence를 조심해야 할까?

유도에는 다음 가정이 숨어 있다.

1. $\widehat\theta$가 1차 조건을 거의 만족한다.
2. Hessian 역작용이 의미 있을 만큼 국소 곡률이 안정적이다.
3. 데이터 제거가 작은 교란이다.
4. 같은 학습 recipe와 최적점 basin, 즉 최적화가 수렴하는 하나의 손실 골짜리 영역을 유지한다.
5. dropout·data order·optimizer state 같은 확률성이 무시되거나 통제된다.

대형 비볼록 모델은 움직여도 손실이 거의 변하지 않는 **flat direction**, 아래가 아니라 위로 휘는 **음의 곡률**, 서로 다른 여러 **basin**을 갖는다. Fine-tuning 한 사례의 삭제가 optimizer 경로를 바꾸면 국소 1차 근사가 실제 재학습 결과와 다를 수 있다. 따라서 작은 subset에서 실제 leave-one-out 또는 leave-group-out 재학습과 순위상관·부호 정확도를 검증해야 한다.

### 3.14 TracIn과 checkpoint 근사인 TracInCP

이상적인 TracIn은 사례 $z_i$가 실제로 사용된 학습 step에서 그 update가 test loss를 얼마나 바꾸었는지 추적한다. 모든 step을 보관하기 어렵기 때문에 실무의 TracInCP는 저장한 checkpoint $c$에서 train gradient와 test gradient가 얼마나 같은 방향이었는지를 다음처럼 누적한다.

$$
\operatorname{TracInCP}(z_i,z_{\mathrm{test}})
\approx
\sum_{c\in\mathcal C}
\eta_c
\nabla_\theta\ell(z_i,\theta_c)^\top
\nabla_\theta\ell(z_{\mathrm{test}},\theta_c).
$$

$\eta_c$는 해당 구간의 학습률, $\mathcal C$는 저장한 checkpoint 집합이다. 한 gradient-descent step $\theta^+=\theta-\eta g_i$에 대해

$$
\ell_{\mathrm{test}}(\theta^+)-\ell_{\mathrm{test}}(\theta)
\approx
-\eta
\nabla\ell_{\mathrm{test}}^\top
\nabla\ell_i.
$$

따라서 내적이 양수면 그 update가 test loss를 낮출 가능성이 있다.

장점은 inverse Hessian을 피하면서 학습 경로의 여러 위치를 일부 반영한다는 것이다. 단점은 TracInCP가 실제 update step을 모두 추적하지 않고 저장 checkpoint에서의 가상 영향으로 대신하며, checkpoint 간격·batch 구성·parameter subset·gradient normalization에 따라 값이 크게 달라진다는 것이다. 이를 “진짜 인과효과”로 읽으면 안 된다.

### 3.15 Shapley 값의 정확한 정의

전체 데이터 인덱스 집합을 $D=\{1,\ldots,n\}$라 하고, 부분집합 $S$로 학습한 모델의 효용을 $v(S)$라 하자. 학습이 확률적이면 먼저

$$
v(S)
=
\mathbb E_{\xi}
\left[
\operatorname{Utility}\bigl(\mathcal A(S;\xi)\bigr)
\right]
$$

처럼 학습 seed·data order·환경 $\xi$에 대한 기대효용으로 정의하거나, 모든 coalition에 공통 random seed를 고정한 조건부 가치를 정의해야 한다. 이 정의를 고정했을 때 사례 $i$의 Shapley 값은

$$
\phi_i
=
\sum_{S\subseteq D\setminus\{i\}}
\frac{|S|!(n-|S|-1)!}{n!}
\left[
v(S\cup\{i\})-v(S)
\right].
$$

동일한 식을 무작위 순열 $\pi$로도 표현할 수 있다.

$$
\phi_i
=
\mathbb E_{\pi}
\left[
v(P_i^\pi\cup\{i\})
-v(P_i^\pi)
\right],
$$

$P_i^\pi$는 순열 $\pi$에서 $i$보다 먼저 온 사례 집합이다.

Shapley 값은 다음 성질을 만족한다.

- **효율성:** $\sum_i\phi_i=v(D)-v(\varnothing)$
- **대칭성:** 모든 coalition에서 같은 기여를 하는 두 사례는 같은 값을 받는다.
- **dummy:** 어느 coalition에도 한계기여가 $0$인 사례의 값은 $0$이다.
- **가법성:** 두 가치함수를 더하면 Shapley 값도 더해진다.

하지만 $v(S)$마다 모델을 다시 학습하면 $2^n$에 가까운 비용이 든다. $K$개 순열만 뽑는 Monte Carlo 근사에서는

$$
\widehat\phi_i
=
\frac1K
\sum_{k=1}^K
\left[
v(P_i^{\pi_k}\cup\{i\})
-v(P_i^{\pi_k})
\right]
$$

를 쓴다. Permutation Monte Carlo 오차와 학습 seed 변동은 서로 다른 불확실성이므로 nested 반복이나 공통 random number 설계로 둘 다 추정하고, 표준오차·중지규칙을 기록해야 한다. 데이터 포인트 대신 사용자·문서·중복 클러스터를 player로 잡는 group Shapley가 삭제 요청 단위에 더 맞을 수 있다.

### 3.16 Influence와 Shapley의 1:1 비교

| 축 | Influence function | Data Shapley |
| --- | --- | --- |
| 질문 | 현재 최적점에서 사례 가중치를 조금 바꾸면? | 모든 학습 조합에서 사례가 평균적으로 얼마나 기여했나? |
| 근사 범위 | 국소·미분 기반 | 조합·상호작용 기반 |
| 핵심 비용 | inverse-Hessian vector product | 반복 재학습·효용 평가 |
| 장점 | 특정 test point 영향 방향을 빠르게 추정 | 대체·상호작용을 가치정의에 반영 |
| 주요 실패 | 비볼록성·큰 삭제·곡률 불안정 | 지수적 비용·학습 확률성·가치함수 의존 |
| 삭제 역할 | 우선순위·영향 후보·sanity check | 사용자·cluster별 가치·영향 감사 |
| 할 수 없는 것 | 삭제 완료 보증 | 삭제 완료 보증 |

### 3.17 SMC와 영향 추정의 공통 수학, 그리고 차이

두 트랙 모두 가중 합을 쓴다.

$$
\text{SMC: }
\sum_iW_t^{(i)}\varphi(X_t^{(i)})
\qquad
\text{Shapley MC: }
\frac1K\sum_k\Delta_i(\pi_k).
$$

그러나 첫 번째 무작위성은 상태 posterior 적분을 근사하고, 두 번째는 coalition 순열 평균을 근사한다. ESS 공식을 Shapley 표본에 그대로 붙이거나, influence 점수를 posterior 확률처럼 해석하면 안 된다. 수학적 모양이 비슷해도 estimand가 다르다.

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Replay 사례의 영향 추정 파이프라인

지속학습 시스템에서 influence 점수 하나를 계산하는 것보다 먼저 단위를 고정해야 한다.

1. **요청 단위:** 한 row, 한 사용자, 한 문서, 한 영상, 한 trajectory 중 무엇인가?
2. **파생 계보:** crop·번역·요약·합성·embedding·hard negative까지 같은 원천에서 나왔는가?
3. **영향 대상:** 전체 validation loss인가, 특정 안전군·언어·행동인가?
4. **학습 범위:** base pretraining, adapter, reward model, replay fine-tuning 중 어디에 들어갔는가?
5. **기준 recipe:** seed, data order, optimizer, tokenizer, preprocessing을 어디까지 고정하는가?

그 뒤 세 층으로 추정한다.

- **빠른 선별:** gradient norm·similarity, nearest neighbor, duplicate lineage로 후보를 좁힌다.
- **중간 비용:** influence function, TracIn, group Shapley 근사를 계산한다.
- **비싼 검증:** 상위·하위 영향 그룹 일부를 실제 leave-group-out 재학습해 근사 순위와 부호를 확인한다.

영향값의 절대 크기는 layer·loss scale·damping에 따라 달라진다. “상위 1%” 같은 순위도 seed별 안정성과 confidence interval 없이 고정 진실로 취급하지 않는다.

### 4.2 Influence를 replay 선택에 쓰는 법

Replay budget이 $M$일 때 다음 세 목적은 충돌할 수 있다.

$$
\underbrace{\text{현재 utility 보존}}_{\text{도움이 큰 사례}}
\qquad
\underbrace{\text{희귀 안전군 coverage}}_{\text{확률은 작아도 필수}}
\qquad
\underbrace{\text{삭제 가능성·개인정보 최소화}}_{\text{보존 비용이 큰 사례 회피}}.
$$

검증 사례 $u$에 대한 removal influence를 $I_{i,u}$라 하고 위험군 가중치 $\alpha_u$를 두면 단순 점수는

$$
s_i
=
\sum_{u\in\mathcal V}
\alpha_u I_{i,u}
$$

처럼 만들 수 있다. 하지만 이 점수만으로 top-$M$을 고르면 비슷한 사례가 중복되고 희귀군이 빠질 수 있다. 실제 선택은 quota·coverage·diversity와 함께 constrained optimization으로 설계해야 한다.

또한 influence가 음수인 사례가 무조건 “나쁜 데이터”라는 뜻은 아니다. noisy label일 수도 있지만, test set의 편향·distribution mismatch·regularization 효과 때문일 수도 있다. 원본 검토와 그룹별 영향을 병행한다.

### 4.3 Machine unlearning의 세 수준

#### 수준 A: 데이터 계층 삭제

원본 저장소, replay buffer, feature store, vector index, cache, 학습 manifest에서 대상과 파생물을 제거한다. 모델 가중치는 아직 바뀌지 않았다.

#### 수준 B: 모델 계층 unlearning

- **Full retraining:** forget set 없이 처음부터 다시 학습한다. 가장 명확한 reference지만 비싸다.
- **Partitioned retraining:** shard·slice를 분리해 영향받은 부분만 다시 학습하고 조합한다.
- **Checkpoint rollback and replay:** 대상이 들어오기 전 checkpoint로 돌아가 이후 허용 데이터만 재적용한다.
- **Approximate update:** gradient ascent, negative preference, retain regularization, adapter 재학습, influence-based Newton update 등을 쓴다.

#### 수준 C: 시스템 계층 정리

Serving replica, quantized export, distillation student, RAG index, prompt cache, safety classifier, backup과 downstream model까지 교체하거나 무효화한다. 모델 하나만 고쳐도 오래된 artifact가 요청에 답하면 삭제 목표는 실패한다.

### 4.4 Exact와 approximate의 경계를 말로 흐리지 않기

먼저 학습 알고리즘 $\mathcal A$와 unlearning 알고리즘 $\mathcal U$의 **확률분포**를 정의해야 한다. 학습 seed·data order·환경 난수 $\xi$와 unlearning 난수 $\zeta$를 포함해

$$
\Theta_{-f}
=
\mathcal A(\mathcal D_r;\xi),
\qquad
\widetilde\Theta
=
\mathcal U\left(
\mathcal A(\mathcal D;\xi),
\mathcal D_f,
\mathcal D_r;
\zeta
\right)
$$

라 하자. **Exact unlearning**의 강한 정의는

$$
\operatorname{Law}(\widetilde\Theta)
=
\operatorname{Law}(\Theta_{-f})
$$

이다. 즉 두 결과를 관찰하는 어떤 검사도 어느 절차에서 나왔는지 구분할 수 없어야 한다. 결정론적 알고리즘이라면 동일 결과로 단순화되지만, 확률적 deep learning에서는 seed 하나의 weight 일치와 분포 동일성을 구분해야 한다.

Approximate 또는 certified removal은 선택한 정의 아래 두 분포가 허용 오차 안에 있음을 보인다. 예를 들어 모든 사건 $E$에 대해 양방향으로

$$
\Pr(\widetilde\Theta\in E)
\le
e^\varepsilon
\Pr(\Theta_{-f}\in E)
+\delta
$$

와 두 분포를 바꾼 대칭 부등식을 요구할 수 있다. 정확한 보증 정의는 방법마다 다르므로 $(\varepsilon,\delta)$, 난수 범위, 반복 삭제 조건을 함께 밝혀야 한다.

실무 감사에서는 동일 seed나 여러 paired seed로 얻은 구체적 draw를

$$
\theta_{-f}\sim\Theta_{-f},
\qquad
\widetilde\theta\sim\widetilde\Theta
$$

라 두고 다음 경험적 거리를 진단한다.

$$
d_{\mathrm{output}}
\left(
p_{\widetilde\theta}(\cdot\mid x),
p_{\theta_{-f}}(\cdot\mid x)
\right),
$$

$$
\Delta_{\mathrm{retain}}
=
R_r(\widetilde\theta)-R_r(\theta_{-f}),
\qquad
\Delta_{\mathrm{forget}}
=
R_f(\widetilde\theta)-R_f(\theta_{-f}).
$$

여기서 forget loss가 커졌다는 사실만으로 충분하지 않다. 무작위로 모델을 망가뜨려도 forget loss는 커진다. Retain utility와 reference closeness를 함께 봐야 한다.

이 출력·손실 검사는 중요한 경험적 증거지만, 그 자체로 위의 모든 사건에 대한 확률분포 보증을 증명하지는 않는다.

### 4.5 삭제 검증의 네 축

| 축 | 핵심 질문 | 대표 검사 | 단독으로 부족한 이유 |
| --- | --- | --- | --- |
| Forget efficacy | 직접·우회 질의에서 대상 정보가 줄었나? | exact prompt, paraphrase, multilingual, image crop, decoding sweep | refusal이나 prompt filter가 숨길 수 있다. |
| Retain utility | 무관한 능력을 보존했나? | task·group별 성능, calibration, latency | 평균만 보면 희귀군 붕괴를 숨긴다. |
| Privacy·extraction | 공격자가 membership·내용을 복원할 수 있나? | membership inference, canary exposure, extraction red team | 공격 실패는 모든 공격에 안전하다는 증명이 아니다. |
| Stability·relearning | 후속 update에서 candidate가 reference보다 비정상적으로 빨리 대상을 회복하나? | 동일 protocol의 small-data relearning, adapter attach, quantization·distillation 후 재검사 | 빠른 학습은 공유 특징과 일반 plasticity 때문일 수도 있어 reference·control 비교가 필요하다. |

가능하면 unlearned model만 보지 말고 세 모델을 함께 비교한다.

1. 원래 모델 $\widehat\theta$
2. forget set 없이 재학습한 reference $\theta_{-f}$
3. unlearned candidate $\widetilde\theta$

Candidate가 원래 모델보다 덜 기억하는 것보다, reference와 얼마나 비슷한지가 더 직접적인 질문이다.

### 4.6 Membership inference는 왜 삭제 증명서가 아닌가?

Membership attack의 정확도가 chance level로 내려갔다고 하자. 가능한 해석은 여러 개다.

- 실제로 대상 영향이 줄었다.
- 공격이 약하거나 잘못 검량됐다.
- 원래 모델도 해당 사례를 구분하지 못했다.
- 출력 온도·refusal policy가 공격 신호만 가렸다.
- 다른 우회 프롬프트나 내부 activation에는 흔적이 남았다.

따라서 공격을 사전 등록하고, 원본·retrain reference·candidate 모두에서 공격의 sensitivity를 확인해야 한다. 여러 공격을 시도하고 가장 좋은 결과만 보고하면 평가자 선택 편향이 생기므로 공격군과 오류예산도 고정한다.

### 4.7 삭제 요청의 운영 수명주기

#### 1단계: 요청 해석과 보존

- 요청자 권한과 대상 ID를 확인한다.
- 원본, 파생물, 중복, 사용자 단위, 법적 보존 예외를 구분한다.
- 조사용 immutable manifest hash를 남기되 삭제 대상 내용을 불필요하게 복제하지 않는다.

#### 2단계: 계보 탐색

$$
\text{source}
\rightarrow
\text{preprocess}
\rightarrow
\text{shard}
\rightarrow
\text{checkpoint}
\rightarrow
\text{adapter·student·index}
\rightarrow
\text{deployment}
$$

각 edge에 version과 digest를 기록한다. 계보를 모르면 영향 추정의 모집단도 정의할 수 없다.

#### 3단계: 격리와 후보 생성

- source·derivative를 향후 학습과 replay에서 차단한다.
- 정확 재학습이 가능한지 먼저 판단한다.
- 불가능하거나 과도하게 비싸면 approximate method와 허용 오차를 사전 등록한다.

#### 4단계: 독립 검증

- unlearning tuning에 쓰지 않은 final forget probes를 사용한다.
- retain·group safety·privacy·relearning을 함께 측정한다.
- seed와 decoding variation을 반복한다.
- candidate 선택 데이터와 final approval 데이터를 분리한다.

#### 5단계: 배포·무효화·감시

- quantized model, cache, RAG index, replica를 원자적으로 교체한다.
- 오래된 artifact의 digest를 denylist에 넣는다.
- 다음 continual-learning cycle 뒤 recurrence probe를 다시 실행한다.
- 증거가 지지하는 범위만 certificate 또는 audit report에 쓴다.

### 4.8 Particle filter와 AI 시스템의 연결

Particle filter는 로봇·에이전트에서 다음처럼 쓰일 수 있다.

- 여러 가능한 객체 위치·접촉 상태를 유지한다.
- VLM 관측 likelihood로 particle을 재가중한다.
- posterior mean뿐 아니라 충돌영역 확률을 계산한다.
- 위험확률의 upper bound가 임계값을 넘으면 safe policy로 전환한다.

그러나 VLM 점수를 likelihood처럼 쓰려면 검량이 필요하다. Softmax confidence는 센서모형 $p(y_t\mid x_t)$와 자동으로 같지 않다. 같은 관측 $y_t$로 proposal $q_t$를 만들고 likelihood를 계산하는 것 자체는 표준적인 observation-conditioned proposal이며, 일반 가중치의 $f_tg_t/q_t$ 보정을 정확히 하면 문제없다. 위험한 경우는 그 $q_t$ 보정을 빠뜨리거나, 검량되지 않은 discriminative score를 $p(y_t\mid x_t)$로 간주하거나, 상관된 모델 오류를 독립 증거처럼 반복 곱하는 것이다. 시뮬레이터의 생성 프레임을 실제 관측처럼 반복 사용하면 sim-to-real 편향이 posterior를 과신하게 만들 수 있다.

### 4.9 수학과 AI 부품의 1:1 매핑

| 수학 부품 | AI 시스템 부품 | 실패 시 증상 |
| --- | --- | --- |
| transition $f_t$ | 환경·사용자·로봇 동역학 | 실제 변화가 particle support 밖으로 나감 |
| likelihood $g_t$ | 센서·VLM 관측 점수 | 그럴듯한 오인식에 weight가 몰림 |
| proposal $q_t$ | candidate state generator | 관측 근처 particle을 거의 못 만듦 |
| normalized weight $W_i$ | 각 세계가 현재 믿을 만한 정도 | 한 가설을 확정값으로 오해 |
| ESS·ancestor diversity | 불확실성 품질 진단 | 계산량은 많지만 실질 가설은 하나 |
| removal influence $\frac1n(H+\lambda I)^{-1}g_i$ | replay 사례 하나를 뺄 때의 국소 파라미터 효과 근사 | 큰 삭제·경로 변화·곡률 근사에서 오판 |
| Shapley marginal contribution | 조합 속 데이터 가치 | 가치함수·재학습 seed에 따라 흔들림 |
| retrain reference | 삭제하지 않았을 세계의 기준 | “잊음”을 단순 성능 저하와 혼동 |
| independent gate | 배포 승인 증거 | 선택·튜닝과 평가 재사용으로 과적합 |

### 4.10 초보자가 자주 하는 오해

1. **Particle 수가 많으면 posterior가 자동으로 정확하다.**
   Proposal과 모형이 나쁘면 많은 particle도 틀린 곳에 몰린다.

2. **재표집은 particle을 새로 발견한다.**
   재표집은 기존 particle을 복제·삭제할 뿐이다. 다양성은 transition이나 move step에서 와야 한다.

3. **ESS $=N/2$는 절대적인 경계다.**
   편리한 휴리스틱일 뿐이며 목표 함수와 비용에 맞춰 검량해야 한다.

4. **Influence 값이 큰 사례가 인과적으로 모델을 만들었다.**
   현재 최적점과 정한 loss 아래의 국소 근사다.

5. **Shapley 값은 데이터의 객관적 가격이다.**
   $v(S)$, 학습 recipe, player 단위가 바뀌면 값도 바뀐다.

6. **삭제 대상에 오답을 내면 unlearning이 끝났다.**
   Refusal·surface suppression일 수 있다. 우회·공격·reference 비교가 필요하다.

7. **Membership attack이 실패했으니 삭제가 증명됐다.**
   특정 공격의 검출력이 부족할 수 있다.

8. **가중치만 고치면 전체 시스템에서 지워졌다.**
   Index, cache, adapter, checkpoint, replica, 파생 데이터가 남을 수 있다.

9. **Approximate unlearning을 exact deletion이라고 부를 수 있다.**
   보증이 없다면 방법·오차·검증 범위를 명확히 적어야 한다.

10. **삭제와 안전한 배포 승인은 같은 단계다.**
    삭제 candidate도 retain regression·latency·group safety를 통과해야 한다.

### 4.11 실전 최소 체크리스트

- Particle filter에서는 seed, particle 수, proposal, likelihood version, ESS threshold, resampling method, 고유 조상 수를 기록한다.
- Influence에서는 loss, parameter subset, Hessian approximation, damping, checkpoint, sign convention을 기록한다.
- Shapley에서는 player 단위, value function, permutation 수, seed, Monte Carlo 표준오차를 기록한다.
- Unlearning에서는 source-to-artifact lineage, forget·retain split, retrain reference, 공격군, 후속 변환과 recurrence test를 기록한다.
- 어떤 경우에도 모델 자체가 만든 confidence를 독립 검증 결과로 재사용하지 않는다.

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **기준 시각:** 2026-10-23 KST. 아래 내용은 NVIDIA·IBM의 공식 발표와 공식 모델 카드를 우선 대조했다. 성능 수치는 별도 표기가 없으면 공급사 보고치이며, 실제 환경의 독립 재현 결과가 아니다.

### 5.1 NVIDIA Cosmos3-H-Dreams: 수술 장면을 행동 조건부로 생성하는 실시간 world model

NVIDIA는 2026년 10월 20일 Cosmos3-H-Dreams를 공개했다. 초기 수술 영상 한 장과 양팔 로봇의 미래 행동 벡터를 받아, 그 행동을 따르는 담낭절제술 장면의 미래 영상을 스트리밍 생성하는 연구용 world model이다.

핵심 구조와 입출력은 다음과 같다.

| 항목 | 공식 공개 내용 | 읽을 때 주의점 |
| --- | --- | --- |
| 기반 | Cosmos3-H-Embodied-Nano를 수술 데이터에 특화 | 그 기반은 Cosmos3-Nano 계열이다. |
| 구조 | Qwen3-VL-8B-Instruct reasoning tower와 latent-diffusion expert를 공유 attention으로 묶은 Mixture-of-Transformers 계열 | 언어 reasoning과 물리 정확도가 자동으로 동일하다는 뜻은 아니다. |
| 규모 | 약 $15.17$B parameters | 활성 연산량·메모리는 runtime 설정과 함께 봐야 한다. |
| 조건 | 초기 RGB frame, 양팔 합계 $28$차원 action vector, 선택적 과거 action $0$–$16$개 | 각 팔은 평행이동 $3$·6D 회전 $6$·jaw $1$·energy $1$·thumbstick button $1$·axes $2$의 $14$차원이며, 과거 action은 실제로 연속한 기록이어야 한다. |
| 해상도·시간 | 기본 $832\times480$, action과 출력 영상의 시간축은 모두 $10$ Hz, 기본 미래 $72$ frames 즉 $7.2$초 | $10$ Hz의 모형 시간축과 아래 wall-clock inference FPS 처리량은 다른 양이다. |
| streaming | 매 다음 $4$ action과 함께 $4$ frames, 즉 $0.4$초 블록 생성 | 자기 생성 history의 오차 누적이 핵심 위험이다. |
| distillation | bidirectional teacher에서 block-causal 4-step student, 이어 그 student를 2-step schedule로 추가 학습한 별도 checkpoint | 처음부터 독립 학습한 모델도, 4-step 모델을 단순히 2-step으로 실행한 것도 아니다. |
| 장기 history | FlashDreams runtime과 변형 FadeMem으로 KV cache를 압축·관리 | cache 효율과 세계상태 정확도는 별도 문제다. |

공식 모델 카드의 단일 장비 측정은 다음과 같다.

- RTX PRO 6000 Blackwell, $832\times480$: 4-step $30.0$ FPS, 2-step $43.2$ FPS
- RTX PRO 6000 Blackwell Max-Q, 같은 해상도: 4-step $20.2$ FPS, 2-step $31.2$ FPS
- DGX Spark, 더 낮은 $512\times288$: 2-step $16.9$ FPS

서로 다른 해상도·하드웨어 값을 같은 열에서 “모델 자체 속도”로 단순 비교하면 안 된다. 2-step은 더 빠르지만 공식 글도 시각 품질 trade-off를 명시한다.

임상 특화 단계는 CMR의 proprietary clinical cholecystectomy video와 robot kinematics를 사용했다. 모델 카드는 원천 $4{,}792$ episodes·약 $17$M frames에서 선택한 $64$개 training recordings·약 $526$ unique minutes를 보고하지만, 이 임상 fine-tuning·distillation 데이터는 공개하지 않았다. Public Open-H-Embodiment는 직접 기반 checkpoint의 초기화 단계에 쓰인 데이터이므로 두 범위를 섞어 “전체 수술 학습 데이터가 공개됐다”고 말하면 안 된다.

Release는 source code·4-step/2-step checkpoint·teacher fine-tuning과 Self-Forcing distillation recipe·FlashDreams runtime을 연결한다. **Source-code repository는 Apache-2.0**이지만, **Cosmos3-H-Dreams checkpoint/model은 NVIDIA Open Model License Agreement** 적용을 받으며 그 밖의 다운로드 asset은 각각의 license terms를 유지한다. “코드가 Apache-2.0”과 “모델이 Apache-2.0”은 같은 말이 아니다.

가장 중요한 안전 경계는 공급사가 직접 쓴 문장에 있다. 이 모델은 research simulator이고 환자 치료에 검증된 임상 시스템이 아니다. 시각적으로 그럴듯한 tissue deformation은 생체역학적 정확도의 증명이 아니며, 공개 데모에서 보지 못한 행동을 그럴듯하게 생성했다고 해서 실제 수술 결과를 정확히 예측한다고 결론낼 수 없다.

**엔지니어 인사이트 (Impact):** 생성 world model은 정책 학습용 데이터를 크게 늘릴 수 있지만 “한 개의 그럴듯한 미래”보다 가능한 미래의 분포가 중요하다. 오늘 배운 particle 관점으로 보면 하나의 action sequence에 대해 서로 다른 seed·model ensemble·dynamics hypothesis를 particle처럼 유지하고, 실제 sensor evidence로 재가중하며, tissue-contact 위험확률과 ancestor diversity를 감시해야 한다. 단, 생성모델의 likelihood나 critic score가 검량된 실제세계 확률은 아니다. Real clinical data에서 control fidelity·anatomical consistency·장기 rollout drift·전문의 평가를 별도로 검증하고, simulator에서 통과한 policy도 실제 시스템의 shadow·safe-stop gate를 거쳐야 한다.

- 공식 발표: [NVIDIA Cosmos3-H-Dreams](https://huggingface.co/blog/nvidia/cosmos3-h-dreams)
- 공식 모델 카드: [nvidia/Cosmos3-H-Dreams](https://huggingface.co/nvidia/Cosmos3-H-Dreams)
- 코드와 recipe: [isaac-for-healthcare/Cosmos3-H-Dreams](https://github.com/isaac-for-healthcare/Cosmos3-H-Dreams)
- streaming runtime: [FlashDreams](https://github.com/NVIDIA/FlashDreams)

### 5.2 IBM Granite Speech 5.0: 정확도·엣지 처리량·화자 귀속을 역할별로 나눈 오픈 음성 모델군

IBM은 2026년 10월 21일 세 가지 음성 모델 업데이트를 발표했다.

| 모델 | 규모·역할 | 핵심 변화 |
| --- | --- | --- |
| Granite Speech 5.0 2B | 고정확도 ASR·음성번역 | 다국어 ASR, 영어 중심 양방향 번역, 언어 ID prompt, 최대 $2{,}000$개 keyword bias |
| Granite Speech 5.0 470M Turbo CTC | 고처리량·on-device ASR | 7개 언어, native punctuation·capitalization, noise·reverb robustness |
| Granite Speech 4.1 2B Plus | 회의·콜·인터뷰 | speaker attribution, timestamp, long-form, keyword bias 개선 |

새 2B 모델은 공개 corpora의 $390{,}000$시간 음성과 일본어 ASR·keyword bias·번역용 synthetic data로 학습됐다고 모델 카드가 설명한다. 영어·프랑스어·독일어·스페인어·포르투갈어·일본어의 ASR, 이 언어들과 영어 사이의 번역, 영어에서 이탈리아어·중국어(만다린) 번역을 지원한다. Apache-2.0이며 Transformers, vLLM, llama.cpp, Apple Silicon용 mlx-audio 경로가 공개됐다.

IBM 발표 기준 2B 모델은 $9$개 public English short-form test set의 평균 WER $5.06\%$로 2026년 10월 21일 당시 Open ASR Leaderboard의 open-weight 모델 중 2위다. 이 순위는 해당 날짜의 snapshot이다. WER는

$$
\operatorname{WER}
=
\frac{S+D+I}{N}
$$

으로, 정답 단어 $N$개에 대한 치환 $S$, 삭제 $D$, 삽입 $I$의 비율이다. 낮을수록 좋지만 번역 품질, 화자 귀속, 개인정보 안전을 한 숫자로 대변하지 않는다.

470M Turbo CTC는 하나의 Conformer block을 compile해 stack에서 graph를 재사용하는 방식으로 공급사 측 Open ASR Leaderboard 처리량을 약 $12{,}600$ RTFx에서 $19{,}870$ RTFx로 높였다고 보고했다. 이는 약

$$
\frac{19{,}870}{12{,}600}
\approx
1.58
$$

배다. 다만 updated multilingual model card는 $19{,}870$ 지점을 2026년 10월 21일 현재 **unofficial**로 표시하고, 공식 leaderboard entry는 이전 English-only release라고 구분한다. Open ASR 측정은 단일 H200에서 dataset별 가능한 최대 batch, warmup과 허용된 compile·optimization을 사용하는 optimized batched offline throughput이다.

RTFx는 오디오 길이 대비 처리 속도이므로 $1{,}000$ RTFx는 해당 측정 조건에서 1초 동안 1,000초 분량을 처리한다는 뜻이다. 이는 휴대기기·edge의 batch-1 지연시간이나 streaming latency가 아니다. Batch, hardware, warmup, audio length가 다르면 직접 비교할 수 없다.

업데이트된 2B Plus에 대해서는 long-form WER $13.53\%\to10.22\%$, speaker-attributed cpWER $30.50\%\to28.48\%$, 전체 음성 timestamp의 average shift인 AAS $210.86\text{ ms}\to98.54\text{ ms}$ 등이 보고됐다. 같은 표 안에서도 error percentage와 시간 오차의 단위를 구분해야 한다.

**엔지니어 인사이트 (Impact):** 하나의 거대 모델로 모든 음성 작업을 처리하기보다 정확도용, 엣지 처리량용, 다화자 귀속용 모델을 분리한 것이 실무적이다. Apache-2.0 weights와 여러 runtime 지원은 온프레미스 음성 파이프라인의 진입 장벽을 낮춘다. 하지만 공개 학습 데이터라는 표현만으로 모든 utterance의 권리·삭제 계보가 자동 해결되는 것은 아니다. 음성은 한 파일에 여러 화자의 생체적 특징과 개인정보가 섞일 수 있다. 삭제 요청 단위를 segment가 아니라 speaker·recording·파생 transcript·embedding까지 확장하고, 오늘 배운 group influence와 retrain reference로 ASR utility와 대상 화자 extraction을 함께 검증해야 한다. 한국어는 발표된 ASR 지원 언어에 포함되지 않으므로 한국어 제품에 바로 일반화해서도 안 된다.

- 공식 발표: [Granite Speech 5.0](https://huggingface.co/blog/ibm-granite/granite-speech-5)
- 2B 모델 카드: [ibm-granite/granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)
- Turbo CTC 모델 카드: [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
- 2B Plus 모델 카드: [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)
- 처리량 측정 규칙: [Open ASR Leaderboard submission template](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md)

### 5.3 두 트렌드를 오늘의 수학으로 읽기

| 비교 축 | Cosmos3-H-Dreams | Granite Speech 5.0 | 오늘의 교훈 |
| --- | --- | --- | --- |
| 입력 스트림 | frame + robot action | audio waveform + prompt·keyword | 관측 전처리와 시간 정렬이 posterior·출력 품질을 좌우한다. |
| 숨은 변수 | tissue·instrument·접촉의 실제 상태 | 발화 내용·언어·화자·경계 | 한 출력만 내더라도 내부 불확실성은 사라지지 않는다. |
| 공개 가치 | 행동 조건부 video world model과 runtime·recipe | Apache-2.0 음성 모델군과 여러 runtime | weight 공개, code 공개, data 재현성, license는 따로 확인한다. |
| headline 수치 | 특정 Blackwell·DGX 장비의 FPS | leaderboard WER·RTFx와 역할별 error rate | hardware·해상도·dataset·metric 범위를 함께 적는다. |
| 영향·삭제 단위 | 임상 video, 환자·procedure, kinematics, generated derivative | speaker, recording, segment, transcript, embedding | 단일 row 삭제로 계보 전체가 사라지지 않는다. |
| 검증 핵심 | control fidelity·장기 drift·biomechanics·expert review | 언어·소음·화자·domain별 WER와 privacy | 공급사 평균과 실제 배포 안전성을 분리한다. |

두 소식의 공통점은 멀티모달 모델이 텍스트 챗봇을 넘어 고위험 시계열로 이동한다는 것이다. 입력 한 건의 provenance와 삭제 가능성, 시간에 따른 불확실성, downstream artifact 계보가 모델 크기만큼 중요한 설계변수가 된다.

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

재표집 직후 네 particle의 가중치가 모두 $1/4$이고, 새 관측에 대한 likelihood가

$$
g(y_t\mid X_t^{(1:4)})
=
(0.60,\;0.25,\;0.10,\;0.05)
$$

라고 하자.

1. Bootstrap particle filter의 정규화 가중치와 ESS는 얼마인가?
2. 재표집을 하면 posterior 불확실성이 사라졌다고 말할 수 있는가?
3. 학습 사례 $z_i$ 제거의 1차 파라미터 근사가 왜

$$
\widehat\theta_{-i}-\widehat\theta
\approx
\frac1nH^{-1}\nabla_\theta\ell(z_i,\widehat\theta)
$$

의 부호를 갖는가?
4. Influence 점수가 작고 membership inference 공격이 실패했다면 삭제 완료를 선언해도 되는가?
5. 가장 강한 비교 기준과 최소 검증 축은 무엇인가?

### 모범 답안

Bootstrap proposal에서는 재표집 직후의 공통 가중치와 transition/proposal 비가 정규화에서 사라지므로 likelihood를 합 $1$로 만들면 된다. 주어진 값의 합이 이미

$$
0.60+0.25+0.10+0.05=1
$$

이므로

$$
W_t=(0.60,\;0.25,\;0.10,\;0.05)
$$

이다.

ESS는

$$
\widehat{\operatorname{ESS}}
=
\frac{1}{
0.60^2+0.25^2+0.10^2+0.05^2
}
=
\frac{1}{0.435}
\approx
2.30.
$$

$N=4$지만 실질적으로 약 $2.3$개의 균등 가중 표본과 비슷한 집중도를 가진다. 임계값이 $0.5N=2$면 재표집하지 않을 수 있고, $0.75N=3$이면 재표집할 수 있다.

재표집은 불확실성을 없애지 않는다. 현재 가중 경험분포를 복제·삭제를 통해 균등 가중 표본으로 바꾸는 Monte Carlo 단계다. 높은 가중치 particle이 여러 번 복제돼도 그것이 참이라는 증명은 아니며, 오히려 다양성이 줄 수 있다. 이후 transition, move step, 새 관측으로 분포를 계속 추적해야 한다.

Influence의 부호는 upweighting 유도에서 나온다.

$$
\left.
\frac{d\widehat\theta_{\varepsilon,i}}{d\varepsilon}
\right|_0
=
-H^{-1}\nabla\ell_i.
$$

제거는 사례 가중치를 약 $1/n$만큼 낮추는 $\varepsilon=-1/n$의 교란이므로 두 음수가 곱해져

$$
\Delta\theta_{-i}
\approx
\frac1nH^{-1}\nabla\ell_i
$$

가 된다. 순수 경험위험의 재정규화 상수는 최적점을 바꾸지 않는다. 근사 오차는 $\varepsilon=0$ 주변의 1차 선형화와 국소 곡률 가정에서 생기며, fixed regularizer가 있다면 데이터 평균과의 상대 scale convention도 고정해야 한다.

Influence가 작고 하나의 membership attack이 실패해도 삭제 완료를 선언할 수 없다.

- Influence는 정한 loss·최적점·곡률 아래의 1차 근사다.
- 큰 group 삭제와 optimizer 경로 변화에는 부정확할 수 있다.
- 공격 실패는 공격의 검출력 부족일 수 있다.
- 파생 데이터, checkpoint, adapter, index, cache에 흔적이 남을 수 있다.
- 후속 fine-tuning에서 지식이 다시 나타날 수 있다. 다만 빠른 relearning만으로 잔존 기억을 단정하지 말고, 같은 protocol의 retrain reference와 무관 지식 plasticity control보다 비정상적으로 빠른지 비교해야 한다.

가장 강한 비교 기준은 가능한 경우 forget set 없이 동일 recipe로 다시 학습한

$$
\theta_{-f}
$$

이다. Candidate $\widetilde\theta$를 원래 모델뿐 아니라 이 retrain reference와 비교해야 한다. 최소 검증 축은 다음 네 가지다.

1. 직접·우회·다국어·multimodal forget efficacy
2. task·group별 retain utility와 calibration
3. membership·extraction 등 privacy 공격
4. 후속 update·adapter·quantization 뒤 recurrence와 relearning

마지막으로 source에서 배포 replica까지 계보를 확인하고, 독립 final gate를 통과한 범위만 주장해야 한다.

$$
\text{낮은 influence}
\neq
\text{삭제}
\neq
\text{검증된 unlearning}
\neq
\text{안전한 재배포}.
$$

---

**다음 학습 예고:** 오늘은 가중 particle로 비선형·비Gaussian posterior를 근사하고, 학습 사례의 국소·조합 기여를 추정해 삭제 검증으로 연결했다. 다음에는 particle smoothing·Rao–Blackwellization·particle MCMC로 전체 경로와 모수를 함께 추론하고, differential privacy와 certified removal의 차이, SISA·checkpoint lineage를 이용한 검증 가능한 모델 수명주기로 확장한다.
