# [2026-10-26] 오늘 학습: 파티클 스무딩·Rao–Blackwellization·Particle MCMC & 차등 개인정보보호·검증 가능한 모델 삭제

> **오늘의 핵심 문장:** 필터링은 지금까지 본 증거로 현재를 추정하지만, 스무딩은 나중에 도착한 증거로 과거의 추정을 고친다. Rao–Blackwellization은 계산할 수 있는 불확실성을 적분으로 없애고, Particle MCMC는 파티클 필터의 무작위 가능도 추정까지 상태에 포함해 모수의 정확한 사후분포를 겨냥한다. 같은 엄밀함은 모델 삭제에도 필요하다. 차등 개인정보보호는 한 사람의 포함 여부에 대한 학습 알고리즘의 민감도를 제한하지만, 이미 만들어진 체크포인트·어댑터·색인·캐시가 실제로 폐기됐다는 증명은 아니다. SISA와 계보 추적은 삭제 범위와 재학습 범위를 계산 가능한 작업으로 바꾼다.

지난 시간에는 비선형·비Gaussian 상태공간모형의 현재 상태분포를 가중 particle로 근사했고, influence function과 Shapley 근사로 학습 사례의 영향을 추정한 뒤 machine unlearning의 검증 기준을 세웠다. 오늘은 그 다음 두 질문을 다룬다.

1. 현재 상태뿐 아니라 **과거 전체 경로와 모르는 모수**까지 추론하려면 어떻게 해야 할까?
2. “한 사례의 영향이 작다”를 넘어 **삭제 가능한 학습 구조와 검증 가능한 모델 계보**를 어떻게 설계할까?

두 트랙은 모두 “숨은 것을 적당히 버리지 말고, 불확실성과 계보를 끝까지 운반하라”는 원칙으로 연결된다.

$$
\underbrace{p(x_t\mid y_{1:t})}_{\text{현재까지로 현재를 추정}}
\longrightarrow
\underbrace{p(x_t\mid y_{1:T})}_{\text{미래 증거로 과거를 수정}}
\longrightarrow
\underbrace{p(\theta,x_{0:T}\mid y_{1:T})}_{\text{모수와 전체 경로를 함께 추론}}
$$

$$
\underbrace{\text{학습 데이터}}_{\text{source}}
\longrightarrow
\underbrace{\text{checkpoint·adapter·index}}_{\text{derived artifacts}}
\longrightarrow
\underbrace{\text{배포 replica·cache}}_{\text{serving descendants}}
$$

> **난이도 표지:** 필터링은 터널을 달리며 앞유리만 보고 현재 위치를 추정하는 일이다. 스무딩은 터널을 나온 뒤 GPS 기록과 전체 영상을 다시 보며 “10초 전에는 어느 갈림길에 있었는가?”를 고치는 일이다. Rao–Blackwellization은 직접 주사위를 던질 필요가 없는 부분을 정확한 평균 계산으로 대체하는 일이다. SISA는 한 권의 거대한 장부 대신 독립된 장부와 중간 저장점을 만들어, 한 영수증 삭제 때문에 모든 장부를 처음부터 다시 쓰지 않게 하는 설계다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 필터링만으로는 부족할까?

시점 $t$에서 사용할 수 있는 관측이 $y_{1:t}$뿐이라면 filtering posterior

$$
p(x_t\mid y_{1:t})
$$

가 최선이다. 하지만 모든 관측 $y_{1:T}$가 모인 뒤에도 과거를 당시 정보만으로 판단할 이유는 없다. 예를 들어 자율주행차가 순간적으로 가려진 보행자를 놓쳤더라도, 몇 프레임 뒤 다시 나타난 위치와 속도는 가려진 구간의 경로를 고치는 단서가 된다.

스무딩(smoothing)은 미래 관측까지 사용한다.

$$
p(x_t\mid y_{1:T}),\qquad t<T.
$$

따라서 filtering과 smoothing의 차이는 “실시간 대 오프라인”만이 아니다. **조건으로 주어진 정보 집합 자체가 다르다.** 같은 시점 $t$를 묻더라도 미래 관측을 추가하면 posterior가 달라진다.

### 1.2 왜 particle의 조상만 거슬러 올라가면 안 될까?

파티클 필터는 재표집할 때 높은 가중치의 조상을 여러 번 복제한다. 현재 시점에는 다양한 particle이 있어 보여도 멀리 과거로 올라가면 대부분 같은 몇 개의 조상으로 합쳐질 수 있다. 이를 path degeneracy라고 한다.

현재 particle 하나를 뽑고 저장된 부모 포인터만 따라가는 ancestral tracing은 구현이 쉽지만, 긴 과거 경로에서는 선택지가 이미 사라졌을 수 있다. Backward simulation은 저장된 부모 하나에 묶이지 않고, 현재 선택한 $x_{t+1}$과 호환되는 시점 $t$의 모든 particle을 다시 비교한다. 미래 상태가 과거 후보를 재평가하는 셈이다.

### 1.3 왜 모든 숨은 변수를 particle로 뽑지 않을까?

Monte Carlo 오차는 표본을 무작위로 뽑기 때문에 생긴다. 그런데 모형의 일부는 조건부로 정확히 적분할 수 있을 때가 있다. 예를 들어 전체 시스템은 비선형이지만, 어떤 연속 상태는 다른 상태가 주어지면 선형 Gaussian Kalman filter로 계산될 수 있다.

이때 계산 가능한 부분까지 particle로 뽑으면 불필요한 잡음이 늘어난다. Rao–Blackwellization은

$$
h(U,V)
\quad\text{대신}\quad
\mathbb E[h(U,V)\mid U]
$$

를 사용해 $V$에 대한 무작위성을 적분으로 제거한다. 같은 정보로 더 낮은 분산을 얻는 것이 핵심이다.

### 1.4 상태뿐 아니라 모수도 모르면 어떻게 할까?

전이잡음의 크기, 센서잡음, 마찰계수 같은 모수 $\theta$를 모르면 목표는 더 커진다.

$$
p(\theta,x_{0:T}\mid y_{1:T}).
$$

고정된 $\theta$에서는 파티클 필터를 돌릴 수 있지만, MCMC의 Metropolis–Hastings 승인비에는 주변가능도

$$
p_\theta(y_{1:T})
=
\int p_\theta(x_{0:T},y_{1:T})\,dx_{0:T}
$$

가 필요하다. 고차원 경로 적분은 보통 닫힌형으로 계산되지 않는다. Particle MCMC는 파티클 필터가 주는 **음이 아닌 불편 가능도 추정량**을 확률적 상태의 일부로 취급해 이 장벽을 넘는다.

### 1.5 왜 개인정보 보호와 삭제를 따로 배워야 할까?

학습 데이터 한 건을 뺐을 때 모델이 크게 바뀐다면 그 사례는 모델 출력에 강한 흔적을 남길 수 있다. Differential Privacy(DP)는 인접한 두 데이터셋에서 학습 결과분포가 크게 달라지지 않도록 제한한다. 이는 “한 사람의 참여 여부를 출력만 보고 알아내기 어렵게 하자”는 강력한 사전 보장이다.

그러나 다음 두 문장은 다르다.

- **DP 학습:** 한 사례의 포함 여부가 결과분포에 미치는 영향을 제한했다.
- **검증된 삭제:** 삭제 요청 뒤 만들어진 모델과 파생 산출물이 정한 재학습 기준선에 충분히 가깝고, 이전 산출물은 더 이상 사용되지 않는다.

DP는 두 번째 문장을 자동으로 증명하지 않는다. 이미 배포된 오래된 체크포인트, vector index, optimizer state, LoRA adapter, distillation student, 로그와 캐시는 별도의 계보·폐기 절차가 필요하다.

### 1.6 왜 SISA가 등장했을까?

가장 직접적인 삭제 방법은 삭제 대상을 뺀 뒤 처음부터 재학습하는 것이다. 하지만 거대 모델의 전체 재학습은 너무 비쌀 수 있다. SISA는 데이터를 여러 shard로 분리하고, 각 shard를 시간 순서의 slice로 나누며, slice마다 checkpoint를 저장한다. 예측 시에는 독립 모델들의 출력을 aggregate한다.

삭제 요청이 오면 해당 사례가 들어간 shard의 직전 checkpoint로 돌아가, 영향받은 stage부터 누적 prefix 학습을 다시 수행한다. 즉, 삭제 효율을 사후 최적화가 아니라 **처음부터의 훈련 구조**로 산다.

### 1.7 오늘의 본질

오늘의 수학과 AI 운영은 모두 다음 구분을 요구한다.

$$
\text{그럴듯한 점 추정}
\neq
\text{불확실성을 보존한 추론}
$$

$$
\text{낮은 노출 위험}
\neq
\text{삭제 요청 처리}
\neq
\text{파생 산출물까지 검증된 폐기}.
$$

무엇을 조건으로 삼았는지, 무엇을 적분했는지, 어떤 무작위성을 남겼는지, 어떤 산출물이 어떤 데이터에서 파생됐는지를 기록해야 주장을 검증할 수 있다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| $X_t$ | 시점 $t$의 숨은 상태 | 위치·속도·장면의 실제 물리 상태처럼 직접 완전히 볼 수 없는 대상이다. |
| $Y_t$ | 시점 $t$의 관측 | 카메라·센서·로그처럼 상태에 대한 불완전한 단서다. |
| $y_{1:T}$ | 시점 $1$부터 $T$까지 실제로 본 관측열 | 대문자는 확률변수, 소문자는 관측된 값을 뜻한다. |
| $\theta$ | 모형의 알려지지 않은 고정 모수 | 전이잡음·센서잡음·마찰계수 등을 묶는다. |
| filtering | 현재까지의 관측으로 현재 상태를 추론 | $p(x_t\mid y_{1:t})$를 계산한다. |
| prediction | 아직 보지 않은 미래 상태를 예측 | $p(x_{t+1}\mid y_{1:t})$를 계산한다. |
| smoothing | 미래 관측까지 사용해 과거 상태를 수정 | $p(x_t\mid y_{1:T})$를 계산한다. |
| fixed-interval smoothing | 정해진 전체 구간 $1{:}T$를 모두 본 뒤 과거 전체를 추론 | 오프라인 분석에 적합하다. |
| fixed-lag smoothing | $L$시점 뒤까지만 기다렸다가 $x_t$를 확정 | 정확도와 지연시간을 교환한다. |
| backward kernel | 다음 상태가 주어졌을 때 직전 상태 후보를 고르는 조건부분포 | 미래에서 과거로 경로를 샘플링한다. |
| FFBSi | Forward Filtering Backward Simulation | 앞 방향 PF 뒤에 뒤 방향 경로 샘플링을 붙인 방법이다. |
| ancestor index $A_t^{(i)}$ | 시점 $t+1$의 child particle $i$가 선택한 시점 $t$ parent의 번호 | 이 시간 규약으로 계보를 저장하지만 장기적으로 다양성이 줄 수 있다. |
| path degeneracy | 오래전 조상이 소수로 합쳐지는 현상 | 단순 ancestral tracing의 긴 경로 추정을 망친다. |
| Rao–Blackwellization | 조건부로 계산 가능한 변수를 적분해 Monte Carlo 잡음을 줄이는 기법 | particle 수를 같은 채로 추정 분산을 낮춘다. |
| sufficient statistic | 모수 추론에 필요한 정보를 잃지 않고 데이터를 압축한 통계량 | 조건부 갱신을 싸게 만들 수 있다. |
| RBPF | Rao–Blackwellized Particle Filter | 일부 상태는 particle, 일부는 Kalman filter 같은 정확 계산으로 다룬다. |
| likelihood $p_\theta(y_{1:T})$ | 모수 $\theta$ 아래에서 실제 관측열이 나올 밀도 | 모수 후보가 데이터를 얼마나 잘 설명하는지 평가한다. |
| evidence | 숨은 상태를 적분한 관측의 주변확률 | likelihood와 같은 역할로 쓰인다. |
| unbiased estimator | 반복 평균이 참값과 일치하는 추정량 | PMMH의 정확한 주변 posterior 보장에 중요하다. |
| pseudo-marginal MCMC | 계산 불가능한 likelihood 대신 적절한 무작위 추정량을 확장 상태에 넣는 MCMC | 주변분포가 목표 posterior가 되도록 한다. |
| PMMH | Particle Marginal Metropolis–Hastings | 제안된 모수마다 PF를 돌려 승인 여부를 정한다. |
| proposal $q(\theta'\mid\theta)$ | 현재 모수에서 새 모수 후보를 만드는 분포 | MH 승인비에서 비대칭성을 보정한다. |
| adjacent datasets $D\sim D'$ | 한 개인의 레코드 하나가 추가·제거되거나 교체된 두 데이터셋 | DP가 비교하는 입력 쌍이다. |
| mechanism $M$ | 데이터셋을 받아 무작위 결과를 내는 알고리즘 | DP 학습 전체를 하나의 확률 메커니즘으로 본다. |
| $\varepsilon$ | DP에서 두 결과분포가 얼마나 달라질 수 있는지 나타내는 주된 예산 | 작을수록 더 강한 안정성이지만 utility 비용이 커질 수 있다. |
| $\delta$ | DP 부등식에 더해지는 additive slack | 흔히 실패확률처럼 직관화하지만 보편적인 “확률 $\delta$로 privacy가 실패”라는 뜻은 아니며, 데이터 규모에 맞게 충분히 작게 정당화해야 한다. |
| clipping bound $C$ | 각 사례 gradient의 최대 $\ell_2$ 크기 | 한 사례가 업데이트에 미치는 민감도를 제한한다. |
| noise multiplier $\sigma$ | clipping scale에 비례해 넣는 Gaussian noise의 크기 | privacy와 학습 정확도의 trade-off를 정한다. |
| privacy accountant | 반복된 학습 step의 누적 privacy loss를 계산하는 회계 도구 | epoch·sampling rate·noise를 최종 $(\varepsilon,\delta)$로 연결한다. |
| post-processing | DP 결과만 사용한 후속 계산은 privacy 보장을 악화시키지 않는 성질 | 양자화·추론 자체에는 추가 원본 접근이 없다면 유용하다. |
| certified removal | unlearning 결과분포가 삭제 대상 없이 재학습한 기준분포와 가깝다는 보장 | DP와 닮은 부등식을 쓰지만 비교 대상이 다르다. |
| SISA | Sharded, Isolated, Sliced, Aggregated 학습 | 재학습 범위를 삭제 대상의 shard·slice 이후로 제한한다. |
| shard | 서로 겹치지 않게 나눈 데이터 조각 | 독립 모델 하나의 학습 데이터가 된다. |
| slice | 한 shard 안의 시간 순서 학습 구간 | 각 경계에 checkpoint를 둔다. |
| aggregation | 여러 shard 모델의 예측을 합치는 규칙 | 투표·평균·학습된 결합기를 쓸 수 있다. |
| lineage DAG | 데이터와 모델 산출물의 파생 관계를 나타낸 방향성 비순환 그래프 | 삭제 시 영향받는 모든 후손을 찾는다. |
| artifact | 데이터 snapshot·checkpoint·adapter·index·cache 같은 저장 산출물 | 삭제 범위의 실제 단위다. |
| content hash | 파일 내용으로 계산한 고유 지문 | 같은 이름의 다른 버전을 구분하고 재현성을 높인다. |
| tombstone | 특정 artifact를 더 이상 사용하면 안 된다고 표시한 폐기 기록 | 오래된 replica가 되살아나는 것을 막는다. |
| transitive closure | 한 노드에서 간선을 따라 도달 가능한 모든 후손의 집합 | source 삭제가 무효화할 전체 범위를 계산한다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 상태공간모형의 결합분포부터 다시 세우기

초기분포, 전이분포, 관측분포를 각각

$$
X_0\sim \mu_\theta(x_0),
\qquad
X_t\mid X_{t-1}\sim f_\theta(x_t\mid x_{t-1}),
\qquad
Y_t\mid X_t\sim g_\theta(y_t\mid x_t)
$$

라고 하자. Markov 가정 아래 전체 결합분포는

$$
p_\theta(x_{0:T},y_{1:T})
=
\mu_\theta(x_0)
\prod_{t=1}^{T}
f_\theta(x_t\mid x_{t-1})
g_\theta(y_t\mid x_t)
$$

로 분해된다. 이 한 줄이 filtering, smoothing, likelihood, parameter inference의 공통 설계도다.

### 3.2 Filtering·prediction·smoothing은 조건이 다르다

Filtering은

$$
p_\theta(x_t\mid y_{1:t})
$$

이고, 한 단계 prediction은

$$
p_\theta(x_{t+1}\mid y_{1:t})
=
\int
f_\theta(x_{t+1}\mid x_t)
p_\theta(x_t\mid y_{1:t})
\,dx_t
$$

이다. 반면 fixed-interval smoothing은

$$
p_\theta(x_t\mid y_{1:T}),\qquad t<T
$$

이다. 조건 집합이 $y_{1:t}$에서 $y_{1:T}$로 커졌으므로 미래 관측이 과거 상태에 대한 정보를 추가한다.

MSE 관점에서도 차이가 보인다. 제곱오차 아래 최적 추정량은 조건부 평균이므로

$$
\widehat x_t^{\text{filter}}
=
\mathbb E[X_t\mid Y_{1:t}],
\qquad
\widehat x_t^{\text{smooth}}
=
\mathbb E[X_t\mid Y_{1:T}].
$$

조건 정보가 더 많은 smoothing 평균은 같은 모형이 맞다는 전제 아래 평균제곱오차를 증가시키지 않는다.

### 3.3 Backward kernel 유도

전체 미래 관측을 이미 봤다고 하자. $x_{t+1}$이 주어지면 Markov 성질 때문에 $X_t$는 $y_{t+1:T}$의 나머지 정보와 조건부 독립이 된다.

$$
p_\theta(x_t\mid x_{t+1},y_{1:T})
=
p_\theta(x_t\mid x_{t+1},y_{1:t}).
$$

Bayes 정리를 적용하면

$$
p_\theta(x_t\mid x_{t+1},y_{1:t})
=
\frac{
p_\theta(x_t\mid y_{1:t})
f_\theta(x_{t+1}\mid x_t)
}{
p_\theta(x_{t+1}\mid y_{1:t})
}.
$$

분모는 $x_t$에 관해 상수이므로

$$
p_\theta(x_t\mid x_{t+1},y_{1:t})
\propto
p_\theta(x_t\mid y_{1:t})
f_\theta(x_{t+1}\mid x_t).
$$

직관은 간단하다.

- 시점 $t$에서 원래 그럴듯했던 상태여야 한다.
- 동시에 이미 선택된 다음 상태 $x_{t+1}$로 전이하기 쉬워야 한다.

### 3.4 Particle backward simulation

파티클 필터가 시점 $t$의 filtering posterior를

$$
\widehat p_\theta^N(dx_t\mid y_{1:t})
=
\sum_{i=1}^{N}W_t^{(i)}\delta_{X_t^{(i)}}(dx_t)
$$

로 근사했다고 하자. 다음 상태 $x_{t+1}$이 선택됐을 때 시점 $t$ particle의 backward probability는

$$
B_t^{(i)}(x_{t+1})
=
\frac{
W_t^{(i)}
f_\theta(x_{t+1}\mid X_t^{(i)})
}{
\sum_{j=1}^{N}
W_t^{(j)}
f_\theta(x_{t+1}\mid X_t^{(j)})
}.
$$

FFBSi 한 경로는 다음처럼 뽑는다.

1. $I_T\sim\operatorname{Categorical}(W_T^{(1:N)})$로 끝점 $X_T^{(I_T)}$를 뽑는다.
2. $t=T-1,\ldots,0$에 대해 $I_t\sim\operatorname{Categorical}(B_t^{(1:N)}(X_{t+1}^{(I_{t+1})}))$를 뽑는다.
3. $(X_0^{(I_0)},\ldots,X_T^{(I_T)})$를 PF가 만든 empirical smoothing approximation의 경로 표본으로 사용한다.

한 경로마다 각 시점에서 $N$개 후보를 비교하면 대략 $O(NT)$가 든다. 모든 particle 쌍의 backward 가중치를 미리 계산하는 구현은 $O(N^2T)$가 될 수 있다. Sparse transition 구조, rejection-based fast FFBSi, tree-based 계산 등으로 줄일 수 있지만 적용 조건을 기록해야 한다. Ancestor sampling은 보통 conditional SMC 안에서 쓰는 PGAS 계열의 별도 방법이지, 단순한 FFBSi 계산 단축과 같지 않다.

### 3.5 단순 조상 추적과 backward simulation의 차이

재표집 때 저장한 부모 인덱스만 따라가면

$$
I_t=A_t^{(I_{t+1})}
$$

로 경로가 결정된다. 이는 forward pass에서 살아남은 genealogy에 갇힌다. Backward simulation은

$$
W_t^{(i)}f_\theta(x_{t+1}\mid X_t^{(i)})
$$

를 다시 계산해 저장된 직접 부모가 아니더라도 다음 상태와 호환되는 과거 particle을 선택할 수 있다. 이미 버린 상태값을 되살릴 수는 없지만, 살아 있는 filtering cloud 안에서 경로 다양성을 회복한다.

### 3.6 Fixed-lag smoothing: 정확도와 지연의 계약

실시간 시스템은 $T$가 끝날 때까지 기다릴 수 없다. Lag $L$을 정해

$$
p(x_t\mid y_{1:t+L})
$$

만 계산하면 $L$ step의 지연으로 미래 정보를 일부 사용한다. 이것은 조건 $y_{1:t+L}$에 대한 올바른 posterior지만, full smoother $p(x_t\mid y_{1:T})$의 근사로 볼 때는 미래 관측을 자른 것이다. 전이계가 과거를 빠르게 잊는다면 먼 미래 관측의 추가 정보가 작아질 수 있지만, 장기 의존성이 강하거나 늦게 드러나는 사건이 있으면 작은 $L$은 full-smoothing truncation error를 키울 수 있다.

따라서 $L$은 단순한 성능 옵션이 아니다.

$$
\text{큰 }L
\Rightarrow
\text{더 많은 미래 증거·더 큰 메모리·더 긴 결정 지연},
$$

$$
\text{작은 }L
\Rightarrow
\text{빠른 확정·더 큰 truncation 위험}.
$$

안전 사건의 사후 감사와 즉시 제어가 같은 lag를 사용할 필요도 없다.

### 3.7 Rao–Blackwell 정리의 분산 감소

적분하려는 양이 $H=h(U,V)$이고 $U$만 샘플링한 뒤

$$
m(U)=\mathbb E[H\mid U]
$$

를 정확히 계산할 수 있다고 하자. 전체분산법칙은

$$
\operatorname{Var}(H)
=
\mathbb E[\operatorname{Var}(H\mid U)]
+
\operatorname{Var}(\mathbb E[H\mid U])
$$

이다. 즉,

$$
\operatorname{Var}(m(U))
=
\operatorname{Var}(H)
-
\mathbb E[\operatorname{Var}(H\mid U)]
\le
\operatorname{Var}(H).
$$

$V$를 직접 뽑는 대신 조건부 평균으로 없애면 그 조건부 무작위성만큼 분산이 줄어든다. 등호는 $U$가 주어졌을 때 $H$에 남는 변동이 없을 때만 성립한다.

독립 표본 $M$개의 평균에서도

$$
\operatorname{Var}\!\left(
\frac1M\sum_{m=1}^{M}m(U^{(m)})
\right)
=
\frac{\operatorname{Var}(m(U))}{M}
$$

이므로 같은 표본 수에서 Monte Carlo 오차가 줄어든다.

### 3.8 Rao–Blackwellized particle filter의 구조

상태를 비선형 부분 $R_t$와 조건부 선형 Gaussian 부분 $Z_t$로 나누자.

$$
X_t=(R_t,Z_t).
$$

$R_{0:t}$가 주어졌을 때 $Z_t$의 posterior가 Gaussian이라면 각 particle은 단일 $Z_t$ 값 대신

$$
p(z_t\mid r_{0:t}^{(i)},y_{1:t})
=
\mathcal N(m_t^{(i)},P_t^{(i)})
$$

라는 Kalman filter 요약을 들고 다닌다.

$$
\left\{
R_{0:t}^{(i)},
W_t^{(i)},
m_t^{(i)},
P_t^{(i)}
\right\}_{i=1}^{N}.
$$

이 방식은 particle 차원을 줄이고 분산을 낮추지만, particle마다 행렬 갱신을 해야 하므로 조건부 선형 상태 차원이 크면 계산비가 커진다. 또한 조건부 Gaussian 가정이 틀리면 정확한 적분이 아니라 잘못된 모형을 정밀하게 계산한 것이 된다.

### 3.9 파티클 필터의 주변가능도 추정

모수 $\theta$를 평가하려면

$$
p_\theta(y_{1:T})
=
\prod_{t=1}^{T}
p_\theta(y_t\mid y_{1:t-1})
$$

가 필요하다. 매 시점 재표집하는 bootstrap particle filter에서는 $X_t^{(i)}$를 predictive mixture에서 뽑고 증분 가중치를

$$
\widetilde w_t^{(i)}
=
g_\theta(y_t\mid X_t^{(i)})
$$

로 둔다. 한 단계 evidence 추정은

$$
\widehat Z_t
=
\frac1N\sum_{i=1}^{N}\widetilde w_t^{(i)}
$$

이고 전체 likelihood 추정량은

$$
\widehat p_\theta(y_{1:T})
=
\prod_{t=1}^{T}\widehat Z_t.
$$

일반 proposal $q_\theta(x_t\mid x_{t-1},y_t)$에서는 증분 가중치에

$$
\frac{
f_\theta(x_t\mid x_{t-1})
g_\theta(y_t\mid x_t)
}{
q_\theta(x_t\mid x_{t-1},y_t)
}
$$

가 들어간다. PF proposal은 target path density가 양수인 영역을 덮어야 한다. 또한 offspring 수 $N_t^{(i)}$가

$$
\mathbb E[N_t^{(i)}\mid W_t^{(1:N)}]
=
NW_t^{(i)}
$$

를 만족하는 의미의 불편 재표집과 표준 SMC 조건 아래 $\widehat p_\theta(y_{1:T})$는 음이 아니며 likelihood에 대해 불편하다.

$$
\mathbb E[\widehat p_\theta(y_{1:T})]
=
p_\theta(y_{1:T}).
$$

주의할 점은

$$
\mathbb E[\log\widehat p_\theta(y_{1:T})]
\neq
\log p_\theta(y_{1:T})
$$

일반적으로 성립한다는 것이다. 수치 계산은

$$
\log\widehat p_\theta(y_{1:T})
=
\sum_{t=1}^{T}
\left[
\operatorname{LSE}_{i=1}^{N}(\log\widetilde w_t^{(i)})
-\log N
\right]
$$

처럼 log-sum-exp로 해야 underflow를 피할 수 있다.

### 3.10 PMMH 승인확률

현재 모수 $\theta$와 그때 PF가 사용한 난수 $u$를 상태로 보자. 새 모수

$$
\theta'\sim q(\theta'\mid\theta)
$$

를 제안하고 새로운 난수 $u'$로 PF를 돌려 $\widehat p_{\theta'}(y_{1:T};u')$를 얻는다. 승인확률은

$$
\alpha
=
\min\!\left{
1,
\frac{
\widehat p_{\theta'}(y_{1:T};u')
p(\theta')
q(\theta\mid\theta')
}{
\widehat p_{\theta}(y_{1:T};u)
p(\theta)
q(\theta'\mid\theta)
}
\right}.
$$

여기서 $p(\theta)$는 prior다. 거절되면 $\theta$뿐 아니라 현재 likelihood 추정에 사용한 $u$와 추정값도 유지한다. 매번 분모를 새로 추정하면 표준 pseudo-marginal Markov chain이 아니다.

왜 무작위 추정량을 썼는데도 정확한가? 확장 목표분포를 개념적으로

$$
\widetilde\pi(\theta,u)
\propto
\widehat p_\theta(y;u)
p(\theta)
m_\theta(u)
$$

로 두면, $u$를 적분했을 때 불편성 때문에

$$
\int\widetilde\pi(\theta,u)\,du
\propto
p_\theta(y)p(\theta)
\propto
p(\theta\mid y)
$$

가 된다. 이 결론에는 posterior가 정규화 가능하고, 각 posterior-support의 $\theta$에서 추정량이 거의 확실히 유한하고 음이 아니며, $u\sim m_\theta$에 대해 불편하고, PF가 실제로 그 $m_\theta$에서 난수를 생성한다는 조건이 필요하다. 또한 parameter proposal $q$가 목표 posterior의 support를 탐색해야 하며, chain 수렴에는 통상적인 irreducibility와 aperiodicity 조건이 필요하다.

유한 $N$은 chain의 혼합 효율을 떨어뜨릴 수 있지만 위 조건이 맞으면 목표 주변분포 자체를 근사 posterior로 바꾸지는 않는다. 다만 가능도 추정 분산이 너무 크면 chain이 우연히 큰 추정값에 오래 붙는 sticky 현상이 생긴다. 현재 상태의 likelihood estimate는 양수인 영역에서 유지하고, 제안 추정값이 $0$이면 그 제안은 거절된다.

### 3.11 Differential Privacy 정의

한 사람의 레코드 하나가 추가·제거된 데이터셋을 인접하다고 쓰자.

$$
D\sim D'.
$$

무작위 메커니즘 $M$이 모든 측정 가능한 결과 집합 $S$에 대해

$$
\Pr[M(D)\in S]
\le
e^{\varepsilon}
\Pr[M(D')\in S]
+\delta
$$

를 만족하면 $(\varepsilon,\delta)$-DP라고 한다. 인접성 정의가 add/remove인지 replace-one인지에 따라 민감도와 해석이 달라지므로 반드시 함께 적어야 한다.

$\varepsilon=0$, $\delta=0$이면 두 결과분포가 완전히 같아 한 레코드의 존재가 아무 영향도 주지 않는다. 실제 학습에서는 utility 때문에 양의 값을 쓴다. $\delta=0$인 pure DP에서는 $e^\varepsilon$가 출력 사건의 확률비를 제한한다. $\delta>0$이면 식은 비율 제한에 additive slack을 더하므로, 이를 모든 실행에 공통인 문자 그대로의 “privacy 실패확률”로 해석하면 안 된다.

### 3.12 DP-SGD의 clipping과 noise

미니배치 $B$의 각 사례 손실에 대해 per-example gradient를 계산한다.

$$
g_i=\nabla_\theta\ell(z_i;\theta).
$$

먼저 $\ell_2$ norm을 $C$로 clip한다.

$$
\bar g_i
=
g_i\min\!\left(1,\frac{C}{\lVert g_i\rVert_2}\right).
$$

그다음 Gaussian noise를 더한 평균을 사용한다.

$$
\widetilde g
=
\frac1{|B|}
\left(
\sum_{i\in B}\bar g_i
+
\mathcal N(0,\sigma^2C^2I)
\right).
$$

업데이트는

$$
\theta\leftarrow\theta-\eta\widetilde g
$$

이다. 역할을 1:1로 구분하면 다음과 같다.

- clipping은 한 사례가 합계 gradient를 움직일 수 있는 최대 크기를 제한한다.
- Gaussian noise는 인접 데이터셋에서 나온 업데이트분포를 겹치게 만든다.
- subsampling은 한 사례가 step에 포함될 확률을 낮춰 privacy amplification을 줄 수 있다.
- accountant는 sampling rate, noise, step 수를 누적 privacy budget으로 바꾼다.

$C$가 너무 작으면 많은 gradient 방향이 잘려 bias가 커지고, 너무 크면 필요한 noise의 절대 크기가 커진다. 따라서 $C$와 $\sigma$는 독립적인 장식 파라미터가 아니다.

### 3.13 Composition과 post-processing

같은 사람의 데이터에 여러 DP 메커니즘을 적용하면 privacy loss가 누적된다. 가장 단순한 보수적 합성은

$$
(\varepsilon_1,\delta_1)
+
(\varepsilon_2,\delta_2)
\Rightarrow
(\varepsilon_1+\varepsilon_2,\delta_1+\delta_2)
$$

이다. 실제 DP-SGD는 Rényi DP나 privacy loss distribution 기반 accountant로 더 타이트하게 계산할 수 있지만, 사용한 sampling 가정과 구현을 명시해야 한다.

반대로 $M(D)$만 입력으로 받는 함수 $f$에 대해

$$
f(M(D))
$$

는 추가 원본 데이터 접근이 없다면 같은 DP 보장을 유지한다. 이것이 post-processing 성질이다. 그러나 후속 fine-tuning이 다시 원본 비공개 데이터에 접근하면 단순 post-processing이 아니며 새로운 privacy 소비를 계산해야 한다.

### 3.14 DP와 certified removal의 닮은꼴과 차이

삭제 알고리즘 $U$, 학습 알고리즘 $A$, 삭제 대상 $D_f$를 생각하자. $U$가 full retraining이나 SISA를 수행할 수 있도록 원 데이터·retained state 중 실제로 접근하는 입력을 명시해야 한다. 여기서는 이를 데이터 인자 $D$로 드러내 쓴다. 대표적인 $(\varepsilon_r,\delta_r)$-certified removal 정의는 unlearned 결과와 삭제 대상을 제외하고 다시 학습한 결과의 분포를 양방향으로 비교한다. 모든 측정 가능한 $S$에 대해

$$
\Pr[U(A(D),D,D_f)\in S]
\le
e^{\varepsilon_r}
\Pr[A(D\setminus D_f)\in S]
+\delta_r,
$$

뿐 아니라

$$
\Pr[A(D\setminus D_f)\in S]
\le
e^{\varepsilon_r}
\Pr[U(A(D),D,D_f)\in S]
+\delta_r
$$

도 요구한다. 두 방향은 한쪽 분포에만 큰 질량이 숨어 있는 경우를 막는다. 문헌마다 삭제 보장의 이름과 세부 입력은 다를 수 있으므로, 실제 보고서에는 사용한 정의와 학습·삭제 알고리즘의 무작위성 범위를 함께 적어야 한다.

모양은 DP와 비슷하지만 비교 대상은 다르다.

| 구분 | Differential Privacy | Certified removal |
| --- | --- | --- |
| 비교 | 인접한 원 데이터셋에서 학습한 결과 | unlearning 결과와 삭제 후 retraining 결과 |
| 시점 | 학습 전 설계하는 사전 안정성 | 삭제 요청 뒤 수행하는 수명주기 보장 |
| 목적 | 참여 여부의 출력 영향 제한 | 삭제 후 모델분포를 reference retrain에 가깝게 만들기 |
| 놓칠 수 있는 것 | 오래된 artifact의 실제 폐기 여부 | 원래 학습의 일반적인 개인정보 노출 |

한 레코드에 대해 매우 강한 DP 학습을 했다면 모델을 그대로 두는 no-op unlearning에도 분포 안정성 논리를 제공할 수 있다. 하지만 group 삭제에서는 privacy 손실이 커지고, checkpoint·raw data·index·cache의 물리적 폐기와 접근통제는 별도다. 따라서 “DP니까 삭제할 필요가 없다”는 문장은 과도하다.

### 3.15 SISA의 재학습 비용

전체 데이터 $D$를 $K$개 독립 shard로 나누고, 각 shard를 $R$개 순서 있는 slice로 나누자.

$$
D
=
\bigsqcup_{k=1}^{K}D_k,
\qquad
D_k
=
\bigsqcup_{r=1}^{R}D_{k,r}.
$$

누적 prefix를

$$
C_{k,r}
=
\bigcup_{j=1}^{r}D_{k,j}
$$

라고 하자. Canonical SISA의 한 학습 단계는 이전 checkpoint에서 시작해 새 slice 하나만 한 번 보는 것이 아니라, 그 시점까지의 누적 prefix로 다시 학습한다.

$$
M_{k,r}
=
\operatorname{Train}(M_{k,r-1},C_{k,r};e_s).
$$

여기서 $e_s$는 각 stage에서 누적 prefix를 보는 epoch 수다. 삭제 사례가 $(k,r)$에 있으면 $M_{k,r-1}$로 돌아가 삭제 사례를 제외한 $C_{k,j}$로 $j=r,\ldots,R$ stage를 다시 수행한다.

전체 데이터 크기를 $n$, monolithic baseline의 epoch 수를 $e_0$, 모든 slice 크기를 $n/(KR)$라고 하자. SISA의 최초 학습 sample-presentation 수를 baseline과 맞추려면

$$
e_s
\sum_{j=1}^{R}\frac{jn}{KR}
=
e_0\frac{n}{K}
$$

이므로

$$
e_s
=
\frac{2e_0}{R+1}
$$

이다. 이 compute-matched 가정에서 slice $r$의 한 사례를 삭제할 때 monolithic 전체 재학습비 대비 SISA 재학습비는

$$
c_r
=
\frac{2}{KR(R+1)}
\sum_{j=r}^{R}j.
$$

삭제 사례가 slice에 균등하게 놓이면

$$
\mathbb E[c_r]
=
\frac{2}{KR(R+1)}
\frac1R
\sum_{r=1}^{R}\sum_{j=r}^{R}j
$$

이고, 각 stage $j$는 가능한 삭제 위치 $r=1,\ldots,j$에 대해 $j$번 나타나므로

$$
\mathbb E[c_r]
=
\frac{2}{KR^2(R+1)}
\sum_{j=1}^{R}j^2
=
\frac{2R+1}{3KR}.
$$

예를 들어 $K=10$, $R=5$이면

$$
\mathbb E[c_r]
=
\frac{11}{150}
\approx
0.0733
$$

으로 이상화된 compute-matched·선형 sample-cost 가정 아래 monolithic 전체 학습의 약 $7.33\%$다. 이는 checkpoint I/O, orchestration, 병렬화, optimizer warm-up 비용을 무시한다. 새 slice만 한 번씩 이어 학습하는 단순 변형에는 더 작은 식이 나오지만, 그것을 canonical 누적-prefix SISA 비용과 섞으면 안 된다.

그러나 이것은 같은 SISA recipe에 대한 재학습 비용 근사다. 다음이 깨지면 독립성이 무너진다.

- shard들이 같은 데이터로 학습한 global tokenizer나 base checkpoint를 공유한다.
- cross-shard normalization 통계나 공동 hard-negative mining을 쓴다.
- 삭제 사례의 증강·중복·합성 파생물이 다른 shard에 있다.
- aggregation model이 모든 shard의 예측으로 별도 학습됐는데 갱신하지 않는다.

또한 SISA ensemble은 단일 전체 데이터 모델과 학습 결과가 같지 않다. 삭제 후 기준선도 “삭제 데이터를 뺀 동일 SISA 절차”여야 정직한 비교다.

### 3.16 Lineage DAG와 전이적 무효화

산출물 계보를 방향성 그래프

$$
G=(V,E)
$$

로 두자. 간선 $(u,v)\in E$는 “artifact $v$가 artifact $u$를 사용해 만들어졌다”는 뜻이다. 삭제 source $s$의 영향범위는 도달 가능한 후손의 전이적 폐쇄다.

$$
\operatorname{Desc}(s)
=
\{v\in V:s\leadsto v\}.
$$

여기서 $s\leadsto v$는 $s$에서 $v$로 가는 유향 경로가 존재한다는 뜻이다.

삭제 작업은 source row만 지우는 것이 아니라

$$
\{s\}\cup\operatorname{Desc}(s)
$$

를 검사해야 한다. 각 노드에는 최소한 다음 메타데이터가 필요하다.

$$
(\text{artifact ID},\text{content hash},\text{parents},\text{recipe},
\text{code SHA},\text{seed},\text{license},\text{status}).
$$

`status`는 active, quarantined, superseded, tombstoned처럼 운영 의미를 가져야 한다. 새 모델을 만들었어도 load balancer가 오래된 replica로 보낼 수 있다면 삭제는 완료되지 않았다.

### 3.17 오늘의 수학이 AI 부품에서 쓰이는 위치

| 수학 개념 | AI·시스템 부품 | 정확한 역할 |
| --- | --- | --- |
| $p(x_t\mid y_{1:T})$ | 비디오·로봇 trajectory 감사 | 이후 프레임으로 가려진 과거 상태를 보정한다. |
| backward kernel | FFBSi·trajectory sampler | 다음 상태와 filtering weight를 함께 사용해 과거 particle을 고른다. |
| 전체분산법칙 | RBPF | 조건부로 계산 가능한 상태를 적분해 sampling variance를 제거한다. |
| 불편 likelihood 추정 | PMMH | 계산 불가능한 상태 경로 적분을 PF의 무작위 추정으로 대체한다. |
| 확장 상태 MH | parameter Bayesian inference | 난수까지 상태에 포함해 $p(\theta\mid y)$를 주변분포로 유지한다. |
| 민감도와 Gaussian noise | DP-SGD | 한 사례의 최대 영향과 결과분포의 구별 가능성을 제한한다. |
| composition | 반복 학습·여러 공개 | privacy budget이 step과 release를 거쳐 누적되는 양을 계산한다. |
| 분포 근접성 | certified removal | unlearned model을 deletion retrain reference와 비교한다. |
| 분할·checkpoint | SISA | 삭제 때 다시 계산할 학습 구간을 제한한다. |
| 그래프 reachability | lineage registry | 삭제 source에서 파생된 checkpoint·adapter·cache를 모두 찾는다. |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 전체 상태를 추론하는 3단계 파이프라인

파티클 기반 시계열 학습을 다음 세 층으로 나누면 혼동이 줄어든다.

```text
[실시간]
관측 y_t -> Particle Filter -> p(x_t | y_1:t) -> 즉시 제어
                              |
                              v
[사후 분석]
저장된 particles/weights -> Backward Smoother -> p(x_0:T | y_1:T)
                              |
                              v
[모수 학습]
PF likelihood estimator <-> PMMH proposals -> p(theta | y_1:T)
```

- PF는 현재 판단을 위한 online component다.
- smoother는 사건 재구성·학습 label 생성·trajectory 분석을 위한 offline 또는 delayed component다.
- PMMH는 전이·관측 모수까지 불확실할 때 쓰는 outer inference loop다.

이 세 가지를 한 번에 실시간으로 돌리려 하면 latency와 메모리 요구가 폭발할 수 있다. 제품 목적별로 경계를 나눠야 한다.

### 4.2 PMMH를 구현할 때 지켜야 할 상태

개념적 pseudocode는 다음과 같다.

```text
theta <- initial parameter
(likelihood_hat, particles) <- particle_filter(theta, fixed_observations, fresh_rng)

repeat:
    theta_new <- propose(theta)
    (likelihood_new, particles_new) <- particle_filter(theta_new, fixed_observations, fresh_rng)
    alpha <- min(1, likelihood_new * prior(theta_new) * reverse_proposal
                    / (likelihood_hat * prior(theta) * forward_proposal))

    if uniform(0, 1) < alpha:
        theta <- theta_new
        likelihood_hat <- likelihood_new
        particles <- particles_new

    record(theta)
```

중요한 점은 거절 시 현재 `likelihood_hat`을 다시 계산하지 않는 것이다. 관측 데이터와 model code, random-number generator version, particle 수, resampling method도 run manifest에 남겨야 재현 가능하다.

### 4.3 DP-SGD는 optimizer 한 줄 변경이 아니다

보통 SGD는 batch 평균 gradient만 계산한다. DP-SGD는 per-example gradient가 필요하므로 메모리·연산 구조가 달라진다. 또한 BatchNorm처럼 한 사례의 activation이 같은 batch의 다른 사례에 의존하는 층은 per-example 민감도 해석을 어렵게 한다. 실무에서는 LayerNorm·GroupNorm 같은 대안을 검토하고, microbatch 구현이 수학적으로 어떤 clipping 단위를 사용하는지 확인해야 한다.

DP 학습 설계에는 최소한 다음 값이 함께 있어야 한다.

| 항목 | 질문 |
| --- | --- |
| privacy unit | 한 row, 한 사용자, 한 대화, 한 환자 중 무엇을 보호하는가? |
| adjacency | add/remove인가 replace-one인가? |
| sampling | Poisson sampling인가 고정크기 without-replacement인가? |
| clipping | per-example인가 per-user인가? $C$는 어떻게 정했는가? |
| noise | $\sigma$와 실제 noise std는 무엇인가? |
| accounting | 어떤 accountant와 가정으로 $(\varepsilon,\delta)$를 계산했는가? |
| releases | checkpoint·metrics·gradient statistics 공개를 모두 합성했는가? |
| utility | 전체 평균뿐 아니라 희귀 group 성능과 calibration을 봤는가? |

한 사용자가 수백 개 레코드를 갖는데 row-level DP만 적용하면 user-level 보호로 바로 읽을 수 없다. Group privacy 비용이 커질 수 있으므로 clipping 단위부터 사용자로 묶어야 할 수 있다.

### 4.4 SISA와 lineage를 결합한 삭제 흐름

```text
[삭제 요청]
     |
     v
subject/record resolver -----> 원본·중복·증강·합성 파생물 식별
     |
     v
lineage DAG reachability ----> 영향받은 shard/slice/checkpoint/adapter/index/cache
     |
     v
SISA retrain planner --------> 가장 이른 영향 slice 직전 checkpoint 선택
     |
     v
재학습 + aggregate 갱신 ----> candidate model family
     |
     v
forget / retain / privacy / recurrence 검증
     |
     v
old artifact tombstone -> replica drain -> canary -> active alias 교체
```

삭제 요청 ID가 단일 row ID와 같다고 가정하면 안 된다. 한 사용자의 원본이 caption, chunk, embedding, synthetic QA, hard negative, distilled trace로 퍼졌다면 entity resolver가 먼저 파생 범위를 찾아야 한다.

### 4.5 검증 가능한 모델 수명주기의 최소 증거

삭제 완료 보고서에는 최소한 다음이 있어야 한다.

1. **범위:** 어떤 subject·record·파생 데이터를 대상으로 했는가?
2. **계보:** 영향을 받은 artifact ID와 hash는 무엇인가?
3. **방법:** full retrain, SISA retrain, certified update 중 무엇을 썼는가?
4. **기준선:** 원래 모델이 아니라 deletion retrain reference와 비교했는가?
5. **Forget:** 직접·우회·다국어·멀티모달 질의에서 대상 정보가 줄었는가?
6. **Retain:** 일반 능력과 안전 성능을 불필요하게 파괴하지 않았는가?
7. **Privacy:** membership·extraction 공격이 충분한 power와 control을 가졌는가?
8. **Recurrence:** 추가 fine-tuning·adapter merge·quantization 뒤 흔적이 다시 나타나는가?
9. **Serving:** 오래된 replica·cache·index가 drain되고 tombstone을 존중하는가?
10. **독립성:** 최종 gate 데이터와 담당자가 unlearning 튜닝에서 분리됐는가?

### 4.6 수학과 삭제 운영의 1:1 연결

Particle smoothing에서 과거 상태의 posterior는 미래 관측과 전이 계보를 모두 사용한다. 모델 삭제도 현재 production alias만 검사해서는 안 되고, source에서 미래 방향으로 파생된 모든 artifact를 따라가야 한다.

$$
\text{Backward smoothing:}
\quad
x_{t+1}\Rightarrow x_t\text{ 후보 재평가}
$$

$$
\text{Forward lineage invalidation:}
\quad
\text{source deletion}\Rightarrow\text{descendant artifact 재평가}
$$

방향은 반대지만 본질은 같다. **국소 상태만 보고 전체 경로를 단정하면 안 된다.**

Rao–Blackwellization과 SISA도 목적은 다르지만 구조적 효율화라는 공통점이 있다.

- Rao–Blackwellization은 정확히 적분 가능한 무작위성을 계산으로 제거한다.
- SISA는 삭제와 무관한 shard의 재학습을 구조적으로 제거한다.

둘 다 가정을 잘못 세우면 효율만 얻고 정답을 잃는다. RBPF는 조건부 모형이 맞아야 하고, SISA는 shard 사이의 실제 의존성을 계보에 기록해야 한다.

### 4.7 초보자가 흔히 하는 오해

#### 오해 1: Smoothing은 filtering curve를 부드럽게 그리는 후처리다

아니다. Smoothing은 미래 관측을 조건에 추가한 새로운 posterior다. 이동평균처럼 숫자 모양만 매끈하게 만드는 것과 다르다.

#### 오해 2: Backward simulation은 죽은 particle을 되살린다

아니다. Forward pass에서 한 번도 보존되지 않은 상태값은 사용할 수 없다. 저장된 각 시점의 filtering particle 집합 안에서 경로 연결을 다시 고른다.

#### 오해 3: Rao–Blackwellization은 언제나 더 싸다

분산은 줄지만 particle마다 Kalman covariance 같은 행렬 계산이 필요하다. 상태 차원과 구조에 따라 총 비용은 커질 수 있다.

#### 오해 4: Unbiased likelihood이면 log-likelihood도 unbiased다

아니다. 로그는 비선형이므로 Jensen 부등식 때문에 일반적으로

$$
\mathbb E[\log\widehat p]<\log p
$$

가 된다. PMMH의 이론은 음이 아닌 likelihood 추정량을 확장 상태에 넣는 데 의존한다.

#### 오해 5: Particle 수가 작아도 PMMH는 정확하니 아무 문제가 없다

정지분포의 주변 목표는 맞을 수 있지만 likelihood noise가 크면 chain이 거의 움직이지 않아 유효표본크기가 매우 작아진다. 정확성과 계산 효율은 별개의 질문이다.

#### 오해 6: DP 모델은 개인 데이터를 기억하지 않는다

DP는 정한 인접성·예산 아래 결과분포의 민감도를 제한한다. $\varepsilon$가 크거나 보호 단위가 잘못됐거나 group contribution이 크면 기대한 보호와 다를 수 있다. 특정 문자열이 절대 출력되지 않는다는 결정론적 보장도 아니다.

#### 오해 7: DP면 삭제 요청을 무시해도 된다

DP는 학습 결과의 안정성 보장이지 raw data, old checkpoint, index, cache의 실제 폐기 기록이 아니다. 법적·운영적 삭제 의무와 certified removal의 비교 기준도 별도다.

#### 오해 8: SISA는 정확한 full retraining과 같은 모델을 만든다

같은 SISA recipe에서 deletion 전후를 비교하는 구조다. 여러 shard 모델을 aggregate한 결과는 처음부터 전체 데이터를 한 모델에 학습한 결과와 일반적으로 다르다.

#### 오해 9: 파일을 지우면 계보에서도 사라진다

백업·registry alias·container layer·quantized copy·adapter·student·vector index·serving cache가 남을 수 있다. 삭제는 파일 연산이 아니라 artifact 상태 전이다.

### 4.8 오늘의 실전 체크리스트

- smoothing 결과에는 사용한 horizon과 lag를 함께 기록한다.
- PF run에는 particle 수, proposal, resampling method, seed 계열, likelihood estimate를 기록한다.
- PMMH 진단에는 trace, autocorrelation, acceptance rate, effective sample size를 포함한다.
- DP 보고에는 privacy unit, adjacency, accountant, $\varepsilon$, $\delta$, clipping, sampling 가정을 함께 적는다.
- SISA manifest에는 shard assignment와 slice checkpoint hash를 불변 기록으로 둔다.
- 삭제 source의 transitive descendants를 찾지 못하면 완료가 아니라 `scope unknown` 상태로 둔다.
- old artifact는 tombstone만 남기지 말고 serving alias·cache·replica가 실제로 참조하지 않는지 확인한다.

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **선정 기준:** 2026년 10월 26일 Asia/Seoul 기준으로 공개 날짜와 접근 상태를 확인했다. 제작사·저자 수치는 독립 재현 결과와 구분하고, 모델 weight·실행 code·학습 data·license의 공개 여부를 각각 따로 본다.

### 5.1 Mundo: 몇 장의 사진을 “시뮬레이션 가능한 세계”로 바꾸는 customized world model

Mundo 팀은 2026년 10월 23일, 한 장소의 여러 시점 이미지(공간 탐색 예시는 1~12장) 또는 video로 편집·탐색·시뮬레이션 가능한 세계를 만드는 시스템을 공개했다. 핵심은 모든 것을 하나의 video generator에 맡기지 않고 표현을 세 층으로 분리한 점이다.

1. 3D foundation model과 detector로 camera pose·geometry·object box의 초기 proxy를 만든다.
2. VLM agent가 여러 view와 render를 비교하는 analysis-by-synthesis loop로 누락된 물체·잘못된 box·문 방향 등을 수정하고, 수치 optimizer가 위치·크기·yaw를 정밀하게 맞춘다.
3. VLM agent가 질량·마찰·applied force 같은 물리 속성을 추정해 scene program에 기록하고, 주로 MuJoCo가 중력·충돌·접촉·robot manipulation에 따른 상태 전이를 계산한다. 이어 proxy frame과 context image를 조건으로 video diffusion model이 최종 영상을 렌더링한다.

상태 전이는 오늘의 표기로

$$
S_{t+1}=f(S_t,a_t)
$$

이고, renderer는 대략

$$
x_{1:T}
\sim
p_\theta\!\left(
\cdot\mid
\pi(S_{1:T},c_{1:T}),
\{(I_k,c_k)\}_{k=1}^{K},
y
\right)
$$

로 읽을 수 있다. 여기서 $S_t$는 object pose와 물리 속성을 담은 명시적 world state, $\pi$는 proxy renderer, $c_t$는 camera, $I_k$는 context view다.

Mundo 설명에 따르면 renderer는 약 4만 쌍의 real/rendered video와 대응 proxy video로 MiniMax-H3의 Ref2VA를 fine-tune했고, 각 clip은 124 frames, $640\times480$ 해상도다. 긴 영상은 124-frame chunk로 생성하면서 직전 chunk의 마지막 34 frames를 다음 chunk의 고정 prefix로 재인코딩한다.

중요한 한계도 분명하다.

- reconstruction은 예시 장면에서 보통 10~20분이 걸린다.
- video rendering은 8대 A800 GPU와 20 inference steps에서 약 1.5 FPS라고 저자들이 보고한다.
- 작은 물체의 fine-grained interaction에서 부유·hallucination이 생길 수 있다.
- 장기 동적 변화 뒤 scene detail을 유지하는 long-horizon memory가 아직 없다.
- 2026년 10월 26일 확인한 공식 GitHub 저장소는 project asset과 README 중심이며, **code와 weights는 11월 말 공개 예정**으로 표시돼 있다. 즉 현재는 완전 재현 가능한 공개 구현이 아니다.

**엔지니어 인사이트 (Impact):** Mundo의 중요한 방향은 생성모델이 물리 상태를 픽셀 속에만 숨기지 않고, 검사 가능한 3D proxy와 code state를 사이에 둔 것이다. 이는 VLM의 의미 추론, numerical fitting의 정밀도, physics engine의 제약, diffusion renderer의 시각 품질을 역할별로 분리한다. 오늘의 smoothing 관점에서는 이후 view가 과거 proxy state와 object identity를 고치도록 posterior를 명시적으로 유지할 여지가 크다. 다만 현재 데모의 단일 proxy를 검량된 posterior로 오해하면 안 된다. 여러 geometry hypothesis를 particle로 유지하고 future view로 backward reweighting하며, uncertainty가 큰 contact를 실제 robot action gate에 전달해야 안전한 Real-to-Sim 파이프라인에 가까워진다.

- 공식 프로젝트: [Mundo: Customized World Model](https://mundo-world-model.github.io/)
- 저자 기술 설명: [Mundo: From seeing a place to simulating what happens there](https://huggingface.co/blog/Luo-Yihang/mundo-world-model)
- 공개 상태 확인: [Luo-Yihang/Mundo](https://github.com/Luo-Yihang/Mundo)

### 5.2 Reflection Beam: 모델 weight와 agent 실행 생태계를 함께 내놓은 501B sparse MoE

Reflection은 2026년 10월 16일 Beam 1.0 weights를 Apache-2.0으로 공개했다. 공식 모델 카드 기준 Beam은 text input·text output 모델이며 다음 구조를 갖는다.

| 항목 | 공식 공개값 |
| --- | --- |
| 전체 / 활성 파라미터 | 501.3B / 23.4B |
| 층 | 3 dense + 49 MoE, 총 52층 |
| expert 수 | 256 |
| attention | Gated GQA + Sliding Window Attention |
| context | 최대 1M tokens |
| vocabulary | 152k tokens |
| 공개된 instruction-tuned weight numerics | BF16, FP8, NVFP4 |
| modality | text only |
| knowledge cutoff | 2026-06-30 |
| license | Apache-2.0 |

Sparse MoE의 핵심은 모든 token이 501.3B 파라미터 전체를 계산하지 않고 일부 expert 경로만 활성화한다는 점이다. 그러나 inactive expert도 checkpoint에는 저장돼야 하므로 “23.4B active”를 “23.4B 모델과 같은 메모리”로 읽으면 안 된다. BF16·FP8·NVFP4 제공은 메모리·통신·정밀도 trade-off를 선택하게 하지만, quantized checkpoint도 별개의 검증 대상이다.

출시 시점에는 BF16·FP8·NVFP4 weights와 speculative drafter, 공개 Mirror·cookbook·verifier 도구, Transformers·vLLM·SGLang을 비롯한 여러 partner integration 경로가 함께 안내됐다. 반면 전체 training code, 완전한 data manifest, model technical report는 공식 공개물에서 확인되지 않는다. 이는 오픈 모델 경쟁이 weight 파일 한 개에서 다음 묶음으로 이동하고 있음을 보여 준다.

$$
\text{model}
+
\text{quantization}
+
\text{serving}
+
\text{agent harness}
+
\text{evaluation}
+
\text{customization recipe}.
$$

공식 모델 카드는 BF16·FP8·NVFP4에 대한 다양한 coding·reasoning·tool-use benchmark를 제공하지만, 이는 제작사가 모델 카드에 공표한 결과이며 독립 재현으로 확인된 값은 아니다. temperature $0.7$, top-$p$ $0.99$라는 공통 decoding 조건은 적혀 있어도 benchmark harness, contamination, hardware, agent tool policy가 다른 외부 결과와 단순 순위 비교하면 안 된다. 공식 설명상 pretraining은 public source와 commercial partner agreement로 얻은 data, mid-training은 선별한 real-world example에서 파생한 data, RL은 주로 synthetic pipeline에 proprietary vendor data와 open-source source를 보탠 구성이다. 이는 범주 수준 설명이며, 개별 학습 example을 역추적할 수 있는 완전한 data manifest는 아니다. 모델 카드에는 model technical report가 `coming`으로 표시돼 있고, 별도의 safety technical report는 이미 공개돼 있다.

**엔지니어 인사이트 (Impact):** Beam은 “open-weight agent model”의 실용성을 보여 주는 동시에 오늘 배운 lineage 문제를 확대한다. 하나의 base에서 BF16·FP8·NVFP4, speculative drafter, fine-tuned adapter, RL checkpoint, hosted endpoint, agent trace와 distillation student가 파생된다. 삭제·라이선스·안전 수정이 필요할 때 base weight 하나만 교체해서는 충분하지 않다. 각 precision과 adapter의 parent hash, 학습 data manifest, serving alias, cache, evaluation result를 DAG로 묶어야 한다. Apache-2.0 weight 공개는 학습 데이터의 완전 공개나 이미 존재하는 파생 배포물의 자동 회수를 의미하지 않는다.

- 공식 출시 발표: [Beam is now available](https://reflection.ai/blog/beam-is-available)
- 공식 모델 카드·weights: [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B)
- 안전성 방법·평가: [Beam Safety Technical Report](https://reflection.ai/beam-safety-report)

### 5.3 두 트렌드를 오늘의 수학으로 읽기

| 비교 축 | Mundo | Beam | 오늘의 교훈 |
| --- | --- | --- | --- |
| 핵심 대상 | 장소별 3D·물리·video world model | text 기반 coding·reasoning·agent LLM | 모델 규모보다 상태와 artifact 경계를 먼저 정의한다. |
| 숨은 상태 | geometry·object pose·physics property | router 경로·long-context agent state | 한 출력만 보고 내부 불확실성을 확정하지 않는다. |
| 시간성 | chunked video와 장기 scene consistency | long-horizon tool execution | 현재 출력뿐 아니라 전체 trajectory를 감사한다. |
| 공개 상태 | project·설명 공개, code/weights는 예정 | 여러 precision weights·일부 OSS 도구·partner integration 공개 | “공개”를 code·weight·data·license로 분해한다. |
| 검증 위험 | 작은 물체·장기 memory·실제 물리 gap | vendor benchmark·data provenance·quantized variant | 저자 수치의 조건과 재현 범위를 함께 적는다. |
| 계보 과제 | context image→proxy→simulation→rendered video | base→precision→adapter→endpoint→agent trace | 파생 DAG 없이는 수정·삭제·회수가 불완전하다. |

두 발표의 공통점은 AI가 단일 정적 checkpoint에서 **오래 살아 있는 상태ful system**으로 이동한다는 것이다. 그러므로 연구자는 accuracy뿐 아니라 posterior uncertainty, session state, artifact lineage, rollback과 deletion evidence를 함께 설계해야 한다.

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

로봇의 관측열 $y_{1:T}$를 설명하는 모수 $\theta$를 추론하려고 한다. 정확한 $p_\theta(y_{1:T})$는 계산할 수 없지만, 파티클 필터가 음이 아닌 불편 추정량 $\widehat p_\theta(y_{1:T};u)$를 준다. 동시에 한 사용자가 자신의 학습 기록 삭제를 요청했고, 이 모델은 DP-SGD로 학습됐으며 SISA checkpoint와 여러 quantized 배포본을 갖고 있다.

다음 세 문장을 한 논리로 설명해 보라.

1. PMMH가 유한 particle의 noisy likelihood를 사용하면서도 왜 단순한 plug-in 근사 MCMC와 다른가?
2. DP 보장이 있는데도 왜 삭제 완료를 바로 선언할 수 없는가?
3. SISA와 lineage DAG를 사용해 무엇을 다시 계산하고 무엇을 폐기·검증해야 하는가?

### 모범 답안

PMMH는 $\widehat p_\theta(y;u)$를 고정된 숫자 근사로 치환하는 것이 아니라, PF 난수 $u$까지 Markov chain의 확장 상태에 포함한다. 확장 목표를

$$
\widetilde\pi(\theta,u)
\propto
\widehat p_\theta(y;u)p(\theta)m_\theta(u)
$$

로 두면 likelihood 추정량의 불편성 때문에

$$
\int\widetilde\pi(\theta,u)\,du
\propto
p_\theta(y)p(\theta)
$$

가 된다. 따라서 $\theta$ 주변분포는 원하는 $p(\theta\mid y)$다. 거절 시 현재 $u$와 현재 likelihood estimate를 유지해야 이 구조가 보존된다. 반면 편향된 deterministic surrogate를 likelihood처럼 넣거나 분모 추정값을 매번 새로 뽑는 알고리즘은 일반적으로 다른 목표를 가진다. Particle 수가 작으면 목표가 즉시 틀리는 것이 아니라 likelihood variance가 커져 chain mixing과 ESS가 나빠진다.

DP-SGD의 보장은 인접 데이터셋 $D\sim D'$에 대해

$$
\Pr[M(D)\in S]
\le
e^\varepsilon\Pr[M(D')\in S]+\delta
$$

라는 학습 결과분포의 안정성이다. 이 식은 한 사례의 참여 영향이 제한됐음을 말하지만 다음을 자동으로 증명하지 않는다.

- 실제 privacy unit이 요청자의 모든 레코드·대화·파생물과 일치하는가?
- 누적 $(\varepsilon,\delta)$가 충분히 강한가?
- raw data와 오래된 checkpoint가 물리적으로 폐기됐는가?
- adapter·quantized copy·student·index·cache·replica가 갱신됐는가?
- unlearned 결과가 deletion retrain reference와 가까운가?

따라서 DP는 중요한 사전 위험 제한이지만 삭제 영수증이 아니다.

SISA에서는 먼저 요청자의 원본·중복·증강·합성 파생물을 entity level로 찾는다. 각 사례가 들어간 shard $k$와 가장 이른 영향 slice $r$을 찾고, 삭제 대상을 제외한 뒤 $M_{k,r-1}$에서 stage $j=r,\ldots,R$을 다시 수행한다. Canonical SISA라면 각 stage $j$는 삭제된 사례를 뺀 누적 prefix $C_{k,j}$를 사용한다. 여러 삭제가 같은 shard에 있으면 가장 이른 영향 slice로 합쳐 계획한다. 그 뒤 aggregate output과 필요하면 aggregation model을 갱신한다.

Lineage DAG에서는 삭제 source에서 도달 가능한 모든 후손

$$
\operatorname{Desc}(s)
$$

을 찾는다. 영향받은 checkpoint, optimizer state, adapter, BF16·FP8·NVFP4 copy, distilled model, embedding index, evaluation cache, container와 serving replica를 quarantined 또는 tombstoned로 바꾸고 새 artifact hash로 대체한다. 마지막으로 동일 SISA deletion-retrain 기준선에 대해 forget efficacy, retain utility, privacy attack, recurrence를 검증하고, 오래된 replica가 traffic을 받지 않는지 확인한 뒤에만 완료를 선언한다.

세 답의 공통 원리는 다음과 같다.

$$
\boxed{
\text{숨은 무작위성과 파생 계보를 상태에서 빼버리면, 편한 답은 얻어도 검증 가능한 답은 잃는다.}
}
$$

---

**다음 학습 예고:** 오늘은 미래 관측으로 과거 경로를 수정하고, 조건부 적분으로 Monte Carlo 분산을 줄이며, PF likelihood를 이용해 모수 posterior까지 추론했다. 다음에는 variational inference·amortized smoothing으로 반복 추론비를 줄이는 방법과, federated learning·secure aggregation·federated unlearning에서 privacy budget과 분산된 계보를 함께 관리하는 방법으로 확장한다.
