# [2026-10-19] 오늘 학습: 전체분산·부트스트랩·분포강건 최적화 & Deep Ensemble 불확실성·강건 정책 학습

> **오늘의 핵심 문장:** 여러 모델이 서로 다른 답을 낸다는 사실은 “모델이 모른다”는 유용한 신호지만, 그 불일치 자체가 곧바로 새 결과의 $95\%$ 예측구간이나 평균의 $95\%$ 신뢰구간이 되지는 않는다. 전체분산법칙으로 불확실성의 출처를 나누고, 부트스트랩 앙상블의 불일치를 별도 검량 데이터로 실제 오차와 연결한 뒤, 배포 시 고객·상황의 비율까지 바뀔 수 있음을 분포강건 최적화로 대비해야 한다.

지난 시간에는 같은 로그에서 많은 정책을 비교할 때 생기는 최댓값 선택 편향을 균등 신뢰경계와 비관적 오프정책 학습으로 다뤘다. 그러나 함수 근사기 하나가 내놓는 표준오차를 정확히 알기 어렵고, 배포 뒤 context 구성까지 바뀐다면 여전히 두 질문이 남는다.

1. **모델이 모르는 곳을 어떻게 발견할까?**
2. **오늘의 평균 사용자 구성이 내일 바뀌어도 좋은 정책을 어떻게 고를까?**

오늘은 이 두 질문을 다음 흐름으로 연결한다.

$$
\text{전체분산법칙}
\longrightarrow
\text{bootstrap deep ensemble}
\longrightarrow
\text{disagreement calibration}
\longrightarrow
\text{group별 보수적 가치}
\longrightarrow
\text{DRO의 최악 혼합비 최적화}
$$

> **난이도 표지:** 오늘 내용은 고급 주제지만 출발점은 단순하다. 여러 선생님이 같은 답을 내면 안심이 되고, 의견이 갈리면 다시 확인하고 싶다. 다만 선생님들이 모두 같은 교재의 같은 오타를 배웠다면 만장일치도 틀릴 수 있다. 앙상블의 불일치를 **측정**하고, 실제 오답률과 대조해 **검량**하며, 구성 변화에 대해 **최악의 경우를 시험**하는 이유가 여기에 있다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 예측값 하나만으로는 부족할까?

회귀 모델이 배송 시간을 $20$분이라고 예측했다고 하자. 이 숫자만으로는 다음 두 상황을 구분할 수 없다.

- 과거에 비슷한 주문을 수십만 건 보아 $20$분이라고 확신한다.
- 처음 보는 지역·날씨라서 잘 모르지만 어쩔 수 없이 $20$분을 출력한다.

두 경우의 점예측은 같아도 행동은 달라야 한다. 첫 번째에는 자동 배차를 해도 되지만, 두 번째에는 범위를 넓혀 안내하거나 사람에게 넘기는 편이 안전하다. 그래서 AI 시스템에는 답뿐 아니라 **그 답을 얼마나 믿어야 하는지**가 필요하다.

### 1.2 왜 불확실성을 둘로 나눌까?

관측값이 흔들리는 이유는 크게 두 가지다.

1. **데이터 자체의 우연성:** 같은 조건에서도 결과가 달라진다. 이를 우연적 불확실성, 즉 **aleatoric uncertainty**라고 한다.
2. **지식 부족:** 학습 데이터가 부족해 어떤 함수가 맞는지 모른다. 이를 인식론적 불확실성, 즉 **epistemic uncertainty**라고 한다.

동전을 던질 때 앞면과 뒷면이 무작위로 나오는 것은 데이터를 더 모아도 사라지지 않는 aleatoric uncertainty다. 반면 동전의 앞면 확률 자체를 아직 잘 모르는 것은 데이터를 더 모으면 줄어드는 epistemic uncertainty다.

이 구분은 대응 방법을 바꾼다.

- aleatoric uncertainty가 크면 더 넓은 예측 구간이나 확률적 의사결정이 필요하다.
- epistemic uncertainty가 크면 데이터 수집, 탐색, 보류, 사람 검토가 필요하다.

### 1.3 왜 모델 여러 개를 학습할까?

신경망은 같은 데이터로 학습해도 초기 가중치, 미니배치 순서, 재표집된 데이터가 달라지면 서로 다른 함수를 배울 수 있다. 관측된 데이터가 충분한 영역에서는 여러 모델이 비슷한 답으로 모이는 경향이 있다. 데이터가 희박하거나 학습 분포 밖인 영역에서는 가능한 함수가 많아 답이 갈릴 수 있다.

따라서 $B$개의 모델을 학습해 예측의 평균과 분산을 보는 **Deep Ensemble**이 등장했다.

$$
\widehat\mu(x)
=
\frac{1}{B}\sum_{b=1}^{B}\widehat\mu_b(x)
$$

$$
s^2_{\mathrm{ens}}(x)
=
\frac{1}{B-1}\sum_{b=1}^{B}
\left(
\widehat\mu_b(x)-\widehat\mu(x)
\right)^2
$$

여기서 $s_{\mathrm{ens}}(x)$가 크면 구성원들이 많이 다투고 있다는 뜻이다.

### 1.4 왜 부트스트랩이 필요한가?

초기값만 바꾼 여러 신경망은 같은 훈련 표본을 모두 본다. 이때 서로 다른 답은 최적화 경로의 차이를 반영하지만, “다른 표본을 받았다면 학습된 함수가 얼마나 달라졌을까?”라는 표본 불확실성을 충분히 드러내지 못할 수 있다.

**부트스트랩(bootstrap)**은 원래 표본 $n$개에서 복원추출로 $n$개를 다시 뽑아 가상 훈련 집합을 만든다. 각 앙상블 구성원이 다른 부트스트랩 표본을 학습하면 데이터 표본 변화에 대한 민감도를 일부 드러낼 수 있다.

그러나 같은 사용자의 여러 요청이나 같은 날짜의 로그가 서로 의존한다면 행 하나씩 뽑아서는 안 된다. 독립에 가까운 사용자·세션·시간 블록을 통째로 재표집해야 한다.

### 1.5 왜 앙상블 분산이 곧 예측구간이나 신뢰구간은 아닐까?

다섯 모델의 예측 표준편차가 $2$라고 해서

$$
\widehat\mu(x)\pm1.96\times2
$$

를 자동으로 $95\%$ 예측구간이라 부를 수는 없다.

- 다섯 모델이 독립적인 사후분포 표본이라는 보장이 없다.
- 모두 같은 편향된 데이터와 구조를 사용해 함께 틀릴 수 있다.
- 구성원 간 상관이 크면 모델 수가 많아도 새 정보가 거의 늘지 않는다.
- 학습 때 보지 못한 종류의 shift에서는 불일치와 실제 오차의 관계도 바뀔 수 있다.

따라서 disagreement는 유용한 **원시 센서**이지, 아직 확률 보장이 붙은 계기판은 아니다. 별도 검량 집합에서 “이 정도 불일치일 때 실제 오차가 어느 정도였는가?”를 연결해야 한다.

### 1.6 왜 conformal calibration을 다시 연결할까?

Conformal prediction은 모델의 내부 확률이 완벽하다고 가정하지 않고, 검량 표본에서 관측한 오차의 순위를 사용한다. 오늘은 conformal을 처음부터 다시 배우기보다, 앙상블 불일치를 실제 오차 척도로 바꾸는 연결 부품으로 사용한다.

회귀 문제에서 앙상블 평균을 $\widehat\mu(x)$, 불일치 척도를 $\widehat\sigma_{\mathrm{ens}}(x)$라 하자. $\mathcal D_{\mathrm{train}}$만 사용해 양의 floor $s_{\min}$과 scale 정의를 고정하고

$$
\widetilde\sigma(x)
=
\max\left\{
\widehat\sigma_{\mathrm{ens}}(x),s_{\min}
\right\}
$$

로 둔 뒤 검량 점수를

$$
S_i
=
\frac{|Y_i-\widehat\mu(X_i)|}
{\widetilde\sigma(X_i)}
$$

로 만들 수 있다. floor는 분모가 $0$이 되는 것을 막는다. 검량 점수의 높은 분위수 $\widehat q$를 구하면

$$
C(x)
=
\left[
\widehat\mu(x)-\widehat q\widetilde\sigma(x),
\widehat\mu(x)+\widehat q\widetilde\sigma(x)
\right]
$$

이라는 예측구간을 만든다. 앙상블 분산의 절대 크기를 그대로 믿지 않고, 검량 데이터가 필요한 배율 $\widehat q$를 결정하게 하는 것이다.

### 1.7 왜 검량만으로 context-mix drift를 해결하지 못할까?

정책의 평균 가치는 어떤 사용자가 얼마나 자주 오는지에 따라 달라진다. 쉬운 요청 $80\%$, 어려운 요청 $20\%$인 오늘의 평균에서 좋은 정책이 내일 어려운 요청이 $40\%$가 되면 나빠질 수 있다.

그룹 $g$의 정책 가치가 $V_g(\pi)$이고 그룹 비율이 $w_g$라면 전체 가치는

$$
V_w(\pi)
=
\sum_{g=1}^{G}w_gV_g(\pi)
$$

다. 여기서 $V_g(\pi)$가 그대로여도 $w_g$가 바뀌면 평균은 달라진다. 이것이 **context-mix drift**다.

Conformal coverage는 예측 집합의 포함률을 다루지만, 어떤 그룹이 얼마나 자주 등장할지까지 자동으로 보호하지 않는다.

### 1.8 왜 분포강건 최적화가 탄생했을까?

분포강건 최적화, 즉 **Distributionally Robust Optimization(DRO)**는 미래 분포 하나를 정확히 맞히려 하지 않는다. 대신 “현재 추정 분포 주변에서 그럴듯한 분포들의 집합”을 만들고, 그 안에서 가장 불리한 분포에서도 좋은 정책을 고른다.

$$
\pi_{\mathrm{rob}}
\in
\arg\max_{\pi\in\Pi}
\min_{Q\in\mathcal P_{\rho}}
\mathbb E_Q[R(\pi)]
$$

- $\Pi$: 후보 정책 집합
- $\mathcal P_{\rho}$: 기준 분포에서 반경 $\rho$ 이내인 모호성 집합
- 안쪽 $\min$: 우리 정책에 가장 불리한 그럴듯한 분포를 찾는 가상 공격자
- 바깥쪽 $\max$: 그 공격을 견디는 정책을 선택하는 학습자

이는 “미래를 정확히 예언하자”가 아니라 “예언이 조금 틀려도 망가지지 않게 하자”는 설계다.

### 1.9 오늘의 세 층을 혼동하지 말자

| 층 | 묻는 질문 | 대표 도구 |
| --- | --- | --- |
| 데이터 소음 | 같은 $x$에서도 $Y$가 얼마나 흔들리는가? | 조건부 분산, 확률적 출력 |
| 모델 지식 부족 | 학습 표본이 달랐다면 함수가 얼마나 달라지는가? | bootstrap, deep ensemble |
| 배포 분포 변화 | 어떤 context가 얼마나 자주 올지가 바뀌면 정책이 견디는가? | ambiguity set, DRO |

한 도구가 세 문제를 모두 자동으로 해결하지 않는다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| random variable $Y$ | 실행 전에는 값이 정해지지 않은 수량 | 실제 보상·오차·배송 시간 등을 나타낸다. |
| expectation $\mathbb E[Y]$ | 같은 실험을 많이 반복했을 때의 평균 | 정책의 평균 품질을 정의한다. |
| variance $\operatorname{Var}(Y)$ | 값이 평균 주변에서 흔들리는 정도 | 불확실성의 크기를 잰다. |
| conditional expectation | 어떤 정보가 주어졌을 때의 평균 | $X=x$ 또는 모델 $\Theta$를 안 뒤의 예측을 뜻한다. |
| law of total variance | 전체 흔들림을 조건 내부와 조건 사이 흔들림으로 나누는 법칙 | aleatoric·epistemic 분해의 뼈대다. |
| aleatoric uncertainty | 데이터 생성 과정 자체의 우연성 | 데이터가 무한히 많아도 남을 수 있다. |
| epistemic uncertainty | 어떤 함수·파라미터가 맞는지 모르는 불확실성 | 관련 데이터를 더 모으면 줄어들 수 있다. |
| parameter $\theta$ | 모델이 학습하는 숫자들의 묶음 | 가중치와 편향을 포함한다. |
| random parameter $\Theta$ | 데이터가 달라질 때 달라질 수 있는 학습 결과를 확률변수로 본 것 | 가능한 모델들의 불확실성을 나타낸다. |
| posterior | 데이터를 본 뒤 가능한 파라미터에 부여하는 분포 | 베이지안 epistemic uncertainty의 이상적 표현이다. |
| ensemble | 여러 모델의 예측을 묶는 방법 | 평균은 안정화하고 불일치는 지식 부족 신호로 쓴다. |
| ensemble member | 앙상블을 이루는 모델 하나 | 서로 다른 seed·표본·구조로 학습할 수 있다. |
| seed | 난수 생성의 시작값 | 초기 가중치와 데이터 순서를 바꾼다. |
| disagreement | 구성원 예측이 서로 다른 정도 | epistemic uncertainty의 대리 신호다. |
| sample variance | IID 표본들이 배후 모집단에서 퍼진 정도의 불편 추정량 | IID 가정 아래 분모에 $B-1$을 쓴다. |
| covariance | 두 양이 함께 움직이는 정도 | 앙상블 구성원 상관을 설명한다. |
| correlation $\rho_c$ | 공분산을 각 표준편차로 표준화한 값 | 구성원 다양성이 실제로 몇 명분인지 결정한다. |
| effective ensemble size | 상관된 $B$개가 독립 모델 몇 개와 비슷한지 나타내는 수 | 3.5절의 동일분산·동일쌍상관 근사에서 모델 수만 세는 오류를 보여 준다. |
| bootstrap | 원표본에서 복원추출해 가상 표본을 만드는 방법 | 표본 변화에 대한 학습기의 민감도를 드러낸다. |
| resampling unit | 재표집하는 독립에 가까운 단위 | 행·사용자·세션·날짜 블록 중 올바른 것을 골라야 한다. |
| out-of-bag sample | 특정 bootstrap 표본에 뽑히지 않은 관측치 | 구성원별 진단에 쓸 수 있지만 naive OOB 재사용은 표준 split calibration을 대체하지 않는다. |
| bagging | bootstrap으로 학습한 예측을 평균내는 방법 | 분산을 줄이고 안정성을 높인다. |
| calibration | 숫자로 표현한 불확실성과 실제 빈도를 맞추는 과정 | disagreement를 실제 오차 규모와 연결한다. |
| calibration set | 모델 학습에 쓰지 않고 검량에 남겨 둔 데이터 | conformal 임계값을 정한다. |
| nonconformity score $S$ | 새 사례가 기존 패턴에서 얼마나 벗어났는지 나타내는 점수 | 큰 값일수록 예측과 덜 맞는다. |
| exchangeability | 표본 순서를 바꿔도 결합분포가 같다는 성질 | split conformal의 유한표본 주변 coverage 조건이다. |
| coverage | 예측구간·집합이 실제 정답을 포함하는 장기 비율 | 폭과 함께 평가해야 한다. |
| marginal coverage | 전체 입력을 평균했을 때의 포함률 | 각 그룹·각 $x$에서 같은 포함률을 뜻하지 않는다. |
| conditional coverage | 특정 $x$나 그룹 안에서의 포함률 | 일반적인 split conformal이 자동 보장하지 않는다. |
| quantile | 데이터를 작은 순서로 세웠을 때 지정한 위치의 값 | conformal 구간의 배율을 결정한다. |
| $s_{\min}$ | train split만으로 고정하는 양의 scale 하한 | disagreement가 $0$이어도 normalized score를 계산하게 한다. |
| context $X$ | 정책이 행동 전에 보는 정보 | 사용자 유형·언어·위험도·시간대 등을 담는다. |
| group $g$ | 운영상 해석 가능한 context 묶음 | 그룹별 가치와 혼합비를 정의한다. |
| mixture weight $w_g$ | 전체 요청 중 그룹 $g$가 차지하는 비율 | 합이 $1$인 확률벡터다. |
| context-mix drift | 그룹 내부 관계는 비슷하지만 그룹 비율이 바뀌는 현상 | 평균 성능 순위를 뒤집을 수 있다. |
| policy $\pi$ | context를 보고 행동을 고르는 규칙 | 오늘 강건하게 학습할 대상이다. |
| group value $V_g(\pi)$ | 그룹 $g$에서 정책 $\pi$의 기대 보상 | 전체 가치를 만드는 부품이다. |
| ambiguity set $\mathcal P_{\rho}$ | 미래 분포로 허용할 그럴듯한 분포들의 집합 | 반경 $\rho$가 보수성의 크기를 정한다. |
| DRO | 모호성 집합 안의 최악 분포에서도 좋은 해를 찾는 최적화 | context 구성 변화에 강한 정책을 만든다. |
| total variation distance | 모든 사건 중 두 분포의 확률 차이 최댓값 | 이산분포에서는 재배치해야 하는 질량 $\frac12\lVert q-w\rVert_1$과 같다. |
| KL divergence | 한 분포가 기준 분포에서 얼마나 정보적으로 멀리 있는지 재는 비대칭 척도 | 부드러운 재가중 ambiguity set을 만든다. |
| Wasserstein distance | 확률질량을 거리 비용을 내고 옮기는 최소 비용 | 연속 context의 근접한 이동을 표현한다. |
| adversarial distribution $q^*$ | 현재 정책에 가장 불리하도록 선택된 분포 | 약한 그룹에 더 큰 가중치를 준다. |
| robust value | 모호성 집합 안에서의 최악 기대 가치 | 평균값 대신 정책 선택 기준으로 쓴다. |
| lower confidence bound | 추정 오차를 감안한 보수적 하한 | 그룹 가치 추정 불확실성을 반영한다. |
| distribution shift | 학습·검량·배포의 분포가 달라지는 현상 | 교환가능성과 평균 성능을 깨뜨릴 수 있다. |
| OOD | out-of-distribution, 학습 분포 밖 | 앙상블 불일치가 커질 수 있지만 반드시 그런 것은 아니다. |
| abstention | 자신 없을 때 자동 결정을 보류하는 것 | 불확실성을 실제 안전 행동으로 바꾼다. |
| selective prediction | 일부 사례만 자동 처리하고 나머지는 보류하는 예측 | coverage·처리율·선택 위험을 함께 본다. |
| proper scoring rule | 참분포를 보고할 때 기대 score가 최대가 되는 규칙 | 부호를 바꾼 proper loss는 기대손실이 최소이며 NLL·Brier loss가 대표적이다. |
| negative log-likelihood | 정답에 준 확률의 음의 로그 | 확률 예측의 날카로움과 오류를 함께 벌점 준다. |
| Brier score | 예측 확률과 원-핫 정답의 제곱오차 | 분류 확률 calibration 진단에 쓰인다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 표기와 문제 설정

학습 데이터를

$$
\mathcal D_{\mathrm{train}}
=
\{(X_i,Y_i)\}_{i=1}^{n_{\mathrm{train}}}
$$

라고 하자. $X_i$는 입력 또는 context, $Y_i$는 결과다. 회귀 모델 하나는 조건부 예측분포

$$
p_{\theta}(y\mid x)
$$

와 그 평균 $\mu_{\theta}(x)$를 학습한다. $\theta$는 신경망 가중치다.

불확실성 분해를 엄밀하게 쓰기 위해 다음 **계층적 예측 구성**을 정의하자.

$$
\Theta\sim\Pi(\cdot\mid\mathcal D),
\qquad
\widetilde Y\mid X=x,\Theta
\sim
p_{\Theta}(\cdot\mid x)
$$

- 베이지안 분석이라면 $\Pi(\cdot\mid\mathcal D)$는 파라미터 posterior다.
- 유한 앙상블의 예측 혼합분포라면 학습된 $B$개 파라미터에 질량 $1/B$씩 둔 경험분포다.
- bootstrap·다른 seed로 얻은 구성원은 이 “가능한 모델 분포”를 근사하려는 공학적 표본이지 자동으로 posterior 표본이 아니다.

이제 새 입력 $x$의 **혼합 예측값** $\widetilde Y$는 두 단계로 흔들린다.

1. 특정 모델 $\Theta=\theta$가 정해진 뒤에도 $\widetilde Y$가 흔들린다.
2. 혼합분포에서 어떤 모델 $\Theta$를 고르는지에 따라 평균 예측이 달라진다.

중요한 경계가 있다. 고정된 데이터에서 optimizer seed만 무작위로 바꿨다고 실제 세계의 $Y$가 그 seed 때문에 생성되는 것은 아니다. $\Theta$가 실제 결과 생성과 무관한 학습 알고리즘 난수일 뿐이라면 실제 $Y$의 조건부분산에 “seed 분산”이 새로 생기지 않는다. 아래 등식은 정의한 posterior·ensemble **혼합 예측분포**에 대해서는 정확하지만, 둘째 항을 현실의 epistemic uncertainty로 해석하는 타당성은 $\Pi$가 plausible model uncertainty를 얼마나 잘 나타내는지에 달려 있다.

### 3.2 전체분산법칙을 직접 유도한다

조건부 평균을

$$
m(\Theta)
=
\mathbb E[\widetilde Y\mid X=x,\Theta,\mathcal D]
$$

라고 하자. 전체 조건부 평균은 반복기대법칙에 의해

$$
\mathbb E[\widetilde Y\mid X=x,\mathcal D]
=
\mathbb E_{\Theta\mid\mathcal D}[m(\Theta)]
$$

다.

분산의 정의에서 시작한다.

$$
\operatorname{Var}(\widetilde Y\mid x,\mathcal D)
=
\mathbb E
\left[
\left(
\widetilde Y-\mathbb E[\widetilde Y\mid x,\mathcal D]
\right)^2
\middle|x,\mathcal D
\right]
$$

괄호 안에 $m(\Theta)$를 더하고 뺀다.

$$
\widetilde Y-\mathbb E[\widetilde Y\mid x,\mathcal D]
=
\bigl(\widetilde Y-m(\Theta)\bigr)
+
\bigl(m(\Theta)-\mathbb E[\widetilde Y\mid x,\mathcal D]\bigr)
$$

제곱하면 세 항이 생긴다.

$$
\begin{aligned}
&\left(\widetilde Y-\mathbb E[\widetilde Y\mid x,\mathcal D]\right)^2\\
&=
\left(\widetilde Y-m(\Theta)\right)^2
+
\left(m(\Theta)-\mathbb E[\widetilde Y\mid x,\mathcal D]\right)^2\\
&\quad+
2\left(\widetilde Y-m(\Theta)\right)
\left(m(\Theta)-\mathbb E[\widetilde Y\mid x,\mathcal D]\right).
\end{aligned}
$$

마지막 교차항을 $\Theta$까지 조건으로 먼저 평균내면

$$
\mathbb E[\widetilde Y-m(\Theta)\mid x,\Theta,\mathcal D]
=0
$$

이므로 사라진다. 따라서

$$
\boxed{
\operatorname{Var}(\widetilde Y\mid x,\mathcal D)
=
\mathbb E_{\Theta\mid\mathcal D}
\left[
\operatorname{Var}(\widetilde Y\mid x,\Theta,\mathcal D)
\right]
+
\operatorname{Var}_{\Theta\mid\mathcal D}
\left(
\mathbb E[\widetilde Y\mid x,\Theta,\mathcal D]
\right)
}
$$

을 얻는다.

- 첫 항은 모델이 고정돼도 남는 조건부 잡음의 평균이므로 aleatoric 성분이다.
- 둘째 항은 가능한 모델들의 평균 예측이 달라지는 정도이므로 epistemic 성분이다.

이 등식은 3.1절에서 지정한 posterior·ensemble predictive mixture에 대해서는 정확하다. 그러나 일반 deep ensemble이 실제 데이터 생성분산을 정확히 이렇게 나눈다는 뜻은 아니다. 구성원이 $\Theta\mid\mathcal D$의 정확한 사후표본이라는 보장도 없으므로, 앙상블 분산을 epistemic uncertainty의 **근사 대리량**으로 사용할 때는 검량이 필요하다.

### 3.3 회귀 Deep Ensemble의 평균과 분산

구성원 $b$가 평균과 조건부 분산을 각각

$$
\widehat\mu_b(x),
\qquad
\widehat\sigma^2_{\mathrm{ale},b}(x)
$$

로 출력한다고 하자. 앙상블 평균은

$$
\widehat\mu(x)
=
\frac{1}{B}
\sum_{b=1}^{B}\widehat\mu_b(x)
$$

다. 먼저 유한한 $B$개 구성원에 동일 가중치 $1/B$를 준 **경험적 혼합분포 자체**의 분산은 전체분산법칙에 의해 정확히

$$
\widehat\sigma^2_{\mathrm{mix}}(x)
=
\underbrace{
\frac{1}{B}\sum_{b=1}^{B}
\widehat\sigma^2_{\mathrm{ale},b}(x)
}_{\text{aleatoric proxy}}
+
\underbrace{
\frac{1}{B}\sum_{b=1}^{B}
\left(
\widehat\mu_b(x)-\widehat\mu(x)
\right)^2
}_{\text{epistemic proxy}}
$$

다. 이 식의 둘째 항 분모는 **현재 $B$개 점에 질량 $1/B$씩 둔 혼합분포**를 계산하므로 $B$다.

반면 $B$개 구성원을 더 큰 가상 모델 생성분포에서 나온 표본으로 보고 그 분포의 예측평균 분산을 추정하려면 표본분산

$$
s^2_{\mathrm{ens}}(x)
=
\frac{1}{B-1}\sum_{b=1}^{B}
\left(
\widehat\mu_b(x)-\widehat\mu(x)
\right)^2,
\qquad B\geq2
$$

를 disagreement 척도로 쓴다. $1/B$와 $1/(B-1)$은 각각 **유한 혼합분포의 분산**과 **배후 모델 분산의 표본 추정량**이라는 다른 대상을 계산한다. 더구나 실제 신경망 구성원이 독립·동일분포 사후표본이라는 보장은 없으므로 어느 식에도 자동 coverage 의미를 붙이면 안 된다.

### 3.4 분류에서는 확률벡터의 불일치를 본다

$K$개 클래스가 있을 때 구성원 $b$의 확률벡터를

$$
\widehat p_b(y=k\mid x)
$$

라고 하자. 평균 확률은

$$
\overline p_k(x)
=
\frac{1}{B}\sum_{b=1}^{B}\widehat p_b(y=k\mid x)
$$

다.

평균 예측의 엔트로피는

$$
H(\overline p)
=
-\sum_{k=1}^{K}\overline p_k\log\overline p_k
$$

이고, 구성원 엔트로피의 평균은

$$
\frac{1}{B}\sum_{b=1}^{B}H(\widehat p_b)
$$

다. 두 값의 차이

$$
\widehat I(Y;\Theta\mid x,\mathcal D)
=
H(\overline p)
-
\frac{1}{B}\sum_{b=1}^{B}H(\widehat p_b)
$$

는 구성원들이 각각은 확신하면서 서로 다른 클래스를 고를 때 커진다. 베이지안 사후 예측 표본이라는 이상화 아래에서는 상호정보량으로 해석할 수 있다. 일반 deep ensemble에서는 역시 epistemic disagreement의 대리량이다.

예를 들어 두 이진 분류기가 각각 $(0.99,0.01)$과 $(0.01,0.99)$를 내면 각 구성원 엔트로피는 작지만 평균은 $(0.5,0.5)$라서 차이가 크다. 반대로 둘 다 $(0.5,0.5)$를 내면 평균 엔트로피는 크지만 구성원 간 불일치는 작다. 전자는 “모델들이 다툰다”, 후자는 “모델마다 데이터 자체가 애매하다고 본다”에 가깝다.

### 3.5 상관된 구성원은 몇 명분인가?

각 구성원의 예측 오차 분산이 $\sigma^2$이고 서로 다른 두 구성원의 오차 상관이 모두 $\rho_c$라고 단순화하자. 평균 오차의 분산은

$$
\begin{aligned}
\operatorname{Var}
\left(
\frac{1}{B}\sum_{b=1}^{B}E_b
\right)
&=
\frac{1}{B^2}
\left[
B\sigma^2+B(B-1)\rho_c\sigma^2
\right]\\
&=
\frac{\sigma^2}{B}
\left[1+(B-1)\rho_c\right].
\end{aligned}
$$

독립 모델 $B_{\mathrm{eff}}$개의 평균 분산 $\sigma^2/B_{\mathrm{eff}}$와 같다고 놓으면

$$
\boxed{
B_{\mathrm{eff}}
=
\frac{B}{1+(B-1)\rho_c}
}
$$

다.

예를 들어 $B=10$, $\rho_c=0.5$이면

$$
B_{\mathrm{eff}}
=
\frac{10}{1+9\times0.5}
\approx1.82
$$

다. 모델 열 개를 돌렸지만 평균오차의 분산 감소 효과는 독립 모델 약 $1.82$개와 같다. 그래서 seed만 바꾸는 것보다 데이터 bootstrap, 구조 변화, feature view 변화 등 **오류의 다양성**이 중요하다. 단, 인위적으로 다양성만 키워 정확도를 망가뜨리는 것도 답은 아니다.

### 3.6 부트스트랩 표본은 어떻게 만들어지는가?

원표본의 인덱스를 $1,\ldots,n$이라 하자. 부트스트랩 표본 $b$는

$$
I_1^{(b)},\ldots,I_n^{(b)}
\overset{\mathrm{iid}}{\sim}
\operatorname{Uniform}\{1,\ldots,n\}
$$

로 인덱스를 복원추출해 만든다.

관측치 $i$가 뽑힌 횟수 $N_i^{(b)}$는 함께

$$
\left(N_1^{(b)},\ldots,N_n^{(b)}\right)
\sim
\operatorname{Multinomial}
\left(n;\frac1n,\ldots,\frac1n\right)
$$

를 따른다.

특정 관측치가 한 번도 뽑히지 않을 확률은

$$
\left(1-\frac1n\right)^n
\longrightarrow
e^{-1}
\approx0.368
$$

이다. 따라서 큰 $n$에서 각 bootstrap 표본에는 원관측치의 약 $63.2\%$가 적어도 한 번 등장하고, 약 $36.8\%$는 out-of-bag으로 남는다.

이 수치는 “훈련 행의 $63.2\%$만 사용한다”는 뜻이 아니다. 중복을 포함한 부트스트랩 표본의 총 행 수는 여전히 $n$이다.

### 3.7 무엇을 재표집해야 하는가?

부트스트랩의 핵심 가정은 재표집 단위가 독립에 가깝다는 것이다.

- 요청이 서로 독립이면 행 단위 bootstrap을 고려할 수 있다.
- 한 사용자의 요청이 묶여 있으면 사용자 단위 cluster bootstrap을 사용한다.
- 시간 의존성이 있으면 연속 구간을 뽑는 block bootstrap을 사용한다.
- 강화학습 trajectory라면 한 trajectory 또는 독립 episode를 통째로 뽑는다.

행 단위로 잘못 쪼개면 같은 사용자의 거의 복제된 정보가 train과 out-of-bag 양쪽에 들어가 불확실성이 작아 보일 수 있다.

또한 bootstrap deep ensemble은 다음을 근사하려는 공학적 방법이다.

$$
\mathcal D
\mapsto
\widehat f_{\mathcal D}
$$

라는 전체 학습 알고리즘이 표본 변화에 얼마나 민감한지 보는 것이다. 베이지안 posterior sampling과 동일하지 않으며, 구조적 편향·누락 변수·모든 구성원이 공유한 데이터 오류를 자동 발견하지 못한다.

### 3.8 앙상블을 conformal로 검량한다

데이터를 적어도 세 부분으로 나눈다.

$$
\mathcal D
=
\mathcal D_{\mathrm{train}}
\cup
\mathcal D_{\mathrm{cal}}
\cup
\mathcal D_{\mathrm{test}}
$$

- $\mathcal D_{\mathrm{train}}$: 앙상블의 모든 구성원을 학습한다.
- $\mathcal D_{\mathrm{cal}}$: disagreement를 실제 오차와 연결한다.
- $\mathcal D_{\mathrm{test}}$: 최종 성능과 coverage를 한 번 평가한다.

$\widehat\mu$, $\widehat\sigma_{\mathrm{ens}}$, $s_{\min}$과 score 정의를 $\mathcal D_{\mathrm{train}}$만으로 고정하고

$$
\widetilde\sigma(x)
=
\max\left\{
\widehat\sigma_{\mathrm{ens}}(x),s_{\min}
\right\}
$$

로 둔다. 이 floor는 disagreement가 $0$일 때 나눗셈이 폭발하는 일을 막는다. 검량 $X$나 $Y$를 비대칭적으로 이용해 $s_{\min}$이나 score를 조정하면 새 $X$가 그 조정에 참여하지 않아 표준 split-conformal의 교환가능성이 깨질 수 있다.

검량 집합 크기를 $n_{\mathrm{cal}}$이라 하고

$$
S_i
=
\frac{|Y_i-\widehat\mu(X_i)|}
{\widetilde\sigma(X_i)}
$$

를 계산한다. 오름차순 점수를

$$
S_{(1)}\leq\cdots\leq S_{(n_{\mathrm{cal}})}
$$

라고 하자. 목표 오류율이 $\alpha\in(0,1)$이면

$$
k
=
\left\lceil
(n_{\mathrm{cal}}+1)(1-\alpha)
\right\rceil
$$

로 두고, $k\leq n_{\mathrm{cal}}$일 때

$$
\widehat q=S_{(k)}
$$

를 쓴다. $k=n_{\mathrm{cal}}+1$이면 유한한 점수만으로 해당 수준을 보장할 수 없어 보수적으로 $\widehat q=+\infty$인 구간을 사용해야 한다.

새 입력의 예측구간은

$$
\boxed{
C(x)
=
\left[
\widehat\mu(x)-\widehat q\widetilde\sigma(x),
\widehat\mu(x)+\widehat q\widetilde\sigma(x)
\right]
}
$$

이다.

$\mathcal D_{\mathrm{train}}$에 조건부로 검량 pair $(X_i,Y_i)$들과 새 pair가 교환가능하면 표준 split conformal의 순위 논리에 의해

$$
\Pr\{Y_{\mathrm{new}}\in C(X_{\mathrm{new}})\}
\geq
1-\alpha
$$

라는 유한표본 주변 coverage를 얻는다.

이 보장은 다음을 뜻하지 않는다.

- 모든 개별 $x$에서 포함확률이 $1-\alpha$다.
- 모든 언어·지역·위험 그룹에서 같은 coverage다.
- 배포 분포가 크게 바뀌어도 그대로 유지된다.
- 구간이 좁거나 의사결정에 유용하다.

따라서 coverage와 함께 평균 폭, 그룹별 coverage, 최악 그룹 폭, abstention rate를 봐야 한다.

### 3.9 검량 배율은 무엇을 고치는가?

만약 앙상블 불일치가 실제 오차보다 전반적으로 절반 크기라면 검량 점수 $S_i$가 대략 $2$ 주변에 모이고 $\widehat q$도 커진다. 반대로 불일치가 오차보다 지나치게 크면 $\widehat q$가 $1$보다 작아질 수 있다.

즉 $\widehat q$는 disagreement를 실제 잔차 단위로 바꾸는 경험적 배율이다. 그러나 하나의 전역 배율로 모든 context의 잘못된 상대 순서를 고칠 수는 없다.

예를 들어 위험 그룹에서 $\widehat\sigma_{\mathrm{ens}}(x)$가 항상 작게 나오고 쉬운 그룹에서만 크게 나온다면 전역 $\widehat q$는 주변 coverage를 맞춰도 위험 그룹을 보호하지 못할 수 있다. 사전에 정한 그룹 안의 exchangeability와 충분한 표본을 정당화할 수 있다면 Mondrian·group-conditional conformal처럼 그룹 안에서 순위를 계산하는 별도 설계를 고려한다. 여러 그룹을 결과를 본 뒤 선택하거나 empirical coverage 주장을 동시에 많이 하면 그때는 multiplicity까지 별도로 조정해야 한다. 다중검정 보정만으로 conditional coverage가 생기는 것은 아니다.

### 3.10 context 혼합모형

$G$개의 그룹이 있고 기준 혼합비가

$$
\widehat w
=
(\widehat w_1,\ldots,\widehat w_G),
\qquad
\widehat w_g\geq0,
\qquad
\sum_{g=1}^{G}\widehat w_g=1
$$

이라고 하자.

정책 $\pi$의 그룹별 가치를 $V_g(\pi)$라 하면 기준 분포 가치는

$$
V_{\widehat w}(\pi)
=
\sum_{g=1}^{G}\widehat w_gV_g(\pi)
$$

다.

하지만 미래 혼합비 $q$는 다를 수 있다. 이산분포의 total variation 거리는

$$
d_{\mathrm{TV}}(q,\widehat w)
=
\frac12
\lVert q-\widehat w\rVert_1
$$

이다. 반경 $\rho$의 모호성 집합을

$$
\mathcal P_{\rho}^{\mathrm{TV}}
=
\left\{
q\in\mathbb R^G:
q_g\geq0,
\ \sum_gq_g=1,
\ \frac12\lVert q-\widehat w\rVert_1\leq\rho
\right\}
$$

로 정의한다.

정책의 강건 가치는

$$
\boxed{
V_{\mathrm{rob}}(\pi)
=
\min_{q\in\mathcal P_{\rho}^{\mathrm{TV}}}
\sum_{g=1}^{G}q_gV_g(\pi)
}
$$

이다. 안쪽 문제는 선형목적함수와 선형제약을 가진 선형계획법이다. 가상 공격자는 허용된 범위에서 가치가 높은 그룹의 질량을 가치가 낮은 그룹으로 옮긴다.

이 해석에는 중요한 경계가 있다.

- 바뀌는 것은 그룹 혼합비 $q_g$뿐이며, $P(X\mid G=g)$와 $P(R,C\mid X,A)$가 유지돼 각 $V_g(\pi)$가 불변이라고 가정한다.
- 정책 행동이 미래 context 도착분포를 바꾸지 않는 contextual-bandit 설정을 가정한다.
- OPE에는 consistency, 기록한 context·history를 조건으로 한 행동의 conditional ignorability, 실제 최종 logging propensity의 정확한 기록이 필요하다.
- target policy가 고를 행동을 logging policy도 양의 확률로 골랐다는 overlap이 필요하다.
- 각 허용 그룹의 가치를 식별할 충분한 데이터가 있어야 한다.
- 실제 미래 혼합비가 선택한 ambiguity set 안에 들어간다는 보장은 별도로 정당화해야 한다.

이 조건이 깨지면 context-mix DRO만으로 concept drift나 장기 상태 전이를 해결할 수 없다.

### 3.11 두 그룹 예제로 평균 순위가 뒤집히는 것을 본다

기준 혼합비가 쉬운 그룹 $80\%$, 어려운 그룹 $20\%$라고 하자.

$$
\widehat w=(0.8,0.2)
$$

두 정책의 그룹별 가치는 다음과 같다.

| 정책 | 쉬운 그룹 | 어려운 그룹 | 기준 평균 |
| --- | ---: | ---: | ---: |
| $\pi_A$ | $0.90$ | $0.40$ | $0.8\times0.90+0.2\times0.40=0.80$ |
| $\pi_B$ | $0.75$ | $0.70$ | $0.8\times0.75+0.2\times0.70=0.74$ |

기준 평균만 보면 $\pi_A$가 좋다.

이제 $\rho=0.2$인 TV ball을 허용하자. 두 그룹에서는 최대 $0.2$의 확률질량을 쉬운 그룹에서 어려운 그룹으로 옮길 수 있다. 최악 혼합비는

$$
q^*=(0.6,0.4)
$$

다.

$$
V_{\mathrm{rob}}(\pi_A)
=
0.6\times0.90+0.4\times0.40
=0.70
$$

$$
V_{\mathrm{rob}}(\pi_B)
=
0.6\times0.75+0.4\times0.70
=0.73
$$

강건 기준에서는 $\pi_B$가 이긴다. $\pi_B$는 기준 평균을 조금 포기하지만 그룹 간 성능 차이가 작아 구성 변화에 덜 취약하다.

여기서 $\rho$는 공짜 안전 손잡이가 아니다.

- 너무 작으면 현실적 drift를 놓친다.
- 너무 크면 관측 가능성이 거의 없는 극단 분포까지 막느라 지나치게 보수적이다.
- 운영에서 관측한 과거 mix 변동, 도메인 지식, 별도 validation 기간으로 정해야 한다.

### 3.12 KL 모호성 집합은 부드러운 재가중을 만든다

기준 비율이 양수라고 하자. KL divergence는

$$
D_{\mathrm{KL}}(q\lVert\widehat w)
=
\sum_{g=1}^{G}
q_g\log\frac{q_g}{\widehat w_g}
$$

이다. KL ball은

$$
\mathcal P_{\rho}^{\mathrm{KL}}
=
\left\{
q\in\Delta_G:
D_{\mathrm{KL}}(q\lVert\widehat w)\leq\rho
\right\}
$$

로 정의한다. $\Delta_G$는 합이 $1$인 확률벡터들의 집합이다.

가치 벡터를 $v_g=V_g(\pi)$라고 하면, 강건 값에는 다음 라그랑주 쌍대 표현이 있다.

$$
\min_{q\in\mathcal P_{\rho}^{\mathrm{KL}}}q^\top v
=
\sup_{\eta>0}
\left\{
-\eta\log
\left(
\sum_{g=1}^{G}\widehat w_g
e^{-v_g/\eta}
\right)
-\eta\rho
\right\}
$$

$0<\eta^*<\infty$인 유한한 dual optimizer가 존재하는 내부 해에서는 최악 재가중치가 정확히

$$
q_g^*
=
\frac{
\widehat w_g e^{-v_g/\eta^*}
}{
\sum_{h=1}^{G}\widehat w_h e^{-v_h/\eta^*}
}
$$

가 된다. 가치가 낮은 그룹일수록 지수적으로 더 큰 가중치를 받는다. $\eta^*$가 작을수록 최저 가치 그룹에 더 집중하고, $\eta^*$가 크면 기준 분포에 가까워진다. $\rho=0$이면 $\eta^*\to\infty$인 극한에서 $q^*=\widehat w$가 되고, ball이 최저 가치 그룹의 점질량을 허용할 만큼 크면 $\eta^*\to0$인 경계해가 나타날 수 있다.

주의할 점은 $\widehat w_g=0$인 그룹에는 $q_g>0$을 줄 수 없다는 것이다. 관측되지 않은 새 그룹의 출현까지 대비하려면 그룹 정의, support 확장, 탐색 데이터 또는 다른 거리 설계가 필요하다.

### 3.13 Wasserstein ball은 가까운 context 이동을 표현한다

그룹이 아니라 연속 context $x$를 다룰 때는 “서울과 부산의 이동”과 “한국어와 전혀 다른 센서 모달리티의 이동”을 똑같이 세고 싶지 않을 수 있다. context 간 임의의 이동 비용 $c(x,x')$를 정의하면 다음 optimal-transport cost로 확률질량을 옮기는 최소 비용을 잴 수 있다.

$$
W_c(Q,\widehat P)
=
\inf_{\gamma\in\Gamma(Q,\widehat P)}
\mathbb E_{(X,X')\sim\gamma}[c(X,X')]
$$

$\Gamma(Q,\widehat P)$는 주변분포가 각각 $Q$와 $\widehat P$인 모든 결합분포의 집합이다. 특히 $c=d$가 metric일 때 이 비용을 $1$-Wasserstein distance $W_1$이라고 부른다.

Wasserstein ambiguity set은

$$
\mathcal P_{\rho}^{W}
=
\{Q:W_c(Q,\widehat P)\leq\rho\}
$$

다. 이제 ground cost가 metric $d$이고 $W_c=W_1$이며, $Q$와 $\widehat P$가 유한한 1차 적률을 갖고, 가치 함수 $f$가 $d$에 대해 $L$-Lipschitz라고 하자. 이 조건에서 Kantorovich–Rubinstein 부등식으로

$$
\left|
\mathbb E_Q[f(X)]-\mathbb E_{\widehat P}[f(X)]
\right|
\leq
L\rho
$$

이므로

$$
\inf_{Q:W_1(Q,\widehat P)\leq\rho}
\mathbb E_Q[f(X)]
\geq
\mathbb E_{\widehat P}[f(X)]-L\rho
$$

를 얻는다. 정책 문제에서는 $f_{\pi}(x)=\mathbb E[R\mid X=x,A\sim\pi(\cdot\mid x)]$로 볼 수 있으며, 이 해석은 context 분포만 움직이고 조건부 reward mechanism은 유지된다는 가정이 필요하다. 입력에 작은 변화가 생겨도 가치가 급격히 변하지 않는 정책이 강건하다는 뜻이지만, context metric $d$가 의미 없으면 Wasserstein ball도 의미가 없다.

### 3.14 분포 불확실성과 가치 추정 불확실성을 함께 다룬다

DRO의 $q$는 **어떤 그룹이 얼마나 자주 올지**의 불확실성을 다룬다. 그런데 $V_g(\pi)$ 자체도 유한한 로그에서 추정했으므로 불확실하다.

그룹별 추정값과 보수적 반경을

$$
\widehat V_g(\pi),
\qquad
r_g(\pi)
$$

라고 하고, 동시적으로 유효한 하한

$$
L_g(\pi)
=
\widehat V_g(\pi)-r_g(\pi)
$$

를 만들었다고 하자. 그러면 이중 비관적 목적은

$$
\boxed{
\max_{\pi\in\Pi}
\min_{q\in\mathcal P_{\rho}}
\sum_{g=1}^{G}q_gL_g(\pi)
}
$$

이다.

- $r_g$는 표본·모델 추정오차를 방어한다.
- $q$는 미래 context 혼합비 변화를 방어한다.

여기서 $L_g(\pi)$는 **그룹 평균 정책가치**에 대한 DR·IPS·AIPW 추정량의 동시 LCB 또는 독립 평가 하한이어야 한다. 3.8절의 conformal 구간은 새 개별 결과 $Y$를 위한 prediction interval이므로, 그 아래 끝점을 그룹 평균가치의 confidence bound로 대신 쓰면 안 된다. 같은 로그에서 여러 정책 중 하나를 고른다면 $L_g(\pi)$는 정책군 전체에 동시에 유효해야 하며, 그렇지 않으면 선택을 끝낸 뒤 독립 final gate에서 잠근 정책을 다시 검증해야 한다.

두 장치는 서로 다른 위험을 다루므로 구분해야 한다. 반대로 같은 불확실성을 두 번 과도하게 벌점 주면 정책이 지나치게 보수적일 수 있다. 반경의 의미와 데이터 분할을 명시해야 한다.

### 3.15 강건 정책의 경사

정책을 파라미터 $\phi$로 나타내고

$$
J(\phi)
=
\min_{q\in\mathcal P_{\rho}}
\sum_gq_gV_g(\pi_{\phi})
$$

라고 하자. ambiguity set이 $\phi$에 의존하지 않고, 현재 $\phi$에서 최악 분포가 유일한 $q^*(\phi)$이며 필요한 미분 가능 조건이 성립하면 Danskin 정리에 따라

$$
\nabla_{\phi}J(\phi)
=
\sum_gq_g^*(\phi)
\nabla_{\phi}V_g(\pi_{\phi})
$$

다. 최악 분포가 여러 개면 $J$는 미분 불가능할 수 있다. active minimizer 집합을

$$
\mathcal Q^*(\phi)
=
\arg\min_{q\in\mathcal P_{\rho}}
\sum_gq_gV_g(\pi_{\phi})
$$

라고 하자. 방향 $d$에 대한 방향미분은 적절한 정칙성 아래

$$
J'(\phi;d)
=
\min_{q\in\mathcal Q^*(\phi)}
\left[
\sum_gq_g\nabla_{\phi}V_g(\pi_{\phi})^\top d
\right]
$$

처럼 모든 active minimizer의 방향기울기 중 최소로 정해진다. 추가 concavity·regularity 없이 active gradient 하나를 보통의 gradient나 subgradient라고 단정하면 안 되며, 실제 구현은 nonsmooth 최적화 규칙을 명시해야 한다.

학습은 직관적으로 두 단계를 번갈아 수행한다.

1. 현재 정책에 가장 불리한 허용 혼합비 $q^*$를 찾는다.
2. 그 혼합비로 가중한 목적을 상승시키는 방향으로 정책을 갱신한다.

이는 모든 그룹을 똑같이 평균내는 학습과 다르다. 다만 한 번의 갱신이 강조된 각 그룹의 가치를 모두 개선한다는 보장은 없다. 강건 목적이 개선되면서 공격자가 다른 약한 그룹으로 이동하면 병목이 차례로 드러날 수 있다.

### 3.16 안전 cost에는 최악 상한을 쓴다

보상 $R$은 높을수록 좋지만 안전 cost $C_j$는 낮을수록 좋다. 따라서 강건 제약 정책은 예를 들어

$$
\max_{\pi}
\min_{q\in\mathcal P_{\rho_R}^{R}}
\sum_gq_gL_{R,g}(\pi)
$$

subject to

$$
\max_{q\in\mathcal P_{\rho_{C,j}}^{C_j}}
\sum_gq_gU_{C_j,g}(\pi)
\leq
d_j
\qquad
\text{for every }j
$$

처럼 쓸 수 있다.

$L_{R,g}$는 보상 하한, $U_{C_j,g}$는 cost 상한이다. 보상과 cost의 현실적 shift 범위가 다르면 ambiguity set과 반경도 별도로 정해야 한다.

### 3.17 오늘 수학과 AI 부품의 1:1 대응

| 수학 부품 | AI 시스템에서의 정확한 역할 |
| --- | --- |
| 전체분산법칙 | 결과 소음과 모델 간 평균 예측 차이를 개념적으로 분리한다. |
| 표본분산 | ensemble disagreement의 크기를 계산한다. |
| 공분산·상관 | 모델 수가 아니라 오류 다양성이 중요한 이유를 설명한다. |
| bootstrap | 학습 표본 변화에 따른 함수의 흔들림을 모사한다. |
| 순위·분위수 | disagreement를 finite-sample conformal 구간으로 검량한다. |
| 확률 simplex | 가능한 context 혼합비를 표현한다. |
| TV·KL·Wasserstein 거리 | “그럴듯한 미래 분포”의 경계를 정의한다. |
| 안쪽 최소화 | 현재 정책의 가장 취약한 context 구성을 찾는다. |
| Danskin 정리 | 최악 분포가 강조한 그룹으로 정책 경사를 계산한다. |
| LCB·UCB | 가치 추정 오차까지 보수적으로 반영한다. |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Deep Ensemble은 어떻게 만드는가?

회귀 문제를 기준으로 가장 단순한 절차는 다음과 같다.

1. 학습·검량·최종 테스트 데이터를 먼저 분리한다.
2. 사용자나 세션 같은 독립 단위로 $B$개의 bootstrap 학습 집합을 만든다.
3. 각 집합에서 다른 seed로 동일한 구조의 신경망을 학습한다.
4. 새 입력에서 $B$개 평균 예측과 disagreement를 계산한다.
5. 검량 집합에서 normalized residual의 conformal 분위수를 구한다.
6. 평균·검량 구간·그룹별 불확실성을 함께 출력한다.

구조는 다음과 같다.

```text
원시 로그
   |
   +--> train 단위별 bootstrap --> model 1 --+
   |                         --> model 2 --+--> 평균 예측 + disagreement
   |                         -->   ...    --+
   |                         --> model B --+
   |
   +--> calibration set ---------------------> 실제 잔차로 배율 q 보정
   |
   +--> untouched test ----------------------> coverage·폭·그룹 위험 평가
```

분류에서는 각 모델의 softmax 확률을 평균할 수 있다. 그러나 logits를 평균한 뒤 softmax하는 것과 softmax 확률을 평균하는 것은 일반적으로 다르다.

$$
\operatorname{softmax}
\left(
\frac1B\sum_bz_b
\right)
\neq
\frac1B\sum_b\operatorname{softmax}(z_b)
$$

서로 다른 예측분포의 mixture를 만들려면 보통 오른쪽처럼 확률을 평균한다.

### 4.2 다양성은 어떻게 만들고 어떻게 망가지는가?

구성원 다양성을 만드는 흔한 축은 다음과 같다.

- 초기 가중치 seed
- 미니배치 순서와 augmentation
- 사용자·세션 단위 bootstrap 표본
- 일부 feature view 또는 모델 구조
- 서로 다른 사전학습 checkpoint

하지만 다양성이 많다고 무조건 좋지는 않다.

- 성능이 매우 나쁜 모델을 섞으면 평균 정확도가 떨어진다.
- 데이터 누수나 동일한 라벨 오류는 모든 모델이 공유한다.
- 서로 다른 모델이 사실상 같은 사전학습 계보라면 오류 상관이 높을 수 있다.
- 하이퍼파라미터를 테스트 데이터에 맞춰 고르면 앙상블도 선택 편향을 겪는다.

구성원별 정확도, pairwise error correlation, ensemble 정확도, OOD 성능을 함께 봐야 한다.

### 4.3 회귀 출력은 평균과 로그분산을 함께 학습할 수 있다

모델이 Gaussian 조건부분포를 가정해

$$
p_{\theta}(y\mid x)
=
\mathcal N
\left(
y;
\mu_{\theta}(x),
\sigma^2_{\theta}(x)
\right)
$$

를 출력한다고 하자. 한 표본의 음의 로그우도는 상수항을 제외하면

$$
\mathcal L(x,y)
=
\frac{(y-\mu_{\theta}(x))^2}
{2\sigma^2_{\theta}(x)}
+
\frac12\log\sigma^2_{\theta}(x)
$$

다.

첫 항만 보면 모델은 어려운 사례에서 $\sigma^2$를 무한히 키워 오차를 숨기고 싶다. 둘째 로그항이 지나치게 큰 분산을 벌점 준다. 실무에서는 양수를 보장하기 위해

$$
\sigma^2_{\theta}(x)
=
\operatorname{softplus}(s_{\theta}(x))+\epsilon
$$

처럼 raw 출력 $s_{\theta}$를 변환한다. 여기서 $\epsilon>0$은 분산이 정확히 $0$이 되지 않게 하는 작은 수치 안정화 상수이며, conformal scale floor $s_{\min}$과는 다른 용도다.

각 구성원의 $\sigma^2_{\theta_b}(x)$는 aleatoric proxy이고, 구성원 평균 $\mu_{\theta_b}(x)$의 분산은 epistemic proxy다.

### 4.4 분류 confidence와 epistemic uncertainty는 다르다

softmax 최댓값이 $0.99$여도 모델이 OOD 입력에 과신할 수 있다. 하나의 모델이 내는 confidence는 “그 모델 내부에서 클래스 점수가 얼마나 벌어졌는가”를 말할 뿐, 다른 plausible model이 동의하는지는 말하지 않는다.

다음 지표를 구분하자.

- **maximum probability:** 평균 예측이 한 클래스에 얼마나 몰렸는가
- **predictive entropy:** 평균 예측분포가 얼마나 퍼졌는가
- **expected entropy:** 각 구성원이 개별적으로 얼마나 애매해하는가
- **entropy difference:** 구성원 사이의 의견 차이에 가까운 신호
- **error calibration:** 확률 $0.8$ 사례가 실제로 약 $80\%$ 맞는가
- **conformal coverage:** 정답 집합이 목표 비율로 실제 정답을 포함하는가

이들은 서로 관련되지만 동일하지 않다.

### 4.5 disagreement를 실제 행동으로 바꾸는 규칙

불확실성 숫자를 계산만 하고 의사결정에 쓰지 않으면 운영 가치가 없다. 예를 들어 다음 규칙을 사전에 정할 수 있다.

| 조건 | 행동 |
| --- | --- |
| 예측구간이 좁고 정책 이점의 LCB가 양수 | 자동 실행 후보 |
| 예측구간이 넓지만 안전 cost 상한은 낮음 | 추가 정보 요청 또는 저위험 fallback |
| 모델 간 클래스 선택이 충돌 | 사람 검토·도구 검색·재측정 |
| OOD 점수와 disagreement가 모두 큼 | 자동 처리 금지 |
| 모든 모델이 동의하지만 drift detector가 경고 | 공통모드 실패를 의심하고 별도 검증 |

threshold는 테스트 결과를 보며 계속 바꾸면 안 된다. 정책 선택용 데이터와 최종 gate 데이터를 분리해야 한다.

### 4.6 context-mix DRO 정책 학습 파이프라인

```text
로그 + propensity + 결과
        |
        +--> 그룹 정의와 support 점검
        |
        +--> 그룹별 cross-fitted OPE 값 V_g(pi)
        |          |
        |          +--> bootstrap ensemble / 영향함수 --> 동시 LCB L_g(pi)
        |
        +--> 과거 기간별 mix 변화 --> ambiguity radius rho
                                      |
후보 정책 pi ------------------------+--> worst-case q*(pi)
                                              |
                                              +--> robust value
                                              +--> safety-cost worst case
                                              +--> 독립 final gate
```

핵심 구현 순서는 다음과 같다.

1. **그룹을 사전 정의한다.** 언어, 요청 난이도, 위험도처럼 운영상 의미 있고 표본이 충분한 축을 쓴다.
2. **각 그룹에서 정책 가치를 추정한다.** 적응 로그라면 consistency·conditional ignorability·overlap을 점검하고 당시의 실제 propensity와 DR/AIPW를 사용한다. Cross-fitting은 nuisance overfit를 줄일 뿐 이 식별 가정을 대신하지 않는다.
3. **그룹별 불확실성을 계산한다.** 독립 단위 bootstrap이나 영향함수로 LCB·UCB를 만든다.
4. **미래 mix 범위를 정한다.** 과거 주별 변동, 계약상 트래픽 변화, 스트레스 시나리오를 반영한다.
5. **후보별 최악 혼합을 푼다.** TV ball이면 작은 선형계획으로 계산할 수 있다.
6. **강건 가치와 안전 제약으로 후보를 고른다.** 그룹별 후보–기준 정책의 paired 개선량 $\Delta_g$에 동시 LCB를 만들고, $\min_{q\in\mathcal P_{\rho}}\sum_gq_gL_{\Delta,g}$가 요구 margin을 넘지 못하면 기준 정책으로 fallback한다.
7. **독립 데이터에서 잠근 후보를 검증한다.** 통과 후 shadow와 무작위 canary로 이어 간다.

### 4.7 반경 $\rho$는 데이터로만 자동 결정되지 않는다

$\rho$는 “얼마나 나쁜 미래까지 대비할 것인가?”라는 운영 선택이다. 과거 변동만으로 정하면 과거에 없던 사건을 놓치고, 임의로 크게 정하면 시스템이 지나치게 보수적이 된다.

다음 세 근거를 함께 기록하는 것이 좋다.

- **경험적 근거:** 과거 기간별 mixture distance의 높은 분위수
- **시나리오 근거:** 신규 국가 출시, 야간 트래픽 급증, 공격 캠페인
- **의사결정 근거:** robust value 곡선이 어느 반경에서 급락하는지

정책별로

$$
\rho
\mapsto
V_{\mathrm{rob},\rho}(\pi)
$$

를 그리면 한 점의 순위보다 훨씬 많은 것을 볼 수 있다. 작은 $\rho$에서 순위가 자주 뒤집히는 정책은 mix에 민감하다.

### 4.8 DRO가 하지 못하는 일

DRO는 ambiguity set 안의 shift만 방어한다.

- 라벨 정의가 바뀌는 concept drift
- 새 행동의 등장
- 기록되지 않은 교란 변수
- reward hacking
- 잘못된 그룹 가치 추정
- 지원 밖의 context

는 별도 대책이 필요하다. 잘못된 ambiguity set에 대한 최적해는 잘못된 문제를 아주 잘 푼 답일 뿐이다.

### 4.9 초보자가 흔히 하는 오해와 교정

| 흔한 오해 | 왜 틀렸는가 | 올바른 습관 |
| --- | --- | --- |
| “모델 다섯 개면 Bayesian posterior 표본 다섯 개다.” | seed·SGD 구성원은 정확한 posterior draw가 아니다. | disagreement를 proxy라 부르고 별도 검량한다. |
| “앙상블 표준편차의 $1.96$배가 $95\%$ 구간이다.” | 정규성·독립성·스케일 정합이 보장되지 않는다. | calibration set에서 conformal 배율을 구한다. |
| “모델들이 동의하면 확실히 맞다.” | 같은 데이터·구조의 공통 편향으로 함께 틀릴 수 있다. | 독립 데이터, shift test, 다른 계보 모델을 사용한다. |
| “행 bootstrap을 하면 언제나 된다.” | 사용자·세션·시간 의존성을 깨뜨린다. | 독립에 가까운 단위를 통째로 재표집한다. |
| “Out-of-bag 데이터가 있으니 calibration split은 필요 없다.” | naive OOB 집합은 구성원마다 다르고 전체 모델 선택에 재사용될 수 있다. | 오늘은 독립 calibration·test를 보존한다. OOB를 쓰려면 jackknife+·CV+ 같은 전용 절차와 가정을 따른다. |
| “Conformal이면 모든 그룹에서 $90\%$ coverage다.” | 표준 보장은 주변 coverage다. | 그룹별 coverage와 표본수를 별도로 보고한다. |
| “DRO는 가능한 모든 미래를 막는다.” | 정한 ambiguity set 밖의 미래는 보호하지 않는다. | 반경과 거리의 운영 의미를 문서화한다. |
| “$\rho$를 크게 할수록 무조건 안전하다.” | 유용성을 잃고 현실성 없는 최악 상황에 최적화할 수 있다. | robustness curve와 비용을 함께 본다. |
| “DRO가 낮은 성능 그룹을 고쳐 준다.” | 모델 구조에 개선 능력이 없으면 가중치만 바뀐다. | 데이터·표현·행동 공간을 함께 개선한다. |
| “앙상블 불일치와 context drift는 같은 불확실성이다.” | 하나는 함수 추정, 다른 하나는 입력 혼합비 문제다. | LCB 반경과 ambiguity set을 분리한다. |

### 4.10 최소 실험 설계

오늘 개념을 작은 프로젝트로 확인하려면 다음 실험이 좋다.

1. 두 개의 입력 그룹을 가진 합성 회귀 데이터를 만든다.
2. 한 그룹의 학습 표본을 의도적으로 적게 둔다.
3. 사용자 단위 bootstrap으로 신경망 또는 작은 회귀 모델 $10$개를 학습한다.
4. 그룹별 RMSE와 ensemble disagreement를 비교한다.
5. 별도 calibration set으로 $90\%$ conformal 구간을 만든다.
6. 전체 coverage와 그룹별 coverage·평균 폭을 측정한다.
7. 미래 그룹 비율을 변화시키며 두 정책의 평균 가치를 계산한다.
8. $\rho\in\{0,0.05,0.10,0.20\}$에서 robust value 곡선을 그린다.

반드시 기록할 표는 다음과 같다.

| 항목 | 전체 | 그룹 1 | 그룹 2 |
| --- | ---: | ---: | ---: |
| 표본 수 |  |  |  |
| RMSE |  |  |  |
| 평균 disagreement |  |  |  |
| $90\%$ conformal coverage |  |  |  |
| 평균 구간 폭 |  |  |  |
| abstention rate |  |  |  |

### 4.11 운영 체크리스트

- [ ] 재표집 단위가 실제 독립 구조와 맞는가?
- [ ] 앙상블 구성원들이 calibration label을 보지 않았는가?
- [ ] 최종 test를 threshold 선택에 재사용하지 않았는가?
- [ ] 구성원별 정확도와 pairwise error correlation을 확인했는가?
- [ ] coverage뿐 아니라 구간 폭과 그룹별 결과를 봤는가?
- [ ] context 그룹이 사후적으로 유리하게 선택되지 않았는가?
- [ ] 각 그룹에 overlap과 충분한 유효표본크기가 있는가?
- [ ] $\rho$와 거리 함수의 현실적 의미를 설명할 수 있는가?
- [ ] 보상과 안전 cost에 별도 최악 방향을 적용했는가?
- [ ] ambiguity set 밖의 shift를 감지할 모니터가 있는가?
- [ ] 기준 정책 fallback과 canary rollback 규칙이 있는가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 기준 시각: **2026-10-19 KST**. 아래 내용은 공식 모델 카드와 논문 원문을 우선해 확인했다. 모델 카드의 벤치마크는 공급사 측 측정이며, 논문은 아직 동료평가가 끝난 출판물이 아닌 arXiv v1이라는 한계를 함께 적는다.

### 5.1 NVIDIA Dory 1.0 4B: 같은 가중치를 반복해 추론 깊이를 조절한다

NVIDIA는 2026년 10월 16일 **NVIDIA-Nemotron-Labs-Dory1.0-4B**를 open-weight로 공개했다. open-weight는 학습된 가중치를 내려받을 수 있다는 뜻이지, 코드·데이터·라이선스까지 모두 전통적인 open source라는 뜻은 아니다. 공식 모델 카드에 따르면 이 모델은 $4$B, 즉 약 $40$억 개($\mathrm{B}=10^9$) 파라미터의 looped, 즉 latent-recurrent normalized-GPT다.

구조를 단순화하면 입력 블록 $I$, 공유 순환 블록 $R$, 출력 블록 $O$로 쓸 수 있다.

$$
h^{(0)}=I(x)
$$

$$
h^{(k+1)}
=
R\left(h^{(k)},I(x);\theta_R\right)
$$

$$
y=O\left(h^{(K)}\right)
$$

$h^{(k)}$는 $k$번째 반복 뒤의 잠재, 즉 토큰을 출력하기 전 내부 표현이고, $K$는 총 반복 횟수다. 같은 파라미터 $\theta_R$를 가진 순환 블록을 $K$번 적용한다. 공식 카드에는 입력·출력 블록이 각각 $8$개 층, 반복 블록이 $20$개 층이라고 적혀 있다. sliding-window attention은 각 토큰이 가까운 구간만 보고 계산량을 줄이는 방식이고, full attention은 허용된 전체 context를 보는 방식이다. Dory는 두 방식을 층 사이에 섞는다.

핵심은 **하나의 checkpoint에서 loop 수를 바꿔 추론 계산량을 조절**한다는 점이다. checkpoint는 학습이 끝난 모델의 가중치와 설정을 저장한 묶음이다.

- $1$ loop: 계산량을 줄이는 빠른 모드
- $2$ loops: 카드가 기본 시작점으로 권하는 균형 모드
- $4$ loops: 모델이 도구를 쓰며 여러 단계를 수행하는 일부 agentic task에서 정확도를 더 중시하는 모드

공식 카드의 NVIDIA 측 동일 **harness**, 즉 같은 평가 코드·설정으로 돌린 결과에서 SWE-bench Verified는 $1/2/4$ loops에 대해 각각 $50.3/59.5/61.8$로 증가했다. SWE-bench Verified는 실제 GitHub 이슈를 해결하도록 하는 코딩 평가다. 그러나 전문가급 과학 질의응답 평가인 GPQA Diamond는 $69.19/72.41/72.16$이어서 loop를 늘린다고 모든 지표가 단조롭게 좋아지는 것은 아니다. 실제 선택에는 토큰당 계산 증가, 출력 토큰 감소, end-to-end latency를 함께 재야 한다.

그 밖의 공개 정보는 다음과 같다.

- 최대 context: $256$K tokens. $\mathrm{K}=10^3$이고 token은 모델이 텍스트를 나눠 읽는 기본 단위다.
- 공개 형식: BF16은 16-bit 부동소수점 가중치, MXFP8·NVFP4는 각각 더 낮은 정밀도의 8-bit·4-bit 계열 형식, Q5_K_M GGUF는 llama.cpp 계열 로컬 실행용 양자화 파일
- 사전학습: 카드 기준 $6$T tokens, 이어 $300$B-token context extension. $\mathrm{T}=10^{12}$다.
- 사용 조건: OpenMDW 1.1 사용자 정의 라이선스
- 언어·입력: 영어 중심의 text input/output
- 공개 당시 고속 LLM serving 엔진 vLLM 지원은 공식 안정판인 upstream release가 아니라, 본 저장소에 합치기 전 변경 제안인 PR 설치가 필요하다고 명시
- llama.cpp도 안정 upstream release가 아니라 공식 카드가 안내한 Dory-enabled source의 고정 revision 빌드가 필요

**엔지니어 인사이트 (Impact):** 기존 test-time scaling은 더 긴 reasoning token을 생성하거나 더 큰 모델로 바꾸는 방식이 많았다. Dory는 출력 전에 잠재표현에서 같은 블록을 반복하는 **깊이 방향의 가변 계산**을 보여 준다. 한 checkpoint로 latency tier를 나눌 수 있다는 점은 serving에 매력적이지만, loop 수를 늘릴수록 무조건 좋아진다는 가정을 버리고 작업별 robust curve를 측정해야 한다. 오늘 배운 관점으로 보면 loop 수 자체를 action으로 두고, context별 가치와 불확실성을 추정해 안전하게 routing하는 문제로 확장할 수 있다. 다만 과거 로그로 loop-routing 정책의 인과적 가치를 평가하려면 당시 routing 확률을 기록한 무작위 탐색과 action overlap이 필요하다.

- 공식 모델 카드: [NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
- 공식 모델 컬렉션: [Nemotron Labs Dory](https://huggingface.co/collections/nvidia/nemotron-labs-dory)
- 라이선스: [OpenMDW 1.1](https://openmdw.ai/license/1-1/)

### 5.2 VLM은 상충하는 텍스트·이미지를 편향된 Bayesian observer처럼 결합하는가?

2026년 10월 14일 arXiv v1으로 제출된 **“Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers”**는 이미지와 텍스트가 서로 다른 답을 지지할 때 VLM이 어느 쪽을 따르는지 연구했다.

연구진은 DocVQA, VQAv2, RealworldQA에서 만든 $2{,}765$개 사례를 사용하고, 텍스트 단어 치환 비율과 이미지 Gaussian blur를 각각 $6$단계로 바꿔 $6\times6$ 신뢰도 격자를 만들었다. 주요 실험은 Qwen2.5-VL-3B, Qwen3.5-4B, Gemma3-4B, GLM-4.6V-Flash 네 개의 open-weight VLM을 대상으로 했다.

텍스트가 지지하는 답을 $a_T$, 이미지가 지지하는 답을 $a_V$라 하자. $t$는 text passage, $v$는 image, $q$는 question이며, 이 절의 $q$는 앞선 DRO의 혼합비 벡터 $q$와 다른 지역 기호다. $P(\cdot)$는 모델이 각 후보 답에 부여하는 확률을 뜻한다. 개념적인 Bayesian 해석은 joint preference를 posterior log-odds

$$
s_J^{\mathrm{ideal}}
=
\log
\frac{P(a_T\mid t,v,q)}
{P(a_V\mid t,v,q)}
$$

로 보는 것이다. $s_J^{\mathrm{ideal}}>0$이면 텍스트 답, $s_J^{\mathrm{ideal}}<0$이면 이미지 답을 더 선호한다. 그러나 논문이 실제로 계산한 probe는 각 후보 답의 **토큰당 평균 log-likelihood 차이**다. context를 $c=(t,v,q)$라고 쓰면

$$
\ell(a\mid c)
=
\frac{1}{|a|}
\sum_{j=1}^{|a|}
\log p(a_j\mid c,a_{<j})
$$

$$
s_J^{\mathrm{probe}}
=
\ell(a_T\mid c)-\ell(a_V\mid c)
$$

로 계산한다. 논문은 이 probe score를 $s_J$로 표기해 posterior log-odds처럼 해석하지만, 두 값의 대응은 근사적이다. 단일 modality probe에서 얻은 evidence를 $e_T,e_V$, 질문만 있을 때의 prior log-odds를 $\pi_0$라 두고 다음 두 파라미터 모형을 적합했다.

$$
\boxed{
s_J
=
\beta_Te_T+\beta_Ve_V+\pi_0
}
$$

$\beta_T$와 $\beta_V$는 각 modality evidence를 얼마나 가중하는지를 나타낸다. modality는 텍스트·이미지처럼 서로 다른 정보 채널을 뜻한다. 핵심 결과는 “항상 텍스트를 믿는다” 같은 고정 규칙보다, **각 modality가 주는 증거의 세기와 modality별 통합 편향을 분리**한 이 단순 모형이 보지 않은 동시 열화 조건을 잘 예측했다는 것이다. OOD는 fitting 때 보지 않은 열화 조합이고, $R^2$는 그 조건에서 관측된 preference 변동을 얼마나 설명했는지 나타내며 $1$에 가까울수록 좋다. 논문 표 1에서 데이터셋 평균 OOD $R^2$는 네 주 모델에 대해 약 $0.692$에서 $0.860$ 범위였다.

하지만 한계도 중요하다.

- 후보 답의 길이 정규화 log-likelihood 차이는 실제 open-ended 생성 빈도와 동일하지 않다.
- modality 가중치는 규범적으로 도출한 값이 아니라 경험적으로 적합한 값이다.
- prior belief를 독립적으로 광범위하게 조작하지 않았다.
- 논문은 arXiv v1이며 코드와 데이터는 “출판 시 공개” 예정으로 적혀 있어 10월 19일 기준 즉시 완전 재현 가능한 공개 저장소가 아니다.

**엔지니어 인사이트 (Impact):** VLM 안전평가에서 “텍스트 편향률” 하나만 보고하면 원인을 놓칠 수 있다. 모델이 텍스트를 따른 이유는 텍스트를 과도하게 가중해서일 수도 있고, 현재 사례에서 텍스트 evidence가 실제로 더 강했기 때문일 수도 있다. 센서 품질별로 evidence strength, integration weight, 실제 오류를 분리해 검량해야 한다. 오늘의 앙상블 관점과 결합하면 modality corruption별 disagreement와 coverage를 측정할 수 있다. DRO에서는 corruption level을 사전 정의 그룹·context로 만들고 그룹 내부 value 안정성을 검증해야 한다. blur·OCR noise가 $P(R\mid X,A)$ 자체를 바꾼다면 3.10절의 mix-only DRO 가정 밖이므로, joint·conditional shift를 다루는 별도 Wasserstein 또는 adversarial DRO 설계가 필요하다.

- 논문 원문: [arXiv:2610.19407](https://arxiv.org/abs/2610.19407)
- HTML 본문: [Vision-Language Models Integrate Conflicting Evidence Like Biased Bayesian Observers](https://arxiv.org/html/2610.19407v1)

### 5.3 두 트렌드를 오늘의 학습과 연결한다

| 최신 사례 | 오늘의 수학 연결 | 실무에서 물어야 할 질문 |
| --- | --- | --- |
| Dory의 loop 수 조절 | context별 가치 함수·강건 정책 | 어떤 요청에 몇 loop를 써야 품질 LCB와 latency UCB를 함께 만족하는가? |
| VLM의 상충 modality 결합 | 조건부 evidence·검량·분포 변화 | blur·OCR noise·도메인 mix가 바뀌어도 coverage와 안전성이 유지되는가? |
| 공급사·저자 보고 점수 | bootstrap·독립 test | 같은 harness·seed·트래픽에서 재현되는가? |
| open-weight 공개 | support·license·runtime | 가중치 접근 가능성과 실제 재현 가능성을 혼동하지 않았는가? |

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

Deep Ensemble의 평균 예측이 정확하고 구성원 표준편차가 작은데도, 왜 그 표준편차를 곧바로 새 결과의 $95\%$ 예측구간이나 평균의 $95\%$ 신뢰구간으로 부를 수 없으며, conformal 검량까지 마친 뒤에도 context-mix DRO가 별도로 필요한가?

### 모범 답안

앙상블 표준편차는 여러 학습 모델의 예측이 얼마나 다른지를 나타내는 **epistemic proxy**일 뿐이다. 구성원들이 정확한 posterior의 독립 표본이라는 보장이 없고, 같은 데이터 편향·구조·사전학습 계보를 공유해 함께 틀릴 수 있으며, 구성원 간 상관도 크다. 따라서 $1.96$을 곱한다고 빈도주의 $95\%$ 보장이 생기지 않는다.

별도 calibration set에서

$$
S_i
=
\frac{|Y_i-\widehat\mu(X_i)|}
{\widetilde\sigma(X_i)}
$$

의 conformal 분위수를 구하면, 교환가능성 아래 새 사례에 대한 **주변 예측 coverage**를 얻을 수 있다. 이 단계는 disagreement의 단위를 실제 잔차와 연결한다.

그러나 conformal coverage는 미래에 쉬운 요청과 어려운 요청이 어떤 비율로 올지를 정하지 않는다. 그룹별 정책 가치가 달라지면 혼합비 $q$만 바뀌어도 평균 정책 가치가 크게 달라질 수 있다. 그래서

$$
\min_{q\in\mathcal P_{\rho}}
\sum_gq_gL_g(\pi)
$$

처럼 그럴듯한 context 혼합비 중 최악의 경우를 따로 계산해야 한다.

정리하면 다음 세 문장은 서로 다른 문제를 해결한다.

1. **Ensemble:** 가능한 함수들이 얼마나 다투는가?
2. **Conformal calibration:** 그 불일치를 실제 오차와 비교했을 때 목표 coverage가 나오는가?
3. **DRO:** 배포 context의 구성 자체가 바뀌어도 정책 가치와 안전 제약이 견디는가?

세 단계가 모두 필요할 수 있지만, 어느 것도 support 밖의 완전히 새로운 현상이나 공통모드 오류를 자동으로 해결하지는 않는다.

---

**다음 학습 예고:** 오늘은 모델 불확실성과 context 혼합비 불확실성을 분리해 강건 정책으로 연결했다. 다음에는 ambiguity radius 자체의 불확실성, group 정의 선택 편향, Wasserstein metric 학습을 다루고, change-point detector가 경보를 울릴 때 online DRO와 safe policy rollback을 어떻게 갱신하는지 살펴본다.
