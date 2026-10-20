# [2026-10-21] 오늘 학습: 상태공간모형·HMM·순차 베이지안 업데이트 & 안전한 지속학습

> **오늘의 핵심 문장:** 관측값은 숨은 운영 체제의 불완전한 그림자다. Hidden Markov Model(HMM)의 필터링은 새 증거가 올 때마다 체제에 대한 믿음을 갱신하지만, 그 사후확률은 체제의 확정 라벨도, 신경망을 즉시 다시 학습해도 된다는 허가증도 아니다. 안전한 지속학습은 새 데이터에 적응하는 **가소성**과 과거 능력을 지키는 **안정성**을, 과거 표본을 다시 섞는 replay와 파라미터 변화를 제한하는 regularization으로 함께 관리한다. 그 뒤 실제 행동 없이 새 모델을 관찰하는 shadow와 소수 트래픽에 제한 노출하는 canary를 통과해야만 새 모델을 승격한다.

지난 시간에는 기준 분포에서 정한 운송 거리 이내의 후보 분포 집합인 **Wasserstein ball**로 불확실성을 표현하고, 변화점 경보가 울렸을 때 반경을 즉석에서 늘리지 말고 `HOLD → 원인 분석 → 새 체제 검량 → 독립 승인 → canary` 순서로 대응하는 법을 배웠다. 그런데 실제 서비스는 단 한 번 정상에서 비정상으로 바뀌고 끝나지 않는다. 평일과 주말, 프로모션, 새 언어 유입, 장애, 복구처럼 여러 체제가 반복해서 나타나며 우리가 보는 것은 체제 자체가 아니라 오류율·입력 통계·latency 같은 시끄러운 관측뿐이다.

오늘은 이 상황을 다음 두 층으로 나눈다.

$$
\underbrace{
p(S_t\mid Y_{1:t})
}_{
\text{숨은 운영 체제에 대한 믿음 갱신}
}
\qquad\neq\qquad
\underbrace{
\theta_t\leftarrow\operatorname{Train}(\theta_{t-1},\mathcal D_t)
}_{
\text{배포 모델 파라미터 갱신}
}
$$

- 왼쪽은 **상태 추론**이다. 고정된 HMM 아래에서 현재가 어떤 체제인지 확률적으로 추정한다.
- 오른쪽은 **모델 적응**이다. 가중치가 바뀌므로 과거 능력, 예측확률과 실제 빈도의 일치인 calibration, 안전 경계가 모두 달라질 수 있다.
- HMM의 전이·관측 모수 자체를 다시 추정하는 일은 또 다른 세 번째 작업이다.

오늘의 연결 흐름은 다음과 같다.

$$
\text{시계열 로그}
\longrightarrow
\text{HMM 예측}
\longrightarrow
\text{Bayes 갱신}
\longrightarrow
\text{WATCH/HOLD 판단}
\longrightarrow
\text{replay·regularization 적응}
\longrightarrow
\text{독립 재검량·회귀시험}
\longrightarrow
\text{shadow·canary·rollback}
$$

> **난이도 표지:** HMM은 “보이지 않는 날씨를 빗소리와 우산 판매량으로 추측하는 모형”이다. 지속학습은 “오늘 날씨에 맞춰 옷장을 바꾸되 겨울옷을 전부 버리지 않는 문제”다. 오늘의 핵심은 복잡한 알고리즘 이름보다 **믿음 갱신과 가중치 갱신을 분리하는 것**이다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 한 시점의 경보만으로는 부족할까?

오류율이 오늘 높아졌다고 하자. 가능한 설명은 여러 가지다.

- 표본이 적어 우연히 높아졌다.
- 주말처럼 이미 알고 있는 체제가 돌아왔다.
- 새 도메인이 유입됐다.
- 배포 버그가 생겼다.
- 실제 정답 규칙이 바뀌었다.

한 시점의 숫자만 보면 이 설명들을 구분하기 어렵다. 하지만 어제와 그제의 상태, 각 체제가 얼마나 오래 지속되는지, 그 체제에서 어떤 관측이 흔한지를 함께 보면 판단이 달라진다. 그래서 시간의 연결 구조를 확률모형 안에 넣는 **상태공간모형(state-space model)**이 필요해졌다.

### 1.2 왜 보이지 않는 상태를 굳이 도입할까?

우리가 정말 알고 싶은 것은 “현재 운영 환경이 정상인가, 계절성 피크인가, 새로운 도메인인가, 사고 상태인가?”다. 그러나 로그에는 `상태=사고`라는 정답 열이 자동으로 생기지 않는다. 대신 다음과 같은 흔적만 보인다.

- 입력 임베딩 거리
- 언어·기기·도메인 구성비
- 응답 latency와 timeout
- 사람이 뒤늦게 확정한 오류·사고 라벨
- 안전비용과 사용자 불만

숨은 상태 $S_t$는 이런 관측을 만들어 내는 공통 원인을 요약하는 **모형 속 변수**다. 여기서 “숨었다”는 말은 신비한 실체라는 뜻이 아니라 직접 관측하지 못해 확률로 추론한다는 뜻이다.

### 1.3 왜 시간순서를 무시하면 안 될까?

같은 경보 하나라도 직전 상황에 따라 의미가 다르다.

- 직전까지 오랫동안 정상이었다면 일회성 잡음일 수 있다.
- 이미 장애 가능성이 높았다면 추가 경보는 장애 지속의 증거가 된다.
- 방금 새 버전을 배포했다면 체제 변화가 아니라 배포 개입의 영향일 수 있다.

HMM은 “현재 상태는 바로 이전 상태에 주로 의존한다”는 Markov 가정과 “현재 관측은 현재 상태가 주어지면 설명된다”는 emission 가정을 사용해 이 시간 의존성을 압축한다.

### 1.4 왜 순차 베이지안 업데이트가 필요한가?

Bayes 정리는 새 증거가 왔을 때 기존 믿음을 어떻게 바꿀지 알려 준다.

$$
\text{posterior}
\propto
\text{likelihood}
\times
\text{prior}
$$

시계열에서는 어제의 posterior가 오늘 예측의 출발점이 된다. 모든 과거 로그를 매 순간 처음부터 읽는 대신, 과거 증거를 요약한 belief를 한 단계 앞으로 전이한 뒤 오늘 관측으로 교정한다. 이것이 **예측–갱신(predict–update)** 구조다.

### 1.5 왜 새 데이터만 fine-tuning하면 과거 능력을 잃을까?

신경망의 한 파라미터는 한 기능만 담당하지 않는다. 한국어, 영어, 코드, 안전 거부, 도구 사용 같은 여러 능력이 같은 가중치를 공유한다. 새 체제 데이터만으로 손실을 줄이면 새 작업에는 좋아지는 방향이 과거 작업에는 나쁜 방향일 수 있다.

이때 이전에 잘하던 능력이 크게 떨어지는 현상을 **파국적 망각(catastrophic forgetting)**이라고 한다. 이는 저장 파일이 지워지는 사건이 아니라, 파라미터 공간에서 새 그래디언트가 과거 기능에 필요한 해를 밀어내는 현상이다.

### 1.6 안정성–가소성 딜레마는 왜 생길까?

- **안정성(stability):** 이미 배운 능력과 안전성을 보존하는 힘
- **가소성(plasticity):** 새 데이터와 새 체제에 빠르게 적응하는 힘

안정성만 극대화하면 모델이 새 환경을 배우지 못한다. 가소성만 극대화하면 과거 능력이 무너진다. replay는 과거 예시를 다시 보여 주고, regularization은 중요한 파라미터가 너무 멀리 움직이지 않게 하며, adapter·expert 분리는 충돌할 공간 자체를 줄인다. 어느 방법도 자동 안전 보장은 아니다.

### 1.7 왜 탐지와 학습 승인을 분리해야 할까?

HMM이 drift 상태의 posterior를 높게 냈다는 것은 “현재 관측이 drift 모형에서 더 그럴듯하다”는 뜻이다. 다음을 뜻하지는 않는다.

1. drift가 확정됐다.
2. 원인이 데이터 변화인지 배포 버그인지 밝혀졌다.
3. 현재 데이터가 학습에 안전하다.
4. 지금 fine-tuning하면 과거 능력이 유지된다.
5. 새 모델의 확률과 threshold가 여전히 calibration되어 있다.

따라서 탐지 루프는 `WATCH/HOLD` 신호를 낼 수 있지만, 적응 학습과 배포 승격은 별도의 데이터·시험·승인 루프를 거쳐야 한다.

### 1.8 오늘의 본질

오늘의 질문은 “모델이 스스로 계속 배우게 만들까?”가 아니다.

> **어떤 체제라고 믿는 근거, 무엇을 새로 학습하는 과정, 과거 능력을 지켰다는 증거, 새 모델을 배포할 권한을 서로 분리했는가?**

이 분리가 없으면 빠른 적응은 빠른 오염과 망각이 될 수 있다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| $t$ | 시간 블록의 번호. 요청 하나가 아니라 시간·배치·세션 단위일 수 있다. | 순차 업데이트의 순서를 표시한다. |
| $S_t$ | 시점 $t$의 숨은 상태 또는 체제 | 정상·피크·새 도메인·사고 같은 직접 보이지 않는 상태다. |
| $Y_t$ | 시점 $t$에 실제로 본 관측 | 입력 통계, latency, 오류 라벨 같은 상태의 흔적이다. |
| $Y_{1:t}$ | $1$부터 $t$까지의 모든 관측 | 지금까지 이용 가능한 정보 집합이다. |
| $U_t$ | 배포 버전·정책·가격·프롬프트처럼 우리가 가한 알려진 개입 | 개입 효과를 자연 drift로 오인하지 않게 조건에 넣는다. |
| 잠재 변수(latent variable) | 직접 볼 수 없지만 관측을 설명하도록 모형 안에 둔 변수 | $S_t$가 대표적인 잠재 변수다. |
| 상태공간모형 | 숨은 상태의 시간 변화와 관측 생성 과정을 함께 표현하는 모형 가족 | HMM과 Kalman filter를 한 틀에서 본다. |
| Markov 성질 | 현재 상태가 주어지면 더 먼 과거 상태가 다음 상태 예측에 추가 정보를 주지 않는다는 가정 | 전체 과거를 직전 belief로 압축하게 해 준다. |
| HMM | 상태가 유한한 이산 값인 상태공간모형 | 반복되는 운영 체제를 확률적으로 추적한다. |
| $K$ | 숨은 상태의 개수 | $S_t\in\{1,\ldots,K\}$로 쓴다. |
| $\boldsymbol\pi$ | 첫 상태에 대한 확률 벡터 | 필터의 시작 prior다. |
| $A_{ij}$ | 상태 $i$에서 다음 시점 상태 $j$로 갈 확률 | 전이행렬의 $i$행 $j$열이다. |
| 전이행렬 $A$ | 모든 $A_{ij}$를 모은 표 | 각 행의 합은 $1$이다. |
| $b_j(y)$ | 상태 $j$일 때 관측 $y$가 나올 확률 또는 밀도 | emission likelihood다. |
| emission | 숨은 상태가 관측을 만들어 내는 확률 규칙 | 상태와 로그를 연결한다. |
| prior | 새 관측을 보기 전 믿음 | 예측 단계의 상태 확률이다. |
| likelihood | 특정 상태를 가정했을 때 실제 관측이 얼마나 그럴듯한지 나타내는 값 | prior를 새 증거로 재가중한다. |
| evidence | 모든 상태를 합친 관측의 전체 예측확률 | posterior를 합이 $1$이 되도록 정규화한다. |
| posterior | 관측까지 본 뒤의 갱신된 믿음 | $p(S_t\mid Y_{1:t})$다. |
| belief $\alpha_t(j)$ | 상태 $j$에 대한 filtering posterior | 운영 판단에 쓰는 확률 벡터다. |
| prediction | 관측 전에 전이만 적용해 다음 상태를 예측하는 단계 | $\alpha_t^{-}$로 쓴다. |
| filtering | 현재까지의 관측만으로 현재 상태를 추론하는 일 | 온라인 의사결정에 사용할 수 있다. |
| smoothing | 미래 관측까지 이용해 과거 상태를 다시 추론하는 일 | 사후 분석용이며 온라인 평가에 쓰면 미래 누출이다. |
| Viterbi decoding | 전체 기간을 통틀어 가장 가능성 높은 상태 경로 하나를 찾는 일 | 시점별 posterior 최댓값을 이어 붙인 것과 다를 수 있다. |
| Bayes factor | 두 상태가 같은 관측을 얼마나 다르게 예측했는지 나타내는 likelihood 비 | 관측이 odds를 얼마나 바꾸는지 보여 준다. |
| predictive surprise | 관측 전 예측확률의 음의 로그 $-\log p(y_t\mid Y_{1:t-1})$ | 예상 밖 관측을 재는 점수지만 자동으로 $p$값은 아니다. |
| dwell time | 한 상태에 연속으로 머무는 시간 | 표준 HMM에서는 geometric 분포를 따른다. |
| Kalman filter | 선형·Gaussian 연속 상태공간모형의 정확한 필터 | HMM과 같은 예측–갱신 뼈대를 공유한다. |
| continual learning | 데이터·과제가 시간에 따라 도착할 때 이전 능력을 보존하며 계속 배우는 설정 | 오늘의 AI 기초 주제다. |
| 파국적 망각 | 새 학습 뒤 과거 과제 성능이 크게 떨어지는 현상 | 새 점수만 보면 놓치기 쉽다. |
| 안정성–가소성 | 과거 보존과 새 적응 사이의 긴장 | replay·regularization의 설계 목표다. |
| replay buffer $\mathcal B$ | 과거 예시 일부를 보관한 작은 저장소 | 새 학습 때 과거 손실도 함께 보게 한다. |
| reservoir sampling | 길이를 모르는 스트림에서 각 항목이 같은 확률로 buffer에 남도록 하는 표집 | 평균 대표성에는 유용하지만 희귀 안전군 보존을 보장하지 않는다. |
| regularization | 파라미터나 출력의 변화를 제한하는 추가 벌점 | 과거 기능을 지키는 근사적 장치다. |
| Fisher information $F_i$ | 파라미터 $\theta_i$ 변화에 과거 모형의 로그가능도가 얼마나 민감했는지 나타내는 국소 곡률 근사 | EWC의 파라미터 중요도 가중치로 쓴다. |
| EWC | 과거에 중요했던 파라미터가 기준점에서 멀리 움직이는 것을 벌주는 방법 | replay와 다른 방식으로 안정성을 높인다. |
| distillation | 이전 모델의 출력분포를 새 모델이 흉내 내게 하는 학습 | 과거 행동을 보존하지만 과거 오류도 보존할 수 있다. |
| event time | 실제 사건이 발생한 시점 | 지연 라벨을 올바른 블록에 귀속하는 기준이다. |
| reveal time | 사건의 정답 라벨을 알게 된 시점 | 당시 온라인 정보와 사후 정보를 구분한다. |
| calibration | 예측확률이나 위험 점수와 실제 빈도가 맞도록 조정하는 일 | 가중치가 바뀌면 다시 확인해야 한다. |
| regression test | 새 버전이 기존 기능을 망가뜨리지 않았는지 확인하는 시험 | old-regime 능력을 지키는 배포 gate다. |
| champion·challenger | 현재 승인 모델과 새 후보 모델 | 독립 시험 뒤에만 challenger를 champion으로 승격한다. |
| hysteresis | 진입·해제 임계값을 다르게 두어 상태가 자주 깜빡이지 않게 하는 규칙 | `WATCH↔HOLD` 운영 진동을 줄인다. |
| EM·Baum–Welch | 숨은 상태의 기대값을 계산하는 E-step과 그 기대값으로 모수를 다시 맞추는 M-step을 반복하는 HMM 학습법 | filtering과 달리 $A,b$ 자체를 학습한다. |
| MAP | posterior가 가장 커지는 파라미터 점, 즉 maximum a posteriori 추정치 | Laplace 근사의 중심 $\theta^\star$다. |
| Hessian·precision | Hessian은 2차 편미분 행렬, precision은 분산의 역수에 해당하는 확실성 크기다. | EWC가 과거 posterior의 국소 곡률을 근사할 때 등장한다. |
| entropy | 확률이 여러 상태에 퍼진 정도를 재는 불확실성 값 | $-\sum_j\alpha_j\log\alpha_j$가 크면 상태가 덜 확실하다. |
| LCB·UCB | 추정오차를 각각 빼고 더한 신뢰 하한·상한 | 개선은 보수적 하한, 안전비용은 보수적 상한으로 검사한다. |
| conformal calibration | calibration 자료의 오차점수 분위수로 prediction set이나 위험 cutoff를 정하는 방법 | 모델이 바뀐 뒤 이전 cutoff를 재사용할 수 있는지 확인한다. |
| hard-stop | 확률모형의 평균 판단과 무관하게 치명 조건이 충족되면 즉시 중지하는 규칙 | 낮은 posterior의 대형 사고를 놓치지 않게 한다. |
| shadow·canary | shadow는 실제 행동 없이 후보 출력을 관찰하고, canary는 작은 실제 트래픽에 제한 노출한다. | 전체 승격 전 단계적 검증이다. |
| rollback | 문제가 생기면 검증된 이전 배포 묶음으로 되돌리는 일 | 가중치뿐 아니라 memory·threshold·runtime도 함께 복원한다. |
| OOD | out-of-distribution의 약자. 학습·검량 분포의 지원범위 밖이거나 매우 드문 입력이다. | 새 모델의 미지 환경 회귀를 따로 본다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 문제 설정: 체제는 보이지 않고 흔적만 보인다

시간 블록 $t=1,\ldots,T$마다 숨은 운영 체제를

$$
S_t\in\{1,\ldots,K\}
$$

라고 하자. 예를 들어 $K=3$이면 상태를 다음처럼 해석할 수 있다.

$$
1=\text{정상},
\qquad
2=\text{예상된 피크},
\qquad
3=\text{새로운 drift 또는 사고}
$$

관측은 벡터일 수 있다.

$$
Y_t
=
\left(
\text{입력 거리},
\text{오류 수},
\text{요청 수},
\text{latency},
\text{도메인 구성비}
\right)_t
$$

상태 이름은 모형이 자동으로 붙여 주지 않는다. 학습된 상태 $1$과 상태 $2$를 바꿔도 같은 likelihood를 가질 수 있는 **label switching**이 있으므로, 상태의 의미는 독립 라벨·운영 지식·사후 점검으로 붙여야 한다.

### 3.2 일반 상태공간모형의 결합분포

가장 기본적인 상태공간모형은 다음처럼 분해된다.

$$
p(s_{1:T},y_{1:T})
=
p(s_1)
\prod_{t=2}^{T}
p(s_t\mid s_{t-1})
\prod_{t=1}^{T}
p(y_t\mid s_t)
$$

이 한 줄에는 두 조건부 독립 가정이 들어 있다.

첫째, 1차 Markov 가정이다.

$$
p(s_t\mid s_{1:t-1})
=
p(s_t\mid s_{t-1})
$$

둘째, 현재 상태가 주어지면 현재 관측은 다른 상태와 과거 관측에 추가로 의존하지 않는다는 emission 가정이다.

$$
p(y_t\mid s_{1:t},y_{1:t-1})
=
p(y_t\mid s_t)
$$

이는 현실의 진리가 아니라 계산 가능한 근사다. 같은 요청에서 입력 거리·오류·latency를 함께 만들면 상태를 조건으로 해도 관측 성분들이 서로 의존할 수 있다. 이 경우 결합 emission을 모델링하거나 잔차 의존성을 반영해야 한다.

### 3.3 배포 개입을 빼면 drift와 버그를 혼동한다

시점 $t$에 모델·프롬프트·retriever·runtime 버전 $U_t$를 바꿨다면 전이와 관측에 이를 조건으로 넣는 편이 정직하다. 먼저 시간 순서를 정해야 한다. 여기서는 $U_t$가 블록 $t$의 상태와 관측이 만들어지기 전에 활성화된 설정이라고 약속한다. 그러면 한 가지 생성 모형은

$$
p(s_{1:T},y_{1:T}\mid u_{1:T})
=
p(s_1\mid u_1)
\prod_{t=2}^{T}
p(s_t\mid s_{t-1},u_t)
\prod_{t=1}^{T}
p(y_t\mid s_t,u_t)
$$

이다. 설정이 블록 끝에 바뀌어 다음 상태부터 영향을 준다면 $u_t$ 대신 전이항에 $u_{t-1}$을 써야 한다. 인덱스보다 중요한 것은 실제 배포 시간과 모형의 시간 순서를 일치시키는 것이다.

또한 운영 정책이 과거 로그를 보고 $U_t$를 적응적으로 선택한다면 전체 생성과정에는 결정 규칙

$$
p(u_t\mid H_{t-1})
$$

도 포함되며, $H_{t-1}$은 선택 당시 이용 가능한 history다. 위 조건부분포는 고정·로그된 $u_{1:T}$ 아래의 **예측 모형**이다. 그렇지 않으면 새 runtime이 latency를 바꾼 효과를 환경 drift라고 부르거나, 새 정책이 노출 대상을 바꾼 효과를 자연 체제 전이라고 오해할 수 있다. 그러나 $U_t$에 조건을 붙였다고 인과효과가 자동 식별되는 것도 아니다. 인과 해석에는 실제 받은 버전의 잠재 결과가 관측 결과와 일치한다는 consistency, 비교할 각 버전이 관련 history에서 양의 확률로 배정된다는 overlap, history를 조건으로 한 뒤 버전 배정과 미래 잠재 결과를 함께 좌우하는 숨은 요인이 없다는 sequential exchangeability 같은 추가 가정 또는 무작위 실험이 필요하다.

### 3.4 연속 상태공간모형과 HMM의 관계

2026-10-15에 본 선형 Gaussian 모형은 연속 상태를 쓴다.

$$
Z_t
=
FZ_{t-1}+\eta_t,
\qquad
\eta_t\sim\mathcal N(0,Q)
$$

$$
Y_t
=
HZ_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,R)
$$

이 조건에서는 Kalman filter가 posterior 평균과 공분산을 정확히 갱신한다. HMM은 상태가 연속 벡터가 아니라 유한 집합인 경우다.

$$
S_t\in\{1,\ldots,K\}
$$

따라서 모든 HMM은 이산 상태공간모형이지만, 모든 상태공간모형이 HMM은 아니다. 비선형·비Gaussian 연속 상태에는 extended Kalman filter, unscented Kalman filter, particle filter 같은 근사가 쓰인다.

### 3.5 HMM의 세 가지 핵심 부품

초기 확률은

$$
\pi_j
=
P(S_1=j),
\qquad
\sum_{j=1}^{K}\pi_j=1
$$

이다. 행 기준 전이행렬은

$$
A_{ij}
=
P(S_t=j\mid S_{t-1}=i),
\qquad
A_{ij}\ge 0,
\qquad
\sum_{j=1}^{K}A_{ij}=1
$$

이다. 관측모형은

$$
b_j(y)
=
p(Y_t=y\mid S_t=j)
$$

이다. $Y_t$가 이산이면 확률질량함수, 연속이면 밀도다. 연속 밀도값은 $1$보다 클 수도 있으며 그 자체가 사건 확률은 아니다.

### 3.6 일반적인 순차 Bayes 필터

직전 filtering posterior를

$$
q_{t-1}(s)
=
p(S_{t-1}=s\mid Y_{1:t-1})
$$

라고 하자. 먼저 상태 전이만 적용해 오늘의 prior를 예측한다.

$$
q_t^{-}(s_t)
=
\sum_{s_{t-1}}
p(s_t\mid s_{t-1})
q_{t-1}(s_{t-1})
$$

연속 상태라면 합 대신 적분을 쓴다.

$$
q_t^{-}(z_t)
=
\int
p(z_t\mid z_{t-1})
q_{t-1}(z_{t-1})
\,dz_{t-1}
$$

그다음 새 관측 $y_t$로 갱신한다.

$$
q_t(s_t)
=
\frac{
p(y_t\mid s_t)q_t^{-}(s_t)
}{
\sum_u p(y_t\mid u)q_t^{-}(u)
}
$$

이 식은 이산 상태의 합 표기다. 연속 상태에서는 갱신 분모도 적분이다.

$$
q_t(z_t)
=
\frac{
p(y_t\mid z_t)q_t^{-}(z_t)
}{
\int p(y_t\mid u)q_t^{-}(u)\,du
}
$$

두 경우의 분모는

$$
p(y_t\mid Y_{1:t-1})
$$

이며, 모든 가능한 상태를 고려했을 때 오늘 관측이 나타날 예측확률이다.

### 3.7 HMM forward filtering 공식

belief를

$$
\alpha_t(j)
=
P(S_t=j\mid Y_{1:t})
$$

라고 쓰자. 예측 단계는

$$
\alpha_t^{-}(j)
=
\sum_{i=1}^{K}
\alpha_{t-1}(i)A_{ij}
$$

이고, 갱신 전의 정규화되지 않은 질량은

$$
\widetilde\alpha_t(j)
=
b_j(y_t)\alpha_t^{-}(j)
$$

이다. evidence를

$$
c_t
=
\sum_{k=1}^{K}
b_k(y_t)\alpha_t^{-}(k)
$$

라고 두면 최종 posterior는

$$
\alpha_t(j)
=
\frac{
b_j(y_t)\alpha_t^{-}(j)
}{c_t}
$$

이다. 반드시 **전이로 예측한 뒤 emission으로 갱신**한다. 어제 posterior에 오늘 likelihood를 바로 곱하면 시점 사이의 체제 전이 가능성을 빠뜨린다.

첫 시점에는 이전 상태가 없으므로 $\alpha_1^{-}(j)=\pi_j$로 시작한다.

$$
c_1
=
\sum_{k=1}^{K}\pi_kb_k(y_1),
\qquad
\alpha_1(j)
=
\frac{\pi_jb_j(y_1)}{c_1}
$$

이 $c_1$부터 이후의 $c_t$까지 곱한 값이 관측열의 likelihood가 된다.

### 3.8 두 상태 HMM을 손으로 계산한다

정상 상태 $N$과 drift 상태 $D$가 있다고 하자. 전이행렬과 직전 posterior가 다음과 같다.

$$
A
=
\begin{bmatrix}
0.95 & 0.05\\
0.20 & 0.80
\end{bmatrix},
\qquad
\boldsymbol\alpha_{t-1}
=
\begin{bmatrix}
0.9 & 0.1
\end{bmatrix}
$$

행은 직전 상태, 열은 새 상태다. 관측 전에 drift일 예측확률은

$$
\begin{aligned}
\alpha_t^{-}(D)
&=
P(S_{t-1}=N\mid Y_{1:t-1})P(D\mid N)\\
&\quad+
P(S_{t-1}=D\mid Y_{1:t-1})P(D\mid D)\\
&=
0.9(0.05)+0.1(0.80)\\
&=
0.125
\end{aligned}
$$

따라서 정상 예측확률은 $0.875$다. 이제 $Y_t=1$을 “경보가 울림”으로 정의하고

$$
P(Y_t=1\mid N)=0.1,
\qquad
P(Y_t=1\mid D)=0.7
$$

라고 하자. 경보 자체의 예측확률은

$$
\begin{aligned}
c_t
&=
P(Y_t=1\mid Y_{1:t-1})\\
&=
0.875(0.1)+0.125(0.7)\\
&=
0.175
\end{aligned}
$$

이다. 경보를 본 뒤 drift posterior는

$$
\begin{aligned}
P(D\mid Y_{1:t})
&=
\frac{P(Y_t=1\mid D)P(D\mid Y_{1:t-1})}{P(Y_t=1\mid Y_{1:t-1})}\\
&=
\frac{0.7(0.125)}{0.175}\\
&=
0.5
\end{aligned}
$$

이다. 강한 경보 하나가 drift 확률을 $12.5\%$에서 $50\%$로 높였지만 drift를 확정하지는 않았다. posterior는 증거와 모형 가정 아래의 **불확실한 믿음**이다.

### 3.9 odds와 Bayes factor로 같은 계산을 본다

관측 전 drift 대 정상 odds는

$$
O_t^{-}
=
\frac{0.125}{0.875}
=
\frac{1}{7}
$$

이다. 경보의 Bayes factor는

$$
\operatorname{BF}_{D:N}
=
\frac{P(Y_t=1\mid D)}{P(Y_t=1\mid N)}
=
\frac{0.7}{0.1}
=
7
$$

이다. posterior odds는

$$
O_t
=
O_t^{-}\operatorname{BF}_{D:N}
=
1
$$

이므로 두 상태의 posterior가 각각 $0.5$가 된다. Bayes factor는 “경보가 drift에서 정상보다 일곱 배 더 그럴듯하다”는 뜻이지, drift 확률이 곧 $700\%$라는 뜻이 아니다.

### 3.10 다음 블록에 경보가 없으면 어떻게 될까?

방금 posterior가 $(N,D)=(0.5,0.5)$였다고 하자. 다음 블록의 drift 예측확률은

$$
\alpha_{t+1}^{-}(D)
=
0.5(0.05)+0.5(0.80)
=
0.425
$$

이다. 이제 경보가 없다는 $Y_{t+1}=0$을 관측하면

$$
P(Y_{t+1}=0\mid N)=0.9,
\qquad
P(Y_{t+1}=0\mid D)=0.3
$$

이므로

$$
\begin{aligned}
P(D\mid Y_{1:t+1})
&=
\frac{0.3(0.425)}{0.9(0.575)+0.3(0.425)}\\
&=
\frac{0.1275}{0.645}\\
&\approx
0.198
\end{aligned}
$$

이다. 정상 관측 하나가 drift belief를 낮췄지만 $0$으로 만들지는 않았다. 전이행렬의 $A_{DD}=0.8$이 drift가 어느 정도 지속될 수 있다고 말하기 때문이다. 이런 시간적 관성이 독립적인 시점별 분류와 HMM의 차이다.

### 3.11 predictive probability와 surprise

다음 관측의 예측분포는 상태 불확실성을 모두 평균낸다.

$$
p(y_{t+1}\mid Y_{1:t})
=
\sum_{i=1}^{K}
\alpha_t(i)
\sum_{j=1}^{K}
A_{ij}b_j(y_{t+1})
$$

실제로 관측된 $y_t$의 surprise는

$$
I_t
=
-\log p(y_t\mid Y_{1:t-1})
=
-\log c_t
$$

로 정의할 수 있다. 예측확률이 작을수록 surprise가 크다. 연속형 관측에서는 여기의 $p(y_t\mid Y_{1:t-1})$가 확률이 아니라 **확률밀도**이므로 단위·좌표변환에 따라 수치가 달라질 수 있다. 따라서 서로 다른 측정 단위의 surprise를 그대로 비교해서는 안 된다. 또한 $I_t$가 크다고 자동으로 유효한 $p$값이나 e-value가 되는 것은 아니다. 같은 스트림을 계속 보고 임계값을 넘을 때까지 기다리면 반복 감시의 오경보 문제가 생긴다. 어제 배운 e-process·Ville 부등식 같은 anytime-valid 설계는 별도로 필요하다.

### 3.12 긴 시퀀스에서 underflow를 막는다

forward joint probability를 정규화 없이 계속 곱하면 $0$에 가까운 수가 누적되어 컴퓨터에서 수치적으로 $0$이 될 수 있다. 두 가지 대표적 해결책이 있다.

첫째, 매 시점 $c_t$로 정규화한다. 이때 관측열의 로그가능도는

$$
\log p(y_{1:T})
=
\sum_{t=1}^{T}\log c_t
$$

로 복원할 수 있다. 둘째, 로그 공간에서 `log-sum-exp`를 쓴다.

$$
\log\sum_i e^{x_i}
=
m+
\log\sum_i e^{x_i-m},
\qquad
m=\max_i x_i
$$

최댓값 $m$을 빼면 큰 음수의 지수값이 모두 $0$으로 사라지는 문제를 줄인다.

### 3.13 filtering·smoothing·Viterbi는 질문이 다르다

세 계산을 혼동하면 미래 정보가 운영 평가에 새어 들어간다.

| 계산 | 수학적 대상 | 답하는 질문 | 사용 시점 |
| --- | --- | --- | --- |
| Filtering | $p(S_t\mid Y_{1:t})$ | 그때까지 본 정보로 지금 어떤 상태인가? | 온라인 감시 |
| Smoothing | $p(S_t\mid Y_{1:T})$ | 전체 사건이 끝난 뒤 과거 상태를 어떻게 재해석할까? | 사후 분석·라벨 개선 |
| Viterbi | $\arg\max_{s_{1:T}}p(s_{1:T}\mid y_{1:T})$ | 가장 가능성 높은 전체 상태 경로 하나는 무엇인가? | 구간 분할·설명 |

Viterbi 경로는 각 시점의 marginal posterior 최댓값을 이어 붙인 것과 일반적으로 같지 않다. 전자는 경로 전체의 전이 일관성을 최적화하고, 후자는 시점마다 따로 고른다.

또한 나중에 도착한 라벨로 smoothing한 결과를 가져와 “그날 실시간 탐지기가 이 상태를 알았다”고 말하면 미래 누출이다. 운영 성능은 그 시점의 filtering 정보만으로 재생해야 한다.

### 3.14 표준 HMM에는 geometric 체류시간 가정이 숨어 있다

시간에 따라 바뀌지 않는 전이행렬을 쓰고 $0\le A_{jj}<1$인 표준 HMM에서, 상태 $j$에 들어온 뒤 $d$개 블록 동안 머물고 다음 블록에서 떠날 확률은

$$
P(D_j=d)
=
A_{jj}^{d-1}(1-A_{jj}),
\qquad
d=1,2,\ldots
$$

이다. 기대 체류시간은

$$
\mathbb E[D_j]
=
\frac{1}{1-A_{jj}}
$$

이다. 이 geometric 분포는 다음 순간 떠날 조건부 확률이 지금까지 얼마나 오래 머물렀는지와 무관한 memoryless 가정을 갖는다. $A_{jj}=1$이면 $j$는 떠나지 않는 absorbing state라 유한한 geometric 체류시간 식을 적용할 수 없다. 전이행렬이 시간이나 개입 $U_t$에 따라 달라져도 체류시간은 일반적으로 위의 한 개 geometric 분포가 아니다.

예를 들어 점검 창이 항상 약 $6$개 블록 지속된다면 geometric 가정은 부적절할 수 있다. 이때는 duration 분포를 명시하는 Hidden Semi-Markov Model(HSMM)이나 경과 시간을 상태에 포함한 확장 상태를 고려한다.

### 3.15 지연 라벨과 누락 관측은 $0$이 아니다

즉시 보이는 입력 특징 $X_t$와 늦게 도착하는 사고 라벨 $R_t$를

$$
Y_t=(X_t,R_t)
$$

로 묶었다고 하자. $R_t$가 아직 도착하지 않았을 때 이를 $0$으로 기록하면 “사고 없음”과 “아직 모름”을 섞는다. 올바른 likelihood는 미관측 성분을 주변화한다.

$$
p(X_t\mid S_t=j)
=
\int
p(X_t,r\mid S_t=j)
\,dr
$$

이산 라벨이면 적분 대신 합을 쓴다. 다만 이 주변화는 “라벨이 늦거나 누락되는 방식”이 상태·결과를 조건으로 추가 정보를 주지 않거나 그 reveal 과정을 이미 모형에 포함했다는 가정 아래 타당하다. 고위험 사건만 더 빨리 확정되는 식의 informative missingness라면 reveal indicator와 reveal time의 분포도 함께 모형화해야 한다. 나중에 $R_t$가 reveal time $r>t$에 도착해도 사건은 event time $t$에 귀속해야 한다. 다음 두 기록을 모두 보존한다.

$$
(t_{\mathrm{event}},t_{\mathrm{reveal}})
$$

- 실시간 재생에서는 $t$ 당시 알 수 있었던 특징만 사용한다.
- 사후 분석에서는 고정 지연 smoothing이나 augmented state로 늦은 라벨을 반영한다.

### 3.16 filtering과 HMM 파라미터 학습은 다르다

지금까지의 filtering은 $A$, $b$, $\boldsymbol\pi$가 고정되었다고 가정했다. 이 모수들을 데이터에서 학습하는 대표적 방법이 Baum–Welch, 즉 HMM의 EM 알고리즘이다.

EM 반복 $m$에서 현재 모수로 전체 구간을 smoothing해

$$
\gamma_t^{(m)}(j)
=
P(S_t=j\mid Y_{1:T})
$$

와

$$
\xi_t^{(m)}(i,j)
=
P(S_{t-1}=i,S_t=j\mid Y_{1:T})
$$

를 얻으면 전이확률을

$$
A_{ij}^{(m+1)}
=
\frac{
\sum_{t=2}^{T}\xi_t^{(m)}(i,j)
}{
\sum_{t=2}^{T}\gamma_{t-1}^{(m)}(i)
}
$$

처럼 갱신할 수 있다. 분자는 $i\to j$ 전이의 기대 횟수, 분모는 상태 $i$에서 출발한 기대 횟수다. 초기분포는 $\pi_j^{(m+1)}=\gamma_1^{(m)}(j)$로 갱신하고, emission 모수의 M-step은 Bernoulli·Gaussian·신경망 등 선택한 분포족에 따라 달라진다. 정확한 E-step과 M-step이면 매 반복 관측 로그가능도는 감소하지 않지만, 전역 최댓값을 보장하지 않아 초기화에 따른 local optimum이 남는다.

하지만 이 계산은 미래 관측을 쓰는 offline 학습이다. 경보가 난 블록으로 상태 수 $K$와 emission을 고르고, 같은 블록에서 탐지 성능을 보고하면 선택과 평가가 섞인다. 학습·검량·최종 gate 블록을 분리해야 한다.

### 3.17 HMM은 숨은 진실 탐지기가 아니라 가정 묶음이다

HMM posterior가 틀릴 수 있는 대표적 이유는 다음과 같다.

- 상태 수 $K$가 너무 작아 새로운 체제를 기존 상태에 억지로 넣는다.
- 상태 수가 너무 커 우연한 잡음을 별도 상태로 만든다.
- emission 분포가 극단값이 자주 나오는 heavy tail, 관측 사이 상관, $0$이 유난히 많은 zero inflation을 설명하지 못한다.
- 전이확률이 계절·정책 버전과 함께 변하는데 고정 $A$를 쓴다.
- 여러 초기값에서 전역 최선이 아닌 서로 다른 주변 최선점인 local optimum에 수렴한다.
- 상태 label이 바뀌거나 서로 분리되지 않는다.

따라서 상태확률이 얼마나 퍼졌는지 재는 posterior entropy, 모형이 만든 가상 관측과 실제 관측을 비교하는 posterior predictive check, 학습에 쓰지 않은 held-out 구간의 log likelihood, 상태별 잔차, 난수 초기값인 여러 seed에서의 안정성, 실제 운영 라벨과의 외부 타당성을 함께 확인해야 한다.

### 3.18 순차 Bayes와 신경망 fine-tuning도 다르다

고정된 확률모형의 파라미터 $\theta$에 대해 새 데이터 블록 $\mathcal D_t$가 조건부 독립이면 정확한 Bayes 갱신은

$$
p(\theta\mid\mathcal D_{1:t})
\propto
p(\mathcal D_t\mid\theta)
p(\theta\mid\mathcal D_{1:t-1})
$$

이다. 이전 posterior 전체를 다음 prior로 넘긴다. 파라미터 자체가 시간에 따라 변한다고 가정한다면 바로 이 식을 쓰는 것이 아니라 먼저

$$
p(\theta_t\mid\mathcal D_{1:t-1})
=
\int
p(\theta_t\mid\theta_{t-1})
p(\theta_{t-1}\mid\mathcal D_{1:t-1})
\,d\theta_{t-1}
$$

로 parameter-state를 예측한 뒤 $p(\mathcal D_t\mid\theta_t)$로 갱신해야 한다. 반면 일반적인 SGD fine-tuning은 점추정 $\theta_{t-1}$에서 새 손실을 낮추는 최적화다.

$$
\theta_t
=
\theta_{t-1}
-
\eta\nabla_\theta
\mathcal L_{\mathrm{new}}(\theta_{t-1})
$$

EWC 같은 방법은 이전 posterior를 대각 Gaussian으로 근사하려 하지만 정확한 Bayes posterior 전달과 같지는 않다.

### 3.19 그래디언트 충돌이 망각을 만드는 원리

현재 파라미터에서 과거 손실과 새 손실의 그래디언트를

$$
g_{\mathrm{old}}
=
\nabla_\theta\mathcal L_{\mathrm{old}}(\theta),
\qquad
g_{\mathrm{new}}
=
\nabla_\theta\mathcal L_{\mathrm{new}}(\theta)
$$

라고 하자. 새 손실만 따라 한 번 갱신하면

$$
\theta'
=
\theta-\eta g_{\mathrm{new}}
$$

이다. 과거 손실을 1차 Taylor 근사하면

$$
\begin{aligned}
\mathcal L_{\mathrm{old}}(\theta')
&\approx
\mathcal L_{\mathrm{old}}(\theta)
+
g_{\mathrm{old}}^{\top}(\theta'-\theta)\\
&=
\mathcal L_{\mathrm{old}}(\theta)
-
\eta g_{\mathrm{old}}^{\top}g_{\mathrm{new}}
\end{aligned}
$$

이다. 따라서

$$
g_{\mathrm{old}}^{\top}g_{\mathrm{new}}<0
$$

이면 새 업데이트가 과거 손실을 증가시키는 방향이다. 더 일반적인 optimizer의 실제 변화량을 $\Delta\theta$라 하면 판단식은 $\Delta\mathcal L_{\mathrm{old}}\approx g_{\mathrm{old}}^\top\Delta\theta$다. 예를 들어 preconditioner $P$를 쓰는 $\Delta\theta=-\eta P g_{\mathrm{new}}$에서는 $g_{\mathrm{old}}^\top P g_{\mathrm{new}}$의 부호를 본다. 내적은 두 학습 목표가 현재 좌표의 파라미터 공간에서 협력하는지 충돌하는지 보여 주지만, parameterization·loss scale·minibatch noise에 불변인 절대 척도는 아니다. 또한 작은 step의 국소 1차 근사라 실제 깊은 신경망의 장기 변화 전체를 보장하지 않는다.

### 3.20 forgetting을 숫자로 정의한다

$a_{r,j}$를 “과제 또는 체제 $r$까지 학습한 뒤, 고정된 과제 $j$ 평가셋에서 얻은 점수”라고 하자. 높은 값이 좋은 지표라면 시점 $T$에서 과제 $j$의 forgetting은

$$
F_{T,j}
=
\max_{r\in\{j,\ldots,T-1\}}
a_{r,j}
-
a_{T,j}
$$

이다. 이전 최고점과 현재점의 차이다. 과거 과제 평균 forgetting은

$$
\overline F_T
=
\frac{1}{T-1}
\sum_{j=1}^{T-1}F_{T,j}
$$

로 쓸 수 있다. $F_{T,j}<0$이면 현재 성능이 과거 최고점보다 높아진 **backward transfer**를 뜻한다. 망각량만 음이 아닌 값으로 보고 싶다면 $F^+_{T,j}=\max(0,F_{T,j})$를 쓴다. 손실·오류율처럼 낮을수록 좋은 지표는 부호 방향을 바꿔야 한다. 여러 시점 점수 중 최댓값을 고르면 평가 잡음 때문에 기준점이 위로 치우칠 수 있으므로, seed·신뢰구간과 고정 평가 프로토콜도 함께 보고한다.

평균 하나만 보고 안전성을 주장하면 안 된다. 희귀한 의료·보안 체제의 $F_{T,j}$가 커도 대량의 일반 질의가 평균을 가릴 수 있으므로 체제·그룹별 회귀 한도를 둔다.

### 3.21 replay는 과거와 현재 손실을 함께 본다

새 데이터 $\mathcal D_t$와 과거 buffer $\mathcal B_{t-1}$를 섞은 가장 단순한 목적함수는

$$
\mathcal L_t(\theta)
=
(1-\beta)
\mathbb E_{z\sim\mathcal D_t}
[\ell(\theta;z)]
+
\beta
\mathbb E_{z\sim\mathcal B_{t-1}}
[\ell(\theta;z)]
$$

이다. $\beta$가 너무 작으면 과거 보호가 약하고, 너무 크면 새 체제 적응이 느리다.

길이를 미리 모르는 스트림에서 크기 $M$의 reservoir를 만들면 $n$개 항목을 본 뒤 각 항목의 포함확률은

$$
P(i\in\mathcal B_n)
=
\min\left(1,\frac{M}{n}\right)
$$

이다. 이 보장은 지금까지 실제로 관측한 유한 스트림의 각 항목에 같은 포함확률을 줄 뿐, 그 스트림 자체가 미래 운영 모집단을 대표한다고 보장하지 않는다. 그룹 quota, 법적 삭제, 중요도 가중 sampling을 섞으면 위의 단순 포함확률 식도 더는 그대로 성립하지 않는다. 또한 $0.1\%$인 치명 안전군이 buffer에 반드시 남는다는 뜻은 아니다. 실무에서는 다음을 분리할 수 있다.

- 전체 대표성을 위한 균등 reservoir
- 사전에 정의한 희귀 안전군별 최소 quota buffer
- 법적 삭제·동의·보존기간을 반영한 replay manifest

위험 점수가 높은 실패만 저장하면 발견에는 유리하지만 전체 과거 분포의 평가 표본은 아니다.

### 3.22 EWC는 중요한 파라미터 이동을 더 세게 벌준다

과거 과제를 학습한 기준점이 $\theta^\star$라고 하자. Elastic Weight Consolidation(EWC)은

$$
\mathcal L_{\mathrm{EWC}}(\theta)
=
\mathcal L_{\mathrm{new}}(\theta)
+
\frac{\lambda}{2}
\sum_i
\widehat F_i
(\theta_i-\theta_i^\star)^2
$$

를 최소화한다. $\widehat F_i$가 크면 파라미터 $i$가 과거 likelihood에 중요하다고 보고 더 강하게 묶는다.

이 식은 과거 posterior를 기준점 근처에서 Gaussian으로 근사한 Laplace 관점과 연결된다. $\theta^\star$가 과거 posterior의 MAP이고 로그 posterior가 그 주변에서 충분히 매끄러우며 국소 2차 근사가 타당하다고 하자. 그러면

$$
-\log p(\theta\mid\mathcal D_{\mathrm{old}})
\approx
\text{const}
+
\frac{1}{2}
(\theta-\theta^\star)^{\top}
\Lambda_{\mathrm{old}}
(\theta-\theta^\star)
$$

이다. $\Lambda_{\mathrm{old}}$는 음의 로그 posterior Hessian이다. 정규성 조건 아래 likelihood Hessian을 Fisher로 근사하면 대략

$$
\Lambda_{\mathrm{old}}
\approx
N_{\mathrm{old}}F
+
\Lambda_{\mathrm{prior}}
$$

로 볼 수 있다. 여기서 $N_{\mathrm{old}}$는 과거 표본수, $\Lambda_{\mathrm{prior}}$는 이전 prior precision이다. 구현이 loss를 합으로 정의하는지 평균으로 정의하는지에 따라 $N_{\mathrm{old}}$와 $\lambda$의 스케일이 달라진다. EWC는 이 행렬의 대각을 $\widehat F_i$로 근사하고 나머지 스케일을 $\lambda$에 흡수하는 경우가 많다.

여기서 데이터 생성분포에 대한 기대값을 쓰는 true Fisher, 관측 라벨을 그대로 넣은 empirical Fisher, 실제 음의 로그가능도의 observed Hessian은 일반적으로 같은 행렬이 아니다. 구현이 무엇을 $\widehat F$라고 부르는지 확인해야 한다. 그러므로 대각 Fisher는 prior precision과 파라미터 사이 상관을 단순화한 국소 근사다. 큰 체제 변화, 신경망의 파라미터 대칭, 여러 과제의 누적에서는 중요도 추정이 부정확할 수 있으며 망각 방지 보장이 아니다.

### 3.23 distillation과 parameter isolation

이전 모델 $p_{\mathrm{old}}$의 출력을 새 모델이 유지하게 하려면 온도 $\tau$를 둔 distillation 손실을 쓸 수 있다.

$$
\mathcal L_{\mathrm{distill}}
=
\mathbb E_{x\sim\mathcal B}
\left[
\operatorname{KL}
\left(
p_{\mathrm{old}}^{(\tau)}(\cdot\mid x)
\,\Vert\,
p_{\theta}^{(\tau)}(\cdot\mid x)
\right)
\right]
$$

이는 출력 행동을 보존하지만 이전 모델의 오류·편향도 함께 보존할 수 있다. Adapter, LoRA, expert routing처럼 새 파라미터 공간을 분리하면 간섭을 줄일 수 있지만 routing 오류, 메모리 증가, 여러 adapter의 충돌과 유지비가 생긴다.

여기서 $\operatorname{KL}(p\Vert q)$는 기준분포 $p$가 중요하게 보는 출력에서 새 분포 $q$가 얼마나 달라졌는지 재는 비대칭 차이다. LoRA는 큰 가중치 전체 대신 작은 저랭크 행렬 두 개를 학습하는 adapter 방식이다.

### 3.24 HMM belief로 적응 강도를 조절할 때의 함정

drift posterior를

$$
q_t
=
P(S_t=D\mid Y_{1:t})
$$

라고 하고 새 데이터 비중을

$$
\lambda_t
=
\lambda_{\min}
+
(\lambda_{\max}-\lambda_{\min})q_t
$$

처럼 정할 수는 있다. 하지만 이는 설계 예시일 뿐 안전 보장이 아니다.

- HMM이 틀리면 학습 비중도 잘못된다.
- 모델 출력이 다시 HMM 관측에 들어가면 자기강화 피드백이 생길 수 있다.
- 작은 posterior의 치명 상태가 평균 목적함수에서 사라질 수 있다.
- 같은 경보 블록으로 $q_t$, 학습 비중, 성능 향상을 모두 고르면 과적합된다.

따라서 posterior 평균 안전비용만 보지 말고, posterior 질량을 적어도 $1-\delta_{\mathrm{state}}$ 포함하는 상태 집합 $\mathcal K_t^{\mathrm{mass}}$를 사전에 정한 정렬·동률 규칙으로 만든다.

$$
\sum_{k\in\mathcal K_t^{\mathrm{mass}}}\alpha_t(k)
\ge
1-\delta_{\mathrm{state}},
\qquad
\mathcal K_t
=
\mathcal K_t^{\mathrm{mass}}
\cup
\mathcal K_{\mathrm{catastrophic}}
$$

그 뒤 각 상태에서 gate를 확인할 수 있다. 상태별 고정 threshold만 쓰면 threshold 바로 아래의 여러 작은 상태가 큰 총질량을 이루어도 모두 빠질 수 있으므로 누적 질량 조건을 함께 둔다. 어떤 알려진 상태에도 맞지 않는 OOD나 posterior predictive failure는 이 집합 계산으로 흡수하지 말고 별도 hard-stop으로 처리한다.

$$
\operatorname{LCB}_k
\left(
V_k(\pi_{\mathrm{new}})
-
V_k(\pi_{\mathrm{champ}})
\right)
\ge 0,
\qquad
\forall k\in\mathcal K_t
$$

$$
\operatorname{UCB}_k
\left(C_k(\pi_{\mathrm{new}})\right)
\le c_k,
\qquad
\forall k\in\mathcal K_t
$$

$V_k$는 상태 $k$의 가치, $C_k$는 안전비용, $c_k$는 한도다. 치명 상태는 posterior가 작아도 강제로 집합에 넣을 수 있다.

여기서 $\operatorname{LCB}$는 추정오차를 뺀 **lower confidence bound(신뢰 하한)**, $\operatorname{UCB}$는 추정오차를 더한 **upper confidence bound(신뢰 상한)**다. 이 부등식이 유효하려면 상태 $k$가 latent라는 사실과 $A,b$의 추정오차를 경계에 전파하거나, 허용한 HMM 모수·상태 할당 전체에 대해 균일하게 성립하도록 만들어야 한다. 로그 정책과 평가 정책이 다르면 행동 overlap과 올바른 propensity 기록도 필요하다. 여러 상태·후보 정책·시점에서 반복해 gate를 보면 동시·순차 추론의 오류율을 통제해야 하며, 같은 데이터로 후보를 고르고 경계를 보고하면 안 된다.

### 3.25 재학습 뒤에는 왜 재검량해야 할까?

가중치가 $\theta_{\mathrm{old}}$에서 $\theta_{\mathrm{new}}$로 바뀌면 같은 입력 $x$에서도 점수분포가 바뀐다.

$$
s_{\theta_{\mathrm{old}}}(x)
\neq
s_{\theta_{\mathrm{new}}}(x)
$$

그러므로 이전 확률 calibration, conformal score 분위수, 안전 threshold, HMM emission이 그대로 유효하다고 가정할 수 없다. 최소한 데이터를 다음처럼 분리한다.

$$
\mathcal D_{\mathrm{adapt}}
\cap
\mathcal D_{\mathrm{cal}}
=
\mathcal D_{\mathrm{adapt}}
\cap
\mathcal D_{\mathrm{gate}}
=
\mathcal D_{\mathrm{cal}}
\cap
\mathcal D_{\mathrm{gate}}
=
\varnothing
$$

- $\mathcal D_{\mathrm{adapt}}$: replay·regularization을 포함한 학습
- $\mathcal D_{\mathrm{cal}}$: 새 모델의 확률·threshold·conformal score, 필요 시 HMM emission 검량
- $\mathcal D_{\mathrm{gate}}$: old/new regime 회귀와 최종 안전 승인

이 식은 세 표본의 **관측 단위가 겹치지 않는다**는 뜻이다. 단순 비중첩만으로 통계적 독립이 생기지는 않는다. 같은 사용자의 인접 대화, 동일 원본의 번역·증강본, 가까운 시간 블록은 서로 의존할 수 있으므로 source·사용자·시간 cluster 단위로 분할하고 남은 의존성이 추론에 어떤 영향을 주는지 확인해야 한다.

### 3.26 어제의 Online DRO와 상태별 안전을 연결한다

상태 $k$마다 별도 기준분포와 Wasserstein 모호성 집합을 둘 수 있다.

$$
\mathcal P_k
=
\left\{
Q:
W_{d_k}
\left(Q,\widehat P_{\mathrm{ref},k}\right)
\le\rho_k
\right\}
$$

현재 belief로 이 집합들을 단순 평균하면 작은 확률의 치명 상태가 희석될 수 있다. 따라서 다음 두 질문을 나눈다.

1. **예측 질문:** posterior 평균에서 가장 가능성 높은 미래 성능은 무엇인가?
2. **안전 질문:** 현재 배제할 수 없는 각 고위험 상태에서 비용 상한을 만족하는가?

HMM은 어떤 상태가 plausible한지 정리하고, DRO와 신뢰경계는 그 상태 안의 분포 불확실성과 추정오차를 다룬다. 둘은 대체재가 아니라 서로 다른 불확실성 층이다.

### 3.27 오늘 수학과 AI 부품의 1:1 대응

| 수학 부품 | AI·운영 부품 | 잘못된 연결 |
| --- | --- | --- |
| 상태 $S_t$ | 정상·피크·새 도메인·사고 체제 | 상태를 관측된 확정 라벨로 취급 |
| 전이 $A_{ij}$ | 체제의 지속·복귀 가능성 | 모델 배포 변경을 자연 전이로 숨김 |
| emission $b_j(y)$ | 입력 거리·오류·latency의 상태별 분포 | 상관된 지표를 독립이라 가정 |
| filtering $\alpha_t$ | 실시간 WATCH/HOLD 근거 | smoothing 결과를 실시간 성능으로 보고 |
| predictive surprise | 예상 밖 블록 점수 | 반복 감시 보정 없는 $p$값으로 해석 |
| 상태별 posterior | 적응 후보 체제의 불확실성 | posterior 임계값을 자동 학습 허가로 사용 |
| 그래디언트 내적 | old/new 학습 충돌 진단 | 양의 내적 하나로 장기 보존 보장 |
| replay 혼합손실 | 과거 예시 재학습 | buffer를 과거 전체의 완전한 대표로 간주 |
| EWC quadratic penalty | 중요 파라미터 anchoring | 정확한 Bayes·무망각 보장으로 해석 |
| 상태별 LCB·UCB | 새 모델의 regime별 안전 gate | posterior 평균 하나로 희귀 위험을 숨김 |
| 독립 calibration | 새 점수·threshold 재설정 | adaptation 데이터로 바로 threshold 선택 |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 지속학습은 무엇이 계속 바뀌는지부터 구분한다

대표적인 continual learning 설정은 다음과 같다.

| 설정 | 바뀌는 것 | 예시 | 주요 위험 |
| --- | --- | --- | --- |
| Task-incremental | 과제와 과제 ID | 번역 뒤 요약을 학습하며 현재 과제 ID를 앎 | 과제별 head·router 의존 |
| Domain-incremental | 입력분포, 정답 공간은 유지 | 새 국가·카메라·사용자군 유입 | 같은 label 의미 아래 covariate shift |
| Class-incremental | 새 class가 추가되고 과제 ID 없음 | 새 결함 종류·새 의도 추가 | 이전 class와 새 class 경쟁 |
| Concept drift | $P(Y\mid X)$ 자체 변화 | 정책·사용자 행동·공격 전략 변화 | 과거 replay가 현재와 충돌 |

실제 서비스는 이 설정들이 섞인다. “지속학습을 지원한다”는 말만으로는 부족하며 어떤 변화, 어떤 정보, 어떤 평가 프로토콜을 가정하는지 명시해야 한다.

### 4.2 세 개의 루프를 분리한다

안전한 구조는 감시·학습·승인을 한 자동 루프로 합치지 않는다.

```text
[불변 원시 로그 + event/reveal time]
                |
                v
      (1) 상태 추론·감시 루프
       HMM filtering + hard-stop
                |
          WATCH / HOLD 신호
                |
                v
      (2) 제한된 적응 학습 루프
   new data + replay + regularization
                |
         frozen challenger 생성
                |
                v
       (3) 독립 승인·배포 루프
 old/new 회귀시험 + 재검량 + final gate
                |
      shadow -> canary -> promote
                |
          실패 시 safe rollback
```

한 루프가 다른 루프의 증거를 덮어쓰지 않도록 데이터 snapshot, 코드 commit, seed, buffer manifest, calibration artifact를 버전으로 고정한다.

### 4.3 HMM 상태와 운영 상태기계는 같은 것이 아니다

HMM 상태는 관측을 설명하는 확률모형의 latent state다. 운영 상태는 사람이 정한 행동 규칙이다.

$$
\text{HMM belief}
\longrightarrow
\text{NORMAL / WATCH / HOLD / SHADOW / CANARY}
$$

예를 들어 사전 등록한 임계값을

$$
q_{\mathrm{enter}}=0.8,
\qquad
q_{\mathrm{exit}}=0.3
$$

으로 두고 $q_t=P(S_t=D\mid Y_{1:t})$라 하자.

- $q_t\ge q_{\mathrm{enter}}$가 연속 $m$블록이면 `HOLD`
- `HOLD` 뒤 $q_t\le q_{\mathrm{exit}}$가 연속 $r$블록이고 성숙 라벨 gate도 통과하면 해제 검토
- 치명 사건 hard-stop은 $q_t$와 무관하게 즉시 작동

진입·해제 임계값을 다르게 두는 hysteresis는 경계 부근에서 상태가 매 블록 뒤집히는 것을 줄인다. 숫자는 예시이며 실제 임계값은 독립 정상·사고 구간에서 검량해야 한다.

### 4.4 적응 학습의 최소 레시피

새 체제에 적응할 때 다음 순서가 안전하다.

1. **원인 분리:** 데이터 파이프라인 오류, 배포 개입, 일시 피크, 지속 drift, 공격을 구분한다.
2. **학습 허용 데이터 확정:** 오염·중복·권리·지연 라벨·source 계보를 검사한다.
3. **replay 구성:** 일반 reservoir와 고위험 그룹 floor를 함께 사용하고 snapshot을 고정한다.
4. **보존 제약 설정:** EWC·distillation·adapter 중 문제에 맞는 장치를 선택한다.
5. **challenger 학습:** champion을 덮어쓰지 않고 별도 artifact로 만든다.
6. **fresh calibration:** 가중치가 바뀐 challenger에 맞춰 확률·threshold·conformal score를 다시 검량한다.
7. **회귀 행렬 평가:** 과거 체제, 새 체제, 안전군, OOD를 각각 본다.
8. **shadow·canary:** 실제 입력에서 행동은 제한한 채 관찰하고 작은 무작위 노출로 확장한다.
9. **완전한 rollback:** 실패하면 검증된 버전 묶음 전체를 복원한다.

### 4.5 replay buffer는 작은 데이터베이스가 아니라 통계 설계다

buffer 항목마다 최소한 다음 provenance를 남긴다.

```text
sample_id
source_id / user_block_id
event_time / reveal_time
regime_label_source
group_membership_version
data_rights / deletion_deadline
model_version_that_generated_the_output
selection_probability_or_sampling_rule
dedup_cluster_id
```

주의할 점은 다음과 같다.

- 같은 대화를 문장 단위로 train과 test에 나누지 않는다.
- 동일 원본의 번역·증강·crop은 source block으로 묶는다.
- failure mining buffer와 대표 평가 buffer를 분리한다.
- 삭제 요청이 오면 파생 embedding·cache·checkpoint에 미친 범위를 추적한다.
- 오래된 안전 사례가 현재 정책상 잘못된 정답이 됐는지 검토한다.

### 4.6 회귀 행렬은 새 점수 하나보다 중요하다

학습 단계 $r$과 평가 체제 $j$의 점수를 행렬로 보관한다.

$$
M_{rj}=a_{r,j}
$$

예를 들어 다음과 같은 표가 필요하다.

| 학습 버전 | 정상 | 주말 피크 | 한국어 의료 | 새 도메인 | 안전 거부 | calibration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| champion | $0.91$ | $0.87$ | $0.82$ | $0.54$ | $0.97$ | $0.03$ |
| challenger | $0.90$ | $0.88$ | $0.80$ | $0.86$ | $0.95$ | $0.07$ |

challenger는 새 도메인 점수가 크게 올랐지만 calibration error가 악화됐고 한국어 의료·안전 거부가 떨어졌다. 단순 평균이 좋아도 미리 정한 회귀 한도를 넘으면 승격하지 않는다.

### 4.7 재검량은 달력과 사건 두 축으로 운영한다

매주 또는 매월이라는 달력 주기만으로는 갑작스러운 shift에 늦다. 반대로 경보 때마다 모든 것을 다시 맞추면 과적합과 운영 진동이 생긴다.

**이벤트 기반 재검량 조건**은 다음을 포함한다.

- 모델·프롬프트·retriever·가중치 수치 정밀도를 줄이는 quantization·runtime 변경
- HMM emission의 posterior predictive failure
- 새 도메인·언어·도구 추가
- conformal coverage 또는 calibration gate 실패
- 장기간 `WATCH/HOLD` 유지

**일정 기반 조건**은 다음을 포함한다.

- 최대 허용 staleness 도달
- 성숙 라벨 최소 표본수 충족
- 가중 표본의 유효표본크기 충족

가중치 $w_i$를 쓴다면

$$
N_{\mathrm{eff}}
=
\frac{\left(\sum_iw_i\right)^2}{\sum_iw_i^2}
$$

도 확인한다. 행 수가 많아도 몇 개 항목에 가중치가 몰리면 실제 정보량은 작다.

### 4.8 재검량 대상은 하나가 아니다

가중치 갱신 뒤 다음 항목의 유효성이 서로 다르게 깨질 수 있다.

1. HMM의 전이·emission·상태 수·임계값
2. 예측 확률의 reliability calibration
3. conformal score와 quantile
4. Online DRO의 reference·ground metric·radius
5. 그룹 정의와 safety threshold
6. replay selection rule과 중요도 가중치

한 경보 블록으로 이 여섯 항목을 모두 바꾸고 같은 블록에서 안전성을 주장하면 “문제를 보고 자를 바꾼 뒤 그 자로 문제없다고 재는” 셈이다.

### 4.9 rollback은 가중치 파일 하나가 아니다

승인 버전은 다음과 같은 묶음이어야 한다.

$$
v
=
\left(
\theta,
\text{prompt},
\text{retriever},
\text{runtime},
A,
b,
\text{state thresholds},
\text{replay manifest},
\text{calibrator},
d,
\rho,
\text{guardrail}
\right)
$$

새 가중치에 옛 calibrator만 붙이거나, 옛 모델에 새 retriever를 붙이면 이전에 검증하지 않은 조합이 된다. rollback 후보도 현재 support에서 안전하다는 증거가 있어야 하며, 없으면 기능 축소·사람 검토 같은 fail-safe가 우선이다.

### 4.10 모델 메모리와 지속학습을 구분한다

“메모리가 있다”는 말은 여러 구조를 가리킨다.

| 메모리 종류 | 무엇이 바뀌나? | 지속 범위 | 파라미터 학습인가? |
| --- | --- | --- | --- |
| 긴 context·KV cache | 현재 요청의 token과 attention의 key/value 중간값 | 요청 또는 session | 아니오 |
| RAG·외부 artifact store | 외부 문서·이미지·표를 검색해 생성에 넣는 저장소 | store 정책에 따라 다름 | 아니오 |
| recurrent latent memory | 고정 크기 latent state | session·trajectory 사이 가능 | 보통 online weight update는 아님 |
| adapter·LoRA 업데이트 | 별도 또는 일부 파라미터 | 새 checkpoint | 예 |
| full fine-tuning | 모델 파라미터 전반 | 새 checkpoint | 예 |

외부 기억이 좋아졌다고 catastrophic forgetting이 해결된 것은 아니며, 가중치를 바꾸지 않았다고 memory poisoning·privacy·stale fact 위험이 없는 것도 아니다.

### 4.11 초보자가 흔히 하는 오해와 교정

| 오해 | 왜 틀렸나? | 교정 |
| --- | --- | --- |
| posterior $0.8$이면 상태가 확정됐다. | 모형·prior·emission 가정 아래의 확률일 뿐이다. | 불확실성과 모형 진단을 함께 보고 hard-stop을 별도 둔다. |
| HMM이 drift라 했으니 즉시 fine-tuning한다. | 탐지와 학습 승인 사이에 원인·데이터 권리·회귀 위험 검토가 빠졌다. | WATCH/HOLD 뒤 독립 적응·승인 루프를 거친다. |
| filtering과 smoothing은 같은 상태 추론이다. | smoothing은 미래 정보를 쓴다. | 온라인 평가는 filtering만 재생한다. |
| 표준 HMM은 어떤 지속시간도 잘 표현한다. | 체류시간이 geometric으로 제한된다. | duration이 중요하면 HSMM을 검토한다. |
| 경보 라벨이 없으면 $0$으로 넣는다. | 미관측과 음성(negative) 라벨을 혼동한다. | missing likelihood로 주변화하고 event/reveal time을 저장한다. |
| replay를 쓰면 망각이 사라진다. | buffer coverage와 학습 충돌이 남는다. | 그룹별 회귀 행렬과 독립 test로 측정한다. |
| EWC는 정확한 Bayesian update다. | 대각 Fisher의 국소 Laplace 근사다. | 근사 범위와 큰 shift에서의 실패를 검증한다. |
| distillation은 과거 정답을 보존한다. | 과거 모델의 출력과 오류를 보존한다. | gold label·독립 감사와 함께 쓴다. |
| 메모리 모델은 자동 continual learner다. | latent state 저장과 영구 weight adaptation은 다르다. | 어떤 상태·가중치가 언제 바뀌는지 명시한다. |
| 평균 점수가 올랐으면 안전하다. | 희귀 고위험 체제의 큰 회귀가 숨을 수 있다. | 상태·그룹별 floor와 비용 UCB를 둔다. |

### 4.12 최소 구현 체크리스트

- [ ] HMM의 $K$, $A$, emission family, 초기화, seed를 버전으로 남겼는가?
- [ ] 배포 개입 $U_t$와 event/reveal time을 로그에 기록했는가?
- [ ] filtering 성능과 smoothing 사후 분석을 분리했는가?
- [ ] forward 계산에 scaling 또는 log-sum-exp를 썼는가?
- [ ] 표준 HMM의 geometric duration 가정을 점검했는가?
- [ ] HMM posterior와 운영 상태기계·hard-stop을 분리했는가?
- [ ] adaptation, calibration, final gate 데이터가 source block 단위로 분리됐는가?
- [ ] replay buffer에 일반 대표 표본과 고위험 그룹 floor가 모두 있는가?
- [ ] old/new regime와 안전 그룹의 회귀 행렬을 보존했는가?
- [ ] 가중치 변경 뒤 calibration·conformal·HMM emission을 재검토했는가?
- [ ] shadow·canary의 중지 규칙과 성숙 라벨 대기 규칙이 사전 등록됐는가?
- [ ] rollback manifest가 모델 밖의 retriever·threshold·memory state까지 포함하는가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **확인 시점:** 2026년 10월 21일 KST. 아래 두 연구의 arXiv v1은 2026년 10월 19일 UTC에 제출되어 10월 20일 KST 공개 배치에 포함됐다. 수치는 저자 보고이며 독립 재현 결과가 아니다.

### 5.1 MemT-1.0: context를 다시 읽지 않는 모델 내부의 상태형 메모리

MemTensor 팀은 foundation model 계산 안에 지속적으로 변하는 latent memory state를 넣은 **MemT-1.0**을 공개했다. 일반적인 긴 context는 과거 token을 다시 attention 대상으로 남기지만, MemT-1.0은 일부 Transformer block 옆에 native memory block을 두고 이전 상호작용을 고정 크기 상태에 누적한다.

핵심 구조는 두 부분이다.

1. **Dynamic local memory:** trajectory마다 달라지는 다채널 행렬 상태다. interaction이 진행되며 바뀐다.
2. **Trainable hyper memory:** query·key·value projection, erase·write·forget gate처럼 “무엇을 어떻게 기억할지”를 학습한 고정 파라미터다.

각 turn 안에서는 Gated DeltaNet-2 기반 token-level update로 다채널 local memory를 갱신한다. 이 다채널 구조와 Mixture-of-Storage(MoS)는 같은 것이 아니다. MoS는 공유 address와 $E$개의 memory expert, storage·utilization용 dense router를 별도로 두어 용량을 확장하는 제안이다. 논문의 주 공개·평가 설정인 Nano와 Flash는 구성표에서 모두 $E=1$이므로, headline 결과를 multi-expert MoS의 직접 효과로 읽으면 안 된다. 논문은 backbone 준비 뒤 continued pre-training과 post-training의 두 단계를 사용했으며 총 $34$B token의 학습 corpus를 구성했다고 보고한다.

공개 모델 가족은 다음과 같다.

| 모델 | Backbone | 총 파라미터 | 활성 파라미터 | trainable memory 파라미터 | dynamic memory state |
| --- | --- | ---: | ---: | ---: | ---: |
| MemT-1.0-Nano | Qwen3.6-35B-A3B | $36$B | $3$B | $451$M | $10$M |
| MemT-1.0-Flash | DeepSeek-V4-Flash | $288$B | $17$B | $3.61$B | $336$M |

dynamic memory state는 모델 파라미터 수가 아니라 **trajectory마다 따로 저장되는 상태 원소 수**다. 정확히는 Nano $10{,}485{,}760$개, Flash $335{,}544{,}320$개이며, 원소당 $4$ byte인 FP32로만 단순 환산하면 각각 $40$ MiB와 $1{,}280$ MiB다.

저자 보고 결과에서 두 모델은 평가한 $11$개 memory benchmark의 parametric-memory 방법 중 평균 기준 1·2위를 차지했다. 그러나 원문과 모델 카드는 긴 history에서 상세 근거를 찾아야 하는 경우 강한 **textual-memory system이 여전히 앞선다**고 명시한다. 즉, 고정 크기 latent state의 압축은 공짜가 아니다.

MemT-1.0-Nano는 $128$K token history 조건에서 full-context KV cache 대비 약 $1.86\times$ end-to-end inference speed와 $66\times$ persistent-storage 절감을 보고했다. 이는 BEAM history $35$개, 전용 GPU $2$장, concurrency·batch size $1$, $512$-token query와 강제 $128$-token generation을 사용한 저자 측 Hugging Face inference 비교다. 속도는 history preparation을 포함하지만 model loading·tokenization·scoring은 제외한다. $66\times$는 trajectory별 논리적 persistent state 비교로, 공유 model weight·embedder·hyper-memory parameter와 현재 step의 KV는 포함하지 않는다. 따라서 모든 workload의 보편적 속도비나 전체 시스템 저장비로 읽으면 안 된다.

공개 범위도 artifact별로 나눠 읽어야 한다.

- Nano weight의 모델 카드는 Apache-2.0을 표시한다.
- Flash weight의 모델 카드는 MIT를 표시한다.
- 공식 GitHub 저장소의 원저작 software는 PolyForm Noncommercial 1.0.0이며 상업 이용을 허용하지 않는다.
- 논문 material은 CC BY-NC-SA 4.0이다.
- 공식 저장소의 구현은 현재 inference·serving 중심이다. 논문에서 말한 $34$B token corpus 전체와 end-to-end training pipeline artifact가 같은 범위로 공개됐다고 보아서는 안 된다.

가장 중요한 한계는 오늘 주제와 정확히 맞닿는다.

- 고정 크기 상태는 history 길이에 대해 $O(1)$ 저장공간을 갖지만, 오래된 세부정보와 새 정보가 같은 용량을 다투므로 **상태 수준의 안정성–가소성 딜레마**가 생긴다.
- agentic coding 실험의 procedural experience는 외부에서 구성·검색해 넣는다. 논문도 autonomous continual learning은 미래 과제라고 명시한다.
- latent memory는 사람이 원문처럼 감사하기 어렵다. 잘못된 사실·prompt injection·민감정보가 쓰였을 때 어떤 부분을 지우고 어떻게 복구할지 별도 통제가 필요하다.
- memory state가 변해도 backbone weight가 매 turn 학습되는 것은 아니다. 이를 full continual fine-tuning과 동일시하면 안 된다.

**엔지니어 인사이트 (Impact):** MemT-1.0이 보여 주는 방향은 긴 history 비용을 “더 큰 context”만으로 해결하지 않고 **고정 크기 recurrent state**로 옮기는 것이다. 오늘의 HMM과 닮은 점은 과거를 현재 상태에 압축해 다음 계산에 넘긴다는 것이다. 다른 점은 MemT state가 확률적으로 해석·검량된 regime posterior가 아니라 학습된 고차원 tensor라는 점이다. 따라서 `memory snapshot`, update code, source provenance, consent, deletion, poisoning test, rollback을 model checkpoint와 함께 버전으로 묶어야 한다.

- 논문: [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825)
- 공식 코드·라이선스: [MemTensor/MemT-1.0](https://github.com/MemTensor/MemT-1.0)
- Nano 모델 카드: [MemTensor/MemT-1.0-Nano](https://huggingface.co/MemTensor/MemT-1.0-Nano)
- Flash 모델 카드: [MemTensor/MemT-1.0-Flash](https://huggingface.co/MemTensor/MemT-1.0-Flash)

### 5.2 OpenVLHarness: VLM의 도구 결과를 질의 안에서 계속 참조하는 외부 멀티모달 메모리

UCLA·Apple 연구진의 **OpenVLHarness**는 VLM 가중치를 fine-tuning하지 않고, 여러 시각·검색 도구와 persistent multimodal memory를 연결하는 training-free harness다. 여기서 “persistent”는 무기한 사용자 기억이 아니라 **한 질문을 해결하는 여러 reasoning step 동안** 이미지·구조화 데이터·웹 근거를 주소 가능한 artifact로 유지한다는 뜻이다. 공개 README는 새 질문마다 fresh memory로 시작한다고 명시한다.

capability layer가 제공하는 아홉 기능은 다음과 같다.

1. visual grounding
2. OCR
3. depth estimation
4. camera trajectory
5. zoom-in
6. text search
7. image search
8. webpage visit
9. Python coding agent

도구의 raw output을 모델이 읽을 수 있는 annotated image, 요약, 수치 데이터로 바꾸고, Python agent가 저장된 box·mask·depth map을 함께 계산한다. 이는 VLM이 내부 추론만 더 오래 하는 대신 외부 근거를 획득·검사·합성하게 만든다.

논문은 $22$개 benchmark와 open·proprietary backbone들을 평가했다. 아래 평균은 accuracy·AP·MRA처럼 서로 다른 지표를 쓰는 $22$개 benchmark 점수의 **비가중 산술평균**이므로 하나의 공통 단위로 해석하면 안 된다. 공개 README의 일곱 backbone 표에서 no-tool 대비 이 평균의 향상은 약 $+9.66$에서 $+12.62$ point였다. 예를 들면 다음과 같다.

| Backbone | Base | OpenVLHarness | 차이 |
| --- | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | $34.24$ | $45.41$ | $+11.17$ |
| Qwen3-VL-32B-Instruct | $38.30$ | $50.92$ | $+12.62$ |
| Kimi K3 (max effort) | $56.43$ | $66.09$ | $+9.66$ |
| GPT-6 Astra (medium effort) | $60.27$ | $70.17$ | $+9.90$ |

Qwen3-VL 결과는 세 번 실행한 평균이고 proprietary GPT 계열 결과는 single run이다. 공개 자료는 Kimi K3의 반복 횟수를 같은 방식으로 명시하지 않으므로 single run이라고 단정하지 않는다. backbone마다 reasoning effort도 같지 않은 항목이 있으므로 이 표를 모델 순위표로 읽기보다 **동일 backbone에 harness를 붙였을 때의 저자 보고 system gain**으로 읽어야 한다.

재현 비용과 공개 범위에는 중요한 경계가 있다.

- OpenVLHarness 본체 코드는 MIT이고 논문은 CC BY 4.0이다.
- evaluation kit의 파생 부분은 Apache-2.0이다. 기본 Depth Anything 3 checkpoint는 CC BY-NC 4.0이라 비상업 조건이 있고, SAM 3 weight는 gated access와 별도 SAM License를 따른다. PaddleOCR·Qwen 계열은 표시된 Apache 계열 조건을 포함해 각 모델·서비스·데이터의 원문 약관을 따로 확인해야 한다.
- 전체 아홉 기능을 쓰려면 SAM 3, PaddleOCR-VL, Depth Anything 3, Qwen3-VL endpoint, 검색 API 등 여러 서비스를 준비해야 하므로 core가 MIT라고 전체 stack의 상업 이용이 자동 허용되는 것은 아니다.
- 검색은 Serper 같은 외부 API와 이미지 업로드 경로를 요구할 수 있어 비용·privacy·availability가 end-to-end 품질에 들어온다.
- 공식 보안 안내에 따르면 기본 Python executor는 생성 코드를 현재 사용자 권한으로 실행하며 filesystem과 network에 접근할 수 있는 **비-sandbox** 방식이다. timeout은 격리 장치가 아니고 분리된 child process가 남을 수 있으므로, 신뢰하지 않은 입력에는 container·VM 격리, network·credential 차단, resource limit가 필요하다.

이 연구는 continual learning과도 구분해야 한다. memory artifact는 질의 안에서 유지되지만 새 질문에는 초기화되며, backbone weight는 바뀌지 않는다. 따라서 benchmark 향상은 “모델이 경험을 영구 학습했다”는 결과가 아니라 **도구와 외부 작업기억을 포함한 시스템이 더 잘 풀었다**는 결과다.

**엔지니어 인사이트 (Impact):** MemT-1.0이 과거를 latent state로 압축한다면 OpenVLHarness는 근거 artifact를 외부에 남겨 다시 참조한다. 후자는 provenance와 재현성이 상대적으로 좋지만 storage·검색·도구 실패가 늘고, 전자는 고정 크기와 속도 이점이 있지만 정확한 내용 감사·삭제가 어렵다. 실제 시스템은 두 방식을 혼합할 가능성이 크다. 이때 HMM emission으로 도구 결과 여러 개를 넣으면 같은 원본 이미지와 같은 보조 모델에서 나온 값들이 상관되어 있으므로 독립 emission으로 곱하지 말아야 한다.

- 논문: [OpenVLHarness: Real World Visual Agentic Tool-Use with Multimodal Memory](https://arxiv.org/abs/2610.26860)
- 공식 코드: [uclanlp/OpenVLHarness](https://github.com/uclanlp/OpenVLHarness)
- 프로젝트·실험표: [OpenVLHarness project page](https://openvlharness.github.io/)

### 5.3 두 트렌드는 “메모리”의 서로 다른 층을 보여 준다

| 비교 축 | MemT-1.0 | OpenVLHarness | 오늘의 교훈 |
| --- | --- | --- | --- |
| 기억 위치 | 모델 내부 latent state | 외부 이미지·표·웹 artifact | state와 evidence store를 구분한다. |
| 지속 범위 | 저장·복원 가능한 trajectory state | 한 query의 여러 reasoning step | “persistent”의 시간 범위를 명시한다. |
| online weight update | local memory state는 바뀌지만 backbone weight 학습과 다름 | 없음 | memory update와 fine-tuning을 구분한다. |
| 감사 가능성 | 압축된 tensor라 직접 해석이 어려움 | 근거 artifact를 주소로 추적 가능 | provenance 수준이 다르다. |
| 용량 trade-off | 고정 크기라 오래된 세부정보 손실 가능 | artifact 수와 비용이 커질 수 있음 | 안정성–가소성과 storage budget을 함께 본다. |
| 공격면 | memory poisoning·민감정보 잔존 | 검색 오염·도구 오류·code execution | memory도 입력 검증과 rollback이 필요하다. |
| 공개 경계 | weight별 라이선스와 NC repo code가 다름 | MIT core와 third-party terms가 다름 | “오픈”을 artifact별로 확인한다. |

두 사례를 한 문장으로 묶으면 다음과 같다.

> **차세대 AI의 경쟁은 더 긴 context뿐 아니라 무엇을 상태로 압축하고 무엇을 근거로 외부에 남길지로 이동하고 있지만, 어느 방식도 안전한 지속학습·검량·삭제·rollback을 자동 해결하지 않는다.**

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

직전 HMM posterior가 $(N,D)=(0.9,0.1)$이고 다음과 같다고 하자.

$$
A
=
\begin{bmatrix}
0.95 & 0.05\\
0.20 & 0.80
\end{bmatrix},
\qquad
P(Y=1\mid N)=0.1,
\qquad
P(Y=1\mid D)=0.7
$$

새 경보 $Y_t=1$이 도착했다.

1. 관측 전 drift 예측확률과 관측 후 drift posterior는 얼마인가?
2. 이 posterior만 보고 오늘 데이터 $200$건에 즉시 fine-tuning해도 되는가?
3. 새 데이터만 학습한 뒤 old-regime 점수가 과거 최고 $0.92$에서 $0.77$로 떨어지고 new-regime 점수가 $0.89$라면 old-regime forgetting은 얼마인가?
4. 같은 $200$건으로 HMM emission, 신경망, conformal threshold를 모두 바꾸고 바로 배포하려는 절차를 어떻게 고쳐야 하는가?

### 모범 답안

먼저 관측 전 drift 확률을 전이로 예측한다.

$$
P(S_t=D\mid Y_{1:t-1})
=
0.9(0.05)+0.1(0.80)
=
0.125
$$

정상 예측확률은 $0.875$다. 경보의 evidence는

$$
P(Y_t=1\mid Y_{1:t-1})
=
0.875(0.1)+0.125(0.7)
=
0.175
$$

이므로 drift posterior는

$$
P(D\mid Y_{1:t})
=
\frac{0.125(0.7)}{0.175}
=
0.5
$$

이다. 경보가 drift belief를 $12.5\%$에서 $50\%$로 올렸지만 drift가 확정된 것은 아니다.

따라서 즉시 fine-tuning하면 안 된다. 사전에 정한 HMM threshold와 별도 hard-stop 규칙에 따라 `WATCH` 또는 `HOLD`하고, 배포 버그·데이터 오류·일시 피크·지속 drift를 먼저 구분한다. posterior는 상태에 대한 믿음이지 학습 데이터의 권리·정확성, 과거 능력 보존, 새 threshold의 calibration을 증명하지 않는다.

old-regime forgetting은

$$
F
=
0.92-0.77
=
0.15
$$

즉 $15$ percentage points다. new-regime 점수 $0.89$가 높거나 두 점수의 평균이 괜찮아도 이 회귀를 상쇄했다고 말할 수 없다.

같은 $200$건으로 모든 것을 바꾸고 평가하는 절차는 다음처럼 고친다.

1. 현재 champion, HMM, calibrator, reference, runtime 버전과 원시 로그를 동결한다.
2. event time과 reveal time을 확인하고 성숙 라벨·중복·오염·권리를 감사한다.
3. adaptation용 block에서 새 데이터와 검증된 replay를 함께 쓰고 EWC·distillation 같은 보존 장치를 적용해 별도 challenger를 학습한다.
4. challenger를 동결한 뒤 겹치지 않는 calibration block으로 확률·conformal threshold를 다시 검량한다.
5. HMM emission을 바꿔야 한다면 별도 구간으로 재적합하고 새로운 state threshold도 독립 시계열에서 검증한다. smoothing 결과를 온라인 탐지 성능처럼 사용하지 않는다.
6. 또 다른 final gate block에서 old/new regime, 희귀 안전군, calibration, 비용 상한을 모두 확인한다.
7. shadow와 작은 무작위 canary를 거쳐 승격하고, 실패하면 memory state·calibrator·retriever까지 포함한 검증된 묶음으로 rollback한다.

핵심은 다음 등식을 거부하는 것이다.

$$
\text{상태 posterior 상승}
\neq
\text{안전한 가중치 업데이트 승인}
$$

---

**다음 학습 예고:** 오늘은 이산 latent regime을 HMM으로 추적하고 replay·regularization으로 지속학습의 망각을 제어했다. 다음에는 HMM 모수학습의 Baum–Welch를 더 깊게 유도하고, geometric 체류시간을 벗어나는 Hidden Semi-Markov Model·Bayesian Online Change-Point Detection을 살펴본 뒤, 제한된 replay 예산을 reservoir·coreset·gradient diversity로 배분하는 법을 연결한다.
