# [2026-10-22] 오늘 학습: Baum–Welch·HSMM·온라인 변화점 추론 & 예산 제약 Replay·순차 승인

> **오늘의 핵심 문장:** 숨은 상태를 추정하는 것, 그 상태모형의 모수를 학습하는 것, 변화 시점을 찾는 것, 새 AI 모델을 학습하고 배포하는 것은 서로 다른 문제다. Baum–Welch는 보이지 않는 상태에 소프트 책임도를 부여해 HMM 모수를 다시 맞추고, HSMM은 상태별 체류시간을 직접 모델링하며, BOCPD는 “마지막 변화가 언제였는가?”에 대한 분포를 갱신한다. 제한된 replay buffer는 과거 전체를 대신하는 **학습용 기억**일 뿐이고, 새 모델의 승인은 독립 평가 스트림에서 반복 관찰에도 유효한 confidence sequence로 따로 증명해야 한다.

지난 시간에는 상태공간모형과 HMM의 예측–갱신으로 숨은 운영 체제에 대한 믿음을 추적하고, replay·EWC·distillation으로 지속학습의 파국적 망각을 줄이는 법을 배웠다. 그러나 세 가지 질문이 남았다.

1. 전이확률과 emission을 처음부터 정확히 모른다면 어떻게 배울까?
2. 장애가 보통 정확히 몇 블록 동안 지속된다면 HMM의 geometric 체류시간 가정으로 충분할까?
3. 과거 데이터를 전부 보관할 수 없고 평가 결과를 매일 들여다본다면, 무엇을 기억하고 언제 새 모델을 승인해야 할까?

오늘은 이 질문을 다음 네 층으로 분리한다.

$$
\underbrace{\theta_{\mathrm{HMM}}\leftarrow\operatorname{EM}(Y)}_{
\text{상태모형의 모수학습}
}
\quad\neq\quad
\underbrace{p(r_t\mid Y_{1:t})}_{
\text{변화 시점 추론}
}
\quad\neq\quad
\underbrace{\theta_{\mathrm{AI}}\leftarrow\operatorname{Train}(\mathcal D_{\mathrm{new}},\mathcal B)}_{
\text{후보 모델 적응}
}
\quad\neq\quad
\underbrace{\operatorname{Approve}(\text{challenger})}_{
\text{독립 배포 승인}
}
$$

- **Baum–Welch:** 숨은 상태의 posterior를 소프트 카운트로 바꾸어 HMM의 초기분포·전이·emission을 학습한다.
- **HSMM:** 상태별 duration 분포를 따로 두어 “매 시점 같은 확률로 상태를 떠난다”는 제약을 푼다.
- **BOCPD:** 이름 붙은 상태보다 마지막 변화 이후의 경과시간인 run length를 온라인으로 추적한다.
- **Replay와 순차 승인:** 제한된 메모리로 과거 능력을 보존하되, 학습에 쓴 기억과 승인용 증거를 분리한다.

오늘의 연결 흐름은 다음과 같다.

$$
\text{시계열 로그}
\longrightarrow
\text{Baum--Welch로 HMM 학습}
\longrightarrow
\text{HSMM·BOCPD로 체류·변화 추론}
\longrightarrow
\text{replay 후보 선별}
\longrightarrow
\text{challenger 학습}
\longrightarrow
\text{confidence sequence 승인}
\longrightarrow
\text{shadow·canary 또는 기각}
$$

> **난이도 표지:** Baum–Welch는 흐린 유리 너머 사람들에게 “A일 확률 $0.8$, B일 확률 $0.2$” 같은 **분수 출석표**를 붙이고, 그 출석표로 반별 통계를 다시 계산하는 과정이다. HSMM은 각 상태에 별도의 체류시간 시계를 달고, BOCPD는 “마지막 변화가 $0,1,2,\ldots$블록 전”이라는 여러 스톱워치를 동시에 추적한다. Replay buffer는 작은 기억 창고이고, confidence sequence는 중간 결과를 여러 번 확인해도 전체 오경보 확률을 지키는 안전띠다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 HMM의 상태만 추론해서는 부족할까?

어제는 초기분포 $\boldsymbol\pi$, 전이행렬 $A$, emission $b_j(y)$가 주어졌다고 가정하고 filtering을 계산했다. 실제 운영에서는 이 숫자를 모르는 경우가 더 많다.

- 정상 상태에서도 경보가 날 확률은 얼마인가?
- 열화 상태가 다음 블록에도 이어질 확률은 얼마인가?
- 복구 상태의 latency 분포는 어떤 평균과 분산을 갖는가?

상태 라벨 $S_t$를 직접 볼 수 있다면 상태별 횟수와 평균을 세면 된다. 하지만 상태가 숨겨져 있으므로 “이 관측은 정상 데이터다”라고 딱 잘라 분류한 뒤 학습하면 초기 오분류가 모수를 왜곡하고, 왜곡된 모수가 다시 더 나쁜 분류를 만든다. 그래서 각 관측을 한 상태에 강제로 넣는 대신 모든 상태에 **posterior 책임도**만큼 나누어 세는 Baum–Welch가 필요하다.

### 1.2 왜 EM은 추측과 재계산을 번갈아 할까?

숨은 상태와 모수를 동시에 모르면 닭과 달걀 문제가 생긴다.

- 모수를 알아야 각 시점의 상태 posterior를 계산할 수 있다.
- 상태를 알아야 전이 횟수와 emission 모수를 계산할 수 있다.

Expectation–Maximization(EM)은 이 문제를 두 단계로 나눈다.

1. **E-step:** 현재 모수 아래에서 숨은 상태와 전이의 posterior 기대 횟수를 계산한다.
2. **M-step:** 그 기대 횟수를 실제 소프트 카운트처럼 사용해 모수를 다시 맞춘다.

EM은 한 번에 정답을 맞히는 마법이 아니다. 현재 설명을 바탕으로 분수 출석표를 만들고, 그 표로 반의 통계를 다시 계산하는 일을 반복한다.

### 1.3 왜 HMM의 체류시간 가정이 문제가 될까?

표준 HMM에서 상태 $j$에 머무는 동안 매 블록 “계속 남을지 떠날지”를 같은 확률 $A_{jj}$로 다시 뽑는다. 따라서 체류시간은 geometric 분포가 된다. 이는 **기억 없음(memoryless)**을 뜻한다.

하지만 현실의 duration은 자주 이와 다르다.

- 정기 점검은 거의 항상 $6$블록 동안 지속된다.
- 장애는 첫 두 블록 안에 복구될 확률이 낮지만, 복구 작업이 시작된 뒤에는 급격히 끝날 수 있다.
- 프로모션은 사전에 정한 기간만큼 지속된다.

이때 상태가 얼마나 오래 지속됐는지 자체가 다음 종료확률을 바꾼다. Hidden Semi-Markov Model(HSMM)은 상태 전이와 별도로 duration 분포를 직접 둔다.

### 1.4 왜 BOCPD는 HMM·HSMM과 다른 질문을 할까?

운영자가 항상 “정상·피크·장애 중 어느 상태인가?”를 묻는 것은 아니다. 때로는 다음 질문이 더 중요하다.

> **현재 구간을 만든 마지막 변화가 언제 일어났는가?**

Bayesian Online Change-Point Detection(BOCPD)은 run length $r_t$, 즉 마지막 변화 이후 얼마나 많은 관측이 이어졌는지에 대한 posterior를 매 관측마다 갱신한다. 새 구간의 모수가 과거 상태 중 하나와 같은지, 완전히 새로운지는 별도 문제다. 따라서 BOCPD 경보를 HMM의 특정 상태 라벨로 곧바로 번역하면 안 된다.

### 1.5 왜 모든 과거 데이터를 그냥 저장하지 못할까?

서비스 로그는 끝없이 늘어난다. 원본 전체를 replay하면 다음 문제가 생긴다.

- 저장비용과 개인정보 보존기간이 커진다.
- 매 학습의 I/O와 GPU 시간이 계속 증가한다.
- 중복이 많은 흔한 사례가 희귀하지만 중요한 안전 사례를 밀어낸다.
- 삭제 요청과 데이터 권리 변경을 반영하기 어려워진다.

그래서 크기 $M$인 replay buffer 안에 과거의 어떤 부분을 남길지 결정해야 한다. Reservoir sampling은 지금까지 본 각 항목에 같은 포함 기회를 주고, coreset은 표현공간의 덮개를 만들며, gradient diversity는 서로 다른 학습 방향을 보존하려 한다.

### 1.6 왜 균등 표본만으로 희귀 안전군을 지키기 어려울까?

희귀 안전사고가 전체의 $0.01\%$라면 크기 $1{,}000$의 완전 균등 reservoir에 들어 있는 기대 개수는 $0.1$개다. 기대값이 $0.1$이라는 것은 최소 한 개가 보장된다는 뜻이 아니다. 대부분의 buffer에 하나도 없을 수 있다.

따라서 실무 buffer는 흔히 다음을 섞는다.

- 전체 분포를 위한 reservoir 영역
- 사전 등록한 고위험 그룹별 최소 quota
- 최근 변화에 적응하기 위한 짧은 recency 영역
- 표현 또는 gradient coverage를 위한 coreset 영역

다만 quota와 어려운 사례 선별은 학습분포를 바꾼다. 전체 모집단 평균 위험을 추정하려면 포함확률을 기록하고 적절히 재가중해야 한다. 반대로 최악 그룹을 직접 최적화하려는 목적이라면 그 목적함수를 명시해야 한다.

### 1.7 왜 평가를 자주 볼수록 고정 시점 신뢰구간이 위험할까?

매일 새 모델과 기존 모델을 비교하고 “처음으로 $95\%$ 신뢰구간이 $0$ 위로 올라온 날” 배포한다고 하자. 각 날짜의 신뢰구간이 따로 $95\%$라 해도 여러 번 들여다보다가 우연히 좋은 날만 멈추면 전체 거짓 승인 확률은 $5\%$보다 커질 수 있다. 이것이 **optional stopping** 문제다.

Confidence sequence는 한 번의 고정 표본크기만 덮는 구간이 아니라, 확률 $1-\delta$ 이상으로 모든 시점의 목표값을 동시에 덮는 구간열이다. 미리 정한 규칙 안에서는 중간에 몇 번 확인하고 언제 멈춰도 오류예산을 지킬 수 있다.

하지만 confidence sequence가 모든 것을 해결하지는 않는다.

- 같은 데이터로 후보를 고르고 평가하면 선택 편향이 남는다.
- 나중에 불리한 그룹을 빼거나 유리한 metric을 추가하면 다중선택 문제가 남는다.
- 분포가 바뀌거나 라벨 누락이 결과 의존적이면 추정 대상 자체가 달라질 수 있다.
- 학습에 쓴 replay 사례를 승인 증거로 재사용하면 독립성이 깨진다.

### 1.8 오늘의 본질

오늘의 핵심 질문은 “더 영리한 buffer 알고리즘은 무엇인가?”가 아니다.

> **숨은 상태 학습, 변화 탐지, 학습용 기억 선택, 후보 모델 훈련, 배포 승인에 각각 어떤 데이터와 오류예산을 썼는지 추적할 수 있는가?**

한 단계의 posterior나 점수가 다음 단계의 허가증은 아니다.

$$
\text{변화 posterior 상승}
\neq
\text{replay 데이터의 정당성}
\neq
\text{challenger의 안전성}
\neq
\text{배포 승인}
$$

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| $Y_{1:T}$ | 시점 $1$부터 $T$까지 실제로 본 관측열 | HMM·HSMM·BOCPD가 설명할 데이터다. |
| $S_t$ | 시점 $t$의 보이지 않는 상태 | 정상·피크·장애 같은 latent regime이다. |
| $K$ | HMM 상태 개수 | $S_t\in\{1,\ldots,K\}$로 쓴다. |
| $\boldsymbol\pi$ | 첫 시점 상태의 확률벡터 | HMM의 초기분포다. Replay 포함확률과 구분한다. |
| $A_{ij}$ | 상태 $i$에서 다음 시점 상태 $j$로 갈 확률 | 전이행렬 $A$의 원소다. |
| emission $b_j(y;\eta_j)$ | 상태 $j$가 관측 $y$를 만들 확률 또는 밀도 | 상태와 로그를 연결하며 $\eta_j$가 그 모수다. |
| likelihood | 모수가 주어졌을 때 실제 관측이 나온 그럴듯함 | Baum–Welch가 반복해 높이는 대상이다. |
| complete data | 관측 $Y$뿐 아니라 숨은 상태 $S$까지 안다고 가정한 자료 | 완전자료 likelihood는 계산과 최적화가 쉽다. |
| incomplete data | 실제처럼 $Y$만 보고 $S$는 모르는 자료 | 상태의 모든 가능성을 합해야 한다. |
| EM | 숨은 변수의 posterior 기대값 계산과 모수 최적화를 번갈아 하는 방법 | Baum–Welch의 일반 뼈대다. |
| Baum–Welch | HMM에 특화된 EM 알고리즘 | forward–backward로 E-step을 수행한다. |
| E-step | 현재 모수에서 숨은 상태의 posterior 책임도를 계산하는 단계 | $\gamma,\xi$를 만든다. |
| M-step | 책임도를 고정하고 기대 완전자료 로그가능도를 최대화하는 단계 | $\pi,A,\eta$를 갱신한다. |
| $\theta^{(m)}$ | EM 반복 $m$번째의 전체 모수 | 다음 E-step의 출발점이다. |
| $Q(\theta\mid\theta^{(m)})$ | 현재 posterior 아래 완전자료 로그가능도의 기대값 | M-step의 목적함수다. |
| $\gamma_t(j)$ | $Y_{1:T}$를 모두 본 뒤 시점 $t$가 상태 $j$일 확률 | 상태 $j$의 소프트 출석표다. |
| $\xi_t(i,j)$ | 시점 $t$의 상태가 $i$, $t+1$이 $j$일 posterior 확률 | $i\to j$ 전이의 소프트 횟수다. |
| $N_j$ | $\sum_t\gamma_t(j)$ | 상태 $j$가 담당한 유효 표본 수다. |
| forward probability | 관측을 왼쪽에서 오른쪽으로 읽으며 누적한 결합확률 | filtering과 likelihood 계산에 쓰인다. |
| backward probability | 미래 관측의 가능성을 오른쪽에서 왼쪽으로 전달한 값 | smoothing 책임도 계산에 쓰인다. |
| scaling | 아주 작은 확률의 연속 곱을 매 시점 정규화하는 기법 | 수치 underflow를 막는다. |
| generalized EM | M-step에서 정확한 최댓값 대신 $Q$를 증가시키는 갱신을 쓰는 EM | 신경망 emission처럼 닫힌형 해가 없을 때 쓴다. |
| 책임도(responsibility) | 한 관측을 각 숨은 상태가 얼마나 책임지는지 나타내는 확률 | hard label 대신 분수 카운트를 만든다. |
| 식별성(identifiability) | 무한한 데이터가 있어도 서로 다른 모수를 관측분포로 구분할 수 있는 성질 | 상태 의미를 믿기 전에 확인해야 한다. |
| label switching | 상태 번호를 동시에 바꿔도 같은 관측분포가 되는 대칭 | `상태 1` 자체에는 고정 의미가 없다. |
| state collapse | 여러 상태가 거의 같은 emission·전이를 배우거나 한 상태의 질량이 사라지는 현상 | $K$가 크거나 초기화가 나쁠 때 생긴다. |
| local optimum | 주변에서는 더 좋아지기 어렵지만 전역 최선은 아닐 수 있는 해 | EM의 단조 증가가 전역 정답을 보장하지 않는 이유다. |
| Dirichlet·Beta prior | 범주확률·Bernoulli 확률에 주는 사전분포 | 0-count와 과도한 확신을 완화한다. |
| MLE | 데이터 likelihood를 가장 크게 하는 모수 | 기본 M-step 해다. |
| MAP | likelihood와 prior를 합친 posterior를 가장 크게 하는 모수 | MLE·posterior mean과 구분한다. |
| 체류시간 $D$ | 한 상태에 연속으로 머문 블록 수 | HMM과 HSMM의 핵심 차이다. |
| geometric 분포 | 매 블록 같은 이탈확률을 갖는 memoryless duration 분포 | 표준 HMM에 암묵적으로 들어 있다. |
| HSMM | explicit duration을 갖는 Hidden Semi-Markov Model | 비-geometric 체류시간을 표현한다. |
| $q_j(d)$ | 상태 $j$의 duration이 정확히 $d$일 확률질량함수 | HSMM의 duration 모수다. |
| survival $P(D\ge d)$ | 상태가 적어도 $d$블록 유지될 확률 | hazard의 분모다. |
| hazard $h_j(d)$ | 이미 $d$블록까지 왔다는 조건에서 그 길이에 종료될 확률 | duration의 나이에 따른 종료 성향이다. |
| right censoring | 관측창이 끝나 아직 segment 종료를 보지 못한 상태 | 완료된 짧은 duration처럼 세면 편향된다. |
| segment | 같은 상태가 이어지는 하나의 연속 구간 | HSMM 계산의 기본 단위다. |
| BOCPD | Bayesian Online Change-Point Detection | 마지막 변화 시점의 불확실성을 온라인 추적한다. |
| run length $r_t$ | 이 문서에서는 새 segment의 첫 관측에서 $0$인 경과 인덱스 | 현재 segment의 관측 수는 $r_t+1$이며 $0,1,2,\ldots$의 posterior를 유지한다. |
| changepoint | 데이터 생성 규칙이 바뀌는 경계 | 곧바로 장애나 새 상태를 뜻하지는 않는다. |
| prior predictive $p_0(y)$ | 새 segment에 데이터가 하나도 없을 때 prior로 만든 예측분포 | BOCPD의 reset 가지에 쓰인다. |
| posterior predictive $p^{(m)}(y)$ | 현재 segment의 $m$개 관측으로 모수를 갱신한 뒤의 다음 관측 예측분포 | BOCPD의 growth 가지에 쓰며 이 문서에서는 $m=r_t+1$이다. |
| replay buffer $\mathcal B$ | 과거 데이터 중 다시 학습할 일부를 담는 제한된 저장소 | 파국적 망각을 줄이는 학습용 기억이다. |
| replay budget $M$ | buffer가 보관할 수 있는 항목 또는 독립 단위의 최대 개수 | 선택 알고리즘의 자원 제약이다. |
| reservoir sampling | 길이를 모르는 스트림에서 각 과거 항목의 포함확률을 같게 만드는 표집 | 평균적 대표성을 위한 기준선이다. |
| 포함확률 $\rho_i$ | 항목 $i$가 buffer 또는 표본에 들어올 확률 | $\pi$와 혼동하지 않고 재가중에 쓴다. |
| stratum | 언어·위험군·상태·출처처럼 사전에 나눈 층 | 희귀군 최소 quota를 둘 수 있다. |
| coreset | 큰 데이터의 기하·손실·그래디언트를 근사하도록 고른 작은 부분집합 | 제한된 budget의 표현력을 높인다. |
| embedding $\phi(z_i)$ | 사례 $z_i$를 수치 벡터로 표현한 것 | 거리 기반 coreset의 좌표다. |
| per-example gradient $g_i$ | 사례 하나가 파라미터를 움직이려는 방향 | 충돌과 다양성을 측정한다. |
| gradient interference | 새 사례와 과거 사례의 gradient 내적이 음수여서 한쪽 개선이 다른 쪽을 해칠 가능성 | 망각 위험의 국소 신호다. |
| importance weight | 표집분포와 목표분포의 차이를 보정하는 가중치 | quota·우선표집 뒤 평균 목표를 복원할 때 쓴다. |
| propensity floor | 포함확률이 너무 작아 역확률 가중치가 폭발하지 않도록 둔 하한 | 분산과 boundedness를 지킨다. |
| champion·challenger | 현재 승인 모델과 새 후보 모델 | 후보가 증거 없이 현재 모델을 덮어쓰지 않게 한다. |
| paired difference $Z_i$ | 같은 평가 단위에서 challenger 점수와 champion 점수의 차이 | 공통 난이도를 상쇄해 비교 분산을 줄인다. |
| filtration $\mathcal F_{t-1}$ | 시점 $t$ 직전까지 알고 있던 모든 정보 | 순차 추론의 조건부 정보 집합이다. |
| confidence sequence | 모든 시점에서 동시에 목표값을 덮는 신뢰구간열 | 반복 확인과 데이터 의존 중지에 대응한다. |
| optional stopping | 중간 결과를 보고 유리한 시점에 실험을 멈추는 행위 | 고정 시점 구간의 오류율을 깨뜨릴 수 있다. |
| alpha ledger | 그룹·metric·후보·재시작에 나눈 총 오류확률의 장부 | 전체 거짓 승인 확률을 관리한다. |
| LCB·UCB | 추정값에서 폭을 빼거나 더한 신뢰 하한·상한 | 효용은 LCB, 안전비용은 UCB로 보수적으로 본다. |
| estimand | 우리가 추정하려는 정확한 모집단 양 | 현재 사용자 평균인지 그룹별 평균인지 먼저 고정한다. |
| ESS | 가중치가 있을 때 실제 정보량이 몇 개 균등 표본과 비슷한지 나타내는 유효표본크기 | 큰 원시 표본 수에 속지 않게 한다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 문제 설정: 무엇을 알고 무엇을 모르는가?

하나의 관측열을

$$
Y_{1:T}=(Y_1,\ldots,Y_T)
$$

라고 하자. 숨은 상태는

$$
S_t\in\{1,\ldots,K\}
$$

이고 HMM 모수 전체를

$$
\theta
=
\left(
\boldsymbol\pi,
A,
\eta_1,\ldots,\eta_K
\right)
$$

라고 쓰자. 각 부품의 shape와 제약은 다음과 같다.

$$
\boldsymbol\pi\in[0,1]^K,
\qquad
\sum_{j=1}^K\pi_j=1,
$$

$$
A\in[0,1]^{K\times K},
\qquad
\sum_{j=1}^KA_{ij}=1
\quad\text{for every }i,
$$

$$
b_j(y;\eta_j)=p(Y_t=y\mid S_t=j).
$$

HMM의 결합분포는

$$
p_\theta(s_{1:T},y_{1:T})
=
\pi_{s_1}b_{s_1}(y_1;\eta_{s_1})
\prod_{t=1}^{T-1}
A_{s_t s_{t+1}}
b_{s_{t+1}}(y_{t+1};\eta_{s_{t+1}})
$$

이다. 상태열 $s_{1:T}$까지 보이면 각 항을 세기 쉽다. 실제로는 상태가 숨겨져 있으므로 관측 likelihood는 가능한 모든 상태열을 합해야 한다.

$$
p_\theta(y_{1:T})
=
\sum_{s_{1:T}}
p_\theta(s_{1:T},y_{1:T})
$$

상태열 후보는 $K^T$개다. Forward 알고리즘이 이 합을 동적계획법으로 계산해 주지만, 로그 안에 거대한 합이 들어간

$$
\ell(\theta)
=
\log p_\theta(y_{1:T})
$$

를 직접 최적화하는 일은 여전히 어렵다. Baum–Welch는 완전자료의 쉬운 구조를 이용한다.

### 3.2 완전자료 로그가능도는 세 개의 장부로 나뉜다

지시함수 $\mathbf 1\{\cdot\}$는 조건이 참이면 $1$, 아니면 $0$이다. 상태열까지 안다고 가정하면 로그가능도는

$$
\begin{aligned}
\log p_\theta(s_{1:T},y_{1:T})
={}&
\sum_{j=1}^K
\mathbf 1\{s_1=j\}\log\pi_j\\
&+
\sum_{t=1}^{T-1}
\sum_{i=1}^K\sum_{j=1}^K
\mathbf 1\{s_t=i,s_{t+1}=j\}\log A_{ij}\\
&+
\sum_{t=1}^T\sum_{j=1}^K
\mathbf 1\{s_t=j\}\log b_j(y_t;\eta_j).
\end{aligned}
$$

세 줄은 각각 다음 장부다.

1. 첫 상태가 무엇이었는가?
2. 각 전이 $i\to j$가 몇 번 일어났는가?
3. 상태 $j$가 어떤 관측을 몇 번 만들었는가?

숨은 지시함수를 posterior 확률로 바꾸면 소프트 장부가 된다.

### 3.3 E-step: 상태와 전이의 소프트 카운트를 만든다

현재 모수를 $\theta^{(m)}$라 하자. E-step은 전체 관측열을 본 smoothing posterior 두 종류를 계산한다.

$$
\gamma_t(j)
=
P_{\theta^{(m)}}
(S_t=j\mid Y_{1:T}=y_{1:T}),
$$

$$
\xi_t(i,j)
=
P_{\theta^{(m)}}
(S_t=i,S_{t+1}=j\mid Y_{1:T}=y_{1:T}).
$$

이 값들은 다음 일관성을 가진다.

$$
\sum_{j=1}^K\gamma_t(j)=1,
$$

$$
\sum_{j=1}^K\xi_t(i,j)=\gamma_t(i),
\qquad
\sum_{i=1}^K\xi_t(i,j)=\gamma_{t+1}(j).
$$

$\gamma_t(j)=0.8$은 시점 $t$에 상태 $j$가 $80\%$ 존재했다는 물리적 뜻이 아니다. 현재 모형과 전체 관측 아래 상태 $j$일 posterior 확률이 $0.8$이라는 뜻이다.

기대 완전자료 로그가능도는

$$
Q(\theta\mid\theta^{(m)})
=
E_{S\mid Y,\theta^{(m)}}
\left[
\log p_\theta(S,Y)
\right]
$$

이고, 지시함수의 기대값을 $\gamma,\xi$로 바꾸면

$$
\begin{aligned}
Q(\theta\mid\theta^{(m)})
={}&
\sum_{j=1}^K\gamma_1(j)\log\pi_j\\
&+
\sum_{t=1}^{T-1}\sum_{i=1}^K\sum_{j=1}^K
\xi_t(i,j)\log A_{ij}\\
&+
\sum_{t=1}^T\sum_{j=1}^K
\gamma_t(j)\log b_j(y_t;\eta_j).
\end{aligned}
$$

이제 $\gamma,\xi$를 고정하면 $\pi$, $A$, 각 emission 모수를 따로 최적화할 수 있다.

### 3.4 Forward–backward로 $\gamma$와 $\xi$를 계산한다

Forward 값의 정규화 전 계산을

$$
\widetilde\alpha_1(j)
=
\pi_j b_j(y_1)
$$

로 두고 scaling 상수와 정규화 값을

$$
c_1=\sum_{j=1}^K\widetilde\alpha_1(j),
\qquad
\alpha_1(j)=\frac{\widetilde\alpha_1(j)}{c_1}
$$

로 둔다. $t\ge2$에서는

$$
\widetilde\alpha_t(j)
=
b_j(y_t)
\sum_{i=1}^K\alpha_{t-1}(i)A_{ij},
$$

$$
c_t=\sum_{j=1}^K\widetilde\alpha_t(j),
\qquad
\alpha_t(j)=\frac{\widetilde\alpha_t(j)}{c_t}.
$$

이 표기에서 관측 로그가능도는

$$
\log p_\theta(y_{1:T})
=
\sum_{t=1}^T\log c_t
$$

이다. 아주 작은 확률을 길게 곱하지 않으므로 underflow를 막는다.

같은 scaling 규약에서 backward 값은

$$
\beta_T(j)=1
$$

로 시작하여

$$
\beta_t(i)
=
\frac{
\sum_{j=1}^K
A_{ij}b_j(y_{t+1})\beta_{t+1}(j)
}{c_{t+1}}
$$

로 계산할 수 있다. 그러면

$$
\gamma_t(j)
=
\frac{\alpha_t(j)\beta_t(j)}
{\sum_{k=1}^K\alpha_t(k)\beta_t(k)}
$$

이고

$$
\xi_t(i,j)
=
\frac{
\alpha_t(i)A_{ij}b_j(y_{t+1})\beta_{t+1}(j)
}{
\sum_{a=1}^K\sum_{b=1}^K
\alpha_t(a)A_{ab}b_b(y_{t+1})\beta_{t+1}(b)
}.
$$

구현에서는 log-space `log-sum-exp` 또는 일관된 scaling 중 하나를 사용한다. Forward는 scaled인데 backward는 unscaled인 값을 섞으면 책임도가 틀어진다.

### 3.5 왜 EM은 관측 likelihood를 감소시키지 않는가?

임의의 상태분포 $q(s)$를 도입하면

$$
\begin{aligned}
\log p_\theta(y)
&=
\log\sum_s q(s)
\frac{p_\theta(y,s)}{q(s)}\\
&\ge
\sum_s q(s)
\log\frac{p_\theta(y,s)}{q(s)}
\equiv
\mathcal L(q,\theta)
\end{aligned}
$$

이다. 마지막 부등식은 로그함수의 오목성에 대한 Jensen 부등식이다. 더 정확히는

$$
\log p_\theta(y)
=
\mathcal L(q,\theta)
+
D_{\mathrm{KL}}
\left(
q(S)\,Vert\,p_\theta(S\mid y)
\right).
$$

E-step에서

$$
q^{(m)}(S)
=
p_{\theta^{(m)}}(S\mid y)
$$

로 두면 KL 항이 $0$이 되어 현재 모수에서 lower bound가 likelihood와 맞닿는다. M-step이

$$
Q(\theta^{(m+1)}\mid\theta^{(m)})
\ge
Q(\theta^{(m)}\mid\theta^{(m)})
$$

를 만족하면

$$
\ell(\theta^{(m+1)})
\ge
\ell(\theta^{(m)})
$$

가 된다.

그러나 이 단조성은 다음을 보장하지 않는다.

- 전역 최댓값에 도달한다.
- 선택한 상태 수 $K$가 옳다.
- 상태가 현실의 원인과 일치한다.
- held-out 예측이 좋아진다.
- 상태의 의미가 반복 실행에서 같은 번호를 갖는다.

신경망 emission처럼 M-step의 닫힌형 해가 없을 때는 gradient step이 실제로 $Q$를 증가시키는지 확인해야 generalized EM의 단조성 설명을 적용할 수 있다.

### 3.6 초기분포의 M-step

$\boldsymbol\pi$에 관련된 항만 떼면

$$
Q_\pi
=
\sum_{j=1}^K\gamma_1(j)\log\pi_j
$$

이고 제약은 $\sum_j\pi_j=1$이다. 라그랑지안을

$$
\mathcal J
=
\sum_j\gamma_1(j)\log\pi_j
+
\lambda\left(\sum_j\pi_j-1\right)
$$

로 두면

$$
\frac{\partial\mathcal J}{\partial\pi_j}
=
\frac{\gamma_1(j)}{\pi_j}+\lambda=0.
$$

합 제약을 사용하면

$$
\pi_j^{\mathrm{new}}
=
\gamma_1(j)
$$

를 얻는다. 독립 시퀀스가 $R$개라면 각 시퀀스의 첫 시점 책임도를 평균한다.

$$
\pi_j^{\mathrm{new}}
=
\frac{1}{R}
\sum_{r=1}^R\gamma_{r,1}(j).
$$

서로 다른 사용자 세션을 하나의 긴 시퀀스로 이어 붙이고 경계에서 가짜 전이를 세면 안 된다.

### 3.7 전이행렬의 M-step

상태 $i$에서 나가는 전이에 관련된 항은

$$
Q_{A_i}
=
\sum_{j=1}^K
\left(
\sum_{t=1}^{T-1}\xi_t(i,j)
\right)
\log A_{ij}
$$

이다. 기대 전이 횟수를

$$
N_{ij}
=
\sum_{t=1}^{T-1}\xi_t(i,j)
$$

라고 두고 $\sum_jA_{ij}=1$ 아래 최적화하면

$$
A_{ij}^{\mathrm{new}}
=
\frac{N_{ij}}{\sum_{k=1}^KN_{ik}}
=
\frac{
\sum_{t=1}^{T-1}\xi_t(i,j)
}{
\sum_{t=1}^{T-1}\gamma_t(i)
}.
$$

분모에 $t=T$의 $\gamma_T(i)$를 넣지 않는다. 마지막 상태 뒤에는 관측된 다음 전이가 없기 때문이다.

### 3.8 범주형 emission의 M-step

관측 $Y_t$가 $C$개 범주 중 하나라고 하자. 상태 $j$의 범주 $c$ 확률을

$$
B_{jc}=P(Y_t=c\mid S_t=j),
\qquad
\sum_{c=1}^CB_{jc}=1
$$

로 둔다. 소프트 범주 횟수는

$$
N_{jc}
=
\sum_{t=1}^T
\gamma_t(j)\mathbf 1\{y_t=c\}
$$

이고 상태 $j$의 총 유효 표본 수는

$$
N_j
=
\sum_{c=1}^CN_{jc}
=
\sum_{t=1}^T\gamma_t(j)
$$

이다. 전이행렬과 같은 라그랑주 승수 계산으로

$$
B_{jc}^{\mathrm{new}}
=
\frac{N_{jc}}{N_j}
$$

를 얻는다. 즉, hard count 대신 posterior로 가중한 count를 쓴다.

### 3.9 Bernoulli emission은 가중 성공률이다

$Y_t\in\{0,1\}$이고

$$
P(Y_t=1\mid S_t=j)=p_j
$$

라 하자. 상태 $j$의 emission 항은

$$
Q_j(p_j)
=
\sum_{t=1}^T\gamma_t(j)
\left[
y_t\log p_j
+
(1-y_t)\log(1-p_j)
\right].
$$

미분하면

$$
\frac{\partial Q_j}{\partial p_j}
=
\sum_t\gamma_t(j)
\left(
\frac{y_t}{p_j}
-
\frac{1-y_t}{1-p_j}
\right).
$$

이를 $0$으로 놓으면

$$
p_j^{\mathrm{new}}
=
\frac{
\sum_{t=1}^T\gamma_t(j)y_t
}{
\sum_{t=1}^T\gamma_t(j)
}
$$

가 된다. 경보 $1$의 posterior 가중 비율이다.

### 3.10 Gaussian emission의 평균과 공분산

$d$차원 관측에서

$$
Y_t\mid S_t=j
\sim
\mathcal N(\mu_j,\Sigma_j)
$$

라고 하자. 상수항을 제외한 목적함수는

$$
Q_j
=
-\frac12
\sum_{t=1}^T\gamma_t(j)
\left[
\log|\Sigma_j|
+
(y_t-\mu_j)^\top
\Sigma_j^{-1}
(y_t-\mu_j)
\right].
$$

가중 Gaussian MLE는

$$
\mu_j^{\mathrm{new}}
=
\frac{
\sum_{t=1}^T\gamma_t(j)y_t
}{N_j}
$$

와

$$
\Sigma_j^{\mathrm{new}}
=
\frac{
\sum_{t=1}^T\gamma_t(j)
(y_t-\mu_j^{\mathrm{new}})
(y_t-\mu_j^{\mathrm{new}})^\top
}{N_j}
$$

이다. 여기서 분모는 가중 MLE의 $N_j$다. 표본공분산의 불편보정 $N_j-1$을 기계적으로 넣지 않는다. 공분산은 반드시 **새 평균** $\mu_j^{\mathrm{new}}$로 중심화한다.

$N_j$가 작거나 관측 차원이 크면 $\Sigma_j$가 singular해질 수 있다. 실무에서는 예를 들어

$$
\Sigma_{j,\lambda}
=
\Sigma_j^{\mathrm{new}}+\lambda I,
\qquad \lambda>0
$$

처럼 regularization을 쓰거나 diagonal·shared covariance를 사용한다. 이는 순수한 unconstrained MLE와 다른 모델링 선택이다. 특히 MLE를 구한 뒤 $\lambda I$를 사후에 더하는 heuristic은 원래 raw likelihood에 대한 정확한 M-step이 아니므로 EM의 likelihood 단조성 보장을 그대로 주장할 수 없다. MAP prior나 제약된 공분산 모형에서 해당 목적을 실제로 최적화했는지 구분해야 한다.

### 3.11 Prior smoothing: MLE·MAP·posterior mean을 구분한다

범주확률에 Dirichlet prior

$$
(B_{j1},\ldots,B_{jC})
\sim
\operatorname{Dirichlet}(\alpha_1,\ldots,\alpha_C)
$$

를 두자. 먼저 숨은 상태 경로 $S_{1:T}$까지 관측되었다면, 실제 범주 count를 $N_{jc}(S,Y)$와 $N_j(S)$라고 쓸 때 정확한 conjugate posterior mean은

$$
E[B_{jc}\mid S,Y]
=
\frac{N_{jc}(S,Y)+\alpha_c}
{N_j(S)+\sum_{r=1}^C\alpha_r}
$$

이다. 그러나 HMM에서는 $S$가 숨겨져 있다. Baum–Welch의 $N_{jc}$와 $N_j$는 현재 파라미터 아래에서 구한 **soft expected count**이므로, 이를 위 식에 대입한

$$
\widetilde B_{jc}^{\text{soft-mean}}
=
\frac{N_{jc}+\alpha_c}
{N_j+\sum_{r=1}^C\alpha_r}
$$

은 exact $E[B_{jc}\mid Y]$가 아니라 soft-count smoothing 또는 variational-style update다. 전체 Bayesian posterior는 가능한 상태 경로와 파라미터를 함께 적분하므로 일반적으로 이 한 줄의 비율로 끝나지 않는다.

한편 모든 $\alpha_c>1$이고 interior mode가 존재할 때 $Q(\theta\mid\theta^{\mathrm{old}})+\log p(\theta)$를 최대화하는 MAP-EM 갱신은

$$
B_{jc}^{\mathrm{MAP}}
=
\frac{N_{jc}+\alpha_c-1}
{N_j+\sum_{r=1}^C(\alpha_r-1)}
$$

이다. 이 갱신이 겨냥하는 것은 plain log-likelihood가 아니라 log posterior, 즉 prior가 더해진 penalized objective다. 두 식은 다르다. 흔히 말하는 add-one smoothing은 prior 표기 관례에 따라 posterior mean으로 설명되는 경우가 많으므로 “MAP와 같다”고 무심코 단정하면 안 된다.

Bernoulli $p_j\sim\operatorname{Beta}(a,b)$에서도 soft expected count를 쓰는 대응 갱신은

$$
\widetilde p_j^{\text{soft-mean}}
=
\frac{a+\sum_t\gamma_t(j)y_t}
{a+b+N_j}
$$

이다. 이것 역시 exact $E[p_j\mid Y]$라고 부르면 안 된다. Prior는 데이터가 없는 상태에서 수치 폭주를 막지만, 잘못 선택한 상태 수나 관측모형을 고쳐 주지는 않는다.

### 3.12 한 번의 Baum–Welch 갱신을 손으로 계산한다

$K=2$, $T=4$이고 E-step 결과가 다음과 같다고 하자.

$$
\begin{array}{c|cccc}
t&1&2&3&4\\
\hline
\gamma_t(1)&0.9&0.7&0.2&0.1\\
\gamma_t(2)&0.1&0.3&0.8&0.9
\end{array}
$$

연속 시점의 pair posterior는

$$
\xi_1
=
\begin{bmatrix}
0.65&0.25\\
0.05&0.05
\end{bmatrix},
\quad
\xi_2
=
\begin{bmatrix}
0.18&0.52\\
0.02&0.28
\end{bmatrix},
\quad
\xi_3
=
\begin{bmatrix}
0.08&0.12\\
0.02&0.78
\end{bmatrix}
$$

라고 하자. 각 행합과 열합이 해당 $\gamma_t,\gamma_{t+1}$와 맞는다.

초기분포는

$$
\boldsymbol\pi^{\mathrm{new}}
=
(0.9,0.1)
$$

이다. 상태 $1$에서 나가는 기대 전이 횟수는

$$
N_{11}=0.65+0.18+0.08=0.91,
$$

$$
N_{12}=0.25+0.52+0.12=0.89
$$

이고 분모는 $0.91+0.89=1.8$이다. 따라서

$$
(A_{11}^{\mathrm{new}},A_{12}^{\mathrm{new}})
=
\left(
\frac{0.91}{1.8},
\frac{0.89}{1.8}
\right)
\approx
(0.506,0.494).
$$

상태 $2$에서는

$$
N_{21}=0.09,
\qquad
N_{22}=1.11,
$$

이므로

$$
(A_{21}^{\mathrm{new}},A_{22}^{\mathrm{new}})
=
(0.075,0.925).
$$

관측이

$$
y=(0,0,1,1)
$$

인 Bernoulli라면

$$
p_1^{\mathrm{new}}
=
\frac{0.2+0.1}{0.9+0.7+0.2+0.1}
=
\frac{0.3}{1.9}
\approx0.158,
$$

$$
p_2^{\mathrm{new}}
=
\frac{0.8+0.9}{0.1+0.3+0.8+0.9}
=
\frac{1.7}{2.1}
\approx0.810.
$$

상태 $2$는 경보를 더 자주 만들고 더 오래 지속되는 쪽으로 갱신됐다. 그러나 이를 곧바로 “상태 $2$는 실제 장애”라고 이름 붙이려면 독립적인 외부 라벨과 운영 근거가 필요하다.

### 3.13 여러 시퀀스·누락 관측·빈 상태

여러 독립 시퀀스가 있을 때 complete likelihood는 시퀀스별 합이다. 전이는 각 시퀀스 내부에서만 센다.

$$
N_{ij}
=
\sum_{r=1}^R
\sum_{t=1}^{T_r-1}
\xi_{r,t}(i,j).
$$

시점 $t$의 emission 관측이 누락됐지만 missingness가 상태와 값에 대해 적절히 무시 가능하다고 가정한다면 그 시점 emission factor를 $1$로 marginalize하고 전이 정보는 유지할 수 있다. 하지만 라벨이 결과가 나쁠 때 더 자주 누락되는 MNAR 구조라면 단순 생략은 편향을 만든다.

$N_j\approx0$인 상태에서는 emission M-step이 불안정하거나 정의되지 않는다. 선택지는 다음과 같지만 training protocol 안에서 미리 정해야 한다.

- prior 또는 이전 모수를 유지한다.
- 질량이 큰 사례 근처로 상태를 재초기화한다.
- 상태를 병합하거나 제거하고 $K$를 다시 선택한다.
- 더 강한 covariance·transition regularization을 둔다.

평가 결과를 본 뒤 가장 좋아 보이는 처리만 고르면 모델 선택 편향이 생긴다.

### 3.14 식별성 1: 상태 번호는 본질이 아니다

상태 라벨의 임의 순열을 $\sigma$라 하자. 모수를

$$
\pi'_j=\pi_{\sigma(j)},
$$

$$
A'_{ij}
=
A_{\sigma(i),\sigma(j)},
$$

$$
b'_j(y)=b_{\sigma(j)}(y)
$$

로 동시에 바꾸면 관측열 분포는 변하지 않는다.

$$
p_{\theta'}(y_{1:T})
=
p_\theta(y_{1:T}).
$$

이것이 label switching이다. 한 실행의 `state 1`과 다음 실행의 `state 1`이 자동으로 같은 의미라는 보장은 없다. 상태 posterior를 replay quota나 운영 정책에 사용할 때 숫자 ID를 그대로 키로 쓰면 재학습 후 의미가 뒤집힐 수 있다.

실무에서는 다음처럼 **사후 정렬**할 수 있다.

- 외부의 고정된 anchor set에서 emission 통계를 비교한다.
- 상태별 latency·오류율·도메인 구성 같은 해석 지표를 함께 기록한다.
- 이전 버전 상태와 새 버전 상태 사이의 최소 비용 matching을 한다.
- 의미가 모호하면 숫자 라벨을 운영 정책에 직접 연결하지 않는다.

“평균 오류율이 작은 순서로 상태 번호를 정했다”는 것은 표시 규칙을 고정한 것이지 모형의 모든 모수가 식별됐다는 증명은 아니다.

### 3.15 식별성 2: 순열을 제외해도 구분되지 않을 수 있다

두 상태의 emission이 완전히 같고 전이구조도 관측에 서로 다른 흔적을 남기지 못하면 관측만으로 둘을 구분할 수 없다. 예를 들어

$$
b_1(y)=b_2(y)
\quad\text{for every }y
$$

이면 현재 관측 하나는 상태 $1$과 $2$를 구분하지 못한다. 충분한 장기 의존성이 있으면 전이 차이가 정보를 줄 수 있지만, 일반적으로 식별성에는 다음과 같은 모형별 조건이 필요하다.

- 올바른 상태 수 $K$
- 각 상태의 양의 점유질량
- 서로 충분히 다른 emission
- 퇴화하지 않은 전이행렬
- 충분히 긴 독립 시퀀스와 지원범위

$K$를 너무 크게 잡으면 하나의 실제 체제를 비슷한 두 상태로 쪼개거나, 한 상태가 거의 쓰이지 않는 state collapse가 생길 수 있다. 반대로 $K$가 작으면 정상 피크와 사고를 한 상태로 뭉칠 수 있다.

Gaussian mixture 성격 때문에 한 상태의 covariance가 특정 관측 주변에서 거의 $0$으로 줄며 likelihood가 비정상적으로 커지는 singularity도 가능하다. 다음을 함께 확인해야 한다.

1. 여러 초기화에서 held-out predictive likelihood가 안정적인가?
2. 상태별 유효 표본 수 $N_j$가 충분한가?
3. covariance 고윳값이 바닥에 붙지 않는가?
4. 사후 상태가 외부 결과·개입과 일관되는가?
5. 상태 수 선택을 같은 final gate 데이터에서 반복하지 않았는가?

### 3.16 관찰만으로 원인을 식별할 수는 없다

상태 $j$에서 오류율과 latency가 함께 높다고 하자. 이는 다음 원인들이 같은 latent state로 묶였을 수 있다.

- 입력 도메인 변화
- 모델 버전 배포
- 검색 시스템 장애
- 트래픽 급증
- 라벨 파이프라인 지연

HMM likelihood가 높다는 사실은 인과 원인을 밝히지 않는다. 알려진 개입 $U_t$를 전이·emission 조건에 넣거나, 무작위 canary·자연실험·원인별 관측을 추가해야 한다.

$$
p(S_t\mid S_{t-1},U_t),
\qquad
p(Y_t\mid S_t,U_t)
$$

처럼 개입을 조건에 넣지 않으면 새 배포가 만든 점프를 자연 drift 상태로 흡수할 수 있다.

### 3.17 표준 HMM의 geometric 체류시간을 유도한다

상태 $j$에 진입한 뒤 정확히 $d$블록 머문다고 하자. 처음 $d-1$번은 자기 상태에 남고, $d$번째 뒤에는 떠나야 한다. 따라서 $A_{jj}<1$일 때

$$
P(D_j=d)
=
A_{jj}^{d-1}(1-A_{jj}),
\qquad d=1,2,\ldots
$$

이다. 기대 체류시간은

$$
E[D_j]
=
\frac{1}{1-A_{jj}}.
$$

이미 $d$블록 머물렀다는 조건에서 다음 블록에도 남을 확률은 언제나

$$
P(D_j\ge d+1\mid D_j\ge d)
=
A_{jj}
$$

로 같다. duration의 나이를 기억하지 않는다.

앞의 수치 예제에서 $A_{22}=0.925$이면

$$
E[D_2]
=
\frac{1}{1-0.925}
\approx13.33
$$

블록이다. 상태를 떠났다는 조건에서 다른 상태 $k\ne j$로 갈 확률은

$$
\widetilde A_{jk}
=
\frac{A_{jk}}{1-A_{jj}}
$$

로 볼 수 있다.

### 3.18 HSMM은 duration을 별도 확률변수로 만든다

HSMM에서는 관측열을 segment 단위로 본다. $m$번째 segment의 상태를 $Z_m$, 길이를 $D_m$이라 하자.

$$
P(D_m=d\mid Z_m=j)=q_j(d).
$$

보통 연속한 같은 상태 segment가 불필요하게 둘로 쪼개지는 모호성을 피하려고 segment 전이행렬에

$$
\widetilde A_{jj}=0
$$

을 둔다. 상태 $j$가 시점 $s$부터 $t$까지 하나의 segment를 만든 emission likelihood를

$$
B_j(s:t)
=
\prod_{u=s}^{t}b_j(y_u)
$$

라고 하자. 여기서는 $q_j(d)=0$ for $d>D_{\max}$인 **hard support**를 둔다. $F_t(j)$를 “상태 $j$의 한 segment가 정확히 $t$에서 끝나고 $y_{1:t}$를 만들 결합확률”로 정의하면

$$
\begin{aligned}
F_t(j)
={}&
\sum_{d=1}^{\min(D_{\max},t)}
q_j(d)B_j(t-d+1:t)\\
&\times
\left[
\mathbf 1\{t=d\}\pi_j
+
\mathbf 1\{t>d\}
\sum_{i\ne j}
F_{t-d}(i)\widetilde A_{ij}
\right].
\end{aligned}
$$

각 후보 길이 $d$마다 “그 길이의 segment emission”과 “그 앞 segment에서 넘어온 확률”을 곱해 합한 것이다. 단순 구현의 시간복잡도는 대략

$$
O(TK^2D_{\max})
$$

이므로 $D_{\max}$가 크면 pruning·구조화된 duration·FFT류 최적화가 필요할 수 있다. $D_{\max}$를 분포의 hard cap이 아니라 계산용 truncation으로만 쓴다면 잘린 tail mass를 버리지 말고 근사오차·재정규화 규칙을 따로 명시해야 한다.

이 정의의 $F_T(j)$는 마지막 segment가 **정확히 $T$에서 완료되었다**는 경계 convention을 포함한다. 따라서 $\sum_jF_T(j)$는 sequence 종료가 segment 경계라는 모형 아래의 likelihood이지, 진행 중인 마지막 segment를 허용하는 right-censored 관측창의 likelihood가 아니다. 후자의 경우에는 아래 survival mass를 포함한 별도 terminal 항이 필요하다.

### 3.19 HSMM duration의 M-step과 검열

HSMM의 E-step이 상태 $j$에서 길이 $d$인 완료 segment의 기대 개수를 $\kappa_{jd}$로 계산했고 censoring이 없다고 하자. 범주형 duration MLE는

$$
q_j^{\mathrm{new}}(d)
=
\frac{\kappa_{jd}}
{\sum_{u=1}^{D_{\max}}\kappa_{ju}}
$$

이다. Poisson·negative binomial 같은 parametric duration을 쓰면 duration이 $1,2,\ldots$에서 시작하도록 shifted 또는 zero-truncated 분포를 정의하고, $\kappa_{jd}$로 가중한 해당 분포의 모수를 최적화한다.

관측창 끝에서 상태가 아직 지속 중이면 그 duration은 right-censored다. 예를 들어 $5$블록까지 관측했지만 종료를 못 봤다면 “길이 $5$”가 아니라

$$
D\ge5
$$

라는 정보다. 이를 완료 길이 $5$로 세면 긴 체류시간을 체계적으로 짧게 추정한다. Censored 길이가 $c$라면 duration 항은

$$
\log \overline Q_j(c),
\qquad
\overline Q_j(c)
=
\begin{cases}
\displaystyle\sum_{d=c}^{D_{\max}}q_j(d),&\text{hard cap},\\
\displaystyle\sum_{d=c}^{\infty}q_j(d),&\text{unbounded duration}
\end{cases}
$$

를 기여한다. 그러므로 censoring이 있으면 위의 단순한 normalized-$\kappa$ 식이 일반적으로 M-step의 완전한 해가 아니며, survival likelihood를 포함한 censoring-aware forward–backward와 최적화가 필요하다.

### 3.20 Duration pmf와 hazard는 서로 변환할 수 있다

양의 정수 duration $D$에 대해

$$
h(d)
=
P(D=d\mid D\ge d)
=
\frac{q(d)}{\sum_{u=d}^{\infty}q(u)}
$$

를 discrete hazard라고 한다. 반대로

$$
q(d)
=
h(d)
\prod_{u=1}^{d-1}(1-h(u)).
$$

상수 hazard $h(d)=h$이면

$$
q(d)=h(1-h)^{d-1}
$$

인 geometric 분포다. HSMM은 $h(d)$가 duration 나이에 따라 달라지도록 허용한다.

### 3.21 BOCPD의 상태는 “마지막 변화 이후 길이”다

여기서는 **새 segment가 $Y_t$ 직전에 시작되면 $r_t=0$**이라는 convention을 사용한다. 다른 문헌은 인덱스를 한 칸 다르게 둘 수 있으므로 구현과 수식을 섞기 전에 convention을 확인해야 한다.

정규화 전 결합확률을

$$
J_t(r)
=
p(r_t=r,y_{1:t})
$$

라고 하자. 이 convention에서는 $r_{t-1}=r$인 현재 segment에 이미 $r+1$개 관측이 들어 있다. 그 충분통계로 계산한 다음 관측의 posterior predictive를 $p^{(r+1)}(y_t)$라고 쓰자. 직전 segment가 $r+1$개 관측 뒤에 끝날 hazard를 $H(r+1)$이라 두면, $t\ge2$에서 두 경로가 있다. 초기조건은

$$
J_1(0)=p_0(y_1)
$$

이다.

**변화가 나서 reset:**

$$
J_t(0)
=
p_0(y_t)
\sum_{r\ge0}
H(r+1)J_{t-1}(r).
$$

**변화가 없어 growth:**

$$
J_t(r+1)
=
p^{(r+1)}(y_t)
\left[1-H(r+1)\right]
J_{t-1}(r).
$$

$p_0(y_t)$는 새 segment의 prior predictive이고, $p^{(r+1)}(y_t)$는 현재 run의 $r+1$개 관측으로 계산한 posterior predictive다. 마지막에

$$
p(r_t=r\mid y_{1:t})
=
\frac{J_t(r)}{\sum_{u\ge0}J_t(u)}
$$

로 정규화한다.

상수 hazard $H(r)=h$는 변화 전 duration에 geometric prior를 둔다. 특정 길이 부근에서 끝날 가능성이 높다면 nonconstant hazard를 사용하되, 운영 결과를 본 뒤 임의로 조정하지 말고 별도 validation에서 검량해야 한다.

### 3.22 Beta–Bernoulli BOCPD 예제

한 segment 안의 이진 경보율을

$$
\vartheta\sim\operatorname{Beta}(a_0,b_0),
\qquad
Y_t\mid\vartheta\sim\operatorname{Bernoulli}(\vartheta)
$$

로 두자. 현재 run 안에 $m$개 관측이 있고 그중 성공이 $s_m$개, 실패가 $f_m$개라면 $s_m+f_m=m$이고 posterior는

$$
\vartheta\mid\mathcal D_m
\sim
\operatorname{Beta}(a_0+s_m,b_0+f_m)
$$

이고 다음 경보의 predictive probability는

$$
p^{(m)}(Y_t=1)
=
\frac{a_0+s_m}
{a_0+b_0+s_m+f_m}.
$$

새 segment의 prior predictive는

$$
p_0(Y_t=1)
=
\frac{a_0}{a_0+b_0}.
$$

예를 들어 $a_0=b_0=1$이고 현재 run의 최근 $4$개가 $(0,0,0,0)$이었다면

$$
p^{(4)}(Y_t=1)=\frac{1}{6}.
$$

이때 직전 시점의 run length는 $r_{t-1}=3$이다. 갑자기 $Y_t=1$이 오면 기존 run에서도 불가능하지는 않지만, 새 segment prior 아래 확률 $1/2$가 더 크다. 이 likelihood 비가 reset 가지의 posterior 질량을 올린다. 경보 하나가 변화확률을 올려도 변화가 확정되는 것은 아니다.

### 3.23 HMM·HSMM·BOCPD는 서로 대체어가 아니다

| 질문 | HMM | HSMM | BOCPD |
| --- | --- | --- | --- |
| 핵심 latent variable | 유한 상태 $S_t$ | 상태와 explicit duration | run length $r_t$ |
| duration 기본값 | geometric | 임의·parametric duration | hazard가 만드는 segment prior |
| 반복 상태 인식 | 자연스럽다 | 자연스럽다 | 기본형은 상태 이름보다 변화 시점에 초점 |
| 온라인 출력 | 상태 filtering posterior | 상태·남은 duration posterior | 마지막 변화 시점 posterior |
| 대표 질문 | “현재 어느 체제인가?” | “어느 체제이며 얼마나 오래 지속되는가?” | “마지막 변화가 언제였는가?” |
| 주의점 | label switching·geometric 가정 | 계산량·censoring·segment 정의 | hazard 민감도·새 segment 의미 |

BOCPD의 높은 reset posterior는 `HOLD`나 조사 우선순위를 줄 수 있지만, 새 모델을 학습하거나 배포해도 된다는 증거는 아니다.

### 3.24 Reservoir sampling의 균등 포함확률을 증명한다

크기 $M$인 buffer에 스트림 항목이 차례로 온다고 하자.

1. 처음 $M$개는 모두 저장한다.
2. $n>M$번째 항목은 확률 $M/n$로 받아들인다.
3. 받아들이면 $M$개 슬롯 중 하나를 균등하게 골라 교체한다.

귀납의 base case $n=M$에서는 처음 $M$개 항목이 모두 들어 있으므로 각 항목에 대해

$$
P(i\in\mathcal B_M)=1=\frac{M}{M}
$$

이다.

새 항목 $n$의 포함확률은 정의상

$$
P(n\in\mathcal B_n)=\frac{M}{n}
$$

이다. 기존 항목 $i<n$가 $n-1$시점에 포함될 확률이 $M/(n-1)$이라고 가정하자. 새 항목이 들어오고 정확히 그 슬롯을 교체할 확률은

$$
\frac{M}{n}\cdot\frac{1}{M}
=
\frac{1}{n}
$$

이므로 기존 항목의 한 단계 생존확률은

$$
1-\frac{1}{n}
=
\frac{n-1}{n}.
$$

따라서

$$
P(i\in\mathcal B_n)
=
\frac{M}{n-1}
\cdot
\frac{n-1}{n}
=
\frac{M}{n}.
$$

귀납적으로 지금까지 본 모든 항목이 같은 포함확률 $M/n$을 가진다.

이 보장이 성립하는 단위가 무엇인지 중요하다. 한 긴 대화의 token $10{,}000$개를 각각 항목으로 세면 그 대화가 buffer를 지배할 수 있다. 사용자·세션·사건 같은 **독립에 가까운 source block**을 먼저 정의해야 한다.

### 3.25 층화 replay와 포함확률

각 항목이 정확히 하나에 속하는 **상호배타적이고 고정된** 그룹 $g=1,\ldots,G$를 사전 등록하고, 관찰 구간 동안 바뀌지 않는 buffer 용량 $M_g$를 주어

$$
\sum_{g=1}^GM_g=M
$$

이라 하자. 그룹 $g$에서 지금까지 $N_g$개를 봤고 내부에서 reservoir를 썼다면 항목의 포함확률은

$$
\rho_i
=
\min\left(1,\frac{M_g}{N_g}\right),
\qquad i\in g
$$

이다. 그룹이 겹치거나 항목을 본 뒤 경계·용량을 바꾸면 이 단순 포함확률은 그대로 성립하지 않으므로 실제 선택정책의 propensity를 다시 계산해야 한다. 희귀 안전군에 큰 $M_g/N_g$를 주면 buffer 안 분포는 원래 모집단과 달라진다.

목표가 원래 모집단의 평균 손실

$$
R(\theta)=E_{Z\sim P}[\ell(\theta;Z)]
$$

이라면 $P$가 $Q$에 대해 절대연속, 즉 $p(z)>0$인 모든 곳에서 $q(z)>0$이고 아래 기대값이 존재한다는 조건에서 표집분포 $Q$ 아래

$$
R(\theta)
=
E_{Z\sim Q}
\left[
\frac{p(Z)}{q(Z)}\ell(\theta;Z)
\right]
$$

처럼 재가중할 수 있다. 그러나 작은 $q(Z)$는 큰 가중치를 만들어 분산을 폭발시킨다. Propensity floor와 clipping이 필요할 수 있다. Clipping은 분산을 줄이는 대신 일반적으로 bias를 만들거나 estimand를 바꾼다.

이 항등식은 **무작위 선택정책에 대한 기대값**의 보정이다. 이미 고정된 하나의 buffer나 결정론적 coreset에서 어떤 영역이 빠져 $q(z)=0$이 되었다면, 사후 가중치만으로 그 누락 영역을 복구할 수 없다. 유한 모집단의 불균등 무복원 표집에서는 단순 density ratio 대신 실제 inclusion probability를 쓰는 Horvitz–Thompson류 설계가 필요할 수도 있다.

목표가 그룹별 최악 위험이라면 평균분포 복원이 아니라

$$
\min_\theta\max_{g\in\mathcal G}R_g(\theta)
$$

같은 목적을 명시해야 한다. “quota를 썼으니 공정하다”는 말만으로는 목표가 정해지지 않는다.

### 3.26 거리 기반 coreset은 공간을 덮으려 한다

각 사례의 표현을 $\phi(z_i)$라 하자. $k$-center형 coreset은

$$
S^\star
\in
\arg\min_{S:\,|S|\le M}
\max_i\min_{j\in S}
d(\phi(z_i),\phi(z_j))
$$

를 근사한다. 모든 사례가 선택된 대표 중 하나와 가깝도록 최대 덮개 반경을 줄인다. Facility-location형 목적은

$$
\max_{S:\,|S|\le M}
\sum_i\max_{j\in S}
\operatorname{sim}(\phi(z_i),\phi(z_j))
$$

처럼 전체가 대표와 높은 유사도를 갖게 한다.

하지만 coreset에는 세 가지 함정이 있다.

1. 표현 $\phi$가 틀리면 중요한 차이를 같은 cluster로 뭉친다.
2. 결정론적 선택은 일반적으로 모집단 평균의 불편표본이 아니다.
3. 현재 모델에 좋은 표현이 새 모델에서도 좋은 replay 좌표라는 보장은 없다.

따라서 coreset만 쓰기보다 uniform reservoir와 고위험 quota를 함께 두고, 선택 버전과 embedding 모델을 기록한다.

### 3.27 Gradient diversity는 서로 다른 학습 방향을 보존한다

사례 $z_i$의 gradient를

$$
g_i
=
\nabla_\theta\ell(\theta;z_i)
$$

라고 하자. 전체 과거 사례 수가 $N$일 때

$$
\overline g_{\mathrm{old}}
=
\frac{1}{N}\sum_{i=1}^N g_i
$$

를 작은 subset의 convex combination으로 근사하는 한 가지 목적은

$$
\min_{\substack{S:\,|S|\le M,\;w_i\ge0\\
\sum_{i\in S}w_i=1}}
\left\|
\overline g_{\mathrm{old}}
-
\sum_{i\in S}w_ig_i
\right\|_2^2
$$

처럼 쓸 수 있다. 여기서 $|S|\le M$이 실제 memory budget이고, $\sum_{i\in S}w_i=1$이 평균의 scale을 맞춘다. 사례가 대표하는 cluster 크기를 따로 반영하는 설계라면 다른 normalization도 가능하지만 반드시 명시해야 한다. 서로 같은 방향의 gradient만 고르지 않도록 정규화한

$$
u_i
=
\frac{g_i}{\|g_i\|_2+\varepsilon}
$$

를 모아 log-determinant 목적

$$
\max_{S:\,|S|\le M}
\log\det
\left(
I+\lambda U_SU_S^\top
\right)
$$

을 사용할 수도 있다. 서로 독립적인 방향이 많을수록 Gram matrix가 더 큰 부피를 갖는다.

새 데이터 gradient $g_{\mathrm{new}}$와 과거 사례 gradient의 내적이

$$
g_{\mathrm{new}}^\top g_i<0
$$

이면 두 손실의 1차 변화 방향이 충돌한다는 신호다. 그러나 충돌 사례만 저장하면 과거분포를 왜곡하고 쉬운 대표 사례를 잃는다. 또한 전체 파라미터 gradient는 너무 크므로 마지막 층, adapter, random sketch 같은 근사를 쓰며 근사 오차를 기록해야 한다.

Gradient는 모델 버전과 파라미터 위치에 의존한다. 어제의 $g_i$가 오늘 checkpoint에서도 같은 의미라는 보장은 없다.

### 3.28 Replay를 섞은 학습 목적함수

새 데이터 분포를 $P_{\mathrm{new}}$, buffer 표집분포를 $Q_{\mathcal B}$라 하자. 하나의 일반적인 목적은

$$
\begin{aligned}
\mathcal L_t(\theta)
={}&
\lambda
E_{Z\sim P_{\mathrm{new}}}
[\ell(\theta;Z)]\\
&+
(1-\lambda)
E_{Z\sim Q_{\mathcal B}}
[w(Z)\ell(\theta;Z)]\\
&+
\lambda_{\mathrm{reg}}
\Omega(\theta,\theta_{t-1})
\end{aligned}
$$

이다. 여기서

- $w(Z)$는 목표에 따라 표집 편향을 보정하는 가중치다.
- $\Omega$는 EWC·distillation·파라미터 이동 벌점일 수 있다.
- $\lambda$는 안정성–가소성 trade-off다.

이 목적의 training loss가 내려갔다고 배포가 승인되는 것은 아니다. Buffer는 학습을 위한 선택된 자료라 평가 모집단의 독립 표본이 아닐 수 있다.

### 3.29 고정 시점 Hoeffding 구간을 매일 반복하면 생기는 문제

동결된 challenger와 champion을 같은 평가 단위에서 비교한 차이를

$$
Z_s
=
\operatorname{score}_{\mathrm{challenger},s}
-
\operatorname{score}_{\mathrm{champion},s}
$$

라 하자. $Z_s\in[a,b]$이고 범위 폭을

$$
R=b-a
$$

라고 하자. 고정된 표본크기 $n$에서 독립·동일분포를 가정한 Hoeffding bound는 대략

$$
P\left(
|\overline Z_n-\mu|\ge\epsilon
\right)
\le
2\exp\left(
-\frac{2n\epsilon^2}{R^2}
\right)
$$

이다. 그러나 $n=100,200,300,\ldots$마다 새 $95\%$ 구간을 만들고 처음 성공할 때 멈추면 각 시점 오류가 누적된다. 한 번의 $n$에 대한 보장이 “모든 $n$ 중 내가 고른 시점”의 보장으로 자동 확장되지 않는다.

### 3.30 오류예산을 시간에 나누어 교육용 confidence sequence를 만든다

각 시점에 쓸 오류예산을

$$
\delta_n
=
\frac{6\delta}{\pi^2n^2}
$$

로 두면 Basel 합 때문에

$$
\sum_{n=1}^{\infty}\delta_n
=
\delta.
$$

각 $n$에서 Hoeffding bound를 $\delta_n$ 수준으로 적용하고 union bound를 쓰면, 확률 $1-\delta$ 이상으로 **모든 $n$에 동시에**

$$
|\overline Z_n-\mu|
\le
R
\sqrt{
\frac{\log(2/\delta_n)}{2n}
}
$$

가 성립한다. 즉

$$
w_n
=
R
\sqrt{
\frac{1}{2n}
\log\left(
\frac{\pi^2n^2}{3\delta}
\right)
}
$$

로 두고

$$
\operatorname{LCB}_n
=
\overline Z_n-w_n,
\qquad
\operatorname{UCB}_n
=
\overline Z_n+w_n
$$

를 사용하면 반복 확인에 유효하다.

$Z_s\in[-1,1]$이면 $R=2$이므로 같은 폭은

$$
w_n
=
\sqrt{
\frac{2}{n}
\log\left(
\frac{\pi^2n^2}{3\delta}
\right)
}.
$$

이 방법은 이해하기 쉽지만 꽤 보수적이다. 실무에서는 variance-adaptive empirical Bernstein confidence sequence나 mixture-martingale boundary가 더 좁을 수 있다. 어느 방법이든 가정과 코드 버전을 미리 고정해야 한다.

독립·동일분포보다 일반적인 순차 설정에서는

$$
m_s
=
E[Z_s\mid\mathcal F_{s-1}]
$$

를 두고 누적 조건부 평균

$$
\overline m_n
=
\frac1n\sum_{s=1}^nm_s
$$

을 덮는 martingale형 confidence sequence를 사용한다. 이는 선택된 평가 스트림의 누적 조건부 평균에 대한 보장이지, 분포가 계속 바뀌는 상황의 “현재 순간 성능”을 자동으로 보장하지 않는다.

### 3.31 숫자로 보는 confidence sequence의 보수성

정확도 차이처럼 $Z_s\in[-1,1]$, 총 오류예산 $\delta=0.05$, $n=10{,}000$이라고 하자.

$$
w_{10{,}000}
=
\sqrt{
\frac{2}{10{,}000}
\log\left(
\frac{\pi^2(10{,}000)^2}{0.15}
\right)
}
\approx0.067.
$$

관측된 평균 개선이

$$
\overline Z_{10{,}000}=0.08
$$

이면

$$
\operatorname{LCB}
\approx0.08-0.067
=
0.013>0.
$$

이 교육용 경계 아래서는 평균 개선이 양수라는 순차 증거가 생긴다. 그러나 그룹 $20$개, 후보 $5$개, metric $3$개에 오류예산을 다시 나누면 각 경계는 더 넓어진다. 대규모 표본이 필요한 이유다.

### 3.32 승인 gate는 여러 부등식의 교집합이다

그룹 $g$에서 효용 차이를

$$
\Delta^U_g
=
U_g(\text{challenger})
-
U_g(\text{champion})
$$

라고 하고, 비용은 클수록 나쁘다고 하여

$$
\Delta^C_g
=
C_g(\text{challenger})
-
C_g(\text{champion})
$$

라고 하자. 사전 등록한 최소 개선 $m_g$와 허용 비용증가 $c_g$에 대해 승인은 적어도

$$
\operatorname{LCB}_{g,n}(\Delta^U_g)
\ge
m_g
$$

및

$$
\operatorname{UCB}_{g,n}(\Delta^C_g)
\le
c_g
$$

를 모든 핵심 그룹에서 만족해야 한다. 여기에 다음 결정적 조건도 붙는다.

- 그룹별 최소 표본 수와 ESS
- latency·비용·메모리 상한
- hard-stop 사건 $0$건 또는 사전 정의한 희귀사건 상한
- OOD·새 언어·장문 회귀시험
- 데이터 권리와 개인정보 삭제 반영
- calibration·threshold 재검량
- shadow와 canary 중지 규칙

오류예산은 그룹·metric·후보·새 test family에 나뉜다.

$$
\sum_{g,m,c,f}
\delta_{g,m,c,f}
\le
\delta_{\mathrm{total}}.
$$

여기서 $f$는 사전 등록된 하나의 test family 또는 protocol revision이다. **같은 confidence sequence 안에서 시간 $n$만 늘려 반복 확인하는 look은 이미 simultaneous coverage로 보호되므로 look마다 오류예산을 다시 나누지 않는다.** 반면 모델·estimand·데이터 포함규칙을 바꾸거나 검정을 리셋해 새 family를 열면 ledger에 새 항목으로 기록해야 한다. 실패한 후보를 버리고 같은 데이터에서 조금 바꾼 후보를 계속 시험하면서 새 오류예산인 것처럼 행동하면 안 된다.

### 3.33 순차 구간이 보호하지 않는 것

Confidence sequence는 “반복해서 본다”는 문제를 해결한다. 다음 문제는 별도다.

1. **후보 선택:** 같은 gate 데이터로 여러 후보를 고른 뒤 승자만 보고하면 편향된다.
2. **그룹 선택:** 결과를 본 뒤 나쁜 그룹을 합치거나 제외하면 simultaneous coverage가 깨진다.
3. **학습–평가 누수:** replay·early stopping에 사용한 항목은 독립 final gate가 아니다.
4. **비정상성:** 사용자 구성과 정책이 바뀌면 누적 평균의 estimand가 현재 배포 목표와 다를 수 있다.
5. **지연 라벨:** 아직 결과가 나온 쉬운 사례만 먼저 보면 informative missingness가 생긴다.
6. **가중치 폭주:** importance-weighted 차이가 bounded하지 않으면 위 Hoeffding 폭을 그대로 쓸 수 없다.

따라서 `탐지 → adaptation → calibration → final sequential gate`의 source block을 분리하고, 지연 결과는 event time·reveal time을 함께 기록해야 한다.

### 3.34 오늘 수학과 AI 부품의 1:1 대응

| 수학 부품 | AI 시스템에서 하는 일 | 금지할 해석 |
| --- | --- | --- |
| $\gamma_t(j)$ | emission 학습의 소프트 가중치 | 진짜 상태 라벨 또는 배포 허가 |
| $\xi_t(i,j)$ | 전이행렬의 기대 count | 관측된 실제 전이 로그 |
| EM의 단조성 | training likelihood가 내려가지 않게 함 | 전역 최적·인과 해석·일반화 보장 |
| label switching | 상태 ID의 버전 간 정렬 필요성 | `state 2`의 영구적 의미 |
| HSMM duration | 점검·장애의 비-geometric 체류 모델링 | 관측창 끝 segment를 완료 duration으로 취급 |
| BOCPD run length | 마지막 변화 시점의 온라인 불확실성 | 새 체제의 정체 또는 재학습 승인 |
| reservoir $\rho_i$ | 모든 과거 source block의 균등 기억 기회 | 희귀 안전군 최소 보장 |
| coreset 거리 | 표현공간 coverage | 모집단의 확률표본 |
| gradient diversity | 서로 다른 학습 방향 보존 | 미래 checkpoint에서도 고정된 중요도 |
| importance weight | 표집과 목표분포 차이 보정 | 작은 propensity에서도 안정적 추정 |
| confidence sequence | 반복 관찰 가능한 승인 경계 | 데이터 누수·후보 선택·drift의 자동 해결 |
| alpha ledger | 여러 그룹·후보·재시작의 총 오류율 관리 | 새 이름을 붙인 무제한 재시험 |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 오늘의 AI 문제: 작은 기억으로 배우되, 기억을 증거로 착각하지 않는다

지속학습 시스템에는 최소 네 종류의 데이터 역할이 있다.

| 데이터 역할 | 목적 | 모델 선택에 사용? | 최종 승인에 사용? |
| --- | --- | --- | --- |
| 탐지 stream | 변화·사고 의심을 빠르게 찾음 | 직접 사용하지 않음 | 직접 사용하지 않음 |
| adaptation set | 새 체제를 학습함 | 학습 자체에 사용 | 불가 |
| replay buffer | 과거 능력을 보존함 | 학습 자체에 사용 | 불가 |
| calibration set | 확률·threshold·conformal cutoff를 다시 맞춤 | 제한적으로 사용 | final gate와 분리 |
| final gate stream | champion과 challenger를 독립 비교 | 사전 등록한 중지·승인만 | 가능 |

Replay buffer에서 잘하는 모델을 고르고 같은 buffer 점수로 배포하면 과적합을 측정하지 못한다. Buffer는 학습과 후보 설계를 위해 적극적으로 선택된 자료이기 때문이다.

### 4.2 세 개의 상태를 구분한다

시스템 설계에서 다음 세 상태를 같은 변수로 만들지 않는다.

1. **환경 상태:** HMM·HSMM이 추론하는 정상·피크·장애 같은 latent regime
2. **변화 상태:** BOCPD가 추론하는 마지막 변화 이후 run length
3. **모델 생명주기 상태:** `CHAMPION`, `TRAINING`, `SHADOW`, `CANARY`, `ROLLED_BACK`

예를 들어 BOCPD의 reset posterior가 높아지면 운영 상태기계는 `INVESTIGATE`나 `HOLD`로 갈 수 있다. 그러나 challenger를 자동 생성하고 즉시 `CANARY`로 보내면 탐지 오경보가 곧 모델 변경으로 증폭된다.

$$
\text{BOCPD alert}
\longrightarrow
\text{HOLD·원인 조사}
\longrightarrow
\text{데이터 성숙·권리 검증}
\longrightarrow
\text{별도 adaptation}
$$

이 순서를 지킨다.

### 4.3 제한된 replay budget을 역할별로 나눈다

총 budget이 $M$이면 하나의 실용적 설계는 다음과 같다.

$$
M
=
M_{\mathrm{uniform}}
+
M_{\mathrm{safety}}
+
M_{\mathrm{coverage}}
+
M_{\mathrm{recent}}.
$$

- $M_{\mathrm{uniform}}$: source-block reservoir로 전체 역사에 균등 기회를 준다.
- $M_{\mathrm{safety}}$: 사전 등록한 희귀 안전군·법적 의무군의 최소 floor다.
- $M_{\mathrm{coverage}}$: embedding coreset 또는 gradient sketch로 다양한 모드를 보존한다.
- $M_{\mathrm{recent}}$: 최근 변화에 빠르게 적응하는 짧은 window다.

이는 유일한 정답이 아니다. 중요한 것은 각 영역의 목적과 포함규칙을 분리하고 버전으로 남기는 것이다. 한 사례가 여러 영역에 중복 저장되면 실제 학습 확률을 기록하고 deduplication 또는 가중치 조정을 해야 한다.

### 4.4 Buffer의 항목 단위부터 설계한다

무엇을 한 항목으로 볼지에 따라 replay 분포가 달라진다.

- 언어 모델: token 하나가 아니라 문서·대화·trajectory 단위
- VLM: crop 하나가 아니라 원본 이미지·문서 페이지·질의 묶음 단위
- 에이전트: 개별 tool call보다 episode 또는 decision state 단위
- 사용자 서비스: 요청 하나보다 사용자·세션·사건 cluster 단위

같은 사용자의 비슷한 요청 $1{,}000$개를 독립 항목으로 저장하면 그 사용자가 buffer와 평가분산을 지배한다. Source ID, near-duplicate hash, consent·retention metadata를 항목에 붙인다.

권장 manifest의 최소 필드는 다음과 같다.

```text
sample_id
source_block_id
event_time
reveal_time
group_tags
data_rights_version
model_version_at_collection
selection_channel
inclusion_probability
embedding_version
gradient_sketch_version
label_version
deletion_tombstone
```

### 4.5 Reservoir는 훌륭한 기준선이지만 완전한 정책은 아니다

Reservoir의 장점은 단순하고, 스트림 길이를 미리 몰라도 과거 각 항목에 같은 포함확률을 준다는 것이다. 시간이 지난 사례를 무조건 버리는 FIFO와 달리 오래된 체제도 확률적으로 남는다.

하지만 다음은 보장하지 않는다.

- 희귀 안전군이 최소 한 개 존재한다.
- 최근 체제가 충분히 반영된다.
- 중복 문서가 제거된다.
- 학습에 유용한 경계 사례가 남는다.
- 개인정보 보존기간과 삭제 요청이 자동 반영된다.

따라서 reservoir를 “공정한 랜덤 기억”으로만 사용하고, 안전 floor·recency·coverage를 별도 정책으로 둔다.

### 4.6 희귀 안전군 floor는 latent state ID가 아니라 안정된 정의를 쓴다

`HMM state 3`에 buffer $100$개를 보장한다고 하자. 다음 재학습에서 label switching이 일어나 state 3이 정상 상태가 되면 quota 의미가 뒤집힌다.

안전 quota는 가능한 한 다음처럼 관측 가능하고 버전이 안정된 조건으로 정의한다.

- 정책 위반 유형
- 언어·지역·접근성 그룹
- 인간이 확정한 incident taxonomy
- 계약상 보호해야 할 고객군
- 외부 benchmark task family

Latent state는 quota 배분의 보조 신호로 사용할 수 있지만, anchor set과 matching으로 상태 의미를 정렬하고 불확실성을 기록해야 한다.

### 4.7 Coreset은 무엇이 서로 다른지 정의해야 한다

Embedding coreset은 거리가 의미 있을 때만 유효하다. 같은 문장이라도 다음 목적의 embedding은 다르다.

- 의미 유사도
- 정책 위험도
- 시각 레이아웃
- 모델 오류 패턴
- tool trajectory 구조

일반 semantic embedding으로 안전상 중요한 미묘한 차이를 가까운 점으로 뭉칠 수 있다. 반대로 노이즈가 큰 embedding의 극단점을 뽑으면 outlier만 가득 찬 buffer가 된다.

권장 절차는 다음과 같다.

1. 목적에 맞는 고정 feature extractor를 선택한다.
2. source block 단위로 중복을 제거한다.
3. 그룹별 최소 floor를 먼저 배정한다.
4. 남은 후보에서 reservoir와 coverage를 혼합한다.
5. embedding 버전이 바뀌면 전체를 즉시 재선별하지 말고 shadow 비교한다.
6. coreset 자체를 final evaluation set으로 사용하지 않는다.

### 4.8 Gradient diversity는 “서로 다른 실수”를 기억하려는 시도다

Embedding이 입력의 겉모습을 본다면 gradient는 현재 모델의 손실 관점에서 사례가 어떤 방향으로 가중치를 움직이는지 본다.

- 같은 의미의 두 사례라도 모델이 하나는 맞히고 하나는 틀리면 gradient가 다를 수 있다.
- 서로 다른 입력이라도 같은 파라미터 방향을 요구하면 gradient가 비슷할 수 있다.
- 새 데이터와 음의 내적을 가진 과거 사례는 망각 위험을 보여 줄 수 있다.

그러나 full gradient를 모든 사례에 저장하는 것은 불가능에 가깝다. 다음 근사를 쓴다.

- 마지막 linear layer gradient
- LoRA·adapter parameter gradient
- random projection sketch
- class prototype gradient
- 여러 mini-batch의 평균 gradient

Gradient diversity score는 모델 checkpoint에 종속된다. `gradient_version = model_hash + loss_version + sketch_seed`처럼 provenance를 남긴다.

### 4.9 Replay sampling과 loss weighting을 분리한다

Buffer에 들어오는 규칙과 학습 mini-batch에서 뽑히는 규칙은 다를 수 있다.

- **Inclusion:** 장기 기억에 무엇을 남길까?
- **Sampling:** 이번 optimizer step에 무엇을 보여 줄까?
- **Weighting:** 그 사례의 loss에 얼마를 곱할까?

희귀군을 buffer와 mini-batch 모두에서 과표집하면서 loss weight도 다시 크게 주면 의도보다 훨씬 큰 영향을 줄 수 있다. 반대로 전체 평균을 목표로 하면서 과표집을 재가중하지 않으면 estimand가 달라진다.

한 batch의 replay loss를

$$
\widehat L_{\mathrm{replay}}
=
\frac1B
\sum_{i\in\mathcal M}
w_i\ell(\theta;z_i)
$$

로 쓰자. 여기서 $\mathcal M$은 이번 optimizer step의 replay mini-batch이고 $|\mathcal M|=B$다. 이때 $w_i$가 다음 중 무엇인지 문서화한다.

- 모집단 평균 복원을 위한 importance weight
- 그룹균형 목적의 설계 weight
- 안전사례 우선순위 weight
- 단순 heuristic hardness weight

이 네 가지를 모두 “sample weight”라고만 부르면 재현과 해석이 불가능하다.

### 4.10 BOCPD와 HSMM은 buffer router가 아니라 신호 제공자다

변화 posterior가 높아지면 최근 영역 $M_{\mathrm{recent}}$의 후보를 늘리거나 조사 queue를 열 수 있다. 하지만 다음 자동화는 위험하다.

```text
if change_probability > 0.9:
    overwrite_champion_with_online_finetune()
```

대신 안전한 인터페이스는 다음과 같다.

```text
if change_probability > alert_threshold:
    freeze_raw_window()
    mark_system_state("HOLD")
    open_investigation()
    wait_for_mature_labels()
    audit_source_and_rights()
    create_candidate_dataset_version()
```

탐지 모델이 학습 데이터 수집에 영향을 주면 selection mechanism 자체도 기록한다. BOCPD 경보 뒤 수집된 표본은 평상시 표본과 다른 propensity를 가진다.

### 4.11 Challenger 학습은 champion을 덮어쓰지 않는다

안전한 지속학습은 새 checkpoint를 별도 challenger로 만든다.

$$
\theta_{\mathrm{challenger}}
=
\operatorname{Train}
\left(
\theta_{\mathrm{champion}},
\mathcal D_{\mathrm{adapt}},
\mathcal B_{\mathrm{replay}},
\text{config hash}
\right).
$$

학습 뒤 다음을 immutable manifest로 묶는다.

- base·adapter·optimizer checkpoint hash
- tokenizer·processor·prompt template
- replay snapshot과 inclusion metadata
- data deletion·license snapshot
- training seed·library·kernel version
- HMM/HSMM/BOCPD 버전
- calibration·threshold 버전
- retriever·tool·runtime 버전

Champion은 새 모델이 final gate와 canary를 통과하기 전까지 그대로 유지한다.

### 4.12 독립 평가에서는 같은 단위를 paired 비교한다

동일한 요청·문서·episode를 두 모델에 모두 실행하면 paired difference를 만들 수 있다.

$$
Z_i
=
u_i(\text{challenger})
-
u_i(\text{champion}).
$$

쉬운 사례와 어려운 사례가 두 모델에 공통으로 작용하므로 독립적인 두 평균을 비교하는 것보다 분산이 줄 수 있다. 단, 다음을 지킨다.

- 변경 대상이 아닌 tool·timeout·외생 조건은 두 모델에서 같게 한다. Retriever처럼 바꾸려는 component는 treatment로 두고 그 버전과 설정을 manifest에 명시한다.
- 실행 순서를 무작위화하거나 공통 외부 변동을 block한다.
- stochastic decode는 seed와 반복 단위를 사전 정의한다.
- 같은 사용자의 여러 요청을 독립 표본처럼 과대 계수하지 않는다.
- 후보가 실패한 사례만 사람이 재라벨하지 않는다.

### 4.13 Confidence sequence 승인기는 이중 gate를 사용한다

효용은 클수록 좋고 안전비용은 작을수록 좋다고 하자.

```text
utility_pass = utility_lcb >= minimum_improvement
safety_pass  = safety_cost_ucb <= maximum_allowed_increase
coverage_pass = all_registered_groups_have_enough_ess
ops_pass = latency_and_cost_within_budget
hard_stop_pass = no_predefined_critical_event

approve = (
    utility_pass
    and safety_pass
    and coverage_pass
    and ops_pass
    and hard_stop_pass
)
```

전체 평균 효용이 좋아도 희귀 안전군 UCB가 기준을 넘으면 승인하지 않는다. 반대로 안전비용이 괜찮아도 효용 LCB가 최소 개선 아래라면 더 기다리거나 기각한다.

### 4.14 Alpha ledger는 실험의 회계장부다

총 허용 거짓 승인 확률이 $\delta_{\mathrm{total}}=0.05$라고 하자. 다음 항목에 예산을 미리 나눈다.

- 전체 효용
- 전체 안전비용
- 핵심 그룹별 회귀
- latency·비용
- 후보 여러 개
- 재시작 또는 새 calibration epoch

예를 들어 후보 $2$개와 핵심 그룹 $5$개에서 효용·비용 두 metric을 단순 Bonferroni로 모두 보면 최소 $20$개 family member가 생긴다. 균등 배분은 각각

$$
\delta_{g,m,c}
=
\frac{0.05}{20}
=
0.0025
$$

가 된다. 이는 보수적이지만 “얼마나 많은 선택을 했는지”를 숨기지 않는다. 상관을 이용하는 더 강한 방법을 쓰더라도 사전 등록과 구현 검증이 필요하다.

새 후보 이름을 붙이거나 날짜를 바꿨다고 오류예산이 자동 초기화되지 않는다. 같은 gate 데이터를 다시 보는 한 family가 이어진다.

### 4.15 지연 라벨은 성숙 cohort로 평가한다

환불·사고·장기 성공 같은 결과는 즉시 나오지 않는다. 오늘까지 reveal된 사례만 무심코 사용하면 결과가 빨리 나오는 쉬운 사례에 편향될 수 있다.

다음 두 시간을 구분한다.

$$
T_i^{\mathrm{event}}
=
\text{예측·행동이 일어난 시각},
$$

$$
T_i^{\mathrm{reveal}}
=
\text{정답을 알게 된 시각}.
$$

평가 cohort는 예를 들어

$$
T_i^{\mathrm{event}}
\le
t-L_{\max}
$$

처럼 최대 성숙 지연을 지난 사건만 사용하거나, censoring·delay mechanism을 명시적으로 모델링한다. `라벨 없음`을 `실패 없음`으로 코딩하지 않는다.

### 4.16 변화 뒤 calibration도 다시 확인한다

Challenger의 가중치가 바뀌면 같은 logit이 같은 실제 확률을 뜻하지 않을 수 있다. Retriever·prompt·tool policy만 바뀌어도 end-to-end score 분포가 달라질 수 있다.

다시 확인할 대상은 다음과 같다.

- 확률 calibration과 reliability curve
- 분류 threshold
- conformal score·quantile
- HMM emission과 transition
- BOCPD predictive model과 hazard
- OOD cutoff
- 비용·latency SLO

이 모든 것을 같은 작은 데이터 block에서 맞추고 평가하면 과적합된다. 적어도 adaptation, calibration, final gate의 역할을 분리한다.

### 4.17 Shadow와 canary는 통계 gate 뒤의 시스템 시험이다

오프라인 confidence sequence를 통과해도 실제 시스템에는 평가 데이터에 없던 상호작용이 있다.

- cache·batching·rate limit
- tool timeout과 retry
- retrieval index freshness
- 사용자 반응에 따른 feedback loop
- 모델 간 traffic interference
- 보안·privacy 경계

**Shadow**에서는 challenger가 실제 요청을 읽고 출력을 만들되 행동에는 반영하지 않는다. **Canary**에서는 작은 무작위 트래픽에 제한적으로 노출한다. Canary도 반복 모니터링이면 anytime-valid rule과 사전 중지조건이 필요하다.

### 4.18 Rollback은 모델 파일 하나를 되돌리는 일이 아니다

다음 묶음을 함께 복원해야 한다.

$$
\text{release bundle}
=
(\text{weights},
\text{tokenizer},
\text{prompt},
\text{retriever},
\text{memory},
\text{calibrator},
\text{threshold},
\text{runtime}).
$$

새 모델만 되돌리고 새 embedding index나 새 memory schema를 남겨 두면 이전 champion이 원래 조건에서 동작하지 않는다. Replay buffer도 오염·삭제 요청·라이선스 변경 시 특정 snapshot으로 복구하거나 재구성할 수 있어야 한다.

### 4.19 초보자가 흔히 하는 오해와 교정

#### 오해 1: Baum–Welch가 진짜 상태를 찾아 준다

아니다. 선택한 $K$, emission 가족, Markov 가정 아래 likelihood가 높은 설명을 찾는다. Label switching, local optimum, 상태 분할, 인과적 모호성이 남는다.

#### 오해 2: EM likelihood가 계속 오르면 일반화도 좋아진다

Training likelihood 단조성은 held-out likelihood·상태 해석·운영 효용을 보장하지 않는다. 독립 validation이 필요하다.

#### 오해 3: $A_{jj}=0.9$이면 상태가 정확히 $10$블록 지속된다

기대값이 $10$블록일 뿐이다. Geometric 분포는 $1,2,3,\ldots$ 모두에 질량이 있다.

#### 오해 4: HSMM은 HMM보다 항상 낫다

Duration 정보가 충분하고 가정이 맞을 때 유용하다. 모수가 늘고 계산량이 커지며 right censoring과 최대 duration 선택이 필요하다.

#### 오해 5: BOCPD가 변화라고 하면 새 모델을 학습해야 한다

변화는 배포 버그·계절성·라벨 지연일 수 있다. 원인 조사와 데이터 성숙·권리 검증이 먼저다.

#### 오해 6: Reservoir면 모든 그룹이 공정하게 보존된다

개별 항목의 포함확률이 같을 뿐 희귀 그룹 최소 개수를 보장하지 않는다.

#### 오해 7: Coreset은 작은 validation set이다

아니다. Coreset은 학습 또는 근사를 위해 의도적으로 선택된 집합이다. 확률표본인 독립 final gate를 대신하지 않는다.

#### 오해 8: Gradient가 큰 사례만 저장하면 된다

큰 gradient는 노이즈·오라벨·outlier일 수 있다. 대표성·안전 floor·중복 제거와 함께 봐야 한다.

#### 오해 9: Confidence sequence면 어떤 분석 변경도 안전하다

반복 관찰과 사전 정의한 중지만 보호한다. 후보·그룹·metric을 결과에 맞춰 바꾸면 별도의 선택 보정이 필요하다.

#### 오해 10: 표본 수가 많으면 지연 라벨 편향은 사라진다

편향된 표본은 많이 모아도 편향이 줄지 않는다. Reveal mechanism과 성숙 cohort를 설계해야 한다.

### 4.20 최소 운영 알고리즘

```text
INPUT
  immutable champion bundle
  raw monitoring stream
  replay budget policy
  preregistered metrics, groups, alpha ledger

MONITOR
  update HMM/HSMM filtering and BOCPD run-length posterior
  on alert: freeze raw window and enter HOLD
  do not update production weights

CURATE
  wait for mature labels
  audit rights, duplicates, source blocks, interventions
  update replay candidates using:
    group safety floors
    source-block reservoir
    embedding/gradient coverage
    bounded recent window
  record inclusion probabilities and selector versions

TRAIN
  create a separate challenger from champion
  mix new data and replay under an explicit target objective
  freeze checkpoint and all dependencies

CALIBRATE
  use a disjoint calibration block
  refit probability/conformal/threshold components as needed

GATE
  evaluate paired champion-versus-challenger differences
  update preregistered confidence sequences
  require utility LCB, safety UCB, ESS, latency and hard-stop gates

DEPLOY
  shadow first
  random canary with an anytime-valid stop rule
  promote only the complete release bundle
  rollback the complete bundle on failure
```

### 4.21 구현 전 체크리스트

- [ ] 독립 시퀀스 경계에서 HMM 전이를 세지 않는가?
- [ ] Forward·backward scaling 규약이 일치하는가?
- [ ] $N_j\approx0$과 singular covariance 처리 규칙이 사전 정의됐는가?
- [ ] 여러 초기화와 held-out predictive likelihood를 비교했는가?
- [ ] 상태 ID를 외부 anchor로 버전 간 정렬했는가?
- [ ] 관측창 끝 duration을 right-censored로 처리했는가?
- [ ] BOCPD run-length convention과 hazard 인덱스가 문서화됐는가?
- [ ] Replay 항목이 token이 아니라 적절한 source block 단위인가?
- [ ] 희귀 안전군 floor와 전체 reservoir가 함께 있는가?
- [ ] Inclusion·sampling·loss weight가 각각 기록됐는가?
- [ ] Coreset·gradient selector 버전과 seed가 남는가?
- [ ] Adaptation·calibration·final gate 데이터가 분리됐는가?
- [ ] 후보·그룹·metric·재시작 전체의 alpha ledger가 있는가?
- [ ] 지연 라벨을 event time과 reveal time으로 구분했는가?
- [ ] Confidence sequence의 boundedness·의존성 가정이 맞는가?
- [ ] Shadow·canary·rollback이 전체 release bundle을 대상으로 하는가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **확인 시점:** 2026년 10월 22일 KST. Nemotron 3.5 Retriever 공식 발표는 2026년 10월 21일 게시됐고, Embed·Rerank 모델 카드는 10월 20일, Parse 모델 카드는 10월 17일 release로 표시된다. R2S 논문 v1은 2026년 10월 20일 17:40 UTC, 즉 10월 21일 02:40 KST에 제출됐다. 아래 성능은 각각 공급사 또는 저자 보고치이며 독립 재현 결과가 아니다.

### 5.1 Nemotron 3.5 Retriever: VLM 시대의 병목을 “답변 모델”보다 앞단에서 줄인다

NVIDIA는 문서 증거를 처리하는 세 단계를 하나의 공개 가중치 제품군으로 확장했다.

$$
\text{document page}
\xrightarrow{\text{Parse}}
\text{구조화된 text·box·layout}
\xrightarrow{\text{Embed}}
\text{후보 검색}
\xrightarrow{\text{Rerank}}
\text{근거 순서 정제}
\xrightarrow{\text{Agent}}
\text{답변·도구 행동}
$$

제품군은 다음과 같다.

| 구성요소 | 역할 | 공개 형식·핵심 사양 |
| --- | --- | --- |
| Nemotron 3.5 Embed 8B BF16 | text query로 text·문서 이미지·image+text를 같은 공간에서 검색 | $8.38$B parameters, $4096$차원 embedding, $32{,}768$ token context |
| Nemotron 3.5 Embed 1B BF16·NVFP4 | 작은 dense retriever | 제품명은 1B지만 실제 $1.55$B parameters·$2048$차원 embedding; BF16은 8B teacher distillation, NVFP4는 PTQ+QAD variant |
| Nemotron 3.5 Rerank 8B BF16 | 초기 후보를 query와 함께 더 비싼 cross-encoding으로 재정렬 | $8.38$B parameters, text·image·combined document 입력 |
| Nemotron 3.5 Parse | 문서 이미지에서 text·semantic class·bounding box·reading order 추출 | 정확히 $630{,}800{,}496$ parameters, C-RADIOv4 encoder + 10-layer mBART decoder |

Embed 8B는 text query만 받고 문서 쪽에는 text, image 또는 둘을 함께 넣을 수 있다. Token-level hidden state를 mean pooling한 뒤 $L_2$ 정규화해 $4096$차원 벡터를 만든다. Model card는 text retrieval을 $34$개 언어, visual retrieval을 $18$개 언어에서 평가했다고 밝힌다. Language-model component는 기존 Nemotron text embedder에서 초기화해 고정하고 vision encoder와 multimodal projector만 추가 학습해 text embedding 공간을 보존했다.

#### 공급사 보고 성능을 정확히 읽기

공식 발표의 ViDoRe v3.1 수치는 다음과 같다.

- Embed 8B BF16: nDCG@10 $61.17$
- Embed 1B BF16: nDCG@10 $58.38$, 즉 발표가 계산한 8B 점수의 $95.4\%$
- Rerank 8B BF16: combined image+text nDCG@10 $60.49$
- 비교한 Qwen3-VL-Reranker-8B: 같은 발표 표에서 $59.17$
- MIRACL Vision에서 Embed 8B: nDCG@10 $72.6$, 비교한 Qwen3-VL-Embedding-8B보다 $4.3$ point 높음

여기서 nDCG@10은 검색 상위 $10$개의 순서가 정답 관련도 순서와 얼마나 맞는지를 $0$에서 $1$ 또는 $0$에서 $100$ scale로 요약한 ranking 지표다. 답변 정확도나 환각률과 같은 값이 아니다.

공식 agentic retrieval 실험은 agent LLM, corpus, tools, task set, execution policy를 고정하고 search tool의 Embed·Rerank 구성만 바꿨다고 설명한다. Rerank를 추가했을 때

- Embed 8B 구성에서는 agent 총 inference token이 약 $14\%$ 감소했고,
- Embed 1B 구성에서는 약 $13\%$ 감소했으며,
- 두 설정 모두 retrieval accuracy가 향상됐다고 보고했다.

이는 더 좋은 근거가 앞에 오면 agent가 반복 검색과 불필요한 문서 검사를 줄일 수 있다는 시스템 효과다. 그러나 모든 agent·corpus에서 $13$–$14\%$가 보장된다는 뜻은 아니다.

NVFP4 1B variant에 대해서는 GB200 측정에서 BF16 $101.3$ documents/s 대비 $128.8$ documents/s, 즉 약 $27.1\%$ 높은 처리량과 BF16 retrieval accuracy의 $99.85\%$ 유지가 보고됐다. 이 역시 특정 hardware·batch·입력 설정의 공급사 측정이다.

#### 공개 범위와 재현 경계

“오픈”을 한 단어로 뭉치면 안 된다.

- Embed·Rerank·Parse weight는 Hugging Face에 공개됐고 상업·비상업 사용 가능하다고 모델 카드가 설명한다.
- 지배 라이선스는 Apache-2.0이 아니라 **OpenMDW-1.1**이다. 재배포 고지·라이선스 의무와 특허 관련 조항을 원문에서 확인해야 한다.
- Embed·Rerank의 Ministral base는 별도의 Apache-2.0 조건을 따른다.
- Model card는 Embed 8B의 학습에 image 또는 image+text $2.3$M samples와 text-only $50$M+ samples를 사용했다고 밝힌다. Public·commercially permissible·synthetic data를 섞었지만 모든 합성 절차와 정확한 mixture weight가 한 줄 recipe로 완전히 재현되는 것은 아니다.
- ViDoRe V3 설명에는 $8$개 public과 $2$개 private dataset이 포함된다. 전체 평가를 외부 사용자가 동일하게 재현할 수 없다는 뜻이다.
- 발표의 ViDoRe **v3.1** $61.17$과 모델 카드의 ViDoRe **V3** combined-input $61.47$은 benchmark revision이 다르다. MTEB 정의상 v3.1은 `.v2` task들과 OCR 기반 markdown을 추가하므로 두 값을 같은 열에 섞어 직접 비교하지 않는다.
- Model card 비교에서는 NVIDIA 모델의 기본 image `longest_edge`가 $1{,}120$ pixels이고 비교 Qwen 모델의 `max_pixels` 설정이 다르다. 계산량·해상도가 완전히 일치하는 대조가 아니다.
- 8B BF16 카드의 hardware requirements에는 A100 $80$GB, H100 $80$GB, GB200 $186$GB가 적혀 있고, 앞의 $101.3\to128.8$ documents/s는 GB200 측정이다. 요구 hardware 목록 전체가 throughput 측정 장비라는 뜻은 아니며 소비자 GPU에서 같은 값을 기대하면 안 된다.

**엔지니어 인사이트 (Impact):** VLM·LLM agent의 품질은 생성 모델 하나가 아니라 `parse → retrieve → rerank → reason` 전체 증거 경로의 곱이다. 새 retriever를 도입하면 embedding 좌표, index, top-$k$, reranker, prompt context가 함께 바뀐다. 따라서 offline nDCG만 보지 말고 같은 query에 대해 champion pipeline과 challenger pipeline을 paired 비교하고, 언어·문서 유형·표·차트 그룹별 효용 LCB와 비용 UCB를 순차 gate로 본다. Index 재구축 뒤에는 rollback 가능한 이전 embedding index를 남겨야 한다.

- 공식 발표: [NVIDIA Nemotron 3.5 Embed, Rerank, and Parse](https://huggingface.co/blog/nvidia/nemotron-retriever-world-class-agentic-retrieval)
- 제품군: [Nemotron 3.5 Retriever collection](https://huggingface.co/collections/nvidia/nemotron-35-retriever)
- Embed 8B 모델 카드: [nvidia/Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16)
- Embed 1B 모델 카드: [nvidia/Nemotron-3.5-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16)
- Rerank 8B 모델 카드: [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16)
- Parse 모델 카드: [nvidia/Nemotron-3.5-Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)
- Benchmark 정의: [MTEB ViDoRe V3·v3.1](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py#L3017-L3043)
- 라이선스 원문: [OpenMDW License 1.1](https://openmdw.ai/license/1-1/)

### 5.2 R2S: 잘 행동하는 teacher가 반드시 잘 점수화하는 것은 아니다

Agentic on-policy distillation(OPD)은 student가 자기 trajectory를 만들고, teacher가 student의 각 token에 dense score를 주어 student를 학습하는 방식이다. 일반적인 직관은 student가 실수해 teacher가 낯선 상태에 들어가면 teacher supervision이 나빠진다는 것이다.

R2S 연구는 다른 실패를 지적한다.

> Teacher는 student가 벗어난 상태에서 직접 행동하면 과제를 끝낼 수 있지만, 다음 token만 국소적으로 점수화할 때는 “한 번 더 둘러보기”, “조금 더 검색하기”처럼 그 순간에는 그럴듯하지만 turn budget을 소모하는 행동을 충분히 벌하지 못한다.

이를 **rollout–scorer gap**이라고 부른다. Outcome-based RL은 episode가 미완료로 끝나면 $0$ reward를 줄 수 있지만, 평범한 next-token scorer는 현재 history만 보고 각각의 지연 token을 평가한다.

#### R2S 파이프라인

R2S(Rollout-to-Scorer Distillation)는 student를 직접 teacher rollout에 맞추기 전에 scorer 자체를 수리한다.

1. 학습 전 student trajectory에서 student와 teacher의 token distribution 차이가 큰 상태를 cell당 최대 $2{,}400$개 채굴한다.
2. 각 상태를 복원하고 teacher가 $K=8$번 takeover해 episode 끝까지 행동한다.
3. Consensus 또는 LLM judge 같은 reward-free check로 통과 가능한 takeover를 고르고, 그중 짧은 rollout을 reference로 선택한다.
4. Teacher가 reference를 context에서 본 상태로 student의 원래 token stream에 부여한 **미래를 반영한 분포**를 target으로 만든다.
5. Reference가 없는 closed-book 상태에서도 그 분포를 내도록 teacher를 한 epoch fine-tuning한다.
6. 수리된 teacher를 다시 동결하고 평범한 OPD의 유일한 scorer로 student를 학습한다.

Student는 teacher takeover text 자체를 직접 보지 않는다. 핵심은 끝까지 본 teacher의 판단을 token-level soft target에 옮기는 것이다.

#### 저자 보고 결과

연구는 두 teacher 모델과 네 student 모델(Qwen3.5-2B/4B, Qwen3-4B/8B), 세 text-agent environment인 ALFWorld·WebShop·ScienceWorld의 $12$개 teacher–student–environment cell을 평가했다. 저자들은 R2S가 비교한 세 최신 agentic OPD 방법 중 cell별 최선보다 평균 $12.8$ success percentage points 높았다고 보고한다.

대표적인 Qwen3.5-9B teacher 결과는 다음과 같다.

| Environment | Student | Vanilla OPD | R2S | 차이 |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 2B | $58.0\pm2.4$ | $77.9\pm1.3$ | $+19.9$ pp |
| ALFWorld | 4B | $65.8\pm2.6$ | $81.9\pm2.4$ | $+16.1$ pp |
| WebShop | 2B | $35.7\pm1.6$ | $57.9\pm1.6$ | $+22.2$ pp |
| WebShop | 4B | $34.6\pm2.9$ | $55.9\pm1.4$ | $+21.3$ pp |
| ScienceWorld | 2B | $23.7\pm2.0$ | $37.9\pm0.6$ | $+14.2$ pp |
| ScienceWorld | 4B | $34.6\pm3.5$ | $47.8\pm3.0$ | $+13.2$ pp |

중요한 통계적 단서가 있다. 주 결과 Table 2에서 각 cell의 student training은 **한 번의 $150$-step run**이고, 표의 $\pm$는 독립 training seed 여러 개의 변동이 아니라 final checkpoint를 세 decoding seed로 평가한 표준편차다. 다만 Figure 3의 Qwen3.5-9B $\to$ 4B 학습곡선은 다섯 방법을 각각 $3$개 training run으로 그렸다. 세 R2S run은 하나의 repaired teacher를 공유하므로 scorer-repair 단계의 run-to-run 변동까지 반복 측정한 것은 아니다.

Qwen3.5-9B $\to$ 4B 설정의 takeover 선택 ablation에서는 다음 success가 보고됐다.

| 선택 규칙 | ALFWorld | WebShop | ScienceWorld |
| --- | ---: | ---: | ---: |
| Vanilla OPD | $65.8\pm2.6$ | $34.6\pm2.9$ | $34.6\pm3.5$ |
| Consensus vote | $81.9\pm2.4$ | $63.1\pm1.6$ | $39.5\pm2.9$ |
| LLM judge | $82.6\pm0.9$ | $55.9\pm1.4$ | $50.3\pm1.5$ |
| Ground-truth reward | $85.9\pm1.2$ | $65.0\pm1.4$ | $49.0\pm1.1$ |

Reward-free 두 규칙 모두 세 환경에서 vanilla OPD를 넘었지만, 어느 규칙이 항상 ground truth와 같은 순위를 내는 것은 아니다. ScienceWorld에서는 이 표의 LLM judge가 ground-truth 선택보다 높게 나왔으며, 이는 제한된 평가 변동 속에서 한 방법이 보편적으로 우월하다는 증거가 아니다.

#### 공개성과 재현 비용

이 연구는 공개 artifact가 비교적 구체적이다.

- 논문 v1과 HTML 전체가 CC BY 4.0으로 공개돼 있다.
- `miles-agent` 저장소의 `r2s` tag에 recipe·prompt·환경 adapter·data digest가 고정돼 있다.
- 저장소 원저작 코드는 Apache-2.0이고 vendored WebShop은 MIT이며, 그 밖의 third-party 구성요소도 각 라이선스를 따른다.
- $24$개 teacher·student checkpoint가 Hugging Face collection에 public·ungated로 공개됐고 각 model card metadata는 Apache-2.0이다. 그래도 base model과 사용 환경의 조건은 함께 확인해야 한다.
- 저장소는 스스로 “personal research code”라고 밝히며, 논문용 코드는 `r2s` tag에 고정돼 있고 main은 크게 삭제·리팩터링될 수 있으므로 paper tag를 사용해야 한다.

재현에는 큰 비용과 조건이 있다.

- 최대 $2{,}400\times8=19{,}200$ teacher takeover episode를 cell마다 생성한다.
- Teacher full-parameter fine-tuning과 student OPD가 필요하다.
- 상태를 정확히 복원할 수 있는 replayable environment가 필요하다.
- 실험은 Qwen3·Qwen3.5와 세 text-agent benchmark에 한정된다.
- Data는 저장소가 재배포하지 않고 upstream에서 받아 digest를 확인하므로 제3자 license와 setup을 따로 해결해야 한다.
- Consensus는 같은 잘못된 결말에 여러 rollout이 모일 수 있고, LLM judge는 자체 편향·prompt 민감도·모델 변화가 있다.

**엔지니어 인사이트 (Impact):** R2S는 “강한 teacher를 고르면 dense supervision도 자동으로 좋다”는 가정을 깨뜨린다. Agent에서는 국소 token score와 장기 episode outcome 사이의 시간적 credit assignment가 핵심이다. 오늘의 replay 관점에서 보면 high-disagreement state 채굴은 일종의 adaptive coreset이고, teacher takeover는 그 상태의 미래 정보를 얻는 비싼 label acquisition이다. 따라서 어떤 상태를 골랐는지, 어떤 checker가 rollout을 통과시켰는지, 실패한 후보를 몇 번 재시험했는지를 기록해야 한다. Reward-free라는 말은 error-free라는 뜻이 아니므로, 학습 뒤에는 별도 episode에서 champion과 paired 비교하고 confidence sequence로 승인해야 한다.

- 논문: [Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation](https://arxiv.org/abs/2610.28586)
- 프로젝트: [R2S project page](https://shuzhao.me/r2s/)
- 재현 코드: [Tree-Shu-Zhao/miles-agent `r2s` tag](https://github.com/Tree-Shu-Zhao/miles-agent/tree/r2s)
- 공개 체크포인트: [R2S Rollout-to-Scorer Distillation collection](https://huggingface.co/collections/TreezzZ/r2s-rollout-to-scorer-distillation-6ad750cab4aa1226dec0a305)

### 5.3 두 트렌드는 “선택된 증거”가 모델보다 앞설 수 있음을 보여 준다

| 비교 축 | Nemotron 3.5 Retriever | R2S | 오늘의 교훈 |
| --- | --- | --- | --- |
| 선택 대상 | corpus의 문서·페이지 | student가 벗어난 상태와 teacher takeover | 모든 데이터를 같은 비용으로 볼 수 없으므로 선택기가 중요하다. |
| 선택 기준 | embedding similarity와 reranker relevance | teacher–student disagreement와 consensus/judge | 선택 기준도 학습·검량·버전 관리 대상이다. |
| downstream 소비자 | VLM·LLM agent | OPD student | 앞단 오류가 뒤 단계에서 증폭될 수 있다. |
| 공개 형태 | OpenMDW-1.1 weights·cards·일부 recipe | Apache-2.0 research code tag·24 checkpoints | “오픈”은 weight·code·data·eval별로 확인한다. |
| 핵심 저자 주장 | 더 나은 retrieval이 agent token을 줄이고 근거 품질을 높임 | rollout-informed scorer가 장기 지연 행동을 줄임 | 큰 생성 모델만 바꾸는 것이 유일한 개선축이 아니다. |
| 주요 검증 한계 | private eval 일부·hardware·resolution·benchmark revision | 주 결과 한 training run/cell·일부 4B 곡선 3 runs이나 single repaired teacher 공유·final checkpoint 3 decode seeds·replayable text env | headline 평균을 배포 보장으로 읽지 않는다. |
| 오늘 수학과 연결 | retrieval 후보는 coreset·ranking 문제 | state mining은 adaptive coreset·future label 문제 | 선택 이후 독립 paired gate와 alpha ledger가 필요하다. |

두 연구를 한 문장으로 묶으면 다음과 같다.

> **AI 시스템의 다음 병목은 더 큰 backbone만이 아니라, 어떤 근거·상태·trajectory를 제한된 예산으로 선택하고 그 선택이 실제 장기 결과를 개선했는지 독립적으로 증명하는 데 있다.**

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

두 상태 HMM의 Bernoulli emission을 Baum–Welch로 학습한다고 하자. 상태 $j$에 대한 관측과 E-step 책임도가

$$
y=(1,0,1,1),
\qquad
\gamma(j)=(0.8,0.2,0.6,0.4)
$$

이다. 또한 운영자가 이 상태의 explicit duration 후보를

$$
q_j(1)=0.2,
\qquad
q_j(2)=0.5,
\qquad
q_j(3)=0.3
$$

으로 추정했다. 새 challenger의 paired utility 차이는 $Z_s\in[-1,1]$이고, 독립 final gate $n=10{,}000$건에서

$$
\overline Z_n=0.08
$$

이었다. 총 오류예산은 $\delta=0.05$다.

1. 상태 $j$의 Bernoulli emission M-step 결과 $p_j^{\mathrm{new}}$는 얼마인가?
2. 상태 $j$와 다른 상태의 번호를 모든 모수에서 동시에 바꾸면 관측 likelihood가 달라지는가?
3. 위 duration의 hazard $h_j(1),h_j(2),h_j(3)$는 얼마이며 geometric duration인가?
4. 교육용 error-spending confidence sequence의 폭 $w_n$과 utility LCB는 대략 얼마인가?
5. BOCPD 변화 posterior가 높고 utility LCB가 $0$보다 크면 곧바로 challenger를 승인해도 되는가?

### 모범 답안

먼저 상태 $j$의 유효 표본 수는

$$
N_j
=
0.8+0.2+0.6+0.4
=
2.0
$$

이다. 경보 $1$의 가중 성공 횟수는

$$
\sum_t\gamma_t(j)y_t
=
0.8(1)+0.2(0)+0.6(1)+0.4(1)
=
1.8
$$

이므로

$$
p_j^{\mathrm{new}}
=
\frac{1.8}{2.0}
=
0.9
$$

이다. 이는 네 시점을 hard assignment한 비율이 아니라 posterior 책임도로 가중한 성공률이다.

상태 번호를 순열 $\sigma$로 바꾸면서 $\boldsymbol\pi$, $A$, emission을 모두 같은 순열로 바꾸면

$$
p_{\theta'}(y_{1:T})
=
p_\theta(y_{1:T})
$$

이므로 관측 likelihood는 달라지지 않는다. 따라서 상태 번호 자체는 식별되지 않는다.

Hazard는 “그 길이까지 살아남았다는 조건에서 정확히 거기서 끝날 확률”이다.

$$
h_j(1)
=
\frac{0.2}{0.2+0.5+0.3}
=
0.2,
$$

$$
h_j(2)
=
\frac{0.5}{0.5+0.3}
=
0.625,
$$

$$
h_j(3)
=
\frac{0.3}{0.3}
=
1.
$$

Hazard가 $0.2,0.625,1$로 일정하지 않으므로 geometric duration이 아니다. 이 구조를 유지하려면 표준 HMM보다 explicit duration HSMM이 자연스럽다.

$Z_s\in[-1,1]$이므로 범위 폭은 $R=2$다. 교육용 순차 폭은

$$
w_n
=
\sqrt{
\frac{2}{n}
\log\left(
\frac{\pi^2n^2}{3\delta}
\right)
}
$$

이고 $n=10{,}000$, $\delta=0.05$를 넣으면

$$
w_n\approx0.067.
$$

따라서 utility LCB는

$$
\operatorname{LCB}
=
0.08-0.067
\approx0.013>0
$$

이다. 사전 정의한 단일 효용 metric의 누적 평균 개선이 양수라는 순차 증거는 생겼다.

그래도 곧바로 승인할 수는 없다.

1. BOCPD posterior는 변화 시점에 대한 믿음이지 모델 업데이트의 허가가 아니다.
2. Utility LCB 하나만 통과했다. 핵심 그룹별 회귀와 안전비용 UCB를 확인해야 한다.
3. $\delta=0.05$를 여러 그룹·metric·후보에 썼다면 alpha ledger에 맞춘 더 작은 개별 예산으로 다시 계산해야 한다.
4. Final gate가 replay·adaptation·calibration과 분리됐는지 확인해야 한다.
5. 지연 라벨의 성숙, ESS, OOD, latency, hard-stop 조건을 통과해야 한다.
6. 그 뒤에도 shadow와 작은 random canary를 거쳐야 한다.

핵심은 다음 네 부등호를 기억하는 것이다.

$$
\text{Baum--Welch likelihood 증가}
\neq
\text{진짜 상태 발견}
\neq
\text{BOCPD 변화 경보}
\neq
\text{안전한 배포 승인}.
$$

---

**다음 학습 예고:** 오늘은 숨은 상태모형의 모수학습과 explicit duration·온라인 변화점 추론을 제한된 replay와 순차 승인에 연결했다. 다음에는 particle filter·sequential Monte Carlo로 비선형·비Gaussian 상태를 근사하고, replay buffer의 influence function·Shapley 근사·데이터 삭제(unlearning)와 결합해 “어떤 기억이 실제 업데이트에 얼마나 영향을 줬는가?”를 추적한다.
