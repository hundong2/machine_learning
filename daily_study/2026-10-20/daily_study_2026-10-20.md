# [2026-10-20] 오늘 학습: 모호성 반경·그룹/거리 선택 편향 & 변화점 연동 Online DRO·안전 롤백

> **오늘의 핵심 문장:** 분포강건 최적화(Distributionally Robust Optimization, DRO)가 안전해 보이는 이유는 “최악의 분포”를 계산하기 때문이지만, 그 최악의 범위를 정하는 반경 $\rho$, 이동 비용을 정하는 거리 $d$, 위험을 나누는 그룹 $g$를 같은 데이터에서 유리하게 고르면 보호막 자체가 과적합될 수 있다. 안정 구간에서 이 세 요소를 검량하고 버전으로 고정한 뒤, 변화점 경보가 울리면 반경을 즉석에서 키우는 대신 배포를 멈추고, 현재 체제에서도 검증된 안전 정책이나 미리 정의한 안전 모드(fail-safe)로 전환한 뒤 새 체제의 독립 데이터로 다시 승인해야 한다.

지난 시간에는 Deep Ensemble과 conformal calibration으로 모델 불확실성을 점검하고, TV·KL·Wasserstein 모호성 집합으로 context 혼합비 변화에 강한 정책을 고르는 법을 배웠다. 당시에는 모호성 집합

$$
\mathcal P_{\rho}^{d}
=
\left\{
Q_X:
W_d(Q_X,\widehat P_{\mathrm{ref}}^X)\le\rho
\right\}
$$

의 반경 $\rho$와 거리 $d$가 이미 주어졌다고 가정했다. 여기서 $Q_X$는 미래 context 분포다. 오늘의 DRO 범위는 $P(X)$의 혼합비가 바뀌는 **covariate·context-mix shift**이며, 같은 $X$에서 정답이나 보상 규칙 $P(R,C\mid X,A)$ 자체가 바뀌는 concept shift는 별도 outcome 감시와 불확실성 경계로 다룬다. 실무에서는 $\rho$와 $d$를 우리가 정하고 그룹을 어떻게 자를지도 우리가 정한다. 오늘은 “강건 최적화기를 돌리는 법”보다 먼저, **강건성의 자와 경계선을 어떻게 검량하고 감시할지**를 배운다.

오늘의 연결 흐름은 다음과 같다.

$$
\text{안정 과거 블록}
\longrightarrow
\text{거리·그룹 사전 정의}
\longrightarrow
\text{반경 검량}
\longrightarrow
\text{독립 안전 게이트}
\longrightarrow
\text{온라인 변화점 감시}
\longrightarrow
\text{hold·rollback·재승인}
$$

> **난이도 표지:** 오늘 내용은 고급 운영 통계지만 직관은 간단하다. 우산의 크기인 $\rho$와 지도에서의 거리인 $d$를 폭우가 온 뒤 마음대로 바꾸면 “원래 우산이 비를 막았는가?”를 평가할 수 없다. 비가 오기 전에 자를 고정하고, 예상 범위를 넘는 폭우가 오면 우산 점수를 조작하는 대신 실내로 대피해야 한다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 DRO 반경을 정해야 할까?

DRO는 하나의 미래 분포를 맞히지 않고 기준 분포 주변의 여러 분포를 고려한다. 이때 $\rho$는 “얼마나 멀리까지 그럴듯한 미래로 볼 것인가?”를 정한다.

- $\rho=0$이면 기준 분포 하나만 믿으므로 경험위험최소화와 비슷해진다.
- $\rho$가 너무 작으면 실제 변화가 집합 밖으로 빠져 보호를 받지 못한다.
- $\rho$가 너무 크면 거의 모든 상황을 걱정해 정책이 지나치게 보수적이거나 쓸모없어질 수 있다.

따라서 $\rho$는 안전을 강하게 만드는 단순 손잡이가 아니다. **평상시 변동과 표본오차를 어느 정도 허용할지 정하는 검량값**이다.

### 1.2 왜 거리 함수가 단순한 기술 세부사항이 아닐까?

Wasserstein 거리는 확률질량을 한 위치에서 다른 위치로 옮기는 최소 비용이다. 두 고객 유형이 얼마나 비슷한지를 거리 $d(x,x')$가 정한다.

예를 들어 “한국어 일반 질문”과 “한국어 의료 질문”을 가깝게 둘지, “한국어 일반 질문”과 “영어 일반 질문”을 가깝게 둘지는 목적에 따라 달라진다. 언어만 중시하면 앞의 두 유형이 가깝고, 위험 도메인을 중시하면 뒤의 두 유형이 더 가까울 수 있다.

즉, Wasserstein DRO의 적대자는 거리표가 싸다고 말하는 방향으로 질량을 많이 옮긴다. 잘못된 거리표는 현실에는 어려운 이동을 싸게 만들거나, 흔한 변화를 불가능한 이동처럼 비싸게 만든다.

### 1.3 왜 그룹 정의도 선택 편향을 만든다고 할까?

운영 대시보드에서 언어, 국가, 기기, 시간대, 신규·기존 고객, 위험 도메인 등 수십 가지 방식으로 데이터를 나눌 수 있다. 그중 가장 나쁜 그룹만 찾아 보고하면 실제 위험을 발견할 수도 있지만, 우연히 나쁘게 나온 조각을 잡을 수도 있다.

같은 데이터에서

1. 가장 나쁜 그룹 정의를 찾고,
2. 그 그룹의 위험을 추정하고,
3. 그 추정치를 최종 안전 증거라고 부르면

탐색과 검증이 섞인다. 이는 여러 정책 중 최고 점수만 골랐을 때 생기는 최댓값 선택 편향과 같은 구조다.

### 1.4 왜 거리도 학습하려 할까?

손으로 만든 거리에는 한계가 있다. 픽셀 공간에서는 밝기만 조금 바뀐 두 이미지가 멀어 보일 수 있고, 문장 표면형은 다르지만 같은 의도를 가진 요청도 멀어 보일 수 있다. 그래서 임베딩 $h(x)$를 만들고, 비슷해야 하는 쌍과 달라야 하는 쌍으로 거리의 가중치를 학습한다.

하지만 학습된 거리는 자연법칙이 아니다. 어떤 쌍을 “비슷하다”고 라벨링했는지, 어떤 임베딩을 사용했는지, 어떤 손실을 최소화했는지에 따라 달라지는 **모델 산출물**이다. 따라서 거리에도 train·calibration·test 분리가 필요하다.

### 1.5 왜 변화점 경보 뒤에 반경만 키우면 안 될까?

새 데이터가 기존 Wasserstein ball 밖에 나타났다고 하자. 가장 쉬운 대응은 $\rho$를 늘려 새 데이터를 다시 ball 안에 넣는 것이다. 그러나 이는 화재경보가 울릴 때 경보 임계값을 올려 소리를 끄는 것과 같다.

경보 뒤 즉시 반경을 키우면 다음 문제가 생긴다.

- 안정기 자료로 얻은 coverage 해석이 사라진다.
- 이상 구간이 새 기준에 섞여 정상처럼 보일 수 있다.
- 정책이 왜 실패했는지 확인하기 전에 탐지기와 보호막을 동시에 바꾸게 된다.
- 커진 집합에 대한 최적 정책이 지나치게 보수적이어도 이를 안전 향상으로 착각할 수 있다.

따라서 빠른 경로는 **hold 또는 rollback**, 느린 경로는 **원인 분석·새 체제 검량·독립 재승인**이어야 한다.

### 1.6 왜 covariate 경보와 성능 경보를 나눌까?

입력 $X$의 분포는 즉시 볼 수 있지만 실제 보상 $R$이나 사고 라벨은 늦게 도착할 수 있다.

- covariate 경보: 언어 비율, 요청 길이, 이미지 품질처럼 즉시 관측되는 변화
- performance 경보: 실제 정답률, 비용, 사고율처럼 라벨이 있어야 확인되는 변화

입력 변화가 커도 모델 성능이 유지될 수 있고, 입력 변화가 작아 보여도 조건부 관계 $P(Y\mid X)$가 바뀌어 성능이 무너질 수 있다. 두 경보는 서로를 대신하지 않는다.

### 1.7 왜 rollback은 “이전 모델 파일 복사”보다 큰 개념일까?

정책은 모델 가중치만이 아니다. 프롬프트, 검색 인덱스, 도구 권한, 라우팅 규칙, 전처리기, 거리 함수, 반경, threshold가 함께 행동을 만든다. 따라서 안전한 rollback은 다음 버전 묶음을 복원해야 한다.

$$
v
=
\left(
\theta,
\text{prompt},
\text{retriever},
\widehat P_{\mathrm{ref}}^X,
\phi,
\rho,
g,
\text{detector},
\text{guardrail},
\text{runtime}
\right)
$$

여기서 $\theta$는 모델 파라미터, $\phi$는 거리 파라미터, $g$는 그룹 규칙이다. 일부만 되돌리면 과거에 검증하지 않은 조합이 된다. 이 묶음도 현재 support에서 안전하다고 확인된 경우에만 fallback으로 쓰며, 그렇지 않으면 fail-safe가 우선이다.

### 1.8 오늘의 본질

오늘의 질문은 “DRO를 더 복잡하게 만드는가?”가 아니다.

> **우리가 만든 불확실성 집합이 실제로 평상시 변화는 포함하고, 구조적 변화는 경보로 분리하며, 데이터 탐색에 과적합되지 않았는가?**

이를 위해서는 최적화보다 먼저 실험 설계, 데이터 분리, 버전 관리, 중지 규칙이 필요하다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| $Z=(X,A,R,C)$ | 한 관측 단위. 입력, 행동, 보상, 안전 비용을 묶어 쓴다. | 로그 한 행의 구성 요소를 나타낸다. |
| $P_{\mathrm{ref}}^X$ | 정상으로 승인된 기준 context 분포 | 새 입력 블록이 얼마나 달라졌는지 비교하는 기준점이다. |
| $\widehat P_{\mathrm{ref}}^X$ | 유한한 기준 데이터로 만든 context 경험분포 | 실제로 계산 가능한 기준 분포다. |
| $Q_X$ | 미래에 올 수 있다고 가정하는 context 분포 | 오늘의 DRO가 바꾸는 대상이다. |
| $v_\pi(x)$ | context $x$에서 정책 $\pi$를 썼을 때의 조건부 기대보상 | context 혼합비가 바뀔 때 정책 가치를 계산한다. |
| $\kappa_\pi(x)$ | context $x$에서 정책 $\pi$의 조건부 기대 안전비용 | 강건 비용 상한을 계산한다. |
| empirical distribution | 관측된 각 데이터 점에 같은 확률질량을 둔 분포 | 표본에서 Wasserstein 거리를 계산하게 한다. |
| ambiguity set $\mathcal P_\rho^d$ | 기준 분포 주변에서 고려할 후보 분포들의 집합 | DRO가 방어하는 범위다. |
| ambiguity radius $\rho$ | 모호성 집합의 크기 | 평상시 변동을 어디까지 허용할지 정한다. |
| ground metric $d$ | 한 개체를 다른 개체로 옮기는 비용 | Wasserstein 이동의 방향과 가격을 정한다. |
| 1-Wasserstein distance $W_d(P,Q)$ | metric $d$를 따라 $P$의 질량을 $Q$로 옮기는 최소 총비용 | 두 분포의 차이를 재는 운송 거리다. |
| optimal transport cost | 주어진 비용표로 질량을 옮기는 최소비용 | 비용표가 metric 조건을 만족하지 않으면 distance 대신 이 이름을 쓴다. |
| coupling $\Gamma$ | 출발 그룹의 질량을 도착 그룹으로 얼마나 보낼지 적은 표 | 최적 운송 계획이다. |
| embedding $h(x)$ | 입력을 숫자 벡터로 바꾼 표현 | 의미 기반 거리의 재료다. |
| metric parameter $\phi$ | 학습된 거리의 가중치 | 어떤 변화가 가깝고 먼지 결정한다. |
| positive semidefinite $M\succeq0$ | 모든 벡터 $u$에 대해 $u^\top Mu\ge0$인 행렬 | 음수가 아닌 Mahalanobis형 거리를 만든다. |
| pseudometric | 서로 다른 두 점의 거리가 $0$일 수도 있는 거리 | 압축 임베딩의 충돌을 정직하게 표현한다. |
| identifiability | 서로 다른 설정이 같은 관측 결과를 만들지 구분 가능한 성질 | $d$의 크기와 $\rho$의 크기를 따로 해석할 때 필요하다. |
| stable block | 정상 체제라고 판정된 시간·세션 묶음 | 반경을 검량하는 교환 단위다. |
| calibration block | 학습에는 쓰지 않고 threshold·분위수를 정하는 블록 | $\rho$에 빈도적 의미를 부여한다. |
| order statistic $\Delta_{(k)}$ | 값을 정렬했을 때 $k$번째 값 | 유한표본 conformal 반경이 된다. |
| exchangeability | 블록 순서를 바꿔도 결합분포가 같다는 조건 | rank 기반 coverage의 핵심 가정이다. |
| IID | independent and identically distributed의 약자. 서로 독립이고 같은 분포에서 나왔다는 뜻 | DKW 같은 고전적 표본 경계의 가정이다. |
| CDF $F(x)$ | 값이 $x$ 이하일 누적확률 | 거리 분위수를 정의한다. |
| DKW inequality | 경험 CDF와 실제 CDF의 최대 차이를 유한표본에서 제한하는 부등식 | 분위수 추정 자체의 오차를 본다. |
| temporal dependence | 가까운 시간의 데이터가 서로 영향을 받는 현상 | 행 단위 무작위 분할을 위험하게 만든다. |
| group map $g_\eta(x)$ | 입력을 그룹 번호로 보내는 규칙 | 언어·위험도·기기별 분석을 정의한다. |
| candidate family $\mathcal H$ | 실제로 검토·보고할 slice 가설들의 사전 정의 집합 | 선택 편향 보정의 범위를 정한다. |
| multiple comparison | 많은 후보를 동시에 보면 우연한 극단값이 생기는 문제 | 동시 신뢰경계나 독립 test가 필요해진다. |
| change point | 데이터 생성 규칙이 바뀐 시점 | 기존 기준 버전을 계속 써도 되는지 판단하게 한다. |
| detector statistic $E_t$ | 시간 $t$까지의 경보 증거를 누적한 수 | anytime alarm을 만들 수 있다. |
| e-process | 귀무가설 아래 평균적으로 불어나지 않도록 만든 비음수 증거 과정 | 반복 관찰에도 전체 오경보를 통제한다. |
| filtration $\mathcal F_t$ | 시점 $t$까지 알고 있는 모든 정보의 묶음 | detector가 미래를 미리 보지 않았음을 표현한다. |
| confidence sequence | 모든 시점을 한꺼번에 덮도록 만든 시간가변 신뢰구간 | canary를 반복 점검해도 coverage를 유지한다. |
| stopping time $\tau$ | 현재와 과거 정보만으로 결정되는 중지 시점 | 경보가 언제 울렸는지 수학적으로 나타낸다. |
| false alarm | 실제 변화가 없는데 경보가 울리는 오류 | 불필요한 rollback 비용을 만든다. |
| detection delay | 실제 변화 뒤 경보까지 걸린 시간 | 빠른 보호와 오경보 사이의 trade-off다. |
| label delay | 실제 성능·사고 라벨이 늦게 도착하는 현상 | 즉시 proxy와 지연 확정 지표를 나누게 한다. |
| online DRO | 시간에 따라 기준·가치 추정을 갱신하는 DRO | 새 체제에 적응하되 오염을 통제해야 한다. |
| champion | 현재 승인되어 실제 트래픽을 받는 정책 | rollback 대상의 기준점이다. |
| challenger | 새로 제안되어 검증 중인 정책 | shadow·canary를 거쳐야 한다. |
| shadow mode | 실제 행동에는 영향 없이 새 정책의 출력을 기록하는 단계 | 안전하게 성능을 비교한다. |
| canary | 작은 무작위 트래픽에만 새 정책을 노출하는 단계 | 전체 배포 전 위험을 제한한다. |
| rollback | 현재 체제에서도 검증된 안전 버전으로 되돌리거나 fail-safe로 전환하는 조치 | 경보 뒤 손실 확산을 막는다. |
| fail-safe | 안전성에 의문이 생겼을 때 미리 정한 더 안전한 동작으로 전환하는 상위 원칙 | 검증된 $v_{\mathrm{safe}}$, 기능 제한, 요청 보류, 사람 검토 등이 후보가 된다. |
| fail-closed | 안전을 입증할 수 없으면 요청을 자동 허용하지 않고 거부·보류하는 fail-safe의 한 방식 | 잘못된 자동 행동보다 보수적 중단을 택한다. |
| re-entry gate | 롤백된 정책이나 새 정책이 다시 들어오기 위한 조건 | “경보가 조용해짐”과 “안전 회복”을 구분한다. |
| frozen evidence window | 의사결정 규칙을 정한 뒤 수정하지 않는 평가 구간 | 반복 엿보기와 데이터 누수를 막는다. |
| Dirac measure $\delta_x$ | 확률질량 $1$을 점 $x$ 하나에 놓는 분포 | 경험분포를 간결하게 쓴다. |
| infimum $\inf$·supremum $\sup$ | 가능한 값의 가장 큰 하한·가장 작은 상한 | 최악의 보상과 비용을 표현한다. |
| LCB·UCB | lower·upper confidence bound, 즉 통계 오차까지 반영한 하한·상한 | DRO 최악값과 추정 불확실성을 함께 gate에 넣는다. |
| audit trail | 데이터·버전·결정 근거의 변경 불가능한 기록 | 사고 뒤 재현과 책임 추적에 필요하다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 문제 설정

시간 블록 $b=1,\dots,m$에서 관측한 context로 경험분포 $\widehat P_b^X$를 만든다. 한 블록은 예를 들어 하루, 한 주, 또는 서로 의존성이 낮은 사용자 묶음일 수 있다. 별도의 기준 블록 집합 $\mathcal I_{\mathrm{ref}}$로

$$
\widehat P_{\mathrm{ref}}^X
=
\frac{1}{|\mathcal I_{\mathrm{ref}}|}
\sum_{b\in\mathcal I_{\mathrm{ref}}}
\widehat P_b^X
$$

를 만든다고 하자. 표기는 각 블록의 크기가 같다는 단순화다. 크기가 다르면 관측 수에 비례한 가중 평균을 사용한다.

정책 $\pi(a\mid x)$가 context $x$에서 행동 $a$를 고를 확률이라고 하자. 행동 전 context에 조건부인 기대보상과 기대비용을

$$
v_\pi(x)
=
\mathbb E_{A\sim\pi(\cdot\mid x)}
\left[
\mathbb E(R\mid X=x,A)
\right]
$$

$$
\kappa_\pi(x)
=
\mathbb E_{A\sim\pi(\cdot\mid x)}
\left[
\mathbb E(C\mid X=x,A)
\right]
$$

로 둔다. 무작위 실험, 또는 “기록된 결과가 실제 실행 행동의 결과이고, 같은 $X$ 안에서는 행동 배정에 숨은 교란이 없으며, 평가할 모든 행동이 양의 확률로 실행되었다”는 consistency·조건부 교환가능성·overlap 가정이 성립하는 로그의 IPS·DR 추정으로 이 함수들을 식별해야 한다. 아래에서는 추정값 $\widehat v_\pi,\widehat\kappa_\pi$가 준비되었다고 가정한다.

오늘의 **context-mix DRO** 강건 하한과 상한은

$$
\underline V_{\rho,d}(\pi)
=
\inf_{Q_X:W_d(Q_X,\widehat P_{\mathrm{ref}}^X)\le\rho}
\mathbb E_{X\sim Q_X}
\left[
\widehat v_\pi(X)
\right]
$$

$$
\overline C_{\rho,d}(\pi)
=
\sup_{Q_X:W_d(Q_X,\widehat P_{\mathrm{ref}}^X)\le\rho}
\mathbb E_{X\sim Q_X}
\left[
\widehat\kappa_\pi(X)
\right]
$$

다. 좋은 정책은 보상 하한은 높고 비용 상한은 낮아야 한다. 이 식은 $Q_X$만 움직이므로 $P(R,C\mid X,A)$가 바뀌는 concept shift를 보호하지 않는다. 결과모형·오프정책 추정오차와 concept shift는 별도 통계 경계와 outcome detector에 넣는다.

### 3.2 이산 Wasserstein 거리를 운송 문제로 본다

metric $d$에 대한 1-Wasserstein 거리는

$$
W_d(P,Q)
=
\inf_{\gamma\in\Pi(P,Q)}
\mathbb E_{(X,X')\sim\gamma}
\left[
d(X,X')
\right]
$$

로 정의한다. $\Pi(P,Q)$는 첫 번째 주변분포가 $P$, 두 번째 주변분포가 $Q$인 모든 결합분포의 집합이다.

이산 그룹에서 기준 분포의 비율을 $p=(p_1,\dots,p_G)$, 후보 분포를 $q=(q_1,\dots,q_G)$라 하자. $D_{gh}$는 그룹 $g$의 질량 한 단위를 그룹 $h$로 옮기는 비용이다.

$$
W_D(p,q)
=
\min_{\Gamma\ge0}
\sum_{g=1}^{G}
\sum_{h=1}^{G}
D_{gh}\Gamma_{gh}
$$

제약은

$$
\sum_{h=1}^{G}\Gamma_{gh}=p_g
$$

$$
\sum_{g=1}^{G}\Gamma_{gh}=q_h
$$

다. $\Gamma_{gh}$가 실제 이동량이다. 행의 합은 출발 질량, 열의 합은 도착 질량을 보존한다. $D$가 대각선에서 $0$이고, $g\ne h$이면 $D_{gh}>0$이며, 비음수·대칭·삼각부등식을 만족하는 ground metric이면 이것은 Wasserstein distance다. 서로 다른 그룹인데도 $D_{gh}=0$을 허용하면 $D$와 $W_D$는 거리는 구분하지 못하는 pseudometric이 된다. 임의 비용표라면 같은 최적화는 가능하지만 정확한 이름은 optimal transport cost다.

두 그룹만 있는 예를 보자.

$$
p=(0.8,0.2)
$$

$$
L=(0.1,0.6)
$$

첫 번째 그룹의 손실은 $0.1$, 두 번째는 $0.6$이다. 현재 평균 손실은

$$
0.8\times0.1+0.2\times0.6=0.20
$$

이다. 그룹 사이 이동비용이 $1$이고 $\rho=0.1$이면 적대자는 첫 번째 그룹의 질량 $0.1$을 두 번째로 옮길 수 있다.

$$
q=(0.7,0.3)
$$

$$
q^\top L
=
0.7\times0.1+0.3\times0.6
=
0.25
$$

같은 $\rho$라도 이동비용을 $0.2$라고 정의하면 $0.5$의 질량까지 옮길 수 있어

$$
q=(0.3,0.7)
$$

$$
q^\top L=0.45
$$

가 된다. **반경만 같다고 같은 강건성 문제가 아니다. 거리와 반경은 한 쌍이다.**

### 3.3 거리의 전체 크기와 반경은 따로 식별되지 않는다

모든 이동비용에 양의 상수 $a$를 곱한 새 거리 $d'=ad$를 생각하자. 운송 목적함수 전체가 $a$배 되므로

$$
W_{d'}(P,Q)
=
aW_d(P,Q)
$$

다. 따라서

$$
W_{d'}(P,Q)\le a\rho
\quad\Longleftrightarrow\quad
W_d(P,Q)\le\rho
$$

이다. $d$를 두 배 키우고 $\rho$도 두 배 키우면 같은 ball이다. 그러므로 학습 거리의 scale을 고정하지 않고 “$\rho=0.3$이 크다”고 말하는 것은 의미가 없다.

실무에서는 다음 중 하나로 scale을 고정할 수 있다.

$$
\operatorname{median}_{(i,j)\in\mathcal S}
d_\phi(x_i,x_j)=1
$$

또는

$$
\operatorname{tr}(M)=p
$$

여기서 $\mathcal S$는 정상 상태의 비교 쌍, $p$는 특징 차원이다. 정규화 규칙도 버전의 일부여야 한다.

### 3.4 안정 블록의 거리로 반경을 검량한다

거리 $d_\phi$와 기준 context 분포 $\widehat P_{\mathrm{ref}}^X$를 먼저 고정한다. 기준을 만드는 데 사용하지 않은 $m$개의 안정 calibration block에 대해

$$
\Delta_b
=
W_{d_\phi}
\left(
\widehat P_b^X,
\widehat P_{\mathrm{ref}}^X
\right),
\qquad
b=1,\dots,m
$$

를 계산한다. $\Delta_b$는 “정상이라고 승인된 블록도 기준에서 이 정도는 흔들렸다”는 점수다.

목표 초과확률을 $0<\alpha<1$이라 하고

$$
k
=
\left\lceil
(m+1)(1-\alpha)
\right\rceil
$$

로 둔다. $\Delta_{(k)}$를 정렬된 $k$번째 점수라 하면

$$
\widehat\rho_\alpha
=
\Delta_{(k)}
$$

를 반경으로 선택한다. 단, $k=m+1$이면 유한한 $k$번째 calibration score가 없으므로 표준 split-conformal 관례에서는 $+\infty$를 둔다. 유한 반경이 가능하려면

$$
\alpha
\ge
\frac{1}{m+1}
$$

이어야 한다. 작은 표본으로 높은 coverage를 주장할 수 없다는 정직한 신호다.

예를 들어 $m=99$, $\alpha=0.05$이면

$$
k
=
\left\lceil100\times0.95\right\rceil
=
95
$$

이므로 95번째 거리로 반경을 잡는다.

### 3.5 왜 이 순위 반경이 coverage를 주는가?

기준 분포가 고정된 조건 아래 calibration score

$$
\Delta_1,\dots,\Delta_m
$$

와 다음 정상 블록의 score $\Delta_{m+1}$가 교환가능하다고 하자. 연속분포에서는 새 score의 순위가 $1,\dots,m+1$에서 대칭이다. 동률이 있으면 모든 동률 순서를 같은 확률로 정하는 무작위 tie-breaking으로 순위를 정의한다.

따라서

$$
\Pr
\left(
\operatorname{rank}(\Delta_{m+1})
\le k
\right)
 =
\frac{k}{m+1}
\ge
1-\alpha
$$

이다. 한편 결정론적 사건 $\Delta_{m+1}\le\Delta_{(k)}$는 이 무작위 순위 사건을 포함하므로 더 보수적이다. 따라서

$$
\Pr
\left(
\Delta_{m+1}
\le
\widehat\rho_\alpha
\right)
\ge
1-\alpha
$$

를 얻는다.

이 보장은 “미래의 모든 분포가 ball 안에 있다”는 뜻이 아니다. **같은 정상 체제에서 나온 다음 블록의 경험분포 거리**에 대한 주변 coverage다.

### 3.6 시간 데이터에서는 무엇이 깨질까?

시간 블록은 계절성, 캠페인, 릴리스, 반복 사용자 때문에 교환가능하지 않을 수 있다. 행을 무작위로 섞으면 같은 사용자의 요청이나 같은 사고의 파생 로그가 train과 calibration에 동시에 들어가 거리 분포가 지나치게 작아질 수 있다.

따라서 다음 원칙이 필요하다.

1. 독립에 가까운 사용자·세션·주 단위로 블록을 만든다.
2. 월요일과 주말처럼 구조가 다르면 계절 조건별 반경을 사전 정의한다.
3. 과거에서 미래로 이동하는 rolling-origin backtest로 초과율을 확인한다.
4. 블록 크기를 비슷하게 하거나 표본 크기에 따른 거리 변화를 따로 보정한다.
5. 이상 사건이 포함된 블록은 사후에 조용히 삭제하지 말고 제외 규칙과 근거를 기록한다.

경험 Wasserstein 거리에는 실제 분포 변화뿐 아니라 표본오차도 들어간다. 작은 블록일수록 같은 모집단에서도 더 멀어 보일 수 있으므로 블록 크기와 $\rho$는 함께 설계해야 한다.

### 3.7 DKW 경계로 분위수 추정 불확실성을 본다

안정 거리의 누적분포함수를 $F$, 경험 누적분포함수를 $\widehat F_m$이라 하자. Dvoretzky–Kiefer–Wolfowitz 부등식은 IID 조건에서

$$
\Pr
\left(
\sup_x
\left|
\widehat F_m(x)-F(x)
\right|
>
\varepsilon
\right)
\le
2e^{-2m\varepsilon^2}
$$

를 준다. 실패확률을 $\delta$로 두면

$$
\varepsilon_{m,\delta}
=
\sqrt{
\frac{\log(2/\delta)}
{2m}
}
$$

이다.

reference와 metric을 독립적으로 고정한 조건 아래 $\Delta_1,\dots,\Delta_m$이 IID라고 하자. 경험 분위수 함수를 $\widehat F_m^{-1}$로 쓰면

$$
\widehat q
=
\widehat F_m^{-1}
\left(
1-\alpha+\varepsilon_{m,\delta}
\right)
$$

에 대해

$$
\Pr_{\mathrm{cal}}
\left\{
F(\widehat q)
\ge
1-\alpha
\right\}
\ge
1-\delta
$$

를 얻는다. 이는 calibration 표본을 다시 뽑았을 때의 신뢰도 $1-\delta$와, 그 event 안에서 다음 거리의 포함확률 $1-\alpha$를 구분한다.

만약

$$
1-\alpha+\varepsilon_{m,\delta}>1
$$

이면 이 DKW 방식으로 유한 경험분위수를 인증할 수 없다. 이때는 $+\infty$, 미리 알려진 거리 상한, 더 많은 독립 블록 중 하나가 필요하다. 다만 DKW는 IID를 요구하므로, 의존 시간자료에서는 블록화·혼합 조건·전용 시계열 방법이 필요하다. conformal 순위 반경과 DKW 보정은 서로 다른 가정과 목적을 가진 도구이지 자동으로 중첩해 쓰는 주문이 아니다.

### 3.8 반경 선택 자체도 검증 데이터에 과적합될 수 있다

후보 반경을

$$
\rho_1<\rho_2<\cdots<\rho_J
$$

로 두고 같은 validation set에서 가장 높은 성능을 내는 $\rho_j$를 고르면 그 성능은 낙관적으로 보인다. $\rho$는 일반 hyperparameter와 마찬가지다.

안전한 분리는 다음과 같다.

- metric-train: 거리 $d_\phi$ 학습
- radius-calibration: $\rho$ 검량
- policy-selection: 후보 정책·반경 조합 선택
- final-gate: 선택이 끝난 한 조합을 최종 평가

데이터가 부족하면 사전 등록한 유한 후보군 전체에 동시 경계를 적용할 수 있지만, 같은 데이터로 무한히 새 후보를 발명하면 그 보정도 무너진다.

### 3.9 그룹 정의 선택 편향을 수식으로 본다

그룹 규칙 후보를

$$
\mathcal G
=
\left\{
g_{\eta_1},\dots,g_{\eta_M}
\right\}
$$

라 하자. 한 그룹 map은 여러 slice를 만들므로 실제 보고 가설의 가족을

$$
\mathcal H
=
\left\{
(\eta,s):
g_\eta\in\mathcal G,
\quad
s\in\operatorname{range}(g_\eta)
\right\}
$$

$$
H
=
|\mathcal H|
$$

로 둔다. 손실이 $[0,1]$ 범위이고 slice $h\in\mathcal H$의 양의 표본 수가 $n_h>0$, 경험 평균이 $\widehat R_h$라고 하자. 사전 고정된 그룹과 IID 관측을 가정하고 표본 수에 조건부로 보면, 고정된 한 slice에 Hoeffding 상한을 써서

$$
\Pr
\left(
R_h
>
\widehat R_h+\epsilon
\right)
\le
e^{-2n_h\epsilon^2}
$$

를 얻는다. 사전 정의한 $H$개 slice 모두에 대해 실패확률 합이 $\delta$ 이하가 되게

$$
\epsilon_h
=
\sqrt{
\frac{\log(H/\delta)}
{2n_h}
}
$$

로 두면 union bound로

$$
\Pr
\left(
\forall h\in\mathcal H:
R_h
\le
\widehat R_h+\epsilon_h
\right)
\ge
1-\delta
$$

를 얻는다. 후보 간 독립성은 필요하지 않지만, 후보군이 데이터를 보기 전에 고정된 유한 집합이어야 한다.

작은 그룹은 $n_h$가 작아 상한이 넓어진다. 이는 귀찮은 벌점이 아니라 “이 그룹에 대해 모른다”는 올바른 표현이다. 그룹을 발견하는 데이터와 최종 추정 데이터를 분리하면 더 유연한 탐색도 가능하다.

### 3.10 학습된 Wasserstein ground metric

임베딩 $h(x)\in\mathbb R^p$를 사용해

$$
d_\phi(x,x')
=
\left\|
L_\phi
\left(
h(x)-h(x')
\right)
\right\|_2
$$

로 정의할 수 있다. 이를 제곱하면

$$
d_\phi^2(x,x')
=
\left(
h(x)-h(x')
\right)^\top
M_\phi
\left(
h(x)-h(x')
\right)
$$

$$
M_\phi
=
L_\phi^\top L_\phi
\succeq0
$$

이다. $M_\phi\succeq0$이므로 제곱값은 음수가 아니다. metric으로 쓰는 것은 $d_\phi$이고, $d_\phi^2$는 일반적으로 삼각부등식을 만족하지 않는 제곱비용이다.

$L_\phi$가 일부 방향을 완전히 지우거나 embedding이 충돌해 $h(x)=h(x')$가 되면, 서로 다른 두 입력의 $d_\phi$가 $0$일 수 있다. 이때 $d_\phi$와 그 위의 $W_{d_\phi}$는 엄밀히 pseudometric이다. 이는 모델이 그 차이를 중요하지 않다고 선언한 결과지만, 비용 $0$인 동치류 안에서 $v_\pi(x)$나 $\kappa_\pi(x)$가 크게 다르면 적대자가 위험한 context로 공짜 이동할 수 있어 강건 목적이 퇴화한다. 안전에 중요한 방향을 지우지 않았는지 별도로 확인해야 한다.

### 3.11 비슷한 쌍과 다른 쌍으로 거리를 학습한다

비슷해야 하는 쌍 집합을 $\mathcal S$, 달라야 하는 쌍 집합을 $\mathcal D$라고 하자. margin을 $\gamma>0$로 두면 contrastive loss의 한 형태는

$$
\mathcal L_{\mathrm{metric}}(\phi)
=
\sum_{(i,j)\in\mathcal S}
d_\phi^2(x_i,x_j)
+
\sum_{(i,k)\in\mathcal D}
\left[
\gamma-d_\phi(x_i,x_k)
\right]_+^2
$$

이다. 여기서

$$
[u]_+
=
\max(u,0)
$$

이다.

- 첫 합은 비슷한 쌍을 가깝게 만든다.
- 둘째 합은 다른 쌍이 margin보다 가까울 때만 벌점을 준다.

하지만 무엇을 비슷하다고 정했는지가 핵심이다. 보상 결과를 본 뒤 “같은 결과면 비슷하다”고 만들면 action 이후 정보를 거리 입력에 섞을 수 있다. 배포 때 사용할 거리는 원칙적으로 행동 전에 관측 가능한 특징으로 계산되어야 하며, outcome-aware 보조 목적을 썼다면 독립 블록에서 정책 가치와 안전성을 다시 검증해야 한다.

### 3.12 같은 데이터로 거리와 반경을 학습하면 왜 위험할까?

거리 학습기가 calibration block들이 기준과 가깝게 보이도록 직접 최적화할 수 있다고 하자.

$$
\widehat\phi
\in
\arg\min_\phi
\sum_{b=1}^{m}
W_{d_\phi}
\left(
\widehat P_b^X,
\widehat P_{\mathrm{ref}}^X
\right)
$$

그 뒤 같은 점수들의 $95\%$ 분위수를 $\widehat\rho$로 잡으면 당연히 작게 나온다. 이는 미래 coverage를 보여 준 것이 아니라 calibration set을 압축한 결과일 수 있다.

따라서

$$
\mathcal D_{\mathrm{metric}}
\cap
\mathcal D_{\mathrm{radius}}
=
\varnothing
$$

가 되도록 독립 분할하는 것이 가장 명확하다. 교차적합을 사용할 수도 있지만, 각 fold에서 metric 학습과 score 계산 순서를 지키고 시간 순서를 보존해야 한다.

### 3.13 Wasserstein 쌍대식은 거리와 반경의 역할을 드러낸다

context 표본 $x_1,\dots,x_n$의 경험분포를

$$
\widehat P_n^X
=
\frac1n
\sum_{i=1}^{n}\delta_{x_i}
$$

$\delta_{x_i}$는 점 $x_i$에 질량 $1$을 놓는 Dirac measure다. $d$가 앞서 정의한 1-Wasserstein ground metric이고 비용이 하반연속이며, 손실 $\ell$이 상반연속이면서 유계이거나 어떤 기준점 $x_0$와 상수 $a,b\ge0$에 대해 $\ell(x)\le a+b\,d(x,x_0)$인 선형 성장 조건을 만족하는 등 표준 정칙성 조건 아래, 주변의 최악 기대손실은 유한하며 다음 쌍대형으로 쓸 수 있다. 초선형으로 커지는 손실은 $\rho>0$이어도 최악값을 무한대로 만들 수 있다.

$$
\sup_{Q_X:W_d(Q_X,\widehat P_n^X)\le\rho}
\mathbb E_{X\sim Q_X}[\ell(X)]
=
\inf_{\lambda\ge0}
\left\{
\lambda\rho
+
\frac1n
\sum_{i=1}^{n}
\sup_{x'}
\left[
\ell(x')
-\lambda d(x',x_i)
\right]
\right\}
$$

$\lambda$는 운송 예산 한 단위의 그림자가격이다.

- $\rho$가 커지면 $\lambda\rho$ 항을 통해 더 큰 이동을 허용한다.
- $d(x',x_i)$가 작은 방향은 적대자가 싸게 이동한다.
- 손실이 크게 증가하면서 거리가 싼 $x'$가 최악 사례가 된다.

그래서 metric learning은 단순 검색 품질 문제가 아니다. 어떤 실패 방향을 DRO가 “그럴듯한 공격”으로 간주할지를 학습하는 일이다.

### 3.14 변화점 경보의 anytime 의미

이 절은 앞서 배운 e-process와 Ville 부등식을 오늘의 반복 drift 감시에 연결하는 복습이다. 정상 체제를 복합 귀무가설 $H_0$라 하자. $E_t$가 $\mathcal F_t$에 맞추어진, 즉 미래가 아니라 시점 $t$까지의 정보만 사용하는 adapted 비음수 과정이고 $E_0\le1$이며, 모든 $P\in H_0$에서

$$
\mathbb E_{P}[E_t\mid\mathcal F_{t-1}]
\le
E_{t-1}
$$

를 만족한다고 하자. $\mathcal F_{t-1}$는 시점 $t-1$까지의 정보다. Ville 부등식에 의해

$$
\sup_{P\in H_0}
\Pr_P
\left(
\sup_{t\ge1}E_t
\ge
\frac1\alpha
\right)
\le
\alpha
$$

이므로

$$
\tau
=
\inf
\left\{
t:
E_t\ge\frac1\alpha
\right\}
$$

를 경보 시점으로 둘 수 있다.

이는 매일 대시보드를 보다가 마음에 들 때 멈추는 반복 엿보기에도 전체 오경보확률을 통제하는 한 방법이다. 다만 detector의 입력과 조건이 올바르게 설계되었을 때의 이야기다. 어떤 e-process도 관측하지 않은 안전 실패를 자동 탐지하지는 못한다.

### 3.15 경보와 Wasserstein 반경의 역할을 분리한다

현재 블록의 거리

$$
D_t
=
W_{d_{\phi_v}}
\left(
\widehat P_t^X,
\widehat P_{\mathrm{ref}}^{X,(v)}
\right)
$$

를 계산하자. 버전 $v$의 정상 반경이 $\rho_v$일 때

$$
D_t\le\rho_v
$$

는 “정상 검량 범위 안”이라는 신호이고

$$
D_t>\rho_v
$$

는 조사 대상이다. 그러나 한 번의 초과만으로 구조적 변화라고 단정해서도 안 된다. 작은 블록, 데이터 지연, 일회성 이벤트가 원인일 수 있다.

$\widehat\rho_\alpha$의 보장은 다음 **한 블록**에 대한 주변 보장이다. 검량 데이터와 그로부터 계산한 임계값을 $\mathcal C$라 하고, 실현된 임계값의 조건부 초과확률을

$$
p_{\mathcal C}
=
\Pr
\left(
D_t>\widehat\rho_\alpha
\mid
\mathcal C
\right)
$$

라 하자. 미래 정상 블록들이 $\mathcal C$에 조건부로 독립·동일분포라면 한 번 이상 초과할 조건부 확률은 $1-(1-p_{\mathcal C})^T$다. split conformal이 보장하는 것은 $\mathbb E_{\mathcal C}[p_{\mathcal C}]\le\alpha$라는 주변 보장이지, 모든 실현된 검량 표본에서 $p_{\mathcal C}\le\alpha$라는 조건부 보장이 아니다. 따라서 $p_{\mathcal C}$를 곧바로 $\alpha$로 바꿔 $1-(1-\alpha)^T$라고 쓸 수 없다. 독립성을 가정하지 않아도 각 시점의 주변 보장과 union bound로

$$
\Pr
\left(
\exists t\le T:
D_t>\rho_v
\right)
\le
\min(1,T\alpha)
$$

라는 거친 상한만 얻는다. 따라서 고정 분위수를 매일 반복 확인하는 것은 anytime change detector가 아니다. 한 번의 초과는 hold·조사 신호로 사용하되, 변화점 주장은 3.14절의 e-process, confidence sequence 또는 사전 정의한 alpha-spending 규칙으로 통제한다.

안전 설계는 두 층으로 나눈다.

1. **신속 보호층:** 초과·성능 경보·안전 비용 위반이 있으면 promotion을 멈추고 고위험 트래픽을 현재 support에서도 검증된 $v_{\mathrm{safe}}$로 보낸다. 그런 정책이 없으면 기능 제한·fail-closed·사람 검토 같은 fail-safe 경로를 사용한다.
2. **통계 확인층:** 순차 detector, 지연 라벨, 운영 사건 기록을 함께 보고 새로운 체제인지 판정한다.

반경은 정상 변동의 경계이고 detector는 그 경계를 넘는 패턴의 지속성과 증거를 판단한다.

### 3.16 새 체제로 적응할 때의 올바른 순서

경보 뒤 새 버전 $v+1$을 만드는 데이터 흐름을 다음처럼 분리한다.

$$
\mathcal D_{\mathrm{investigate}}
\rightarrow
\mathcal D_{\mathrm{metric}}
\rightarrow
\mathcal D_{\mathrm{radius}}
\rightarrow
\mathcal D_{\mathrm{select}}
\rightarrow
\mathcal D_{\mathrm{gate}}
$$

- $\mathcal D_{\mathrm{investigate}}$: 변화 원인과 데이터 오류를 찾는다.
- $\mathcal D_{\mathrm{metric}}$: 기존 거리의 의미가 유지되는지 감사하고, 근거가 있을 때만 $d_{\phi_{v+1}}$를 다시 학습한다.
- $\mathcal D_{\mathrm{radius}}$: 새 정상 반경 $\rho_{v+1}$을 검량한다.
- $\mathcal D_{\mathrm{select}}$: challenger를 고른다.
- $\mathcal D_{\mathrm{gate}}$: 선택이 끝난 뒤 안전 조건을 최종 확인한다.

최종 gate에서는 같은 후보분포 아래 정책과 기준 정책의 차이를 비교하는 paired robust improvement를

$$
\underline\Delta_{\mathrm{DRO}}(\pi,\pi_0)
=
\inf_{Q_X\in\mathcal P_{\rho}^{d}}
\mathbb E_{X\sim Q_X}
\left[
\widehat v_\pi(X)
-
\widehat v_{\pi_0}(X)
\right]
$$

로 정의한다. 두 정책에 서로 다른 최악분포를 적용한 값을 따로 계산해 빼면 paired 차이의 의미가 깨질 수 있다. 이 DRO 값만으로 통계적 신뢰구간이 자동 생기는 것도 아니다. radius coverage, 블록 내부 표본오차, 결과모형·IPS·DR 추정오차를 함께 반영한 최종 LCB·UCB를 만든다.

$$
\operatorname{LCB}_{\mathrm{rob}}
\left(
\underline\Delta_{\mathrm{DRO}}(\pi,\pi_0)
\right)
\ge
-\varepsilon_V
$$

$$
\operatorname{UCB}_{\mathrm{rob}}
\left(
\overline C_{\rho,d}(\pi)
\right)
\le
c_{\max}
$$

$\varepsilon_V$는 허용 가능한 최대 가치 저하, $c_{\max}$는 안전 비용 한도다. 이 둘을 만족해도 canary에서 같은 규칙을 다시 감시한다.

### 3.17 오늘 수학과 AI 부품의 1:1 대응

| 수학 부품 | AI 시스템 부품 | 실패하면 생기는 일 |
| --- | --- | --- |
| $d_\phi$ | context·언어·modality 변화의 의미 체계 | 현실적으로 가까운 실패를 놓치거나 엉뚱한 이동에 과잉 대비한다. |
| scale 정규화 | metric 버전 비교 | $\rho$ 숫자의 크기를 잘못 비교한다. |
| $\widehat\rho_\alpha$ | 정상 변동 허용선 | 너무 잦은 경보 또는 위험한 과소 방어가 생긴다. |
| block exchangeability | 시간 분할 설계 | nominal coverage가 실제 운영에서 무너진다. |
| union bound·독립 test | post-hoc slice 탐색 보정 | 우연히 나쁜 그룹을 확정적 위험으로 착각한다. |
| e-process·stopping time | 24시간 변화점 감시 | 반복 관찰로 오경보가 누적된다. |
| robust LCB·UCB | 재배포 승인 조건 | 평균 점수는 좋아도 최악 위험이 커질 수 있다. |
| 버전 $v$ | 모델·전처리·prompt·reference·metric·반경·group·detector·안전 fallback·runtime 묶음 | 검증하지 않은 혼합 구성이 배포된다. |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 Online DRO란 무엇인가?

**Online**은 데이터가 한 번에 모두 주어지는 것이 아니라 시간 순서로 도착한다는 뜻이다. Online DRO는 새 블록이 올 때 기준 분포, 그룹별 가치, 불확실성 추정을 갱신해 변화에 대응한다.

그러나 “매 순간 최신 데이터로 다시 최적화”와 “안전하게 온라인 적응”은 다르다. 안전한 online DRO에는 두 속도가 필요하다.

- 빠른 제어 루프: 경보, 트래픽 축소, hold, rollback
- 느린 학습 루프: 원인 조사, 데이터 정제, metric·radius 재검량, challenger 선택, 재승인

빠른 루프가 피해를 제한하고, 느린 루프가 통계적 정당성을 복구한다.

### 4.2 운영 상태 기계

상태를 다음처럼 정의할 수 있다.

| 상태 | 실제 트래픽 | 허용되는 일 | 다음 상태 조건 |
| --- | --- | --- | --- |
| NORMAL | champion $v$가 받음 | 감시·로그·정기 shadow | 경보 시 HOLD |
| HOLD | 신규 promotion 중지 | 데이터 무결성 점검·고위험 수동 검토 | 중대 위반이면 ROLLBACK, 오경보면 NORMAL |
| ROLLBACK | 현재 체제에서도 안전하다고 검증된 $v_{\mathrm{safe}}$ | 영향 범위 차단·사고 분석 | 새 후보가 준비되면 SHADOW |
| SHADOW | champion이 행동, challenger는 기록만 | 새 metric·radius·정책 평가 | gate 통과 시 CANARY |
| CANARY | 작은 무작위 비율만 challenger | propensity·성과·비용 감시 | 통과 시 NORMAL의 새 버전, challenger gate 위반 시 canary 중단·안전한 champion 복귀, 시스템 전체나 champion 위반 시 ROLLBACK·fail-safe |

상태 전이는 사람이 읽을 수 있는 규칙으로 사전에 정의해야 한다. $v_{\mathrm{safe}}$가 새 support에서도 안전하다는 근거가 없으면 단순히 직전 버전 $v-1$로 보내지 않고, 기능 제한·fail-closed·사람 검토 같은 fail-safe로 전환한다. 여기서 fail-closed는 안전을 입증할 수 없을 때 요청을 거부하거나 보류하는 fail-safe의 한 구현이다. “담당자가 괜찮아 보이면 전면 배포”는 재현 가능한 안전 절차가 아니다.

### 4.3 경보가 울렸을 때의 최소 대응 순서

1. 경보 시간, detector 버전, 입력 schema, 모델 버전을 고정 기록한다.
2. 데이터 파이프라인 오류인지 실제 분포 변화인지 분리한다.
3. 신규 promotion과 자동 재학습을 멈춘다.
4. 안전 비용이 즉시 증가했거나 고위험 support 밖이면 검증된 $v_{\mathrm{safe}}$로 rollback하고, 그런 fallback이 없으면 fail-closed·사람 검토 같은 fail-safe로 보낸다.
5. 지연 라벨이 도착할 때까지 proxy 경보와 실제 outcome을 구분해 표시한다.
6. 새 체제가 지속되면 새로운 reference·radius 버전을 별도 데이터로 만들고, 기존 metric의 의미가 깨졌다는 근거가 있을 때만 metric도 다시 학습한다.
7. shadow와 무작위 canary로 독립 gate를 통과한 뒤 재진입한다.

### 4.4 왜 현재 블록을 즉시 reference에 섞지 않을까?

rolling reference를 다음처럼 쓸 수 있다. 여기서 $0\le\beta\le1$은 새 블록에 주는 가중치다. $\beta=0$이면 기준을 전혀 갱신하지 않고, $\beta=1$이면 기준을 새 블록으로 완전히 바꾼다.

$$
\widehat P_{\mathrm{ref},t}^{X}
=
(1-\beta)
\widehat P_{\mathrm{ref},t-1}^{X}
+
\beta\widehat P_t^X
$$

로 갱신하면 적응은 빠르다. 하지만 사고 블록도 곧 정상 기준에 흡수된다. 이를 **baseline poisoning**이라고 생각할 수 있다.

따라서 새 블록에는 먼저 상태 라벨이 필요하다.

- accepted stable: 지연 라벨과 검토를 통과해 reference 후보가 됨
- quarantine: 경보 또는 데이터 품질 문제로 격리
- incident: 명확한 실패 사례로 reference에서 제외하되 stress test에는 보존

reference 갱신은 accepted stable 블록만 사용하고, 어떤 규칙으로 승인했는지 기록해야 한다. reference가 바뀌면 모든 거리 score와 정상 귀무가설도 바뀐다. 따라서 매 갱신을 새 버전으로 만들고, 독립 안정 블록에서 $\rho$를 다시 검량하며, detector 상태를 새 epoch에서 재시작·재검증한다. 다만 오경보 예산을 버전마다 무제한으로 새로 주면 전체 수명 동안의 오류율이 커진다. 보장이 버전별임을 명시하거나, 버전 $v$에 $\alpha_v$를 배정해 $\sum_v\alpha_v\le\alpha_{\mathrm{total}}$을 지키거나, epoch를 가로지르는 anytime-valid 절차를 사용해야 한다.

### 4.5 covariate detector와 outcome detector를 함께 쓴다

즉시 관측 가능한 score를

$$
D_t^{X}
=
W_{d_\phi}
\left(
\widehat P_t^X,
\widehat P_{\mathrm{ref}}^X
\right)
$$

라 하자. 지연 성능 score는 raw 평균 차이가 아니라 비교 대상을 맞춘 값이어야 한다. 고정된 monitored policy $\pi_{\mathrm{mon}}$을 공통 context mix에 표준화하고, 성숙한 라벨만 사용한 가치 추정치를 $\widehat V_t^{\mathrm{std}}(\pi_{\mathrm{mon}})$라 쓰면

$$
D_t^{R}
=
\widehat V_{\mathrm{ref}}^{\mathrm{std}}
\left(
\pi_{\mathrm{mon}}
\right)
-
\widehat V_t^{\mathrm{std}}
\left(
\pi_{\mathrm{mon}}
\right)
$$

또는 같은 방식의 안전 비용 증가량으로 만들 수 있다. 동일 정책 로그가 없으면 무작위 canary나 propensity를 기록한 IPS·DR 추정으로 action mix를 맞춘다. context mix, 정책 버전, 라벨 성숙도가 다른 raw 평균끼리 빼면 concept shift와 단순 구성 변화를 구분할 수 없다.

운영 규칙의 예는 다음과 같다.

- $D_t^X$만 상승: hold·조사·고위험 slice 강화, 성능 실패로 단정하지 않음
- $D_t^R$만 상승: concept shift, 라벨 파이프라인, 정책 실행 오류 조사
- 둘 다 상승: rollback 우선순위를 높임
- 둘 다 안정: 경보 부재일 뿐 미관측 위험이 없다는 증명은 아님

$X$, outcome, 여러 slice detector를 동시에 OR로 묶으면 전체 오경보율이 커진다. 사전 정의한 alpha 배분, 결합 e-process 또는 계층적 경보 규칙으로 가족 전체의 오류 예산을 관리해야 한다.

### 4.6 그룹 정의를 운영 계약으로 만든다

그룹은 보고서 장식이 아니라 안전 계약이다. 각 그룹 규칙에는 다음 메타데이터가 필요하다.

- 사용 특징과 시점: action 이전에 관측 가능한가?
- 결측값 처리: “unknown”을 버리지 않고 별도 그룹으로 남기는가?
- 최소 표본 수와 작은 그룹의 pooling 규칙
- 교차 그룹: 예를 들어 언어 $\times$ 위험 도메인
- 생성 날짜, 코드 hash, 승인자, 폐기 날짜
- 변경 시 과거 대시보드와의 비교 방법

새 그룹을 사고 뒤 발견하는 것은 가치가 있다. 다만 그 그룹은 **탐색적 발견**으로 표시하고, 다음 독립 기간에서 위험을 확인하기 전까지 확정된 빈도 추정치처럼 말하지 않는다.

### 4.7 학습 거리를 위한 세 가지 방어선

1. **의미 방어선:** 도메인 전문가가 must-link와 cannot-link 쌍을 검토한다.
2. **통계 방어선:** metric-train과 radius-calibration을 분리한다.
3. **운영 방어선:** 배포 후 거리 분포 자체와 nearest-neighbor 사례를 감시한다.

특히 embedding model이 업데이트되면 같은 입력도 좌표가 달라진다. 따라서

$$
\phi_v
\neq
\phi_{v+1}
$$

이면 과거 $\rho_v$를 그대로 재사용해서는 안 된다. metric이 바뀌면 distance score 전체를 새로 검량해야 한다.

### 4.8 rollback 대상은 완전한 배포 묶음이다

다음 항목을 하나의 immutable manifest로 관리한다.

- model checkpoint와 tokenizer
- prompt·system policy
- retrieval corpus와 index revision
- tool 목록과 권한
- preprocessing·feature schema
- reference snapshot·source block ID·승인 규칙
- group map $g_\eta$
- embedding·metric $d_\phi$와 metric-train data ID
- radius $\rho$와 radius-calibration data ID
- detector 설정·epoch·threshold·남은 alpha budget
- $v_{\mathrm{safe}}$ 또는 fail-safe routing 규칙
- serving runtime·quantization·sampling 설정
- 데이터·평가 코드 revision

quantization이나 runtime kernel 변경도 출력과 latency를 바꿀 수 있으므로 “같은 모델”이라고 자동 간주하지 않는다.

### 4.9 label delay가 있을 때 canary를 어떻게 할까?

즉시 proxy만 좋아졌다고 canary를 확대하면 늦게 도착한 사고 라벨을 보기 전에 전면 노출될 수 있다. 따라서 노출 상한을 라벨 성숙도와 연결하되, 성숙 비율 하나만 통과 조건으로 쓰지 않는다.

단계 $k$의 노출률을 $e_k$, 최소 라벨 성숙 비율을 $q_k$, 최소 성숙 표본 수를 $n_k^{\min}$이라 하자. 확대 조건은 다음 조건의 교집합이다.

$$
\frac{\text{matured labels}}
{\text{eligible canary events}}
\ge
q_k
$$

$$
n_{\mathrm{mature},k}
\ge
n_k^{\min}
$$

$$
\operatorname{LCB}_{\mathrm{rob}}
\left(
\underline\Delta_{\mathrm{DRO}}
\right)
\ge
-\varepsilon_V
$$

$$
\operatorname{UCB}_{\mathrm{rob}}
\left(
\overline C_{\rho,d}(\pi)
\right)
\le
c_{\max}
$$

여기에 활성 변화점 경보가 없어야 비로소 $e_{k+1}>e_k$를 허용한다. canary를 여러 번 중간 점검한다면 LCB·UCB도 confidence sequence나 alpha-spending처럼 anytime-valid해야 한다. 결과에 따라 라벨 도착 속도가 달라지는 경우에는 고정 판정시점이나 censoring 보정을 사전 정의한다.

challenger의 gate가 깨지면 현재 노출을 즉시

$$
e(t^+)=0
$$

으로 만들어 canary를 중단하고 안전한 champion으로 복귀한다. champion 자체가 새 체제에서 안전하지 않다면 이는 단순 challenger 중단이 아니라 $v_{\mathrm{safe}}$ 또는 fail-safe로 가는 시스템 rollback이다. 정확한 수치는 서비스 위험과 라벨 지연 분포에 맞게 사전 정의한다.

### 4.10 초보자가 흔히 하는 오해와 교정

| 오해 | 왜 틀렸는가 | 올바른 관점 |
| --- | --- | --- |
| $\rho$가 클수록 무조건 더 안전하다. | 지정한 ball 안에서는 보수적이지만 정책이 무용해질 수 있고 ball 밖 구조 변화는 여전히 놓친다. | coverage·utility·stress 결과를 함께 본다. |
| Wasserstein이면 자동으로 의미적 거리다. | ground metric이 픽셀·임베딩·수작업 비용 중 무엇인지에 따라 전혀 다른 이동을 허용한다. | $d$의 의미와 버전을 검증한다. |
| 가장 나쁜 slice를 찾았으니 그 위험률이 정확하다. | 많은 slice 중 극단값을 골랐으므로 선택 편향이 있다. | 발견과 추정을 분리하거나 동시 경계를 쓴다. |
| 경보가 울리면 모델이 나빠졌다는 뜻이다. | 데이터 오류·일회성 이벤트·표본오차일 수 있다. | hold 후 원인과 outcome을 확인한다. |
| 경보가 울리면 $\rho$를 키우면 된다. | 기존 검량을 무효화하고 이상을 정상에 흡수할 수 있다. | 먼저 보호하고 새 체제를 별도 검량한다. |
| embedding을 업데이트해도 $\rho$는 재사용할 수 있다. | 거리 단위와 방향이 달라졌다. | metric 변경마다 score와 반경을 재검량한다. |
| rollback은 checkpoint만 되돌리면 된다. | prompt·도구·index·runtime 조합이 행동을 바꾼다. | 전체 manifest를 원자적으로 복원한다. |
| 라벨이 없어도 입력 drift만으로 안전 실패를 증명한다. | $P(X)$ 변화와 $P(Y\mid X)$ 변화는 다르다. | proxy와 지연 outcome을 분리한다. |

### 4.11 최소 구현 체크리스트

- [ ] metric-train, radius-calibration, policy-selection, final-gate의 시간 구간이 분리되어 있는가?
- [ ] 거리 scale 정규화가 문서화되어 있는가?
- [ ] 그룹 후보군과 탐색적 그룹을 구분했는가?
- [ ] calibration score의 블록 단위가 실제 독립 단위와 맞는가?
- [ ] 새 블록을 reference에 넣기 전 quarantine 절차가 있는가?
- [ ] covariate와 outcome detector를 별도로 운영하는가?
- [ ] 경보 후 자동 재학습·promotion을 멈추는가?
- [ ] fallback이 실제로 호출 가능한지 정기적으로 연습하는가?
- [ ] model·전처리·prompt·reference·metric·radius·group·detector·안전 fallback·runtime을 한 버전으로 복원할 수 있는가?
- [ ] canary 로그에 실제 노출확률과 지연 라벨 상태가 남는가?
- [ ] re-entry gate가 선택 데이터와 독립적인가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 기준 시각: **2026-10-20 KST**. 아래 내용은 공급사 공식 발표, 공식 모델 카드·라이선스, 기술 보고서와 공개 저장소를 우선해 확인했다. 공급사 벤치마크는 독립 재현 결과가 아니며, “가중치 공개”와 “제약 없는 오픈소스”를 구분한다.

### 5.1 Meta Muse Spark Open: 대형 멀티모달 MoE의 가중치 공개

Meta는 2026년 10월 15일 **Muse Spark Open**을 공개하고 이를 현재까지 가장 유능한 미국산 open-weight 모델이라고 소개했다. 이는 공급사의 표현이며 독립 판정은 아니다. Hugging Face checkpoint는 gated access로, 로그인 뒤 라이선스 동의와 신원·조직 정보 제출을 요구한다. 공식 모델 카드 기준 핵심 사양은 다음과 같다.

- 전체 파라미터 $816$B, 토큰당 활성 파라미터 약 $42$B
- 각 MoE layer의 $256$개 expert 중 token마다 top-$8$ expert를 선택하는 sparse Mixture-of-Experts
- text·image·audio·video 입력, text 출력
- 최대 $1$M-token context
- BF16과 NVFP4 checkpoint
- SGLang, vLLM, Transformers, TensorRT-LLM, llama.cpp, MLX 등 지원

MoE를 초보자 비유로 보면 $816$B명의 직원 전체가 매 단어마다 일하는 조직이 아니다. 각 MoE layer의 라우터가 token마다 맞는 $8$개 전문팀을 고른다. 토큰당 계산은 전체 크기보다 작아지지만, **전체 expert 가중치의 저장 비용은 남고, expert parallel 배포에서는 라우팅된 token activation을 해당 expert가 있는 장치로 보내는 통신 비용이 생긴다.**

공식 카드가 적은 자체 호스팅 요구량은 NVFP4도 최소 약 $700$GB VRAM, BF16은 약 $2$TB다. Hugging Face의 실행 가이드도 3-bit 양자화가 $256$GB가 넘는 RAM을 요구한다고 적는다. “데스크톱 실행”은 일반 노트북 한 대가 아니라, 대용량 통합 메모리를 가진 고가 장비나 여러 장비를 전제로 읽어야 한다.

지원 runtime 목록이 모든 modality와 context 길이를 같은 수준으로 지원한다는 뜻도 아니다. 공개 당시 공식 가이드의 SGLang 경로는 네 입력 modality를 다루지만, vLLM 참조 통합 예시는 text·image와 $128$K 설정을 안내한다. $1$M endpoint 예시는 $8$장의 H200을 사용한다. 도입 전에는 선택한 runtime의 실제 modality·context 조합을 따로 확인해야 한다.

라이선스도 중요하다. 이는 OSI 의미의 일반 오픈소스가 아니라 **Muse Model License를 적용한 open-weight**다.

- 각 제품·서비스의 UI에 눈에 띄게 표시하거나, 동반 문서·웹사이트·설정 메뉴에 “Built with Muse Spark Open”을 표시해야 하며 파생 모델에도 적용된다.
- 라이선시와 계열사의 합산 총매출이 임의의 연속 $12$개월 동안 미화 $20$M을 초과하면, 제3자 Model-as-a-Service에 Software 또는 Derivatives를 제공하기 전에 Meta에 통지하고 별도 서면 계약을 체결해야 한다.
- 사용 정책과 금지 용도가 결합되어 있다.

공식 평가표는 여러 agentic·coding·multimodal 점수를 제시하지만, 모델마다 reasoning effort, harness, 공급사 보고 여부가 다르다. 최대 $1$M context도 “어떤 위치의 정보를 얼마나 정확히 사용하는가”를 자동 보장하지 않는다.

**엔지니어 인사이트 (Impact):** 멀티모달 agent의 개방 경쟁이 작은 연구 모델을 넘어 대형 MoE까지 확장되었다는 점은 크다. 하지만 실제 도입 가능성은 parameter count보다 메모리, expert parallelism, 라이선스, tool 권한, 장문·다중모달 drift 감시에 좌우된다. 오늘의 관점에서는 text·image·audio·video 품질을 하나의 거리로 뭉개지 말고 modality별 metric과 교차 그룹을 사전 정의해야 한다. runtime·quantization 변경도 새 버전으로 보고, 장시간 agent가 새로운 tool·화면 상태를 만났을 때 변화점 경보와 rollback이 작동하는지 독립 canary에서 확인해야 한다.

- 공식 발표: [Introducing Muse Spark Open](https://research.meta.ai/blog/muse-spark-open)
- 공식 모델 카드·라이선스: [Muse-Spark-Open-816B-A42B](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
- 실행·튜닝 가이드: [Welcome Muse Spark Open](https://huggingface.co/blog/meta-muse-spark-open)

### 5.2 NVIDIA·AI4Bharat: 2.83T Indic 합성 토큰보다 중요한 공개 생성 공정

NVIDIA와 IIT Madras의 AI4Bharat는 2026년 10월 16일 **Nemotron-pretraining-Indic-v1** 계획과 데이터 생성 파이프라인을 발표했다.

공식 발표와 보고서가 집계한 보존 코퍼스는 Gemma-4 tokenizer 기준

$$
2{,}601.0\text{B native-script tokens}
+
224.6\text{B romanized tokens}
=
2{,}825.6\text{B tokens}
$$

이다. 대상은 인도 헌법에 열거된 $22$개 언어이며, $12$개 언어에는 native script와 Latin 문자 romanization을 모두 만들어 총 $34$개 language-script 구성을 이룬다.

대표적인 $12$개 언어 파이프라인은 다음 네 단계다.

$$
\text{English source}
\rightarrow
\text{document translation}
\rightarrow
\text{quality score}
\rightarrow
\text{targeted repair}
\rightarrow
\text{recheck}
$$

Gemma-4-31B-it가 문서 번역과 품질 평가, 저점수 결과의 수리를 맡는다. $100$점 rubric에서 $75$점 이상만 보존하며 source ID와 quality metadata를 남긴다. 다른 $10$개 저자원 언어에는 Bodhan AI의 Indic Translate를 사용하고 Gemma가 평가한다. Apache-2.0 생성 plugin은 공개되어 있어 prompt, windowing, script remapping, 검수 과정을 수정할 수 있다.

숫자를 정확히 구분해야 한다.

- $2.8256$T는 보존된 **합성 출력 코퍼스 inventory**다.
- 고유 English source token 수나 한 모델의 실제 학습량이 아니다.
- Nemotron 3.5 Super의 Indic continued-pretraining은 명목상 $100$B-token budget에서 $12$개 언어 합성 subset과 자연 Indic 데이터에 합계 $10$B, 이전 mixture 비율을 유지한 데이터에 $90$B를 설정상 배정했다. 이는 실제 고유 token 수 측정치가 아니다.
- 이 stage 전후 다섯 Indic benchmark는 NVIDIA 보고 기준 $+4.98$에서 $+9.82$ percentage points 향상됐지만, 합성 데이터 단독 효과와 자연 데이터 효과를 분리한 인과 추정은 아니다.
- 2026년 10월 20일 기준 공식 글은 전체 코퍼스를 **releasing soon**이라고 표현한다. 기술 보고서와 생성 plugin은 공개되었지만 전체 $2.83$T corpus가 이미 즉시 다운로드 가능하다고 쓰면 안 된다.

한계도 분명하다.

- $12$개 언어 workflow에서는 같은 모델이 번역과 채점을 맡아 자기 출력 선호가 생길 수 있다.
- 사람 검수는 표본 단위이며 언어별 coverage가 균일하지 않다.
- source ID 중복 제거는 의미가 비슷한 문서까지 제거하는 semantic dedup이 아니다.
- 영어 원문 번역은 Indic 언어로 원래 작성된 텍스트를 대체하지 못하며, 현지 지식·문화 coverage를 보장하지 않는다.
- 동일 source ID에서 여러 언어·문자 파생본이 만들어졌다는 사실로부터, 이를 train과 test에 나누면 누수가 생길 수 있다고 **엔지니어링상 추론**할 수 있다.

**엔지니어 인사이트 (Impact):** 이번 공개의 핵심은 토큰 수만이 아니다. 모델 가중치 경쟁과 함께 **데이터 계보, 자동 품질평가, 수리 규칙, 원문 추적이 가능한 생성 공정**이 공개 자산이 되고 있다. 오늘의 수학으로 보면 언어와 문자를 group으로 삼고, 동일 source의 파생본을 하나의 block으로 묶어 분할해야 한다. native와 romanized 텍스트 사이 거리를 임의로 작게 두기보다 실제 downstream 오류와 native reviewer 판단으로 metric을 학습하고 독립 블록에서 반경을 검량해야 한다. 언어별 유입 비율이나 quality-score 분포가 바뀌면 online detector가 경보를 울리고, 자동으로 acceptance threshold를 낮추기보다 생성·채점 모델 버전을 동결한 채 원인을 조사해야 한다.

- 공식 발표: [Scaling Indic Pretraining: 2.8 Trillion Tokens and an Open Data Generation Pipeline](https://huggingface.co/blog/nvidia/scaling-indic-pretraining)
- 공식 기술 보고서: [Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs](https://huggingface.co/datasets/nvidia/nemotron-indic-pretraining-report/blob/main/Scaling%20Synthetic%20Pre-training%20Data%20and%20Evaluation%20for%20Indic%20LLMs.pdf)
- 공개 생성 plugin: [Nemotron Indic Translation Plugin](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/tree/main/plugins/data-designer-nemotron-indic-translation)
- 모델 기술 보고서: [Nemotron 3.5 Super and Lightning](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)

### 5.3 두 트렌드를 오늘의 학습과 연결한다

| 최신 사례 | 오늘의 수학 연결 | 실무에서 물어야 할 질문 |
| --- | --- | --- |
| Muse Spark의 네 가지 입력 modality | 교차 그룹·학습 ground metric | audio noise와 image blur를 같은 거리 단위로 비교해도 되는가? |
| $816$B total, $42$B active MoE | context별 routing·운영 drift | expert 부하와 task mix가 바뀔 때 latency·품질 경보가 함께 울리는가? |
| NVFP4·BF16·여러 runtime | 완전한 배포 버전 manifest | quantization 변경 뒤 과거 $\rho$와 안전 gate를 재사용해도 되는가? |
| Indic의 $34$ language-script 구성 | 사전 정의 group·동시 추론 | 작은 언어 그룹의 불확실성을 숨기지 않았는가? |
| 동일 source의 다국어 파생본 | block split·누수 방지 | source ID 단위로 train·test가 분리되었는가? |
| 번역기와 judge가 같은 모델 | 상관 오류·독립 감사 | 자동 점수가 놓치는 공통모드 오류를 native reviewer가 확인하는가? |
| open-weight와 open pipeline | 접근성의 서로 다른 층 | 가중치·코드·데이터·라이선스·하드웨어 중 실제로 열린 것은 무엇인가? |

두 사례를 한 문장으로 묶으면 다음과 같다.

> **개방 경쟁은 모델 가중치와 데이터 생성 공정 양쪽으로 확장되고 있지만, 실제 신뢰성은 하드웨어·라이선스·거리 정의·독립 검증·롤백 가능성이 결정한다.**

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

안정 과거 블록의 $95\%$ Wasserstein 거리 분위수로 $\rho$를 정해 배포했는데, 오늘 블록의 거리가 $\rho$를 넘었다. 왜 $\rho$를 오늘 거리까지 즉시 키워 서비스를 계속하면 안 되며, 어떤 순서로 online DRO와 정책을 갱신해야 하는가?

### 모범 답안

$\rho$는 원래 거리 $d_\phi$, 기준 context 분포 $\widehat P_{\mathrm{ref}}^X$, 정상 블록 정의가 고정되어 있고 다음 블록이 같은 정상 체제에서 온다는 조건 아래 검량되었다. $95\%$ cutoff라면 같은 정상 체제의 다음 한 블록에 대한 **주변 초과확률**은 최대 약 $5\%$이므로 한 번의 초과만으로 변화가 증명되지는 않는다. 이 말은 실현된 하나의 검량 표본에 조건부인 초과확률이 반드시 $5\%$라는 뜻은 아니다. 그러나 오늘의 초과가 구조적 변화나 사고라면 그 블록을 포함해 $\rho$를 키우는 순간, 이상을 정상 기준에 흡수하고 기존 coverage 해석을 잃는다. 경보가 사라져도 안전이 회복된 것이 아니라 자를 바꾼 것뿐일 수 있다.

따라서 먼저 해야 할 일은 다음과 같다.

1. metric·reference·radius·detector·정책·runtime 버전을 고정 기록한다.
2. promotion과 자동 reference 갱신을 멈춘다.
3. 데이터 오류, 일회성 사건, 지속적 변화, 실제 안전 비용 증가를 구분한다.
4. 중대 위험이면 현재 support에서도 검증된 $v_{\mathrm{safe}}$로 rollback하고, 그런 fallback이 없으면 fail-safe·사람 검토로 전환한다.
5. 새 체제가 지속되면 기존 metric의 의미를 먼저 감사한다. 유효하면 유지하고, 깨졌다는 근거가 있을 때만 별도 데이터로 다시 학습한다. 어느 경우든 새 reference와 다른 안정 블록으로 새 $\rho$를 검량하고 detector 상태를 새 epoch에서 재시작·재검증한다. 오경보 보장이 버전별인지 명시하고, 전체 수명 보장이 필요하면 버전별 $\alpha_v$가 $\sum_v\alpha_v\le\alpha_{\mathrm{total}}$을 만족하도록 배분하거나 epoch 횡단 anytime-valid 절차를 쓴다.
6. 또 다른 데이터에서 challenger를 선택하고 독립 final gate에서 radius·표본·결과모형·오프정책 오차를 반영한 robust 가치 LCB와 비용 UCB를 확인한다.
7. shadow와 무작위 canary를 거쳐 재진입하고, 지연 라벨이 성숙하기 전에는 노출을 성급히 확대하지 않는다.

특히 metric, radius, 그룹 정의를 같은 데이터에서 고르고 최종 안전성을 주장하면 선택 편향이 생긴다. 그러므로 새 버전은

$$
\left(
\widehat P_{\mathrm{ref}}^{X,(v+1)},
\phi_{v+1},
\rho_{v+1},
g_{v+1},
\text{detector}_{v+1},
\pi_{v+1},
\text{runtime}_{v+1}
\right)
$$

을 하나의 묶음으로 독립 승인해야 한다. 핵심은 **반경을 늘려 경보를 없애는 것**이 아니라, **경보 동안 피해를 제한하고 명시한 새 정상성 가정을 독립적으로 재검증하는 것**이다.

---

**다음 학습 예고:** 오늘은 고정된 DRO의 숨은 설계 변수와 변화점 대응을 연결했다. 다음에는 체제가 여러 개 번갈아 나타나는 상황을 상태공간모형·Hidden Markov Model로 표현하고, 순차 베이지안 업데이트와 replay·regularization을 이용한 안전한 continual learning에서 안정성–가소성, catastrophic forgetting, 재검량 주기를 다룬다.
