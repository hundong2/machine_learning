# [2026-09-23] 오늘 학습: 라그랑주 승수·KL 신뢰영역 & 제약 정책 최적화·Canary 순차 검정

> **오늘의 핵심 문장:** 안전한 AI 정책은 보상 하나를 무작정 키우지 않는다. 성능은 최대화하되, 정책 변화량과 안전 비용에는 명시적인 예산을 두고, 오프라인 통과 뒤에도 시간에 따라 유효한 증거가 쌓일 때만 배포 범위를 넓힌다.

지난 시간에는 과거의 다중 턴 로그로 후보 정책을 평가하기 위해 **PDIS(per-decision importance sampling)**와 **sequential DR**, 그리고 개선량의 **신뢰 하한(LCB)**을 배웠다. 하지만 “이 후보가 과거 로그에서 좋아 보인다”와 “다음 정책을 어떤 방향으로 얼마나 업데이트할 것인가”는 다른 문제다. 보상만 크게 만드는 업데이트는 로그가 거의 없는 행동 영역으로 뛰어갈 수 있고, 평균 성공률을 올리면서도 권한 위반·유해 응답·비용을 악화시킬 수 있다.

오늘은 이 빈틈을 메운다. 먼저 **라그랑주 승수**와 **KKT 조건**으로 “목표를 키우되 제약을 지킨다”는 문장을 수식으로 바꾼다. 이어 정책의 변화량을 **KL 발산 예산**으로 제한하고, KL의 국소 이차형식이 **피셔 정보행렬**과 **자연 그래디언트**로 이어지는 과정을 유도한다. 이 수학을 **TRPO·PPO·CPO와 primal-dual 학습**에 연결한 뒤, 작은 실제 트래픽을 받는 **canary 배포**에서 매번 결과를 들여다봐도 오류율을 통제하는 간단한 **confidence sequence(모든 시점에 동시에 유효한 신뢰구간 열)**까지 이어 간다.

> **난이도 표지:** 9월 2일부터 14일까지의 PCA·Attention 핵심 기초를 마친 뒤 진행하는 고급 확장이다. 새로운 기호는 모두 풀어 쓰되, 9월 17일의 KL 비음수성 및 9월 21~22일의 LCB·UCB 유도는 오늘의 제약식 안에서 재사용한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 보상 하나만 최대화하면 위험할까?

고객 지원 에이전트의 보상을 다음처럼 만들었다고 하자.

$$
R=\text{문제 해결 성공}-0.01\times\text{사용 토큰 수}.
$$

이 숫자만 최대화하면 에이전트는 성공률을 높이기 위해 확인 절차를 생략하거나, 필요 이상의 도구 권한을 사용하거나, 드물지만 큰 피해를 일으키는 행동을 선택할 수 있다. 안전 위반에 큰 음의 보상을 주면 해결될 것 같지만 현실에서는 다음 문제가 생긴다.

- 성공률은 퍼센트, 지연은 밀리초, 비용은 달러, 개인정보 노출은 사건 수처럼 단위가 다르다.
- 매우 드문 사고에 얼마의 벌점을 줘야 하는지 미리 정하기 어렵다.
- 벌점 계수 하나가 바뀌면 “좋은 정책”의 뜻도 바뀐다.
- 평균 보상은 좋아도 절대 넘으면 안 되는 규제·권한 한도를 위반할 수 있다.

그래서 안전 문제는 보통 **가중합 하나**가 아니라 다음과 같은 **제약 최적화**로 쓰는 편이 명확하다.

$$
\begin{aligned}
\max_\theta\quad &J_R(\theta)\\
\text{subject to}\quad
&J_{C_k}(\theta)\le d_k,
\qquad k=1,\ldots,K.
\end{aligned}
$$

$J_R$은 기대 성능, $J_{C_k}$는 $k$번째 기대 안전 비용, $d_k$는 허용 한도다. 예를 들어 성공률은 높이되 권한 위반률은 $0.1\%$ 이하, 평균 지연은 $2$초 이하, 작업당 비용은 $0.03\,\text{달러}$ 이하로 요구할 수 있다.

### 1.2 왜 “한 번에 크게 좋아지는 업데이트”를 경계할까?

정책 그래디언트는 현재 정책 근처에서 어느 방향이 좋아 보이는지 알려 준다. 그러나 그 기울기는 현재 정책이 방문한 상태와 행동에서 얻은 표본으로 계산된다. 파라미터를 크게 바꾸면 새 정책은 다른 상태를 방문하고 다른 행동을 내므로, 옛 데이터에서 만든 국소 근사가 더 이상 믿을 만하지 않다.

산길에서 발밑의 경사만 보고 수백 미터를 한 번에 뛰는 것과 같다. 경사는 “바로 근처”의 방향 정보이지 멀리 떨어진 지형의 보증서가 아니다. 그래서 다음 업데이트가 현재 정책에서 너무 멀리 가지 못하도록 **trust region(신뢰영역)**을 둔다.

정책에서는 두 행동 분포의 차이를 재는 **KL divergence(KL 발산)**가 대표적인 거리 역할을 한다. 엄밀히 KL은 대칭도 아니고 삼각부등식도 만족하지 않아 수학적 거리는 아니지만, 정책이 확률질량을 얼마나 재배치했는지 측정하는 데 유용하다.

$$
\mathbb E_{h\sim d_{\pi_{\mathrm{old}}}}
\left[
D_{\mathrm{KL}}
\bigl(
\pi_{\mathrm{old}}(\cdot\mid h)
\,\|\,
\pi_\theta(\cdot\mid h)
\bigr)
\right]
\le \delta.
$$

$\delta$는 한 번의 업데이트가 쓸 수 있는 **KL 예산**이다. “무조건 작은 파라미터 변화”가 아니라 “행동 분포가 평균적으로 너무 많이 바뀌지 않게 하라”는 뜻이다.

### 1.3 왜 벌점 계수를 사람이 고정하지 않고 라그랑주 승수를 학습할까?

보상과 비용을

$$
J_R(\theta)-\lambda J_C(\theta)
$$

처럼 합치면 $\lambda$가 안전의 가격이 된다. 그러나 고정된 $\lambda$는 환경이 바뀌어도 같은 가격을 강요한다. 비용이 한도를 훨씬 밑도는데도 성능을 지나치게 희생하거나, 반대로 위반이 계속되는데도 벌점이 약할 수 있다.

**라그랑주 승수(Lagrange multiplier)**는 제약이 빡빡할수록 커지고 여유가 있으면 작아지는 적응형 그림자 가격이다. 정책 파라미터 $\theta$는 보상을 높이는 방향으로 움직이고, 승수 $\lambda$는 위반을 더 비싸게 만드는 방향으로 움직인다. 이 둘의 균형을 설명하는 조건이 **KKT(Karush–Kuhn–Tucker) 조건**이다.

### 1.4 왜 오프라인 LCB를 통과해도 Canary가 필요할까?

오프정책 평가는 과거 로그에 없는 새 UI, 새 사용자 행동, 도구 장애, 지연된 부작용을 완전히 알 수 없다. 따라서 오프라인 LCB는 “위험한 후보를 먼저 거르는 문”이지 실제 배포를 대체하는 마지막 증명이 아니다.

**Canary 배포**는 후보 정책을 작은 무작위 트래픽에만 노출해 실제 결과를 확인하는 단계다. 그런데 매시간 지표를 보고 유리해 보이는 순간에 배포를 확대하면, 고정 표본용 $p$-value나 신뢰구간의 오류 보장이 깨진다. 동전을 계속 던지며 앞면이 유난히 많이 나온 순간만 골라 발표하는 것과 같다. 이를 **optional stopping(데이터를 보며 중단 시점을 선택하는 것)** 문제라고 한다.

따라서 canary에서는 “$n=10{,}000$일 때 한 번 본다”는 고정 시점 검정이나, 어느 시점에 보더라도 동시에 유효한 **순차 검정·confidence sequence**가 필요하다. 안전 제약은 학습할 때만 필요한 것이 아니라 배포 판단에도 이어진다.

### 1.5 역사적 흐름

1. **라그랑주·KKT:** 목적함수와 제약을 승수로 연결하고 최적점의 필요조건을 설명했다.
2. **자연 그래디언트:** 파라미터 좌표의 유클리드 길이 대신 확률분포 변화의 기하를 따라 움직이려 했다.
3. **TRPO:** 정책 개선의 surrogate objective를 KL 신뢰영역 안에서 최적화했다.
4. **PPO:** 복잡한 constrained solve 대신 확률비 clipping으로 큰 업데이트를 억제하는 실용적 근사를 제공했다.
5. **CPO·primal-dual safe RL:** 보상뿐 아니라 기대 안전 비용을 명시적 제약으로 다뤘다.
6. **순차 canary:** 오프라인 통과 뒤 실제 트래픽에서 품질과 안전의 시간에 따른 증거를 동시에 관리한다.

오늘의 큰 관점은 **학습 제약 → 오프라인 평가 → 온라인 배포 게이트**가 서로 분리된 체크박스가 아니라 하나의 안전 루프라는 것이다.

## 2. 친절한 용어 사전

| 기호·용어 | 초보자를 위한 뜻 | 오늘의 역할 |
| --- | --- | --- |
| $\theta$ | 신경망의 모든 가중치를 모은 파라미터 벡터 | 업데이트할 대상이다. |
| $\pi_\theta(a\mid h)$ | 역사 $h$에서 행동 $a$를 고를 확률 | LLM의 다음 토큰·도구 호출 정책을 나타낸다. |
| $\pi_{\mathrm{old}}$ | 업데이트 직전의 기준 정책 | 새 정책이 너무 멀리 가지 않는 기준점이다. |
| $d_\pi(h)$ | 정책 $\pi$가 역사·상태 $h$를 방문하는 분포 | 정책이 바뀌면 학습 데이터의 분포도 바뀜을 나타낸다. |
| $\tau$ | 한 과제의 시작부터 종료까지 이어진 상태·행동·결과 기록 | 정책의 보상과 비용을 계산하는 한 궤적이다. |
| $H$ | 한 궤적이 가질 수 있는 최대 단계 수 | 보상·비용을 어느 시점까지 합할지 정한다. |
| $\gamma\in[0,1]$ | 먼 미래 값을 얼마나 작게 볼지 정하는 할인율 | $\gamma=1$이면 모든 시점을 같은 비중으로 센다. |
| $r_t$, $c_{k,t}$ | $t$시점의 성능 보상과 $k$번째 안전 비용 | 궤적 전체의 $G_R$, $G_{C_k}$로 누적된다. |
| $J_R(\theta)$ | 정책의 기대 누적 보상 | 성공률·품질처럼 최대화할 목표다. |
| $J_{C_k}(\theta)$ | $k$번째 기대 누적 비용 | 안전 위반·지연·금액처럼 제한할 값이다. |
| $K$ | 동시에 관리하는 안전·운영 제약의 개수 | $k=1,\ldots,K$가 각 제약을 순회한다. |
| $d_k$ | $k$번째 비용의 허용 한도 | 제품·법·운영 요구를 수치로 표현한다. |
| feasible set | 모든 제약을 만족하는 파라미터 집합 | 정책이 머물러야 할 안전한 후보 영역이다. |
| active constraint | 최적점에서 등호로 꽉 찬 제약 | 성능을 더 높이려 할 때 실제로 발목을 잡는 한도다. |
| slack | 한도에서 실제 비용을 뺀 여유 | $d_k-J_{C_k}(\theta)$로 생각할 수 있다. |
| 라그랑지안 $\mathcal L$ | 목적과 제약을 승수로 묶은 함수 | constrained problem을 saddle-point 문제로 바꾼다. |
| $\lambda_k\ge0$ | $k$번째 라그랑주 승수 | 제약을 한 단위 완화했을 때 얻을 수 있는 목적의 국소 가치다. |
| dual variable | 원래 파라미터가 아닌 제약의 가격 변수 | 여기서는 $\lambda_k$다. |
| KKT 조건 | 최적점에서 목적 기울기와 제약 기울기가 균형을 이루는 조건 묶음 | feasibility와 최적성의 구조를 검사한다. |
| complementary slackness | 제약에 여유가 있으면 그 승수는 $0$이어야 한다는 조건 | $\lambda_k(J_{C_k}-d_k)=0$이다. |
| trust region | 현재 정책 주변에서 근사를 믿기로 한 영역 | 한 번에 너무 큰 정책 이동을 막는다. |
| $\delta$ | 허용할 평균 KL 변화량 | 한 번의 정책 업데이트 예산이다. |
| score function | $\nabla_\theta\log\pi_\theta(a\mid h)$ | 확률정책의 기울기를 표현하는 기본 벡터다. |
| 피셔 정보행렬 $F$ | score function의 바깥곱 평균 | KL의 국소 곡률을 나타낸다. |
| Hessian $\nabla_\theta^2 f$ | 함수 $f$의 모든 2차 편미분을 모은 행렬 | 각 방향으로 얼마나 휘는지 나타낸다. |
| $F\succ0$ | 영벡터가 아닌 모든 $v$에 대해 $v^\top Fv>0$인 양의 정부호 | KL 신뢰영역을 닫힌 타원체로 만들고 $F^{-1}$을 정의한다. |
| $o(\|\Delta\theta\|^2)$ | $\Delta\theta\to0$일 때 제곱항보다 더 빠르게 작아지는 나머지 | KL의 국소 이차근사에서 생략되는 고차 효과다. |
| 자연 그래디언트 | $F^{-1}g$ 방향의 업데이트 | 분포 변화량을 기준으로 가장 효율적인 상승 방향이다. |
| surrogate objective | 새 정책의 실제 성능 대신 옛 로그에서 계산하는 국소 대리 목적 | TRPO·PPO가 최적화한다. |
| advantage $A^{\pi_{\mathrm{old}}}(h,a)$ | 그 상태에서 평균적인 행동보다 $a$가 얼마나 더 좋았는지 | 어떤 행동의 확률을 올릴지 알려 준다. |
| 확률비 $r_\theta$ | $\pi_\theta(a\mid h)/\pi_{\mathrm{old}}(a\mid h)$ | 옛 정책 표본으로 새 정책 surrogate를 계산한다. |
| TRPO | Trust Region Policy Optimization | KL 제약 아래 대리 목적을 개선한다. |
| PPO | Proximal Policy Optimization | 확률비를 잘라 큰 업데이트를 억제하는 실용적 방법이다. |
| CPO | Constrained Policy Optimization | trust region과 기대 비용 제약을 함께 둔다. |
| primal variable | 원래 최적화 문제의 변수 | 정책 파라미터 $\theta$다. |
| primal-dual update | 정책과 승수를 번갈아 업데이트하는 방법 | 위반에 따라 안전 가격을 자동 조정한다. |
| $m$, $\eta_\theta$, $\eta_\lambda$ | 반복 번호, 정책 학습률, 승수 학습률 | 한 번에 각각 얼마나 움직일지 정한다. |
| $\nu\ge0$ | KL 제약에 붙는 라그랑주 승수 | 한 번의 정책 이동 예산에 대한 그림자 가격이다. |
| $I$, $\xi>0$·damping | 대각선이 $1$인 단위행렬, 작은 안정화 값, 그 기법 | $F+\xi I$를 써 역문제를 안정화한다. |
| conjugate gradient | 역행렬을 만들지 않고 $Fx=g$를 반복적으로 푸는 방법 | 거대한 신경망에서 자연 그래디언트를 근사한다. |
| $\epsilon$ | PPO가 확률비를 자르는 폭 | 보통 $1-\epsilon$과 $1+\epsilon$ 사이를 기준으로 삼는다. |
| $D_{\mathrm{TV}}(P,Q)$ | 두 분포 $P,Q$가 사건에 주는 확률의 최대 차이 | KL이 행동 통계 변화에 주는 상한을 연결한다. |
| $f(a)\in[0,1]$ | 행동 $a$를 bounded score로 바꾸는 함수 | 두 정책 아래 기대 score 차이를 비교한다. |
| canary | 새 버전을 소수 실제 트래픽에 먼저 노출하는 배포 | 오프라인에서 놓친 위험을 제한된 범위에서 확인한다. |
| optional stopping | 중간 결과를 본 뒤 유리한 시점에 멈추는 행위 | 고정 시점 통계 검정의 오류율을 깨뜨릴 수 있다. |
| confidence sequence | 모든 관측 시점에 동시에 참값을 덮도록 설계한 구간열 | canary를 계속 관찰하면서도 오류율을 통제한다. |
| $\alpha$ | 허용할 거짓 경보 확률 | 예를 들어 $0.05$면 장기 오류 예산을 $5\%$로 둔다. |
| alpha spending | 여러 시점·지표에 $\alpha$를 나누어 배정하는 규칙 | 반복 확인과 다중 안전 지표를 함께 통제한다. |
| $\overline X_t$ | 처음 $t$개 결과의 표본평균 | 시간에 따라 갱신되는 canary 지표다. |
| $\delta_{\min}$, $\eta_k$ | 요구하는 최소 품질 개선과 허용하는 $k$번째 비용 증가 | canary 확대 여부의 사전 기준이다. |
| union bound | 여러 실패 사건 중 하나라도 일어날 확률을 각 확률의 합으로 상한하는 법 | 시점별 $\alpha_t$를 전체 $\alpha$로 묶는다. |
| shadow 배포 | 후보가 행동을 계산하지만 사용자에게 적용하지 않는 단계 | 권한 없이 로그·비용·파싱 실패를 먼저 점검한다. |
| rollback | 새 정책의 트래픽을 즉시 기준 정책으로 되돌리는 것 | 순차 안전 경보의 실제 대응이다. |

오늘의 수식은 유한 길이 에피소드와 미분 가능한 확률정책을 가정한다. 기대값은 존재하고, 정책 그래디언트와 비용 그래디언트를 추정할 로그가 있으며, 국소 최적점 근처에서는 필요한 미분 가능성과 **제약 정칙성(constraint qualification, 활성 제약의 기울기가 최적점의 가능한 방향을 제대로 표현하게 하는 조건)**이 성립한다고 둔다. KKT는 이런 조건 아래의 필요조건이며, 비볼록 신경망에서는 전역 최적성 보증이 아니다.

## 3. 수학의 해부학 (증명과 원리)

### 3.1 제약 정책 최적화 문제 세우기

한 에피소드의 보상과 $k$번째 비용의 할인합을 각각

$$
G_R=\sum_{t=1}^{H}\gamma^{t-1}r_t,
\qquad
G_{C_k}=\sum_{t=1}^{H}\gamma^{t-1}c_{k,t}
$$

라고 하자. 그러면 정책의 기대 성능과 기대 비용은

$$
J_R(\theta)
=\mathbb E_{\tau\sim\pi_\theta}[G_R],
\qquad
J_{C_k}(\theta)
=\mathbb E_{\tau\sim\pi_\theta}[G_{C_k}]
$$

이다. 안전한 업데이트는 다음 문제로 쓸 수 있다.

$$
\boxed{
\begin{aligned}
\max_\theta\quad &J_R(\theta)\\
\text{s.t.}\quad
&g_k(\theta)
=J_{C_k}(\theta)-d_k\le0,
\quad k=1,\ldots,K,\\
&\overline D_{\mathrm{KL}}
(\pi_{\mathrm{old}}\|\pi_\theta)
\le\delta.
\end{aligned}}
$$

여기서 $g_k(\theta)>0$이면 $k$번째 비용을 초과했다. KL 제약도 하나의 추가 비용처럼 보이지만 역할은 다르다. $J_{C_k}\le d_k$는 배포 정책 자체의 안전·운영 요구이고, KL 제약은 **이번 한 번의 업데이트**가 국소 근사 밖으로 나가는 것을 막는 알고리즘적 안정장치다.

### 3.2 라그랑지안과 KKT 조건

KL 제약함수도

$$
h_{\mathrm{KL}}(\theta)
=\overline D_{\mathrm{KL}}
(\pi_{\mathrm{old}}\|\pi_\theta)-\delta
\le0
$$

로 쓰자. 최대화 문제에서 비용 제약 $g_k(\theta)\le0$와 KL 제약을 함께 다루는 라그랑지안은

$$
\mathcal L(\theta,\lambda,\nu)
=J_R(\theta)
-\sum_{k=1}^{K}\lambda_k g_k(\theta)
-\nu h_{\mathrm{KL}}(\theta),
\qquad \lambda_k\ge0,\quad\nu\ge0
$$

이다. 위반 $g_k>0$에는 $-\lambda_k g_k$라는 벌점이 붙고, KL 예산 초과 $h_{\mathrm{KL}}>0$에는 $-\nu h_{\mathrm{KL}}$가 붙는다. 충분히 안전해 $g_k<0$이면 비용 항은 양수가 되지만, $\lambda$는 임의의 보너스를 얻기 위한 변수가 아니라 **dual problem(제약 가격을 골라 원래 최대화 문제의 상한을 가장 타이트하게 만드는 짝 문제)**이 정하는 값이다.

정칙성이 성립하는 국소 최적점 $(\theta^*,\lambda^*,\nu^*)$에서는 다음 KKT 조건이 필요하다.

1. **Primal feasibility**

$$
g_k(\theta^*)\le0,
\qquad
h_{\mathrm{KL}}(\theta^*)\le0.
$$

2. **Dual feasibility**

$$
\lambda_k^*\ge0,
\qquad
\nu^*\ge0.
$$

3. **Complementary slackness**

$$
\lambda_k^*g_k(\theta^*)=0,
\qquad
\nu^*h_{\mathrm{KL}}(\theta^*)=0.
$$

4. **Stationarity**

$$
\nabla_\theta J_R(\theta^*)
-\sum_{k=1}^{K}
\lambda_k^*\nabla_\theta g_k(\theta^*)
-\nu^*\nabla_\theta h_{\mathrm{KL}}(\theta^*)
=0.
$$

Complementary slackness는 두 경우를 말한다.

- 제약에 여유가 있어 $g_k(\theta^*)<0$이면 반드시 $\lambda_k^*=0$이다.
- 제약이 성능을 실제로 제한하면 보통 $g_k(\theta^*)=0$이고 $\lambda_k^*>0$일 수 있다.
- KL 예산이 남으면 $\nu^*=0$이고, KL 경계가 업데이트를 제한하면 $h_{\mathrm{KL}}(\theta^*)=0$과 $\nu^*>0$이 함께 가능하다.

즉 안전 비용이나 KL 예산이 경계를 이루는 최적점에서는 성능 기울기와 활성 제약 기울기의 조합이 균형을 이룬다.

### 3.3 한 행동 확률로 보는 KKT 손계산

먼저 KL 제약은 잠시 제외하고 비용 제약의 KKT만 손으로 계산해 보자. $q\in[0,1]$를 “위험하지만 성공 가능성이 큰 도구”를 고를 확률이라고 하자.

$$
J_R(q)=0.60+0.30q,
\qquad
J_C(q)=0.01+0.25q.
$$

안전 비용 한도가 $d=0.06$이면

$$
0.01+0.25q\le0.06
\quad\Longleftrightarrow\quad
q\le0.20.
$$

보상만 보면 $q=1$을 택하지만 feasible set에서는 $q^*=0.20$이 최적이다. 제약함수는

$$
g(q)=0.01+0.25q-0.06
$$

이고 라그랑지안은

$$
\mathcal L(q,\lambda)
=0.60+0.30q
-\lambda(0.01+0.25q-0.06)
$$

이다. Stationarity를 적용하면

$$
\frac{\partial\mathcal L}{\partial q}
=0.30-0.25\lambda=0
\quad\Longrightarrow\quad
\lambda^*=1.2.
$$

$q^*=0.20$에서는 $g(q^*)=0$이므로 complementary slackness도 성립한다. $\lambda^*=1.2$는 비용 한도를 아주 조금 완화할 때 목적값이 국소적으로 얼마나 늘 수 있는지 나타내는 **그림자 가격**이다. 단위가 서로 다른 비용끼리 $\lambda$의 숫자 크기를 그대로 비교해서는 안 된다.

### 3.4 Primal-dual 업데이트는 어떻게 움직일까?

비용 제약의 라그랑지안은 정책 $\theta$에 대해서는 최대화하고, 제약 가격 $\lambda$에 대해서는 최소화하는 **saddle-point(한 변수 방향에서는 봉우리, 다른 변수 방향에서는 골짜기가 되는 균형점)** 문제로 볼 수 있다.

$$
\max_\theta\min_{\lambda\ge0}
\mathcal L(\theta,\lambda).
$$

확률적 그래디언트로 단순화하면

$$
\theta_{m+1}
=\theta_m
+\eta_\theta
\left(
\widehat\nabla J_R
-\sum_k\lambda_{k,m}\widehat\nabla J_{C_k}
\right),
$$

$$
\lambda_{k,m+1}
=\left[
\lambda_{k,m}
+\eta_\lambda
(\widehat J_{C_k}-d_k)
\right]_+,
$$

여기서 $[x]_+=\max(0,x)$다. 두 번째 식의 부호를 꼭 확인하자.

- 비용이 한도를 넘으면 $\widehat J_{C_k}-d_k>0$이므로 $\lambda_k$가 커진다.
- 비용에 여유가 있으면 $\lambda_k$가 작아진다.
- $\lambda_k$는 음수가 될 수 없으므로 $0$에 투영한다.

그러나 이 동역학이 자동으로 매 순간 제약을 지킨다는 뜻은 아니다. 승수 업데이트가 느리면 위반이 오래 지속되고, 너무 빠르면 정책과 승수가 진동할 수 있다. 비정상 환경, 편향된 비용 추정, 서로 충돌하는 제약에서는 feasible policy 자체가 없을 수도 있다. 따라서 학습 로그에는 성능뿐 아니라 각 제약의 slack과 $\lambda_k$ 궤적을 함께 남겨야 한다.

위 식은 비용 승수의 가장 단순한 업데이트만 보여 준다. KL 제약은 보통 다음 절의 이차 trust-region solver와 line search로 직접 처리하거나, 별도 $\nu$·KL penalty를 적응시킨다. 따라서 이 비용 primal-dual 식만 실행했다고 KL 예산까지 지켜지는 것은 아니다.

### 3.5 KL의 국소 이차형식과 피셔 정보행렬

현재 파라미터를 $\theta_0$라 하고 새 파라미터를

$$
\theta=\theta_0+\Delta\theta
$$

로 쓰자. 한 역사 $h$에서의 KL은

$$
D_h(\theta_0\|\theta)
=\mathbb E_{a\sim\pi_{\theta_0}}
\left[
\log\pi_{\theta_0}(a\mid h)
-\log\pi_\theta(a\mid h)
\right]
$$

이다. $\theta=\theta_0$에서는 두 분포가 같으므로 $D_h=0$이다. 1차 미분은

$$
\left.
\nabla_\theta D_h(\theta_0\|\theta)
\right|_{\theta=\theta_0}
=-
\mathbb E_{a\sim\pi_{\theta_0}}
\left[
\nabla_\theta\log\pi_\theta(a\mid h)
\right]_{\theta=\theta_0}.
$$

그런데 score function의 기대값은

$$
\begin{aligned}
\mathbb E_{a\sim\pi_\theta}
[\nabla_\theta\log\pi_\theta(a\mid h)]
&=\sum_a\pi_\theta(a\mid h)
\frac{\nabla_\theta\pi_\theta(a\mid h)}
{\pi_\theta(a\mid h)}\\
&=\nabla_\theta\sum_a\pi_\theta(a\mid h)\\
&=\nabla_\theta 1=0.
\end{aligned}
$$

행동의 **support(양의 확률을 받는 행동들의 집합)**가 파라미터 근처에서 바뀌지 않고 미분과 합을 교환할 수 있다는 정규성 아래, score 평균 $0$을 한 번 더 미분하면

$$
\mathbb E_{a\sim\pi_{\theta_0}}
\left[
\nabla_\theta^2\log\pi_{\theta_0}(a\mid h)
+\bigl(\nabla_\theta\log\pi_{\theta_0}(a\mid h)\bigr)
\bigl(\nabla_\theta\log\pi_{\theta_0}(a\mid h)\bigr)^\top
\right]=0.
$$

그러므로 KL의 Hessian은

$$
\left.
\nabla_\theta^2D_h(\theta_0\|\theta)
\right|_{\theta=\theta_0}
=-
\mathbb E
\left[
\nabla_\theta^2\log\pi_{\theta_0}(a\mid h)
\right]
=\mathbb E
\left[
\bigl(\nabla_\theta\log\pi_{\theta_0}(a\mid h)\bigr)
\bigl(\nabla_\theta\log\pi_{\theta_0}(a\mid h)\bigr)^\top
\right].
$$

따라서 KL은 현재 정책에서 1차항이 사라지고, 테일러 전개의 첫 비영 항이 이차항이 된다.

$$
\overline D_{\mathrm{KL}}
(\theta_0,\theta_0+\Delta\theta)
=
\frac12
\Delta\theta^\top F\Delta\theta
+o(\|\Delta\theta\|^2).
$$

여기서

$$
F
=\mathbb E_{
h\sim d_{\pi_{\theta_0}},
a\sim\pi_{\theta_0}(\cdot\mid h)}
\left[
s_\theta(h,a)s_\theta(h,a)^\top
\right],
$$

$$
s_\theta(h,a)
=\left.
\nabla_\theta\log\pi_\theta(a\mid h)
\right|_{\theta=\theta_0}
$$

가 **피셔 정보행렬**이다. $F$는 양의 준정부호이므로 $\Delta\theta^\top F\Delta\theta\ge0$이다. 같은 유클리드 길이의 파라미터 변화라도 행동 확률을 크게 바꾸는 방향은 $F$가 큰 비용을 부여한다.

### 3.6 자연 그래디언트 신뢰영역 해 구하기

현재 정책 주변에서 보상 목적을 1차로 근사하자.

$$
J_R(\theta_0+\Delta\theta)
\approx
J_R(\theta_0)+g^\top\Delta\theta,
\qquad
g=\nabla_\theta J_R(\theta_0).
$$

유도를 위해 식별 가능한 정책의 **접공간(tangent space, 현재 정책에서 아주 작게 움직일 수 있는 국소 방향들의 공간)**에서 $F\succ0$이고 $g\ne0$라고 가정하자. 실제로 $F$가 특이하거나 수치적으로 불안정하면 뒤에서 설명할 damping으로 $F_\xi=F+\xi I\succ0$를 사용한다. 상수항을 버리면 다음 문제가 된다.

$$
\begin{aligned}
\max_{\Delta\theta}\quad
&g^\top\Delta\theta\\
\text{s.t.}\quad
&\frac12\Delta\theta^\top F\Delta\theta\le\delta.
\end{aligned}
$$

라그랑지안을

$$
\mathcal L(\Delta\theta,\nu)
=g^\top\Delta\theta
-\nu
\left(
\frac12\Delta\theta^\top F\Delta\theta-\delta
\right),
\qquad\nu\ge0
$$

로 두면 stationarity는

$$
g-\nu F\Delta\theta=0
$$

이므로

$$
\Delta\theta=\frac1\nu F^{-1}g.
$$

보상 기울기가 $0$이 아니고 경계에서 KL 예산을 모두 쓴다면

$$
\frac12
\left(\frac1\nu F^{-1}g\right)^\top
F
\left(\frac1\nu F^{-1}g\right)
=\delta.
$$

경계가 활성화될 때 dual feasibility를 만족하는 양의 근을 택하면

$$
\nu
=\sqrt{
\frac{g^\top F^{-1}g}{2\delta}
}
$$

이고 최종 스텝은

$$
\boxed{
\Delta\theta^*
=\sqrt{
\frac{2\delta}
{g^\top F^{-1}g}
}
F^{-1}g
}
$$

이다. $F^{-1}g$가 **자연 그래디언트** 방향이다. 실제 신경망에서는 $F$를 직접 만들거나 역행렬로 계산하지 않고, 행렬–벡터 곱과 **conjugate gradient(켤레기울기법)**를 이용한다. $F$가 특이하거나 추정 잡음이 크면 **damping(대각 안정화)**으로 $F_\xi=F+\xi I$를 쓰고, 식의 $F$를 $F_\xi$로 바꾼다.

### 3.7 비용 제약을 더하면 CPO의 국소 문제가 된다

$k$번째 비용 제약을 현재 정책에서 1차로 근사하면

$$
J_{C_k}(\theta_0+\Delta\theta)-d_k
\approx
c_k+b_k^\top\Delta\theta,
$$

$$
c_k=J_{C_k}(\theta_0)-d_k,
\qquad
b_k=\nabla_\theta J_{C_k}(\theta_0).
$$

따라서 CPO 계열의 핵심 국소 문제는

$$
\boxed{
\begin{aligned}
\max_{\Delta\theta}\quad
&g^\top\Delta\theta\\
\text{s.t.}\quad
&c_k+b_k^\top\Delta\theta\le0,
\quad k=1,\ldots,K,\\
&\frac12\Delta\theta^\top F\Delta\theta\le\delta
\end{aligned}}
$$

가 된다. $F\succ0$이면 기하학적으로 KL 제약은 타원체, 비용 제약은 반공간이다. $F$가 양의 준정부호일 뿐이면 퇴화한 타원체 또는 원통형 영역이 될 수 있어 damping이 필요하다. 자연 그래디언트 방향이 안전 반공간 밖을 향하면, 해는 성능 상승을 일부 포기하고 KL 영역과 안전 경계가 만나는 쪽으로 휘어진다.

현재 정책부터 이미 $c_k>0$인 경우에는 “허용된 KL 영역 안의 복구 업데이트로 feasible set에 돌아갈 수 있는가”를 먼저 확인해야 한다. 국소 비용 반공간과 KL 영역의 교집합이 비어 있으면 그 스텝에서는 정상적인 개선 해가 없다. 이때는 보상 개선을 잠시 포기한 cost-recovery objective, 실행 안전 필터, 더 안전한 탐색 데이터, 권한 축소, 또는 제약·모델의 feasibility 자체를 점검해야 한다. 단순히 스텝이나 $\delta$를 더 줄이면 가능한 영역도 줄어들므로 해결책이 아닐 수 있다. 수치 solver가 답을 냈다는 사실이 실제 제약 충족을 뜻하지 않는다.

### 3.8 TRPO의 surrogate와 PPO clipping

옛 정책의 로그에서 새 정책의 행동 확률 변화를 반영하는 비율은

$$
r_\theta(h,a)
=\frac{\pi_\theta(a\mid h)}
{\pi_{\mathrm{old}}(a\mid h)}
$$

이다. 옛 정책의 advantage를 사용한 대표적 surrogate는

$$
L(\theta)
=\mathbb E_{
h\sim d_{\pi_{\mathrm{old}}},
a\sim\pi_{\mathrm{old}}(\cdot\mid h)}
\left[
r_\theta(h,a)
A^{\pi_{\mathrm{old}}}(h,a)
\right].
$$

여기서 advantage는 옛 정책으로 추정해 $\theta$에 대해 고정된 값으로 취급한다. TRPO는 이 surrogate를 키우되 평균 KL을 제한한다. PPO의 clipped objective는

$$
L^{\mathrm{clip}}(\theta)
=\mathbb E
\left[
\min\left(
r_\theta A,
\operatorname{clip}
(r_\theta,1-\epsilon,1+\epsilon)A
\right)
\right]
$$

이다. 좋은 행동 $A>0$의 확률을 지나치게 올리거나 나쁜 행동 $A<0$의 확률을 지나치게 내릴 때 추가 이득을 잘라낸다.

중요한 구분은 다음과 같다.

- PPO clipping은 **샘플된 행동의 확률비 목적**을 자르는 것이지 모든 상태의 KL을 하드하게 제한하지 않는다.
- 구현에서 평균 KL을 따로 측정해 early stopping을 걸 수 있지만, 이것도 드문 상태의 큰 변화를 숨길 수 있다.
- PPO clipping만으로 안전 비용 $J_C\le d$가 보장되지 않는다. 비용 advantage, 라그랑주 승수, 별도의 projection·shield가 필요하다.
- TRPO·CPO의 이론적 보장은 정확한 기대값과 근사 조건에 의존하며, 유한 배치·함수 근사·비정상 환경에서는 경험적 검증이 필요하다.

### 3.9 KL 예산은 무엇을 보장하고 무엇을 보장하지 않을까?

자연로그를 쓰고

$$
D_{\mathrm{TV}}(P,Q)
=\frac12\sum_a|P(a)-Q(a)|
=\sup_A|P(A)-Q(A)|
$$

로 정의하자. 한 상태에서 Pinsker 부등식은

$$
D_{\mathrm{TV}}(P,Q)
\le
\sqrt{\frac12D_{\mathrm{KL}}(P\|Q)}
$$

를 준다. $f(a)\in[0,1]$이면

$$
\left|
\mathbb E_P[f]-\mathbb E_Q[f]
\right|
\le D_{\mathrm{TV}}(P,Q)
$$

이므로 작은 KL은 그 상태에서 bounded action statistic의 급격한 변화를 제한한다.

하지만 실제 제약은 흔히

$$
\overline D_{\mathrm{KL}}
=\mathbb E_{h\sim d_{\pi_{\mathrm{old}}}}
[D_h]
$$

라는 **옛 정책 방문분포 아래 평균**이다. 따라서 다음은 자동 보장되지 않는다.

- 모든 역사에서의 최대 KL
- 새 정책이 새롭게 자주 방문하는 상태의 KL
- 긴 궤적 전체의 실패확률
- 상태 방문 변화나 모델 밖 실행기로 연결되는 권한 위반 같은 시스템 부작용
- 모델 밖의 parser·tool·UI 변경

KL budget은 유용한 속도 제한이지 방호벽 전체가 아니다. 평균 KL, 상위 분위수 KL, 최대 관측 KL, 안전 비용, 실제 canary 결과를 따로 봐야 한다.

### 3.10 장난감 정책에 비용 제약과 KL 제약을 함께 적용하기

3.3의 위험 행동 확률 $q$를 다시 보자. 현재 정책은 $q_0=0.10$이고, 비용 제약만 보면 $q\le0.20$까지 갈 수 있다. Bernoulli 정책의 KL은

$$
D_{\mathrm{KL}}
(\operatorname{Bern}(q_0)\|\operatorname{Bern}(q))
=q_0\log\frac{q_0}{q}
+(1-q_0)\log\frac{1-q_0}{1-q}.
$$

$q$ 자체를 파라미터로 볼 때 현재점의 피셔 정보는

$$
F(q_0)=\frac1{q_0(1-q_0)}.
$$

KL 예산이 $\delta=0.01$이면 국소 이차근사로 허용되는 양의 변화는

$$
\frac12
\frac{(q-q_0)^2}{q_0(1-q_0)}
\le0.01,
$$

$$
q-q_0
\le
\sqrt{2(0.01)(0.10)(0.90)}
\approx0.0424.
$$

따라서 국소 이차근사에서는 이번 스텝이 대략 $q\le0.1424$로 제한된다. 이 점의 정확한 Bernoulli KL은 약 $0.00809$이고, 정확한 식에서 $D_{\mathrm{KL}}=0.01$이 되는 양의 경계는 $q\approx0.14768$이다. 국소 근사는 현재점 근처의 계산 도구이지 exact constraint가 아니므로, 실제 line search에서는 정확한 KL을 다시 계산한다. 어느 쪽이든 비용 제약의 $0.20$보다 KL trust region이 더 빡빡하다. 다음 배치에서 새 기준점과 실제 KL·비용을 다시 측정한 뒤 조금 더 움직일 수 있다. 이처럼 비용 제약은 **어디까지 가도 되는가**, KL 예산은 **이번에 얼마나 움직일 것인가**를 답한다.

### 3.11 매번 들여다봐도 유효한 Canary confidence sequence

고정된 한 시점이 아니라 모든 시점 $t=1,2,\ldots$에서 동시에 유효한 간단한 구간을 만들어 보자. 사용자 단위 결과 $X_i\in[0,1]$가 독립이고 공통의 시간불변 평균 $\mu$를 갖는다고 가정한다. 시점별 오류 예산을

$$
\alpha_t
=\frac{6\alpha}{\pi^2t^2}
$$

로 나누면

$$
\sum_{t=1}^{\infty}\alpha_t=\alpha
$$

이다. Hoeffding 부등식에 따라 고정된 $t$에서

$$
P\left(
|\overline X_t-\mu|>
\sqrt{\frac{\log(2/\alpha_t)}{2t}}
\right)
\le\alpha_t.
$$

폭을

$$
w_t
=\sqrt{\frac{\log(2/\alpha_t)}{2t}}
$$

라고 하자. **Union bound**는 “여러 실패 사건 중 하나라도 일어날 확률은 각 실패확률의 합보다 크지 않다”는 법칙이다. 이를 쓰면

$$
P\left(
\exists t\ge1:
|\overline X_t-\mu|>w_t
\right)
\le
\sum_{t=1}^{\infty}\alpha_t
=\alpha.
$$

따라서

$$
\boxed{
[L_t,U_t]
=\left[
\max(0,\overline X_t-w_t),
\min(1,\overline X_t+w_t)
\right]
}
$$

는 단순하지만 모든 $t$에 동시에 유효한 confidence sequence다. 구간의 동시 coverage는 데이터에 따라 고른 stopping time에도 유지된다. 다만 지표·대상·세그먼트·후보 선택을 결과를 본 뒤 바꾸면 다른 선택 문제가 생기므로 그 절차를 사전 등록하거나 별도로 보정해야 한다. 이 구간은 설명하기 쉬운 대신 최신 mixture·martingale 기반 구간보다 넓을 수 있다. 트래픽 확대와 사용자 구성 변화로 평균 자체가 drift하면 하나의 고정 $\mu$를 위한 이 단순 구간을 그대로 적용할 수 없으므로 단계·시간대별 설계를 다시 해야 한다.

기준군과 canary군을 무작위 배정하고, 전체 오류 예산 안에서 각 군의 실패확률을 나눠 배정하자. 전체 시점 $t$까지 각 군에 쌓인 표본 수를 $n_0(t)$, $n_1(t)$라 하고 confidence sequence $[L_{0,n_0(t)},U_{0,n_0(t)}]$, $[L_{1,n_1(t)},U_{1,n_1(t)}]$를 만들면 평균 차이

$$
\Delta=\mu_1-\mu_0
$$

의 동시 유효 구간은 보수적으로

$$
[L_{1,n_1(t)}-U_{0,n_0(t)},\;
U_{1,n_1(t)}-L_{0,n_0(t)}]
$$

로 얻을 수 있다. 품질에는 하한, 안전 비용에는 상한을 사용한다.

$$
\text{확대 조건:}
\quad
L_t^{\Delta R}\ge\delta_{\min},
\qquad
U_t^{C_{1,k}}\le d_k,
\qquad
U_t^{\Delta C_k}\le\eta_k
\quad\forall k.
$$

최소 노출량·최소 관찰 시간·지연 보상 완료율도 미리 정해야 한다. 사용자 한 명이 여러 요청을 보낸다면 요청 행을 독립 표본으로 세면 안 되고 사용자·세션 단위로 집계하거나 cluster-aware 순차 방법을 써야 한다. 지표·세그먼트·후보가 여러 개면 $\alpha$를 다시 나누거나 전체 선택 절차를 포함한 방법을 써야 한다.

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 안전한 정책 개선의 전체 루프

~~~text
보상·안전 비용·한도 사전 정의
  ↓
기준 정책으로 로그 수집 + 실제 행동확률 기록
  ↓
오프정책 평가(PDIS·Sequential DR) + ESS·지원집합 진단
  ↓
개선량 LCB와 비용 UCB를 통과한 후보만 학습/선정
  ↓
KL trust region + 명시적 비용 제약으로 제한된 업데이트
  ↓
독립 로그에서 재평가 + 제약 slack·KL 감사
  ↓
Shadow: 행동은 계산하되 외부 효과는 차단
  ↓
무작위 소규모 Canary + time-uniform 품질/안전 구간
  ↓
통과 시 단계적 확대 / 위반 증거 시 즉시 rollback
  ↓
분포 변화·새 도구·새 UI를 반영해 루프 반복
~~~

이 흐름에서 같은 데이터를 후보 생성, 승수 조정, 임계값 선택, 최종 통과 판정에 모두 쓰면 낙관적 편향이 생긴다. 학습·모델 선택·최종 게이트의 로그를 시간 또는 사용자 단위로 분리하고, canary 규칙은 트래픽을 보기 전에 고정한다.

### 4.2 TRPO·PPO·CPO·primal-dual의 차이

| 방법 | 큰 업데이트 제어 | 안전 비용 제약 | 장점 | 주의점 |
| --- | --- | --- | --- | --- |
| TRPO | 평균 KL trust region을 constrained solve로 근사 | 기본형에는 없음 | 분포 기하와 개선 근사가 명시적 | 구현이 복잡하고 평균 KL이 희귀 상태를 숨길 수 있다. |
| PPO-Clip | 샘플 확률비를 $[1-\epsilon,1+\epsilon]$ 주변에서 clip | 기본형에는 없음 | 미니배치 SGD로 구현이 쉽다 | clip은 하드 KL 제약도, 안전 보증도 아니다. |
| PPO-KL | 목적에 KL penalty를 더하거나 target KL에서 중단 | 별도 설계 필요 | 실무적 업데이트 크기 조정 | penalty 계수와 target이 환경별로 민감하다. |
| CPO | KL 이차제약 | 기대 비용의 선형화 제약 | 성능·안전 비용을 같은 국소 문제에서 다룬다 | 추정 오차·solver 오차·비볼록성 때문에 실제 검증이 필요하다. |
| Lagrangian PPO | PPO 목적에서 $\lambda_k$로 비용 advantage를 감점 | 승수를 동적으로 학습 | 여러 비용에 확장하기 쉽다 | 제약 위반 진동, 승수 폭주, infeasibility를 감시해야 한다. |
| Safety shield | 학습과 별개로 금지 행동을 실행 직전 차단 | 규칙 기반 hard constraint 가능 | 즉각적 방어선 | 허용 행동 안의 품질과 장기 부작용까지 해결하지 않는다. |

방법 이름보다 중요한 것은 실제 구현이 무엇을 제한하는지다. “PPO를 썼다”는 말만으로 KL, 권한 위반, 비용 한도 중 어느 것도 자동 보장되지 않는다.

### 4.3 LLM/VLM 에이전트의 보상과 비용을 분리하기

| 종류 | 예시 | 관측 방법 | 흔한 함정 |
| --- | --- | --- | --- |
| 성능 보상 | 과제 성공, 정답률, 사용자 문제 해결 | deterministic verifier, 사람 평가, 결과 상태 확인 | 형식만 맞춘 보상 해킹, judge 편향 |
| 안전 비용 | 승인 없는 결제·삭제·외부 전송 | 권한 로그, side-effect 감사, 정책 위반 판정 | 드문 사건이라 평균 추정이 매우 불안정 |
| 개인정보 비용 | 민감정보가 모델·도구·로그에 노출됨 | DLP 규칙, 정밀 표본 감사 | 탐지기 자체의 false negative |
| 운영 비용 | 토큰, GPU 시간, API 금액 | 계량 로그 | 재시도·cache miss·도구 비용 누락 |
| 지연 비용 | p50·p95·timeout | end-to-end trace | 평균만 보면 긴 꼬리를 숨김 |
| 인간 개입 비용 | 검토·승인 요청 횟수 | workflow 로그 | 무조건 승인을 요구해 안전 수치만 좋게 만듦 |

평균 제약 하나로 충분하지 않을 수 있다. 예를 들어 평균 지연 $\le2$초와 p99 지연 $\le10$초는 다른 요구다. 권한 위반은 평균 비용보다 **사건 확률**이나 “한 건도 없어야 하는 hard rule”로 다뤄야 할 수 있다. 심각도가 다른 사건을 임의의 숫자로 더하기 전에 제품·보안·법무가 단위를 합의해야 한다.

### 4.4 LLM 토큰 정책에서 KL은 어디에 적용할까?

LLM 응답 $y=(y_1,\ldots,y_T)$의 확률은

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t})
$$

이므로 시퀀스 로그확률 차이는 토큰별 차이의 합이다.

$$
\log\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{old}}(y\mid x)}
=\sum_{t=1}^{T}
\log\frac{
\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{\mathrm{old}}(y_t\mid x,y_{<t})}.
$$

실무에서는 유효 토큰 평균 KL, 시퀀스 합 KL, 턴 단위 KL이 서로 다른 길이 편향을 만든다. padding·prompt token을 포함할지, stop token과 tool schema token을 포함할지, 샘플링 후처리 전 모델 분포와 실제 실행 정책 중 무엇을 비교할지 명시해야 한다.

도구 에이전트에서는 텍스트 KL이 작아도 parser 경계 하나를 넘어 `read_file`이 `delete_file`로 바뀔 수 있다. 따라서 토큰 KL과 별도로 구조화 행동의 변화, 도구별 호출률, 권한 수준, 실제 side effect를 측정한다.

### 4.5 Canary를 실제로 운영하는 규칙

1. **실험 단위 고정:** 같은 사용자가 기준군과 후보군을 오가며 오염되지 않도록 사용자·조직 단위로 무작위 배정한다.
2. **노출 전 등록:** 핵심 품질 지표, 안전 비용, 최소 개선폭, 한도, $\alpha$, 최소 표본, 최대 기간, 중단·확대 조건을 미리 기록한다.
3. **Shadow 먼저:** tool call을 생성하되 외부 효과를 차단해 schema 오류, 비용, 지연, 금지 도구 의도를 측정한다.
4. **아주 작은 Canary:** 예를 들어 $1\%$에서 시작하되 고위험 도구는 read-only 또는 추가 인간 승인으로 제한한다.
5. **동시 유효 구간 갱신:** 품질의 LCB, 각 안전 비용의 UCB, 표본 완료율을 같은 cadence로 갱신한다.
6. **확대와 중단을 비대칭으로 설계:** 위험 증거에는 빠르게 rollback하고, 성공 증거에는 최소 시간·표본을 요구한다.
7. **단계적 확대:** $1\%\rightarrow5\%\rightarrow20\%\rightarrow50\%$처럼 각 단계에서 새 환경과 부하를 다시 확인한다.
8. **사후 학습 분리:** canary에서 임계값을 바꾸거나 후보를 수정했다면 같은 데이터로 최종 통과를 선언하지 않는다.

Canary는 피해를 허용하는 면허가 아니다. 되돌릴 수 없는 결제·삭제·의료 결정에는 트래픽 비율과 별개로 승인을 요구하거나 기능을 비활성화해야 한다.

### 4.6 수학 부품과 AI 시스템의 1:1 연결

| 수학 부품 | AI 엔지니어링 부품 | 확인 질문 |
| --- | --- | --- |
| $J_R(\theta)$ | 과제 성공·품질의 기대값 | 대리 judge 점수가 실제 사용자 가치와 맞는가? |
| $J_{C_k}(\theta)\le d_k$ | 권한·유해성·비용·지연 SLO | 평균과 꼬리 위험을 구분했는가? |
| $\lambda_k$ | 비용 위반의 동적 가격 | 위반 시 커지고 여유 시 줄어드는가? |
| KKT primal feasibility | 후보 정책의 실제 안전 한도 충족 | 추정치가 아니라 UCB까지 한도 안인가? |
| complementary slackness | 여유 있는 제약의 승수는 $0$ | 불필요한 벌점이 성능을 누르고 있지 않은가? |
| $F^{-1}g$ | KL 기하를 따른 정책 업데이트 | 파라미터 크기가 아니라 분포 변화로 스텝을 재는가? |
| KL budget $\delta$ | 한 번의 허용 정책 변화량 | 평균뿐 아니라 tail·도구별 KL도 보는가? |
| CPO 반공간 | 안전 비용의 국소 허용 방향 | 현재 정책이 이미 위반 중이면 복구 계획이 있는가? |
| PPO clip | 샘플 확률비의 대리 안정장치 | 이를 hard safety guarantee로 오해하지 않는가? |
| Confidence sequence | 반복 확인 가능한 canary 구간 | 중간 결과를 본 뒤 규칙을 바꾸지 않았는가? |
| alpha spending | 여러 시점·지표의 오류 예산 | 세그먼트·후보 탐색까지 포함했는가? |
| rollback | 제약 위반 증거에 대한 실행 | 자동화되어 있고 실제 복구 시간을 측정했는가? |

### 4.7 초보자가 흔히 하는 오해

| 오해 | 바로잡기 |
| --- | --- |
| “벌점을 충분히 크게 주면 hard constraint와 같다.” | 유한한 벌점은 보상 이득과 교환될 수 있다. 절대 금지 행동은 실행 차단과 명시적 제약이 필요하다. |
| “$\lambda$가 크면 안전한 모델이다.” | 큰 $\lambda$는 지속적 위반이나 단위 스케일 문제의 신호일 수도 있다. 제약 slack과 함께 봐야 한다. |
| “KKT를 만족하면 전역 최적해다.” | 비볼록 신경망에서는 보통 국소 필요조건일 뿐이다. 정칙성과 추정 정확도도 필요하다. |
| “파라미터의 $L_2$ 변화가 작으면 정책도 거의 같다.” | 민감한 방향의 작은 파라미터 변화가 출력 확률을 크게 바꿀 수 있다. KL·행동 지표가 더 직접적이다. |
| “KL이 $0.01$이면 모든 프롬프트에서 변화가 작다.” | 평균 KL은 희귀 프롬프트의 큰 변화를 숨길 수 있다. 분위수·최대·세그먼트별 값을 본다. |
| “PPO clipping이 trust region을 정확히 구현한다.” | 샘플 surrogate의 이득을 자를 뿐 hard KL ball을 직접 강제하지 않는다. |
| “CPO를 쓰면 배포가 안전하다.” | 비용 정의·측정·환경 변화·solver 오차가 틀리면 제약도 틀린다. 독립 평가와 canary가 필요하다. |
| “Canary가 작으면 통계는 중요하지 않다.” | 작은 표본일수록 변동성이 크며 반복 확인의 우연한 경보가 더 문제다. |
| “매시간 일반 $95\%$ 신뢰구간을 보고 통과 순간 멈춰도 된다.” | 고정 시점 구간을 반복 확인하면 전체 오류율이 커진다. 순차 유효 방법이 필요하다. |
| “요청 백만 건이면 표본 백만 개다.” | 같은 사용자·조직의 요청은 상관될 수 있다. 무작위 배정 단위와 분석 단위를 맞춰야 한다. |
| “안전 UCB만 한도 아래면 바로 확대한다.” | 지연 보상, 데이터 누락, 최소 관찰 기간, 미리 정한 품질 기준도 충족해야 한다. |

### 4.8 엔지니어의 최소 감사 체크리스트

1. 보상과 각 안전 비용의 단위·집계 창·지연 시간을 문서화한다.
2. hard-ban 행동과 기대 비용 제약을 분리한다.
3. 기준 정책의 실제 sampler·tool router·후처리를 포함한 행동 확률을 기록한다.
4. 업데이트마다 평균·p95·최대 KL, entropy, 도구별 호출률을 남긴다.
5. 각 비용의 추정치·UCB·slack과 $\lambda_k$ 궤적을 함께 본다.
6. KL·비용 국소 근사 뒤에는 실제 새 정책으로 line search와 재평가를 수행한다.
7. PPO clipping을 사용해도 target KL·비용 제약·gradient norm을 별도 감시한다.
8. 학습용, 후보 선택용, 최종 오프라인 게이트용 로그를 분리한다.
9. Shadow에서 외부 side effect를 차단하고 parser·권한·비용을 검증한다.
10. Canary의 단위, traffic ramp, $\alpha$, 최소 표본·기간, 확대·중단·rollback 규칙을 사전 등록한다.
11. 사용자·조직·시간 군집과 지연 보상을 반영한 순차 추론을 사용한다.
12. 고위험 기능은 canary 비율과 무관하게 인간 승인·rate limit·egress 제한을 유지한다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

### 5.1 MiMo-V2.6-Pro-RL: 멀티도메인 RL을 한 번에 섞는 오픈웨이트 옴니모달 에이전트

[Xiaomi MiMo-V2.6-Pro-RL 공식 모델 카드](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)는 Hugging Face API의 `createdAt` 기준 2026년 9월 22일 00:39 KST(9월 21일 15:39 UTC)에 공개됐으며, 모델 카드 메타데이터에는 MIT 라이선스로 표시되어 있다. 모델 카드가 보고하는 구조는 sparse MoE(Mixture of Experts, 많은 전문가 중 일부만 토큰마다 활성화하는 구조) $1.02$T 총 파라미터·$42$B 활성 파라미터, $1$M token context, text·image·video·audio 입력, $681$M MiMo ViT, 두 종류의 audio encoder, 5-layer **speculative decoder(본 모델의 다음 토큰을 미리 여러 개 제안해 검증·채택 속도를 높이는 작은 초안 디코더)**다.

오늘 주제와 직접 맞닿는 부분은 **You Only RL Once**라는 mixed RL 설계다. 코딩, 일반 에이전트, 시각, 사이버보안 과제와 여러 **harness(모델·도구·환경·채점기를 묶어 과제를 실행하는 껍질)**를 한 배치에 섞어 도메인별 RL run을 따로 하지 않았다고 보고한다. 완전 비동기 GRPO를 $1{,}568$ prompts $\times16$ **rollouts(정책이 환경에서 처음부터 끝까지 수행한 실행 궤적)** per step 규모로 돌린다. 그중 코드 에이전트 과제의 서로 다른 subset에서 단순 pass/fail이 구분하지 못하는 성공 패치의 품질을 비교하기 위해 다음 두 장치를 사용한다.

- **GRS(Groupwise Reward Synthesis):** 선택된 high-pass-rate 코드 과제에서 서로 다른 rollout을 비교해 과제별 **rubric(좋은 해법을 판정하는 세부 채점 기준)**을 오프라인으로 만들고 test outcome과 합친다.
- **GAR(Groupwise Advantage Redistribution):** 나머지 코드 에이전트 과제에서 통과한 patch를 접근법·정밀성·최소 변경·부작용·완성도로 순위화하고, grader가 더 높은 품질로 판단한 pass 쪽으로 양의 advantage를 재배분한다. 길이 제어는 보고서의 별도 group-relative length penalty가 담당한다.

혼합 RL 뒤에는 **MOPD2(Multi-Prefix Multi-Teacher On-Policy Distillation)**로 teacher trajectory와 SFT prefix의 중간 의사결정 지점을 재사용한다. 모델 카드는 self-correction cold start, environment hardening, adversarial screening, verifier cross-check로 reward hacking을 억제했다고 설명한다. 저자 보고 평가는 DeepSWE v1.1 $71.9$, Toolathlon-Verified $76.9$, OSWorld-Verified $82.0$, MiMo VisualCoding $72.3$ 등이며, 이는 제공자가 선택한 harness와 조건에서 나온 수치다.

**엔지니어 인사이트 (Impact):** RL compute만 키우는 것이 아니라 환경 다양성·grader compute·검증기를 함께 확장하는 흐름이 뚜렷하다. 오늘 배운 관점에서는 GRS·GAR가 $J_R$을 더 세밀하게 만드는 시도이고, environment hardening·verifier cross-check는 보상 해킹을 줄이는 방어선이다. 그러나 공식 카드에는 안전 비용별 한도, 위반 UCB, 독립 canary 결과가 제약 최적화 형태로 공개되어 있지 않다. “정렬된 RL”이라는 문구를 hard safety guarantee로 읽어서는 안 된다.

또한 open-weight와 가벼운 self-hosting은 같은 말이 아니다. 공식 SGLang 권장 예시는 2 nodes, tensor parallel $16$, data parallel $2$, expert parallel $16$을 사용한다. 이는 최소 요구사항의 증명이 아니라 제공자가 제시한 한 배포 구성이다. 가중치 접근성은 연구·감사 가능성을 넓히지만, 실제 재현에는 대규모 하드웨어와 custom code 신뢰 경계가 필요하다. [공식 기술 보고서](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)의 학습·평가 조건을 확인하고, 공급사 수치는 독립 benchmark와 운영 비용 측정으로 다시 검증해야 한다.

### 5.2 VLM-in-Sandbox: 모델보다 먼저 “시각 증거 상태”에 예산을 걸다

[VLM-in-Sandbox 사전논문(arXiv v1, 2026-09-21 UTC)](https://arxiv.org/abs/2609.24362)은 학습 없이 VLM 에이전트의 중간 시각 증거를 관리하는 **Visual Workspace**를 제안한다. VLM은 crop, mask, overlay, plot처럼 새 이미지를 만들 수 있지만, 모두 대화에 계속 붙이면 visual token이 누적되고 파일로만 남기면 모델이 보지 못한다.

제안된 runtime은 다음 세 부품으로 이 문제를 푼다.

1. **Image Ledger:** 각 이미지에 stable ID, 경로, provenance(어디서 만들어졌는지), parent–child 관계를 기록한다.
2. **Active Visual Context:** 다음 요청에 실제 이미지로 넣을 자리를 `focus`와 `aux` 두 슬롯으로 제한한다.
3. **Promote:** 모델이 ledger의 특정 자산을 명시적으로 골라 두 슬롯 중 하나로 다시 올린다. 나머지는 메타데이터와 파일로 남는다.

Prompt compiler는 최근 3개 action–observation turn, 더 오래된 기록의 결정적 요약, ledger, active slots, 남은 step budget으로 매 요청을 다시 만든다. 즉 “이미지를 생성하는 일”과 “다음 추론에서 어떤 이미지를 보게 할지”를 분리한다.

저자들은 모델마다 7개 benchmark의 $6{,}350$-example collection을 사용해 총 4개 VLM에서 Vanilla VLM·append-only sandbox·제안법을 비교했다. 모든 모델의 sample-weighted average에서 제안법이 가장 높았고, GPT-4.1-mini를 사용한 compiler-matched $2\times2$ 실험 $N=1{,}260$에서는 자동으로 모든 이미지를 유지하는 matched control보다 정확도가 $1.83$%p 높고 total tokens가 $18.6\%$ 적었다. 정확도 차이의 paired bootstrap $95\%$ CI는 $[0.40,3.25]$%p, exact McNemar $p=0.014$였다. 두 슬롯에서 “모델이 고르게 하는 효과”만 recency와 비교한 $+0.63$%p는 CI가 $[-0.16,1.51]$%p로 $0$을 포함해 통계적으로 확정적이지 않았다. token 절감의 더 큰 원인은 선택 자체보다 bounded retention이었다.

한계도 중요하다. Tool-enabled reasoning은 단일 Vanilla call보다 여전히 대략 한 order of magnitude, 즉 약 $10$배 규모의 total tokens를 쓰며, 실험은 static image에 한정됐다. Qwen3.5-9B는 Vanilla 대비 $734$ rescues와 동시에 $411$ regressions를 만들었고 RealWorldQA에서는 긴 artifact-heavy trajectory와 함께 음의 셀이 있었다. 시각 증거를 잘 골라도 해석을 틀릴 수 있다. arXiv v1 논문 자체는 CC BY 4.0이지만, 해당 버전의 메타데이터와 본문에는 저자의 구현 저장소가 제시되지 않았다. 공개 논문의 라이선스와 즉시 재현 가능한 오픈소스 구현을 구분해야 한다.

**엔지니어 인사이트 (Impact):** 이 연구의 두 active slots는 학습의 KL budget과 대상은 다르지만, 둘 다 “더 많은 자유가 항상 더 좋은가?”에 **명시적 예산**으로 답한다. 멀티모달 에이전트의 품질은 모델 가중치뿐 아니라 어떤 증거를 상태로 보존·노출하는지에 좌우된다. 실제 도입 canary에서는 정답률만 보지 말고 total/uncached tokens, time-to-first-token, end-to-end latency, tool failure, rescue와 regression을 함께 제약 지표로 둬야 한다. Docker sandbox도 파일·네트워크·권한 policy가 올바를 때만 방어선이므로, “sandbox 사용” 자체를 안전 보증으로 간주해서는 안 된다.

두 소식의 공통 방향은 선명하다. MiMo-V2.6은 학습 단계에서 환경·grader·RL을 함께 키우고, VLM-in-Sandbox는 추론 단계에서 증거 상태를 제한하고 선택한다. 다음 세대 에이전트의 경쟁력은 모델 크기 하나보다 **학습 신호, 상태 예산, 실행 권한, 검증·배포 제약을 한 시스템으로 조립하는 능력**에서 갈릴 가능성이 크다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

**질문:** 한 LLM 에이전트 후보가 PPO-Clip 학습에서 평균 KL $0.008$로 목표 $0.01$보다 작고, 오프라인 추정 성공률도 기준보다 $3$%p 높다. 그러나 권한 위반 비용 추정치는 한도 $0.1\%$에 가까운 $0.09\%$다. 이 숫자만으로 전체 배포해도 될까? 오늘 배운 수학과 배포 절차를 이용해 답하라.

**모범 답안:** 전체 배포하면 안 된다. 첫째, PPO clipping과 평균 KL $0.008$은 샘플된 평균 정책 변화가 작다는 신호일 뿐 hard KL trust region이나 모든 프롬프트의 작은 변화를 보장하지 않는다. 세그먼트별·도구별 p95와 최대 KL, 구조화 행동 변화도 확인해야 한다. 둘째, 권한 위반의 점추정치 $0.09\%$가 한도 $0.1\%$ 아래라는 사실만으로 제약 충족을 주장할 수 없다. 독립 로그에서 비용 UCB가 한도 아래인지, ESS와 지원집합이 충분한지, 여러 후보를 고른 선택 편향이 없는지 확인한다. 필요하면 $J_C\le d$를 명시한 CPO 또는 Lagrangian update와 실행 직전 permission shield를 함께 쓴다.

오프라인 게이트를 통과한 뒤에도 shadow에서 실제 parser·tool router·side effect를 점검하고, 사용자 단위로 무작위 배정한 작은 canary를 시작한다. 품질 차이의 time-uniform LCB가 최소 개선폭을 넘고 권한 위반 차이의 time-uniform UCB가 허용 증가량 아래이며, 미리 정한 최소 표본·기간과 지연 보상 완료율을 만족할 때만 단계적으로 확대한다. 위반 증거가 생기면 즉시 rollback한다. 요약하면 **KL은 업데이트 크기, 비용 제약은 허용 영역, confidence sequence는 온라인 증거의 시간축**을 각각 담당한다.

> **다음 연결:** 오늘 사용한 단순 alpha-spending confidence sequence를 더 날카로운 e-process·martingale 검정으로 확장하고, 희귀 안전 사건의 zero-count 문제와 delayed outcome을 다루면 실제 canary 관측 설계를 한 단계 더 깊게 이해할 수 있다.
