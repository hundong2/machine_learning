# [2026-10-27] 오늘 학습: 변분추론·ELBO·상각 스무딩 & 연합학습·보안 집계·검증 가능한 연합 삭제

> **오늘의 핵심 문장:** 파티클 방법이 숨은 경로를 표본으로 추적한다면, 변분추론은 계산하기 쉬운 분포를 최적화해 사후분포를 근사한다. 상각 추론은 새 데이터마다 처음부터 최적화하지 않고 하나의 추론 네트워크가 반복 계산을 대신한다. 연합학습도 원본 데이터를 한곳에 모으지 않고 업데이트를 모으지만, 보안 집계는 업데이트를 숨길 뿐 개인정보보호나 삭제를 자동 보장하지 않는다. 빠른 추론과 삭제 가능한 분산 학습은 모두 **무엇을 근사했고, 무엇을 숨겼으며, 어떤 오차와 계보가 남았는가**를 기록해야 검증할 수 있다.

지난 시간에는 파티클 스무딩으로 미래 관측을 이용해 과거 경로를 고치고, Rao–Blackwellization으로 계산 가능한 불확실성을 적분했으며, Particle MCMC로 모수의 사후분포까지 겨냥했다. 또한 DP-SGD·SISA·lineage DAG를 통해 개인정보 위험과 삭제 증거가 서로 다른 문제임을 배웠다. 오늘은 그 다음 두 질문을 다룬다.

1. 매번 파티클 필터나 최적화를 다시 돌리지 않고도 새 관측의 사후분포를 빠르게 구할 수 있을까?
2. 데이터가 여러 기기·기관에 흩어져 있을 때 원본을 모으지 않고 학습하면서, 개별 업데이트 보호·누적 개인정보 예산·삭제 가능한 계보까지 함께 설계할 수 있을까?

두 트랙은 다음과 같은 공통 구조를 가진다.

$$
\underbrace{\text{비싼 정확 추론}}_{\text{MCMC·particle smoothing}}
\longrightarrow
\underbrace{\text{학습된 근사 추론}}_{\text{variational·amortized smoothing}}
\longrightarrow
\underbrace{\text{빠른 반복 사용}}_{\text{새 시계열에 한 번의 순전파}}
$$

$$
\underbrace{\text{분산된 원본 데이터}}_{\text{client}}
\longrightarrow
\underbrace{\text{보호된 합}}_{\text{secure aggregation·DP}}
\longrightarrow
\underbrace{\text{전역 모델과 파생물}}_{\text{checkpoint·adapter·index·endpoint}}.
$$

> **난이도 표지:** 변분추론은 복잡한 산맥의 정확한 지도를 매번 그리는 대신, 중요한 지형을 담는 접이식 지도를 학습하는 일이다. 상각 추론은 도시마다 새 지도를 손으로 그리지 않고 지도 제작 기계를 만드는 일이다. 연합학습은 여러 병원이 환자 기록을 보내는 대신 각자 계산한 학습 방향을 합치는 일이다. 보안 집계는 봉투를 개별로 열지 않고 합계만 읽게 하지만, 합계가 안전한지와 과거 기여가 삭제됐는지는 별도의 계약이다.

---

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 왜 정확한 사후분포는 자주 계산하기 어려울까?

관측 $x$와 잠재변수 $z$의 결합분포 $p_\theta(x,z)$를 안다고 하자. 관측 뒤 잠재변수에 대한 믿음인 posterior는

$$
p_\theta(z\mid x)
=
\frac{p_\theta(x,z)}{p_\theta(x)}
$$

이다. 문제는 분모인 evidence다.

$$
p_\theta(x)
=
\int p_\theta(x,z)\,dz.
$$

$z$가 고차원 연속변수이거나 긴 시계열 $z_{0:T}$이면 이 적분은 닫힌형으로 계산되지 않는 경우가 많다. 지난 시간의 Particle MCMC는 올바른 목표분포를 겨냥할 수 있지만, 모수 후보마다 파티클 필터를 다시 돌려야 한다. 정확성은 강력하지만 새 데이터가 들어올 때마다 반복할 계산비가 크다.

### 1.2 왜 “분포를 최적화한다”는 생각이 필요했을까?

변분추론(Variational Inference, VI)은 계산하기 어려운 posterior를 직접 적분하는 대신, 다루기 쉬운 후보분포 집합 $\mathcal Q$ 안에서 가장 비슷한 분포 $q$를 찾는다.

$$
q^*
=
\arg\min_{q\in\mathcal Q}
D_{\mathrm{KL}}
\left(
q(z)\,\|\,p_\theta(z\mid x)
\right).
$$

적분 문제를 최적화 문제로 바꾼 셈이다. 신경망·자동미분·미니배치 경사하강법을 사용할 수 있다는 장점이 있지만, $\mathcal Q$가 표현하지 못하는 모양은 복원할 수 없다. 즉, 계산비를 줄이는 대신 **근사 family의 가정**이라는 빚을 진다.

### 1.3 왜 데이터마다 별도 최적화를 하지 않을까?

고전적인 VI는 관측 $x_i$마다 별도의 변분 파라미터 $\lambda_i$를 최적화할 수 있다. 데이터가 백만 개라면 백만 번의 작은 최적화가 필요하다. 상각 추론(amortized inference)은 하나의 encoder가

$$
x_i
\longmapsto
\phi
\longmapsto
q_\phi(z_i\mid x_i)
$$

를 예측하게 한다. 학습 때 큰 비용을 미리 지불하고, 새 사례에서는 한 번의 순전파로 근사 posterior를 얻는다. 대출 원금을 여러 달에 나누어 갚듯, 개별 추론 비용을 공유 파라미터 $\phi$에 “상각”한다.

다만 속도는 무료가 아니다. 제한된 함수 근사기와 학습 절차 때문에 각 사례에 최적인 $q_i^*$에 도달하지 못하는 **상각 오차**가 추가된다.

### 1.4 왜 시계열에서는 상각 “스무딩”이 필요한가?

필터링은 $z_t$를 추론할 때 $y_{1:t}$만 사용한다. 스무딩은 미래 관측 $y_{t+1:T}$까지 사용한다.

$$
p(z_t\mid y_{1:t})
\neq
p(z_t\mid y_{1:T})
\quad\text{일반적으로}.
$$

가려진 보행자가 몇 프레임 뒤 다시 나타났다면 미래 프레임은 과거 위치를 고치는 중요한 증거다. Bidirectional encoder나 전체 시퀀스 encoder가 $y_{1:T}$를 읽고 $q_\phi(z_{0:T}\mid y_{1:T})$를 곧바로 출력하도록 학습하면, 매 시퀀스마다 FFBSi를 반복하지 않고 빠른 근사 smoothing을 할 수 있다. 그러나 미래를 읽은 결과를 실시간 예측이라고 부르면 데이터 누수다.

### 1.5 왜 연합학습이 필요했을까?

스마트폰의 입력 기록, 병원의 환자 자료, 공장의 센서 로그처럼 원본 데이터를 중앙 서버로 옮기기 어렵거나 옮겨서는 안 되는 상황이 있다. 연합학습(Federated Learning, FL)은 서버가 전역 모델을 배포하고, 각 client가 로컬 데이터로 계산한 업데이트를 보내며, 서버가 이를 합쳐 다음 전역 모델을 만드는 방식이다.

하지만 “원본을 보내지 않았다”와 “정보가 새지 않는다”는 같은 문장이 아니다. Gradient와 model delta에도 학습 사례의 정보가 담길 수 있다. 참여자가 서로 다른 분포를 가지면 단순 평균은 편향되거나 불안정할 수 있고, 악성 client는 poisoning update를 보낼 수도 있다.

### 1.6 왜 보안 집계와 차등 개인정보보호가 둘 다 필요한가?

보안 집계(Secure Aggregation)는 서버가 각 client의 업데이트를 보지 못하고 합만 복원하게 하는 암호 프로토콜이다. 차등 개인정보보호(Differential Privacy, DP)는 공개된 학습 결과분포가 한 privacy unit의 참여 여부에 크게 의존하지 않도록 clipping과 무작위 noise로 제한한다.

둘의 질문은 다르다.

$$
\begin{aligned}
\text{Secure aggregation: }&
\text{서버가 전송 중 무엇을 볼 수 있는가?}\\
\text{Differential privacy: }&
\text{공개 결과에서 한 참여자의 영향을 얼마나 추론할 수 있는가?}
\end{aligned}
$$

암호화된 합도 cohort가 너무 작거나 여러 round의 합을 차분하면 정보를 드러낼 수 있다. 반대로 DP noise가 있어도 서버가 개별 raw update를 본다면 전송 단계의 노출은 남는다. threat model에 따라 두 기술을 결합해야 한다.

### 1.7 왜 연합 삭제는 중앙 삭제보다 더 어려울까?

Client $u$가 round $r$에 보낸 업데이트를 나중에 찾을 수 있다고 해도 단순히 빼면 충분하지 않다. 이후 client들은 이미 $u$의 영향이 섞인 $w_{r+1}$에서 학습했으므로, 뒤의 모든 업데이트 경로가 달라졌다.

$$
\Delta_s^{(k)}
=
\Delta_s^{(k)}(w_s,D_k),
\qquad s>r.
$$

$w_s$가 바뀌면 $\Delta_s^{(k)}$도 바뀐다. 더구나 보안 집계는 개별 업데이트를 애초에 복원하지 못하게 설계될 수 있다. 따라서 정확한 삭제를 원한다면 첫 영향 이전의 checkpoint, 참여 cohort와 데이터 snapshot, code·seed·optimizer state, privacy accountant, 파생 adapter·index·cache를 연결한 분산 lineage가 필요하다.

### 1.8 오늘의 본질

오늘의 두 트랙은 “요약”의 힘과 한계를 함께 가르친다.

$$
\boxed{
\text{요약은 계산과 노출을 줄이지만, 버린 정보와 남은 오차를 자동으로 증명하지 않는다.}
}
$$

- $q_\phi$는 복잡한 posterior를 요약하지만 approximation gap과 amortization gap을 남긴다.
- Secure sum은 개별 update를 숨기지만 DP·무결성·삭제를 보장하지 않는다.
- 전역 checkpoint는 많은 client의 학습을 요약하지만, 삭제 영향의 경계를 보존하려면 별도 lineage가 필요하다.

---

## 2. 친절한 용어 사전

| 용어·기호 | 초보자 해설 | 오늘의 역할 |
| --- | --- | --- |
| $x$ | 관측한 한 사례 | 이미지·문장·센서열처럼 실제로 입력된 데이터다. |
| $y_{1:T}$ | 시점 $1$부터 $T$까지의 관측열 | 상각 스무딩 encoder가 읽는 전체 증거다. |
| $z$ | 직접 보이지 않는 잠재변수 | 관측을 만든 숨은 요인·표현이다. |
| $z_{0:T}$ | 시간에 따라 변하는 잠재상태 경로 | 동작·위치·장면 상태의 숨은 궤적이다. |
| $\theta$ | 생성모형의 파라미터 | $p_\theta(x,z)$가 데이터를 만드는 규칙을 정한다. |
| $\phi$ | 추론모형의 파라미터 | $q_\phi(z\mid x)$가 posterior를 얼마나 잘 흉내 내는지 정한다. |
| prior $p(z)$ | 관측 전 잠재변수에 대한 믿음 | posterior가 과도하게 제멋대로 되지 않게 기준을 준다. |
| likelihood $p_\theta(x\mid z)$ | 잠재변수 $z$가 주어졌을 때 관측 $x$의 확률·밀도 | decoder가 관측을 얼마나 잘 설명하는지 나타낸다. |
| posterior $p_\theta(z\mid x)$ | 관측 뒤 갱신한 잠재변수의 분포 | 계산하고 싶은 목표다. |
| evidence $p_\theta(x)$ | 잠재변수를 모두 적분한 관측의 주변확률 | posterior의 정규화 상수이며 대개 계산이 어렵다. |
| variational family $\mathcal Q$ | 계산 가능한 후보분포 모음 | Gaussian·normalizing flow 등 허용할 근사 형태를 정한다. |
| $q_\phi(z\mid x)$ | 참 posterior를 흉내 내는 분포 | encoder가 파라미터를 출력한다. |
| $D_{\mathrm{KL}}(q\|p)$ | $q$가 $p$와 얼마나 다른지 재는 비대칭 발산 | 음수가 아니지만 거리 metric은 아니다. |
| ELBO $\mathcal L$ | Evidence Lower BOund | $\log p_\theta(x)$보다 크지 않은 학습 목적함수다. |
| encoder | 관측에서 근사 posterior의 파라미터를 만드는 함수 | $\mu_\phi(x)$와 $\sigma_\phi(x)$ 등을 출력한다. |
| decoder | 잠재변수에서 관측분포를 정의하는 함수 | $p_\theta(x\mid z)$를 계산한다. |
| amortization | 여러 사례의 추론 비용을 공유 파라미터에 나누어 미리 지불 | 새 사례의 추론을 빠르게 한다. |
| approximation gap | 선택한 $\mathcal Q$가 참 posterior를 담지 못해 생기는 오차 | 더 유연한 family로 줄일 수 있다. |
| amortization gap | encoder가 사례별 최적 $q_x^*$를 출력하지 못해 생기는 추가 오차 | 더 좋은 encoder·학습·test-time refinement로 줄일 수 있다. |
| reparameterization | 무작위 표본을 고정 잡음과 미분 가능한 함수로 다시 표현 | 확률 노드를 통과해 backpropagation하게 한다. |
| $\odot$ | 성분별 곱 | Gaussian 표본 $z=\mu+\sigma\odot\epsilon$에 쓰인다. |
| score-function estimator | $\nabla_\phi\log q_\phi$를 이용하는 확률적 gradient 추정법 | 이산 잠재변수에도 적용할 수 있지만 분산이 클 수 있다. |
| smoothing | 미래 관측까지 사용해 과거 상태를 추론 | $p(z_t\mid y_{1:T})$가 목표다. |
| fixed-lag smoothing | 미래를 $L$시점까지만 기다려 과거를 수정 | 정확도와 지연을 교환한다. |
| inference network | 입력에서 변분분포 파라미터를 내는 신경망 | 상각 추론기의 다른 이름이다. |
| client | 자신의 로컬 데이터와 계산을 가진 참여자 | 휴대전화·병원·조직 하나가 될 수 있다. |
| server | 전역 모델을 배포하고 업데이트를 집계하는 주체 | 원본 데이터를 반드시 보유할 필요는 없다. |
| round $r$ | 전역 배포부터 로컬 학습·집계까지 한 주기 | privacy budget과 lineage의 시간 단위다. |
| local step $E$ | 한 round 안에서 client가 수행하는 최적화 횟수 | 많을수록 통신은 줄지만 client drift가 커질 수 있다. |
| non-IID | client별 데이터분포가 서로 동일하지 않은 상태 | 병원별 환자군·기기별 사용 습관 차이다. |
| FedAvg | 로컬 학습 결과를 가중 평균하는 대표적 FL 알고리즘 | 전역 모델을 갱신한다. |
| client drift | 로컬 목적함수를 여러 step 따라가며 전역 방향에서 벗어나는 현상 | non-IID와 큰 $E$에서 심해질 수 있다. |
| model delta $\Delta^{(k)}$ | client $k$의 학습 전후 모델 차이 | 서버에 보낼 수 있는 업데이트 표현이다. |
| secure aggregation | 개별 update 대신 정해진 집계값만 복원하는 암호 프로토콜 | 전송·집계 단계의 기밀성을 높인다. |
| mask | 실제 update에 더하는 상쇄용 무작위 벡터 | 모든 client의 합에서는 사라지도록 만든다. |
| cohort $S_r$ | round $r$에 선택된 client 집합 | 가중치·보안 임계값·DP sampling을 정한다. |
| dropout | 선택됐지만 집계 완료 전에 이탈한 client | mask 복구와 noise 보장을 어렵게 한다. |
| clipping bound $C$ | update의 $\ell_2$ 노름 상한 | 한 client가 합에 미치는 최대 영향을 제한한다. |
| client-level DP | client 한 명의 전체 기여를 privacy unit으로 보는 DP | record-level DP보다 더 큰 단위를 보호한다. |
| record-level DP | 레코드 한 건을 privacy unit으로 보는 DP | 한 client의 전체 참여 보호와 같지 않다. |
| privacy accountant | 여러 round의 sampling·noise를 누적해 $(\varepsilon,\delta)$를 계산 | privacy budget의 장부다. |
| add/remove adjacency | client 하나가 있거나 없는 데이터셋을 비교 | clipping된 합의 민감도가 보통 $C$가 된다. |
| replace-one adjacency | 한 client의 기여를 다른 하나로 교체한 데이터셋을 비교 | clipping만으로는 민감도가 최대 $2C$가 될 수 있다. |
| federated unlearning | 특정 client·record의 영향을 연합 모델에서 제거하려는 절차 | 중앙 raw data 없이 삭제 증거를 만들어야 한다. |
| exact unlearning | 삭제 후 처음부터 다시 학습한 결과분포와 정한 의미에서 같은 절차 | 매우 강하지만 비쌀 수 있다. |
| approximate removal | 정한 거리·공격·certificate 아래 재학습 기준선에 가깝다고 주장 | 평가 범위와 실패 조건을 명시해야 한다. |
| lineage DAG | 산출물의 부모–자식 관계를 나타낸 방향성 비순환 그래프 | 삭제 영향의 전이적 폐쇄를 찾는다. |
| immutable manifest | 변경 대신 새 버전을 만드는 실행·데이터·산출물 기록 | 나중에 어떤 입력으로 무엇을 만들었는지 증명한다. |
| content hash | 내용에서 계산한 짧은 지문 | artifact 동일성과 변조를 확인한다. |
| tombstone | 더 이상 사용하면 안 된다는 폐기 표식 | 파일 삭제와 별개로 배포 금지 상태를 기록한다. |
| transitive invalidation | 영향받은 노드의 모든 후손을 무효화 | adapter·index·cache까지 삭제 범위를 확장한다. |

---

## 3. 수학의 해부학 (증명과 원리)

### 3.1 잠재변수 모형과 계산하기 어려운 분모

생성모형은 잠재변수 $z$를 먼저 뽑고 관측 $x$를 만든다고 가정한다.

$$
p_\theta(x,z)
=
p(z)p_\theta(x\mid z).
$$

Bayes 법칙에 따라

$$
p_\theta(z\mid x)
=
\frac{p(z)p_\theta(x\mid z)}
{\int p(z')p_\theta(x\mid z')\,dz'}.
$$

분모의 적분은 $z$의 모든 가능한 값을 합친다. 차원이 커지면 격자 적분은 차원의 저주를 겪고, 비선형 decoder에서는 Gaussian 적분 공식도 쓸 수 없다. VI는 $p_\theta(z\mid x)$ 대신 계산·표본추출·밀도평가가 쉬운 $q_\phi(z\mid x)$를 둔다.

### 3.2 KL 발산이 음수가 아닌 이유

연속분포의 KL 발산은

$$
D_{\mathrm{KL}}(q\|p)
=
\int q(z)
\log\frac{q(z)}{p(z)}
\,dz
$$

이다. $q(z)>0$인 곳에서 $p(z)=0$이면 KL은 무한대다. 유한한 경우 $p$가 $q$를 지지한다고 가정하고, 오목함수 $\log$에 Jensen 부등식을 적용하면

$$
\begin{aligned}
-D_{\mathrm{KL}}(q\|p)
&=
\mathbb E_q
\left[
\log\frac{p(z)}{q(z)}
\right]\\
&\le
\log
\mathbb E_q
\left[
\frac{p(z)}{q(z)}
\right]\\
&=
\log
\int_{\{z:q(z)>0\}}
p(z)\,dz\\
&\le
0.
\end{aligned}
$$

따라서

$$
D_{\mathrm{KL}}(q\|p)\ge 0.
$$

두 분포가 거의 모든 곳에서 같을 때만 등호가 성립한다. 하지만 대칭은 아니다.

$$
D_{\mathrm{KL}}(q\|p)
\neq
D_{\mathrm{KL}}(p\|q)
\quad\text{일반적으로}.
$$

### 3.3 ELBO 항등식의 단계별 유도

관측 $x$를 고정하고 표기를 줄이면

$$
\begin{aligned}
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\|
p_\theta(z\mid x)
\right)
&=
\mathbb E_{q_\phi}
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(z\mid x)
\right].
\end{aligned}
$$

Bayes 식

$$
\log p_\theta(z\mid x)
=
\log p_\theta(x,z)
-
\log p_\theta(x)
$$

을 대입한다.

$$
\begin{aligned}
D_{\mathrm{KL}}(q_\phi\|p_\theta)
&=
\mathbb E_{q_\phi}
\left[
\log q_\phi(z\mid x)
-
\log p_\theta(x,z)
+
\log p_\theta(x)
\right]\\
&=
\log p_\theta(x)
-
\mathbb E_{q_\phi}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
\end{aligned}
$$

기댓값 부분을 ELBO라 정의한다.

$$
\mathcal L(\theta,\phi;x)
=
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right].
$$

그러면 가장 중요한 항등식이 나온다.

$$
\boxed{
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)
\|
p_\theta(z\mid x)
\right)
}
$$

KL이 음수가 아니므로

$$
\mathcal L(\theta,\phi;x)
\le
\log p_\theta(x).
$$

즉 ELBO는 log evidence의 하한이다. $\theta$가 고정돼 있을 때 ELBO를 $\phi$에 대해 최대화하는 일은 reverse KL을 최소화하는 일과 같다. 그러나 $\theta$도 함께 바뀌면 생성모형 자체도 움직인다는 점을 잊으면 안 된다.

### 3.4 Jensen 부등식으로 다시 보는 ELBO

같은 결과를 더 직접적으로 볼 수 있다. 이 유도에서는 $p_\theta(x,z)>0$인 곳에서 $q_\phi(z\mid x)>0$이라고 가정한다. 즉 $q_\phi$가 결합분포의 관련 support를 모두 덮어 importance ratio가 정의된다고 가정한다. 앞 절의 ELBO–KL 항등식 자체에는 상황에 따라 이보다 약한 조건을 사용할 수 있다.

$$
\begin{aligned}
\log p_\theta(x)
&=
\log\int p_\theta(x,z)\,dz\\
&=
\log\int
q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\,dz\\
&=
\log
\mathbb E_{q_\phi}
\left[
\frac{p_\theta(x,z)}
{q_\phi(z\mid x)}
\right]\\
&\ge
\mathbb E_{q_\phi}
\left[
\log p_\theta(x,z)
-
\log q_\phi(z\mid x)
\right]\\
&=
\mathcal L(\theta,\phi;x).
\end{aligned}
$$

Jensen의 등호는

$$
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
$$

가 $q_\phi$ 아래 거의 확실하게 상수일 때 성립하며, 이는 $q_\phi(z\mid x)=p_\theta(z\mid x)$와 연결된다.

### 3.5 재현 항과 prior 정규화 항

$p_\theta(x,z)=p(z)p_\theta(x\mid z)$를 대입하면

$$
\begin{aligned}
\mathcal L
&=
\mathbb E_{q_\phi}
\left[
\log p_\theta(x\mid z)
+
\log p(z)
-
\log q_\phi(z\mid x)
\right]\\
&=
\underbrace{
\mathbb E_{q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
}_{\text{관측 재현}}
-
\underbrace{
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\|p(z)
\right)
}_{\text{prior와의 괴리}}.
\end{aligned}
$$

첫 항은 $z$가 $x$를 잘 설명하게 한다. 둘째 항은 각 입력의 posterior가 prior에서 무한히 멀어지는 것을 막는다. VAE에서는 흔히 encoder가 $q_\phi(z\mid x)$를, decoder가 $p_\theta(x\mid z)$를 구현한다.

“재현 오차 + 정규화”라는 말은 직관에는 좋지만 likelihood에 따라 정확한 형태가 달라진다. 예를 들어

$$
p_\theta(x\mid z)
=
\mathcal N
\left(
x;f_\theta(z),\tau^2I
\right)
$$

이면 상수항을 제외한 음의 log likelihood는

$$
-\log p_\theta(x\mid z)
=
\frac{1}{2\tau^2}
\|x-f_\theta(z)\|_2^2
+
\text{constant}.
$$

따라서 MSE처럼 보이지만, 분산 $\tau^2$가 가중치를 결정한다.

### 3.6 Reverse KL과 mode-seeking의 정확한 의미

VI에서 자주 최소화하는 것은

$$
D_{\mathrm{KL}}(q\|p)
$$

이다. $q(z)>0$인데 $p(z)$가 매우 작으면 큰 벌점을 받는다. 반면 $p$가 큰 어느 mode에 $q$가 질량을 두지 않아도, 그 영역은 $q$의 기댓값에 거의 나타나지 않는다. 제한된 unimodal family가 여러 mode의 posterior를 근사하면 하나의 mode를 선택하는 경향이 생길 수 있어 mode-seeking이라 부른다.

그러나 이것은 모든 모형·family·최적화에서 반드시 mode 하나만 고른다는 정리가 아니다. “reverse KL의 적분 가중치가 $q$에 있다”는 구조적 경향이다. 더 유연한 mixture나 flow를 사용하면 approximation gap을 줄일 수 있다.

### 3.7 대각 Gaussian KL의 닫힌형

VAE에서 자주 쓰는

$$
q_\phi(z\mid x)
=
\mathcal N
\left(
z;\mu,\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2)
\right),
\qquad
p(z)=\mathcal N(0,I)
$$

를 생각하자. 두 Gaussian의 KL 공식을 적용하면

$$
\boxed{
D_{\mathrm{KL}}(q_\phi\|p)
=
\frac{1}{2}
\sum_{j=1}^{d}
\left(
\mu_j^2
+
\sigma_j^2
-
\log\sigma_j^2
-
1
\right)
}
$$

이다. $\mu=0$, $\sigma_j^2=1$이면 KL은 $0$이다. 구현에서는 양수인 표준편차를 직접 출력하기보다

$$
\ell_j=\log\sigma_j^2,
\qquad
\sigma_j=\exp\left(\frac{\ell_j}{2}\right)
$$

처럼 log variance를 출력하거나 softplus로 양수를 만든다.

### 3.8 재매개변수화로 확률 노드를 미분하기

목표에

$$
\mathbb E_{q_\phi(z\mid x)}[f_\theta(z)]
$$

가 있으면 표본 $z$의 분포 자체가 $\phi$에 의존한다. 대각 Gaussian에서는

$$
\epsilon\sim\mathcal N(0,I),
\qquad
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
$$

로 다시 쓸 수 있다. 무작위성은 파라미터와 무관한 $\epsilon$으로 옮겨지고, $z$는 $\phi$의 미분 가능한 함수가 된다.

적절한 미분·적분 교환 조건 아래

$$
\begin{aligned}
\nabla_\phi
\mathbb E_{q_\phi(z\mid x)}
[f_\theta(z)]
&=
\nabla_\phi
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[
f_\theta
\left(
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
\right)
\right]\\
&=
\mathbb E_{\epsilon}
\left[
\nabla_\phi
f_\theta
\left(
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
\right)
\right].
\end{aligned}
$$

한두 개의 $\epsilon$ 표본으로 Monte Carlo 근사하고 backpropagation할 수 있다. 이산 잠재변수에는 이 Gaussian 공식을 그대로 적용할 수 없다. Score-function estimator, 정확한 합산, Gumbel 계열 relaxation 등 별도 선택이 필요하다.

### 3.9 상각 오차는 어디에서 생길까?

생성 파라미터 $\theta$와 variational family $\mathcal Q$를 고정하자. 각 사례 $x$를 개별 최적화했을 때의 최선 후보를

$$
q_x^*
=
\arg\max_{q\in\mathcal Q}
\mathcal L(q;x)
$$

라 두자. Encoder가 출력한 $q_\phi(\cdot\mid x)$의 총 inference gap은

$$
\log p_\theta(x)
-
\mathcal L(q_\phi;x)
$$

이다. 중간에 $\mathcal L(q_x^*;x)$를 더하고 빼면

$$
\begin{aligned}
\log p_\theta(x)
-
\mathcal L(q_\phi;x)
&=
\underbrace{
\log p_\theta(x)
-
\mathcal L(q_x^*;x)
}_{\text{approximation gap}}
\\
&\quad+
\underbrace{
\mathcal L(q_x^*;x)
-
\mathcal L(q_\phi;x)
}_{\text{amortization gap}}.
\end{aligned}
$$

첫 항은 $\mathcal Q$ 자체의 표현 한계다. 둘째 항은 공유 encoder가 해당 사례의 최적 후보에 도달하지 못한 정도다. 더 유연한 posterior family는 첫 항을, 더 나은 encoder·optimizer·test-time refinement는 둘째 항을 줄이는 방향이다. 실제로 $q_x^*$의 전역 최적값을 정확히 구하기 어렵다면 이 분해도 추정치라는 점을 밝혀야 한다.

### 3.10 시계열 생성모형의 결합분포

Markov 상태공간모형을

$$
p_\theta(z_{0:T},y_{1:T})
=
p_\theta(z_0)
\prod_{t=1}^{T}
p_\theta(z_t\mid z_{t-1})
p_\theta(y_t\mid z_t)
$$

로 쓰자. 정확한 smoothing posterior

$$
p_\theta(z_{0:T}\mid y_{1:T})
$$

가 어렵다면 전체 관측열을 읽는 근사분포를 둔다.

$$
q_\phi(z_{0:T}\mid y_{1:T})
=
q_\phi(z_0\mid c_0)
\prod_{t=1}^{T}
q_\phi(z_t\mid z_{t-1},c_t),
$$

여기서

$$
c_t=h_\phi(y_{1:T},t)
$$

는 bidirectional RNN·Transformer 같은 encoder가 만든 문맥이다. 이 factorization은 한 가지 설계 예시일 뿐 유일한 정답이 아니다. Reverse-time factorization, structured Gaussian, flow 등 다른 선택도 가능하다.

### 3.11 Sequence ELBO 전개

일반 ELBO에 시계열 결합분포를 넣으면

$$
\begin{aligned}
\mathcal L_{\mathrm{seq}}
&=
\mathbb E_q
\Bigg[
\log p_\theta(z_0)
+
\sum_{t=1}^{T}
\log p_\theta(z_t\mid z_{t-1})
+
\sum_{t=1}^{T}
\log p_\theta(y_t\mid z_t)\\
&\qquad\qquad
-
\log q_\phi(z_0\mid c_0)
-
\sum_{t=1}^{T}
\log q_\phi(z_t\mid z_{t-1},c_t)
\Bigg].
\end{aligned}
$$

항을 묶으면

$$
\begin{aligned}
\mathcal L_{\mathrm{seq}}
&=
\sum_{t=1}^{T}
\mathbb E_q
\left[
\log p_\theta(y_t\mid z_t)
\right]\\
&\quad-
D_{\mathrm{KL}}
\left(
q_\phi(z_0\mid c_0)
\|
p_\theta(z_0)
\right)\\
&\quad-
\sum_{t=1}^{T}
\mathbb E_{q(z_{t-1}\mid y_{1:T})}
\left[
D_{\mathrm{KL}}
\left(
q_\phi(z_t\mid z_{t-1},c_t)
\|
p_\theta(z_t\mid z_{t-1})
\right)
\right].
\end{aligned}
$$

첫 줄은 각 시점의 관측을 설명하게 한다. 나머지는 근사 초기분포와 전이를 생성모형의 prior·transition에 맞춘다. 한 번 학습한 $\phi$는 새 시퀀스에도 빠르게 적용되지만, 학습분포 밖의 길이·동역학·센서 결손에서는 오차가 커질 수 있다.

### 3.12 Filtering·smoothing·fixed-lag의 정보 경계

동일한 $z_t$를 추론해도 입력으로 허용한 관측 범위가 다르다.

$$
\begin{aligned}
\text{filter: }&
q_\phi(z_t\mid y_{1:t}),\\
\text{full smoother: }&
q_\phi(z_t\mid y_{1:T}),\\
\text{fixed-lag smoother: }&
q_\phi
\left(
z_t
\mid
y_{1:\min(T,t+L)}
\right).
\end{aligned}
$$

Full smoother는 $y_{t+1:T}$를 쓰므로 offline reconstruction에는 적합하지만, 시점 $t$에 결정을 내려야 하는 온라인 서비스에는 사용할 수 없다. Fixed-lag $L$은 $L$만큼의 지연을 허용해 일부 미래 증거를 얻는다. 평가 데이터에서도 각 방식이 실제 배포 시점에 가질 수 있는 정보만 제공해야 한다.

Padding mask와 결측 mask도 구분해야 한다. Padding은 존재하지 않는 시간 위치이고, 결측 관측은 시간은 존재하지만 센서값이 없는 것이다. 둘을 같은 $0$으로 넣으면 encoder가 “값이 0”과 “관측하지 못함”을 혼동한다.

### 3.13 Importance weight로 근사 posterior를 진단하기

$z^{(k)}\sim q_\phi(z\mid x)$를 뽑고

$$
w_k
=
\frac{p_\theta(x,z^{(k)})}
{q_\phi(z^{(k)}\mid x)}
$$

를 계산하면

$$
\mathbb E_q[w_k]
=
p_\theta(x).
$$

따라서

$$
\widehat p_\theta(x)
=
\frac{1}{K}
\sum_{k=1}^{K}w_k
$$

는 적절한 support 조건 아래 evidence의 불편 추정량이다. 그러나

$$
\log\widehat p_\theta(x)
$$

는 로그의 오목성 때문에 일반적으로 $\log p_\theta(x)$의 불편 추정량이 아니다. 정규화한 weight

$$
\bar w_k
=
\frac{w_k}{\sum_{j=1}^{K}w_j}
$$

의 집중 정도를

$$
\operatorname{ESS}_{\mathrm{IS}}
=
\frac{1}{\sum_{k=1}^{K}\bar w_k^2}
=
\frac{\left(\sum_k w_k\right)^2}
{\sum_k w_k^2}
$$

로 볼 수 있다. ESS가 $1$에 가까우면 한 표본이 거의 모든 질량을 차지해 $q_\phi$가 target의 중요한 영역을 잘 덮지 못했을 가능성이 있다. 이는 유한 표본 진단이지 참 posterior 오차의 완전한 증명은 아니다.

Importance-weighted bound는

$$
\mathcal L_K
=
\mathbb E
\left[
\log
\frac{1}{K}
\sum_{k=1}^{K}
\frac{p_\theta(x,z^{(k)})}
{q_\phi(z^{(k)}\mid x)}
\right]
$$

이다. 표준 조건에서 $\mathcal L_1$은 ELBO이고 $K$가 커질수록 evidence에 더 가까운 하한을 만들 수 있다. 그렇다고 큰 $K$가 언제나 inference network의 유한표본 gradient 품질이나 wall-clock 효율을 개선하는 것은 아니다. 목적함수의 tightness와 최적화 난이도를 함께 측정해야 한다.

### 3.14 연합 목적함수와 가중치

Client $k$의 데이터셋을 $D_k$, 크기를 $n_k$, 전체 크기를 $N=\sum_{k=1}^{K}n_k$라 하자. 전체 레코드의 평균 위험을 목표로 한다면

$$
F(w)
=
\sum_{k=1}^{K}
\frac{n_k}{N}
F_k(w),
\qquad
F_k(w)
=
\frac{1}{n_k}
\sum_{i\in D_k}
\ell(w;x_i).
$$

여기서 $w$는 모델 파라미터다. $n_k/N$는 “각 client를 똑같이 중요하게 본다”가 아니라 “각 레코드에 같은 가중치를 준다”는 선택이다. Client별 공정성을 원한다면

$$
F_{\mathrm{client}}(w)
=
\frac{1}{K}
\sum_{k=1}^{K}F_k(w)
$$

처럼 다른 목적을 택할 수 있다. 어느 가중치가 옳은지는 학습 목적·sampling 설계·privacy unit에 달려 있다.

### 3.15 FedAvg 한 round의 수식

Round $r$에서 선택된 client 집합을 $S_r$라 하자. 각 client는 같은 전역 시작점에서 출발한다.

$$
w_{r,0}^{(k)}=w_r.
$$

$E$번의 로컬 gradient step을 수행한다.

$$
w_{r,e+1}^{(k)}
=
w_{r,e}^{(k)}
-
\eta
\nabla F_k
\left(
w_{r,e}^{(k)}
\right),
\qquad
e=0,\ldots,E-1.
$$

Client update를

$$
\Delta_r^{(k)}
=
w_{r,E}^{(k)}-w_r
$$

라 하면 서버 갱신은

$$
w_{r+1}
=
w_r
+
\sum_{k\in S_r}
\alpha_{r,k}
\Delta_r^{(k)},
\qquad
\alpha_{r,k}\ge 0,
\quad
\sum_{k\in S_r}\alpha_{r,k}=1
$$

로 쓸 수 있다. 참여 client 사이의 데이터 크기 가중 평균이라면

$$
\alpha_{r,k}
=
\frac{n_k}
{\sum_{j\in S_r}n_j}.
$$

부분 참여인데 무조건 $n_k/N$만 사용하면 참여자 가중치 합이 $1$이 아닐 수 있다. 전체 population gradient의 불편 추정이 목적이라면 client sampling probability와 inverse-probability correction까지 명시해야 한다. FedAvg라는 이름만으로 estimator가 자동 결정되지는 않는다.

### 3.16 Local step과 client drift

$E=1$이고 모든 client가 $w_r$에서 gradient를 계산한다면

$$
\Delta_r^{(k)}
\approx
-\eta\nabla F_k(w_r).
$$

하지만 $E>1$이면 뒤의 gradient는 이동한 로컬 파라미터에서 계산된다.

$$
\Delta_r^{(k)}
=
-\eta
\sum_{e=0}^{E-1}
\nabla F_k
\left(
w_{r,e}^{(k)}
\right).
$$

$F_k$가 client마다 다르면 로컬 궤적도 달라진다. 큰 $E$는 통신 횟수를 줄이지만 non-IID 환경에서 전역 목적의 하강 방향과 멀어질 수 있다. 이를 client drift라고 한다. 작은 학습률, proximal penalty, control variate, 적절한 client sampling 등이 완화책이 될 수 있지만 만능은 아니다.

### 3.17 Pairwise mask가 합에서 사라지는 이유

직관을 위해 모든 client가 끝까지 참여하고 계산이 어떤 유한 ring에서 이루어진다고 하자. 두 client의 순서 있는 mask를 $m_{ij}=-m_{ji}$가 되도록 정한다. Client $i$는

$$
\widetilde\Delta^{(i)}
=
\Delta^{(i)}
+
\sum_{j\ne i}m_{ij}
$$

를 보낸다. 서버가 모두 합하면 각 pair의 mask가 한 번은 더해지고 한 번은 빠진다.

$$
\begin{aligned}
\sum_i\widetilde\Delta^{(i)}
&=
\sum_i\Delta^{(i)}
+
\sum_{i<j}
\left(
m_{ij}+m_{ji}
\right)\\
&=
\sum_i\Delta^{(i)}.
\end{aligned}
$$

서버는 합을 얻지만 개별 $\Delta^{(i)}$는 mask 때문에 바로 읽을 수 없다. 실제 보안은 이 대수 한 줄만으로 성립하지 않는다. Key agreement, authenticated channel, 정수 양자화·overflow, 악성 참여자, 서버 공모, replay 방지와 프로토콜 증명이 필요하다.

### 3.18 Client dropout과 복구 임계값

Client $j$가 masked update를 보내기 전에 이탈하면 다른 client가 더한 $m_{ij}$의 짝이 사라질 수 있다. 실제 secure aggregation protocol은 secret sharing 등을 사용해 정해진 조건에서 이탈자의 미상쇄 mask만 복구한다.

복구 threshold $t$는 보통 다음 두 목표 사이의 계약이다.

- 충분한 참여자가 남으면 합을 완성할 수 있어야 한다.
- 너무 적은 주체나 서버와의 공모만으로 생존 client의 비밀 mask를 복원해서는 안 된다.

구체적인 $t$의 의미와 허용하는 dropout·collusion 수는 프로토콜마다 다르다. “$t$명 이상이면 익명”처럼 일반화하면 안 된다. 또한 작은 cohort의 합 자체나 서로 거의 같은 두 cohort 합의 차이는 암호를 깨지 않아도 개인 업데이트에 가까운 정보를 줄 수 있다.

### 3.19 Client-level clipping과 Gaussian mechanism

각 client의 전체 update를 한 privacy unit으로 보호한다고 하자. 먼저

$$
\overline\Delta_r^{(k)}
=
\Delta_r^{(k)}
\min
\left(
1,
\frac{C}
{\|\Delta_r^{(k)}\|_2}
\right)
$$

로 $\ell_2$ 노름을 $C$ 이하로 제한한다. 정확히는 $\Delta=0$일 때 나눗셈 예외를 별도로 처리한다. 고정된 $m$개 참여 슬롯 중 불참자를 $0$ 기여로 나타내는 단순 예에서

$$
\widetilde\Delta_r
=
\frac{1}{m}
\left(
\sum_{k\in S_r}
\overline\Delta_r^{(k)}
+
\xi_r
\right),
\qquad
\xi_r
\sim
\mathcal N(0,\sigma^2C^2I)
$$

를 공개한다.

Add/remove adjacency에서 “합”의 민감도는 최대 $C$다. 공개된 고정 분모 $m$을 사용하고 불참자를 $0$으로 나타낸 평균의 민감도는 최대 $C/m$이다. 실현된 참여자 수로 정규화해 인접 입력 사이에서 분모까지 달라지면 별도의 민감도 분석이 필요하다. Replace-one adjacency에서는 빠진 client와 들어온 client의 차이가 생겨 최대 $2C$ 또는 고정 평균 기준 $2C/m$가 될 수 있다.

이 단순식은 실제 FL의 가변 cohort·가중치·sampling을 모두 덮지 않는다. 예를 들어 데이터 크기 가중치가 무제한이면 큰 client 하나의 영향이 커진다. Client-level DP를 주장하려면 가중치나 기여를 함께 cap하고, adjacency·denominator 공개·실패 round 처리까지 accountant가 분석한 메커니즘과 일치해야 한다.

### 3.20 Privacy budget은 round를 넘어 누적된다

Round $r$의 메커니즘이 $(\varepsilon_r,\delta_r)$-DP라고 해서 전체 학습도 같은 숫자를 갖는 것은 아니다. 가장 단순한 composition 상한은

$$
\varepsilon_{\mathrm{total}}
\le
\sum_{r=1}^{R}\varepsilon_r,
\qquad
\delta_{\mathrm{total}}
\le
\sum_{r=1}^{R}\delta_r
$$

이다. 실제로는 Rényi DP·privacy loss distribution 같은 더 정교한 accountant가 더 타이트한 값을 줄 수 있다. 무작위 subsampling은 조건이 맞으면 privacy amplification을 주지만 다음이 명시돼야 한다.

- client population과 한 round의 sampling 방식,
- 참여확률과 재참여 규칙,
- clipping 단위와 noise 위치,
- abort·retry·adaptive selection 처리,
- add/remove인지 replace-one인지,
- 한 사람의 여러 계정·기기를 어떻게 privacy unit으로 묶는지.

DP noise를 여러 client가 나누어 생성할 수도 있지만, dropout이나 일부 client–server 공모 뒤에도 목표 noise 분산이 남는다는 protocol-level 보장이 필요하다. “각자 조금씩 noise를 넣었다”만으로 중앙 DP 수치가 성립하지 않는다.

### 3.21 Secure aggregation·DP·robustness의 세 경계

세 기술은 서로 다른 실패를 다룬다.

| 질문 | 대표 수단 | 자동으로 해결하지 않는 것 |
| --- | --- | --- |
| 서버가 개별 update를 볼 수 있는가? | secure aggregation | 공개 합의 추론 위험, poisoning, 삭제 |
| 한 client가 공개 결과에 미치는 영향은 제한됐는가? | client-level DP | 전송 중 raw update 노출, 무결성, 과거 artifact 폐기 |
| 악성 update가 모델을 망치는가? | validation·robust aggregation·anomaly control | 암호학적 기밀성, formal DP, 삭제 증거 |

보안 집계가 개별 update를 숨기면 서버가 per-client anomaly를 검사하기 어려워질 수도 있다. 따라서 clipping·range proof·aggregate-level check·trusted execution·다단계 승인 같은 설계가 threat model에 맞게 필요하다. 기밀성과 무결성 사이의 긴장을 숨겨서는 안 된다.

### 3.22 연합 변분추론으로 두 트랙 연결하기

전역 잠재변수 $\Theta$, client별 잠재변수 $Z_k$, 로컬 데이터 $D_k$를 둔 계층모형을 생각하자.

$$
p(\theta,z_{1:K},D_{1:K})
=
p(\theta)
\prod_{k=1}^{K}
p(z_k)
p(D_k\mid\theta,z_k).
$$

근사분포를

$$
q_\lambda(\theta)
\prod_{k=1}^{K}
q_\phi(z_k\mid D_k)
$$

로 두면 ELBO는

$$
\begin{aligned}
\mathcal L
&=
-
D_{\mathrm{KL}}
\left(
q_\lambda(\theta)\|p(\theta)
\right)\\
&\quad+
\sum_{k=1}^{K}
\Bigg\{
\mathbb E_q
\left[
\log p(D_k\mid\theta,z_k)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z_k\mid D_k)\|p(z_k)
\right)
\Bigg\}.
\end{aligned}
$$

Client별 합으로 나뉘는 항은 로컬에서 추정할 수 있고, 서버는 gradient나 sufficient statistic의 합을 사용할 수 있다. 이것이 두 트랙의 수학적 접점이다. 그러나 “ELBO를 분산 계산했다”는 사실은 privacy 보장이 아니다. 로컬 gradient가 민감하면 secure aggregation과 DP가 여전히 필요하다.

### 3.23 연합 삭제의 기준선

학습 알고리즘 $A$가 무작위 seed·client sampling·local optimizer까지 포함한다고 하자. Client $u$의 데이터를 제거한 기준선은

$$
W^{-u}
\sim
A(D\setminus D_u)
$$

이다. Unlearning 알고리즘 $U$의 출력은

$$
\widehat W^{-u}
\sim
U(W,D_u,\mathcal M)
$$

처럼 쓸 수 있다. 여기서 $\mathcal M$은 checkpoint·manifest·저장된 통계 같은 보조 정보다.

Exact unlearning은 정한 관측 가능 범위에서 $\widehat W^{-u}$의 분포가 삭제 재학습 기준선 $W^{-u}$의 분포와 같다는 강한 목표다. Approximate 또는 certified removal은 두 분포가 정한 발산·privacy형 부등식·certificate 아래 가깝다고 주장한다. Forget-set 정확도 하락이나 membership inference attack 하나의 실패만으로 분포 동등성을 증명할 수는 없다.

### 3.24 왜 과거 update 하나를 빼는 것으로 부족한가?

Round $r$의 전역 갱신을

$$
w_{r+1}
=
G_r
\left(
w_r,
\{\Delta_r^{(k)}(w_r,D_k)\}_{k\in S_r}
\right)
$$

라 하자. Client $u$를 제거하면 $w_{r+1}^{-u}$가 달라진다. 다음 round의 다른 client $k$도 다른 시작점에서 학습한다.

$$
\Delta_{r+1}^{(k)}
\left(
w_{r+1}^{-u},D_k
\right)
\neq
\Delta_{r+1}^{(k)}
\left(
w_{r+1},D_k
\right)
\quad\text{일반적으로}.
$$

따라서

$$
w_R
-
\alpha_{r,u}\Delta_r^{(u)}
$$

는 보통 $u$ 없이 다시 학습한 $w_R^{-u}$가 아니다. 정확한 방향은 첫 영향 이전의 안전 checkpoint로 돌아가 이후 참여·난수·데이터 버전 정책을 정해 재실행하는 것이다. 계산비를 줄이려면 처음부터 cohort·shard·checkpoint 경계를 설계하거나, 명시된 오차 보장을 가진 근사를 사용해야 한다.

보안 집계는 이 문제를 더 선명하게 한다. 서버가 $\Delta_r^{(u)}$를 본 적이 없다면 사후 subtraction 자체가 불가능할 수 있다. 이는 secure aggregation의 실패가 아니라 의도된 기밀성이다. 삭제 가능성은 개별 update 보관이 아니라 재실행 가능한 checkpoint·cohort manifest·client 측 삭제 증명 같은 별도 구조로 준비해야 한다.

### 3.25 분산 lineage DAG와 전이적 무효화

산출물 계보를 DAG $G=(V,E)$로 두자. Edge $a\to b$는 “$b$를 만들 때 $a$를 사용했다”는 뜻이다.

$$
\text{client data snapshot}
\rightarrow
\text{local training job}
\rightarrow
\text{round cohort commitment}
\rightarrow
\text{secure aggregate}
\rightarrow
\text{optimizer state}
\rightarrow
\text{global checkpoint}.
$$

그 뒤에도

$$
\text{global checkpoint}
\rightarrow
\{\text{adapter, quantized copy, student, index, cache, endpoint}\}
$$

가 이어진다. 삭제 source $v$의 모든 후손은

$$
\operatorname{Desc}(v)
=
\{u\in V:v\leadsto u\}
$$

이다. 정확한 영향 범위는 단순 파일명 검색이 아니라 graph reachability 문제다.

Manifest에는 최소한 다음이 필요하다.

- 부모 artifact ID와 content hash,
- client·cohort의 가명화된 commitment와 sampling 설정,
- 데이터 snapshot version과 삭제 요청의 identity resolution,
- code·container·dependency·randomness policy,
- clipping·noise·accountant state,
- local optimizer와 server optimizer state,
- 생성한 checkpoint·adapter·index·배포 alias.

참여 manifest 자체도 민감할 수 있다. 누가 어느 round에 참여했는지가 건강 상태나 조직 활동을 드러낼 수 있기 때문이다. 따라서 lineage는 감사 가능해야 하지만 최소 공개·접근제어·가명화·보존기간을 함께 가져야 한다.

### 3.26 오늘의 수학이 AI 부품에서 쓰이는 위치

| 수학 원리 | AI·시스템 부품 | 놓치면 생기는 오류 |
| --- | --- | --- |
| KL 비음수성과 ELBO 항등식 | Encoder–decoder 공동 학습 | ELBO를 정확한 evidence로 오해한다. |
| Gaussian KL | VAE latent regularization | 분산·log variance를 혼동한다. |
| 재매개변수화 | Stochastic latent를 지나는 backpropagation | 표본 노드를 미분 불가능하다고 포기한다. |
| 상각 오차 분해 | Inference network 진단 | 빠른 encoder를 사례별 최적 posterior로 착각한다. |
| 시계열 factorization | Amortized smoother | 미래 관측을 온라인 입력으로 누출한다. |
| Importance weight와 ESS | Posterior coverage 진단 | 높은 ELBO 하나만으로 불확실성을 확신한다. |
| 가중합과 sampling estimator | FedAvg server update | 참여 subset의 가중치 의미를 잃는다. |
| Pairwise mask 상쇄 | Secure aggregation | 합 보존과 개인 기밀성을 같은 증명으로 취급한다. |
| Norm sensitivity | Client-level DP clipping | record-level·client-level 보장을 혼용한다. |
| Composition | Multi-round privacy accountant | round별 $\varepsilon$ 하나만 보고 전체 예산을 누락한다. |
| 함수 합성의 경로 의존성 | Federated unlearning | 과거 delta 하나를 빼면 재학습과 같다고 믿는다. |
| DAG reachability | Transitive invalidation | Base checkpoint만 바꾸고 adapter·cache를 남긴다. |

---

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 VAE 한 번의 순전파를 부품별로 보기

입력 batch를

$$
X\in\mathbb R^{B\times d_x}
$$

라 하자. Encoder는 각 사례에 대해 잠재차원 $d_z$의 평균과 log variance를 출력한다.

$$
\mu_\phi(X),\ell_\phi(X)
\in
\mathbb R^{B\times d_z}.
$$

표준편차와 표본은

$$
\sigma
=
\exp\left(\frac{1}{2}\ell\right),
\qquad
\epsilon
\sim
\mathcal N(0,I),
\qquad
Z
=
\mu+\sigma\odot\epsilon
$$

로 만든다. Decoder는 $Z$에서 likelihood의 파라미터를 출력한다.

$$
Z
\xrightarrow{f_\theta}
\text{parameters of }p_\theta(X\mid Z).
$$

학습에서는 음의 ELBO를 최소화한다.

$$
\mathcal J(\theta,\phi;X)
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\mathcal L(\theta,\phi;x_i).
$$

Gaussian likelihood라면 reconstruction 항이 제곱오차처럼 보일 수 있고, Bernoulli likelihood라면 binary cross-entropy 형태가 나올 수 있다. 데이터 범위와 likelihood 가정이 맞지 않는데 손실 이름만 복사하면 확률모형의 의미가 깨진다.

흔히

$$
\mathcal L_\beta
=
\mathbb E_q[\log p_\theta(x\mid z)]
-
\beta
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
$$

처럼 $\beta$를 사용한다. $\beta=1$은 표준 ELBO다. $\beta\neq1$은 표현·재현 trade-off를 바꾼 목적함수이며 일반적으로 원래 모형의 같은 log evidence에 대한 표준 ELBO라고 부르면 안 된다.

### 4.2 Posterior collapse는 왜 생길까?

Decoder가 $z$ 없이도 $x$를 잘 예측할 만큼 강하면

$$
q_\phi(z\mid x)
\approx
p(z)
$$

가 되어 KL 항이 거의 $0$이 될 수 있다. 이때 latent가 입력 정보를 거의 담지 않는 현상을 posterior collapse라 한다. 낮은 KL 하나를 “완벽한 posterior”라고 해석하면 안 된다. Decoder가 latent를 실제 사용하는지 다음을 함께 본다.

- Reconstruction이 $z$를 바꿀 때 의미 있게 변하는가?
- Dimension별 KL과 active unit 수는 어떤가?
- Prior sample과 posterior sample의 품질 차이는 어떤가?
- Mutual-information proxy나 downstream latent utility가 남는가?

KL warm-up·free bits·decoder capacity 조절은 완화법이지만, 모형과 데이터에 따라 trade-off가 있다.

### 4.3 사례별 최적화와 상각 encoder의 차이

사례별 VI는 새 $x$가 올 때마다 변분 파라미터 $\lambda_x$를 초기화하고 여러 gradient step을 수행한다.

$$
\lambda_x^{(s+1)}
=
\lambda_x^{(s)}
+
\rho
\nabla_{\lambda}
\mathcal L(\lambda_x^{(s)};x).
$$

상각 추론은

$$
\lambda_x
=
g_\phi(x)
$$

를 한 번 계산한다. 빠르지만 학습분포 밖의 사례에서는 $g_\phi$가 나쁜 초기값을 줄 수 있다. 절충안은 test-time refinement다.

$$
\lambda_x^{(0)}
=
g_\phi(x),
\qquad
\lambda_x^{(s+1)}
=
\lambda_x^{(s)}
+
\rho\nabla_\lambda\mathcal L.
$$

Encoder의 속도를 얻고 소수 step으로 amortization gap을 줄인다. 다만 test-time optimization 시간과 stopping rule을 평가 비용에 포함해야 한다.

### 4.4 상각 스무더의 텐서 흐름

관측 batch의 shape를

$$
Y
\in
\mathbb R^{B\times T\times d_y},
\qquad
M
\in
\{0,1\}^{B\times T}
$$

라 하자. $M$은 실제 유효 시점을 표시한다. 한 가지 구현 흐름은 다음과 같다.

$$
Y,M
\xrightarrow{\text{bidirectional encoder}}
C
\in
\mathbb R^{B\times T\times d_c}
$$

$$
C
\xrightarrow{\text{posterior heads}}
\mu_{0:T},\ell_{0:T}
\in
\mathbb R^{B\times(T+1)\times d_z}
$$

$$
\epsilon_{0:T}
\longrightarrow
Z_{0:T}
\longrightarrow
\text{emission·transition ELBO}.
$$

실제 structured posterior는 $z_{t-1}$이나 $z_{t+1}$에도 조건을 걸 수 있다. 중요한 것은 다음 계약이다.

- Full smoothing이면 encoder가 미래 $y_{t+1:T}$를 보았다고 명시한다.
- Fixed-lag이면 최대 $L$시점 미래만 볼 수 있도록 attention mask를 건다.
- Missing observation mask와 padding mask를 분리한다.
- Train·validation·deployment의 관측 가능 범위를 같게 한다.
- 새 센서·길이·동역학에서 importance-weight·calibration 진단을 다시 한다.

### 4.5 변분추론과 Particle MCMC 중 무엇을 써야 할까?

| 기준 | Amortized VI | Particle MCMC |
| --- | --- | --- |
| 목표 | 선택한 family 안의 빠른 근사 | 조건 아래 정확한 posterior를 주변분포로 겨냥 |
| 초기 비용 | Encoder 학습이 큼 | 별도 encoder 학습은 불필요할 수 있음 |
| 새 사례 비용 | 순전파로 빠름 | Chain·particle 계산을 반복 |
| 주된 오차 | Approximation·amortization·optimization gap | 유한 chain의 mixing·Monte Carlo error |
| 다봉 posterior | Family가 빈약하면 mode 누락 가능 | 충분한 mixing이 어려울 수 있음 |
| 진단 | ELBO gap proxy·importance ESS·calibration | $\widehat R$·ESS·trace·likelihood variance |
| 적합한 상황 | 반복되는 대규모 online/offline inference | 소수 사례의 정밀한 연구·검증 |

두 방법은 경쟁자만이 아니다. Particle·MCMC 결과로 inference network를 검증하거나 가르칠 수 있고, encoder를 proposal로 사용해 particle 효율을 높일 수도 있다.

### 4.6 연합학습 한 round의 안전한 순서

한 round를 다음 순서로 설계할 수 있다.

$$
\boxed{
\text{배포}
\rightarrow
\text{로컬 학습}
\rightarrow
\text{client clipping}
\rightarrow
\text{secure masking}
\rightarrow
\text{합 복원}
\rightarrow
\text{noise·accounting}
\rightarrow
\text{검증 gate}
}
$$

1. **배포:** 서버는 부모 checkpoint hash, round ID, optimizer 설정, clipping bound, cohort 정책을 서명된 manifest와 함께 보낸다.
2. **로컬 학습:** Client는 승인된 데이터 snapshot과 code version으로 $E$ step 학습한다.
3. **Client clipping:** 전체 update를 $C$에 맞춰 제한한다. 보안 집계 뒤에는 개별 update를 clip할 수 없으므로 이 단계의 정직성·검증 방법이 중요하다.
4. **Secure masking:** Client는 protocol이 정한 mask·commitment를 만들고 masked update를 보낸다.
5. **합 복원:** Dropout·threshold 조건을 확인한 뒤 허용된 aggregate만 복원한다. 조건이 깨지면 round를 abort한다.
6. **Noise·accounting:** 신뢰하는 서버가 중앙 noise를 더하거나, 검증된 distributed noise protocol을 사용한다. 실제 참여와 실패 정보를 accountant에 반영한다.
7. **검증 gate:** Utility·그룹 성능·안전·poisoning aggregate check를 통과한 경우만 새 generation으로 승격한다.

순서는 threat model에 따라 달라질 수 있다. 예를 들어 서버를 DP noise에 대해 신뢰하지 않는다면 noise를 client 쪽에서 분산 생성해야 한다. 어떤 순서를 택했는지와 그 순서가 보장하는 공격자를 문서화해야 한다.

### 4.7 Secure aggregation이 만드는 삭제의 역설

삭제를 쉽게 하려고 모든 개별 update를 서버에 평문으로 영구 보관하면 사후 subtraction 연구는 쉬워질 수 있다. 그러나 이는 secure aggregation이 줄이려던 노출을 다시 만든다. 반대로 개별 update를 전혀 복원하지 않으면 특정 client의 delta만 정확히 빼기 어렵다.

이 긴장은 “기밀성을 포기하거나 삭제를 포기하라”는 뜻이 아니다. 삭제 단위를 사전에 구조화해야 한다는 뜻이다.

- Round·cohort 전의 checkpoint를 충분히 보존한다.
- Client가 참여한 cohort의 가명 commitment를 남긴다.
- Cohort·shard가 서로 미치는 영향을 제한하는 학습 구조를 고려한다.
- Local raw data와 local optimizer artifact의 삭제 책임을 client 측까지 정의한다.
- 서버는 aggregate와 후손 artifact의 계보를 보존한다.
- 정확 재실행이 불가능하면 approximate guarantee와 실패 범위를 명시한다.

보존하는 metadata가 새 개인정보가 되지 않도록 접근권한·암호화·retention policy도 필요하다.

### 4.8 연합 삭제의 단계별 수명주기

삭제 요청을 받았다고 가정하자.

1. **요청 인증과 범위 결정:** 요청자가 어느 사람·기관·기기·계정과 연결되는지 확인한다. 한 사람이 여러 client ID를 가질 수 있고 한 client에 여러 사람이 섞일 수 있다.
2. **Source 식별:** 원본 레코드뿐 아니라 중복·증강·합성 파생물과 local cache를 찾는다.
3. **즉시 격리:** 영향받을 가능성이 있는 model generation과 artifact를 quarantine하고, 새 fine-tuning·distillation의 부모로 사용하지 못하게 한다.
4. **계보 폐쇄 계산:** 첫 참여 round와 $\operatorname{Desc}(v)$를 찾아 checkpoint·optimizer state·adapter·student·index·cache·endpoint를 목록화한다.
5. **보장 선택:** Full retraining, 영향 이전 checkpoint replay, 구조화된 shard retraining, 또는 명시된 certified approximation 중 요구 수준에 맞는 절차를 고른다.
6. **재계산:** 삭제 대상을 제외하고 필요한 round·artifact를 새 immutable generation으로 만든다.
7. **검증:** Retain utility, forget criterion, privacy attack, distributional distance 또는 certificate, 재학습 reproducibility를 평가한다.
8. **세대 교체:** 새 endpoint alias를 원자적으로 전환하고 옛 replica·cache·download channel을 tombstone 처리한다.
9. **증거 보존:** 원본 민감정보를 과도하게 남기지 않으면서 실행 manifest, hash, 승인자, accountant state, 검증 결과를 감사 기록으로 남긴다.

이미 외부 사용자가 내려받은 checkpoint나 그 checkpoint가 만든 출력까지 기술적으로 회수할 수 있는 것은 아니다. 삭제 완료 문구에는 통제 가능한 시스템 경계가 포함돼야 한다.

### 4.9 무엇을 검증해야 “삭제됐다”고 말할 수 있을까?

한 숫자로는 부족하다.

| 검증 축 | 질문 | 예시 |
| --- | --- | --- |
| Forget efficacy | 삭제 대상의 특별한 영향이 줄었는가? | 대상별 loss·exposure·behavior test |
| Retain utility | 남겨야 할 데이터의 성능이 유지됐는가? | Holdout·그룹별 성능·calibration |
| Reference closeness | 삭제 재학습 기준선과 가까운가? | Parameter·prediction distribution·certificate |
| Privacy | 남은 모델에서 참여 여부 공격 위험은 어떤가? | Membership·inversion audit, DP accounting |
| Recurrence | 추가 fine-tuning 뒤 삭제 행동이 되살아나는가? | Relearning·adversarial prompt test |
| Lineage closure | 영향받은 후손이 모두 교체됐는가? | DAG reachability·serving inventory |
| Operational revocation | 옛 generation이 실제 traffic을 받지 않는가? | Registry·replica·cache·alias audit |

Behavioral forgetting과 data deletion은 구분해야 한다. 특정 단어·스타일을 생성하지 않게 만든 편집은 유용할 수 있지만, 학습 레코드를 제외한 재학습 결과분포와 같다는 증거는 아니다.

### 4.10 수학과 구현의 1:1 연결

| 오늘의 수학 | 실제 구현 질문 |
| --- | --- |
| $\log p=\mathcal L+D_{\mathrm{KL}}$ | ELBO 상승이 posterior gap 감소인지 생성모형 변화인지 분리했는가? |
| Approximation·amortization gap | Family 확장과 test-time refinement 중 무엇이 병목인가? |
| $q(z_{0:T}\mid y_{1:T})$ | 미래 관측을 사용할 수 있는 offline 작업인가? |
| Weighted client sum | 목표 population과 참여 sampling에 맞는 $\alpha_{r,k}$인가? |
| Mask 상쇄 | Dropout·collusion·overflow에서도 protocol 증명이 유지되는가? |
| Sensitivity $C$ 또는 $2C$ | Add/remove와 replace-one 중 어느 adjacency인가? |
| Privacy composition | 실패·retry·adaptive participation도 accountant에 들어갔는가? |
| Path dependence | 삭제 영향 이전 checkpoint에서 후속 round를 다시 계산했는가? |
| $\operatorname{Desc}(v)$ | Quantized copy·adapter·student·index·cache가 모두 포함됐는가? |

### 4.11 초보자가 흔히 하는 오해

#### 오해 1: ELBO가 높으면 참 posterior를 정확히 찾았다

ELBO와 log evidence의 차이는 KL이지만 log evidence를 모르면 gap을 직접 알기 어렵다. 또한 $\theta$도 함께 학습되므로 높은 training ELBO가 posterior calibration·test likelihood·sample quality를 자동 보장하지 않는다.

#### 오해 2: KL은 두 분포 사이의 거리다

KL은 비대칭이고 triangle inequality도 일반적으로 만족하지 않는다. 따라서 metric처럼 “두 배 멀다”라고 기하학적으로 해석하면 안 된다.

#### 오해 3: Reverse KL은 언제나 mode 하나만 고른다

제한된 family에서 그런 경향이 있지만 절대 법칙은 아니다. Posterior shape, family, optimization에 따라 결과가 다르다.

#### 오해 4: 재매개변수화는 모든 잠재변수에 그대로 쓸 수 있다

Gaussian 연속변수의 위치–척도 표기는 대표 예다. 이산변수에는 같은 미분 경로가 없으므로 다른 estimator나 relaxation이 필요하다.

#### 오해 5: 상각 추론은 정확한 추론을 공짜로 빠르게 만든다

공유 encoder는 inference cost를 줄이는 대신 amortization gap과 분포 이동 위험을 만든다. 새 domain에서는 refinement·retraining·진단이 필요하다.

#### 오해 6: Bidirectional encoder는 언제나 causal encoder보다 좋다

Offline smoothing에는 미래 증거가 유용하지만 실시간 의사결정에는 미래가 존재하지 않는다. 잘못 쓰면 leakage다.

#### 오해 7: FedAvg는 client 모델의 단순 평균이다

목표가 record-average인지 client-average인지, client sampling이 균등한지, 부분 참여를 어떻게 보정하는지에 따라 가중치가 달라진다.

#### 오해 8: 원본 데이터를 보내지 않으면 개인정보가 보호된다

Update도 정보를 담는다. Secure aggregation·DP·최소 cohort·접근제어·감사가 별도로 필요하다.

#### 오해 9: Secure aggregation은 DP다

Secure aggregation은 누가 개별 update를 볼 수 있는지 제한한다. DP는 공개 결과분포의 민감도를 제한한다. 암호학적 기밀성과 통계적 안정성은 다른 보장이다.

#### 오해 10: Secure aggregation은 poisoning도 막는다

오히려 개별 update가 숨겨지면 anomaly inspection이 어려워질 수 있다. Clipping과 무결성·robustness 설계를 별도로 해야 한다.

#### 오해 11: Client-level DP와 record-level DP는 같은 보호다

Record 하나를 빼는 인접성과 client의 모든 레코드를 빼는 인접성은 민감도가 다르다. Accountant의 privacy unit을 먼저 확인해야 한다.

#### 오해 12: 삭제하면 과거 privacy budget이 환불된다

이미 공개된 DP 출력이 소비한 privacy loss는 사후 삭제로 되돌아가지 않는다. 새 모델 release까지 포함해 전체 수명주기의 composition을 관리해야 한다.

#### 오해 13: Client update만 빼면 federated unlearning이다

후속 client들이 달라진 전역 모델에서 학습했기 때문에 경로 의존성이 남는다. Full retraining 또는 영향 이전 checkpoint에서의 재실행이 기준선이다.

#### 오해 14: Tombstone이면 모든 복사본이 사라진다

통제 중인 registry·replica·cache에는 강제할 수 있지만 외부로 반출된 checkpoint는 자동 회수되지 않는다. 시스템 경계를 정직하게 써야 한다.

#### 오해 15: 참여 manifest는 민감하지 않다

특정 병원·기기가 어느 시점에 참여했는지도 민감할 수 있다. 감사 가능성과 metadata 최소화를 함께 설계한다.

### 4.12 오늘의 실전 체크리스트

- [ ] Posterior family, support, latent dimension과 likelihood를 명시했는가?
- [ ] ELBO의 reconstruction·KL을 batch·dimension별로 따로 모니터링하는가?
- [ ] Approximation gap과 amortization gap을 구분할 진단 또는 refinement가 있는가?
- [ ] Smoother가 볼 수 있는 미래 범위와 배포 latency가 일치하는가?
- [ ] Missing mask와 padding mask를 구분했는가?
- [ ] FL의 목표 population, client sampling, aggregation weight를 문서화했는가?
- [ ] Client-level clipping이 secure masking 전에 적용되고 bound가 검증되는가?
- [ ] Secure aggregation의 dropout·collusion·최소 cohort 조건을 명시했는가?
- [ ] DP adjacency, sampling, noise, retry를 실제 accountant에 반영했는가?
- [ ] Poisoning 방어가 기밀성 설계와 충돌하지 않는지 검토했는가?
- [ ] 삭제 요청의 entity-to-client mapping과 중복·파생 데이터를 찾을 수 있는가?
- [ ] 첫 영향 이전 checkpoint와 후속 lineage를 재구성할 수 있는가?
- [ ] 옛 adapter·index·student·cache·replica까지 전이적으로 무효화하는가?
- [ ] Forget·retain·privacy·recurrence·serving 검증을 모두 통과했는가?
- [ ] 통제할 수 없는 외부 복사본과 과거 privacy loss의 한계를 공개했는가?

---

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> **선정 기준:** 2026-10-27 KST 기준 최근 공개물 중, 원 논문과 공개 저장소를 함께 확인할 수 있고 오늘의 “상태·근사·삭제 증거”와 직접 연결되는 두 항목을 골랐다. 아래 수치는 모두 2026-10-23 제출된 v1 preprint의 저자 보고치이며 독립 재현 결과가 아니다.

### 5.1 Async-RBench: 비동기 subagent 결과를 “받았는가”가 아니라 “올바르게 반영했는가”를 평가

LLM agent가 여러 subagent를 동시에 실행하면 main agent는 기다리지 않고 일을 계속할 수 있다. 문제는 늦게 도착한 결과가 현재 상태와 충돌할 수 있다는 점이다. 이미 수정한 파일을 오래된 보고서가 되돌릴 수도 있고, 새 변경이 앞선 테스트를 무효화할 수도 있으며, 같은 외부 행동을 중복 실행할 수도 있다.

Async-RBench는 이 능력을 분리해 측정한다. 공개된 구성은 다음과 같다.

| 항목 | 논문·저장소의 공개 설정 |
| --- | --- |
| 실행 가능한 task | 200개 |
| 원천 family | GAIA2·MultiAgentBench·OSWorld·SWE-bench·Terminal-Bench에서 각 40개 |
| 비동기 scenario | 8종, 각 25개 |
| 평가 agent | 9개 |
| 반복 | task·delivery mode별 3회 |
| 주 실험 episode | 10,800개 |
| 비교 | 모든 결과를 모아 주는 Batched 대 도착하는 즉시 주는 Async |

각 target event에는 evaluator만 보는 Event Response Contract(ERC)가 있다.

$$
\Gamma_i
=
\left(
C_i,R_i,F_i,V_i,O_i
\right).
$$

- $C_i$: 새 결과 때문에 반드시 바꿔야 할 것(Change)
- $R_i$: 이미 맞으므로 보존해야 할 것(Preserve)
- $F_i$: 해서는 안 되는 부작용(Forbid)
- $V_i$: 다시 확인해야 할 검증(Verify)
- $O_i$: 종료 시 만족해야 할 결과(Outcome)

Process score $P_i$는 적용 가능한 $C,R,F,V$의 평균이고 outcome score를 $Q_i$라 하면

$$
\operatorname{DRS}
=
\frac{100}{n}
\sum_{i=1}^{n}
\frac{P_i+Q_i}{2}
$$

로 Dynamic Replanning Score를 만든다. 최종 성공만 보는 대신 “필요한 수정·보존·금지·재검증”과 결과를 반씩 본다는 점이 핵심이다.

저자 보고 결과에서 9개 agent의 DRS는 $19.2$부터 $45.1$ 사이였다. 평균 task success point estimate는 Batched의 $29.7\%$에서 Async의 $26.0\%$로 낮아졌다. 세 agent에 각 subagent 결과가 제시될 때마다 완료된 작업과 기존 artifact는 유지하되 남은 계획 전체를 다시 세우라고 강제해도 DRS 증가는 $2.6$–$3.5$점이었고 model call은 $7.5\%$–$10.2\%$ 늘었다. 즉 “항상 남은 계획을 다시 세워라”는 규칙만으로는 어떤 증거가 무엇을 무효화했는지 추적하는 문제를 해결하지 못했다.

**공개성과 한계:** 저장소에는 200개 public task bundle, 실행 framework, test, configuration, paper artifact가 있고 code·documentation은 Apache-2.0이다. Task material은 원천 benchmark의 별도 조건을 유지한다. 반면 solution, hidden test, event schedule, verifier logic, credential, experiment trace는 public package에서 제외되고 formal scoring에는 별도 배포 judge bundle이 필요하다. 논문도 한 task에 target event 하나만 격리해 겹치는 interruption을 제외하며, controller의 재현 가능한 도착 방식이 실제 wall-clock arrival을 완전히 모사하지 않고, main과 subagent에 같은 model을 썼으며, 사람이 만든 ERC·verifier가 유효한 대응을 빠뜨릴 수 있다고 밝힌다.

**엔지니어 인사이트 (Impact):** Agent 병렬화는 latency 최적화가 아니라 **상태 일관성과 증거 계보** 문제다. Subagent 보고서에는 어떤 artifact version을 읽었는지, 어느 가정에 의존했는지, 어떤 검증을 통과했는지가 붙어야 한다. 새 결과가 오면 lineage graph에서 영향받은 후손만 무효화하고, 영향받지 않은 작업은 보존하며, 외부 side effect는 idempotency key로 중복을 막아야 한다. 오늘 배운 $\operatorname{Desc}(v)$와 transitive invalidation이 모델 삭제뿐 아니라 agent orchestration에도 그대로 등장한다.

- 원 논문: [Async-RBench: Benchmarking Asynchronous Task Execution in LLM Agents](https://arxiv.org/abs/2610.32429)
- 공식 공개 저장소: [pope8zdt/Async-RBench](https://github.com/pope8zdt/Async-RBench)

### 5.2 ConceptACP: Sparse Autoencoder로 diffusion concept의 위치를 찾고 한 block만 표적 수정

Concept Acupuncturist(ConceptACP)는 diffusion image model에서 원치 않는 concept를 억제할 때 전체 U-Net이나 전체 denoising trajectory를 매번 최적화하는 대신, pretrained Sparse Autoencoder(SAE)로 concept 관련 feature와 가장 관련 높은 cross-attention block을 찾는다.

핵심 흐름은 다음과 같다.

$$
\text{target·non-target prompt}
\rightarrow
\text{SAE feature relevance}
\rightarrow
\text{target block 선택}
\rightarrow
\text{feature suppression}
\rightarrow
\text{수정된 block-output target}
\rightarrow
\text{해당 block만 fine-tuning}.
$$

SAE를 최종 생성 때 붙이는 외부 module로 쓰는 것이 아니라, feature를 억제한 block output을 supervision target으로 만든 뒤 그 효과를 model parameter에 내재화한다. 논문 설명상 선택된 block의 여러 timestep activation을 한 번의 forward pass에서 수집하므로 fine-tuning마다 전체 diffusion trajectory를 반복할 필요가 없고, 배포 시 추가 SAE module도 필요 없다.

주 실험은 SDXL-Turbo와 EvalIGMU의 네 target을 사용한다.

| Target | 평가 prompt 수 | prompt당 image 수 |
| --- | ---: | ---: |
| Nudity | 250 | 5 |
| Van Gogh style | 250 | 5 |
| Church | 300 | 5 |
| Parachute | 300 | 5 |

저자는 표준 one-step generation에서 best-performing baseline 대비 forgetting success rate가 네 target 평균 $36.34$ percentage point 개선됐다고 보고한다. Full diffusion 기반 prompt-to-latent 설계와 비교해 block-level latent-I/O fine-tuning 시간이 $53.6\%$로 줄어, 논문은 이를 계산비 $46.4\%$ 감소로 요약한다. 이 표의 iteration·epoch 시간은 data loading·I/O를 제외한 측정이므로 end-to-end wall-clock 절감률과 같다고 단정하면 안 된다. 논문은 SDXL의 30-step과 SD v1.4의 50-step 설정에도 별도 실험을 제시한다.

그러나 이 결과를 “검증된 데이터 삭제”로 읽으면 안 된다.

- 연구 대상은 특정 concept의 **생성 행동 억제**이지, 특정 학습 레코드를 뺀 재학습 결과분포와의 동등성 certificate가 아니다.
- 논문 자체가 classifier 기반 output metric만으로 진짜 forgetting을 판단하기에는 불완전하다고 지적한다.
- 네 concept와 선택한 Stable Diffusion 계열에서의 결과이므로 임의의 정체성·스타일·안전 concept로 자동 일반화되지 않는다.
- 저자 표에서 SDXL-Turbo 4-step nudity에 대한 I2P·MMA·UnlearnDiffAtk 세 조건의 평균 attack success rate는 ConceptPrune $2.80\%$, ConceptACP $8.63\%$로, ConceptACP가 모든 공격 조건에서 일률적으로 최고인 것은 아니다.
- 더 강한 adaptive attack, concept별 SAE 학습, layer·concept에 적응하는 feature selection은 논문이 future work로 남겼다.

**오픈소스 상태:** 공개 GitHub 저장소는 MIT LICENSE 파일을 포함하고 SDXL·SD 1.5용 source, configuration, checkpoint·data 디렉터리가 있는 source tree를 제공한다. README는 Python 3.13과 외부 pretrained SAE를 요구한다. 다만 2026-10-27 현재 clone 예시에 placeholder 계정명이 남아 있고 README의 License 절에도 아직 “[Your License Here]”가 남아 있어 LICENSE 파일과 문서가 완전히 정리되지 않았다. Release package·tag도 성숙하지 않은 매우 초기 연구 코드다. 논문 수치의 재현에는 benchmark data·classifier·외부 model·GPU 환경과 정확한 revision을 고정해야 한다.

**엔지니어 인사이트 (Impact):** ConceptACP의 중요한 방향은 interpretability를 사후 설명에 그치지 않고 **어디를 얼마나 수정할지 정하는 localizer**로 썼다는 점이다. 이는 거대한 model 전체를 고치는 비용을 줄일 수 있다. 동시에 오늘의 연합 삭제 관점에서는 경고도 준다. “행동이 보이지 않는다”는 것과 “데이터 기여가 제거됐다”는 것은 다르다. 실제 삭제 주장을 하려면 target behavior metric, retain utility, adversarial recovery, 내부 feature 변화, retraining reference, artifact lineage를 서로 대체하지 말고 함께 제시해야 한다.

- 원 논문: [Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders](https://arxiv.org/abs/2610.32059)
- 공식 공개 저장소: [Eran-Xia/ConceptACP](https://github.com/Eran-Xia/ConceptACP)

### 5.3 두 트렌드를 오늘의 수학으로 읽기

| 비교 축 | Async-RBench | ConceptACP | 오늘의 교훈 |
| --- | --- | --- | --- |
| 다루는 상태 | Agent 작업·검증·subagent 결과 | Diffusion block·SAE feature·생성 행동 | 최종 출력만으로 내부 상태를 판단하지 않는다. |
| 요약 | DRS와 task success | FSR·FID·feature metric·ASR | 한 지표가 전체 보장을 대신하지 않는다. |
| 근사·선택 | Event contract가 영향 범위를 지정 | Feature·block relevance가 수정 범위를 선택 | 선택한 범위 밖의 누락을 별도로 감사한다. |
| 계보 | 새 결과가 앞선 work·test를 무효화 | Edited block에서 새 checkpoint가 파생 | Parent version과 후손 artifact를 연결한다. |
| 공개 범위 | Public cases·framework, judge 일부 비공개 | Source 공개, 외부 SAE·평가 dependency 필요 | “오픈소스”를 code·data·judge·weight·license로 분해한다. |
| 과잉 주장 위험 | Async가 낮은 성공률의 원인이라고 단정 | Behavioral suppression을 certified deletion으로 단정 | 실험 설계가 식별하는 주장만 한다. |

두 연구는 서로 다른 분야처럼 보이지만 공통 질문을 던진다.

$$
\boxed{
\text{결과가 좋아 보이는가?}
\quad\text{보다}\quad
\text{그 결과를 지지하는 상태·과정·증거가 보존됐는가?}
}
$$

---

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

100개 병원이 환자 시계열 $y_{1:T}$에서 숨은 질병 상태 $z_{0:T}$를 추론하는 상각 스무더

$$
q_\phi(z_{0:T}\mid y_{1:T})
$$

를 연합학습한다. 각 병원은 FedAvg update를 client 단위로 clipping하고 secure aggregation으로 합치며, client-level DP accountant를 운영한다. Validation ELBO가 계속 높아졌고 서비스는 전체 시계열을 받은 뒤 과거 상태를 잘 복원한다.

이때 병원 A가 자신의 전체 기여 삭제를 요청했다. 다음 네 문장을 하나의 논리로 설명해 보라.

1. 높은 ELBO가 왜 참 smoothing posterior를 정확히 찾았다는 증명은 아닌가?
2. 이 full smoother를 왜 시점 $t$의 실시간 경보에 그대로 쓰면 안 되는가?
3. Secure aggregation과 client-level DP가 있는데도 왜 “A는 이미 삭제됐다”고 말할 수 없는가?
4. 어떤 수학적 기준선과 artifact 절차를 거쳐야 검증 가능한 삭제에 가까워지는가?

### 모범 답안

첫째, ELBO 항등식은

$$
\log p_\theta(y_{1:T})
=
\mathcal L_{\mathrm{seq}}(\theta,\phi;y_{1:T})
+
D_{\mathrm{KL}}
\left(
q_\phi(z_{0:T}\mid y_{1:T})
\|
p_\theta(z_{0:T}\mid y_{1:T})
\right)
$$

이다. $\theta$와 같은 관측을 고정하면 ELBO 상승은 이 항등식에 따라 reverse KL 감소를 뜻한다. 하지만 현재처럼 $\theta$와 $\phi$를 공동 학습할 때는 $\log p_\theta(y_{1:T})$ 자체도 움직이므로, ELBO가 커졌다는 사실만으로 오른쪽 KL이 작아졌다고 단정할 수 없다. 또한 제한된 variational family가 다봉·강한 시간 상관 posterior를 표현하지 못할 수 있으며, encoder는 사례별 최적 $q_y^*$에 못 미치는 amortization gap을 가질 수 있다.

따라서 held-out likelihood bound, importance weight의 concentration·ESS, posterior predictive calibration, synthetic data에서 아는 latent recovery, 일부 사례의 particle smoother·MCMC reference, test-time refinement 전후를 함께 본다. 높은 training ELBO 하나는 유용한 학습 신호이지 정확한 posterior의 certificate가 아니다.

둘째, full smoother는

$$
q_\phi(z_t\mid y_{1:T})
$$

를 계산할 때 미래 관측 $y_{t+1:T}$를 사용한다. 시점 $t$의 실시간 경보에는 그 미래가 아직 존재하지 않는다. Offline chart review에는 full smoothing을 쓸 수 있지만 online alert에는

$$
q_\phi(z_t\mid y_{1:t})
$$

인 filter 또는 허용 지연 $L$을 명시한

$$
q_\phi
\left(
z_t\mid y_{1:\min(T,t+L)}
\right)
$$

를 사용해야 한다. 평가에서도 같은 정보 경계를 지켜야 한다.

셋째, secure aggregation은 서버가 A의 개별 update를 보지 못하고 합만 복원하게 한다. Client-level DP는 정한 adjacency와 누적 $(\varepsilon,\delta)$ 아래 공개 모델분포가 A의 참여 여부에 덜 민감하도록 제한한다. 둘 다 A가 참여한 aggregate가 실제로 새 checkpoint에서 제거됐다는 명제가 아니다.

특히 DP는 과거 공개 모델이 소비한 privacy loss를 환불하지 않는다. Secure aggregation은 오히려 서버가 A의 raw delta를 보관하지 않게 하므로 사후에 그 delta만 빼는 절차를 불가능하게 할 수 있다. 또한 A가 round $r$에 영향을 주면 이후 병원들은 달라진 $w_{r+1}$에서 로컬 학습했으므로 모든 후속 update가 경로 의존적으로 달라진다.

넷째, 기준선은 개념적으로

$$
W^{-A}
\sim
A_{\mathrm{train}}(D\setminus D_A)
$$

이다. 삭제 알고리즘 출력

$$
\widehat W^{-A}
\sim
U(W,D_A,\mathcal M)
$$

가 이 재학습 기준선과 같거나, 미리 정한 distance·certificate 아래 가깝다는 주장을 해야 한다. 이를 위해 다음 순서를 따른다.

1. A라는 법적·조직적 주체를 실제 client ID·data snapshot·중복·증강 데이터에 정확히 연결한다.
2. A가 처음 참여한 round 이전의 신뢰 가능한 checkpoint와 당시 code·seed policy·cohort·optimizer·accountant manifest를 찾는다.
3. Lineage DAG에서 A의 source node로부터 도달 가능한 aggregate, optimizer state, global checkpoint, smoother encoder, quantized copy, adapter, student, index, cache, endpoint를 계산한다.
4. 요구된 보장에 따라 full retraining, checkpoint replay, 사전 구조화한 cohort·shard retraining 또는 명시된 certified approximation을 수행한다.
5. 새 model generation에서 retain utility, smoothing calibration, forget criterion, membership·inversion audit, adversarial recovery, reference closeness를 검증한다.
6. 새 hash와 parent manifest를 등록하고 endpoint alias를 전환한 뒤 옛 generation·replica·cache를 quarantine·tombstone 처리한다.
7. Client 측 raw data·local optimizer artifact 삭제와 서버 측 후손 폐기를 각각 증명하되, 외부로 이미 반출된 checkpoint와 과거 privacy loss는 회수할 수 없다는 시스템 경계를 명시한다.

네 답의 공통 원리는 다음과 같다.

$$
\boxed{
\text{좋은 근사·보호된 합·작은 개인정보 영향}
\neq
\text{정확한 posterior·실시간 정당성·검증된 삭제}.
}
$$

각 주장은 서로 다른 기준선, 관측 가능 정보, threat model과 증거를 요구한다.

---

**다음 학습 예고:** 오늘은 ELBO·재매개변수화·상각 스무딩으로 반복 추론비를 줄이고, FedAvg·secure aggregation·client-level DP·federated unlearning을 분산 lineage와 연결했다. 다음에는 normalizing flow·importance-weighted ELBO·posterior calibration으로 근사 family와 진단을 확장하고, Byzantine-robust federated learning·verifiable aggregation·privacy-preserving audit로 기밀성·무결성·감사 가능성의 충돌을 다룬다.
