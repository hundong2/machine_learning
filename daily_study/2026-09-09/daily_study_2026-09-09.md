# [2026-09-09] 오늘 학습: 고윳값·고유벡터 & 과적합·정규화

> **오늘의 핵심 문장:** 고유벡터는 행렬이 섞지 않고 크기만 바꾸는 특별한 방향이며, 정규화는 모델이 훈련 데이터의 우연한 방향을 지나치게 증폭하지 못하도록 학습 규칙에 제약을 더한다.

지난 학습에서는 평균 중심화된 데이터 행렬 $\mathbf{X}_c$로 공분산 행렬

$$
\boldsymbol{\Sigma}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
$$

을 만들었다. 이 행렬은 데이터가 여러 방향으로 얼마나 퍼졌는지를 모두 담지만, 행렬의 숫자만 바라봐서는 **어느 방향의 분산이 가장 큰지** 바로 알기 어렵다. 오늘은 공분산 행렬이 방향은 유지한 채 길이만 바꾸는 벡터를 찾아 이 질문에 답한다.

동시에 신경망에서는 또 다른 질문이 생긴다. 모델이 훈련 데이터를 거의 외울 만큼 잘 맞추는데, 왜 처음 보는 데이터에서는 실패할까? 답은 **과적합**과 **일반화**의 차이에 있다. 이를 줄이기 위해 파라미터 크기, 뉴런 의존성, 학습 시간 등에 제약을 거는 여러 **정규화** 전략을 사용한다.

오늘의 연결 고리는 다음과 같다. 아래 연결은 평균 중심화된 입력을 쓰는 최소제곱 선형 회귀에서 데이터 손실의 헤시안이 $\boldsymbol{\Sigma}$가 되는 경우를 나타낸다.

$$
\underbrace{\boldsymbol{\Sigma}}_{\text{데이터의 방향별 분산}}
\xrightarrow{\boldsymbol{\Sigma}\mathbf{q}_i=\lambda_i\mathbf{q}_i}
\underbrace{(\lambda_i,\mathbf{q}_i)}_{\text{분산 크기와 특별한 방향}}
\xrightarrow{\text{L2 정규화}}
\underbrace{\lambda_i+\alpha}_{\text{모든 방향의 곡률 보강}}
\longrightarrow
\underbrace{\text{계수 민감도 완화}}_{\text{저분산·정보 부족 방향}}
$$

여기서 $\lambda_i$는 행렬의 고윳값이고, $\mathbf{q}_i$는 그 고유벡터다. $\alpha>0$는 오늘 사용할 L2 정규화 강도다. 두 기호를 모두 $\lambda$로 쓰는 책도 많지만, 오늘은 고윳값과 정규화 강도를 혼동하지 않도록 서로 다른 글자를 사용한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 행렬이 모든 방향을 뒤섞는다면 어떻게 핵심 방향을 찾을까?

행렬은 벡터를 다른 벡터로 바꾸는 선형 변환이다. 일반적인 벡터 $\mathbf{x}$에 행렬 $\mathbf{A}$를 곱하면 길이뿐 아니라 방향도 바뀐다.

$$
\mathbf{x}\longmapsto\mathbf{A}\mathbf{x}
$$

예를 들어 종이에 그린 화살표를 늘이고, 누르고, 회전하고, 기울이는 변환을 생각할 수 있다. 모든 화살표의 변화를 하나씩 추적하면 복잡하다. 그런데 어떤 특별한 방향은 변환 뒤에도 같은 직선 위에 남고 크기와 부호만 바뀐다.

$$
\boxed{
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
}
$$

이때 $\mathbf{v}\ne\mathbf{0}$가 **고유벡터**, $\lambda$가 **고윳값**이다. 고유벡터는 행렬이 자신의 성질을 가장 단순하게 드러내는 좌표축과 같다. 충분한 수의 선형독립 고유벡터가 있을 때 복잡한 변환을 이 축들로 분해하면 “어느 방향을 얼마나 늘이거나 줄이는가?”라는 쉬운 문제로 바뀐다. 특히 오늘 다루는 실수 대칭 공분산 행렬에서는 항상 이런 정규직교 고유기저가 존재한다.

### 1.2 공분산 행렬에서 고유벡터는 왜 특별할까?

길이가 $1$인 단위벡터 $\mathbf{u}$ 방향으로 중심화된 데이터를 투영하면 스칼라 값 $z=\mathbf{u}^{\top}\mathbf{x}_c$를 얻는다. 이 투영값의 분산은 지난 학습에서

$$
\operatorname{Var}(z)
=\mathbf{u}^{\top}\boldsymbol{\Sigma}\mathbf{u}
$$

임을 배웠다. 따라서 가장 큰 분산 방향을 찾는 문제는

$$
\max_{\lVert\mathbf{u}\rVert_2=1}
\mathbf{u}^{\top}\boldsymbol{\Sigma}\mathbf{u}
$$

가 된다. 오늘 증명하겠지만 이 최댓값은 가장 큰 고윳값 $\lambda_1$이고, 최댓값을 만드는 방향은 그에 대응하는 단위 고유벡터 $\mathbf{q}_1$이다. 즉, 고유벡터는 데이터의 분산을 최대한 보존하는 새 축을 찾는 **주성분분석(PCA)**이 사용할 “데이터가 가장 넓게 퍼진 축”의 언어다.

### 1.3 훈련 점수가 좋은데 왜 실전에서는 틀릴까?

다음 두 학생을 생각해 보자.

- 학생 A는 연습문제의 숫자와 답을 통째로 외웠다.
- 학생 B는 문제를 푸는 규칙을 이해해 숫자가 바뀌어도 적용한다.

연습문제를 그대로 다시 내면 A가 완벽해 보일 수 있다. 그러나 새로운 문제를 내면 B가 더 안정적이다. 머신러닝에서도 훈련 데이터의 우연한 잡음, 배경, 문장 표현까지 외운 모델은 훈련 손실은 낮지만 새로운 데이터에서 성능이 떨어질 수 있다. 이것이 **과적합**이다.

우리가 진짜 원하는 것은 훈련 샘플 자체의 암기가 아니라, 같은 데이터 생성 원리에서 나온 처음 보는 샘플에도 잘 작동하는 **일반화**다.

### 1.4 왜 큰 모델을 그냥 오래 학습하면 안 될까?

표현력이 큰 모델은 유용한 규칙과 불필요한 우연을 모두 표현할 수 있다. 데이터가 적거나 편향되어 있고 학습을 오래 계속하면, 손실을 더 줄이는 마지막 몇 단계가 실제 규칙보다 개별 샘플의 노이즈를 맞추는 방향으로 갈 수 있다.

정규화는 모델에게 다음과 같은 추가 선호를 부여한다.

- 같은 훈련 손실이라면 지나치게 큰 가중치보다 작은 가중치를 선호한다.
- 특정 뉴런 하나에만 의존하지 않도록 일부 경로를 무작위로 끈다.
- 검증 성능이 나빠지기 시작하면 학습을 멈춘다.
- 의미가 같은 입력에서는 비슷한 출력을 내도록 행동 자체를 제약한다.

정규화는 “복잡한 모델은 나쁘다”는 선언이 아니다. 제한된 데이터만 보고 선택해야 할 때, 여러 가능한 해 중 더 안정적인 해를 선호하도록 만드는 **귀납적 편향**이다.

### 1.5 오늘 두 트랙이 만나는 지점: 작은 고윳값 방향은 왜 위험할까?

선형 회귀에서 어떤 방향의 데이터 분산, 즉 공분산 행렬의 고윳값 $\lambda_i$가 매우 작다고 하자. 그 방향에는 관측 정보가 거의 없다. 그런데 훈련 데이터의 작은 잡음까지 맞추려 하면 그 약한 방향의 가중치가 크게 흔들릴 수 있다.

평균 중심화된 최소제곱 선형 회귀에서는 데이터 손실의 헤시안이 $\boldsymbol{\Sigma}$다. 목적함수에 L2 정규화를 더하면 이 헤시안이

$$
\boldsymbol{\Sigma}
\longrightarrow
\boldsymbol{\Sigma}+\alpha\mathbf{I}
$$

로 바뀐다. 모든 고유방향의 고윳값은

$$
\lambda_i
\longrightarrow
\lambda_i+\alpha
$$

로 이동한다. 특히 원래 $\lambda_i$가 작았던 정보 부족 방향에서 상대적인 안정화 효과가 크다. 이것이 고윳값·고유벡터가 단지 PCA의 도구가 아니라 정규화와 최적화를 이해하는 언어이기도 한 이유다.

## 2. 친절한 용어 사전

### 2.1 수학 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 고유벡터 | $\mathbf{v}$ | 행렬을 곱해도 같은 직선 방향에 남는 영벡터가 아닌 벡터다. 길이는 임의이므로 보통 단위벡터로 만든다. |
| 고윳값 | $\lambda$ | 고유벡터 방향을 행렬이 몇 배로 바꾸는지 나타내는 스칼라다. 음수면 방향의 앞뒤도 뒤집힌다. |
| 고유방정식 | $\mathbf{A}\mathbf{v}=\lambda\mathbf{v}$ | 행렬 작용이 특정 벡터에서는 단순한 스칼라배가 된다는 식이다. |
| 단위행렬 | $\mathbf{I}$ | 어떤 벡터에 곱해도 그대로 두는 행렬이다. 숫자 $1$의 행렬 버전이다. |
| 행렬식 | $\det(\mathbf{A})$ | 선형 변환이 넓이·부피를 얼마나 바꾸는지 나타내며, $0$이면 공간의 적어도 한 방향이 눌려 역행렬이 없다. |
| 특성방정식 | $\det(\mathbf{A}-\lambda\mathbf{I})=0$ | 영벡터가 아닌 고유벡터가 존재하도록 만드는 $\lambda$의 조건이다. |
| 특성다항식 | $p(\lambda)$ | $\det(\mathbf{A}-\lambda\mathbf{I})$를 $\lambda$에 대한 다항식으로 본 것이다. 그 근이 고윳값이다. |
| 고유공간 | $\ker(\mathbf{A}-\lambda\mathbf{I})$ | 같은 고윳값에 속하는 모든 고유벡터와 영벡터의 집합이다. |
| 직교 | $\mathbf{u}^{\top}\mathbf{v}=0$ | 두 벡터의 내적이 $0$인 상태다. 기하학적으로 서로 수직이다. |
| 정규직교 | $\mathbf{q}_i^{\top}\mathbf{q}_j=\delta_{ij}$ | 서로 직교하면서 각 길이가 $1$인 벡터 모음이다. |
| 대칭행렬 | $\mathbf{A}=\mathbf{A}^{\top}$ | 주대각선을 기준으로 양쪽 값이 같은 행렬이다. 실수 대칭행렬은 정규직교 고유벡터 기저를 갖는다. |
| 스펙트럼 분해 | $\mathbf{A}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}$ | 대칭행렬을 고유벡터 축으로 회전하고, 축별로 고윳값만큼 늘인 뒤 되돌리는 분해다. |
| 레일리 몫 | $R(\mathbf{u})=\frac{\mathbf{u}^{\top}\mathbf{A}\mathbf{u}}{\mathbf{u}^{\top}\mathbf{u}}$ | 방향 $\mathbf{u}$에서 행렬이 만드는 이차 크기를 길이로 보정한 값이다. |
| 대각합 | $\operatorname{tr}(\mathbf{A})$ | 행렬 주대각선 원소의 합이다. 고윳값의 합과 같고, 공분산 행렬에서는 전체 특징 분산과 같다. |
| 중복 고윳값 |  | 같은 고윳값이 특성다항식의 근으로 여러 번 나타나는 경우다. 최대 고윳값이 중복되면 최대 분산 방향 하나가 유일하지 않을 수 있다. |
| 영공간 | $\ker(\mathbf{A})$ | $\mathbf{A}\mathbf{x}=\mathbf{0}$이 되는 모든 벡터의 집합이다. |

여기서 $\delta_{ij}$는 크로네커 델타다. $i=j$이면 $1$, 다르면 $0$이다.

### 2.2 AI 학습 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 파라미터 | $\boldsymbol{\theta}$, $\mathbf{w}$ | 훈련 데이터로부터 모델이 직접 조정하는 가중치와 편향이다. |
| 하이퍼파라미터 | $\alpha$, $p$ 등 | 학습률, 정규화 강도, 드롭 확률처럼 학습 절차를 정하는 설정이다. 검증 세트로 선택한다. |
| 미니배치 |  | 전체 훈련 데이터 중 한 번의 순전파·역전파에 함께 사용하는 작은 샘플 묶음이다. |
| 훈련 세트 | $D_{\mathrm{train}}$ | 모델 파라미터를 실제로 업데이트하는 데 쓰는 데이터다. |
| 검증 세트 | $D_{\mathrm{val}}$ | 학습률, 정규화 강도, 학습 중단 시점처럼 설정을 고르는 데이터다. 가중치 업데이트에 직접 쓰지 않는다. |
| 테스트 세트 | $D_{\mathrm{test}}$ | 모든 선택이 끝난 뒤 최종 일반화 성능을 한 번 평가하는 봉인된 데이터다. |
| 경험위험 | $\widehat{R}_{\mathrm{train}}$ | 유한한 훈련 샘플에서 계산한 평균 손실이다. |
| 기대위험 | $R$ | 실제 데이터 분포에서 앞으로 만날 모든 샘플에 대한 평균 손실의 이상적 개념이다. |
| 일반화 |  | 학습에 쓰지 않은 같은 과업의 데이터에서도 성능을 유지하는 능력이다. |
| 일반화 간극의 경험적 추정 | $\widehat{R}_{\mathrm{val}}-\widehat{R}_{\mathrm{train}}$ | 검증 손실과 훈련 손실의 차이로 실제 일반화 간극을 추정한 값이다. 큰 양의 차이는 과적합 신호일 수 있다. |
| 과적합 | overfitting | 훈련 데이터의 규칙뿐 아니라 우연한 잡음까지 맞춰 새 데이터 성능이 나빠지는 상태다. |
| 과소적합 | underfitting | 모델이나 학습이 너무 제한되어 훈련 데이터의 기본 규칙조차 충분히 배우지 못한 상태다. |
| 모델 용량 | capacity | 모델이 표현할 수 있는 함수의 복잡도와 다양성을 뜻한다. 파라미터 수만으로 완전히 결정되지는 않는다. |
| 귀납적 편향 | inductive bias | 제한된 데이터에서 어떤 종류의 해를 더 선호하도록 만드는 가정이다. |
| 정규화 | regularization | 훈련 손실만 줄이는 것 외에 안정적이고 단순한 해를 선호하게 하는 전략들의 총칭이다. |
| L2 정규화 | $\frac{\alpha}{2}\lVert\mathbf{w}\rVert_2^2$ | 큰 가중치에 제곱 비례 비용을 부과한다. 선형 회귀에서는 릿지 회귀 또는 릿지 정규화라고 부른다. |
| 가중치 감쇠 | weight decay | 업데이트할 때 가중치를 조금씩 줄이는 구현 방식이다. 기본 확률적 경사하강법(SGD)에서는 L2와 같지만 모든 최적화기에서 자동으로 같은 것은 아니다. |
| Dropout |  | 훈련 중 일부 활성값을 무작위로 $0$으로 만들어 특정 경로에 대한 과도한 의존을 줄이는 기법이다. |
| 유지 확률 | $q$ | Dropout에서 한 활성값을 남길 확률이다. 드롭 확률이 $p$라면 $q=1-p$다. |
| Batch Normalization | BatchNorm | 미니배치의 특징별 통계로 활성값을 정규화하고 학습 가능한 크기·이동을 적용한다. |
| Layer Normalization | LayerNorm | 샘플 또는 토큰 하나의 특징축 통계로 정규화한다. Transformer에서 널리 쓰인다. |
| Early Stopping |  | 검증 성능이 더 좋아지지 않으면 학습을 멈추고 가장 좋았던 체크포인트를 선택한다. |
| 데이터 누수 | leakage | 검증·테스트 정보가 훈련 과정에 섞여 실제보다 성능이 좋아 보이는 문제다. |
| 분포 이동 | distribution shift | 배포 데이터의 조건이 훈련 데이터와 달라지는 현상이다. |

### 2.3 오늘의 트렌드 언어

| 용어 | 영문·표기 | 초보자 해설 |
|---|---|---|
| VLM | Vision-Language Model | 이미지·영상 같은 시각 정보와 언어를 함께 처리하는 비전-언어 모델이다. |
| 멀티모달 추론 | multimodal reasoning | 텍스트뿐 아니라 이미지, 영상, 오디오 등 여러 종류의 입력을 함께 사용해 판단하는 과정이다. |
| 지각 근거화 | perceptual grounding | 답을 이미지 영역, 영상 프레임, 시간 구간, 오디오 사건 같은 실제 입력 근거와 연결하는 것이다. |
| 증거 정렬 | evidence alignment | 모델의 답과 설명이 과업에 필요한 실제 지각 증거로 뒷받침되는 성질이다. |
| 지름길 학습 | shortcut learning | 본질적 규칙 대신 배경, 문구, 위치처럼 우연히 정답과 함께 나타난 단서를 이용하는 현상이다. |
| 반사실적 평가 | counterfactual evaluation | 핵심 영역을 가리거나 오디오를 끄는 등 한 요소를 바꾼 뒤 예측 변화로 실제 의존성을 검사하는 평가다. |
| 보상 모델 | reward model | 상태나 행동 결과를 보고 얼마나 좋은지 스칼라 점수를 내는 모델이다. 강화학습은 이 점수를 학습 신호로 사용할 수 있다. |
| 궤적 | $\tau$ | 로봇이 시간에 따라 관측하고 행동한 연속 기록이다. |
| 패러프레이즈 | paraphrase | 의미는 유지하면서 단어나 문장 구조를 바꾼 표현이다. |
| 불변성 | invariance | 의미가 같은 변형을 가해도 필요한 출력은 유지되는 성질이다. |
| SCR | Score Crossing Rate | 같은 궤적이 표현만 바뀐 지시문 때문에 실패와 성공 양쪽 점수를 모두 받는 비율이다. |
| 사전논문 | preprint | 동료 심사나 최종 출판 전에 빠르게 공개된 연구 원고다. 결과를 유용한 초기 증거로 보되 독립 재현과 한계를 함께 확인해야 한다. |

## 3. 수학의 해부학 (증명과 원리)

### 3.1 고유방정식은 “방향이 유지된다”를 식으로 쓴 것이다

이 절에서는 $\mathbf{A}\in\mathbb{R}^{d\times d}$의 실수 고유쌍 $\lambda\in\mathbb{R}$, $\mathbf{v}\in\mathbb{R}^d$를 먼저 다룬다. 일반 실수행렬에는 복소 고유쌍도 있을 수 있지만, 뒤에서 집중할 실수 대칭행렬과 공분산 행렬의 고윳값은 모두 실수다. 고유방정식

$$
\mathbf{A}\mathbf{v}=\lambda\mathbf{v}
$$

라면 $\mathbf{A}\mathbf{v}$는 $\mathbf{v}$와 같은 직선 위에 있다.

- $\lambda>1$: 같은 방향으로 길이가 늘어난다.
- $0<\lambda<1$: 같은 방향으로 길이가 줄어든다.
- $\lambda=0$: 그 방향이 영벡터로 눌린다.
- $\lambda<0$: 직선은 유지되지만 앞뒤가 뒤집힌다.

영벡터 $\mathbf{0}$는 어떤 $\lambda$에도

$$
\mathbf{A}\mathbf{0}=\lambda\mathbf{0}
$$

를 만족하므로 방향 정보를 주지 못한다. 그래서 고유벡터 정의에서 반드시 $\mathbf{v}\ne\mathbf{0}$라고 한다.

또한 $\mathbf{v}$가 고유벡터라면 $c\ne0$인 임의의 스칼라 $c$에 대해

$$
\mathbf{A}(c\mathbf{v})
=c\mathbf{A}\mathbf{v}
=c\lambda\mathbf{v}
=\lambda(c\mathbf{v})
$$

이므로 $c\mathbf{v}$도 같은 고윳값의 고유벡터다. 고유벡터의 길이는 본질이 아니고 방향이 본질이므로 보통

$$
\mathbf{q}=\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}
$$

로 정규화한다.

### 3.2 왜 $\det(\mathbf{A}-\lambda\mathbf{I})=0$을 풀까?

고유방정식을 한쪽으로 모으면

$$
\mathbf{A}\mathbf{v}-\lambda\mathbf{v}=\mathbf{0}
$$

이다. $\lambda\mathbf{v}=\lambda\mathbf{I}\mathbf{v}$이므로

$$
(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=\mathbf{0}
$$

가 된다. 만약 $\mathbf{A}-\lambda\mathbf{I}$가 역행렬을 가진다면 양변에 역행렬을 곱해

$$
\mathbf{v}=\mathbf{0}
$$

만 남는다. 하지만 고유벡터는 영벡터가 아니어야 한다. 따라서 $\mathbf{A}-\lambda\mathbf{I}$는 역행렬이 없어야 하며, 정사각행렬이 특이행렬일 조건은

$$
\boxed{
\det(\mathbf{A}-\lambda\mathbf{I})=0
}
$$

이다.

$2\times2$ 행렬

$$
\mathbf{A}
=\begin{bmatrix}
a&b\\
c&d
\end{bmatrix}
$$

라면

$$
\begin{aligned}
\det(\mathbf{A}-\lambda\mathbf{I})
&=
\det
\begin{bmatrix}
a-\lambda&b\\
c&d-\lambda
\end{bmatrix}\\
&=(a-\lambda)(d-\lambda)-bc
\end{aligned}
$$

이다. 이 이차방정식의 근을 먼저 구하고, 각 근을 $(\mathbf{A}-\lambda\mathbf{I})\mathbf{v}=\mathbf{0}$에 넣어 대응하는 방향을 구한다.

### 3.3 공분산 행렬 예제로 고유쌍을 직접 구해 보자

다음 공분산 행렬을 생각하자.

$$
\boldsymbol{\Sigma}
=\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}
$$

특성방정식은

$$
\begin{aligned}
0
&=\det(\boldsymbol{\Sigma}-\lambda\mathbf{I})\\
&=
\det
\begin{bmatrix}
2-\lambda&1\\
1&2-\lambda
\end{bmatrix}\\
&=(2-\lambda)^2-1\\
&=\lambda^2-4\lambda+3\\
&=(\lambda-3)(\lambda-1)
\end{aligned}
$$

이므로 고윳값은

$$
\lambda_1=3,
\qquad
\lambda_2=1
$$

이다.

$\lambda_1=3$을 넣으면

$$
\begin{bmatrix}
-1&1\\
1&-1
\end{bmatrix}
\begin{bmatrix}
v_1\\v_2
\end{bmatrix}
=\mathbf{0}
$$

이므로 $v_1=v_2$다. 단위 고유벡터 하나는

$$
\mathbf{q}_1
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}
$$

이다. $\lambda_2=1$에서는 $v_1=-v_2$이므로

$$
\mathbf{q}_2
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\-1
\end{bmatrix}
$$

를 얻는다. 두 방향은 실제로 직교한다.

$$
\mathbf{q}_1^{\top}\mathbf{q}_2
=\frac{1}{2}(1-1)
=0
$$

기하학적으로 데이터는 $(1,1)$ 방향에서 분산 $3$, $(1,-1)$ 방향에서 분산 $1$을 가진다. 첫 번째 방향의 분산은 두 번째 방향의 세 배이며, 표준편차로 표현한 퍼짐은 $\sqrt{3}$배다.

### 3.4 공분산 행렬의 고윳값은 왜 음수가 될 수 없을까?

공분산 행렬은

$$
\boldsymbol{\Sigma}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
$$

이므로 임의의 벡터 $\mathbf{z}$에 대해

$$
\begin{aligned}
\mathbf{z}^{\top}\boldsymbol{\Sigma}\mathbf{z}
&=\frac{1}{N}
\mathbf{z}^{\top}\mathbf{X}_c^{\top}\mathbf{X}_c\mathbf{z}\\
&=\frac{1}{N}
\lVert\mathbf{X}_c\mathbf{z}\rVert_2^2\\
&\ge0
\end{aligned}
$$

이다. 즉, 공분산 행렬은 양의 준정부호다.

이제 $\mathbf{q}_i$가 단위 고유벡터라면

$$
\boldsymbol{\Sigma}\mathbf{q}_i
=\lambda_i\mathbf{q}_i
$$

이고

$$
\begin{aligned}
\mathbf{q}_i^{\top}\boldsymbol{\Sigma}\mathbf{q}_i
&=\mathbf{q}_i^{\top}(\lambda_i\mathbf{q}_i)\\
&=\lambda_i\mathbf{q}_i^{\top}\mathbf{q}_i\\
&=\lambda_i
\end{aligned}
$$

이다. 왼쪽은 양의 준정부호 성질 때문에 음수가 아니므로

$$
\boxed{\lambda_i\ge0}
$$

이다. 공분산 행렬의 고윳값은 해당 고유방향의 분산 그 자체이기 때문에 음수가 될 수 없다는 뜻이기도 하다.

### 3.5 실수 대칭행렬의 서로 다른 고유방향은 왜 직교할까?

$\mathbf{A}=\mathbf{A}^{\top}$이고

$$
\mathbf{A}\mathbf{u}=\lambda\mathbf{u},
\qquad
\mathbf{A}\mathbf{v}=\mu\mathbf{v}
$$

이며 $\lambda\ne\mu$라고 하자. 한편

$$
\mathbf{u}^{\top}\mathbf{A}\mathbf{v}
=\mathbf{u}^{\top}(\mu\mathbf{v})
=\mu\mathbf{u}^{\top}\mathbf{v}
$$

이다. 대칭성을 사용하면

$$
\begin{aligned}
\mathbf{u}^{\top}\mathbf{A}\mathbf{v}
&=(\mathbf{A}^{\top}\mathbf{u})^{\top}\mathbf{v}\\
&=(\mathbf{A}\mathbf{u})^{\top}\mathbf{v}\\
&=(\lambda\mathbf{u})^{\top}\mathbf{v}\\
&=\lambda\mathbf{u}^{\top}\mathbf{v}
\end{aligned}
$$

이다. 따라서

$$
(\lambda-\mu)\mathbf{u}^{\top}\mathbf{v}=0
$$

이고 $\lambda\ne\mu$이므로

$$
\boxed{\mathbf{u}^{\top}\mathbf{v}=0}
$$

이다. 지금까지 증명한 것은 **이미 존재하는 서로 다른 고유쌍의 고유벡터가 직교한다**는 사실이다. 여기에 “모든 실수 대칭행렬은 실수 고윳값과 $d$개의 정규직교 고유벡터를 갖는다”는 **스펙트럼 정리**를 적용한다. 같은 고윳값이 중복될 때는 그 고유공간 안에서 그람–슈미트 과정 등으로 정규직교 기저를 고를 수 있다. 스펙트럼 정리의 완전한 존재·대각화 증명은 별도의 정리이며, 여기서는 다음 분해의 출발점으로 사용한다.

### 3.6 스펙트럼 분해는 복잡한 행렬을 축별 스칼라배로 바꾼다

정규직교 고유벡터를 열로 모은 행렬을

$$
\mathbf{Q}
=\begin{bmatrix}
\mathbf{q}_1&\cdots&\mathbf{q}_d
\end{bmatrix}
$$

라 하고, 고윳값을 대각선에 놓은 행렬을

$$
\boldsymbol{\Lambda}
=\operatorname{diag}(\lambda_1,\ldots,\lambda_d)
$$

라 하자. $\mathbf{Q}$의 열들이 정규직교이므로

$$
\mathbf{Q}^{\top}\mathbf{Q}
=\mathbf{Q}\mathbf{Q}^{\top}
=\mathbf{I}
$$

이다. 고유방정식을 열별로 모으면

$$
\mathbf{A}\mathbf{Q}
=\mathbf{Q}\boldsymbol{\Lambda}
$$

이고 오른쪽에 $\mathbf{Q}^{\top}$를 곱해

$$
\boxed{
\mathbf{A}
=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}
}
$$

를 얻는다.

임의의 벡터 $\mathbf{x}$에 대한 작용은 세 단계다.

$$
\mathbf{x}
\xrightarrow{\mathbf{Q}^{\top}}
\text{고유축 좌표}
\xrightarrow{\boldsymbol{\Lambda}}
\text{축별로 }\lambda_i\text{배}
\xrightarrow{\mathbf{Q}}
\text{원래 좌표계}
$$

행렬의 거듭제곱도 쉬워진다.

$$
\mathbf{A}^k
=\mathbf{Q}\boldsymbol{\Lambda}^k\mathbf{Q}^{\top}
$$

공분산 행렬에서 $\lambda_1>\lambda_2$, $\lambda_1>0$이고 초기 벡터가 $\mathbf{q}_1$ 성분을 조금이라도 가진다면, 반복해서 곱하고 길이를 $1$로 맞추는 거듭제곱 방법은 가장 큰 고유방향으로 가까워진다.

### 3.7 레일리 몫의 최댓값이 가장 큰 고윳값임을 증명하자

공분산 행렬의 고윳값을

$$
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge0
$$

로 정렬하자. 임의의 단위벡터 $\mathbf{u}$는 정규직교 고유기저에서

$$
\mathbf{u}=\mathbf{Q}\mathbf{a}
$$

로 쓸 수 있다. 회전은 길이를 보존하므로

$$
\lVert\mathbf{u}\rVert_2^2
=\lVert\mathbf{a}\rVert_2^2
=\sum_{i=1}^{d}a_i^2
=1
$$

이다. 방향 분산은

$$
\begin{aligned}
\mathbf{u}^{\top}\boldsymbol{\Sigma}\mathbf{u}
&=(\mathbf{Q}\mathbf{a})^{\top}
(\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top})
(\mathbf{Q}\mathbf{a})\\
&=\mathbf{a}^{\top}\boldsymbol{\Lambda}\mathbf{a}\\
&=\sum_{i=1}^{d}\lambda_i a_i^2\\
&\le\lambda_1\sum_{i=1}^{d}a_i^2\\
&=\lambda_1
\end{aligned}
$$

이다. $\mathbf{a}=(1,0,\ldots,0)^{\top}$, 즉 $\mathbf{u}=\mathbf{q}_1$일 때 등호가 성립한다. 따라서

$$
\boxed{
\max_{\lVert\mathbf{u}\rVert_2=1}
\mathbf{u}^{\top}\boldsymbol{\Sigma}\mathbf{u}
=\lambda_1
}
$$

이고 최대 분산 방향은 $\mathbf{q}_1$이다. 이것이 다음 학습에서 PCA의 첫 번째 주성분을 얻는 핵심 증명이다.

최대 고윳값이 중복되면 최대 고유공간 안의 모든 단위벡터가 같은 최댓값을 만든다. 이때 “첫 번째 방향”은 하나로 유일하지 않지만, 그 방향들이 만드는 부분공간은 의미가 있다.

### 3.8 고윳값의 합은 전체 분산이다

공분산 행렬의 대각선에는 각 특징의 분산이 있다.

$$
\operatorname{tr}(\boldsymbol{\Sigma})
=\sum_{j=1}^{d}\operatorname{Var}(X_j)
$$

한편 스펙트럼 분해와 대각합의 순환 성질을 사용하면

$$
\begin{aligned}
\operatorname{tr}(\boldsymbol{\Sigma})
&=\operatorname{tr}
(\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top})\\
&=\operatorname{tr}
(\boldsymbol{\Lambda}\mathbf{Q}^{\top}\mathbf{Q})\\
&=\operatorname{tr}(\boldsymbol{\Lambda})\\
&=\sum_{i=1}^{d}\lambda_i
\end{aligned}
$$

이다. 즉,

$$
\boxed{
\text{원래 좌표축의 전체 분산}
=\text{고유축의 전체 분산}
}
$$

이다. 좌표계를 회전해도 총분산은 사라지지 않고 고유방향들로 재배분된다. PCA는 이 중 큰 고윳값 방향부터 남겨 총분산을 최대한 보존한다.

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 훈련 손실과 우리가 진짜 원하는 위험은 다르다

샘플 $(\mathbf{x}_i,y_i)$가 $N$개인 훈련 세트에서 경험위험은

$$
\widehat{R}_{\mathrm{train}}(\boldsymbol{\theta})
=\frac{1}{N}
\sum_{i=1}^{N}
\ell(f_{\boldsymbol{\theta}}(\mathbf{x}_i),y_i)
$$

이다. 여기서 $f_{\boldsymbol{\theta}}$는 파라미터 $\boldsymbol{\theta}$를 가진 모델이고, $\ell$은 샘플 하나의 손실 함수다.

하지만 진짜 목표는 아직 보지 못한 데이터까지 포함하는 기대위험

$$
R(\boldsymbol{\theta})
=\mathbb{E}_{(\mathbf{x},y)\sim P}
\left[
\ell(f_{\boldsymbol{\theta}}(\mathbf{x}),y)
\right]
$$

을 작게 만드는 것이다. $P$는 현실의 데이터 생성 분포다. 우리는 $P$ 전체를 볼 수 없으므로 별도의 검증 세트로 이를 추정한다.

```text
전체 데이터
├─ 훈련 세트: 파라미터 학습
├─ 검증 세트: 하이퍼파라미터·중단 시점 선택
└─ 테스트 세트: 모든 선택 후 최종 1회 평가
```

검증 세트를 반복해서 보며 수십 번 설정을 고르면 검증 세트에도 간접 과적합할 수 있다. 그래서 테스트 세트는 마지막까지 봉인하고, 큰 프로젝트에서는 별도의 개발용 테스트나 **교차검증**, 즉 데이터를 여러 묶음으로 번갈아 나누어 훈련·검증하는 평가 절차를 사용한다.

### 4.2 과적합과 과소적합을 학습 곡선으로 읽자

| 상태 | 훈련 손실 | 검증 손실 | 해석 |
|---|---:|---:|---|
| 과소적합 | 높음 | 높음 | 기본 규칙도 충분히 배우지 못했다. |
| 적절한 적합 | 낮음 | 낮음 | 훈련 규칙이 새 데이터에도 이어진다. |
| 과적합 | 매우 낮음 | 상대적으로 높음 | 훈련 샘플의 우연까지 외웠을 가능성이 크다. |

학습이 진행될 때 훈련 손실은 계속 내려가지만 검증 손실이 어느 순간 다시 오를 수 있다.

```text
손실
높음 |\
     | \        검증 손실
     |  \______/\
     |          \  훈련 손실
낮음 |___________\________ 학습 스텝
                 ↑
           좋은 중단 후보
```

단 한 번의 검증 손실 상승만으로 과적합을 단정하지는 않는다. 미니배치 잡음과 평가 분산이 있으므로 여러 시점의 추세, 반복 표본에서 참값을 포함하도록 설계한 추정 범위인 **신뢰구간**, 과업 지표를 함께 본다.

### 4.3 편향–분산 관점은 왜 정규화가 필요한지 보여 준다

제곱오차와 몇 가지 표준 가정 아래, 입력 $\mathbf{x}$를 고정하고 훈련 세트 $D$와 관측 잡음 $\varepsilon$에 대해 평균낸 기대 예측오차는 개념적으로

$$
\mathbb{E}_{D,\varepsilon}
\left[
(y-\widehat{f}_D(\mathbf{x}))^2
\right]
=\operatorname{Bias}_D
\left[\widehat{f}_D(\mathbf{x})\right]^2
+\operatorname{Var}_D
\left[\widehat{f}_D(\mathbf{x})\right]
+\sigma_{\varepsilon}^2
$$

로 분해할 수 있다.

- **편향:** 모델의 평균 예측이 실제 규칙에서 체계적으로 벗어난 정도다.
- **분산:** 훈련 데이터를 조금 바꿨을 때 학습된 모델이 얼마나 흔들리는가다.
- **불가약 잡음:** 입력만으로는 제거할 수 없는 관측 잡음이다.

여기서 통계적 **편향(bias)**은 신경망 층의 $\mathbf{W}\mathbf{x}+\mathbf{b}$에 들어가는 학습 파라미터 **편향 $\mathbf{b}$**와 같은 단어를 쓰지만 다른 개념이다.

모델을 지나치게 제한하면 편향이 커지고, 제한이 전혀 없으면 데이터 변화에 대한 분산이 커질 수 있다. 정규화 강도 $\alpha$는 이 균형을 조절하는 하이퍼파라미터이며 검증 성능으로 선택해야 한다.

### 4.4 L2 정규화: 큰 가중치에 부드러운 비용을 더한다

데이터 손실 $L_{\mathrm{data}}$에 L2 벌점을 더한 목적함수는

$$
\boxed{
J(\mathbf{w})
=L_{\mathrm{data}}(\mathbf{w})
+\frac{\alpha}{2}
\lVert\mathbf{w}\rVert_2^2
}
$$

이다. 그래디언트는

$$
\nabla_{\mathbf{w}}J
=\nabla_{\mathbf{w}}L_{\mathrm{data}}
+\alpha\mathbf{w}
$$

이다. 기본 경사하강법으로 업데이트하면

$$
\begin{aligned}
\mathbf{w}_{t+1}
&=\mathbf{w}_t
-\eta
\left(
\nabla L_{\mathrm{data}}(\mathbf{w}_t)
+\alpha\mathbf{w}_t
\right)\\
&=(1-\eta\alpha)\mathbf{w}_t
-\eta\nabla L_{\mathrm{data}}(\mathbf{w}_t)
\end{aligned}
$$

이다. 첫 항 때문에 매 스텝 가중치가 조금 줄어들어 **가중치 감쇠**라는 이름이 붙었다.

이 등가성은 기본 SGD에서는 정확하지만, Adam처럼 좌표별 적응형 스케일을 쓰는 최적화기에서는 손실에 L2 항을 넣는 것과 파라미터 감쇠를 분리하는 것이 일반적으로 같지 않다. AdamW는 감쇠를 그래디언트 적응 변환과 분리한다. 실무에서는 편향과 정규화 층의 크기·이동 파라미터를 감쇠 대상에서 제외하는 경우가 많지만, 이는 아키텍처와 실험으로 결정해야 한다.

### 4.5 L2 정규화를 고유방향에서 보면 무엇이 보일까?

평균 중심화된 선형 회귀를 생각하자.

$$
J(\mathbf{w})
=\frac{1}{2N}
\lVert\mathbf{X}_c\mathbf{w}-\mathbf{y}\rVert_2^2
+\frac{\alpha}{2}\lVert\mathbf{w}\rVert_2^2
$$

그래디언트를 $0$으로 두면

$$
\left(
\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
+\alpha\mathbf{I}
\right)
\mathbf{w}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{y}
$$

이다. $\boldsymbol{\Sigma}=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c$와 $\mathbf{b}=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{y}$를 쓰면

$$
(\boldsymbol{\Sigma}+\alpha\mathbf{I})\mathbf{w}
=\mathbf{b}
$$

이다. 공분산 행렬의 스펙트럼 분해

$$
\boldsymbol{\Sigma}
=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}
$$

를 대입하면

$$
\boldsymbol{\Sigma}+\alpha\mathbf{I}
=\mathbf{Q}
(\boldsymbol{\Lambda}+\alpha\mathbf{I})
\mathbf{Q}^{\top}
$$

이므로 해는

$$
\boxed{
\mathbf{w}_{\alpha}
=\mathbf{Q}
\operatorname{diag}
\left(
\frac{1}{\lambda_1+\alpha},
\ldots,
\frac{1}{\lambda_d+\alpha}
\right)
\mathbf{Q}^{\top}\mathbf{b}
}
$$

이다.

정규화가 없고 $\lambda_i>0$이면 $i$번째 고유방향의 계수는 $(\mathbf{q}_i^{\top}\mathbf{b})/\lambda_i$다. L2를 넣으면

$$
\frac{\mathbf{q}_i^{\top}\mathbf{b}}{\lambda_i}
\longrightarrow
\frac{\mathbf{q}_i^{\top}\mathbf{b}}{\lambda_i+\alpha}
$$

로 바뀌며, 상대 축소 비율은

$$
\frac{\lambda_i}{\lambda_i+\alpha}
$$

다. $\lambda_i$가 작은 저분산 방향일수록 상대적으로 더 많이 축소된다. 데이터가 거의 알려 주지 않는 방향에서 잡음 때문에 계수가 폭증하는 것을 막는 효과다.

곡률 관점에서도 모든 이계 편미분을 모은 행렬인 헤시안은

$$
\nabla^2J
=\boldsymbol{\Sigma}+\alpha\mathbf{I}
$$

이고 고윳값은 모두 $\lambda_i+\alpha$가 된다. $\alpha>0$이면 원래 영 고윳값이 있더라도 선형 회귀 목적함수는 모든 방향에서 양의 곡률을 갖는다. 다만 L2가 항상 테스트 성능을 높인다는 뜻은 아니다. $\alpha$가 너무 크면 유용한 신호까지 줄여 과소적합을 만든다.

### 4.6 Dropout: 한 경로에만 의존하지 못하게 한다

은닉 활성값 $\mathbf{h}$의 각 성분에 대해 마스크

$$
m_j\sim\operatorname{Bernoulli}(q)
$$

를 뽑자. 이는 $m_j$가 확률 $q$로 $1$, 확률 $1-q$로 $0$이 되는 베르누이 마스크라는 뜻이다. 유지 확률은 $q$, 드롭 확률은 $p=1-q$다. 훈련 중 **inverted dropout**은

$$
\widetilde{h}_j
=\frac{m_j}{q}h_j
$$

를 사용한다. 기대값은

$$
\mathbb{E}[\widetilde{h}_j]
=\frac{\mathbb{E}[m_j]}{q}h_j
=h_j
$$

이므로 훈련 중 평균 크기가 유지된다. 평가 및 일반 추론 단계에서는 마스크를 끄고 $\mathbf{h}$를 그대로 사용한다.

Dropout은 매 스텝 서로 다른 부분 네트워크를 학습시키는 효과를 내고, 특징들이 특정 동료 뉴런 하나에만 의존하는 **공동 적응**을 줄일 수 있다. 흔히 여러 부분 모델을 평균내는 앙상블에 비유하지만 정확히 독립 모델들을 학습해 평균내는 것과 동일하지는 않다.

주의할 점은 다음과 같다.

- 드롭 확률이 너무 크면 정보가 지나치게 사라져 과소적합한다.
- 합성곱망, 순환망, Transformer에서는 넣는 위치와 비율의 효과가 다르다.
- 이미 데이터가 매우 크거나 다른 강한 정규화가 있으면 이득이 작거나 음수가 될 수 있다.
- 검증·테스트 때 Dropout을 켜 두면 일반 평가가 확률적으로 흔들린다. 불확실성 추정을 위한 Monte Carlo Dropout은 별도의 의도적 기법이다.

### 4.7 BatchNorm과 LayerNorm: 이름이 비슷해도 L2와 같은 정규화가 아니다

한국어에서는 regularization과 normalization을 모두 “정규화”라고 부르는 경우가 있어 혼동하기 쉽다. 두 개념은 다르다.

- **Regularization:** 모델의 유효 복잡도나 행동을 제약해 일반화를 돕는 전략이다.
- **Normalization:** 값의 중심과 스케일을 조절해 수치와 최적화 흐름을 안정화하는 변환이다.

미니배치의 한 특징에 값 $x_1,\ldots,x_m$이 있다면 BatchNorm은 훈련 중

$$
\mu_B
=\frac{1}{m}\sum_{i=1}^{m}x_i
$$

$$
\sigma_B^2
=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2
$$

를 구하고

$$
\widehat{x}_i
=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}
$$

$$
y_i
=\gamma\widehat{x}_i+\beta
$$

로 변환한다. $\varepsilon>0$는 분모가 $0$에 가까워지는 것을 막고, $\gamma,\beta$는 모델이 필요한 크기와 중심을 다시 학습하게 한다. 평가 시에는 보통 훈련 중 누적한 이동 통계를 사용한다. 작은 배치에서는 통계가 불안정할 수 있다.

LayerNorm은 샘플 또는 Transformer의 토큰 하나에서 특징축 $j=1,\ldots,d$를 따라

$$
\mu_i
=\frac{1}{d}\sum_{j=1}^{d}x_{ij}
$$

$$
\sigma_i^2
=\frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\mu_i)^2
$$

를 계산한다. 다른 샘플에 의존하지 않으므로 훈련과 추론에서 같은 방식으로 현재 샘플 통계를 사용할 수 있다. 정확한 정규화 축은 구현과 아키텍처를 확인해야 한다.

BatchNorm의 배치 통계 잡음이 부수적으로 정규화 효과를 낼 수는 있다. 하지만 BatchNorm과 LayerNorm의 주된 목적을 “과적합 방지 벌점”으로만 이해하면 안 된다. 특히 Transformer의 LayerNorm은 깊은 잔차 경로의 최적화와 수치 안정성에 핵심적이며 L2나 Dropout을 자동으로 대체하지 않는다.

### 4.8 Early Stopping과 데이터 증강: 파라미터가 아닌 학습 과정과 입력을 제약한다

Early Stopping은 검증 지표가 가장 좋았던 시점의 체크포인트를 저장하고, 일정한 **patience** 동안 개선이 없으면 중단한다. patience는 몇 번의 평가 구간을 기다릴지 정하는 값이다.

```text
매 평가 시점
1. 검증 지표가 최고 기록보다 좋아졌는가?
2. 그렇다면 체크포인트 저장, 대기 횟수 초기화
3. 아니라면 대기 횟수 증가
4. 대기 횟수가 patience에 도달하면 중단
```

데이터 증강은 이미지 자르기, 색 변화, 의미 보존 패러프레이즈처럼 정답을 바꾸지 않는 변환을 추가해 모델이 표면적 우연보다 불변 규칙을 배우도록 한다. 단, 실제로 의미가 보존되는 변환인지 도메인 지식으로 확인해야 한다. 숫자 `6`을 무심코 회전해 `9`처럼 만들거나, 의료 영상의 좌우 정보를 뒤집으면 라벨이 달라질 수 있다.

### 4.9 수학 부품과 AI 동작의 1:1 연결

| 수학 개념 | AI에서 맡는 역할 |
|---|---|
| 공분산 행렬 $\boldsymbol{\Sigma}$ | 평균 중심화된 최소제곱 선형 회귀에서 데이터의 방향별 정보량이자 데이터 손실의 헤시안이다. |
| 고유벡터 $\mathbf{q}_i$ | 서로 섞이지 않는 데이터·곡률의 기본 방향을 제공한다. |
| 고윳값 $\lambda_i$ | 고유방향의 분산 또는 곡률 크기를 나타낸다. |
| 작은 $\lambda_i$ | 데이터가 그 방향을 약하게 제약하므로 잡음에 민감한 계수가 생길 수 있음을 알린다. |
| $\alpha\mathbf{I}$ | L2가 모든 방향의 곡률에 같은 양 $\alpha$를 더한다. |
| $\frac{1}{\lambda_i+\alpha}$ | 릿지 해에서 $\mathbf{b}$의 $i$번째 고유방향 성분을 계수로 보내는 역곡률이다. 정규화 없는 최소제곱(OLS) 해 대비 상대 축소율은 $\frac{\lambda_i}{\lambda_i+\alpha}$다. |
| 레일리 몫 | 가장 큰 분산 방향을 찾는 PCA 목적을 나타낸다. |
| 분산 | Dropout 예측 변동, 데이터 변형에 대한 출력 변동, 모델의 학습 데이터 민감도를 측정하는 언어가 된다. |

### 4.10 실전에서 정규화 강도를 고르는 최소 절차

1. 먼저 데이터 분할을 고정하고 중복 샘플이나 동일 사용자의 데이터가 분할을 가로질러 새지 않는지 확인한다.
2. 정규화가 없는 기준 모델의 훈련·검증 곡선을 기록한다.
3. $\alpha$, 드롭 확률, 데이터 증강 강도, Early Stopping patience를 한꺼번에 무작정 바꾸지 말고 작은 실험표로 비교한다.
4. 평균 성능뿐 아니라 난수 생성의 출발값인 여러 **시드(seed)**에 따른 변동과 중요한 부분집합 성능을 본다.
5. 선택이 끝난 뒤에만 테스트 세트를 평가한다.
6. 배포 분포가 바뀔 수 있다면 시간, 장치, 언어, 문구, 배경별 스트레스 테스트를 따로 둔다.

### 4.11 초보자가 흔히 하는 오해와 주의할 점

#### 오해 1: “모든 벡터가 고유벡터다.”

일반 벡터는 행렬 변환 뒤 방향이 바뀐다. 같은 직선 위에 남는 영벡터가 아닌 특별한 방향만 고유벡터다.

#### 오해 2: “고유벡터는 하나의 정확한 화살표다.”

$\mathbf{v}$와 $c\mathbf{v}$는 같은 고유방향이다. 부호가 반대인 $-\mathbf{v}$도 같은 축을 나타낸다.

#### 오해 3: “모든 행렬의 고윳값은 실수이고 고유벡터는 직교한다.”

일반 실수 행렬은 복소 고윳값을 갖거나 고유벡터가 직교하지 않을 수 있다. 오늘의 강한 성질은 공분산 행렬이 실수 대칭행렬이기 때문에 성립한다.

#### 오해 4: “가장 큰 고윳값의 고유벡터는 언제나 유일하다.”

최대 고윳값이 중복되면 최대 고유공간 안의 여러 방향이 같은 최대 분산을 만든다.

#### 오해 5: “훈련 정확도가 $100\%$면 모델이 완성됐다.”

훈련 데이터 암기일 수 있다. 검증·테스트와 실제 배포 조건에서 일반화를 확인해야 한다.

#### 오해 6: “검증 세트는 훈련에 전혀 영향을 주지 않는다.”

검증 세트로 모델과 설정을 반복 선택하면 그 정보가 선택 과정에 들어간다. 그래서 별도의 테스트 세트가 필요하다.

#### 오해 7: “정규화는 무조건 성능을 높인다.”

너무 강한 L2, Dropout, 증강은 유용한 신호까지 억제해 과소적합을 만든다. 강도는 검증으로 고른다.

#### 오해 8: “L2 정규화는 작은 고윳값 방향만 줄인다.”

모든 방향을 줄이지만 상대 축소율 $\lambda_i/(\lambda_i+\alpha)$ 때문에 작은 고윳값 방향이 더 강하게 축소된다.

#### 오해 9: “L2와 weight decay는 어떤 최적화기에서도 같다.”

기본 SGD에서는 같지만 적응형 최적화기에서는 일반적으로 다르다. AdamW처럼 감쇠를 분리한 구현과 손실에 L2를 더한 구현을 구분해야 한다.

#### 오해 10: “Normalization은 이름 그대로 과적합을 막는 regularization이다.”

Normalization의 주된 역할은 활성값의 스케일과 최적화 흐름을 다루는 것이다. 일반화에 간접 효과가 있을 수 있지만 L2·Dropout과 같은 개념은 아니다.

#### 오해 11: “Dropout은 추론 때도 켜야 학습과 같다.”

일반 추론에서는 끈다. inverted dropout이 훈련 중 기대 크기를 맞춰 두었기 때문이다. 불확실성 추정을 위한 반복 확률 추론은 별도 설정이다.

#### 오해 12: “PCA는 오늘 완성됐다.”

오늘은 최대 분산 방향이 최대 고윳값의 고유벡터임을 증명했다. 다음에는 중심화, 성분 선택, 투영, 재구성, 설명분산비를 하나의 PCA 알고리즘으로 완성한다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 조사 기준 시각: 2026-09-09 02:37 (Asia/Seoul). 이전 학습에서 다룬 K2 Horizon과 비디오 VLM Attention Knockout은 반복하지 않았다. 오늘은 2026-09-08 열린 공식 워크숍과 arXiv 표기 2026-09-04에 제출된 최근 VLM 사전논문을 1차 출처로 확인했다. 후자의 제출 시각은 2026-09-04 17:47:58 UTC, 즉 2026-09-05 02:47:58 KST다. 워크숍의 문제 제기와 논문 저자의 실험 결과는 독립 재현된 업계 표준과 구분해 읽어야 한다.

### 5.1 BEAM 2 @ ECCV 2026: “정답을 맞혔는가?”에서 “올바른 증거로 맞혔는가?”로

2026-09-08 스웨덴 말뫼에서 열린 ECCV 2026의 **BEAM 2: Benchmarking Evidence-Aligned Multimodal Reasoning** 워크숍은 멀티모달 평가의 중심을 최종 정확도 하나에서 **증거 정렬**로 넓히자는 흐름을 보여 준다.

VLM이 질문에 맞는 답을 냈더라도 실제 이미지나 영상을 사용했다고 단정할 수 없다. 문장에 자주 등장하는 답, 선택지 위치, 배경과 라벨의 우연한 상관만으로 맞혔을 수 있기 때문이다. BEAM 2가 강조한 평가 단위는 다음과 같다.

- 정답 정확도와 지각 근거화 품질을 함께 본다.
- 객체를 둘러싼 사각형인 바운딩 박스, 대표 프레임, 시간 구간, 오디오 사건의 발생 시각처럼 검증 가능한 근거 위치를 사용한다.
- 이미지 영역 마스킹, 오디오 음소거, 객체·사건 교체 같은 반사실적 개입으로 지름길을 검사한다.
- 답뿐 아니라 구조화된 증거 위치와 짧은 근거 설명을 출력하게 한다.

오늘의 과적합과 연결하면, 언어적 지름길은 훈련·기존 벤치마크에서만 통하는 우연한 특징에 적합한 사례다. 답 정확도만 있는 검증 세트는 이 과적합을 놓칠 수 있다. 입력의 본질적 근거를 제거했을 때 답이 적절히 변하는지, 무관한 표면 요소를 바꿨을 때 답이 유지되는지를 함께 시험해야 한다.

이를 개념적으로 두 축으로 기록할 수 있다.

$$
\text{평가 벡터}
=
\begin{bmatrix}
\text{답 정확도}\\
\text{근거 정렬 품질}
\end{bmatrix}
$$

두 값을 하나의 점수로 합칠 수도 있지만, 합산 점수만 보고 한 축의 실패를 숨기지 않도록 원래 두 지표도 함께 공개해야 한다.

**엔지니어 인사이트 (Impact)**

- 의료 영상, 로봇, 감시·안전 시스템에서는 “우연히 맞은 답”이 다음 입력에서 치명적 실패로 이어질 수 있다. 정답 지표와 근거 위치 지표를 별도 회귀 테스트로 관리해야 한다.
- 프롬프트와 선택지 순서, 이미지 마스킹, 핵심 프레임 제거, 오디오 제거를 지속적 통합(CI) 자동 평가 조합에 넣으면 표면 단서 의존성을 더 빨리 발견할 수 있다.
- 모델이 내놓은 자연어 설명은 그 자체로 내부 계산의 충실한 증거가 아니다. 설명의 그럴듯함과 실제 입력 개입에 대한 민감도를 분리해 평가해야 한다.
- BEAM 2는 하나의 완성된 범용 지표를 발표한 사건이라기보다 여러 데이터셋·프로토콜·지표가 향해야 할 연구 의제를 모은 워크숍이다. 워크숍의 방향성을 곧 표준 확립이나 성능 향상의 증거로 해석하면 안 된다.

**공식 자료:** [BEAM 2 워크숍 페이지](https://beamv2-eccv-workshop.github.io/) · [ECCV 2026 BEAM OpenReview](https://openreview.net/group?id=thecvf.com/ECCV/2026/Workshop/BEAM)

### 5.2 RoboRMBench: 같은 로봇 행동인데 문장만 바꾸자 보상이 뒤집혔다

arXiv 표기 2026-09-04에 제출된 사전논문 **Same Trajectory, Contradictory Rewards (RoboRMBench)**는 VLM을 로봇의 보상 모델로 사용할 때 필요한 **패러프레이즈 불변성**을 점검한다. 같은 로봇 궤적과 같은 목표 의미를 유지하고 지시문 표현만 바꿨는데 보상이 크게 달라진다면, 강화학습은 행동 품질이 아니라 문구의 우연을 최적화할 수 있다.

저자들은 실제 로봇 궤적 $2{,}390$개와 세 LLM 판정기의 보수적 의미 동등성 필터를 통과한 패러프레이즈 $21{,}673$개를 구성했다. 필터 자체는 후보 $210$개를 별도로 뽑아 사람 판정과 대조했으며, 전체 패러프레이즈를 사람이 모두 검수한 것은 아니다. 데이터는 로봇 구성 $14$종과 다양한 조작 과제를 포함하며, 표현 변화는 다음 세 단계다.

1. 어휘 치환
2. 문장 구조 재구성
3. 행동 중심 표현과 목표 상태 중심 표현 사이의 관점 전환

궤적 $\tau$에 의미가 같은 지시문 $g_1,\ldots,g_K$가 있고 보상 모델 점수가

$$
r_k=R_{\boldsymbol{\theta}}(\tau,g_k)
$$

라고 하자. 논문은 점수 $1,2$를 실패, $4,5$를 성공으로 두고 같은 궤적에서 양쪽 범주가 모두 나타나는지를 본다. 점수 $3$은 중간값으로 두어 이 이진 비교에서 제외한다. 궤적이 $M$개라면 이를 단순화한 SCR은

$$
\operatorname{SCR}
=\frac{1}{M}
\sum_{i=1}^{M}
\mathbf{1}
\left[
\min_k r_{ik}\le2
\;\land\;
\max_k r_{ik}\ge4
\right]
$$

처럼 쓸 수 있다. $\mathbf{1}[\cdot]$은 조건이 참이면 $1$, 거짓이면 $0$인 지시함수다.

저자 보고에 따르면 관점 전환 조건에서 Gemini 2.5 Flash-Lite의 SCR은 $0.607$, Llama 4 Scout는 $0.557$이었다. 즉, 해당 설정에서는 절반이 넘는 궤적이 표현 변화만으로 실패·성공 경계를 모두 건넜다. 반면 궤적 보상에 특화된 RoboReward-4B(RR-4B)와 RoboReward-8B(RR-8B)는 같은 조건에서 각각 $0.086$, $0.114$였다. 더 큰 범용 모델이 자동으로 더 불변적인 보상 모델이 되는 것은 아니었다.

특히 오늘의 정규화와 직접 맞닿는 완화 실험이 있다. 저자들은 예측 손실에 의미가 같은 문장들 사이 출력 분산을 벌점으로 더했다.

$$
\boxed{
\mathcal{L}
=\mathcal{L}_{\mathrm{pred}}
+\lambda
\operatorname{Var}_{g_k\in G(\tau)}
\left[
R_{\boldsymbol{\theta}}(\tau,g_k)
\right]
}
$$

여기서 이 식의 $\lambda>0$는 논문이 사용한 출력 일관성 정규화 강도이며, 앞 절의 고윳값 $\lambda_i$와는 다른 역할이다. L2가 **파라미터 크기**를 제약한다면 이 출력 분산 벌점은 의미가 같은 입력들에서 **함수의 출력 변화**를 제약한다. 논문의 Qwen3-VL-4B 실험에서 관점 전환 SCR은 $0.200$에서 $0.057$로, 평균 절대오차(ME)는 $1.020$에서 $0.372$로 낮아졌다. 이는 저자 실험 한 설정의 결과이며 다른 데이터·모델에서 그대로 재현된다고 가정하면 안 된다.

**엔지니어 인사이트 (Impact)**

- 보상 모델은 평균 오차만 평가해서는 부족하다. 의미 보존 패러프레이즈, 표현 순서, 카메라 시점 같은 변형에 대한 불변성 지표를 별도로 둬야 한다.
- 같은 의미의 문장 여러 개로 점수를 평균내는 추론 시 앙상블은 변동을 줄일 수 있지만 호출 비용이 늘고 문제를 완전히 없애지 못한다. 훈련 시 일관성 벌점은 추가 추론 호출 없이 행동을 제약하는 대안이다.
- 작은 과업 특화 모델이 큰 범용 VLM보다 안정적이었다는 결과는 “모델 규모”와 “배포 과업 일반화”가 같은 축이 아님을 보여 준다. 대상 분포에 맞는 검증과 감독이 중요하다.
- 연구는 영어 지시문과 에피소드 종료 시점의 보상에 한정된다. 다국어, 한 문장 안에서 여러 언어를 섞는 코드 스위칭, 매 스텝 보상으로 일반화하려면 추가 실험이 필요하다.
- 사전논문 원고는 CC BY 4.0으로 읽을 수 있지만 이것이 벤치마크 데이터·코드의 라이선스를 자동으로 정하지는 않는다. 조사 시점의 v1 arXiv 문서에는 RoboRMBench 코드·데이터를 직접 내려받는 공식 공개 저장소 링크가 연결되어 있지 않았다. 논문 공개와 엔드투엔드 재현 자산 공개를 구분해야 한다.

**원문:** [arXiv 초록](https://arxiv.org/abs/2609.05401) · [arXiv HTML 전체 논문](https://arxiv.org/html/2609.05401)

### 5.3 두 소식이 말하는 공통 흐름: 일반화는 평균 점수 하나가 아니다

BEAM 2는 모델이 올바른 **지각 증거**를 사용했는지 묻고, RoboRMBench는 같은 의미를 다른 **언어 표현**으로 말해도 판단이 유지되는지 묻는다. 둘은 모두 기존 시험 한 장의 평균 정확도로는 실제 일반화를 충분히 볼 수 없다고 말한다.

오늘의 수학 언어로 비유하면 평균 점수 하나는 고차원 성능 공간을 한 축에만 투영한 값과 같다. 실제 시스템은 정확도, 근거화, 표현 불변성, 예측 확률과 실제 정답률을 맞추는 보정, 지연, 비용 등 여러 방향을 가진다. 어떤 방향의 실패가 중요한지는 배포 위험이 결정한다.

따라서 실전의 검증 세트는 단순히 훈련 세트에서 무작위로 떼어 낸 샘플만이 아니라, 모델이 지름길을 썼는지 드러내는 **의도적 분포 이동과 불변성 테스트**를 포함해야 한다. 정규화는 학습 단계의 선호이고, 스트레스 테스트는 그 선호가 실제 행동으로 이어졌는지 확인하는 평가 단계다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

평균 중심화된 선형 회귀 문제의 공분산 행렬과 입력–정답 상관 벡터가 다음과 같다고 하자.

$$
\boldsymbol{\Sigma}
=\begin{bmatrix}
2&1\\
1&2
\end{bmatrix},
\qquad
\mathbf{b}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{y}
=\begin{bmatrix}
4\\2
\end{bmatrix}
$$

1. $\boldsymbol{\Sigma}$의 고윳값과 단위 고유벡터를 구하라.
2. 어느 방향이 최대 분산 방향이며, 그 분산은 얼마인가?
3. L2 강도 $\alpha=1$일 때 릿지 해

$$
\mathbf{w}_{\alpha}
=(\boldsymbol{\Sigma}+\alpha\mathbf{I})^{-1}\mathbf{b}
$$

를 고유방향별로 구하라.
4. 정규화가 없을 때와 비교해 어떤 고유방향이 상대적으로 더 많이 줄어드는가?
5. 이 결과가 과적합, Dropout, LayerNorm과 각각 어떤 관계인지 설명하라.

### 모범 답안

앞서 유도했듯 고윳값과 단위 고유벡터는

$$
\lambda_1=3,
\qquad
\mathbf{q}_1
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}
$$

$$
\lambda_2=1,
\qquad
\mathbf{q}_2
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\-1
\end{bmatrix}
$$

이다. 레일리 몫의 최댓값은 가장 큰 고윳값이므로 최대 분산 방향은 $\mathbf{q}_1$이고 분산은 $3$이다.

$\mathbf{b}$를 고유기저에 투영하면

$$
\begin{aligned}
b_1
&=\mathbf{q}_1^{\top}\mathbf{b}
=\frac{1}{\sqrt{2}}(4+2)
=3\sqrt{2},\\
b_2
&=\mathbf{q}_2^{\top}\mathbf{b}
=\frac{1}{\sqrt{2}}(4-2)
=\sqrt{2}
\end{aligned}
$$

이다. 정규화가 없을 때 고유방향 계수는

$$
c_1^{(0)}
=\frac{b_1}{\lambda_1}
=\sqrt{2},
\qquad
c_2^{(0)}
=\frac{b_2}{\lambda_2}
=\sqrt{2}
$$

다. $\alpha=1$일 때는

$$
c_1^{(1)}
=\frac{b_1}{\lambda_1+1}
=\frac{3\sqrt{2}}{4}
$$

$$
c_2^{(1)}
=\frac{b_2}{\lambda_2+1}
=\frac{\sqrt{2}}{2}
$$

이다. 원래 좌표로 돌아오면

$$
\begin{aligned}
\mathbf{w}_{1}
&=c_1^{(1)}\mathbf{q}_1
+c_2^{(1)}\mathbf{q}_2\\
&=\frac{3\sqrt{2}}{4}
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\1
\end{bmatrix}
+\frac{\sqrt{2}}{2}
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\-1
\end{bmatrix}\\
&=
\begin{bmatrix}
3/4\\3/4
\end{bmatrix}
+
\begin{bmatrix}
1/2\\-1/2
\end{bmatrix}\\
&=
\boxed{
\begin{bmatrix}
5/4\\1/4
\end{bmatrix}
}
\end{aligned}
$$

이다. 직접 계산해도

$$
\boldsymbol{\Sigma}+\mathbf{I}
=\begin{bmatrix}
3&1\\
1&3
\end{bmatrix}
$$

이고

$$
\begin{bmatrix}
3&1\\
1&3
\end{bmatrix}
\begin{bmatrix}
5/4\\1/4
\end{bmatrix}
=
\begin{bmatrix}
4\\2
\end{bmatrix}
$$

이므로 맞다.

고유방향별 상대 축소율은

$$
\frac{\lambda_1}{\lambda_1+\alpha}
=\frac{3}{4}
$$

과

$$
\frac{\lambda_2}{\lambda_2+\alpha}
=\frac{1}{2}
$$

이다. 따라서 분산이 작은 $\mathbf{q}_2$ 방향이 더 강하게 줄어든다. 이 방향은 데이터가 덜 뒷받침하므로 샘플 잡음에 따라 계수가 흔들리기 쉽고, L2는 그 민감도를 낮춘다. 이것이 일반화에 도움이 될 수 있지만 $\alpha$가 지나치게 크면 두 방향의 유용한 신호까지 줄어 과소적합한다.

Dropout은 이 스펙트럼 공식을 그대로 적용하는 방법이 아니다. 활성 경로를 무작위로 끄는 확률적 정규화로 특정 뉴런 조합에 대한 의존을 줄인다. LayerNorm은 각 샘플·토큰의 특징 스케일을 조절해 최적화를 안정화하는 normalization이며, 파라미터 크기를 고유방향별로 축소하는 L2와 목적과 메커니즘이 다르다.

---

**다음 연결 고리:** 이제 평균 중심화, 공분산 행렬, 고윳값·고유벡터, 최대 분산 증명이 모두 준비되었다. 다음에는 이 부품들을 **PCA의 투영·차원 축소·재구성·설명분산비**로 조립하고, AI 기초에서는 임베딩과 어텐션으로 넘어갈 준비를 한다.
