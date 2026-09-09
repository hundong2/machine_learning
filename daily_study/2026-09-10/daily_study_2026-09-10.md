# [2026-09-10] 오늘 학습: 주성분분석(PCA) & 임베딩 기초

> **오늘의 핵심 문장:** PCA는 데이터가 가장 많이 퍼진 직교 방향을 골라 정보 손실을 계산할 수 있는 저차원 좌표로 바꾸고, 임베딩은 학습 목표에 유용하도록 이산적인 대상을 연속적인 벡터 좌표로 바꾼다.

지난 학습에서는 평균 중심화된 데이터의 공분산 행렬

$$
\boldsymbol{\Sigma}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
$$

과 그 고유방정식

$$
\boldsymbol{\Sigma}\mathbf{q}_i
=\lambda_i\mathbf{q}_i
$$

를 배웠다. 가장 큰 고윳값의 고유벡터가 최대 분산 방향이라는 사실도 증명했다. 오늘은 이 부품들을 실제 알고리즘으로 조립한다.

표본을 행으로 쌓는 표기에서 원본 특징 수를 $d$, 남길 주성분 수를 $k$라 하자. 큰 고윳값에 대응하는 단위 고유벡터를 열로 모으면

$$
\mathbf{Q}_k
=
\begin{bmatrix}
\mathbf{q}_1&\cdots&\mathbf{q}_k
\end{bmatrix}
\in\mathbb{R}^{d\times k}
$$

이고 PCA의 핵심 흐름은 다음과 같다.

$$
\underbrace{\mathbf{X}}_{N\times d}
\xrightarrow{\text{평균 중심화}}
\underbrace{\mathbf{X}_c}_{N\times d}
\xrightarrow{\text{주성분 좌표}}
\underbrace{\mathbf{Z}=\mathbf{X}_c\mathbf{Q}_k}_{N\times k}
\xrightarrow{\text{원래 공간으로 복원}}
\underbrace{\widehat{\mathbf{X}}
=\mathbf{Z}\mathbf{Q}_k^{\top}
+\mathbf{1}\boldsymbol{\mu}^{\top}}_{N\times d}
$$

AI 기초 트랙에서는 단어·토큰처럼 셀 수는 있지만 크기 순서가 없는 대상을 벡터로 바꾼다. 어휘 크기가 $V$, 임베딩 차원이 $d_e$일 때 임베딩 표

$$
\mathbf{E}\in\mathbb{R}^{V\times d_e}
$$

의 $i$번째 행을 꺼내 토큰 $i$의 벡터로 사용한다. 원-핫 벡터 $\mathbf{o}_i$로 쓰면

$$
\underbrace{\mathbf{e}_i}_{d_e\times1}
=
\underbrace{\mathbf{E}^{\top}}_{d_e\times V}
\underbrace{\mathbf{o}_i}_{V\times1}
$$

이다. PCA와 임베딩은 모두 “대상을 벡터 좌표로 표현한다”는 공통점이 있지만, 좌표를 정하는 목적과 학습 방법은 다르다. 오늘은 이 공통점과 차이를 정확히 구분한다.

## 1. 지식의 씨앗: 이 개념들은 왜 탄생했을까?

### 1.1 특징이 많으면 정보도 항상 많을까?

학생 $N$명의 키, 몸무게, 팔 길이, 다리 길이, 신발 크기를 기록했다고 하자. 특징은 다섯 개지만 서로 강하게 연관되어 있을 가능성이 크다. 키가 큰 사람은 평균적으로 팔과 다리도 길기 때문이다.

이때 다섯 숫자를 모두 독립적인 정보처럼 저장하면 같은 “전체적인 체격” 신호를 여러 번 기록하게 된다. 센서 데이터에서도 이웃한 센서가 비슷한 값을 내고, 이미지에서도 이웃한 픽셀이 비슷하며, 임베딩의 여러 좌표도 서로 상관될 수 있다.

차원이 커질수록 다음 문제가 생긴다.

- 저장 공간과 계산량이 늘어난다.
- 사람은 고차원 구조를 직접 그려 보기 어렵다.
- 샘플 수에 비해 특징이 지나치게 많으면 잡음 방향까지 학습하기 쉽다.
- 서로 상관된 특징 때문에 어떤 변화가 본질적인지 해석하기 어려워진다.

따라서 단순히 열 몇 개를 버리는 것이 아니라, 여러 원래 특징을 섞어 **데이터가 가장 크게 변하는 새 축**을 만들고 중요한 축부터 남기고 싶다. 이것이 주성분분석, 즉 PCA가 해결하려는 문제다.

### 1.2 왜 원래 좌표축이 아니라 회전한 축을 찾을까?

2차원 점들이 오른쪽 위 대각선 방향으로 길게 퍼져 있다고 생각하자. $x_1$축만 남기거나 $x_2$축만 남기면 대각선 구조를 충분히 담지 못한다. 반면 두 축을 섞은

$$
\mathbf{q}_1
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

방향으로 투영하면 긴 퍼짐을 한 숫자로 요약할 수 있다.

PCA는 원래 열을 그대로 선택하는 **특징 선택**이 아니다. 원래 특징의 선형결합으로 서로 직교하는 새 축을 만드는 **특징 추출**이다. “키 열만 남긴다”가 아니라 “키·팔 길이·다리 길이를 섞어 전체 체격 축을 만든다”에 가깝다.

### 1.3 왜 평균을 먼저 빼야 할까?

PCA가 찾으려는 것은 원점에서 멀리 떨어진 방향이 아니라 **평균 주변에서 데이터가 변하는 방향**이다. 모든 점에 같은 상수 벡터를 더해 멀리 옮겨도 점들 사이의 모양은 바뀌지 않는다. 그런데 원점 기준의 제곱 크기를 그대로 사용하면 이 평행이동이 주성분을 왜곡할 수 있다.

그래서 각 표본 $\mathbf{x}_n$에서 평균 $\boldsymbol{\mu}$를 빼

$$
\mathbf{x}_{c,n}
=\mathbf{x}_n-\boldsymbol{\mu}
$$

로 만든다. 이는 데이터 구름의 중심을 원점으로 옮기는 과정이다. PCA의 투영 방향은 이 중심화된 구름에서 찾는다.

### 1.4 압축한 뒤 무엇을 잃었는지 어떻게 알까?

압축은 마술이 아니다. $d$차원을 $k<d$차원으로 줄이면 일반적으로 일부 정보가 사라진다. 좋은 차원 축소는 “손실이 없다”고 말하는 대신 다음을 수치로 답한다.

1. 남긴 축들이 전체 분산 중 얼마를 보존했는가?
2. 원래 공간으로 되돌렸을 때 평균적으로 얼마나 틀리는가?
3. 그 손실이 실제 과업의 정확도·지연·메모리에 어떤 영향을 주는가?

PCA에서는 첫 질문을 **설명분산비**, 두 번째 질문을 **재구성 오차**로 측정할 수 있다. 두 양은 서로 독립된 이야기가 아니라 같은 고윳값을 보존한 쪽과 버린 쪽에서 바라본 것이다.

### 1.5 토큰 번호를 그대로 신경망에 넣으면 왜 안 될까?

어휘가 다음과 같이 번호로 저장되어 있다고 하자.

$$
\text{고양이}\mapsto 17,\qquad
\text{강아지}\mapsto 18,\qquad
\text{우주선}\mapsto 900
$$

이 번호는 데이터베이스의 사물함 번호와 같다. $18$이 $17$보다 의미상 “하나 더 크다”거나, 우주선이 고양이보다 약 $53$배 큰 의미를 가진다는 뜻이 아니다. 정수 ID를 실수 하나로 신경망에 넣으면 모델은 존재하지 않는 순서와 거리를 보게 된다.

원-핫 표현은 이 문제를 피한다. 토큰 $i$만 $1$이고 나머지는 $0$인 $V$차원 벡터를 사용하면 임의의 크기 순서는 사라진다. 그러나 어휘가 수십만 개라면 대부분이 $0$인 거대한 벡터는 비효율적이고, 서로 다른 원-핫 벡터의 내적은 언제나 $0$이라 의미의 가까움을 표현하지 못한다.

### 1.6 임베딩은 무엇을 해결하려고 태어났을까?

임베딩은 각 이산 항목을 비교적 작은 실수 벡터로 바꾼다. 비슷한 문맥에서 비슷한 역할을 하는 토큰이 학습 목표를 잘 풀려면, 모델은 이들을 이후 층이 활용하기 좋은 위치에 놓을 수 있다.

중요한 점은 사람이 “고양이는 이 좌표”라고 직접 지정하지 않는다는 것이다. 임베딩 표도 다른 가중치처럼 손실 함수의 그래디언트로 업데이트된다. 의미적 구조는 사전의 정의를 그대로 옮긴 결과라기보다, 데이터·토큰화·학습 목표·모델 구조가 함께 만든 **과업 의존적 기하학**이다.

### 1.7 오늘 두 트랙이 만나는 지점과 갈라지는 지점

PCA와 임베딩은 둘 다 고차원 또는 이산 대상을 더 다루기 쉬운 벡터 표현으로 바꾼다.

$$
\text{복잡한 대상}
\longrightarrow
\text{벡터 좌표}
\longrightarrow
\text{거리·내적·학습}
$$

그러나 좌표를 고르는 규칙은 다르다.

- PCA는 관측 데이터의 분산을 최대한 보존하는 **직교 선형 축**을 닫힌 형태의 선형대수로 찾는다.
- 신경망 임베딩은 예측 손실을 줄이도록 역전파로 학습되는 **대체로 비직교인 파라미터 표**다.
- PCA의 축은 고윳값 순서로 중요도를 정할 수 있지만, 일반 임베딩의 개별 좌표에는 자동으로 “첫 번째가 가장 중요하다”는 순서가 없다.
- PCA는 임베딩을 시각화하거나 압축하는 후처리 도구가 될 수 있지만, PCA 자체가 단어의 의미를 만들어 주지는 않는다.

## 2. 친절한 용어 사전

### 2.1 PCA의 수학 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 표본 수 | $N$ | 관측한 데이터 행의 개수다. |
| 원래 특징 수 | $d$ | 각 표본을 설명하는 원래 숫자의 개수다. |
| 남길 차원 | $k$ | PCA 뒤에 보존할 주성분의 수다. 보통 $1\le k<d$다. |
| 데이터 행렬 | $\mathbf{X}\in\mathbb{R}^{N\times d}$ | 표본을 행으로, 특징을 열로 쌓은 행렬이다. |
| 특징 평균 | $\boldsymbol{\mu}\in\mathbb{R}^{d}$ | 각 열의 평균을 모은 벡터다. |
| 중심화 행렬 | $\mathbf{X}_c$ | 각 행에서 특징 평균을 뺀 행렬이다. 각 열의 평균은 $0$이다. |
| 공분산 행렬 | $\boldsymbol{\Sigma}$ | 특징들이 함께 변하는 정도와 모든 방향의 분산을 담은 $d\times d$ 대칭행렬이다. |
| 주성분 방향 | $\mathbf{q}_i$ | 공분산 행렬의 단위 고유벡터다. 고윳값이 큰 순서로 번호를 붙인다. |
| 주성분 고윳값 | $\lambda_i$ | $\mathbf{q}_i$ 방향으로 투영한 데이터의 분산이다. |
| 주성분 행렬 | $\mathbf{Q}_k$ | 앞의 $k$개 주성분 방향을 열로 모은 $d\times k$ 행렬이다. |
| 점수·좌표 행렬 | $\mathbf{Z}$ | 각 표본을 주성분 축에서 나타낸 $N\times k$ 행렬이다. 문헌에서는 score라고도 한다. |
| 로딩 | loading | 원래 특징이 주성분 축에 얼마나 기여하는지 나타내는 고유벡터 성분이다. 분야와 소프트웨어에 따라 스케일을 포함한 정의가 다를 수 있다. |
| 투영 행렬 | $\mathbf{P}_k=\mathbf{Q}_k\mathbf{Q}_k^{\top}$ | 원래 벡터를 선택한 $k$차원 부분공간 위로 정사영하는 $d\times d$ 행렬이다. |
| 재구성 | $\widehat{\mathbf{X}}$ | 저차원 좌표를 원래 특징 공간의 근삿값으로 되돌린 결과다. |
| 재구성 오차 | $\lVert\mathbf{X}-\widehat{\mathbf{X}}\rVert_F^2$ | 원본과 복원본의 모든 원소 차이를 제곱해 더한 값이다. |
| 프로베니우스 노름 | $\lVert\mathbf{A}\rVert_F$ | 행렬의 모든 원소 제곱합의 제곱근이다. 행렬 전체의 유클리드 길이와 같다. |
| 설명분산비 | $\operatorname{EVR}_i$ | 전체 분산 중 $i$번째 주성분이 차지하는 비율이다. |
| 누적 설명분산비 | $\operatorname{CEVR}(k)$ | 앞의 $k$개 주성분이 함께 보존하는 전체 분산의 비율이다. |
| 표준화 |  | 평균을 뺀 뒤 각 특징을 표준편차로 나누어 단위를 맞추는 과정이다. 중심화와 같은 말이 아니다. |
| 특잇값분해 | SVD | 직사각행렬을 직교 방향과 축별 크기로 분해하는 방법이다. PCA를 공분산 행렬을 직접 만들지 않고 계산할 수 있게 한다. |
| 랭크 | $r$ | 행렬에서 서로 독립인 방향의 수다. 중심화된 $N\times d$ 행렬의 랭크는 최대 $\min(N-1,d)$다. |

### 2.2 임베딩의 AI 언어

| 용어 | 표기 | 초보자 해설 |
|---|---|---|
| 어휘집 | vocabulary | 모델이 구분하는 토큰 종류의 전체 목록이다. |
| 어휘 크기 | $V$ | 서로 다른 토큰 종류의 개수다. |
| 토큰 | token | 모델이 한 단위로 처리하는 텍스트 조각이다. 단어 전체, 서브워드, 문자, 특수 기호일 수 있다. |
| 토큰 ID | $i$ | 토큰을 저장·조회하기 위한 정수 주소다. 숫자 크기에 의미는 없다. |
| 원-핫 벡터 | $\mathbf{o}_i$ | $i$번째 원소만 $1$이고 나머지는 $0$인 $V$차원 벡터다. |
| 임베딩 | $\mathbf{e}_i$ | 이산 항목을 나타내는 학습 가능한 저차원 실수 벡터다. |
| 임베딩 차원 | $d_e$ | 한 토큰 벡터에 들어가는 실수 좌표의 개수다. |
| 임베딩 표 | $\mathbf{E}\in\mathbb{R}^{V\times d_e}$ | 모든 토큰의 임베딩을 행으로 저장한 학습 파라미터다. |
| 룩업 | lookup | 토큰 ID로 임베딩 표의 해당 행을 꺼내는 연산이다. |
| 희소 표현 | sparse representation | 원-핫처럼 대부분의 원소가 $0$인 표현이다. |
| 밀집 표현 | dense representation | 임베딩처럼 비교적 적은 차원의 많은 좌표가 정보를 나누어 담는 표현이다. |
| 잠재 공간 | latent space | 모델이 학습한 벡터 좌표 공간이다. 각 축의 의미가 사람이 붙인 이름으로 직접 드러나지 않을 수 있다. |
| 코사인 유사도 | $\cos\theta$ | 두 벡터의 길이를 제거하고 방향이 얼마나 비슷한지 재는 값이다. |
| 정적 임베딩 |  | 같은 토큰 ID에 문맥과 무관하게 같은 기본 벡터를 주는 표현이다. |
| 문맥 임베딩 |  | Transformer의 여러 층을 거쳐 주변 토큰에 따라 달라진 표현이다. |
| 위치 임베딩 |  | 토큰의 순서나 상대 위치 정보를 표현에 더하거나 결합하는 벡터다. |
| 멀티모달 임베딩 |  | 텍스트·이미지·오디오 같은 서로 다른 입력을 비교하거나 함께 처리할 수 있게 만든 벡터 표현이다. |
| 배치 | $B$ | 한 번의 계산에 함께 처리하는 표본 수다. |
| 시퀀스 길이 | $T$ | 한 표본에 들어 있는 토큰 위치 수다. |

### 2.3 오늘의 트렌드 언어

| 용어 | 영문·표기 | 초보자 해설 |
|---|---|---|
| 벡터 검색 | vector search | 질의와 데이터의 임베딩을 비교해 가까운 항목을 찾는 검색 방식이다. |
| 밀집 임베딩 | dense embedding | 고정 길이 벡터의 대부분 좌표가 값을 갖는 표현이다. 의미적 유사도 검색에 자주 쓴다. |
| 희소 임베딩 | sparse embedding | 큰 어휘 좌표 중 일부에만 값이 있는 표현이다. 특정 토큰 신호를 보존하기 쉽다. |
| 정확 최근접 이웃 | exact nearest neighbor | 후보를 근사적으로 줄이지 않고 정의된 거리에서 정확한 가까운 이웃을 찾는 계산이다. |
| 근사 최근접 이웃 | ANN | 속도와 메모리를 얻는 대신 일부 이웃을 놓칠 수 있는 근사 검색이다. |
| 정답 집합 | ground truth | 근사 검색의 재현율을 평가하기 위한 정확한 상위 결과 목록이다. |
| 재현율 | recall | 정확한 정답 이웃 중 검색 시스템이 실제로 찾아낸 비율이다. |
| 체화 에이전트 | embodied agent | 이미지·센서로 환경을 보고 실제 또는 시뮬레이션 공간에서 행동하는 에이전트다. |
| 어포던스 | affordance | 현재 물체와 상태에서 어떤 행동이 가능한지를 뜻한다. |
| 제약 디코딩 | constrained decoding | 생성 도중 허용되는 다음 토큰만 남겨 형식·규칙을 강제하는 방법이다. |
| 로짓 | logit | Softmax로 확률을 만들기 전 각 다음 토큰 후보의 점수다. |
| 하드 마스크 | hard mask | 불가능한 후보의 로짓을 $-\infty$로 보내 선택 확률을 정확히 $0$으로 만드는 제약이다. |
| DFA | deterministic finite automaton | 현재 상태와 다음 기호로 다음 상태가 하나로 정해지는 유한 상태 기계다. 허용된 문법을 검사할 수 있다. |
| HMM | hidden Markov model | 직접 보이지 않는 상태와 관측·전이 확률을 이용하는 확률 모델이다. |
| 동결 모델 | frozen model | 해당 실험에서 가중치를 업데이트하지 않고 그대로 사용하는 모델이다. |
| 테스트 시점 적응 | test-time adaptation | 배포·평가 입력을 처리하는 시점에 가벼운 구성요소를 새 환경에 맞추는 방법이다. |

## 3. 수학의 해부학 (증명과 원리)

### 3.1 표기와 shape를 먼저 고정하자

$n$번째 표본을 열벡터

$$
\mathbf{x}_n\in\mathbb{R}^{d}
$$

라고 하고, 데이터 행렬에는 그 전치를 행으로 쌓는다.

$$
\mathbf{X}
=
\begin{bmatrix}
\mathbf{x}_1^{\top}\\
\vdots\\
\mathbf{x}_N^{\top}
\end{bmatrix}
\in\mathbb{R}^{N\times d}
$$

특징 평균은

$$
\boldsymbol{\mu}
=\frac{1}{N}\sum_{n=1}^{N}\mathbf{x}_n
\in\mathbb{R}^{d}
$$

이다. $\mathbf{1}\in\mathbb{R}^{N}$을 모든 원소가 $1$인 열벡터라 하면 평균 행을 $N$번 반복한 행렬은

$$
\mathbf{1}\boldsymbol{\mu}^{\top}
\in\mathbb{R}^{N\times d}
$$

이고, 중심화 행렬은

$$
\boxed{
\mathbf{X}_c
=\mathbf{X}-\mathbf{1}\boldsymbol{\mu}^{\top}
}
$$

이다. 실제 구현에서 훈련 데이터의 평균을 저장해야 새 데이터에도 **같은 평균**을 뺄 수 있다.

오늘은 모공분산 형태의 분모 $N$을 일관되게 사용한다.

$$
\boxed{
\boldsymbol{\Sigma}
=\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c
}
$$

표본공분산 관례의 분모 $N-1$을 사용해도 고유벡터는 같고 모든 고윳값의 공통 스케일만 바뀐다. 단, 설명분산비는 공통 스케일이 약분되지만 재구성 오차와 고윳값 합의 등식에서는 어느 분모를 썼는지 맞춰야 한다.

### 3.2 공분산 행렬의 고유쌍을 큰 순서로 정렬한다

공분산 행렬은 실수 대칭이고 양의 준정부호이므로 정규직교 고유기저를 갖고 고윳값은 음수가 아니다.

$$
\boldsymbol{\Sigma}
=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top}
$$

여기서

$$
\mathbf{Q}
=
\begin{bmatrix}
\mathbf{q}_1&\cdots&\mathbf{q}_d
\end{bmatrix},
\qquad
\mathbf{Q}^{\top}\mathbf{Q}
=\mathbf{I}_d
$$

이고

$$
\boldsymbol{\Lambda}
=\operatorname{diag}(\lambda_1,\ldots,\lambda_d),
\qquad
\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d\ge0
$$

로 정렬한다. $\operatorname{diag}(a_1,\ldots,a_m)$은 주대각선에 $a_1,\ldots,a_m$을 놓고 나머지를 $0$으로 만든 대각행렬이다. 앞의 $k$개 방향만 모으면

$$
\mathbf{Q}_k
=
\begin{bmatrix}
\mathbf{q}_1&\cdots&\mathbf{q}_k
\end{bmatrix}
\in\mathbb{R}^{d\times k}
$$

이며

$$
\mathbf{Q}_k^{\top}\mathbf{Q}_k
=\mathbf{I}_k
$$

다. $\mathbf{I}_m$은 $m\times m$ 단위행렬이다. 따라서 $\mathbf{Q}_k^{\top}\mathbf{Q}_k=\mathbf{I}_k$는 같은 주성분 축의 내적은 $1$, 서로 다른 축의 내적은 $0$이라는 뜻이다. 뒤에서 단위행렬의 아래첨자를 생략할 때는 행렬곱에 맞는 크기를 뜻한다.

### 3.3 투영: $d$개의 숫자를 $k$개의 주성분 좌표로 바꾼다

새 표본 $\mathbf{x}$의 중심화 벡터를

$$
\mathbf{x}_c
=\mathbf{x}-\boldsymbol{\mu}
$$

라 하자. 각 주성분 방향과 내적하면 저차원 좌표

$$
\boxed{
\mathbf{z}
=\mathbf{Q}_k^{\top}\mathbf{x}_c
\in\mathbb{R}^{k}
}
$$

를 얻는다. $i$번째 좌표는

$$
z_i
=\mathbf{q}_i^{\top}\mathbf{x}_c
$$

이므로 이전에 배운 정사영 계수 그 자체다.

표본 전체를 행으로 계산하면

$$
\boxed{
\mathbf{Z}
=\mathbf{X}_c\mathbf{Q}_k
\in\mathbb{R}^{N\times k}
}
$$

다. 행 표기와 열 표기에서 곱셈 순서가 달라 보이지만 같은 연산이다.

### 3.4 재구성: 저차원 좌표를 원래 공간의 근삿값으로 되돌린다

저차원 좌표 $\mathbf{z}$를 원래 중심화 공간으로 되돌리면

$$
\widehat{\mathbf{x}}_c
=\mathbf{Q}_k\mathbf{z}
$$

이다. $\mathbf{z}=\mathbf{Q}_k^{\top}\mathbf{x}_c$를 대입하면

$$
\widehat{\mathbf{x}}_c
=\mathbf{Q}_k\mathbf{Q}_k^{\top}\mathbf{x}_c
$$

이고 평균을 다시 더하면

$$
\boxed{
\widehat{\mathbf{x}}
=\boldsymbol{\mu}
+\mathbf{Q}_k\mathbf{Q}_k^{\top}
(\mathbf{x}-\boldsymbol{\mu})
}
$$

이다.

행렬 전체에서는

$$
\boxed{
\widehat{\mathbf{X}}
=\mathbf{X}_c\mathbf{Q}_k\mathbf{Q}_k^{\top}
+\mathbf{1}\boldsymbol{\mu}^{\top}
}
$$

다. $k=d$이고 모든 고유벡터를 사용하면 $\mathbf{Q}\mathbf{Q}^{\top}=\mathbf{I}$이므로 수치 오차를 제외하고 원본을 정확히 복원한다. $k<d$이면 버린 방향의 성분만큼 오차가 남는다.

### 3.5 $\mathbf{P}_k=\mathbf{Q}_k\mathbf{Q}_k^{\top}$가 정사영 행렬임을 증명하자

투영 행렬을

$$
\mathbf{P}_k
=\mathbf{Q}_k\mathbf{Q}_k^{\top}
$$

라고 하자. 먼저 대칭성은

$$
\mathbf{P}_k^{\top}
=
(\mathbf{Q}_k\mathbf{Q}_k^{\top})^{\top}
=\mathbf{Q}_k\mathbf{Q}_k^{\top}
=\mathbf{P}_k
$$

이다.

한 번 투영한 벡터를 다시 투영해도 바뀌지 않는 멱등성은

$$
\begin{aligned}
\mathbf{P}_k^2
&=
\mathbf{Q}_k\mathbf{Q}_k^{\top}
\mathbf{Q}_k\mathbf{Q}_k^{\top}\\
&=
\mathbf{Q}_k
(\mathbf{Q}_k^{\top}\mathbf{Q}_k)
\mathbf{Q}_k^{\top}\\
&=
\mathbf{Q}_k\mathbf{I}_k\mathbf{Q}_k^{\top}\\
&=\mathbf{P}_k
\end{aligned}
$$

로 확인된다.

잔차를

$$
\mathbf{r}
=\mathbf{x}_c-\widehat{\mathbf{x}}_c
=(\mathbf{I}-\mathbf{P}_k)\mathbf{x}_c
$$

라고 하면 선택한 축들과의 내적은

$$
\begin{aligned}
\mathbf{Q}_k^{\top}\mathbf{r}
&=
\mathbf{Q}_k^{\top}
(\mathbf{I}-\mathbf{Q}_k\mathbf{Q}_k^{\top})
\mathbf{x}_c\\
&=
(\mathbf{Q}_k^{\top}
-\mathbf{I}_k\mathbf{Q}_k^{\top})
\mathbf{x}_c\\
&=\mathbf{0}
\end{aligned}
$$

이다. 즉, 복원값은 선택한 주성분 부분공간 안에 있고 잔차는 그 공간에 수직이다. 이것이 단순한 좌표 삭제가 아니라 **가장 가까운 점으로의 정사영**인 이유다.

### 3.6 주성분 좌표의 공분산은 왜 대각행렬일까?

저차원 좌표 행렬은 $\mathbf{Z}=\mathbf{X}_c\mathbf{Q}_k$이고 각 열의 평균도 $0$이다. 따라서 그 공분산은

$$
\begin{aligned}
\boldsymbol{\Sigma}_{\mathbf{Z}}
&=
\frac{1}{N}\mathbf{Z}^{\top}\mathbf{Z}\\
&=
\frac{1}{N}
\mathbf{Q}_k^{\top}
\mathbf{X}_c^{\top}\mathbf{X}_c
\mathbf{Q}_k\\
&=
\mathbf{Q}_k^{\top}
\boldsymbol{\Sigma}
\mathbf{Q}_k\\
&=
\operatorname{diag}
(\lambda_1,\ldots,\lambda_k)
\end{aligned}
$$

이다.

대각 원소는 각 주성분의 분산이고 비대각 원소는 서로 다른 주성분 사이의 공분산이다. 비대각 원소가 $0$이므로 PCA 좌표들은 선형 상관이 없다. 그러나 **공분산이 $0$이라고 통계적으로 독립인 것은 아니다**. 공동분포가 다변량 가우시안인 경우처럼 추가 조건이 있을 때만 무상관에서 독립을 결론낼 수 있다.

### 3.7 설명분산비는 무엇을 설명한다는 뜻일까?

중심화된 데이터의 전체 특징 분산은 공분산 행렬의 대각합이고 고윳값 합과 같다.

$$
\operatorname{tr}(\boldsymbol{\Sigma})
=\sum_{j=1}^{d}\operatorname{Var}(X_j)
=\sum_{i=1}^{d}\lambda_i
$$

$i$번째 주성분의 설명분산비는

$$
\boxed{
\operatorname{EVR}_i
=
\frac{\lambda_i}
{\sum_{j=1}^{d}\lambda_j}
}
$$

이고 앞의 $k$개 누적 설명분산비는

$$
\boxed{
\operatorname{CEVR}(k)
=
\frac{\sum_{i=1}^{k}\lambda_i}
{\sum_{j=1}^{d}\lambda_j}
}
$$

이다.

예를 들어 고윳값이 $6,3,1$이면 전체 분산은 $10$이고 첫 주성분은 $60\%$, 앞의 두 주성분은 $90\%$를 보존한다. 여기서 “설명”은 정답의 원인이나 인과를 설명한다는 뜻이 아니다. **입력 데이터의 총 분산 중 해당 축이 차지하는 비율**이라는 제한된 뜻이다.

모든 중심화 표본이 영벡터라면 $\boldsymbol{\Sigma}=\mathbf{0}$이고 전체 분산도 $0$이다. 이때 설명분산비는 $0/0$이라 정의되지 않으며, 데이터에 변동이 없으므로 주성분 방향도 하나로 정해지지 않는다.

### 3.8 최대 분산 보존과 최소 재구성 오차는 같은 문제다

임의의 정규직교 열을 가진

$$
\mathbf{W}\in\mathbb{R}^{d\times k},
\qquad
\mathbf{W}^{\top}\mathbf{W}
=\mathbf{I}_k
$$

를 생각하자. 이 부분공간에 투영했을 때 보존되는 총분산은

$$
\operatorname{tr}
(\mathbf{W}^{\top}\boldsymbol{\Sigma}\mathbf{W})
$$

이다.

전체 정규직교 고유기저 $\mathbf{q}_1,\ldots,\mathbf{q}_d$에서

$$
a_i
=\lVert\mathbf{q}_i^{\top}\mathbf{W}\rVert_2^2
$$

라고 두자. $\mathbf{P}_{\mathbf{W}}=\mathbf{W}\mathbf{W}^{\top}$는 $\mathbf{W}$의 열공간으로 향하는 정사영이므로

$$
\begin{aligned}
a_i
&=
\mathbf{q}_i^{\top}
\mathbf{W}\mathbf{W}^{\top}
\mathbf{q}_i\\
&=
\lVert
\mathbf{P}_{\mathbf{W}}\mathbf{q}_i
\rVert_2^2\\
&\le
\lVert\mathbf{q}_i\rVert_2^2
=1
\end{aligned}
$$

이다. 완전한 고유기저는 $\sum_{i=1}^{d}\mathbf{q}_i\mathbf{q}_i^{\top}=\mathbf{I}_d$를 만족하므로

$$
\begin{aligned}
\sum_{i=1}^{d}a_i
&=
\operatorname{tr}
\left[
\mathbf{W}^{\top}
\left(
\sum_{i=1}^{d}
\mathbf{q}_i\mathbf{q}_i^{\top}
\right)
\mathbf{W}
\right]\\
&=
\operatorname{tr}
(\mathbf{W}^{\top}\mathbf{W})\\
&=
\operatorname{tr}(\mathbf{I}_k)
=k
\end{aligned}
$$

이다. 따라서

$$
0\le a_i\le1,
\qquad
\sum_{i=1}^{d}a_i=k
$$

이고 보존 분산은

$$
\operatorname{tr}
(\mathbf{W}^{\top}\boldsymbol{\Sigma}\mathbf{W})
=\sum_{i=1}^{d}\lambda_i a_i
$$

가 된다. $0\le a_i\le1$, $\sum_i a_i=k$만 남긴 더 넓은 가중치 문제에서도 큰 $\lambda_i$부터 $k$개에 $1$을 배정할 때 합이 최대다. 이는 원래 부분공간 문제의 상한이고, 실제로 $\mathbf{W}=\mathbf{Q}_k$가 이 가중치를 만들어 상한을 달성한다. 따라서

$$
\operatorname{tr}
(\mathbf{W}^{\top}\boldsymbol{\Sigma}\mathbf{W})
\le
\sum_{i=1}^{k}\lambda_i
$$

이다. 등호는 $\mathbf{W}$의 열공간이 앞의 $k$개 고유방향이 만드는 부분공간일 때 성립한다. 경계의 고윳값이 중복되면 그 고유공간 안에서 기저 자체는 유일하지 않을 수 있다.

이제 앞에서 정의한 $\mathbf{P}_{\mathbf{W}}$로 재구성 오차를 보자.

$$
\begin{aligned}
\frac{1}{N}
\lVert
\mathbf{X}_c-\mathbf{X}_c\mathbf{P}_{\mathbf{W}}
\rVert_F^2
&=
\operatorname{tr}
\left[
(\mathbf{I}-\mathbf{P}_{\mathbf{W}})
\boldsymbol{\Sigma}
(\mathbf{I}-\mathbf{P}_{\mathbf{W}})
\right]\\
&=
\operatorname{tr}
\left[
\boldsymbol{\Sigma}
(\mathbf{I}-\mathbf{P}_{\mathbf{W}})^2
\right]\\
&=
\operatorname{tr}
\left[
\boldsymbol{\Sigma}
(\mathbf{I}-\mathbf{P}_{\mathbf{W}})
\right]\\
&=
\operatorname{tr}(\boldsymbol{\Sigma})
-
\operatorname{tr}
(\mathbf{W}^{\top}\boldsymbol{\Sigma}\mathbf{W})
\end{aligned}
$$

이다. 첫 항은 어떤 $\mathbf{W}$를 골라도 일정하다. 따라서 보존 분산을 최대화하는 것은 재구성 오차를 최소화하는 것과 정확히 같다. 최적의 PCA 부분공간에서는

$$
\boxed{
\frac{1}{N}
\lVert
\mathbf{X}_c-\mathbf{X}_c\mathbf{Q}_k\mathbf{Q}_k^{\top}
\rVert_F^2
=
\sum_{i=k+1}^{d}\lambda_i
}
$$

가 된다.

표본공분산 분모 $N-1$을 사용했다면 왼쪽도 $1/(N-1)$로 나눌 때 같은 등식이 성립한다.

### 3.9 $2\times2$ 예제로 투영과 손실을 직접 계산하자

지난 학습의 공분산 행렬을 다시 사용하자.

$$
\boldsymbol{\Sigma}
=
\begin{bmatrix}
2&1\\
1&2
\end{bmatrix}
$$

고유쌍은

$$
\lambda_1=3,
\qquad
\mathbf{q}_1
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

과

$$
\lambda_2=1,
\qquad
\mathbf{q}_2
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
-1
\end{bmatrix}
$$

이다. $k=1$이면

$$
\operatorname{CEVR}(1)
=\frac{3}{3+1}
=0.75
$$

이므로 총분산의 $75\%$를 보존한다.

평균이 이미 $0$이고 새 점이

$$
\mathbf{x}_c
=
\begin{bmatrix}
2\\
0
\end{bmatrix}
$$

라고 하자. 저차원 좌표는

$$
z
=\mathbf{q}_1^{\top}\mathbf{x}_c
=\frac{1}{\sqrt{2}}(2+0)
=\sqrt{2}
$$

이고 재구성은

$$
\widehat{\mathbf{x}}_c
=\mathbf{q}_1z
=
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1\\
1
\end{bmatrix}
\sqrt{2}
=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

이다. 잔차와 제곱오차는

$$
\mathbf{r}
=
\begin{bmatrix}
2\\
0
\end{bmatrix}
-
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
1\\
-1
\end{bmatrix}
$$

$$
\lVert\mathbf{r}\rVert_2^2
=1^2+(-1)^2
=2
$$

이다. 이 한 점의 오차 $2$와 데이터 전체의 평균 재구성 오차 $\lambda_2=1$은 같은 값일 필요가 없다. 후자는 모든 표본에 대해 평균낸 값이다.

### 3.10 표준화가 필요한 경우와 위험한 경우

센서 A가 미터 단위로 $0$에서 $2$ 사이이고 센서 B가 밀리미터 단위로 $0$에서 $2000$ 사이라 하자. 물리적으로 같은 크기의 변화라도 숫자 분산은 B가 훨씬 크게 보인다. PCA는 큰 분산을 우선하므로 단위 선택이 주성분을 지배할 수 있다.

각 특징의 표준편차 $s_j$가 양수일 때

$$
x_{nj}^{\mathrm{std}}
=
\frac{x_{nj}-\mu_j}{s_j}
$$

로 표준화하면 각 특징의 분산 규모를 맞출 수 있다. 이는 상관행렬 기반 PCA와 연결된다.

하지만 표준화가 언제나 정답은 아니다.

- 단위와 절대 변동 크기 자체가 중요한 물리 문제에서는 큰 분산을 유지해야 할 수 있다.
- 거의 상수인 특징은 $s_j$가 매우 작아 잡음을 크게 증폭할 수 있다.
- 훈련·검증·테스트를 나누기 전에 전체 데이터 평균과 표준편차를 계산하면 데이터 누수가 생긴다.

전처리 통계는 훈련 세트에서만 추정하고 이후 데이터에 그대로 적용해야 한다.

### 3.11 실제 계산에서는 SVD를 왜 자주 사용할까?

중심화 행렬의 랭크가 $r$일 때 compact SVD, 즉 랭크-$r$ 얇은 특잇값분해를

$$
\mathbf{X}_c
=\mathbf{U}_r\mathbf{S}_r\mathbf{V}_r^{\top}
$$

라 하자. 각 행렬의 shape는

$$
\mathbf{U}_r\in\mathbb{R}^{N\times r},
\qquad
\mathbf{S}_r\in\mathbb{R}^{r\times r},
\qquad
\mathbf{V}_r\in\mathbb{R}^{d\times r}
$$

이고 두 방향 행렬의 열은 정규직교하므로

$$
\mathbf{U}_r^{\top}\mathbf{U}_r
=
\mathbf{V}_r^{\top}\mathbf{V}_r
=\mathbf{I}_r
$$

이다. 여기서 $r=\operatorname{rank}(\mathbf{X}_c)$이고 $\mathbf{S}_r$의 대각 원소를

$$
s_1\ge s_2\ge\cdots\ge s_r>0
$$

라 하자. 그러면

$$
\begin{aligned}
\boldsymbol{\Sigma}
&=
\frac{1}{N}\mathbf{X}_c^{\top}\mathbf{X}_c\\
&=
\frac{1}{N}
\mathbf{V}_r\mathbf{S}_r^2\mathbf{V}_r^{\top}
\end{aligned}
$$

이므로

$$
\boxed{
\mathbf{q}_i=\mathbf{v}_i,
\qquad
\lambda_i=\frac{s_i^2}{N}
}
$$

이다. 이 관계는 $i=1,\ldots,r$에 대해 성립한다. $r<d$이면 나머지 $d-r$개 고윳값은 $0$이고, 그 영공간 안의 고유기저는 하나로 정해지지 않을 수 있다. $1\le k\le r$일 때 PCA 좌표도

$$
\mathbf{Z}_k
=\mathbf{X}_c\mathbf{V}_k
=\mathbf{U}_k\mathbf{S}_k
$$

로 바로 얻는다.

$d$가 매우 크거나 수치 안정성이 중요할 때 공분산 행렬을 명시적으로 만든 뒤 고유분해하기보다 $\mathbf{X}_c$에 SVD를 적용하는 구현이 흔하다. 대규모 데이터에서는 전체 SVD 대신 randomized SVD나 incremental PCA 같은 근사·스트리밍 방법을 쓸 수 있으며, 그때는 근사 오차와 난수 시드를 기록해야 한다.

### 3.12 PCA 알고리즘을 한 장으로 정리하자

1. 훈련 데이터의 결측값·이상값·단위를 점검한다.
2. 훈련 데이터 평균 $\boldsymbol{\mu}$를 계산하고 중심화한다.
3. 필요하다면 훈련 데이터 표준편차로 표준화한다.
4. 공분산 고유분해 또는 중심화 행렬의 SVD를 계산한다.
5. 고윳값을 큰 순서로 정렬하고 설명분산비를 확인한다.
6. 비용과 과업 성능을 함께 보며 $k$를 정한다.
7. $\mathbf{Q}_k$로 훈련·검증·테스트와 새 입력을 변환한다.
8. 필요하면 재구성 오차와 원래 과업 지표를 함께 측정한다.
9. 평균, 스케일, $\mathbf{Q}_k$, 라이브러리 버전을 모델과 함께 저장한다.

설명분산비가 높아도 정답 예측에 중요한 저분산 신호를 버릴 수 있다. PCA의 목표는 입력 분산 보존이지 라벨 예측 성능 보존이 아니므로, 최종 $k$는 반드시 실제 과업 지표로도 검증해야 한다.

## 4. 🤖 인공지능 기초 빌드업 (Core AI Fundamentals)

### 4.1 범주를 숫자로 저장하는 것과 숫자의 크기를 학습하는 것은 다르다

컴퓨터는 문자열 토큰을 곧바로 행렬곱에 넣을 수 없으므로 각 토큰에 정수 ID를 붙인다. 그러나 ID는 주소일 뿐 수치형 특징이 아니다.

예를 들어

$$
\text{서울}\mapsto2,\qquad
\text{부산}\mapsto3,\qquad
\text{제주}\mapsto100
$$

이라는 사전이 있더라도 다음 관계는 성립하지 않는다.

$$
\operatorname{meaning}(\text{제주})
=
50\operatorname{meaning}(\text{서울})
$$

ID 하나를 실수 입력으로 쓰면 선형층은 $wi+b$처럼 ID의 대소와 간격에 반응한다. 사전 순서를 무작위로 바꾸기만 해도 입력의 기하학이 완전히 달라진다. 의미가 같은 데이터 표현이 주소 배정에 따라 달라지는 잘못된 귀납적 편향이다.

### 4.2 원-핫 벡터는 거짓 순서를 없애지만 의미적 거리를 만들지 못한다

어휘 크기가 $V=5$이고 토큰 ID가 $i=3$이라 하자. 열벡터 원-핫 표현은

$$
\mathbf{o}_3
=
\begin{bmatrix}
0\\
0\\
1\\
0\\
0
\end{bmatrix}
\in\mathbb{R}^{5}
$$

이다. 서로 다른 두 토큰 $i\ne j$에 대해

$$
\mathbf{o}_i^{\top}\mathbf{o}_j=0
$$

이고 각 길이는 $1$이므로 모든 다른 토큰 쌍의 유클리드 거리는

$$
\lVert\mathbf{o}_i-\mathbf{o}_j\rVert_2
=\sqrt{2}
$$

다. 고양이–강아지와 고양이–미분방정식이 원-핫 공간에서는 똑같이 멀다.

원-핫은 범주 사이에 근거 없는 순서를 부여하지 않는다는 장점이 있지만, 큰 $V$에서 메모리가 낭비되고 학습 가능한 유사도 구조가 없다.

### 4.3 임베딩 룩업은 원-핫 벡터와 행렬을 곱한 것과 같다

임베딩 표를

$$
\mathbf{E}
=
\begin{bmatrix}
\mathbf{e}_1^{\top}\\
\mathbf{e}_2^{\top}\\
\vdots\\
\mathbf{e}_V^{\top}
\end{bmatrix}
\in\mathbb{R}^{V\times d_e}
$$

라고 하자. 토큰 $i$의 열벡터 임베딩은

$$
\boxed{
\mathbf{e}_i
=\mathbf{E}^{\top}\mathbf{o}_i
=\mathbf{E}_{i,:}^{\top}
}
$$

이다. $\mathbf{o}_i$에서 $1$인 위치만 남기므로 행렬곱 결과가 정확히 $\mathbf{E}$의 $i$번째 행이다.

실제 라이브러리는 거대한 원-핫 벡터를 만들고 곱하지 않는다. ID를 배열 인덱스로 사용해 필요한 행만 꺼낸다. 수학적으로는 같은 연산이지만 계산과 메모리는 훨씬 효율적이다.

### 4.4 배치와 시퀀스에서는 shape가 어떻게 변할까?

토큰 ID 텐서가

$$
\mathbf{T}_{\mathrm{id}}
\in\{1,\ldots,V\}^{B\times T}
$$

라고 하자. 여기서는 수식을 읽기 쉽게 토큰 ID를 $1$부터 $V$까지 붙였다. 실제 프로그래밍 라이브러리는 보통 $0$부터 $V-1$까지 인덱싱하므로 데이터 사전과 라이브러리 규약을 맞춰야 한다. $B$는 배치 크기이고 $T$는 시퀀스 길이다. 임베딩 룩업 뒤의 출력은

$$
\boxed{
\mathbf{H}
\in\mathbb{R}^{B\times T\times d_e}
}
$$

가 된다.

예를 들어 $B=2$, $T=4$, $d_e=3$이면 토큰 ID 숫자 $2\times4$개 각각이 길이 $3$인 벡터로 바뀌어 출력 shape는 $2\times4\times3$이다.

원-핫 행을 모두 평평하게 쌓은

$$
\mathbf{O}\in\mathbb{R}^{(BT)\times V}
$$

를 상상하면

$$
\mathbf{H}_{\mathrm{flat}}
=\mathbf{O}\mathbf{E}
\in\mathbb{R}^{(BT)\times d_e}
$$

이고 이를 다시 $B\times T\times d_e$로 reshape한 것과 같다. 평평하게 센 $m$번째 위치의 토큰 ID가 $i_m$이면 $\mathbf{H}_{\mathrm{flat}}$의 $m$번째 행은 $\mathbf{e}_{i_m}^{\top}$이다.

### 4.5 임베딩 표는 역전파로 어떻게 학습될까?

최종 손실 $\mathcal{L}\in\mathbb{R}$은 스칼라라고 하자. 평평한 임베딩 출력의 상류 그래디언트를

$$
\mathbf{G}
=\frac{\partial\mathcal{L}}
{\partial\mathbf{H}_{\mathrm{flat}}}
\in\mathbb{R}^{(BT)\times d_e}
$$

라고 하자. $\mathbf{G}_{m,:}$는 $\mathbf{H}_{\mathrm{flat}}$의 $m$번째 행과 같은 shape의 행벡터다. $\mathbf{H}_{\mathrm{flat}}=\mathbf{O}\mathbf{E}$이므로

$$
\boxed{
\frac{\partial\mathcal{L}}{\partial\mathbf{E}}
=\mathbf{O}^{\top}\mathbf{G}
}
$$

이다.

원-핫의 성질 때문에 현재 미니배치에 등장한 토큰의 행에만 룩업 경로의 그래디언트가 모인다. 같은 토큰이 여러 번 등장하면 그 위치들의 그래디언트가 같은 행에 더해진다.

토큰 ID $i$가 위치 집합 $\mathcal{S}_i$에 나타났다면

$$
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{i,:}}
=
\sum_{m\in\mathcal{S}_i}
\mathbf{G}_{m,:}
$$

이다. 학습률이 $\eta$인 기본 경사하강법에서는

$$
\mathbf{E}_{i,:}
\leftarrow
\mathbf{E}_{i,:}
-
\eta
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{i,:}}
$$

로 업데이트된다.

이 설명은 $\mathbf{E}$가 입력 룩업에만 사용되고 별도 정규화 항이 없다는 단순한 경우다. 출력 분류 가중치와 임베딩 표를 공유하는 **weight tying**을 쓰거나 전체 표에 정규화를 적용하면 미니배치에 직접 등장하지 않은 행에도 다른 경로의 그래디언트가 생길 수 있다.

### 4.6 “가까운 벡터는 비슷한 의미”가 자동으로 보장될까?

$\mathbf{a}\ne\mathbf{0}$, $\mathbf{b}\ne\mathbf{0}$인 두 임베딩의 내적은

$$
\mathbf{a}^{\top}\mathbf{b}
=
\lVert\mathbf{a}\rVert_2
\lVert\mathbf{b}\rVert_2
\cos\theta
$$

이고 코사인 유사도는

$$
\boxed{
\operatorname{cos}(\mathbf{a},\mathbf{b})
=
\frac{\mathbf{a}^{\top}\mathbf{b}}
{\lVert\mathbf{a}\rVert_2
\lVert\mathbf{b}\rVert_2}
}
$$

이다. 단위 정규화된 벡터에서는 내적과 코사인 유사도가 같다. 둘 중 하나가 영벡터면 분모가 $0$이므로 코사인 유사도는 정의되지 않는다. PCA 투영 뒤 영벡터가 생길 수 있으므로 검색 구현은 이 경우의 처리 규칙도 정해야 한다.

하지만 임베딩이 가까워지는 이유는 학습 목표에 달려 있다.

- 다음 토큰 예측 모델은 비슷한 문맥에서 비슷한 예측 역할을 하는 표현을 가까이 둘 수 있다.
- 검색 모델은 관련 질의–문서 쌍의 점수를 높이고 무관한 쌍의 점수를 낮추도록 학습한다.
- 추천 모델은 비슷한 사용자 반응을 만드는 항목을 가까이 둘 수 있다.
- 이미지–텍스트 대조학습은 짝이 맞는 이미지와 문장의 유사도를 높이도록 학습할 수 있다.

따라서 “가깝다”는 것은 인간의 모든 의미에서 동일하다는 선언이 아니다. 훈련 데이터의 편향, 빈도, 언어, 손실 함수가 정한 특정 관계에서 가깝다는 뜻이다. 동일한 철자의 토큰도 문맥에 따라 뜻이 달라질 수 있으므로 정적 룩업 벡터만으로 다의성을 완전히 해결할 수 없다.

### 4.7 정적 토큰 임베딩과 문맥 임베딩을 구분하자

Transformer 입력의 첫 단계에서 토큰 ID $i_t$는 기본 임베딩

$$
\mathbf{e}_{i_t}
=\mathbf{E}^{\top}\mathbf{o}_{i_t}
$$

를 얻는다. 여기에 위치 정보 등을 결합해 첫 층에 넣는다. 이 시점의 같은 ID는 기본적으로 같은 행을 조회한다.

하지만 여러 self-attention 층과 MLP를 지난 위치 $t$의 은닉표현

$$
\mathbf{h}_t^{(\ell)}
$$

은 주변 토큰에 따라 달라진다. 예를 들어 “배를 먹다”와 “배를 타다”처럼 같은 표면 토큰이 다른 문맥에 있으면 뒤쪽 층의 표현이 달라질 수 있다. 이를 문맥 임베딩 또는 문맥화된 표현이라고 부른다.

임베딩이라는 말은 입력 표의 행, 문장 전체 벡터, 중간층 토큰 표현, 검색용 출력 벡터를 모두 가리킬 수 있다. 실무 문서에서는 **어느 모델의 어느 층에서 어떤 풀링과 정규화를 거친 벡터인지**를 명시해야 한다.

### 4.8 위치와 멀티모달 정보도 같은 벡터 공간으로 들어온다

토큰 내용만 있으면 “개가 사람을 물었다”와 “사람이 개를 물었다”의 토큰 집합이 같아 순서를 구분하기 어렵다. 그래서 모델은 절대 위치 임베딩, 상대 위치 편향, 회전 위치 표현 등으로 순서 정보를 넣는다. 구체적인 방식은 아키텍처마다 다르다.

개념적으로 단순한 덧셈 방식은

$$
\mathbf{h}_t^{(0)}
=
\mathbf{e}_{i_t}
+
\mathbf{p}_t
$$

처럼 토큰 임베딩 $\mathbf{e}_{i_t}$와 위치 표현 $\mathbf{p}_t$를 같은 차원에서 결합한다.

VLM에서는 이미지 패치나 비전 인코더 출력

$$
\mathbf{v}_1,\ldots,\mathbf{v}_M
$$

을 언어 모델이 처리할 수 있는 차원으로 투영한다. 이후 텍스트 토큰과 함께 attention에 넣거나 cross-attention으로 연결한다. 이때 “같은 차원”이라는 것과 “완벽히 의미가 정렬됐다”는 것은 다르다. 정렬은 학습 데이터와 목적함수로 만들어지고 별도 평가가 필요하다.

### 4.9 임베딩과 PCA의 1:1 연결 및 결정적 차이

| 관점 | PCA | 신경망 임베딩 |
|---|---|---|
| 입력 | 연속형 데이터 행렬 | 토큰 ID, 항목 ID, 이미지·문장 등 |
| 학습 목표 | 입력 분산 최대 보존, 재구성 오차 최소화 | 다음 토큰, 분류, 검색, 추천 등 과업 손실 최소화 |
| 변환 | 평균 중심화 뒤 직교 선형 투영 | 룩업, 신경망 인코더, 비선형 변환 가능 |
| 축의 성질 | 주성분이 서로 직교하며 고윳값 순서가 있다 | 일반적으로 직교하지 않고 좌표별 중요도 순서가 없다 |
| 해의 식별성 | 고윳값이 서로 다르면 축은 부호를 제외하고 정해진다 | 회전·스케일 등 여러 동등한 표현이 가능할 수 있다 |
| 새 데이터 | 저장한 평균·축으로 고정 변환 | 학습한 표 또는 인코더로 변환 |
| 손실 측정 | 설명분산비와 재구성 오차를 직접 계산 | 다운스트림 과업 지표, 검색 recall, 정확도 등으로 검증 |

둘의 연결은 다음과 같이 사용할 수 있다.

$$
\underbrace{\mathbf{E}}_{V\times d_e}
\xrightarrow{\text{행 평균 중심화}}
\underbrace{\mathbf{E}_c}_{V\times d_e}
\xrightarrow{\text{PCA}}
\underbrace{\mathbf{Z}_E}_{V\times k}
$$

이 도식은 임베딩 표의 각 어휘 행을 같은 가중치로 본다. 배포용 압축에서는 실제 질의·문서 분포를 대표하는 피팅 표본에서 평균과 축을 추정하거나, 토큰·문서 빈도에 따른 가중치를 명시해야 한다.

- $k=2$ 또는 $3$으로 줄여 임베딩의 거친 군집을 시각화할 수 있다.
- 더 큰 $k$로 줄여 저장량과 검색 계산을 낮추는 후보를 만들 수 있다.
- 고윳값 스펙트럼으로 임베딩 분산이 몇 방향에 집중됐는지 살펴볼 수 있다.

그러나 PCA에서 분산이 작다고 검색·분류에 쓸모없는 방향이라고 단정할 수 없다. 라벨이나 희귀 개념을 구분하는 신호가 저분산 방향에 있을 수 있다. 압축 전후에 실제 검색 recall, 순위, 편향, 희귀 언어 성능을 다시 측정해야 한다.

### 4.10 임베딩을 PCA로 압축할 때 저장량은 어떻게 바뀔까?

$M$개의 벡터를 float32로 저장하고 원래 차원이 $d_e$, 압축 차원이 $k$라 하자. 벡터 본체의 대략적인 저장량은

$$
4Md_e\ \text{bytes}
$$

에서

$$
4Mk\ \text{bytes}
$$

로 줄어든다. 압축비는

$$
\frac{k}{d_e}
$$

다. 예를 들어 $d_e=768$, $k=192$이면 벡터 본체는 원래의 $1/4$이다.

하지만 실제 시스템 저장량에는 ID, 메타데이터, 인덱스 그래프, 정렬 패딩, 양자화 코드북이 추가된다. 또한 PCA 행렬 $\mathbf{Q}_k$와 평균도 저장해야 한다. “차원을 $1/4$로 줄였으니 전체 데이터베이스도 정확히 $1/4$”이라고 말하면 안 된다.

질의와 문서에 동일한 중심화·투영을 적용해야 한다.

$$
\mathbf{z}_{q}
=\mathbf{Q}_k^{\top}
(\mathbf{e}_{q}-\boldsymbol{\mu}),
\qquad
\mathbf{z}_{d}
=\mathbf{Q}_k^{\top}
(\mathbf{e}_{d}-\boldsymbol{\mu})
$$

코사인 검색을 쓴다면 투영 후 벡터를 다시 단위 정규화할지 여부도 학습·평가 파이프라인과 맞춰야 한다. 중심화와 투영은 원래 단위 길이를 보존하지 않는다.

### 4.11 실전 임베딩 파이프라인의 최소 구조도

~~~text
원시 문장
  -> 토크나이저
  -> 토큰 ID
  -> 임베딩 룩업
  -> 위치 정보 결합
  -> Transformer 문맥화
  -> 토큰 풀링 또는 전용 출력
  -> 검색용 벡터
  -> 선택적 PCA·양자화
  -> 벡터 인덱스
  -> top-k 후보
  -> 재순위화·최종 응답
~~~

각 화살표는 버전과 계약이 필요하다. 토크나이저나 평균 벡터가 바뀌면 같은 문장도 다른 검색 좌표가 된다. 문서 인덱스는 옛 모델인데 질의만 새 모델을 쓰는 식의 버전 불일치는 유사도 공간 자체를 깨뜨릴 수 있다.

### 4.12 수학 부품과 AI 동작의 1:1 연결

| 수학 개념 | AI에서 맡는 역할 |
|---|---|
| 원-핫 기저벡터 $\mathbf{o}_i$ | 토큰 ID를 거짓 순서 없이 특정 임베딩 행 선택으로 바꾼다. |
| 행렬곱 $\mathbf{E}^{\top}\mathbf{o}_i$ | 임베딩 룩업이 선형대수적으로 행 선택과 같음을 보여 준다. |
| 내적 $\mathbf{a}^{\top}\mathbf{b}$ | 다음 토큰 점수, attention 점수, 벡터 검색 유사도의 기본 부품이 된다. |
| 노름과 코사인 | 벡터 길이 효과와 방향 유사도를 분리한다. |
| 그래디언트 $\partial\mathcal{L}/\partial\mathbf{E}$ | 과업 손실이 어떤 토큰 임베딩 행을 어느 방향으로 움직일지 정한다. |
| 평균 중심화 | PCA가 임베딩 구름의 위치가 아니라 변동 방향을 분석하게 한다. |
| 고유벡터 $\mathbf{q}_i$ | 임베딩 분산을 많이 담는 직교 후처리 축을 제공한다. |
| 고윳값 $\lambda_i$ | 각 PCA 축이 담는 임베딩 분산량을 나타낸다. |
| $\mathbf{Q}_k^{\top}\mathbf{e}_c$ | 임베딩을 $d_e$차원에서 $k$차원으로 압축한다. |
| 버린 고윳값 합 | 훈련 표본에 대한 평균 제곱 재구성 손실을 나타낸다. |

### 4.13 임베딩과 PCA를 실전에 적용하는 최소 절차

1. 과업을 먼저 정한다. 의미 검색, 분류, 추천은 좋은 임베딩의 기준이 다르다.
2. 모델·토크나이저·출력 층·풀링·단위 정규화 규칙을 고정한다.
3. 실제 배포 분포를 대표하는 문서와 질의 검증 세트를 만든다.
4. 원본 차원의 정확도, recall, 지연, 메모리를 기준선으로 기록한다.
5. PCA 평균과 축은 훈련 또는 별도 피팅 세트에서만 계산한다.
6. 여러 $k$에서 누적 설명분산비와 실제 과업 지표를 모두 측정한다.
7. 희귀 토큰, 다국어, 긴 문서, 도메인별 부분집합 성능도 확인한다.
8. 변환 버전과 인덱스 버전을 함께 배포하고 롤백 경로를 둔다.
9. 데이터 분포가 바뀌면 고윳값 스펙트럼과 검색 성능을 다시 점검한다.

### 4.14 초보자가 흔히 하는 오해와 주의할 점

#### 오해 1: “PCA는 원래 특징 중 중요한 열 $k$개를 고른다.”

PCA는 원래 특징의 선형결합으로 새 축을 만든다. 원래 열을 그대로 고르는 특징 선택과 다르다.

#### 오해 2: “중심화와 표준화는 같은 과정이다.”

중심화는 평균을 빼고, 표준화는 추가로 표준편차로 나눈다. 문제의 단위와 목적에 따라 표준화 여부를 결정한다.

#### 오해 3: “설명분산비 $95\%$면 정답 정보도 $95\%$ 남는다.”

설명분산비는 입력 분산의 비율이다. 저분산 방향에 예측에 중요한 신호가 있으면 과업 성능은 크게 떨어질 수 있다.

#### 오해 4: “PCA로 줄이면 새 좌표가 원래 변수처럼 직접 해석된다.”

각 주성분은 여러 원래 특징의 혼합이다. 로딩을 살펴볼 수 있지만 인과적 의미를 자동으로 주지는 않는다.

#### 오해 5: “고유벡터의 부호가 실행할 때마다 바뀌면 PCA가 틀렸다.”

$\mathbf{q}$와 $-\mathbf{q}$는 같은 축이다. 좌표 부호도 함께 바뀌므로 투영 부분공간과 재구성은 같다.

#### 오해 6: “서로 무상관인 주성분은 서로 독립이다.”

공분산 $0$은 선형 상관이 없다는 뜻이다. 독립은 더 강한 조건이며 일반 분포에서 자동으로 성립하지 않는다.

#### 오해 7: “토큰 ID가 가까우면 뜻도 가깝다.”

ID는 사전의 주소다. 거리와 대소에는 의미가 없다.

#### 오해 8: “임베딩 룩업은 원-핫보다 완전히 다른 수학이다.”

계산 구현은 행 조회지만 수학적으로는 원-핫 벡터와 임베딩 행렬의 곱과 같다.

#### 오해 9: “임베딩의 각 좌표에는 사람이 읽을 수 있는 고정 의미가 있다.”

정보는 여러 좌표에 분산되어 있고 회전된 동등 표현도 가능하다. 개별 축을 곧바로 ‘성별 축’, ‘감정 축’이라고 단정하면 안 된다.

#### 오해 10: “코사인 유사도가 높으면 두 대상은 모든 면에서 같다.”

선택한 모델과 학습 목표가 강조한 관계에서 방향이 비슷하다는 뜻이다. 편향과 과업 불일치를 별도로 평가해야 한다.

#### 오해 11: “PCA를 임베딩에 적용해 설명분산이 높으면 검색 품질도 보장된다.”

PCA는 라벨이나 검색 순위를 보지 않는다. 압축 후 실제 top-$k$ recall과 순위 지표를 다시 측정해야 한다.

#### 오해 12: “PCA 변환은 질의에만 적용해도 된다.”

질의와 문서는 같은 평균, 축, 정규화 규칙을 사용해야 같은 좌표계에서 비교할 수 있다.

#### 오해 13: “입력 임베딩이 문장의 최종 의미 벡터다.”

입력 임베딩은 토큰 ID의 기본 벡터다. 문맥화 층, 풀링, 학습 목표를 거친 출력 표현과 구분해야 한다.

#### 오해 14: “임베딩이 준비되면 attention도 이미 이해한 것이다.”

임베딩은 토큰을 벡터로 바꾸는 출발점이다. attention은 현재 문맥에서 어떤 다른 위치의 정보를 얼마나 섞을지 정하는 다음 단계다.

## 5. 💡 오늘의 AI 트렌드 & 오픈소스 (Must-Read)

> 조사 기준 시각: 2026-09-10 08:38 (Asia/Seoul). 이전 학습에서 다룬 GPT-6 Astra, K2 Horizon, NeoMME, BEAM 2, RoboRMBench는 반복하지 않았다. 오늘은 임베딩이 실제 검색 인프라로 확장되는 사례와 VLM의 확률적 생성을 실행 규칙으로 제한하는 최신 연구를 공식 발표·데이터 카드·논문·저장소에서 교차 확인했다. 아래 성능과 규모는 공개 주체 또는 논문 저자의 보고이며 독립 재현 결과와 구분해야 한다.

### 5.1 Qdrant-FineWeb-10B: 임베딩 하나가 아니라 검색 전체 생애주기를 공개하다

Qdrant가 2026-09-01 공개한 **Qdrant-FineWeb-10B**는 FineWeb의 문서

$$
10{,}074{,}324{,}060
$$

개를 임베딩한 대규모 벡터 검색 벤치마크다. 현재 데이터 카드가 표시하는 전체 파일 크기는 약 $46.5$ TB다. 릴리스 글의 구성요소별 저장량 표기와 현재 카드의 전체 크기가 정확히 일치하지 않으므로, 실제 사용자는 다운로드 전에 Hub의 최신 파일 목록과 저장 형식을 다시 확인해야 한다.

각 문서는 Alibaba-NLP의 gte-multilingual-base를 사용한 두 표현을 함께 가진다.

- 길이 $768$인 단위 정규화 밀집 임베딩
- 모델의 $250{,}048$개 토큰 어휘 좌표 중 일부만 값이 있는 비정규화 희소 임베딩

밀집 벡터 $\mathbf{d}$와 질의 $\mathbf{q}$가 모두 단위벡터라면

$$
\operatorname{cos}(\mathbf{q},\mathbf{d})
=
\frac{\mathbf{q}^{\top}\mathbf{d}}
{\lVert\mathbf{q}\rVert_2\lVert\mathbf{d}\rVert_2}
=
\mathbf{q}^{\top}\mathbf{d}
$$

이므로 코사인 유사도와 내적 순위가 같다. 오늘 배운 노름과 내적이 인터넷 규모 검색의 실제 점수 함수가 된 사례다.

현재 데이터 카드는 MS MARCO에서 온 질의 $119{,}953$개를 네 집합으로 구분한다.

| 질의 집합 | 개수 | 검색 종류 |
|---|---:|---|
| 일반 밀집 질의 | $100{,}000$ | 단위 밀집 벡터 검색 |
| 일반 희소 질의 | $10{,}000$ | 희소 내적 검색 |
| 텍스트 필터 밀집 질의 | $4{,}953$ | 키워드·도메인 필터 결합 |
| 구조 필터 밀집 질의 | $5{,}000$ | 날짜·언어 점수·크롤 집합 필터 결합 |

각 질의에는 전체 100억 문서를 대상으로 계산한 정확한 결과가 최대 top-$1000$까지 연결된다. 단, 텍스트 필터 질의 $4{,}953$개 중 $557$개는 필터를 만족하는 문서 자체가 $1000$개보다 적다. 필터 결과가 하나도 없던 $47$개 질의는 공개 집합에서 제외되었다.

근사 검색이 반환한 최대 $K$개 결과 집합을 $A_K$, 사용할 수 있는 정확 정답을 최대 $K$개까지 모은 집합을 $G_K$라 하자. $|G_K|>0$일 때 recall은

$$
\operatorname{Recall@}K
=
\frac{
\left|
A_K\cap G_K
\right|
}{|G_K|}
$$

로 정의할 수 있다. 정답이 정확히 $K$개인 일반적인 경우에는 분모가 $K$가 된다. Supernova는 짧은 정답 집합을 자체 길이로 나눈 지표를 별도로 다루며, 거리 동점 때문에 top-$K$ 경계가 모호할 때 recall의 하한과 상한을 함께 보고할 수 있다.

함께 공개된 **Supernova**는 Apache-2.0 라이선스의 오픈소스 도구다. 임베딩 생성, 정확한 최근접 이웃 계산, 데이터베이스 적재, 부하 시험을 모듈로 나눈다. QPS와 $p_{50}$·$p_{95}$·$p_{99}$ 지연, 인덱스 생성 시간, 정답 대비 recall을 함께 측정하도록 설계되었다.

이 공개가 오늘의 PCA와 만나는 지점은 명확하다. 100억 개의 $768$차원 float 벡터는 차원 하나의 비용도 100억 번 반복한다. PCA로 $k<768$로 줄이면 저장·전송·내적 계산을 낮출 후보가 되지만, 실제로 채택하려면 다음 곡선을 측정해야 한다.

$$
k
\longmapsto
\left(
\text{저장량},
\text{지연},
\operatorname{Recall@}k',
\text{희귀 질의 성능}
\right)
$$

여기서 왼쪽의 $k$는 PCA 차원이고 $\operatorname{Recall@}k'$의 $k'$는 검색 결과 개수이므로 서로 다른 기호 역할이다. 실험 문서에서는 혼동을 피하도록 각각 $d_{\mathrm{PCA}}$와 $K_{\mathrm{search}}$처럼 이름을 분리하는 편이 좋다.

**엔지니어 인사이트 (Impact)**

- 작은 장난감 데이터의 빠른 ANN 결과가 100억 벡터에서도 유지된다고 가정할 수 없다. 인덱스 생성, 장애 복구, 필터 선택도, tail latency까지 포함한 시스템 벤치마크가 필요하다.
- 정확한 top-$1000$은 압축·양자화·ANN의 속도 이득과 검색 손실을 재는 공통 기준을 제공한다. 평균 recall뿐 아니라 질의 유형별 recall을 분리해야 한다.
- 데이터 카드에 따르면 원래 정답은 bfloat16 GPU 계산, 재생성 스크립트는 float32 임베딩을 사용한다. 아주 가까운 동점과 top-$1000$ 경계에서는 순서나 포함 여부가 달라질 수 있다. “정확 검색”에도 수치 형식과 재현 절차가 필요하다.
- 데이터베이스 묶음은 ODC-BY-1.0이지만 원본 웹페이지는 각자의 조건을 유지한다. MS MARCO 질의는 별도의 비상업 연구 조건이며 질의 텍스트와 파생 임베딩이 데이터셋에 직접 재배포되지 않는다. 한 저장소의 라이선스 배지만 보고 전체 파이프라인의 상업 이용 가능성을 단정하면 안 된다.
- 전체 $46.5$ TB를 내려받는 것은 초보자 실습 범위를 넘는다. 먼저 작은 shard와 적은 질의로 변환 계약을 검증한 뒤 규모를 키워야 한다.

**공식 자료:** [Qdrant 발표 글](https://huggingface.co/blog/Qdrant/fineweb-10b-release) · [FineWeb-10B 데이터 카드](https://huggingface.co/datasets/Qdrant/FineWeb-10B) · [Supernova 저장소](https://github.com/qdrant-labs/supernova)

### 5.2 CLAMP: VLM이 “그럴듯한 행동” 대신 “허용된 행동”만 생성하게 만들기

2026-09-08 11:38:09 UTC, 즉 같은 날 20:38:09 KST에 arXiv v1이 제출된 **CLAMP: Constrained Decoding for Vision-Language Embodied Planning**은 VLM 기반 에이전트의 다음 문제에서 출발한다.

> 문법적으로 유창한 계획이 실제로 실행 가능한 계획이라는 보장은 없다.

VLM은 이미지에 보이지 않는 물체를 언급하거나, 현재 상태에서 불가능한 행동을 고르거나, 실행기가 요구하는 문법을 어길 수 있다. CLAMP는 기본 VLM의 가중치를 동결한 채 디코딩 바깥에 두 종류의 장치를 붙인다.

1. 초기 장면에서 관측된 물체, 행동 문법, 현재 상태의 실행 가능성을 DFA 기반 하드 제약으로 만든다.
2. HMM 기반 상태 추적과 lookahead로 남은 행동 예산 안에서 목표에 도달할 가능성이 높은 허용 후보를 재가중한다.

기본 VLM이 다음 토큰 $v$에 주는 로짓을 $\ell_t(v)$라 하고 현재 제약 상태에서 허용 여부를 나타내는 마스크를

$$
m_t(v)
=
\begin{cases}
0,&v\text{가 허용됨},\\
-\infty,&v\text{가 금지됨}
\end{cases}
$$

라고 하자. 하드 제약 뒤의 확률은 개념적으로

$$
\widetilde{p}_t(v)
=
\operatorname{Softmax}
\left(
\ell_t(v)+m_t(v)
\right)
$$

이다. 금지 후보는 $e^{-\infty}=0$이므로 확률이 정확히 $0$이 된다. HMM lookahead는 이때 남은 허용 후보 사이의 우선순위를 추가로 조정한다.

여기서 오늘의 임베딩과 중요한 대비가 생긴다.

$$
\underbrace{\text{임베딩·VLM 로짓}}_{\text{무엇이 그럴듯한가?}}
\quad+\quad
\underbrace{\text{기호 제약}}_{\text{무엇이 허용되는가?}}
\quad\longrightarrow\quad
\underbrace{\text{실행 계획 후보}}_{\text{둘을 모두 만족}}
$$

벡터 공간의 유사도와 학습된 확률은 부드러운 선호를 제공하지만, 로봇의 행동 문법이나 “보이지 않는 물체를 집지 말 것” 같은 규칙을 자동으로 보장하지 않는다. 반대로 기호 규칙만으로는 자연어와 이미지의 풍부한 모호성을 해석하기 어렵다. CLAMP는 두 계층을 분리해 결합한다.

논문 저자가 동일한 Qwen3-VL-8B 기반 설정에서 보고한 VLABench macro 점수는 다음과 같다.

| 구성 | VLABench macro |
|---|---:|
| 기본 VLM | $28.7$ |
| 관측 기반 DFA 하드 제약 | $34.1$ |
| 하드 제약 + text-conditioned HMM guidance | $37.1$ |
| supervised vision HMM + per-instance calibration | $38.7$ |

또한 SafeAgentBench의 긴 horizon $n=50$ 실험에서 저자가 보고한 per-constraint violation rate는 Policy 단독 $0.41$에서 **Policy + recovery** $0.05$로 줄었다. 여기서 recovery는 단순한 한 번의 CLAMP 디코딩이 아니라 rewind·retry와 검증된 강제 행동 삽입을 포함하므로, 이 감소 전체를 하드 마스크 하나의 효과로 해석하면 안 된다. 이 숫자는 논문 설정의 결과이며 실제 로봇 안전 인증이나 독립 재현을 뜻하지 않는다. 마지막 $38.7$ 구성은 benchmark ground-truth skill–image 쌍으로 감독한 vision HMM과 샘플별 calibration을 포함한다. 따라서 $37.1$에 단일 부품 하나만 공정하게 더한 supervision-matched 비교로 단순화하면 안 된다.

공개 저장소에는 제약 생성, DFA 컴파일, 선택적 HMM prior·테스트 시점 적응, Hugging Face·vLLM 디코딩, 소규모 프롬프트 부분집합, 설정과 테스트가 들어 있다. 그러나 저장소 설명은 다음을 명시적으로 제외한다.

- 모델 가중치
- 전체 벤치마크 데이터와 시뮬레이터 자산
- 원시 궤적과 논문 수치의 원시 실행 결과
- 비공개 실험 자료와 자격 증명

또한 이 GitHub 저장소의 CLAMP-specific 파일에는 포괄적인 재사용 라이선스를 적용하지 않았고, 포함된 외부 구성요소는 각각의 원래 조건을 따른다. 논문 원고의 공개 조건과 코드의 재배포 조건도 별개다. 코드를 읽을 수 있다는 사실과 자유롭게 재배포할 수 있다는 사실을 구분해야 한다.

**엔지니어 인사이트 (Impact)**

- 모델 크기를 키우거나 다시 학습하는 것만이 신뢰성을 높이는 방법은 아니다. 명시적 스키마와 상태 제약을 Softmax 전에 적용하면 불가능한 출력을 생성 공간에서 제거할 수 있다.
- 제약의 품질이 곧 상한이 된다. 장면 인식이 물체를 놓치거나 사람이 작성한 전이 규칙이 틀리면 올바른 행동까지 막을 수 있다. 규칙 버전, 관측 근거, 차단 사유를 로그로 남겨야 한다.
- 논문 설정의 VLM은 생성 동안 초기 관측을 고정해 사용하며 중간 실행 관측을 생성 피드백으로 쓰지 않는다. 실제 배포에서는 상태 추정, 저수준 motion planning, 실행 모니터링, 긴급 정지가 별도로 필요하다.
- 모든 제약 적용 점수가 $-\infty$가 되거나 남은 행동 예산으로 목표에 도달할 수 없으면 공개된 CLAMP-standard는 명시적으로 FAIL을 반환한다. 실제 제품은 그다음에 중단, 재관측, 사람 승인 중 무엇을 할지 별도 운영 정책으로 정해야 한다.
- 벤치마크 성공률과 규칙 위반률을 따로 봐야 한다. 안전 제약을 늘려 위반률은 낮췄지만 모든 작업을 거부해 성공률도 낮아지는 시스템은 유용하지 않을 수 있다.

**원문과 구현:** [arXiv 초록·제출 이력](https://arxiv.org/abs/2609.08602) · [CLAMP 공개 저장소](https://github.com/HLR/CLAMP)

### 5.3 두 소식이 보여 주는 공통 흐름: 좋은 벡터만으로 제품은 완성되지 않는다

Qdrant-FineWeb-10B는 임베딩 모델 다음에 정확 정답 계산, 인덱스 적재, 부하 시험, 라이선스와 수치 재현성이 필요함을 보여 준다. CLAMP는 VLM 표현 다음에 문법, 상태, 실행 가능성, 모니터링이 필요함을 보여 준다.

이를 한 문장으로 줄이면 다음과 같다.

$$
\boxed{
\text{학습된 표현}
+
\text{검증 가능한 외부 계약}
=
\text{운영 가능한 AI 시스템의 출발점}
}
$$

PCA의 설명분산비도 같은 교훈을 준다. 내부 표현을 잘 압축했다는 지표 하나만으로 검색 품질이나 행동 안전을 대신할 수 없다. 각 단계가 무엇을 보장하고 무엇을 보장하지 않는지 경계를 명시하고, 최종 과업의 실패 비용에 맞는 지표를 별도로 측정해야 한다.

## 6. 오늘의 메타인지 질문 (스스로 묻고 답하기)

### 질문

어휘 크기가 $V=4$, 임베딩 차원이 $d_e=2$인 임베딩 표가 다음과 같다고 하자.

$$
\mathbf{E}
=
\begin{bmatrix}
3&1\\
3&-1\\
-3&1\\
-3&-1
\end{bmatrix}
$$

각 행은 토큰 $1,2,3,4$의 임베딩이다. 네 토큰을 같은 빈도로 관측했다고 가정하고 공분산 분모는 $N=4$를 사용한다.

1. 행 평균 $\boldsymbol{\mu}$와 중심화된 임베딩 표 $\mathbf{E}_c$를 구하라.
2. 공분산 행렬 $\boldsymbol{\Sigma}_E=\frac{1}{4}\mathbf{E}_c^{\top}\mathbf{E}_c$의 고윳값과 단위 고유벡터를 구하라.
3. $k=1$ PCA 좌표와 네 임베딩의 재구성값을 구하라.
4. 누적 설명분산비와 표본당 평균 제곱 재구성 오차를 구하고, 버린 고윳값과 비교하라.
5. 토큰 $2$의 원-핫 벡터를 쓰고 $\mathbf{E}^{\top}\mathbf{o}_2$가 룩업과 같은지 확인하라.
6. 원래 공간에서 토큰 $1$과 $2$의 코사인 유사도, 1차원 PCA 뒤의 코사인 유사도를 구하라. 이 변화가 검색에 주는 경고는 무엇인가?
7. 한 미니배치의 토큰 ID가 $[2,2,4]$이고 세 위치의 임베딩 출력에 대한 상류 그래디언트가 각각 행벡터 $\mathbf{g}_1,\mathbf{g}_2,\mathbf{g}_3\in\mathbb{R}^{1\times2}$라면 임베딩 표의 어느 행이 어떻게 업데이트되는가?
8. CLAMP 사례를 이용해 “임베딩 유사도가 높으면 그 행동은 실행해도 된다”가 왜 성립하지 않는지 설명하라.

### 모범 답안

#### 1. 평균과 중심화

첫 좌표와 둘째 좌표의 합이 모두 $0$이므로

$$
\boldsymbol{\mu}
=
\frac{1}{4}
\begin{bmatrix}
3+3-3-3\\
1-1+1-1
\end{bmatrix}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

이다. 따라서

$$
\mathbf{E}_c=\mathbf{E}
$$

다.

#### 2. 공분산과 고유쌍

공분산은

$$
\begin{aligned}
\boldsymbol{\Sigma}_E
&=
\frac{1}{4}
\begin{bmatrix}
3&3&-3&-3\\
1&-1&1&-1
\end{bmatrix}
\begin{bmatrix}
3&1\\
3&-1\\
-3&1\\
-3&-1
\end{bmatrix}\\
&=
\frac{1}{4}
\begin{bmatrix}
36&0\\
0&4
\end{bmatrix}\\
&=
\begin{bmatrix}
9&0\\
0&1
\end{bmatrix}
\end{aligned}
$$

이다. 따라서 큰 순서의 고유쌍은

$$
\lambda_1=9,
\qquad
\mathbf{q}_1
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

과

$$
\lambda_2=1,
\qquad
\mathbf{q}_2
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

이다. 각 고유벡터의 부호를 반대로 골라도 같은 PCA 축이다.

#### 3. 1차원 좌표와 재구성

$\mathbf{Q}_1=\mathbf{q}_1$이므로

$$
\mathbf{Z}
=\mathbf{E}_c\mathbf{Q}_1
=
\begin{bmatrix}
3\\
3\\
-3\\
-3
\end{bmatrix}
$$

이다. 재구성은

$$
\widehat{\mathbf{E}}
=
\mathbf{Z}\mathbf{Q}_1^{\top}
+\mathbf{1}\boldsymbol{\mu}^{\top}
=
\begin{bmatrix}
3&0\\
3&0\\
-3&0\\
-3&0
\end{bmatrix}
$$

이다. 둘째 좌표의 $1$과 $-1$ 정보가 사라졌다.

#### 4. 설명분산비와 재구성 오차

누적 설명분산비는

$$
\operatorname{CEVR}(1)
=
\frac{\lambda_1}{\lambda_1+\lambda_2}
=
\frac{9}{10}
=0.9
$$

이다.

각 행의 재구성 잔차는 둘째 좌표만 $\pm1$이고 제곱오차는 모두 $1$이다. 따라서

$$
\frac{1}{4}
\lVert
\mathbf{E}_c
-\widehat{\mathbf{E}}_c
\rVert_F^2
=
\frac{1+1+1+1}{4}
=1
$$

이고 이는 버린 고윳값

$$
\lambda_2=1
$$

과 정확히 같다.

#### 5. 원-핫과 룩업의 동치

토큰 $2$의 원-핫 열벡터는

$$
\mathbf{o}_2
=
\begin{bmatrix}
0\\
1\\
0\\
0
\end{bmatrix}
$$

이다. 따라서

$$
\mathbf{E}^{\top}\mathbf{o}_2
=
\begin{bmatrix}
3&3&-3&-3\\
1&-1&1&-1
\end{bmatrix}
\begin{bmatrix}
0\\
1\\
0\\
0
\end{bmatrix}
=
\begin{bmatrix}
3\\
-1
\end{bmatrix}
$$

이고 이는 $\mathbf{E}$의 둘째 행을 전치한 것과 같다.

#### 6. 압축 전후 코사인 유사도

원래 토큰 $1$과 $2$의 임베딩을

$$
\mathbf{e}_1
=
\begin{bmatrix}
3\\
1
\end{bmatrix},
\qquad
\mathbf{e}_2
=
\begin{bmatrix}
3\\
-1
\end{bmatrix}
$$

라 하면

$$
\mathbf{e}_1^{\top}\mathbf{e}_2
=9-1
=8
$$

이고 두 벡터의 길이는 모두 $\sqrt{10}$이다. 따라서

$$
\operatorname{cos}(\mathbf{e}_1,\mathbf{e}_2)
=
\frac{8}{10}
=0.8
$$

이다.

1차원 PCA 좌표는 둘 다 $3$이다. 영벡터가 아니므로 1차원에서 단위 정규화하면 둘 다 $+1$이고 코사인 유사도는

$$
1
$$

이 된다. 설명분산의 $90\%$를 보존했지만 두 토큰을 구분하던 둘째 좌표가 사라져 둘이 완전히 같은 방향으로 보인다. 검색에서는 서로 다른 문서가 동점이 되거나 순위가 바뀔 수 있으므로 설명분산비만 보지 말고 압축 전후 recall을 측정해야 한다.

#### 7. 반복 토큰의 임베딩 그래디언트

ID $2$가 두 번, ID $4$가 한 번 등장했으므로 룩업 경로의 행별 그래디언트는

$$
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{2,:}}
=
\mathbf{g}_1+\mathbf{g}_2
$$

와

$$
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{4,:}}
=
\mathbf{g}_3
$$

이다. ID $1$과 $3$은 이 단순한 룩업 경로에서 등장하지 않았으므로

$$
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{1,:}}
=
\frac{\partial\mathcal{L}}
{\partial\mathbf{E}_{3,:}}
=\mathbf{0}
$$

이다. 기본 경사하강법은 해당 그래디언트의 반대 방향으로 둘째·넷째 행을 움직인다.

#### 8. 유사도와 실행 가능성의 차이

임베딩과 VLM 로짓은 학습 데이터에서 어떤 물체·행동이 문맥상 그럴듯한지를 점수화한다. 하지만 높은 점수는 다음을 보장하지 않는다.

- 물체가 현재 카메라에 실제로 보인다.
- 로봇 팔이 그 물체에 닿을 수 있다.
- 선행 행동이 완료되어 현재 상태에서 실행 가능하다.
- 출력 문자열이 실행기의 문법을 만족한다.
- 안전 규칙을 위반하지 않는다.

CLAMP의 DFA 하드 마스크는 이런 명시적 규칙에 어긋나는 후보의 로짓을 $-\infty$로 보내 확률을 $0$으로 만든다. HMM guidance는 남은 허용 후보 사이에서 목표 도달 가능성을 비교한다. 따라서 임베딩 기반 선호와 실행 가능성 제약은 서로 대체하는 것이 아니라 서로 다른 실패를 막는 두 계층이다.

---

**다음 연결 고리:** 임베딩으로 토큰을 벡터로 만들었고 내적이 유사도와 점수의 핵심 부품임을 확인했다. 다음에는 각 토큰이 문맥 속 다른 토큰을 얼마나 참고할지 정하는 **Query·Key·Value와 scaled dot-product attention**을 행렬 shape, Softmax, 마스킹까지 단계적으로 연결한다.
