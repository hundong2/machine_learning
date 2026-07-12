# Code Function Guide

노트북을 처음 열었을 때 함수 이름이 낯설면 논문 아이디어보다 코드 문법이 먼저 발목을 잡습니다. 이 문서는 각 실습에서 반복해서 등장하는 Python, NumPy, Matplotlib 표현을 작게 풀어둔 안내서입니다.

## 배열 만들기와 확인

| 코드 | 뜻 | 초보자 체크 |
|---|---|---|
| `np.array([...])` | 리스트를 숫자 배열로 바꿉니다. | 이미지도 결국 숫자 격자입니다. |
| `np.zeros((h, w))` | 모든 값이 0인 `h x w` 배열을 만듭니다. | 빈 이미지, 빈 출력 feature map을 만들 때 씁니다. |
| `x.shape` | 배열의 크기를 확인합니다. | CNN/ViT 코드는 shape 추적이 절반입니다. |
| `reshape(-1)` | 배열을 한 줄 벡터로 펼칩니다. | ViT/MAE의 patch flattening과 연결됩니다. |

## 난수와 장난감 데이터

| 코드 | 뜻 | 왜 쓰는가 |
|---|---|---|
| `np.random.seed(숫자)` | 무작위 결과를 재현 가능하게 고정합니다. | 같은 그림과 숫자가 다시 나오게 합니다. |
| `np.random.normal(0, 1, shape)` | 평균 0, 표준편차 1인 노이즈를 만듭니다. | 손글씨 흔들림, diffusion 노이즈, 임베딩 잡음을 흉내 냅니다. |
| `np.random.permutation(n)` | 0부터 `n-1`까지 순서를 섞습니다. | MAE에서 어떤 패치를 가릴지 고를 때 씁니다. |

## 신경망 계산의 핵심 표현

| 코드 | 뜻 | 연결되는 논문 |
|---|---|---|
| `x @ weight` | 입력 벡터와 가중치 행렬의 곱입니다. | AlexNet의 선형층, ResNet 블록, ViT attention, CLIP similarity |
| `np.maximum(x, 0)` | ReLU입니다. | AlexNet의 학습 속도 개선 |
| `np.sum(patch * kernel)` | 필터와 이미지 조각을 곱해 합칩니다. | LeNet-5의 합성곱 |
| `np.max(region)` | 영역 안의 최댓값을 고릅니다. | LeNet-5의 pooling |
| `np.linalg.norm(x, axis=1)` | 벡터 길이를 계산합니다. | CLIP에서 cosine similarity를 만들기 전 정규화 |

## 그림 그리기

| 코드 | 뜻 | 읽는 법 |
|---|---|---|
| `plt.imshow(image, cmap="gray")` | 2차원 배열을 흑백 이미지로 보여줍니다. | 밝을수록 값이 큽니다. |
| `plt.imshow(matrix, cmap="viridis")` | 행렬 값을 색으로 보여줍니다. | attention이나 similarity 행렬에서 강한 연결을 찾습니다. |
| `plt.hist(values.ravel())` | 많은 숫자의 분포를 막대그래프로 봅니다. | ReLU와 tanh 활성값 분포를 비교합니다. |
| `plt.yscale("log")` | y축을 로그 스케일로 바꿉니다. | 기울기가 아주 빠르게 작아지는 현상을 보기 좋게 만듭니다. |

## 헷갈릴 때 보는 규칙

- 코드가 길어 보이면 먼저 `입력 shape -> 함수 -> 출력 shape`만 적어봅니다.
- `axis`가 헷갈리면 작은 2x3 배열을 직접 만들고 같은 함수를 적용해 봅니다.
- `@`가 나오면 "두 벡터/행렬이 서로 얼마나 섞이는가"라고 읽습니다.
- `softmax` 결과는 항상 합이 1인지 확인합니다.
- 그림이 이상하면 값 범위가 0에서 1인지, 또는 음수가 포함되는지 먼저 확인합니다.
