# 더빈-왓슨(Durbin-Watson) 통계량: 오차의 독립성 검증

더빈-왓슨 통계량은 회귀 분석의 결과로 나온 오차(잔차)들이 서로 독립적인지, 아니면 특정 패턴(자기상관)을 보이는지를 검사하는 지표입니다. 특히 시계열 데이터(시간 순서가 있는 데이터) 분석에서 오차들이 서로 영향을 주고받는지 확인하는 데 필수적이며, 통계량이 2에 가까울수록 오차들이 독립적이고 이상적인 상태임을 의미합니다.

## 예시 파일

아래 예시 파일은 (1) 자기상관이 없는 이상적인 경우, (2) 양의 자기상관이 있는 경우, (3) 음의 자기상관이 있는 경우를 각각 인위적으로 만들어, 더빈-왓슨 통계량이 어떻게 변하는지 보여줍니다.

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 그래프 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def run_regression_and_plot(X, y, title):
    """주어진 데이터로 회귀분석을 실행하고 잔차와 더빈-왓슨 통계량을 출력/시각화합니다."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    residuals = model.resid
    
    # 더빈-왓슨 통계량 계산
    dw_stat = sm.stats.durbin_watson(residuals)
    
    # 잔차 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(residuals, marker='o', linestyle='-')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{title}\nDurbin-Watson: {dw_stat:.2f}', fontsize=14)
    plt.xlabel('데이터 순서 (시간)')
    plt.ylabel('잔차 (오차)')
    plt.grid(True)
    plt.show()
    
    return dw_stat

# 1. 기본 데이터 생성
np.random.seed(42)
X = np.arange(50)
# 실제 관계: y = 2x + noise
y_base = 2 * X

# --- 시나리오 1: 자기상관이 없는 경우 (이상적) ---
# 오차들이 완전히 무작위적임
residuals_ideal = np.random.randn(50) * 10
y_ideal = y_base + residuals_ideal
run_regression_and_plot(X, y_ideal, "시나리오 1: 자기상관 없음 (이상적)")

# --- 시나리오 2: 양의 자기상관이 있는 경우 ---
# 이전 오차가 다음 오차에 긍정적인 영향을 줌 (오차가 비슷한 경향을 보임)
residuals_positive = np.zeros(50)
residuals_positive[0] = np.random.randn() * 10
for t in range(1, 50):
    # 이전 잔차의 80% + 새로운 랜덤 노이즈
    residuals_positive[t] = 0.8 * residuals_positive[t-1] + np.random.randn() * 5
y_positive = y_base + residuals_positive
run_regression_and_plot(X, y_positive, "시나리오 2: 양의 자기상관 (DW < 2)")

# --- 시나리오 3: 음의 자기상관이 있는 경우 ---
# 이전 오차가 다음 오차에 부정적인 영향을 줌 (오차가 부호를 바꾸는 경향을 보임)
residuals_negative = np.zeros(50)
residuals_negative[0] = np.random.randn() * 10
for t in range(1, 50):
    # 이전 잔차의 -80% + 새로운 랜덤 노이즈
    residuals_negative[t] = -0.8 * residuals_negative[t-1] + np.random.randn() * 5
y_negative = y_base + residuals_negative
run_regression_and_plot(X, y_negative, "시나리오 3: 음의 자기상관 (DW > 2)")

```

## 답변

### 오차의 독립성이란 무엇이며 왜 중요한가?

선형 회귀 분석이 신뢰성을 가지려면 몇 가지 기본 가정을 만족해야 합니다. 그중 **"오차의 독립성"**은 "하나의 오차(실제값 - 예측값)는 다른 오차와 아무런 관련이 없어야 한다"는 가정입니다.

-   **쉬운 비유**: 양궁 선수가 과녁을 쏠 때, 첫 번째 화살이 10점 오른쪽에 맞았다고 해서 두 번째 화살도 반드시 10점 오른쪽에 맞으리란 법은 없어야 합니다. 매번의 실수는 독립적이고 무작위여야 좋은 선수입니다. 만약 첫 번째 실수가 두 번째 실수에 계속 영향을 준다면(예: 바람의 방향을 잘못 계산), 그 선수의 점수는 신뢰하기 어렵습니다.
-   **중요한 이유**: 만약 오차들이 서로 관련이 있다면(자기상관이 있다면), 회귀 모델의 여러 통계치(예: 계수의 표준오차, p-value)가 왜곡되어 잘못된 결론을 내릴 수 있습니다. 예를 들어, "이 변수는 유의미하다"라고 잘못 판단할 수 있습니다.

### 더빈-왓슨 통계량의 수학적 원리

더빈-왓슨 통계량은 이웃한 오차들 간의 관계를 측정하여 오차의 독립성을 검사합니다. 그 공식은 다음과 같습니다.

`DW = Σ(eₜ - eₜ₋₁)² / Σ(eₜ)²`

-   `eₜ`: 현재 시점(t)의 오차(잔차)
-   `eₜ₋₁`: 바로 이전 시점(t-1)의 오차(잔차)

이 공식이 어떻게 0에서 4 사이의 값을 가지며 자기상관을 탐지하는지 직관적으로 이해해 봅시다.

1.  **분자 `Σ(eₜ - eₜ₋₁)²`**: **"이웃한 오차들 간의 차이"**의 제곱합입니다.
    -   **양의 자기상관**: 오차들이 비슷한 경향을 보이면(+, +, +, -, -, -), 이웃한 오차 `eₜ`와 `eₜ₋₁`의 값 차이가 매우 작아집니다. 따라서 분자 전체가 작아집니다.
    -   **음의 자기상관**: 오차들이 번갈아 나타나면(+, -, +, -, +), 이웃한 오차의 값 차이가 매우 커집니다. 따라서 분자 전체가 커집니다.
    -   **자기상관 없음**: 오차들이 무작위라면, 차이는 위 두 경우의 중간 정도가 됩니다.

2.  **분모 `Σ(eₜ)²`**: 전체 오차의 제곱합으로, 분자를 **정규화(normalize)**하는 역할을 합니다.

3.  **결과 해석**:
    -   **DW ≈ 2 (자기상관 없음)**: 분자가 분모의 약 2배일 때입니다. 이는 오차들이 서로 독립적이고 무작위적이라는 이상적인 상태를 의미합니다.
    -   **DW → 0 (양의 자기상관)**: 분자가 매우 작아져서 통계량이 0에 가까워집니다. 이는 이전 오차가 다음 오차와 매우 비슷하다는 강력한 신호입니다. (예: 주가가 한번 오르기 시작하면 계속 오르는 경향)
    -   **DW → 4 (음의 자기상관)**: 분자가 매우 커져서 통계량이 4에 가까워집니다. 이는 이전 오차가 다음 오차와 정반대의 값을 가지려는 경향이 강하다는 신호입니다. (예: 모델이 과도하게 보정하여 예측값이 실제값 주변을 계속 위아래로 교차하는 경우)

### 요약 및 판단 기준

-   **Durbin-Watson ≈ 2**: 오차의 독립성 가정을 만족한다고 볼 수 있습니다. (일반적으로 1.5 ~ 2.5 사이를 허용 범위로 보기도 합니다.)
-   **Durbin-Watson < 1.5**: **양의 자기상관**을 의심할 수 있습니다.
-   **Durbin-Watson > 2.5**: **음의 자기상관**을 의심할 수 있습니다.

더빈-왓슨 통계량은 회귀 모델의 신뢰도를 평가하는 중요한 지표이므로, `statsmodels`의 `summary()` 결과에서 항상 확인하는 것이 좋습니다.

### 추가 자료

-   [Wikipedia: Durbin–Watson statistic (영문)](https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic)
-   [Statology: A Guide to the Durbin-Watson Statistic (영문)](https://www.statology.org/durbin-watson-statistic/)