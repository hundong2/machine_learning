# 브루스-파간(Breusch-Pagan) 검정: 오차의 등분산성 진단

브루스-파간 검정은 회귀 모델의 오차(잔차)들이 일정한 패턴 없이 골고루 흩어져 있는지(등분산성), 아니면 특정 패턴을 보이며 흩어져 있는지(이분산성)를 통계적으로 확인하는 방법입니다. 이 검정의 핵심은 **p-value**를 통해 "이분산성이 존재한다"는 의심이 통계적으로 유의미한지를 판단하는 것입니다.

## 예시 파일

아래 예시 파일은 (1) 등분산성을 만족하는 이상적인 데이터와 (2) 이분산성을 보이는 문제가 있는 데이터를 각각 만들어, 브루스-파간 검정 결과와 잔차 그래프가 어떻게 달라지는지 명확하게 보여줍니다.

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

# 그래프 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def run_bp_test(X, y, title):
    """주어진 데이터로 회귀분석 및 브루스-파간 검정을 실행하고 결과를 시각화합니다."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    # 예측값과 잔차 계산
    predicted_values = model.predict(X_const)
    residuals = model.resid
    
    # 브루스-파간 검정 실행
    # bp_test[0]은 LM 통계량, bp_test[1]은 p-value
    bp_test = het_breuschpagan(residuals, model.model.exog)
    p_value = bp_test[1]
    
    # 잔차 산점도 시각화
    plt.figure(figsize=(10, 5))
    plt.scatter(predicted_values, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{title}\nBreusch-Pagan p-value: {p_value:.4f}', fontsize=14)
    plt.xlabel('예측값 (Fitted values)')
    plt.ylabel('잔차 (Residuals)')
    plt.grid(True)
    plt.show()
    
    print(f"--- {title} ---")
    if p_value > 0.05:
        print(f"p-value({p_value:.4f}) > 0.05 이므로, 등분산성 가정을 만족한다고 볼 수 있습니다. (Good)")
    else:
        print(f"p-value({p_value:.4f}) <= 0.05 이므로, 이분산성이 존재할 가능성이 높습니다. (Bad)")
    print("-" * (len(title) + 6) + "\n")


# --- 시나리오 1: 등분산성 (Homoscedasticity) - 좋은 경우 ---
# X값에 관계없이 오차의 퍼짐 정도가 일정함
np.random.seed(0)
X1 = np.arange(1, 101)
# y = 5 + 2x + noise (오차의 표준편차가 30으로 일정)
y1 = 5 + 2 * X1 + np.random.randn(100) * 30
run_bp_test(X1, y1, "시나리오 1: 등분산성 (이상적인 경우)")


# --- 시나리오 2: 이분산성 (Heteroscedasticity) - 나쁜 경우 ---
# X값이 커질수록 오차의 퍼짐 정도도 함께 커짐 (깔때기 모양)
np.random.seed(0)
X2 = np.arange(1, 101)
# y = 5 + 2x + noise (오차의 표준편차가 X값에 비례하여 커짐)
y2 = 5 + 2 * X2 + np.random.randn(100) * (X2 * 0.5)
run_bp_test(X2, y2, "시나리오 2: 이분산성 (문제가 있는 경우)")

```

## 답변

### 등분산성 vs 이분산성: 왜 중요한가?

선형 회귀 분석이 신뢰성을 가지려면, 모델이 예측하고 남은 **오차(잔차)들이 예측값의 크기와 상관없이 일정하게 흩어져 있어야** 합니다. 이를 **등분산성(Homoscedasticity)**이라고 하며, 이는 모델이 모든 구간에서 일관된 예측력을 보인다는 의미입니다.

-   **쉬운 비유 (등분산성)**: 과녁의 중앙을 향해 샷건을 쐈을 때, 총알 자국들이 과녁 전체에 걸쳐 **균일한 너비로 퍼져있는 상태**입니다. 이는 사격이 안정적임을 의미합니다.
-   **쉬운 비유 (이분산성)**: 샷건을 쐈는데, 과녁의 왼쪽에서는 총알이 좁게 모여있고 오른쪽으로 갈수록 **깔때기처럼 넓게 퍼지는 상태**입니다. 이는 사격이 불안정하며, 특정 조건에서 예측 오차가 커짐을 의미합니다.

만약 **이분산성(Heteroscedasticity)**이 존재하면, 회귀 계수 자체는 괜찮을지 몰라도 그 계수가 얼마나 신뢰할 수 있는지를 나타내는 통계치(표준오차, p-value)가 왜곡됩니다. 결국 "이 변수는 통계적으로 유의미하다" 또는 "유의미하지 않다"는 판단을 잘못 내릴 위험이 커집니다.

### 브루스-파간 검정의 원리

브루스-파간 검정은 "과연 독립 변수(X)가 오차의 분산(퍼짐 정도)을 설명할 수 있는가?"라는 질문에 답하는 방식으로 이분산성을 탐지합니다.

1.  **1단계: 기본 회귀 분석 실행**
    먼저, 원래의 회귀 모델(`y = β₀ + β₁x`)을 학습시켜 각 데이터 포인트의 **잔차(residual)**를 구합니다.

2.  **2단계: 잔차를 이용한 보조 회귀 분석**
    이 검정의 핵심 아이디어입니다. 잔차의 분산(퍼짐 정도)을 대변하는 **잔차의 제곱(`residual²`)**을 새로운 종속 변수로 놓고, 원래의 독립 변수(X)를 사용하여 또 다른 회귀 분석(보조 회귀)을 실행합니다.
    `residual² = γ₀ + γ₁x`

3.  **3단계: 결과 해석**
    만약 보조 회귀 분석에서 독립 변수(X)가 잔차의 제곱(`residual²`)을 잘 설명한다면(즉, 보조 회귀 모델이 통계적으로 유의미하다면), 이는 **"X값에 따라 오차의 분산이 변한다"**는 뜻이 됩니다. 이것이 바로 이분산성의 증거입니다.

### p-value를 이용한 최종 판단

브루스-파간 검정은 위 3단계의 결과를 바탕으로 통계량과 **p-value**를 계산합니다. 우리는 이 p-value를 보고 가설을 검증합니다.

-   **귀무가설 (H₀)**: **등분산성이 존재한다.** (오차의 분산은 독립 변수와 관련이 없다. 우리가 바라는 좋은 상황)
-   **대립가설 (H₁)**: **이분산성이 존재한다.** (오차의 분산은 독립 변수와 관련이 있다. 문제가 있는 상황)

우리가 통계에서 일반적으로 사용하는 유의수준 0.05(5%)를 기준으로 다음과 같이 판단합니다.

-   **`p-value > 0.05`**: 귀무가설을 기각할 충분한 근거가 없다. 즉, **등분산성 가정을 만족한다고 볼 수 있다.** (잔차 그래프가 균일하게 퍼져있을 가능성이 높음)
-   **`p-value <= 0.05`**: 귀무가설을 기각한다. 즉, **이분산성이 존재한다고 볼 수 있다.** (잔차 그래프가 깔때기 모양 등 특정 패턴을 보일 가능성이 높음)

따라서 브루스-파간 검정에서는 **p-value가 0.05보다 크게 나오는 것이 좋은 결과**입니다.

### 추가 자료

-   [Wikipedia: Breusch–Pagan test (영문)](https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test)
-   [Statology: How to Perform a Breusch-Pagan Test in Python (영문)](https://www.statology.org/breusch-pagan-test-python/)