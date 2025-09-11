# `add_constant`의 역할과 수학적 원리

`add_constant`는 기울기(slope)가 아닌 **y절편(y-intercept)**을 추정하기 위해 사용됩니다. 이는 회귀식을 완전한 형태(`y = β₀ + β₁x`)로 만들어, 회귀선이 원점(0,0)을 지나야 한다는 제약 없이 데이터에 가장 적합한 선을 찾도록 도와줍니다.

## 예시 파일

아래 예시 코드는 `add_constant`를 사용했을 때와 사용하지 않았을 때의 회귀선이 어떻게 달라지는지를 명확하게 보여줍니다.

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. 데이터 생성 (실제 y절편이 50인 데이터)
np.random.seed(0)
X = np.arange(0, 10, 0.5)
# y = 50 + 3x + noise 형태의 데이터
y = 50 + 3 * X + np.random.randn(len(X)) * 8

# 2. add_constant를 사용한 경우 (y절편 O)
# X 데이터에 [1, 1, ..., 1] 형태의 상수항 열이 추가됨
X_with_const = sm.add_constant(X)
model_with_const = sm.OLS(y, X_with_const).fit()

# 3. add_constant를 사용하지 않은 경우 (y절편 X, 원점을 지나도록 강제)
model_without_const = sm.OLS(y, X).fit()

# 4. 결과 시각화
plt.figure(figsize=(12, 6))
plt.scatter(X, y, label='Original Data') # 원본 데이터

# add_constant 사용 O -> y절편을 자유롭게 찾음
plt.plot(X, model_with_const.predict(X_with_const), color='red', 
         label=f'With Constant (y-intercept={model_with_const.params[0]:.2f})')

# add_constant 사용 X -> y절편이 0으로 고정됨
plt.plot(X, model_without_const.predict(X), color='green', 
         label='Without Constant (forced through origin)')

plt.title('Effect of add_constant')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print("--- With Constant (add_constant 사용) ---")
print(model_with_const.summary())
print("\n--- Without Constant (add_constant 미사용) ---")
print(model_without_const.summary())
```

## 답변

사용자님의 질문은 매우 중요한 포인트입니다. 하지만 약간의 오해가 있습니다. `add_constant`는 기울기(`β₁`)가 아닌 **y절편(`β₀`)**을 추정할 수 있도록 모델을 도와주는 함수입니다.

### 수학적 원리: 왜 상수항 '1'이 필요한가?

1.  **선형 회귀의 기본 방정식**
    우리가 찾고 싶은 이상적인 선형 관계는 다음과 같은 방정식으로 표현됩니다.
    `y = β₀ + β₁x + ε`
    -   `y`: 종속 변수 (우리가 예측하려는 값)
    -   `x`: 독립 변수 (예측에 사용하는 값)
    -   `β₁`: **기울기(slope)**. `x`가 1단위 증가할 때 `y`가 얼마나 변하는지를 나타냅니다.
    -   `β₀`: **y절편(y-intercept)**. `x`가 0일 때의 `y`값, 즉 회귀선이 y축과 만나는 지점입니다.
    -   `ε`: 오차항. 실제 값과 예측 값의 차이를 의미합니다.

2.  **방정식을 행렬 형태로 변환**
    컴퓨터는 이 문제를 행렬 연산을 통해 풉니다. 위 방정식을 행렬 형태로 바꾸면 다음과 같습니다.
    `y = Xβ`
    -   `y`: 종속 변수 벡터
    -   `X`: 독립 변수 행렬
    -   `β`: 우리가 추정해야 할 계수(기울기와 절편) 벡터

    여기서 `β` 벡터는 `[β₀, β₁]` 두 개의 값을 가집니다. 이 두 값을 모두 추정하려면, `X` 행렬도 두 개의 열을 가져야 합니다. 하나는 `β₁`에 대응하는 `x` 데이터 열이고, 다른 하나는 `β₀`에 대응하는 열입니다.

3.  **y절편(`β₀`)을 위한 '가짜' 변수**
    원래 방정식 `y = β₀ + β₁x`를 다시 보면, `β₀`는 `β₀ * 1`과 같습니다. 즉, y절편 `β₀`는 **값이 항상 1인 '가짜' 변수에 곱해지는 계수**라고 생각할 수 있습니다.

    따라서 `statsmodels`와 같은 라이브러리가 `β₀`를 추정하게 하려면, 우리는 독립 변수 데이터 `X`에 **모든 값이 1인 열을 인위적으로 추가**해야 합니다.

    바로 이 역할을 하는 함수가 `sm.add_constant(X)` 입니다. 이 함수는 기존 `X` 데이터의 맨 앞에 모든 요소가 `1.0`인 열을 추가하여 모델이 y절편(`β₀`)을 계산할 수 있는 완전한 형태의 독립 변수 행렬을 만들어줍니다.

### `add_constant`를 사용하지 않으면 어떻게 되나요?

만약 `add_constant`를 사용하지 않으면, 모델은 y절편 `β₀`의 존재를 모르게 됩니다. 따라서 회귀 방정식은 `y = β₁x` 형태로 축소됩니다. 이 방정식은 `x`가 0일 때 `y`도 항상 0이므로, 회귀선이 **반드시 원점(0, 0)을 지나도록 강제**됩니다.

대부분의 실제 데이터에서 독립 변수가 0일 때 종속 변수도 0이 되는 경우는 드물기 때문에, `add_constant`를 사용하여 y절편을 모델이 자유롭게 찾도록 허용하는 것이 훨씬 더 정확하고 일반적인 분석 방법입니다.

### 추가 자료

-   [Statsmodels 문서: add_constant](https://www.statsmodels.org/stable/generated/statsmodels.tools.tools.add_constant.html)
-   [StackExchange: Why is a constant term needed in linear regression? (영문)](https://stats.stackexchange.com/questions/25954/why-is-a-constant-term-needed-in-linear-regression)