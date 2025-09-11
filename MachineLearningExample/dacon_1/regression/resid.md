# `model.resid`란 무엇인가? (잔차의 의미)

`model.resid`는 `statsmodels` 라이브러리로 회귀 분석을 실행한 후, 모델이 예측한 값과 실제 값 사이의 **오차(error)**, 즉 **잔차(residual)**들을 모아놓은 배열입니다. 이 잔차들을 분석하는 것은 모델이 데이터를 얼마나 잘 설명하는지, 그리고 회귀 분석의 기본 가정들을 만족하는지를 진단하는 데 매우 중요합니다.

## 예시 파일

아래 예시 파일은 간단한 회귀 모델을 만들고, `results.resid`가 실제로 "실제 값 - 예측 값"과 동일함을 보여줍니다. 또한 잔차가 무엇인지 시각적으로 표현합니다.

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 그래프 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 가상의 데이터 생성
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y = 10 + 2x + noise
y = 10 + 2 * X + np.random.randn(len(X)) * 2

# 2. 회귀 모델 학습
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const)
results = model.fit()

# 3. 예측값과 잔차(residual) 계산
predicted_values = results.predict(X_const)
residuals = results.resid  # model.fit()의 결과 객체에서 잔차를 바로 가져옴

# 4. 잔차가 (실제값 - 예측값)과 동일한지 확인
manual_residuals = y - predicted_values
print("--- results.resid와 수동 계산 결과 비교 ---")
print(f"results.resid: \n{np.round(residuals, 4)}")
print(f"수동 계산 (y - ŷ): \n{np.round(manual_residuals, 4)}")
print(f"두 결과가 동일한가? {np.allclose(residuals, manual_residuals)}")


# 5. 잔차 시각화
plt.figure(figsize=(10, 6))
# 회귀선과 실제 데이터 포인트
plt.scatter(X, y, color='blue', label='실제 값 (y)')
plt.plot(X, predicted_values, color='red', label='회귀선 (예측 값, ŷ)')

# 각 데이터 포인트와 회귀선 사이의 오차(잔차)를 선으로 표시
for i in range(len(X)):
    plt.vlines(X[i], ymin=predicted_values[i], ymax=y[i], color='green', linestyle='--', lw=1)

plt.title('회귀선과 잔차(Residuals)의 시각적 표현', fontsize=15)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

## 답변

### `results.resid`의 정의와 역할

`results.resid`는 `statsmodels` 회귀 분석 결과 객체(`results`)에 저장된 **잔차(residuals)** 배열입니다. 여기서 잔차란 모델이 데이터를 완벽하게 예측하지 못해서 발생하는 "예측 오차"를 의미합니다.

-   **수학적 정의**: `잔차(e) = 실제 관측값(y) - 모델의 예측값(ŷ)`

-   **쉬운 비유**: 내일 기온을 25℃로 예측했는데 실제 기온이 27℃였다면, 예측 오차(잔차)는 `27 - 25 = 2℃`가 됩니다. `results.resid`는 모든 데이터 포인트에 대한 이러한 예측 오차들을 모아놓은 것입니다.

### 잔차(Residuals)는 왜 중요한가?

회귀 모델은 단순히 선 하나를 긋는 것에서 끝나지 않습니다. 그 모델이 통계적으로 신뢰할 수 있는지 확인하는 과정이 반드시 필요하며, 이때 **잔차를 분석하는 것이 핵심적인 모델 진단 방법**입니다.

이상적인 회귀 모델의 잔차는 어떠한 패턴도 보이지 않는 **무작위적인 백색소음(white noise)**과 같아야 합니다. 만약 잔차에 특정 패턴이 보인다면, 이는 모델이 데이터의 특정 정보를 제대로 포착하지 못했다는 신호이며, 모델의 기본 가정이 깨졌을 수 있음을 의미합니다.

사용자님의 노트북 코드에서 `results.resid`는 다음과 같은 중요한 검증 작업에 사용되고 있습니다.

1.  **오차의 독립성 검증 (Durbin-Watson Test)**:
    -   `sm.stats.durbin_watson(results.resid)`
    -   잔차들이 서로 영향을 주는지(자기상관) 검사합니다. 잔차는 서로 독립적이어야 합니다.

2.  **오차의 정규성 검증 (Shapiro-Wilk, K-S Test)**:
    -   `stats.shapiro(results.resid)`
    -   잔차들이 평균이 0인 정규분포를 따르는지 검사합니다.

3.  **오차의 정규성 시각적 검증 (Q-Q Plot)**:
    -   `sm.qqplot(results.resid, line='s')`
    -   잔차의 분포가 정규분포와 얼마나 비슷한지를 시각적으로 확인합니다. 점들이 직선에 가까울수록 정규성을 만족합니다.

4.  **오차의 등분산성 검증 (선택 사항)**:
    -   예측값에 따라 잔차의 흩어짐(분산)이 일정해야 합니다. 만약 예측값이 커질수록 잔차도 커지는 등 특정 패턴을 보이면 등분산성 가정이 깨진 것입니다.

결론적으로, `results.resid`는 모델의 예측이 얼마나 빗나갔는지를 보여주는 "성적표"이자, 모델의 건강 상태를 진단하는 데 사용하는 가장 중요한 "청진기"와 같은 역할을 합니다.

### 추가 자료

-   [Statsmodels 공식 문서: OLSResults.resid](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.resid.html)
-   [Wikipedia: Errors and residuals (영문)](https://en.wikipedia.org/wiki/Errors_and_residuals)
-   [Khan Academy: Introduction to residuals](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/regression-library/a/introduction-to-residuals)