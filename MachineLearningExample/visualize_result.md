# 예측치와 실제값 비교 시각화 방법

예측 모델의 성능을 시각적으로 평가하는 여러 가지 효과적인 방법들을 소개합니다.

## 예시 파일
[회귀 모델 시각화 예시](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)


````python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# 1. 산점도 (Scatter Plot) - 가장 기본적이고 효과적
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_valid, y_pred, alpha=0.6)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('실제값 (Actual)')
plt.ylabel('예측값 (Predicted)')
plt.title('예측값 vs 실제값')

# 2. 잔차 플롯 (Residual Plot)
plt.subplot(1, 2, 2)
residuals = y_valid - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측값 (Predicted)')
plt.ylabel('잔차 (Residuals)')
plt.title('잔차 플롯')
plt.tight_layout()
plt.show()

# 3. 성능 지표 출력
mse = mean_squared_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)
print(f'MSE: {mse:.4f}')
print(f'R²: {r2:.4f}')
````

**주요 비교 방법들:**
- **산점도**: 완벽한 예측이면 대각선 위에 점들이 위치
- **잔차 플롯**: 잔차가 0 주변에 무작위로 분포하면 좋은 모델
- **히스토그램**: 예측값과 실제값의 분포 비교

### 추가 자료
- [Matplotlib 회귀 시각화](https://matplotlib.org/stable/gallery/statistics/errorbar_limits_simple.html)
- [Seaborn 회귀 플롯](https://seaborn.pydata.org/examples/anscombes_quartet.html)