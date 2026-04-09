# 선형보간 계산방법 (Linear Interpolation)

선형보간은 두 점 사이의 직선을 이용하여 중간값을 추정하는 방법으로, 결측치 처리에서 가장 많이 사용되는 기법입니다.

## 예시 파일

````python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 📊 선형보간 계산방법 완전 가이드 ===")

# 선형보간 공식 예시 데이터
print("📋 기본 개념:")
print("선형보간은 두 알려진 점 사이에서 직선을 그어 중간값을 추정하는 방법입니다.")
print()

# 실제 계산 예시
print("🔢 계산 예시:")
x0, y0 = 1, 10  # 첫 번째 알려진 점
x2, y2 = 3, 30  # 두 번째 알려진 점
x1 = 2          # 보간하려는 지점

# 수동 계산
y1_manual = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
print(f"점 ({x0}, {y0})과 ({x2}, {y2}) 사이에서 x={x1}일 때의 y값:")
print(f"수동 계산: y = {y0} + ({y2} - {y0}) × ({x1} - {x0}) / ({x2} - {x0}) = {y1_manual}")

# Pandas를 이용한 보간
data = pd.Series([10, np.nan, 30], index=[1, 2, 3])
interpolated = data.interpolate(method='linear')
print(f"Pandas 보간: {interpolated.iloc[1]}")
print()

print("=== 📐 선형보간 수학적 공식 ===")
print()
print("두 점 (x₀, y₀)과 (x₂, y₂) 사이에서 x₁에서의 y₁을 구하는 공식:")
print()
print("LaTeX 형식:")
print("y_1 = y_0 + \\frac{(y_2 - y_0)(x_1 - x_0)}{x_2 - x_0}")
print()
print("또는 비율을 이용한 형태:")
print("y_1 = y_0 + (y_2 - y_0) \\times \\frac{x_1 - x_0}{x_2 - x_0}")
print()

print("=== 📊 실무 예시: 온도 데이터 보간 ===")

# 온도 데이터 예시
dates = pd.date_range('2024-01-01', periods=7, freq='D')
temperatures = [22, np.nan, np.nan, 18, np.nan, 24, 26]
temp_data = pd.Series(temperatures, index=dates, name='Temperature')

print("🌡️ 원본 온도 데이터:")
print(temp_data)
print()

# 선형보간 적용
temp_interpolated = temp_data.interpolate(method='linear')
print("🔧 선형보간 후:")
print(temp_interpolated)
print()

# 각 보간값의 계산 과정 설명
print("💡 보간 계산 과정:")
print("• 1월 2일: 22 + (18-22) × (1/3) = 22 + (-4) × (1/3) = 20.67")
print("• 1월 3일: 22 + (18-22) × (2/3) = 22 + (-4) × (2/3) = 19.33")
print("• 1월 5일: 18 + (24-18) × (1/2) = 18 + 6 × 0.5 = 21.00")
print()

print("=== 🎯 다양한 보간 방법 비교 ===")

# 다양한 보간 방법
methods = ['linear', 'quadratic', 'cubic']
comparison_data = pd.Series([10, np.nan, np.nan, np.nan, 50], index=[0, 1, 2, 3, 4])

print("📊 원본 데이터:", comparison_data.dropna().tolist())
print()

for method in methods:
    try:
        result = comparison_data.interpolate(method=method)
        print(f"{method.capitalize()} 보간: {result.tolist()}")
    except:
        print(f"{method.capitalize()} 보간: 사용 불가 (데이터 부족)")

print()
print("=== ⚠️ 선형보간 주의사항 ===")
print()
print("✅ 장점:")
print("• 간단하고 직관적")
print("• 계산이 빠름")
print("• 안정적인 결과")
print()
print("❌ 단점:")
print("• 실제 데이터 패턴을 단순화")
print("• 급격한 변화 무시")
print("• 경계값에서 부정확할 수 있음")
print()
print("🎯 적용 가이드:")
print("• 데이터가 선형적 경향을 보일 때")
print("• 결측치가 연속적이지 않을 때")
print("• 빠른 처리가 필요할 때")

print("\n✅ 완료: 선형보간 계산방법 완전 마스터!")
````

## 답변

선형보간의 수학적 공식을 LaTeX 문법으로 자세히 설명드리겠습니다.

### 📐 기본 선형보간 공식

두 점 $(x_0, y_0)$과 $(x_2, y_2)$ 사이에서 $x_1$에서의 $y_1$을 구하는 공식:

$$y_1 = y_0 + \frac{(y_2 - y_0)(x_1 - x_0)}{x_2 - x_0}$$

### 🔍 공식 변형

**비율을 이용한 형태:**
$$y_1 = y_0 + (y_2 - y_0) \times \frac{x_1 - x_0}{x_2 - x_0}$$

**가중평균 형태:**
$$y_1 = \frac{(x_2 - x_1)y_0 + (x_1 - x_0)y_2}{x_2 - x_0}$$

**일반적인 직선의 방정식:**
$$y = y_0 + m(x - x_0)$$

여기서 기울기 $m = \frac{y_2 - y_0}{x_2 - x_0}$

### 📊 시간 기반 보간 공식

시간 데이터에서 $t_0, t_1, t_2$ 시점의 값들:

$$v(t_1) = v(t_0) + \frac{v(t_2) - v(t_0)}{t_2 - t_0} \times (t_1 - t_0)$$

### 🧮 다차원 선형보간

**2차원 이중선형보간 (Bilinear Interpolation):**
$$f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy$$

**3차원 삼중선형보간 (Trilinear Interpolation):**
$$f(x,y,z) = \sum_{i,j,k \in \{0,1\}} f(i,j,k) \times i^x(1-i)^{1-x} \times j^y(1-j)^{1-y} \times k^z(1-k)^{1-z}$$

### 📈 오차 분석

**선형보간 오차의 상한:**
$$|f(x) - P_1(x)| \leq \frac{M_2}{8}h^2$$

여기서:
- $M_2 = \max_{x_0 \leq x \leq x_2} |f''(x)|$
- $h = x_2 - x_0$ (구간의 길이)

### 🎯 실제 계산 예시

데이터: $(1, 10)$, $(?, ?)$, $(3, 30)$에서 $x=2$일 때:

$$y = 10 + \frac{(30-10)(2-1)}{3-1} = 10 + \frac{20 \times 1}{2} = 10 + 10 = 20$$

### 추가 자료

- [수치해석 - 보간법](https://en.wikipedia.org/wiki/Linear_interpolation)
- [Pandas 보간 함수 문서](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html)
- [NumPy 보간 함수](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
- [SciPy 보간 모듈](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
