# fit_transform과 transform의 차이점

`fit_transform`과 `transform`은 scikit-learn에서 데이터 전처리 시 매우 중요한 개념입니다.

## 예시 파일
과적합 방지 노트북

## 답변

`fit_transform`과 `transform`의 핵심 차이는 **학습 여부**에 있습니다.

### 🔍 핵심 개념

**fit_transform**: **학습 + 변환**을 동시에 수행
**transform**: 이미 학습된 규칙으로 **변환만** 수행

### ⚡ 실제 예시로 이해하기

```python
from sklearn.preprocessing import StandardScaler

# 1. fit_transform: 훈련 데이터에 사용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 평균, 표준편차 학습 + 변환

# 2. transform: 테스트 데이터에 사용  
X_test_scaled = scaler.transform(X_test)  # 학습된 규칙으로 변환만
```

### 📊 동작 과정 상세 분석

**fit_transform 내부 동작**:
1. **fit**: 훈련 데이터에서 통계값 계산 (평균, 표준편차 등)
2. **transform**: 계산된 통계값으로 데이터 변환

**transform 내부 동작**:
1. 이미 학습된 통계값 사용
2. 새로운 데이터를 동일한 규칙으로 변환

### 🎯 노트북 코드 예시 분석

노트북의 24번 셀에서 올바른 사용법을 확인할 수 있습니다:

```python
# ✅ 올바른 사용법
scaler = StandardScaler()
X_train[normalize_cols] = scaler.fit_transform(X_train[normalize_cols])  # 훈련 데이터: 학습+변환
X_test[normalize_cols] = scaler.transform(X_test[normalize_cols])        # 테스트 데이터: 변환만
```

### ⚠️ 잘못된 사용법과 문제점

```python
# ❌ 잘못된 사용법 - 절대 하면 안됨!
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train)

scaler2 = StandardScaler()  # 새로운 스케일러!
X_test_scaled = scaler2.fit_transform(X_test)  # 테스트에서 다시 학습!
```

**문제점**:
- 훈련/테스트 데이터가 **다른 기준**으로 변환됨
- 데이터 누출(Data Leakage) 발생
- 모델 성능 평가가 부정확해짐

### 📈 구체적인 수치 예시

```python
# 훈련 데이터
train_data = [1, 2, 3, 4, 5]  # 평균: 3, 표준편차: 1.58

# fit_transform으로 학습된 규칙
# 변환공식: (값 - 3) / 1.58

# 테스트 데이터
test_data = [6, 7, 8]

# ✅ 올바른 방법: 훈련 데이터 규칙 적용
# (6-3)/1.58 = 1.90, (7-3)/1.58 = 2.53, (8-3)/1.58 = 3.16

# ❌ 잘못된 방법: 테스트 데이터로 새로 학습
# 테스트 평균: 7, 표준편차: 1
# (6-7)/1 = -1, (7-7)/1 = 0, (8-7)/1 = 1  # 완전히 다른 결과!
```

### 🔧 다른 전처리 기법에서의 적용

**MinMaxScaler 상세 예시**:
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 예시 데이터
X_train = np.array([[1], [3], [5], [7], [9]])    # 최소: 1, 최대: 9
X_test = np.array([[2], [10], [15]])             # 최소: 2, 최대: 15

print("원본 데이터:")
print(f"훈련 데이터: {X_train.flatten()}")  # [1 3 5 7 9]
print(f"테스트 데이터: {X_test.flatten()}")  # [2 10 15]

# ✅ 올바른 방법: 훈련 데이터 기준으로 학습
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터의 min=1, max=9 학습
X_test_scaled = scaler.transform(X_test)        # 같은 min=1, max=9 기준 적용

print("\n✅ 올바른 방법 결과:")
print(f"훈련 스케일링: {X_train_scaled.flatten()}")  # [0.0, 0.25, 0.5, 0.75, 1.0]
print(f"테스트 스케일링: {X_test_scaled.flatten()}")  # [0.125, 1.125, 1.75] (범위 벗어남 가능!)

# ❌ 잘못된 방법: 각각 따로 학습
scaler_wrong = MinMaxScaler()
X_test_wrong = scaler_wrong.fit_transform(X_test)  # 테스트 데이터의 min=2, max=15로 새로 학습

print("\n❌ 잘못된 방법 결과:")
print(f"테스트 잘못 스케일링: {X_test_wrong.flatten()}")  # [0.0, 0.615, 1.0] (완전히 다른 의미!)
```

**왜 이렇게 해야 할까요?**

1. **모델이 훈련 데이터 기준으로 학습했기 때문**
   - 모델은 [0.0, 0.25, 0.5, 0.75, 1.0] 범위의 데이터로 학습
   - 테스트할 때도 같은 기준으로 변환해야 함

2. **일관성 유지**
   - 값 2는 훈련 기준으로 0.125 (1과 3 사이)
   - 테스트 기준으로는 0.0 (최솟값) - 완전히 다른 의미!

3. **실제 운영 환경 시뮬레이션**
   - 새로운 데이터가 올 때마다 다시 학습할 수 없음
   - 미리 정한 기준으로 변환해야 함

**LabelEncoder**:
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)  # 클래스 매핑 학습
y_test_encoded = encoder.transform(y_test)        # 같은 매핑 적용
```

### 💡 핵심 원칙

1. **훈련 데이터**: `fit_transform` 사용 (학습 + 변환)
2. **테스트 데이터**: `transform` 사용 (변환만)
3. **같은 스케일러 객체** 사용 (다른 객체 생성 금지)
4. **일관된 변환 규칙** 유지

### 🚨 핵심 이해: 왜 다르게 처리해야 하나?

**현실 상황 비유**:
- 훈련 데이터 = 시험 문제 (기준 설정)
- 테스트 데이터 = 실제 현장 (기준 적용)

**구체적인 이유**:

1. **모델은 특정 기준으로 학습함**
   ```python
   # 모델이 보는 세상: "1~9 범위를 0~1로 변환한 데이터"
   # 값 5 → 0.5로 변환된 데이터로 학습
   # 따라서 새로운 값 5도 반드시 0.5로 변환해야 함
   ```

2. **테스트 데이터로 새로 학습하면 기준이 바뀜**
   ```python
   # ❌ 잘못된 예
   # 훈련: 값 5 → 0.5 (1~9 기준)
   # 테스트: 값 5 → 0.23 (2~15 기준) ← 모델이 혼란!
   ```

3. **실제 서비스에서는 새 데이터가 계속 들어옴**
   ```python
   # 운영 중인 모델에 새 데이터 하나씩 들어올 때
   # 매번 다시 학습할 수 없음!
   # 미리 정한 기준 사용해야 함
   ```

### 🎯 실무 베스트 프랙티스

```python
# 1. 스케일러 생성
scaler = StandardScaler()

# 2. 훈련 데이터로 학습 및 변환
X_train_scaled = scaler.fit_transform(X_train)

# 3. 동일한 스케일러로 테스트 데이터 변환
X_test_scaled = scaler.transform(X_test)

# 4. 새로운 데이터도 같은 방식
X_new_scaled = scaler.transform(X_new)
```

### 추가 자료

- [scikit-learn 전처리 가이드](https://scikit-learn.org/stable/modules/preprocessing.html)
- [데이터 누출 방지 방법](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [StandardScaler 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---
🎯 **핵심 요약**: `fit_transform`은 훈련 데이터에서 **학습+변환**, `transform`은 테스트 데이터에서 **변환만**! 동일한 변환 규칙을 유지하는 것이 머신러닝 성공의 핵심입니다.
