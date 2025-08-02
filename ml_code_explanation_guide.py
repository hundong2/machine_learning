# 머신러닝 당뇨병 예측 모델 - 초보자를 위한 상세 주석 가이드

"""
=== 당뇨병 예측 머신러닝 프로젝트 완전 분석 ===

이 파일은 기존 goodexample.ipynb의 모든 코드에 대한 상세한 설명을 제공합니다.
각 함수, 모델, 기법에 대한 이론적 배경과 실제 사용법을 초보자 관점에서 설명합니다.

주요 학습 내용:
1. 데이터 전처리 (결측치, 이상치 처리)
2. 데이터 시각화 (히트맵, 박스플롯)
3. 머신러닝 모델 (RandomForest, XGBoost, SVM)
4. 앙상블 기법 (Voting, Stacking)
5. 고급 기법 (Pseudolabeling, Threshold Tuning)
6. 모델 평가 (Cross-validation, ROC-AUC, F1-Score)
"""

# ============================================================================
# 1. 라이브러리 임포트 및 설명
# ============================================================================

# 기본 데이터 처리 라이브러리
import pandas as pd  
# pandas: 테이블 형태의 데이터를 다루는 라이브러리
# DataFrame, Series 등의 자료구조 제공
# CSV 파일 읽기/쓰기, 데이터 필터링, 그룹화 등 가능

import numpy as np   
# numpy: 수치 계산을 위한 라이브러리
# 다차원 배열(ndarray) 제공, 수학 함수들 포함
# 머신러닝의 기반이 되는 라이브러리

# 데이터 시각화 라이브러리
import matplotlib.pyplot as plt  
# matplotlib: Python의 기본 시각화 라이브러리
# 선 그래프, 히스토그램, 산점도 등 다양한 그래프 생성
# plt.figure(), plt.subplot(), plt.show() 등의 함수 사용

import seaborn as sns            
# seaborn: matplotlib 기반의 통계 시각화 라이브러리
# 더 예쁘고 통계적인 그래프를 쉽게 생성
# heatmap, boxplot, distplot 등 제공

# 머신러닝 라이브러리 (scikit-learn)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
"""
train_test_split: 데이터를 훈련용/테스트용으로 분할
- test_size: 테스트 데이터 비율 (0.2 = 20%)
- random_state: 재현 가능한 결과를 위한 시드값
- stratify: 클래스 비율을 유지하며 분할

cross_val_score: 교차 검증을 통한 모델 성능 평가
- 데이터를 k개 폴드로 나누어 k번 학습/평가 반복
- 과적합을 방지하고 일반화 성능을 정확히 측정

StratifiedKFold: 계층화된 교차 검증
- 각 폴드에서 클래스 비율을 동일하게 유지
- 불균형 데이터에서 특히 중요
"""

from sklearn.preprocessing import StandardScaler
"""
StandardScaler: 특성 표준화 (정규화)
- 평균을 0, 표준편차를 1로 변환 (z-score normalization)
- 공식: (x - 평균) / 표준편차
- 서로 다른 스케일의 특성들을 동일한 범위로 조정
- SVM, 로지스틱 회귀 등에서 필수적
"""

from sklearn.ensemble import VotingClassifier, StackingClassifier
"""
VotingClassifier: 투표 방식 앙상블
- Hard Voting: 각 모델의 예측 클래스 중 다수결로 결정
- Soft Voting: 각 모델의 확률을 평균내어 결정 (더 정확)
- 서로 다른 알고리즘의 장점을 결합

StackingClassifier: 스태킹 앙상블
- 1단계: 여러 기본 모델(base model)들이 예측
- 2단계: 메타 모델(meta model)이 기본 모델들의 예측을 입력으로 최종 예측
- 더 복잡하지만 일반적으로 성능이 더 좋음
"""

from sklearn.linear_model import LogisticRegression
"""
LogisticRegression: 로지스틱 회귀
- 분류 문제를 위한 선형 모델
- 시그모이드 함수를 사용하여 확률을 출력 (0~1 사이)
- 해석하기 쉽고 빠른 학습 속도
- 선형적으로 분리 가능한 데이터에 효과적
"""

from sklearn.ensemble import RandomForestClassifier
"""
RandomForestClassifier: 랜덤 포레스트
- 여러 개의 결정 트리를 결합한 앙상블 모델
- 배깅(Bagging) + 랜덤 특성 선택
- 장점: 과적합 방지, 특성 중요도 제공, 안정적 성능
- 단점: 해석이 어려움, 메모리 사용량 높음
"""

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve
"""
평가 지표들:

accuracy_score: 정확도
- (올바른 예측 수) / (전체 예측 수)
- 가장 직관적이지만 불균형 데이터에서 misleading할 수 있음

roc_auc_score: ROC-AUC 점수
- ROC 곡선 아래 면적 (Area Under the Curve)
- 0.5는 랜덤, 1.0은 완벽한 분류기
- 불균형 데이터에서도 안정적인 평가 지표

f1_score: F1 점수
- Precision과 Recall의 조화평균
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- 정밀도와 재현율의 균형을 고려

precision_recall_curve: 정밀도-재현율 곡선
- 다양한 임계값에서의 정밀도와 재현율 계산
- 최적 임계값 찾기에 사용
"""

from sklearn.svm import SVC
"""
SVC: Support Vector Classifier (서포트 벡터 머신)
- 데이터를 고차원 공간으로 매핑하여 분리 평면 찾기
- 커널 트릭을 사용하여 비선형 데이터도 처리 가능
- 장점: 고차원 데이터에 효과적, 메모리 효율적
- 단점: 큰 데이터셋에서 느림, 확률 출력 기본 제공 안 함
"""

from xgboost import XGBClassifier
"""
XGBClassifier: XGBoost (eXtreme Gradient Boosting)
- 그래디언트 부스팅의 최적화된 구현
- 부스팅: 약한 학습기들을 순차적으로 학습하여 결합
- 각 단계에서 이전 모델의 오류를 보정
- 캐글 등 머신러닝 경진대회에서 우수한 성능
- 특성: 빠른 속도, 메모리 효율성, 과적합 방지 기능
"""

# ============================================================================
# 2. 데이터 로딩 및 탐색
# ============================================================================

# CSV 파일 읽기
train = pd.read_csv('train.csv')        # 훈련 데이터 (정답 포함)
test = pd.read_csv('test.csv')          # 테스트 데이터 (정답 없음)
submission = pd.read_csv('sample_submission.csv')  # 제출 형식 파일

# 데이터 정보 확인
print(train.info())  # 데이터 타입, 결측치, 메모리 사용량 등
print(test.info())

"""
.info() 메서드 해석:
- Non-Null Count: 결측치가 아닌 데이터 개수
- Dtype: 데이터 타입 (int64, float64, object 등)
- Memory usage: 메모리 사용량

일반적으로 확인해야 할 것들:
1. 데이터 크기 (행, 열 개수)
2. 결측치 존재 여부
3. 데이터 타입의 적절성
4. 메모리 사용량 (큰 데이터의 경우)
"""

# ============================================================================
# 3. 데이터 전처리 - 결측치 및 이상치 처리
# ============================================================================

# 0을 결측치로 간주하여 시각화
train_zero = train.replace(0, np.nan)
"""
의료 데이터의 특성:
- Glucose(혈당), BloodPressure(혈압), BMI 등이 0인 것은 비현실적
- 실제로는 측정되지 않았거나 기록되지 않은 값일 가능성
- 따라서 0을 결측치(NaN)로 처리하여 패턴 확인
"""

# 히트맵으로 결측치 패턴 시각화
plt.figure(figsize=(10, 6))
sns.heatmap(train_zero.isnull(), cbar=False, cmap='viridis')
plt.show()

"""
sns.heatmap() 매개변수 설명:
- data: 시각화할 데이터 (여기서는 결측치 여부)
- cbar: 컬러바 표시 여부
- cmap: 색상 맵 ('viridis', 'plasma', 'coolwarm' 등)

히트맵 해석:
- 어두운 색: 결측치 위치
- 밝은 색: 정상 데이터 위치
- 패턴을 통해 결측치의 원인 추론 가능
"""

# 박스플롯으로 이상치 확인
plt.figure(figsize=(15, 10))

columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i, column in enumerate(columns_to_check):
    plt.subplot(2, 3, i + 1)  # 2행 3열의 부분 그래프
    sns.boxplot(x=train[column])
    plt.title(f'{column} Box Plot')

plt.tight_layout()  # 그래프들 간의 간격 자동 조정
plt.show()

"""
박스플롯(Box Plot) 해석:
- 상자: 1사분위수(Q1) ~ 3사분위수(Q3), IQR(Interquartile Range)
- 중앙선: 중위수(median)
- 수염: Q1-1.5*IQR ~ Q3+1.5*IQR
- 점들: 이상치(outliers)

plt.subplot(rows, cols, index):
- 여러 그래프를 한 화면에 배치
- 2행 3열 격자에서 index번째 위치에 그래프 생성
"""

# ============================================================================
# 4. 도메인 지식을 활용한 이상치 처리
# ============================================================================

# Insulin 이상치 처리 (500 이상)
insulin_median = train['Insulin'].median()
train.loc[train['Insulin'] >= 500, 'Insulin'] = insulin_median
test.loc[test['Insulin'] >= 500, 'Insulin'] = insulin_median

"""
도메인 지식 적용:
- 공복 시 인슐린 수치는 0일 수 있음 (정상)
- 당뇨 환자도 최대 600 정도까지 측정됨
- 500 이상은 극히 예외적이므로 중위수로 대체

.loc[] 인덱서:
- 조건에 맞는 행과 열을 선택하여 값 변경
- train.loc[조건, '열이름'] = 새로운_값
"""

# BMI 이상치 처리 (0 또는 50 이상)
bmi_median = train['BMI'].median()
train.loc[(train['BMI'] == 0) | (train['BMI'] >= 50), 'BMI'] = bmi_median
test.loc[(test['BMI'] == 0) | (test['BMI'] >= 50), 'BMI'] = bmi_median

"""
BMI(Body Mass Index) 해석:
- BMI = 체중(kg) / 키(m)²
- BMI 50 = 170cm에서 약 145kg
- 의학적으로 극도비만 수준이므로 이상치로 판단

논리 연산자:
- | : OR 연산자
- & : AND 연산자
- ~ : NOT 연산자
"""

# BloodPressure 이상치 처리 (0인 경우만)
bloodpressure_median = train['BloodPressure'].median()
train.loc[train['BloodPressure'] == 0, 'BloodPressure'] = bloodpressure_median
test.loc[test['BloodPressure'] == 0, 'BloodPressure'] = bloodpressure_median

"""
혈압 관련 도메인 지식:
- 혈압이 0인 것은 불가능 (생명 위험)
- 고혈압, 저혈압도 당뇨와 관련이 있으므로 보존
- 0인 값만 중위수로 대체
"""

# Glucose 이상치 처리 (0인 경우)
glucose_median = train['Glucose'].median()
train.loc[train['Glucose'] == 0, 'Glucose'] = glucose_median
test.loc[test['Glucose'] == 0, 'Glucose'] = glucose_median

"""
혈당(Glucose) 관련:
- 혈당이 0인 것은 의학적으로 불가능
- 당뇨 진단의 핵심 지표이므로 정확한 값이 중요
"""

# SkinThickness 특별 처리 (99를 0으로 변경)
train.loc[train['SkinThickness'] == 99, 'SkinThickness'] = 0
test.loc[test['SkinThickness'] == 99, 'SkinThickness'] = 0

"""
데이터 오류 수정:
- 99는 일반적으로 '결측치 코드'로 사용됨
- 실제 피부 두께가 99mm는 비현실적
- 도메인 전문가의 판단으로 0(결측치)으로 변경
"""

# ============================================================================
# 5. 특성과 타겟 분리
# ============================================================================

# 독립변수(features)와 종속변수(target) 분리
X = train.drop(columns=['ID', 'Outcome'])  # ID와 정답 제거
y = train['Outcome']                       # 정답 (0: 정상, 1: 당뇨)
X_test = test.drop(columns=['ID'])         # 테스트 데이터에서 ID 제거

"""
머신러닝 데이터 구조:
- X: 독립변수, 특성(features), 입력값
  - 모델이 학습할 패턴이 들어있는 데이터
- y: 종속변수, 타겟(target), 정답(label)
  - 모델이 예측해야 할 값

.drop() 메서드:
- columns: 제거할 열 이름들
- axis=1: 열 방향으로 제거 (기본값)
- inplace=False: 원본 보존하고 새 객체 반환 (기본값)
"""

# ============================================================================
# 6. 데이터 분할
# ============================================================================

# 훈련/검증 데이터 분할 (70:30 비율)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.3,      # 검증 데이터 30%
    random_state=42     # 재현 가능한 결과
)

"""
train_test_split 전략:
- 일반적 비율: 80:20, 70:30, 60:20:20 (train:val:test)
- test_size: 검증 데이터 비율
- random_state: 동일한 분할 결과를 위한 시드값
- stratify=y: 클래스 비율을 유지하며 분할 (불균형 데이터에 중요)

분할 이유:
- 과적합 방지
- 모델의 일반화 능력 평가
- 하이퍼파라미터 튜닝
"""

# ============================================================================
# 7. 데이터 표준화 (스케일링)
# ============================================================================

# 표준화 객체 생성 및 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터로 학습 후 변환
X_val_scaled = scaler.transform(X_val)          # 검증 데이터 변환만
X_test_scaled = scaler.transform(X_test)        # 테스트 데이터 변환만

"""
StandardScaler 중요 개념:

fit(): 평균과 표준편차 계산
- 훈련 데이터에서만 통계량 계산
- 데이터 리키지(data leakage) 방지

transform(): 실제 표준화 적용
- (값 - 평균) / 표준편차

fit_transform(): fit() + transform()
- 훈련 데이터에서만 사용

주의사항:
- 검증/테스트 데이터는 transform()만 사용
- 각 데이터셋별로 따로 fit() 하면 안 됨 (데이터 리키지)
"""

# ============================================================================
# 8. 앙상블 모델 - Voting Classifier
# ============================================================================

# 소프트 투표 방식의 앙상블 모델 구성
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),    # 랜덤 포레스트
        ('xgb', XGBClassifier(random_state=42)),           # XGBoost
        ('svc', SVC(probability=True, random_state=42))     # SVM
    ],
    voting='soft'  # 확률 기반 투표
)

"""
VotingClassifier 상세 설명:

estimators: (이름, 모델) 튜플의 리스트
- 이름은 식별자 역할 (디버깅, 시각화에 사용)
- 각기 다른 알고리즘으로 다양성 확보

voting 방식:
- 'hard': 각 모델의 예측 클래스를 다수결로 결정
- 'soft': 각 모델의 확률을 평균내어 결정 (더 정확)

SVC의 probability=True:
- SVM은 기본적으로 확률 출력 안 함
- 소프트 투표를 위해 확률 출력 활성화
- 계산 시간이 다소 증가하지만 성능 향상
"""

# ============================================================================
# 9. 교차 검증을 통한 모델 평가
# ============================================================================

# 계층화된 K-Fold 교차 검증 설정
kfold = StratifiedKFold(
    n_splits=5,         # 5-fold 교차 검증
    shuffle=True,       # 데이터 섞기
    random_state=42     # 재현 가능한 결과
)

# 교차 검증 수행
cross_val_scores = cross_val_score(
    voting_clf,         # 사용할 모델
    X_train_scaled,     # 훈련 특성
    y_train,           # 훈련 타겟
    cv=kfold,          # 교차 검증 전략
    scoring='roc_auc'  # 평가 지표
)

print("Cross-validation ROC-AUC scores:", cross_val_scores)
print("Mean cross-validation ROC-AUC:", np.mean(cross_val_scores))

"""
교차 검증(Cross-Validation) 개념:

StratifiedKFold:
- 각 폴드에서 클래스 비율 유지
- 불균형 데이터에서 특히 중요
- n_splits: 폴드 개수 (일반적으로 5 또는 10)

교차 검증 과정:
1. 데이터를 5개 폴드로 분할
2. 4개 폴드로 학습, 1개 폴드로 평가 (5번 반복)
3. 5개의 성능 점수 평균으로 최종 성능 추정

장점:
- 모든 데이터를 학습과 평가에 활용
- 더 안정적이고 신뢰할 수 있는 성능 추정
- 과적합 여부 판단 가능
"""

# 모델 학습
voting_clf.fit(X_train_scaled, y_train)

# ============================================================================
# 10. 고급 기법 - Pseudolabeling
# ============================================================================

# 테스트 데이터로 예측하여 의사 라벨 생성
pseudo_labels = voting_clf.predict(X_test_scaled)
pseudo_data = pd.DataFrame(X_test_scaled, columns=X.columns)
pseudo_data['Outcome'] = pseudo_labels

"""
Pseudolabeling (의사 라벨링):

개념:
- 학습된 모델로 테스트 데이터의 라벨을 예측
- 예측된 라벨을 "정답"으로 간주하여 훈련 데이터에 추가
- 확장된 데이터로 모델을 재학습

장점:
- 더 많은 데이터로 학습 가능
- 일반화 성능 향상 가능
- 준지도 학습(Semi-supervised Learning)의 한 형태

주의사항:
- 잘못된 예측이 포함될 수 있음
- 모델이 자신의 편향을 강화할 위험
- 신중한 사용 필요
"""

# 훈련 데이터 확장
X_exp = np.vstack((X_train_scaled, X_test_scaled))  # 세로로 결합
y_exp = np.hstack((y_train, pseudo_labels))         # 가로로 결합

# 확장된 데이터로 모델 재학습
voting_clf.fit(X_exp, y_exp)

"""
NumPy 배열 결합:

np.vstack(): 세로 방향 결합 (Vertical Stack)
- 행(row) 방향으로 배열을 쌓음
- 열 개수가 같아야 함

np.hstack(): 가로 방향 결합 (Horizontal Stack)
- 열(column) 방향으로 배열을 연결
- 행 개수가 같아야 함

데이터 확장 효과:
- 원래 훈련 데이터 + 의사 라벨링된 테스트 데이터
- 더 큰 학습 데이터셋으로 모델 성능 향상 기대
"""

# ============================================================================
# 11. 스태킹 앙상블
# ============================================================================

# 스태킹 분류기 구성
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(),  # 메타 학습기
    cv=kfold                               # 교차 검증 전략
)

stacking_clf.fit(X_exp, y_exp)

"""
Stacking (스태킹) 앙상블:

2단계 학습 과정:
1. Base Models (기본 모델들):
   - RandomForest, XGBoost, SVM이 각각 예측
   - 교차 검증을 통해 예측값 생성

2. Meta Model (메타 모델):
   - 기본 모델들의 예측값을 특성으로 사용
   - 최종 예측을 수행하는 모델
   - 여기서는 LogisticRegression 사용

Voting vs Stacking:
- Voting: 단순 평균 또는 다수결
- Stacking: 메타 모델이 최적 조합 학습

장점:
- 각 모델의 강점을 더 효과적으로 결합
- 일반적으로 Voting보다 높은 성능

단점:
- 계산 복잡도 증가
- 과적합 위험 증가
- 해석 어려움
"""

# ============================================================================
# 12. 임계값 튜닝 (Threshold Tuning)
# ============================================================================

# 검증 데이터에 대한 확률 예측
y_pred_prob = stacking_clf.predict_proba(X_val_scaled)[:, 1]

# 정밀도-재현율 곡선 계산
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)

"""
predict_proba() 메서드:
- 각 클래스에 대한 확률 반환
- 이진 분류: [클래스0 확률, 클래스1 확률]
- [:, 1]: 양성 클래스(클래스1)의 확률만 선택

precision_recall_curve():
- 다양한 임계값에서의 정밀도와 재현율 계산
- 임계값을 조정하여 성능 최적화 가능

precision (정밀도):
- 양성으로 예측한 것 중 실제 양성 비율
- TP / (TP + FP)
- "예측의 정확성"

recall (재현율, 민감도):
- 실제 양성 중 올바르게 예측한 비율  
- TP / (TP + FN)
- "실제 양성의 검출율"
"""

# 최적 F1 점수를 위한 임계값 찾기
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]
print("Best Threshold for F1 Score:", best_threshold)

# 최적 임계값으로 예측
y_pred_val = (y_pred_prob >= best_threshold).astype(int)

"""
F1 Score:
- 정밀도와 재현율의 조화평균
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- 불균형 데이터에서 중요한 지표
- 1에 가까울수록 좋음

임계값(Threshold) 조정:
- 기본값: 0.5
- 임계값 ↑: 정밀도 ↑, 재현율 ↓
- 임계값 ↓: 정밀도 ↓, 재현율 ↑
- 비즈니스 목적에 따라 조정

np.argmax():
- 배열에서 최대값의 인덱스 반환
- 최고 F1 점수를 가진 임계값의 위치 찾기
"""

# ============================================================================
# 13. 최종 성능 평가
# ============================================================================

# 다양한 평가 지표로 성능 측정
accuracy_val = accuracy_score(y_val, y_pred_val)
roc_auc_val = roc_auc_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)

print("Validation accuracy:", accuracy_val)
print("Validation ROC-AUC:", roc_auc_val)
print("Validation F1 Score:", f1_val)

"""
종합적 성능 평가의 중요성:

Accuracy (정확도):
- 전체 예측 중 맞춘 비율
- 직관적이지만 불균형 데이터에서 misleading

ROC-AUC:
- 모든 임계값에서의 성능을 종합한 지표
- 클래스 불균형에 상대적으로 robust
- 0.5는 랜덤, 1.0은 완벽

F1 Score:
- 정밀도와 재현율의 균형
- 불균형 데이터에서 특히 중요
- 의료 진단 등에서 핵심 지표

평가 지표 선택 기준:
- 비즈니스 목적
- 데이터 특성 (균형/불균형)
- 거짓 양성 vs 거짓 음성의 비용
"""

# ============================================================================
# 14. 최종 모델 학습 및 예측
# ============================================================================

# 전체 데이터로 모델 재학습
X_final = np.vstack((X_train_scaled, X_val_scaled))
y_final = np.hstack((y_train, y_val))

voting_clf.fit(X_final, y_final)

"""
최종 모델 전략:

모든 가용 데이터 활용:
- 검증 데이터까지 포함하여 최대한 많은 데이터로 학습
- 일반적으로 더 많은 데이터 = 더 좋은 성능

주의사항:
- 이 시점 이후로는 성능 평가 불가
- 교차 검증 점수를 최종 성능으로 간주
- 테스트 데이터로는 절대 평가하면 안 됨 (data leakage)
"""

# 최종 테스트 예측 및 제출 파일 생성
y_test_pred_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_pred_prob >= best_threshold).astype(int)

submission['Outcome'] = y_test_pred
submission.to_csv('submission.csv', index=False)

"""
제출 파일 생성:

과정:
1. 테스트 데이터에 대한 확률 예측
2. 최적 임계값 적용하여 클래스 예측
3. 제출 형식에 맞게 저장

to_csv() 매개변수:
- index=False: 행 인덱스 제외
- 캐글 등 경진대회의 제출 형식에 맞춤

최종 워크플로우:
1. 데이터 전처리
2. 모델 학습 및 검증
3. 하이퍼파라미터 튜닝
4. 최종 예측 및 제출
"""

# ============================================================================
# 추가 학습 자료 및 참고사항
# ============================================================================

"""
=== 머신러닝 프로젝트 체크리스트 ===

1. 데이터 이해:
   - 도메인 지식 습득
   - 데이터 탐색적 분석 (EDA)
   - 특성별 분포 및 관계 파악

2. 데이터 전처리:
   - 결측치 처리
   - 이상치 처리  
   - 특성 스케일링
   - 특성 선택/생성

3. 모델 선택:
   - 문제 유형 파악 (분류/회귀)
   - 데이터 크기 고려
   - 해석 가능성 vs 성능 trade-off

4. 모델 평가:
   - 적절한 평가 지표 선택
   - 교차 검증 수행
   - 과적합/언더피팅 체크

5. 모델 개선:
   - 하이퍼파라미터 튜닝
   - 앙상블 기법 적용
   - 특성 엔지니어링

6. 배포 준비:
   - 모델 저장
   - 추론 파이프라인 구축
   - 모니터링 체계 마련

=== 권장 학습 순서 ===

1. 기초 개념: 선형회귀, 로지스틱 회귀
2. 트리 기반: 결정트리, 랜덤포레스트
3. 앙상블: 배깅, 부스팅, 스태킹
4. 고급 기법: SVM, 신경망
5. 평가 및 검증: 교차검증, 평가지표
6. 실전 기법: 특성선택, 하이퍼파라미터 튜닝

=== 추천 도서 및 자료 ===

1. 도서:
   - "핸즈온 머신러닝" (오렐리앙 제롱)
   - "파이썬 머신러닝 완벽 가이드" (권철민)
   - "밑바닥부터 시작하는 머신러닝" (조엘 그루스)

2. 온라인 강의:
   - Coursera Machine Learning (Andrew Ng)
   - Kaggle Learn
   - Fast.ai

3. 실습 플랫폼:
   - Kaggle Competitions
   - Google Colab
   - Jupyter Notebook

이 코드는 실제 데이터 사이언스 프로젝트의 전 과정을 포함하고 있으며,
각 단계별 이론과 실습을 통해 머신러닝의 핵심을 이해할 수 있습니다.
"""
