# sklearn과 PyTorch 지원 기능 비교표

sklearn과 PyTorch의 지원 범위와 한계점을 비교하여 각 라이브러리의 특성을 이해할 수 있습니다.

## 예시 파일
[sklearn vs PyTorch 비교 예시](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/neural_networks_supervised.rst)

## 답변

| 기능/특징 | sklearn | PyTorch |
|-----------|---------|---------|
| **딥러닝 지원** | ❌ 제한적 (간단한 MLP만) | ✅ 완전 지원 |
| **GPU 가속** | ❌ 미지원 | ✅ 완전 지원 |
| **전통적 ML 알고리즘** | ✅ 완전 지원 | ❌ 미지원 |
| **자동 미분** | ❌ 미지원 | ✅ 완전 지원 |
| **복잡한 신경망 구조** | ❌ 미지원 | ✅ 완전 지원 |
| **전처리 도구** | ✅ 풍부한 지원 | ❌ 제한적 |
| **하이퍼파라미터 튜닝** | ✅ GridSearch, RandomSearch | ❌ 수동 구현 필요 |
| **모델 해석도구** | ✅ feature_importance 등 | ❌ 제한적 |
| **시계열 분석** | ❌ 제한적 | ✅ RNN, LSTM 등 |
| **컴퓨터 비전** | ❌ 미지원 | ✅ CNN 등 |
| **자연어처리** | ❌ 기본 기능만 | ✅ Transformer 등 |
| **분산 학습** | ❌ 미지원 | ✅ 지원 |
| **모바일 배포** | ❌ 미지원 | ✅ TorchScript |
| **산업용 파이프라인** | ✅ Pipeline 지원 | ❌ 수동 구현 |

### sklearn이 지원하지 않는 것들
- 복잡한 딥러닝 모델 (CNN, RNN, Transformer)
- GPU 연산
- 자동 미분 (Automatic Differentiation)
- 동적 계산 그래프

### PyTorch가 지원하지 않는 것들
- 전통적 머신러닝 알고리즘 (Random Forest, SVM 등)
- 자동 하이퍼파라미터 튜닝
- 풍부한 전처리 도구
- 간단한 모델 파이프라인

### 추가 자료
- [Scikit-learn 공식 문서](https://scikit-learn.org/stable/user_guide.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [ML vs DL 비교 가이드](https://www.tensorflow.org/guide)