# Kaggle 30-Day Guide 🏆

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-orange)](https://lightgbm.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-competition--ready-20BEFF)](https://www.kaggle.com/)

완전 초보자부터 **Kaggle 대회 참가 가능 수준**까지 **4주 만에** 도달할 수 있도록 설계된
**15개의 Jupyter Notebook 시리즈**입니다.

수학 기초 · Pandas/NumPy · Gradient Boosting · Optuna · Stacking · Deep Learning · GitHub 배포까지
**실전에서 쓰는 모든 것**을 한 곳에 담았습니다.

---

## 📚 커리큘럼

### Week 1 · 기초 다지기
| 일자 | 노트북 | 내용 |
|------|--------|------|
| Day 1-2 | [01_Kaggle_Setup](notebooks/01_Week1_D01-02_Kaggle_Setup.ipynb) | Kaggle 가입, API, CLI, 대회 선택 전략 |
| Day 3-4 | [02_NumPy_Pandas](notebooks/02_Week1_D03-04_NumPy_Pandas.ipynb) | Vectorization, DataFrame 30패턴 |
| Day 5-7 | [03_Visualization_EDA](notebooks/03_Week1_D05-07_Visualization_EDA.ipynb) | EDA 10단계 체크리스트, 신호 탐지 |

### Week 2 · ML 기초 완성
| 일자 | 노트북 | 내용 |
|------|--------|------|
| Day 8-9 | [04_Math_Foundation](notebooks/04_Week2_D08-09_Math_Foundation.ipynb) | 선형대수, 확률/통계, 경사하강법 |
| Day 10-11 | [05_ML_Basics](notebooks/05_Week2_D10-11_ML_Basics.ipynb) | sklearn, Pipeline, 모델 비교 |
| Day 12-14 | [06_Feature_Engineering](notebooks/06_Week2_D12-14_Feature_Engineering.ipynb) | 결측/인코딩/상호작용/Target Encoding |

### Week 3 · 실전 모델링
| 일자 | 노트북 | 내용 |
|------|--------|------|
| Day 15-16 | [07_CV_Metrics](notebooks/07_Week3_D15-16_CV_Metrics.ipynb) | K-Fold/Stratified/Group/TimeSeries, Adversarial Val |
| Day 17-19 | [08_Gradient_Boosting](notebooks/08_Week3_D17-19_Gradient_Boosting.ipynb) | XGBoost vs LightGBM vs CatBoost |
| Day 20-21 | [09_Optuna_Ensemble](notebooks/09_Week3_D20-21_Optuna_Ensemble.ipynb) | Bayesian Opt, Stacking/Blending |

### Week 4 · 실전 대회
| 일자 | 노트북 | 내용 |
|------|--------|------|
| Day 22-23 | [10_Titanic_Project](notebooks/10_Week4_D22-23_Titanic_Project.ipynb) | 분류 E2E (FE → K-Fold → Stacking → Submit) |
| Day 24-25 | [11_HousePrices_Project](notebooks/11_Week4_D24-25_HousePrices_Project.ipynb) | 회귀 E2E (log-target, 4모델 stacking) |
| Day 26-27 | [12_Deep_Learning](notebooks/12_Week4_D26-27_Deep_Learning.ipynb) | PyTorch MLP & CNN, K-Fold NN |
| Day 28 | [13_Advanced_Techniques](notebooks/13_Week4_D28_Advanced_Techniques.ipynb) | NLP, MixUp/CutMix, Pseudo-labeling, TTA |
| Day 29-30 | [14_Workflow_Submission](notebooks/14_Week4_D29-30_Workflow_Submission.ipynb) | 재현성, 실험 관리, 제출 체크리스트 |
| FINAL | [15_GitHub_Upload_Guide](notebooks/15_FINAL_GitHub_Upload_Guide.ipynb) | 이 레포를 GitHub에 올리는 방법 |

---

## 🚀 시작하기

### 1) 레포지토리 클론
```bash
git clone https://github.com/<YOUR_USERNAME>/kaggle-guide.git
cd kaggle-guide
```

### 2) 가상 환경 구성
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Jupyter 실행
```bash
jupyter lab
# 또는
jupyter notebook
```

`notebooks/` 폴더에서 **01번부터 순서대로** 진행하세요.

---

## 📦 주요 라이브러리

- **데이터**: `numpy`, `pandas`, `scipy`
- **시각화**: `matplotlib`, `seaborn`
- **ML 기본**: `scikit-learn`
- **Gradient Boosting**: `lightgbm`, `xgboost`, `catboost`
- **튜닝**: `optuna`
- **딥러닝**: `torch`
- **Kaggle**: `kaggle` (CLI)

---

## 💡 이 가이드의 특징

- ✅ **완전 초보자용**: 수학과 Python 기초부터 차근차근
- ✅ **실전 지향**: 모든 코드가 실제 Kaggle 대회에 그대로 적용 가능
- ✅ **재현 가능**: 모든 노트북이 seed 고정, 독립 실행 가능
- ✅ **한국어**: 전 내용이 한국어로 작성됨
- ✅ **Data Leakage 방지**: 올바른 K-Fold Target Encoding, Pipeline 사용
- ✅ **상위권 기법**: Stacking, Pseudo-labeling, TTA, MixUp 등 포함

---

## 🏆 이 가이드를 마치면...

- LightGBM/XGBoost/CatBoost를 **자유자재로** 튜닝
- 로컬 CV와 Public LB 점수 **차이 해석** 가능
- Titanic, House Prices 같은 Getting Started 대회 **Top 10%** 도전 가능
- Playground 시리즈 **메달권** 경쟁 가능
- 자신만의 **experiment tracking pipeline** 확립

---

## 📝 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

## 🙌 기여

이슈와 Pull Request는 언제나 환영합니다!

## 📧 Contact

질문이나 피드백은 GitHub Issues에 남겨주세요.

---

**Happy Kaggling! 🚀**
