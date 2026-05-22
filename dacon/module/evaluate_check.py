import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_classification_model_detailed(y_true, y_pred, class_names=['Negative(0)', 'Positive(1)']):
    """
    분류 모델의 성능 지표와 혼동 행렬 분석(TP, FP, FN, TN 의미 해석)을 출력하는 함수.
    이진 분류(Binary Classification)에 최적화되어 있습니다.
    """
    print("=" * 60)
    print(" 🌟 분류 모델 상세 성능 평가 리포트 🌟 ")
    print("=" * 60)
    
    # 1. 종합 성능 지표 출력
    print("\n[1. 성능 지표 요약]")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 2. Confusion Matrix 요소 추출 및 해석
    cm = confusion_matrix(y_true, y_pred)
    # 이진 분류의 경우 matrix는 [[TN, FP], [FN, TP]] 형태입니다.
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print("-" * 60)
        print("[2. 혼동 행렬 상세 분석]")
        print(f"▶ TN (True Negative) : {tn:4d}개 -> 실제 0을 0으로 잘 예측함 (정상 판단)")
        print(f"▶ FP (False Positive): {fp:4d}개 -> 실제 0인데 1로 잘못 예측함 (1종 오류 / 거짓 알람)")
        print(f"▶ FN (False Negative): {fn:4d}개 -> 실제 1인데 0으로 잘못 예측함 (2종 오류 / 진짜를 놓침)")
        print(f"▶ TP (True Positive) : {tp:4d}개 -> 실제 1을 1로 잘 예측함 (정탐)")
        
        print("\n[💡 지표 해석 가이드]")
        print("- TP가 높을 때 (좋음): 모델이 우리가 찾고자 하는 타겟(1)을 아주 잘 잡아내고 있다는 의미입니다.")
        print("- FP가 높을 때 (나쁨): 거짓 알람이 많다는 의미입니다. (예: 정상 메일을 스팸으로 분류해 휴지통으로 보냄)")
        print("- FN이 높을 때 (가장 위험): 진짜를 놓치는 경우입니다. (예: 암 환자를 정상으로 분류해 치료를 놓침) 모델이 너무 둔감할 때 발생합니다.")
        print("- F1-Score (높을수록 좋음): 정밀도와 재현율의 균형을 나타내며, 1에 가까울수록(높을수록) 모델 검출 능력이 매우 훌륭하다는 뜻입니다.")
    
    # 3. 시각화
    print("-" * 60)
    print("[3. 혼동 행렬 시각화]")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Confusion Matrix")
    plt.show()
    