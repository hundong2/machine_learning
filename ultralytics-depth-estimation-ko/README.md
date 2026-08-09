# Ultralytics YOLO26 단안 깊이 추정 한글 학습 자료

한 장의 RGB 이미지로 각 픽셀까지의 거리를 추정하는 **단안 깊이 추정(monocular depth estimation)** 을 Ultralytics YOLO26으로 익히는 실습형 자료입니다. 공식 [Monocular Depth Estimation 문서](https://docs.ultralytics.com/tasks/depth/)를 2026-08-04 기준으로 번역·재구성했습니다.

> 이 저장소의 설명은 학습을 위한 비공식 한글 번역·해설입니다. 최신 옵션과 수치는 항상 공식 문서를 우선하세요. 원문을 통째로 복제하지 않고 핵심 개념과 사용법을 실습 순서에 맞게 재구성했습니다.

## 학습 목표

이 과정을 마치면 다음을 할 수 있습니다.

1. 단안 깊이 추정의 출력과 한계를 설명한다.
2. `yolo26n-depth.pt`로 이미지의 미터 단위 깊이 맵을 얻는다.
3. 원본 깊이 배열, 색상 맵, 오버레이를 저장하고 특정 픽셀의 거리를 읽는다.
4. 사용자 데이터셋의 RGB 이미지와 `.npy` 깊이 파일을 올바르게 짝짓는다.
5. 학습, 검증, 스케일 보정, ONNX 내보내기 명령을 구분해 사용한다.

## 폴더 구성

```text
ultralytics-depth-estimation-ko/
├── README.md
├── requirements.txt
├── docs/
│   ├── 01_공식문서_한글해설.md
│   └── 02_실습_가이드.md
├── notebooks/
│   └── 01_depth_estimation_hands_on.ipynb
└── scripts/
    ├── check_environment.py
    ├── predict_depth.py
    └── validate_depth_dataset.py
```

## 10분 빠른 시작

PowerShell에서 이 폴더로 이동한 뒤 실행합니다.

```powershell
cd D:\workspace\machine_learning\ultralytics-depth-estimation-ko
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/predict_depth.py --source "https://ultralytics.com/images/bus.jpg"
```

첫 실행 때 모델 가중치가 자동 다운로드됩니다. 기본 결과는 `outputs/`에 저장됩니다.

```text
outputs/
├── depth_raw.npy       # 계산에 사용하는 float32 미터 값
├── depth_colored.png   # 가까운 곳이 따뜻한 색인 상대 깊이 시각화
├── depth_metric.png    # 0~20 m 고정 범위 시각화
└── depth_overlay.png   # 원본 이미지와 깊이 맵의 합성 결과
```

Jupyter 실습을 선호하면 다음을 실행합니다.

```powershell
jupyter lab notebooks/01_depth_estimation_hands_on.ipynb
```

## 권장 학습 순서

1. [공식 문서 한글 해설](docs/01_공식문서_한글해설.md)
2. [단계별 실습 가이드](docs/02_실습_가이드.md)
3. [실습 노트북](notebooks/01_depth_estimation_hands_on.ipynb)
4. `scripts/predict_depth.py`에 자신의 사진 전달
5. 사용자 데이터가 있다면 `scripts/validate_depth_dataset.py`로 구조 검사

## 실행 환경 메모

- Python 3.10~3.13을 권장합니다.
- CPU에서도 실행되지만 GPU보다 느립니다. `--device 0`은 첫 CUDA GPU, `--device cpu`는 CPU입니다.
- 공식 가중치는 768×768로 학습되었으므로 정확도를 중시하면 기본 `--imgsz 768`을 유지하세요.
- `ultralytics`는 AGPL-3.0 라이선스를 사용합니다. 제품·서비스에 적용하기 전 [Ultralytics 라이선스 안내](https://www.ultralytics.com/license)를 확인하세요.

## 출처

- [Ultralytics 단안 깊이 추정 공식 문서](https://docs.ultralytics.com/tasks/depth/)
- [Ultralytics 깊이 데이터셋 안내](https://docs.ultralytics.com/datasets/depth/)
- [Ultralytics Python 패키지](https://pypi.org/project/ultralytics/)
