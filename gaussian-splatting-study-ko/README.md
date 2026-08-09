# 3D Gaussian Splatting: 원리부터 실무까지

3D Gaussian Splatting(3DGS)을 처음 접하는 사람이 Python·NumPy 기초부터 수식, 렌더링, 학습, 실제 촬영 데이터 처리, 배포와 최신 연구까지 순서대로 학습하도록 만든 한글 과정입니다.

자료 기준일은 **2026-08-09**입니다. 원 논문, 공식 구현, 학회 공개 논문과 공식 문서를 우선 출처로 사용했습니다.

## 이 폴더에서 배우는 것

```text
Python/NumPy 기초
  → 1D·2D Gaussian 이해
  → 3D Gaussian의 위치·크기·회전·색·불투명도
  → 카메라 좌표와 3D→2D 투영
  → 깊이 정렬과 alpha compositing
  → gradient 학습과 densification
  → 직접 촬영·COLMAP·Splatfacto
  → 품질 평가·압축·웹/XR 배포
  → 2023~2026 연구 흐름
```

## 권장 학습 순서

| 단계 | 자료 | 결과 |
|---:|---|---|
| 0 | [학습 로드맵](docs/00_학습_로드맵.md) | 환경과 전체 구조 이해 |
| 1 | [Python·NumPy 기초](docs/01_Python_NumPy_기초.md) | 실습 코드 문법 이해 |
| 2 | [수학 기초](docs/02_수학_기초.md) | Gaussian·행렬·카메라 이해 |
| 3 | [3DGS 핵심 원리](docs/03_3DGS_핵심_원리.md) | 전체 알고리즘 설명 가능 |
| 4 | [직접 구현 실습](docs/04_직접_구현_실습.md) | CPU 미니 렌더러 실행 |
| 5 | [실제 프로젝트](docs/05_실무_프로젝트.md) | 촬영부터 PLY 내보내기 |
| 6 | [운영·최적화](docs/06_운영과_최적화.md) | 품질·속도·메모리 관리 |
| 7 | [최신 연구 방향](docs/07_최신_연구_방향.md) | 연구 지형과 다음 학습 선택 |
| 상시 | [용어·단축어 사전](docs/08_용어와_단축어.md) | 약어를 빠짐없이 복습 |

## 바로 실행하기

PowerShell에서 다음을 실행합니다.

```powershell
cd D:\workspace\machine_learning\gaussian-splatting-study-ko
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/01_gaussian_1d.py
python scripts/02_gaussian_2d.py
python scripts/03_camera_projection.py
python scripts/04_mini_splat_renderer.py
pytest -q
```

모든 입문 실습은 CPU에서 실행됩니다. 실제 3DGS 학습은 CUDA GPU가 사실상 필요하며, 별도 설치가 필요한 Nerfstudio 또는 원 공식 구현의 명령은 [실무 프로젝트](docs/05_실무_프로젝트.md)에 분리했습니다.

## 코드 주석 원칙

- 실행되는 각 코드 줄 바로 위에 그 줄의 목적을 설명합니다.
- 변수 이름에 약어가 있으면 첫 등장 시 전체 영어와 한글 뜻을 씁니다.
- `@`, `**`, `[:, None]`, `np.newaxis` 같은 축약 문법은 코드 안과 기초 문서에서 모두 풀이합니다.
- 실제 CUDA 래스터라이저를 숨긴 채 “마법처럼” 호출하지 않고, CPU 미니 구현으로 투영·정렬·합성을 먼저 확인합니다.
- 교육용 미니 구현과 실무용 고속 구현의 차이를 명시합니다.

## 핵심 출처

- [3D Gaussian Splatting 원 논문 및 프로젝트](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Graphdeco 공식 구현](https://github.com/graphdeco-inria/gaussian-splatting)
- [gsplat 공식 문서](https://docs.gsplat.studio/main/)
- [Nerfstudio Splatfacto 문서](https://docs.nerf.studio/nerfology/methods/splat.html)
- [COLMAP 공식 튜토리얼](https://colmap.github.io/tutorial.html)
- [Khronos KHR_gaussian_splatting 발표](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release)

## 중요한 현실적 한계

3DGS는 보이는 장면을 새 시점에서 사실적으로 재현하는 **radiance-field 표현**입니다. 정확한 CAD 메시, 충돌 판정용 표면, 숨겨진 뒷면까지 자동 복원해 주는 만능 3D 스캐너가 아닙니다. 반사·투명체, 움직이는 물체, 노출 변화, 흐린 사진, 부족한 시점 중첩에서는 품질이 크게 떨어질 수 있습니다.
