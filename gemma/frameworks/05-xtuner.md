# 05. XTuner: 보관 예제와 최신 V1 구분하기

## 먼저 알아야 할 상태

Google Gemma 튜닝 페이지의 XTuner 링크는 `google-gemma/cookbook/.archive/Gemma/[Gemma_1]...`에 있는 과거 Gemma 1 notebook입니다. 한편 XTuner는 2025년 V1에서 초대형 MoE 학습 엔진 중심으로 크게 개편됐습니다. 현재 V1 지원표에 원하는 Gemma 버전이 명시되지 않았다면 “예전에 Gemma가 됐으니 현재도 된다”고 가정하면 안 됩니다.

공식 출발점:

- [Google의 보관 Gemma 1 + XTuner notebook](https://github.com/google-gemma/cookbook/blob/main/.archive/Gemma/%5BGemma_1%5DFinetune_with_XTuner.ipynb)
- [XTuner GitHub](https://github.com/InternLM/xtuner)
- [XTuner 문서](https://xtuner.readthedocs.io/en/latest/)

이 장의 목표는 오래된 예제를 무리하게 최신 환경에 이식하는 것이 아니라, 버전을 식별하고 재현하거나 안전하게 마이그레이션하는 능력을 기르는 것입니다.

## 트랙 A: 보관 예제 재현

### 1단계: 일회용 환경 사용

로컬 기본 환경에 설치하지 말고 Colab 또는 별도 `.venv-xtuner-legacy`를 사용합니다.

```bash
python -m venv .venv-xtuner-legacy
source .venv-xtuner-legacy/bin/activate
```

### 2단계: notebook의 설치 셀 기록

실행 전 다음을 표로 옮깁니다.

| 항목 | notebook 값 |
|---|---|
| 작성/수정 시점 |  |
| XTuner 버전/revision |  |
| Transformers 버전 |  |
| 모델 |  |
| 데이터 |  |
| LoRA/QLoRA |  |
| CUDA/PyTorch |  |

버전이 명시되지 않았다면 현재 최신을 설치하지 않습니다. notebook commit 시점의 lock/release를 조사해 재현 가능한 revision을 정합니다.

### 3단계: 셀을 네 구간으로 실행

1. 환경 설치
2. 모델 접근과 기준 추론
3. 데이터/설정 preview
4. 10~20 step 학습과 추론

각 구간 후 런타임 상태를 저장합니다.

```bash
pip freeze > xtuner-legacy-freeze.txt
```

### 4단계: 보관 예제의 성공 정의

- notebook 원본 모델과 데이터로 최소 step 학습
- adapter/checkpoint 생성
- 원문과 같은 추론 경로 실행
- 환경 revision과 변경점 기록

Gemma 3/4로 모델만 바꿔 성공하는 것은 이 단계의 목표가 아닙니다.

## 트랙 B: 최신 XTuner V1 평가

### 1단계: 지원표 확인

최신 README와 모델 문서에서 다음을 확인합니다.

- Gemma의 정확한 버전/크기가 지원되는가?
- SFT가 지원되는가?
- LoRA/QLoRA가 지원되는가?
- 원하는 GPU/CUDA가 검증됐는가?
- 저장 형식이 기존 생태계와 호환되는가?

하나라도 불명확하면 프로덕션 후보에서 제외하고 작은 기술 검증만 합니다.

### 2단계: 현재 CLI 확인

XTuner V1 공식 문서의 학습 진입점 예:

```bash
python xtuner/v1/train/cli/sft.py --help
```

설정 파일 실행:

```bash
python xtuner/v1/train/cli/sft.py --config path/to/config.py
```

과거의 `xtuner train ...` 명령과 최신 V1 명령을 섞지 않습니다.

### 3단계: 설정 구조 읽기

최신 문서의 `TrainerConfig`는 모델, tokenizer, checkpoint, optimizer, dataloader, learning rate, work directory를 명시합니다.

```python
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.train import TrainerConfig

optim_cfg = AdamWConfig(lr=1e-4)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
```

Gemma 지원이 확인된 경우에만 해당 버전의 `model_cfg`와 dataloader를 공식 예제에 따라 추가합니다. 다른 모델 설정의 class name을 Gemma로 추측해 바꾸지 마세요.

## 마이그레이션 실습: XTuner 보관 예제 → Hugging Face

현실적인 전문가 과제입니다.

1. 보관 notebook에서 다음을 추출합니다.
   - model ID
   - prompt template
   - dataset mapping
   - max length
   - LoRA target/rank/alpha
   - optimizer/lr/scheduler
   - effective batch
2. 같은 값을 `labs/hf_sft.py` 인자로 옮깁니다.
3. seed와 데이터 순서를 고정합니다.
4. trainable parameter 수를 비교합니다.
5. 같은 20 step 후 loss와 고정 평가 출력을 비교합니다.
6. 차이를 “프레임워크”, “커널/dtype”, “데이터 전처리”, “버전”으로 분류합니다.

## 의사결정 기준

XTuner를 계속 사용할 조건:

- 조직에 기존 XTuner 운영 경험이 있음
- 원하는 Gemma 모델이 현재 지원표에 있음
- 대규모 학습 기능이 실제 요구사항과 맞음
- checkpoint와 배포 경로를 검증함

다른 도구를 선택할 조건:

- 단일 GPU Gemma LoRA가 목적
- 최신 Gemma 지원이 불명확
- 보관 notebook 외에 유지되는 예제가 없음
- 빠른 결과가 중요

이 경우 Hugging Face, Unsloth, LLaMA-Factory, Axolotl이 더 단순할 수 있습니다.

## 디버깅

- 인터넷의 과거 config를 최신 V1에서 실행하고 있지 않은가?
- 문서 URL의 `stable`, `latest`, `docs` 버전이 같은가?
- model class가 현재 engine에 등록되어 있는가?
- conversion/merge 스크립트가 checkpoint 버전과 맞는가?
- “지원”이 full training만 의미하고 LoRA는 미지원인 것은 아닌가?

## 완료 기준

- [ ] Google 링크가 보관 예제임을 확인했다.
- [ ] legacy와 V1 환경을 분리했다.
- [ ] legacy notebook의 핵심 설정을 표로 추출했다.
- [ ] 현재 지원표에서 원하는 Gemma 모델을 확인했다.
- [ ] 동일 설정을 유지되는 프레임워크로 마이그레이션했다.

