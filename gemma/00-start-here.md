# 00. 학습 환경과 첫 실행

## 목표

이 장을 마치면 “학습을 시작하기 전에 실패할 문제”를 대부분 제거할 수 있습니다. 아직 모델을 튜닝하지 않습니다. GPU, 계정, 토큰, 채팅 템플릿, 기준 출력을 확인하는 것이 목표입니다.

## 1단계: 실행 환경 선택

### 무료로 개념부터

- 데이터 검증과 평가 스크립트: CPU만으로 실행
- 모델 추론·짧은 LoRA: Google Colab의 GPU 사용
- Keras 분산: Kaggle 또는 Colab TPU 사용 가능 여부 확인

### 로컬 학습

- Windows 네이티브보다 WSL2 Ubuntu를 권장합니다.
- NVIDIA 드라이버가 WSL에서 보이는지 `nvidia-smi`로 확인합니다.
- Python 3.11 또는 3.12의 별도 가상환경을 프레임워크마다 만듭니다.
- 디스크 여유 공간을 모델 크기의 최소 2~3배 확보합니다. 원본, 캐시, 어댑터, 병합본이 동시에 생길 수 있습니다.

### 클라우드 학습

- GPU 종류와 수량뿐 아니라 리전 할당량을 먼저 확인합니다.
- GKE/Vertex 실습은 과금됩니다. 예산 알림을 만든 뒤 시작하고 종료 절차까지 읽은 후 실행하세요.

## 2단계: 저장소와 공통 실습 확인

저장소 루트에서 다음을 실행합니다.

```powershell
python gemma/labs/validate_dataset.py
python gemma/labs/evaluate_outputs.py --demo
```

Linux/WSL에서는 같은 명령을 `/` 경로 표기로 실행하면 됩니다.

기대 결과:

- 데이터 구조 오류 0개
- 중복 질문 0개
- 평가 데모에 exact match, keyword recall, 형식 준수율이 출력됨

이 단계는 GPU나 외부 패키지가 필요 없습니다.

## 3단계: 계정과 모델 접근

### Hugging Face

1. Hugging Face 계정을 만듭니다.
2. `google/gemma-3-1b-it` 모델 페이지에서 Gemma 이용약관에 동의합니다.
3. 읽기 권한 토큰을 발급합니다.
4. 토큰은 코드나 Markdown에 쓰지 말고 환경 변수 또는 비밀 저장소에 둡니다.

```bash
pip install -U huggingface_hub
huggingface-cli login
```

자동화 환경에서는 `HF_TOKEN`을 비밀 변수로 주입합니다.

### Kaggle

Keras 공식 예제는 Kaggle 모델을 사용하기도 합니다.

1. Kaggle에서 Gemma 접근을 승인받습니다.
2. Kaggle 설정에서 API 토큰을 만듭니다.
3. `KAGGLE_USERNAME`, `KAGGLE_KEY`를 환경 변수 또는 Colab Secrets에 저장합니다.

비밀 파일을 저장소에 복사하지 마세요. 최상단 `.gitignore`는 `.env`를 제외하지만, 커밋 전 `git diff --cached`로 다시 확인해야 합니다.

## 4단계: 프레임워크별 환경을 분리

LLM 튜닝 패키지는 PyTorch, CUDA, JAX, Triton 버전 요구가 서로 다릅니다. 한 환경에 모두 설치하지 않습니다.

```bash
python -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install -U pip
```

PowerShell:

```powershell
py -3.12 -m venv .venv-hf
.\.venv-hf\Scripts\Activate.ps1
python -m pip install -U pip
```

추천 환경 이름:

- `.venv-keras`
- `.venv-jax`
- `.venv-hf`
- `.venv-unsloth`
- `.venv-axolotl`
- `.venv-xtuner-legacy`

## 5단계: 하드웨어 상태 기록

아래 결과를 실험 노트에 붙여 넣습니다.

```bash
nvidia-smi
python -c "import platform; print(platform.platform()); print(platform.python_version())"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

JAX 환경:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

재현 보고서에는 최소한 다음을 기록합니다.

- OS, Python
- GPU/TPU 종류와 개수
- 드라이버, CUDA 런타임
- 핵심 패키지와 버전
- 모델 ID와 정확한 revision
- seed, 데이터 revision

## 6단계: 메모리 예산 잡기

파라미터가 `N`개이고 가중치가 `b`바이트라면 가중치만 대략 `N × b` 바이트입니다.

- FP32: 파라미터당 4바이트
- BF16/FP16: 2바이트
- INT8: 약 1바이트
- 4비트: 약 0.5바이트와 추가 메타데이터

학습은 가중치 외에 활성값, 그래디언트, 옵티마이저 상태가 필요합니다. 따라서 단순 가중치 크기보다 훨씬 큽니다. LoRA는 그래디언트·옵티마이저 상태를 크게 줄이고, QLoRA는 고정된 기본 가중치도 4비트로 줄입니다.

초기 안전 설정:

```text
model: Gemma 3 1B IT
sequence length: 256
micro batch: 1
gradient accumulation: 8
LoRA rank: 8
max steps: 20
evaluation samples: 10
```

스모크 테스트 성공 후 한 번에 하나씩 키웁니다.

## 7단계: 첫 추론과 기준 출력

Hugging Face 환경에서:

```bash
pip install -U "transformers>=4.50" torch accelerate
```

```python
import torch
from transformers import pipeline

model_id = "google/gemma-3-1b-it"
generator = pipeline(
    "text-generation",
    model=model_id,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

messages = [
    {"role": "user", "content": "반품 접수 절차를 세 단계로 설명해 주세요."}
]
result = generator(messages, max_new_tokens=128, do_sample=False)
print(result[0]["generated_text"])
```

실패 시 순서대로 확인합니다.

1. 모델 이용약관 동의 여부
2. 토큰에 read 권한이 있는지
3. `transformers`가 Gemma 3을 지원하는 버전인지
4. GPU dtype이 BF16을 지원하는지
5. OOM이면 CPU가 아니라 4비트 로드가 필요한지

## 8단계: 채팅 템플릿을 눈으로 확인

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
messages = [
    {"role": "user", "content": "배송 조회는 어디서 하나요?"},
    {"role": "assistant", "content": "주문 내역에서 배송 조회를 선택하세요."},
]

rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)
print(rendered)
```

확인할 점:

- 역할 순서가 `user → assistant`인지
- 시작/종료 토큰이 모델 템플릿에 맞는지
- 템플릿을 수동 적용한 뒤 Trainer가 또 적용하지 않는지
- 학습과 추론이 같은 템플릿을 쓰는지

## 통과 기준

- [ ] 공통 데이터 검증 스크립트가 성공한다.
- [ ] 모델 약관과 인증 설정을 완료했다.
- [ ] 프레임워크별 가상환경을 분리했다.
- [ ] 하드웨어와 패키지 버전을 기록했다.
- [ ] Gemma 기준 출력 5개 이상을 저장했다.
- [ ] 렌더링된 채팅 템플릿을 직접 확인했다.

다음: [LLM·SFT·LoRA 기초](./01-foundations.md)

