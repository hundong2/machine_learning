# 06. Unsloth 저메모리 QLoRA

## 언제 선택하나

Unsloth는 단일 GPU와 Colab에서 LoRA/QLoRA를 빠르고 적은 메모리로 실행하는 데 초점을 둡니다. 모델별 notebook이 잘 준비되어 있어 첫 성공이 빠르지만, 최적화 patch와 빠르게 변하는 패키지 조합 때문에 성공한 환경을 잠그는 습관이 중요합니다.

공식 출발점:

- [Gemma 3 실행 및 튜닝](https://docs.unsloth.ai/basics/tutorial-how-to-run-and-fine-tune-gemma-3)
- [Unsloth 튜토리얼](https://docs.unsloth.ai/basics/tutorial)
- [LoRA 하이퍼파라미터](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia)
- [공식 notebooks](https://github.com/unslothai/notebooks)

Google Gemma 튜닝 페이지의 링크는 Gemma 2 보관 notebook입니다. 최신 Gemma는 Unsloth의 현재 모델별 notebook을 사용하세요.

## 1단계: 환경 선택

가장 쉬운 경로는 공식 Gemma 3 Colab notebook을 복사해 자신의 Drive에 저장하는 것입니다. 로컬:

```bash
python -m venv .venv-unsloth
source .venv-unsloth/bin/activate
pip install -U pip
pip install unsloth
```

Unsloth 문서의 현재 Python/CUDA/PyTorch 호환표를 먼저 봅니다. 설치 후:

```bash
python -c "import unsloth, torch; print(torch.__version__, torch.version.cuda)"
pip freeze > outputs/unsloth-environment.txt
```

## 2단계: 4비트 모델 로드

Gemma 3 텍스트 1B의 개념 예:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=512,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)
```

설치 버전의 공식 notebook이 `FastLanguageModel` 또는 `FastModel` 중 무엇을 쓰는지 그대로 따릅니다. Gemma 3 4B 이상은 비전 processor를 반환할 수 있으므로 1B 텍스트 모델 코드를 단순 복사하지 않습니다.

## 3단계: LoRA 삽입

```python
model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()
```

`r`, alpha, target modules를 실험 기록에 남깁니다. “Unsloth 기본값”도 버전이 바뀔 수 있으므로 resolved 값을 저장합니다.

## 4단계: 데이터 렌더링

공통 JSONL의 `messages`를 tokenizer의 chat template로 변환합니다.

```python
def render(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }
```

첫 5개 렌더링을 출력합니다. Gemma는 BOS를 자동 추가하는 경로가 있으므로 수동 BOS와 tokenizer 자동 BOS가 중복되지 않는지 확인합니다.

## 5단계: SFTTrainer

Unsloth는 TRL의 `SFTTrainer`와 통합됩니다.

```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=20,
        warmup_ratio=0.05,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs/unsloth-gemma3-smoke",
        report_to="none",
    ),
)
stats = trainer.train()
```

API 이름은 TRL/Unsloth 버전별로 달라질 수 있습니다. 공식 notebook의 현재 인자를 우선합니다.

## 6단계: VRAM 측정

```python
import torch

torch.cuda.reset_peak_memory_stats()
trainer.train()
peak_gib = torch.cuda.max_memory_allocated() / 1024**3
print(f"peak allocated: {peak_gib:.2f} GiB")
```

HF QLoRA와 공정 비교:

- 같은 모델 revision
- 같은 max length
- 같은 micro/effective batch
- 같은 steps와 데이터 순서
- 같은 target modules/rank
- 첫 compile/warmup 제외

## 7단계: 저장

어댑터:

```python
model.save_pretrained("outputs/unsloth-gemma3-adapter")
tokenizer.save_pretrained("outputs/unsloth-gemma3-adapter")
```

병합 또는 GGUF:

```python
model.save_pretrained_merged(
    "outputs/unsloth-gemma3-merged",
    tokenizer,
    save_method="merged_16bit",
)
```

GGUF export 옵션은 설치 버전의 공식 문서를 확인합니다. export 성공만 보지 말고 대상 런타임에서 같은 chat template과 EOS를 사용해 출력 품질을 검증합니다.

## 8단계: Gemma 3 VLM 확장

4B 이상 멀티모달 모델은 processor, image collator, 이미지 토큰을 사용합니다.

- text-only dataset과 VLM dataset을 혼동하지 않기
- `max_length` 절단으로 image token을 잃지 않기
- vision tower를 freeze할지 명시
- image resolution과 batch가 VRAM에 미치는 영향 측정
- 텍스트 1B 실습이 성공한 뒤 공식 Vision notebook으로 이동

## 디버깅

- notebook 그대로인데 오류: 패키지 업데이트 후 runtime 재시작 여부
- processor에 `encode` 없음: VLM processor 내부 tokenizer 구분
- T4에서 overflow/NaN: Gemma 3용 Unsloth 최신 patch와 dtype 확인
- export 후 반복 출력: chat template, BOS/EOS 불일치
- save 중 OOM: 저장 메모리 상한 옵션과 CPU merge 검토
- Colab 재연결 후 파일 소실: Drive 또는 Hub에 명시적으로 저장

## 전문가 확장

1. LoRA/QLoRA의 품질·속도·메모리를 seed 3개로 비교
2. Unsloth gradient checkpointing과 기본 checkpointing 비교
3. 16bit merge와 GGUF 양자화별 회귀 평가
4. HF vanilla pipeline과 logits/출력 비교
5. 모델별 patch를 제거했을 때 정확성 차이가 아니라 실행 안정성 차이 분석

## 완료 기준

- [ ] 현재 Gemma 모델용 공식 notebook을 사용했다.
- [ ] 4비트 모델과 LoRA target을 확인했다.
- [ ] 렌더링된 chat template을 검사했다.
- [ ] 피크 VRAM과 처리량을 측정했다.
- [ ] adapter 및 한 가지 배포 형식을 다시 로드했다.

