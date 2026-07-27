# 03. Hugging Face Transformers + TRL + PEFT

## 언제 선택하나

가장 범용적인 기준 트랙입니다. Transformers가 모델을, Datasets가 데이터를, TRL의 `SFTTrainer`가 SFT를, PEFT가 LoRA를, Accelerate가 장치 실행을 담당합니다. 도구가 나뉜 만큼 각 경계를 이해해야 디버깅할 수 있습니다.

공식 출발점:

- [Hugging Face Gemma PEFT 가이드](https://huggingface.co/blog/gemma-peft)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/main/sft_trainer)
- [TRL PEFT 통합](https://huggingface.co/docs/trl/main/peft_integration)
- [Transformers PEFT](https://huggingface.co/docs/transformers/peft)

## 1단계: 환경

```bash
python -m venv .venv-hf
source .venv-hf/bin/activate
pip install -U pip
pip install -U torch transformers datasets accelerate trl peft
```

QLoRA:

```bash
pip install -U bitsandbytes
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers, trl, peft; print(transformers.__version__, trl.__version__, peft.__version__)"
```

성공한 버전을 즉시 저장합니다.

```bash
pip freeze > outputs/hf-environment.txt
```

## 2단계: 제공 스크립트 보기

```bash
python gemma/labs/hf_sft.py --help
```

스크립트는 다음을 수행합니다.

1. JSONL 로드와 train/eval 분리
2. Gemma 토크나이저와 모델 로드
3. 선택적으로 4비트 QLoRA
4. LoRA 어댑터 삽입
5. completion-only SFT
6. 평가와 어댑터 저장

먼저 코드를 읽고 `model`, `dataset`, `peft_config`, `SFTConfig`의 책임을 구분하세요.

## 3단계: 20-step LoRA

```bash
python gemma/labs/hf_sft.py \
  --model-id google/gemma-3-1b-it \
  --train-file gemma/labs/data/train.jsonl \
  --eval-file gemma/labs/data/eval.jsonl \
  --output-dir outputs/hf-gemma3-1b-lora-smoke \
  --max-steps 20 \
  --max-length 256 \
  --lora-r 8
```

CPU 실행은 학습 시간이 매우 길 수 있습니다. 스크립트는 GPU가 없으면 명시적으로 중단하도록 설계되어 있습니다.

확인:

- trainable parameter 비율
- 첫 loss와 마지막 loss
- eval loss
- peak VRAM
- `adapter_config.json`, adapter weight 존재

## 4단계: QLoRA

```bash
python gemma/labs/hf_sft.py \
  --model-id google/gemma-3-1b-it \
  --train-file gemma/labs/data/train.jsonl \
  --eval-file gemma/labs/data/eval.jsonl \
  --output-dir outputs/hf-gemma3-1b-qlora-smoke \
  --max-steps 20 \
  --max-length 256 \
  --lora-r 8 \
  --load-in-4bit
```

비교표:

| 항목 | LoRA BF16 | QLoRA 4-bit |
|---|---:|---:|
| peak allocated VRAM |  |  |
| step/s |  |  |
| final eval loss |  |  |
| adapter size |  |  |
| 고정 평가 점수 |  |  |

4비트가 언제나 빠른 것은 아닙니다. 양자화·dequantize 연산과 GPU 커널에 따라 달라집니다.

## 5단계: 데이터와 loss 영역 확인

공통 `messages`의 마지막 assistant를 `prompt`와 `completion`으로 분리합니다. 최신 TRL은 prompt-completion 데이터에서 completion 토큰만 loss에 포함할 수 있습니다.

검증 과제:

1. Trainer가 전처리한 한 배치를 가져옵니다.
2. `labels == -100` 위치를 출력합니다.
3. user 토큰과 padding이 마스킹됐는지 확인합니다.
4. assistant 토큰 중 실제 학습되는 비율을 계산합니다.

Gemma 채팅 템플릿이 `assistant_only_loss=True`용 generation mask를 제공한다고 가정하지 마세요. prompt-completion 형식과 실제 label 검사를 사용합니다.

## 6단계: LoRA target 비교

PEFT는 Gemma 같은 일반 아키텍처의 기본 target을 알 수 있지만, 실험 목적을 분명히 하려면 직접 기록합니다.

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

실험:

- A: attention의 `q_proj`, `v_proj`
- B: attention의 q/k/v/o
- C: attention + MLP 전체

각 실험에서 학습 파라미터, VRAM, 처리량, 평가값을 비교합니다.

## 7단계: 체크포인트 재개

학습 중간 체크포인트가 생기도록 `save_steps`를 줄입니다.

```text
save_steps=10
save_total_limit=2
```

의도적으로 10 step에서 중단한 뒤:

```python
trainer.train(resume_from_checkpoint="outputs/.../checkpoint-10")
```

다음을 확인합니다.

- global step이 이어지는가?
- optimizer/scheduler가 복원되는가?
- `save_only_model`로 저장한 결과는 왜 엄밀한 재개가 안 되는가?

## 8단계: 어댑터 추론과 병합

어댑터 로드:

```python
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "outputs/hf-gemma3-1b-lora-smoke",
    device_map="auto",
)
```

병합:

```python
merged = model.merge_and_unload()
merged.save_pretrained(
    "outputs/hf-gemma3-1b-merged",
    safe_serialization=True,
)
```

토크나이저도 함께 저장합니다. 병합 전/후 greedy 출력과 logits 차이를 작은 허용오차로 비교합니다. QLoRA의 양자화된 base에서 바로 병합하지 말고 비양자화 base를 다시 로드하는 경로를 검토합니다.

## 9단계: Accelerate 멀티 GPU

```bash
accelerate config
accelerate launch gemma/labs/hf_sft.py ...
```

데이터 병렬에서는 effective batch가 GPU 수만큼 변할 수 있습니다. 공정 비교를 위해 gradient accumulation을 조정합니다.

FSDP/DeepSpeed는 단일 GPU 실험이 정확히 재현된 뒤 도입합니다. “실행된다”보다 checkpoint 저장·재개·병합이 정확한지 먼저 검증합니다.

## 디버깅

- 401/403: 모델 약관과 HF 토큰
- `KeyError`/class 없음: Gemma 지원 Transformers 버전
- bitsandbytes 오류: OS, CUDA, GPU compute capability
- NaN: FP16 대신 BF16, 학습률, clipping, 데이터
- 응답 반복: 템플릿, EOS, 생성 설정
- loss가 너무 빨리 0: 중복·누수·label mask
- 저장 후 품질 급락: tokenizer/chat template 미저장 또는 잘못된 merge

## 전문가 확장

- seed 3개와 bootstrap 신뢰구간
- packing 전후 throughput/padding 비율
- FSDP와 DeepSpeed ZeRO-2/3 비교
- VLM Gemma 3에서 image token truncation 방지
- DPO/ORPO는 SFT 기준선이 통과한 뒤 적용
- Hub model card에 base model, dataset, license, eval, limitations 기록

## 완료 기준

- [ ] LoRA와 QLoRA 스모크 테스트를 비교했다.
- [ ] completion-only label mask를 확인했다.
- [ ] checkpoint 재개를 시험했다.
- [ ] 어댑터와 병합 모델 모두 추론했다.
- [ ] 고정 평가 세트로 튜닝 전후를 비교했다.

