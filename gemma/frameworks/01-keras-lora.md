# 01. Keras + KerasHub로 LoRA 튜닝

## 언제 선택하나

Keras는 모델 로드, LoRA 활성화, compile, fit을 익숙한 딥러닝 흐름으로 보여 줍니다. 처음으로 “학습 가능한 파라미터가 어떻게 줄어드는지” 확인하기 좋습니다. Keras 3은 TensorFlow, JAX, PyTorch 백엔드를 지원하지만 모델·연산별 지원 차이가 있으므로 이 실습은 JAX를 기본으로 합니다.

공식 출발점:

- [Google Keras LoRA 가이드](https://ai.google.dev/gemma/docs/core/lora_tuning?hl=ko)
- [KerasHub Gemma 3 API](https://keras.io/keras_hub/api/models/gemma3/)
- [Keras LoRA/QLoRA 예제](https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/)

> 이름 변경 주의: Google 원문 일부는 `keras_nlp`를 사용합니다. 현재 학습에서는 후속 패키지인 `keras_hub`를 우선 사용합니다. 두 패키지의 import를 한 노트북에서 섞지 마세요.

## 실습 1: 환경과 백엔드

```bash
python -m venv .venv-keras
source .venv-keras/bin/activate
pip install -U pip keras keras-hub jax
```

NVIDIA GPU용 JAX 설치는 CUDA 버전에 따라 달라집니다. [JAX 설치 문서](https://docs.jax.dev/en/latest/installation.html)에서 현재 명령을 확인하세요.

백엔드는 Keras를 import하기 전에 설정합니다.

```python
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import keras
import keras_hub

print(keras.backend.backend())
```

성공 기준: `jax`가 출력되고 `jax.devices()`에 원하는 가속기가 보입니다.

## 실습 2: 모델 로드와 기준 생성

```python
model = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-1b-it"
)
model.summary()

prompt = (
    "<start_of_turn>user\n"
    "반품 절차를 세 단계로 알려 주세요."
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)
print(model.generate(prompt, max_length=160))
```

모델 페이지 접근 승인이 필요합니다. Hugging Face 경로가 환경에서 동작하지 않으면 KerasHub의 현재 preset 목록에서 Gemma 3 1B preset을 선택합니다.

기준 출력, 생성 파라미터, 모델 preset을 실험 노트에 저장합니다.

## 실습 3: 학습 문자열 만들기

공통 JSONL을 Keras가 받을 문자열로 변환합니다.

```python
import json
from pathlib import Path

def to_gemma_text(messages):
    pieces = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        pieces.append(
            f"<start_of_turn>{role}\n"
            f"{message['content']}<end_of_turn>\n"
        )
    return "".join(pieces)

rows = [
    json.loads(line)
    for line in Path("gemma/labs/data/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
]
train_texts = [to_gemma_text(row["messages"]) for row in rows]
print(train_texts[0])
```

검사:

- assistant 역할이 Gemma 템플릿의 `model`로 변환됐는가?
- 빈 응답이 없는가?
- 같은 템플릿을 전처리기가 다시 추가하지 않는가?

수동 템플릿 대신 KerasHub 전처리 API를 사용할 수 있다면 그 경로를 우선하고, 렌더링 결과가 같은지 비교하세요.

## 실습 4: LoRA 활성화

```python
before = sum(v.size for v in model.trainable_weights)
model.backbone.enable_lora(rank=8)
after = sum(v.size for v in model.trainable_weights)

print(f"trainable before: {before:,}")
print(f"trainable after : {after:,}")
print(f"ratio           : {after / before:.6%}")
model.summary()
```

관찰 과제:

1. rank 4, 8, 16에서 학습 파라미터 수를 표로 기록합니다.
2. rank가 2배일 때 파라미터가 정확히 2배인지 확인합니다.
3. 어떤 계층에 LoRA 변수가 생겼는지 이름을 출력합니다.

```python
for variable in model.trainable_weights[:20]:
    print(variable.path, variable.shape)
```

## 실습 5: 20-step 스모크 학습

```python
model.preprocessor.sequence_length = 256

optimizer = keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
)
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    train_texts,
    batch_size=1,
    epochs=1,
)
```

공통 샘플은 매우 작으므로 품질 학습이 아니라 파이프라인 확인용입니다. 실제 학습에서는 validation과 checkpoint callback을 추가합니다.

## 실습 6: 전후 비교

학습 전과 같은 prompt, sampler, 길이를 사용합니다.

```python
model.compile(sampler="greedy")
print(model.generate(prompt, max_length=160))
```

다음 표를 채웁니다.

| 항목 | 튜닝 전 | 튜닝 후 |
|---|---|---|
| 필수 절차 포함 |  |  |
| 모르는 주문 상태 단정 |  |  |
| 답변 문장 수 |  |  |
| 생성 시간 |  |  |

## 실습 7: QLoRA 비교

Keras 공식 예제는 모델 양자화 후 LoRA를 활성화하는 순서를 사용합니다.

```python
quant_model = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-1b-it"
)
quant_model.quantize("int8")
quant_model.backbone.enable_lora(rank=8)
quant_model.summary()
```

지원되는 양자화 형식은 백엔드와 모델에 따라 다릅니다. “QLoRA”라는 이름만 보고 Hugging Face bitsandbytes의 NF4 구현과 완전히 같다고 가정하지 마세요. Keras 공식 예제는 지원 형식과 구현 차이를 명시합니다.

## 실습 8: 저장과 다시 로드

```python
model.save_to_preset("outputs/keras-gemma3-lora")
```

저장 후 새 프로세스에서 불러와 같은 고정 프롬프트를 생성합니다. 디스크에 기본 모델 전체가 저장됐는지, LoRA 변수만 저장됐는지 파일 목록과 크기로 확인합니다. 다른 생태계로 옮길 때는 원하는 출력 형식(Safetensors, Keras preset 등)을 먼저 정하고 변환 지원을 확인하세요.

## 디버깅

- import 전에 `KERAS_BACKEND`를 설정했는가?
- `keras_nlp` 예제와 `keras_hub` API를 섞지 않았는가?
- preset이 해당 모델 클래스를 지원하는가?
- JAX가 GPU/TPU가 아닌 CPU를 보고 있지 않은가?
- XLA 선할당 때문에 OOM이면 메모리 비율을 낮췄는가?
- 첫 step의 JIT 컴파일 시간을 정상 반복 시간과 구분했는가?

## 전문가 확장

1. rank와 학습률의 3×3 실험을 seed 3개로 반복합니다.
2. 모든 토큰 loss와 assistant-only loss를 비교합니다.
3. JAX/PyTorch 백엔드에서 같은 seed의 차이를 분석합니다.
4. Keras 분산 장의 DeviceMesh로 같은 코드를 확장합니다.
5. preset 저장물을 HF Safetensors로 내보낼 수 있는지 왕복 검증합니다.

## 완료 기준

- [ ] 백엔드와 장치를 확인했다.
- [ ] 튜닝 전 기준 출력을 저장했다.
- [ ] LoRA 전후 학습 파라미터 수를 비교했다.
- [ ] 20-step 이하 스모크 학습을 완료했다.
- [ ] 새 프로세스에서 저장 결과를 다시 로드했다.

