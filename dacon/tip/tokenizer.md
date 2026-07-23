보통 토크나이저는 모델과 함께 재사용해야 하는 핵심 학습 결과물로 취급합니다. BPE 토크나이저에는 단순 단어 목록뿐 아니라 다음 정보가 저장됩니다.

- 토큰과 ID의 대응 관계
- BPE 병합 규칙
- 정규화 규칙
- 공백·문장부호 분리 규칙
- `[PAD]`, `[UNK]` 같은 특수 토큰
- 후처리 및 디코딩 규칙

모델은 토큰 자체가 아니라 토큰 ID를 학습하므로, 모델 학습 때 사용한 토크나이저를 추론 때도 그대로 사용해야 합니다.

## 1. 가장 간단한 방법: JSON 파일 하나로 저장

`tokenizers` 라이브러리만 사용하는 경우입니다.

### 저장

```python
# 학습 완료 후 저장
tokenizer.save(
    "my_bpe_tokenizer.json"
)
```

폴더를 구분하고 싶다면 다음과 같이 저장합니다.

```python
from pathlib import Path

save_dir = Path("artifacts/tokenizer")
save_dir.mkdir(
    parents=True,
    exist_ok=True
)

tokenizer.save(
    str(save_dir / "tokenizer.json")
)
```

`tokenizer.json` 하나에 vocabulary, BPE 병합 규칙, 전처리 설정 등이 직렬화됩니다. 이는 Hugging Face Tokenizers가 공식적으로 제공하는 저장 형식입니다. [Tokenizer 저장·불러오기 API](https://huggingface.co/docs/tokenizers/main/api/tokenizer)

### 불러오기

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "artifacts/tokenizer/tokenizer.json"
)
```

### 사용하기

```python
text = "It is a truth universally acknowledged."

encoded = tokenizer.encode(text)

print("토큰:", encoded.tokens)
print("토큰 ID:", encoded.ids)
```

출력 예시는 다음과 같은 형태입니다.

```text
토큰: ['It', 'is', 'a', 'truth', 'universally', 'acknowledged', '.']
토큰 ID: [351, 84, 23, 729, 1842, 2911, 17]
```

복원은 `decode()`로 합니다.

```python
decoded_text = tokenizer.decode(
    encoded.ids
)

print(decoded_text)
```

여러 문장을 처리할 때는 `encode_batch()`를 사용합니다.

```python
texts = [
    "This is the first sentence.",
    "This is another sentence."
]

encoded_batch = tokenizer.encode_batch(texts)

for encoded in encoded_batch:
    print(encoded.tokens)
    print(encoded.ids)
```

## 2. 실무에서 편리한 방법: Transformers 형식으로 저장

BERT, GPT 등의 모델이나 PyTorch `DataLoader`와 함께 사용한다면 `PreTrainedTokenizerFast`로 변환하는 것이 편리합니다.

먼저 설치합니다.

```bash
uv add transformers
```

### 기존 토크나이저 변환

```python
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]"
)
```

여기서 `tokenizer`는 앞에서 학습한 `tokenizers.Tokenizer` 객체입니다.

### 폴더로 저장

```python
fast_tokenizer.save_pretrained(
    "artifacts/my_bpe_tokenizer"
)
```

폴더 내부에는 보통 다음 파일들이 생성됩니다.

```text
artifacts/my_bpe_tokenizer/
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

파일 역할은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| `tokenizer.json` | vocabulary, BPE 규칙, 전처리 파이프라인 |
| `tokenizer_config.json` | 최대 길이 등의 토크나이저 설정 |
| `special_tokens_map.json` | `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]` 설정 |

Transformers 토크나이저는 이 폴더 전체를 `save_pretrained()`와 `from_pretrained()`로 저장·복원하도록 설계되어 있습니다. [Transformers Tokenizer 문서](https://huggingface.co/docs/transformers/main_classes/tokenizer)

### 다시 불러오기

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/my_bpe_tokenizer",
    use_fast=True
)
```

또는 클래스를 명시할 수 있습니다.

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "artifacts/my_bpe_tokenizer"
)
```

## 3. 모델에 입력할 형태로 변환

Transformers 형식의 장점은 문자열을 바로 PyTorch Tensor로 만들 수 있다는 것입니다.

```python
texts = [
    "This is the first sentence.",
    "This is a longer second sentence."
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
)

print(batch)
```

결과는 대략 다음 형태입니다.

```python
{
    "input_ids": tensor([
        [101, 251, 75, 912, 102,   0,   0],
        [101, 251, 75, 344, 817, 912, 102]
    ]),
    "attention_mask": tensor([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1]
    ])
}
```

각 항목의 의미는 다음과 같습니다.

- `input_ids`: 각 토큰을 정수 ID로 변환한 값
- `attention_mask=1`: 실제 토큰
- `attention_mask=0`: 길이를 맞추기 위해 추가한 패딩

GPU 모델에 전달하려면 다음처럼 이동합니다.

```python
batch = {
    key: value.to(device)
    for key, value in batch.items()
}

outputs = model(**batch)
```

## 4. 직접 만든 PyTorch 모델에서 사용

직접 Transformer나 RNN을 구현했다면 `input_ids`를 `nn.Embedding`에 입력합니다.

```python
import torch
import torch.nn as nn

vocab_size = len(tokenizer)
embedding_dim = 256

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=tokenizer.pad_token_id
)

batch = tokenizer(
    ["This is a sentence."],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

input_ids = batch["input_ids"]

embedded = embedding(input_ids)

print("input_ids 크기:", input_ids.shape)
print("embedding 크기:", embedded.shape)
```

예를 들어 입력 크기가 다음과 같다면,

```text
input_ids 크기: torch.Size([1, 6])
```

임베딩 결과는 다음과 같습니다.

```text
embedding 크기: torch.Size([1, 6, 256])
```

각 토큰 ID가 길이 256인 학습 가능한 벡터로 변환된 것입니다.

`vocab_size`에는 다음을 권장합니다.

```python
vocab_size = len(tokenizer)
```

`tokenizer.vocab_size`는 기본 vocabulary만 셀 수 있지만, `len(tokenizer)`는 나중에 추가한 특수 토큰까지 포함하기 때문입니다.

## 5. 모델과 토크나이저 함께 저장하기

실무에서는 보통 다음 구조로 관리합니다.

```text
artifacts/my_model/
├── model.pt
├── config.json
└── tokenizer/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

### 모델 저장

```python
import torch

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id
    },
    "artifacts/my_model/model.pt"
)

tokenizer.save_pretrained(
    "artifacts/my_model/tokenizer"
)
```

### 모델과 토크나이저 불러오기

```python
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/my_model/tokenizer"
)

checkpoint = torch.load(
    "artifacts/my_model/model.pt",
    map_location=device,
    weights_only=True
)

model = MyModel(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()
```

불러온 다음 설정이 일치하는지 검사하는 것도 좋습니다.

```python
assert len(tokenizer) == checkpoint["vocab_size"]
assert tokenizer.pad_token_id == checkpoint["pad_token_id"]
```

## 6. 추론 전체 예제

```python
from transformers import AutoTokenizer
import torch

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    "artifacts/my_model/tokenizer"
)

# 체크포인트 로드
checkpoint = torch.load(
    "artifacts/my_model/model.pt",
    map_location=device,
    weights_only=True
)

# 모델 생성 및 가중치 복원
model = MyModel(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# 입력 문자열 전처리
texts = [
    "It is a truth universally acknowledged."
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# 추론
with torch.no_grad():
    outputs = model(**inputs)

print(outputs)
```

## 꼭 기억할 점

토크나이저를 새로 훈련하면 같은 문장도 전혀 다른 토큰 ID로 변환될 수 있습니다.

예를 들어 기존 토크나이저에서 `apple`이 150번이었다고 해도 새 토크나이저에서는 827번일 수 있습니다. 하지만 모델의 임베딩 150번 위치는 기존 `apple`의 의미를 학습한 상태입니다. 따라서 모델 학습 후 토크나이저만 새로 훈련하면 입력과 모델의 임베딩이 서로 맞지 않게 됩니다.

그래서 실무 원칙은 다음과 같습니다.

- 모델과 토크나이저를 항상 한 묶음으로 버전 관리합니다.
- 모델 학습이 끝난 뒤 토크나이저를 다시 훈련하지 않습니다.
- 특수 토큰 ID와 vocabulary 크기를 함께 기록합니다.
- 간단한 실험은 `tokenizer.json`으로 저장합니다.
- Transformers 모델과 연결할 때는 `save_pretrained()` 형식을 사용합니다.
- 토크나이저 객체 전체를 `torch.save()`로 저장하기보다는 공식 JSON/폴더 형식을 사용합니다.