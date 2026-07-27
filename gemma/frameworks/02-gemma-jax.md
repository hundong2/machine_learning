# 02. Gemma 라이브러리 + JAX/Kauldron

## 언제 선택하나

Google DeepMind의 `gemma` 라이브러리는 Gemma 구조, 토크나이저, 체크포인트, LoRA, 샤딩을 JAX 생태계에서 직접 다루게 해 줍니다. Keras보다 명시적이며 연구 코드와 TPU 확장에 적합합니다.

공식 출발점:

- [Gemma 라이브러리 Finetuning](https://gemma-llm.readthedocs.io/en/latest/colab_finetuning.html)
- [Gemma 라이브러리 API](https://gemma-llm.readthedocs.io/)
- [Kauldron](https://kauldron.readthedocs.io/)

공식 튜닝 예제는 Kauldron이 체크포인트, 학습 루프, 평가, 메트릭, 샤딩을 관리합니다.

## 1단계: 새 환경

```bash
python -m venv .venv-jax
source .venv-jax/bin/activate
pip install -U pip
pip install gemma
```

가속기별 JAX wheel은 공식 설치 지침을 따릅니다. 확인:

```python
import jax
print(jax.__version__)
print(jax.devices())
```

JAX는 프로세스 시작 때 GPU 메모리를 선할당할 수 있습니다.

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
```

환경 변수는 JAX import보다 먼저 설정합니다.

## 2단계: 토크나이저

```python
from gemma import gm

tokenizer = gm.text.Gemma3Tokenizer()
ids = tokenizer.encode("반품 절차를 알려 주세요.", add_bos=True)
print(ids)
print(tokenizer.decode(ids))
```

실습:

1. BOS/EOS 포함 여부를 바꿔 길이를 비교합니다.
2. 직접 쓴 `<start_of_turn>`이 특수 토큰으로 처리되는지 확인합니다.
3. 공통 데이터 5개를 encode→decode하고 손실되는 문자가 없는지 봅니다.

## 3단계: 데이터 파이프라인 이해

Kauldron 데이터 변환의 핵심 출력은 다음 세 텐서입니다.

```text
input      : 모델에 들어가는 토큰
target     : 다음 토큰 정답
loss_mask  : 실제 loss를 계산할 위치
```

공식 예제 구조:

```python
from kauldron import kd

ds = kd.data.py.Tfds(
    name="mtnt/en-fr",
    split="train",
    shuffle=True,
    batch_size=8,
    transforms=[
        gm.data.Seq2SeqTask(
            in_prompt="src",
            in_response="dst",
            out_input="input",
            out_target="target",
            out_target_mask="loss_mask",
            tokenizer=tokenizer,
            max_length=200,
            truncate=True,
        ),
    ],
)
```

우리 JSONL을 사용할 때는 Kauldron의 현재 JSON source에 `prompt`, `response` 필드를 노출하거나, 로더에서 공통 `messages`의 마지막 user/assistant를 두 필드로 변환합니다. 먼저 작은 배치를 화면에 출력해 shape와 mask를 확인한 뒤 학습하세요.

필수 검사:

```python
example = ds[0]
print(example["input"].shape)
print(example["target"].shape)
print(example["loss_mask"].shape)
print(tokenizer.decode(example["input"][0]))
```

## 4단계: 모델과 손실의 key 연결

Kauldron은 문자열 key로 배치, 예측, 파라미터를 연결합니다.

```python
model = gm.nn.Gemma3_1B(tokens="batch.input")

loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
    logits="preds.logits",
    labels="batch.target",
    mask="batch.loss_mask",
)
```

실습:

1. `loss_mask`를 모두 1로 만든 잘못된 버전과 비교합니다.
2. assistant 응답 영역의 mask만 1인지 토큰별로 출력합니다.
3. padding 위치의 mask가 0인지 확인합니다.

## 5단계: Trainer 스모크 테스트

```python
import optax

trainer = kd.train.Trainer(
    seed=42,
    workdir="/tmp/gemma-jax-smoke",
    train_ds=ds,
    model=model,
    init_transform=gm.ckpts.LoadCheckpoint(
        path=gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
    ),
    num_train_steps=20,
    train_losses={"loss": loss},
    optimizer=optax.adafactor(learning_rate=1e-4),
)

state, aux = trainer.train()
```

체크포인트 enum과 모델 클래스는 설치한 `gemma` 버전에서 확인합니다. 최신 라이브러리는 Gemma 3/3n/4 클래스를 제공하지만 체크포인트 접근 권한과 장치 메모리는 별도 조건입니다.

## 6단계: 체크포인트

주기 저장:

```python
checkpointer = kd.ckpts.Checkpointer(save_interval_steps=10)
```

수동 파라미터 저장:

```python
gm.ckpts.save_params(state.params, "/tmp/gemma-jax-smoke/final")
```

전문가가 반드시 확인할 것:

- 파라미터뿐 아니라 optimizer, step, dataset state가 복원되는가?
- 중단 후 재개한 run이 연속 run과 같은 결과를 내는가?
- 샤딩된 체크포인트가 다른 device mesh에서 로드되는가?

## 7단계: 생성 평가

```python
sampler = gm.text.ChatSampler(
    model=model,
    params=state.params,
    tokenizer=tokenizer,
)
```

설치 버전의 sampler 호출법에 맞춰 공통 eval prompt를 생성합니다. JIT warmup 생성은 시간 측정에서 제외하고, 같은 prompt와 sampling method로 튜닝 전후를 비교합니다.

## 8단계: LoRA

`gemma.gm.nn.LoRA`와 `gemma.peft` API는 모델 파라미터를 어댑터로 분리·병합할 수 있습니다. 버전 변화가 있는 저수준 API이므로 다음 순서로 실습합니다.

1. 설치 버전 API에서 `gm.nn.LoRA`, `gm.peft` 시그니처 확인
2. attention projection 하나에 adapter 적용
3. trainable tree의 leaf 수와 바이트 계산
4. 기본 파라미터가 실제로 변하지 않는지 hash 비교
5. adapter 분리 저장
6. merge 후 sampling 출력 비교

```python
import inspect
print(inspect.signature(gm.nn.LoRA))
print(dir(gm.peft))
```

문서 버전과 설치 버전이 다르면 추측으로 코드를 수정하지 말고 API reference와 release를 고정합니다.

## 9단계: JAX 전문가 과제

- `jax.tree.map`으로 파라미터 dtype·크기 요약기 작성
- `jax.jit` 첫 호출과 steady-state 처리량 분리
- `jax.profiler`로 입력 파이프라인과 연산 병목 구분
- device mesh를 1D/2D로 바꿔 sharding 비교
- seed, 데이터 shuffle state, 체크포인트 복원의 결정성 검증
- full fine-tuning과 LoRA의 optimizer state 크기 비교

## 디버깅

- `jax.devices()`가 CPU만 표시: accelerator wheel 또는 런타임 설정 문제
- 첫 step이 매우 느림: XLA compile인지 확인
- 재컴파일 반복: batch shape가 계속 변하는지 확인
- OOM: 시퀀스/배치 축소, 선할당 비율 조절, LoRA/샤딩
- checkpoint 오류: 모델 클래스·revision·mesh 불일치 확인
- loss가 0 또는 NaN: mask 합, dtype, label shift 확인

## 완료 기준

- [ ] JAX 장치와 메모리 정책을 확인했다.
- [ ] `input/target/loss_mask`를 토큰 수준에서 검증했다.
- [ ] 20-step Trainer를 완료했다.
- [ ] 중단/재개 체크포인트를 시험했다.
- [ ] JIT 시간과 실제 처리량을 분리해 기록했다.

