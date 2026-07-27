# 08. Keras 분산 조정: DeviceMesh와 LayoutMap

## 언제 선택하나

모델이 한 장치에 들어가지 않거나 TPU/GPU 여러 개를 사용해야 할 때 Keras Distribution API로 가중치와 텐서 배치를 지정합니다. Google 공식 가이드는 JAX 백엔드와 TPU 모델 병렬화를 중심으로 설명합니다.

공식 출발점:

- [Keras를 사용한 Gemma 분산 조정](https://ai.google.dev/gemma/docs/core/distributed_tuning?hl=ko)
- [Keras 3 분산 학습](https://keras.io/guides/distribution/)

## 먼저 구분하기

- 데이터 병렬: 모델 복제본마다 다른 미니배치
- 모델 병렬: 한 모델의 가중치/연산을 여러 장치에 분할
- 파이프라인 병렬: layer 구간을 장치별로 배치
- 텐서 병렬: 큰 행렬 연산 자체를 여러 장치에 분할

공식 Gemma 예제의 `LayoutMap`은 특정 가중치 축을 device mesh의 `model` 축으로 샤딩합니다.

## 1단계: 장치 확인

```python
import jax
print(jax.devices())
print("device count:", jax.device_count())
```

무료 TPU 런타임의 장치 구성은 시점에 따라 달라질 수 있습니다. 코드에서 8개라고 고정하기 전에 실제 수를 확인합니다.

## 2단계: Keras JAX 백엔드

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import keras
import keras_hub
```

## 3단계: DeviceMesh

장치가 8개라면 예시 1×8 mesh:

```python
devices = jax.devices()
device_mesh = keras.distribution.DeviceMesh(
    shape=(1, len(devices)),
    axis_names=("batch", "model"),
    devices=devices,
)
```

`batch=1`, `model=8`은 모델 병렬 중심입니다. 2×4 mesh는 데이터 2-way와 모델 4-way를 결합할 수 있지만 global batch와 shard compatibility를 확인해야 합니다.

실습:

1. 1×N mesh 생성
2. 가능한 경우 2×(N/2) mesh 생성
3. 각 경우 global batch 조건 기록

## 4단계: LayoutMap

```python
layout_map = keras.distribution.LayoutMap(device_mesh)
model_dim = "model"

layout_map["token_embedding/embeddings"] = (model_dim, None)
layout_map["decoder_block.*attention.*(query|key|value).*kernel"] = (
    model_dim, None, None
)
layout_map["decoder_block.*attention_output.*kernel"] = (
    model_dim, None, None
)
layout_map["decoder_block.*ffw_gating.*kernel"] = (None, model_dim)
layout_map["decoder_block.*ffw_linear.*kernel"] = (model_dim, None)
```

정규식은 모델 버전의 실제 variable path와 일치해야 합니다. Gemma 1 예제의 path가 Gemma 3/4에 그대로 맞는다고 가정하지 않습니다.

검증 코드:

```python
for variable in model.weights:
    if "decoder_block_0" in variable.path:
        print(variable.path, variable.shape, variable.value.sharding.spec)
```

원하는 큰 행렬이 shard되지 않고 복제됐다면 OOM이 날 수 있습니다.

## 5단계: 분산 전략 적용 후 모델 로드

```python
strategy = keras.distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch",
)
keras.distribution.set_distribution(strategy)

model = keras_hub.models.Gemma3CausalLM.from_preset(
    "hf://google/gemma-3-1b-it"
)
```

분산 설정은 모델 변수를 만들기 전에 적용합니다.

## 6단계: shard 감사

다음 표를 자동 생성합니다.

| variable regex | shape | sharding spec | replicated bytes | sharded bytes |
|---|---:|---|---:|---:|

전문가 과제:

```python
total_bytes = 0
for variable in model.weights:
    size = variable.size * variable.dtype.itemsize
    total_bytes += size
print(total_bytes / 1024**3)
```

각 장치가 실제로 보유한 local shard 크기는 JAX 배열의 addressable shards로 확인합니다.

## 7단계: LoRA와 전체 튜닝 비교

### LoRA

```python
model.backbone.enable_lora(rank=8)
```

### 전체 튜닝

LoRA를 활성화하지 않고 모든 가중치를 학습합니다. 먼저 작은 모델·짧은 sequence·몇 step으로만 확인합니다.

비교:

- 장치당 가중치
- optimizer state
- activation
- 통신량
- compile time
- steps/s
- checkpoint 저장 시간

## 8단계: 입력 배치의 shard

모델 weight만 나눠도 입력 batch가 전략과 맞지 않으면 성능이 낮거나 shape 오류가 납니다.

- global batch가 batch mesh 축으로 나누어지는가?
- 각 device의 local batch가 0이 되지 않는가?
- 마지막 incomplete batch를 어떻게 처리하는가?
- variable sequence shape 때문에 재컴파일하지 않는가?

## 9단계: 장애 복구

분산 학습에서 checkpoint는 선택이 아닙니다.

1. 5~10 step마다 작은 테스트 checkpoint
2. 프로세스 종료
3. 동일 mesh에서 재개
4. 다른 mesh에서 restore 가능 여부 확인
5. optimizer/step/data position 연속성 확인

TPU VM이나 spot 인스턴스를 쓰면 object storage에 비동기 저장하는 전략을 검토합니다.

## 디버깅

- layout 정규식이 variable path와 불일치
- mesh shape 곱과 device 수 불일치
- global batch가 batch 축으로 나누어지지 않음
- 모델 생성 후 strategy를 설정함
- 일부 큰 tensor가 replicated 상태
- JIT compile 동안 OOM과 학습 OOM을 혼동
- checkpoint mesh metadata 불일치

## 전문가 확장

- 1D vs 2D mesh의 throughput/통신 비교
- 자동 LayoutMap과 수동 LayoutMap 비교
- tensor shape divisibility 정적 검사기
- LoRA adapter 변수의 replicated/sharded 전략 비교
- Gemma 3→4에서 variable path 변경 감지 테스트
- GKE TPU 또는 Cloud TPU로 notebook 외 실행 전환

## 완료 기준

- [ ] 데이터·모델 병렬 차이를 설명한다.
- [ ] 실제 device mesh를 출력했다.
- [ ] variable별 sharding spec을 감사했다.
- [ ] 분산 5-step 학습과 checkpoint 복원을 수행했다.
- [ ] 장치당 메모리와 처리량을 단일 장치 기준과 비교했다.

