<!-- curriculum: cycle=3; level=production-engineering; source_index=2/18; source=02-1.LazyLinear.md; part=1/1 -->

# LazyLinear 운영: 지연 초기화를 정적 릴리스 계약으로 봉인하는 법

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-01 |
| 회차·수준 | 3회차 · 실무 엔지니어 |
| 현재 소스 | 2/18 · `02-1.LazyLinear.md` |
| Part | 1/1 |
| 이전 소스 | `02.ClassificationForHands.md` |
| 다음 소스 | `02-02.AdaptiveAvgPooll2d.md` |
| 실행 검증 환경 | Python 3.12.12 · PyTorch 2.13.0 · NumPy 2.3.5 · C++17 · C# 7.2/Mono |

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. `LazyLinear`의 materialization을 개발 편의 기능이 아니라 모델 수명 주기의 build 단계로 정의한다.
2. 입력 shape, 전처리, class 순서, weight hash를 하나의 release manifest로 묶는다.
3. 여러 학습 replica가 서로 다른 첫 초기값을 갖지 않도록 materialize와 동기화 순서를 설계한다.
4. optimizer, DDP/FSDP, checkpoint, compile, ONNX export의 안전한 선후 관계를 설명한다.
5. cold-start, parameter·gradient·optimizer state의 메모리 비용을 계산한다.
6. Python, C++, C#에서 같은 affine head의 shape·layout·dtype·출력을 검증한다.
7. 잘못된 health check, checkpoint 교체, 가변 해상도 요청 때문에 생기는 운영 장애를 분리 진단한다.

## 선수 지식과 기호

| 기호 | 뜻 | 예시 |
| --- | --- | --- |
| $N$ | batch 크기 | 1, 32 |
| $C,H,W$ | 채널, 높이, 너비 | 16, 8, 10 |
| $F$ | flatten 뒤 feature 수 | $CHW=1280$ |
| $K$ | class 또는 출력 feature 수 | 5 |
| $W_{\mathrm{fc}}$ | affine weight | $(K,F)$ |
| $b$ | affine bias | $(K)$ |
| $S$ | versioned input specification | layout·shape·dtype·전처리 |

PyTorch의 이미지 feature map은 이 글에서 NCHW를 사용한다.

$$
X_0 \in \mathbb{R}^{N \times C_0 \times H_0 \times W_0}
$$

backbone 출력과 flatten 결과는 다음과 같다.

$$
X_L \in \mathbb{R}^{N \times C_L \times H_L \times W_L}
\longrightarrow
Z \in \mathbb{R}^{N \times F},
\qquad F=C_LH_LW_L
$$

affine head는 다음 계산을 수행한다.

$$
Y=ZW_{\mathrm{fc}}^{\mathsf T}+b,
\qquad
W_{\mathrm{fc}} \in \mathbb{R}^{K \times F},
\qquad
Y \in \mathbb{R}^{N \times K}
$$

## 1. 세 번째 회차에서 바뀌는 질문

1회차는 첫 batch가 `in_features`를 정한다는 직관과 가변 해상도 함정을 다뤘다. 2회차는 build API, optimizer·checkpoint 순서, 단위 테스트를 구현했다. 실무 엔지니어의 질문은 한 단계 더 바뀐다.

> 어느 요청이 우연히 weight shape을 정하게 둘 것인가가 아니라, 누가 어떤 계약으로 모델을 build하고 검증한 뒤 불변 artifact로 배포할 것인가?

`LazyLinear`는 서비스가 요청마다 shape을 적응하는 계층이 아니다. 배포 artifact 안의 `Gemm` 또는 `Linear`는 결국 정적인 $K \times F$ weight를 가져야 한다. 따라서 lazy state는 개발·조립 단계에서만 허용하고, registry에 올리는 release bundle에는 남기지 않는 것이 안전하다.

## 2. 원본의 실제 전제와 운영 교정

원본은 수동 `Linear`, `LazyLinear`, Global Average Pooling(GAP)을 비교한다. 핵심 아이디어는 유용하지만 운영 계약으로 옮길 때 다음을 바로잡아야 한다.

| 원본의 표현 또는 생략 | 운영 관점의 교정 |
| --- | --- |
| 첫 dummy를 통과시키면 된다 | dummy는 versioned input spec에서 생성하고 build job에서만 실행해야 한다. health check나 실제 요청이 build를 대신하면 안 된다. |
| `LazyLinear`이 차원 고정 문제를 해결한다 | 코드 작성 시 수동 계산을 늦출 뿐이다. materialize 뒤 $F$는 고정되고 다른 flatten 길이는 실패한다. |
| lazy는 최적화 기법이다 | 여기서는 수명 주기 편의 기능이다. parameter 수와 추론 FLOPs를 줄이지 않으며 cold-start 비용을 추가할 수 있다. |
| `LazyConv2d`가 입력 채널을 자동 추론한다 | 첫 입력 채널로 고정될 뿐, 이후 RGB와 grayscale을 모두 자동 수용하지 않는다. |
| GAP면 어떤 크기든 받는다 | backbone의 최소 크기, stride, positional encoding, 전처리 제약을 모두 통과할 때만 공간 크기에 덜 민감하다. |
| GAP가 곧 FCN이다 | 뒤에 `Linear`가 있으면 문자 그대로 합성곱만 있는 네트워크는 아니다. 여기서 중요한 성질은 classifier 입력 $F=C_L$가 공간 크기와 분리된다는 점이다. |
| flatten head의 큰 parameter 수가 lazy로 개선된다 | lazy는 같은 $F$를 뒤늦게 알 뿐 parameter 수는 동일하다. parameter를 줄이는 것은 GAP 같은 구조 변경이다. |

PyTorch에서는 concrete `state_dict`를 lazy module에 로드해 parameter를 구체화할 수도 있다. 그러므로 “반드시 dummy forward만 가능하다”도 지나치게 강한 문장이다. 다만 버전별 속성 갱신과 입력 계약 검증을 우연에 맡기지 않기 위해, 이 글의 신규 build는 대표 입력으로 materialize하고 checkpoint restore는 manifest와 smoke test까지 통과시키는 절차를 사용한다.

## 3. 모델 수명 주기: lazy 구간을 짧게 만든다

운영 모델을 다음 상태 기계로 관리한다.

```text
[SOURCE]
  topology에 LazyLinear 존재
        |
        | build(spec, seed)
        v
[MATERIALIZED]
  모든 parameter가 concrete shape·dtype·device를 가짐
        |
        | replica sync / checkpoint load / smoke test
        v
[VERIFIED]
  input spec, class map, golden output, hash 확정
        |
        | export + package
        v
[RELEASE BUNDLE]
  model artifact + manifest + golden vectors
        |
        | canary -> promote 또는 rollback
        v
[SERVING]
  요청은 shape guard 뒤 concrete model만 호출
```

금지할 역방향 전이는 다음과 같다.

- serving process가 첫 고객 요청으로 materialize한다.
- replica마다 독립적으로 random weight를 materialize한 뒤 학습을 시작한다.
- manifest는 그대로 두고 model file만 교체한다.
- export 뒤 입력 해상도나 class 순서를 바꾼다.
- 이미 검증된 bundle 안에서 다시 lazy module을 만든다.

## 4. 입력 사양은 shape보다 큰 ABI다

다음 필드를 하나의 `InputSpec`으로 versioning한다.

| 필드 | 예시 | 누락 시 장애 |
| --- | --- | --- |
| layout | `NCHW` | NHWC를 NCHW로 해석해 channel·공간 축이 바뀜 |
| symbolic batch | `N>=1` | batch 1만 export해 batch 8 요청이 실패할 수 있음 |
| channel | 3 | grayscale 또는 alpha 입력이 다른 $F$를 만들거나 conv에서 실패 |
| height·width | 32·40 | flatten head의 $F$가 바뀜 |
| dtype | `float32` | `uint8` 값 범위나 FP16 연산 경계가 달라짐 |
| color order | RGB | BGR 입력으로 정확도 저하 |
| resize | bilinear, align convention | 같은 크기여도 pixel 값이 달라짐 |
| normalization | mean·std | logits가 체계적으로 이동 |
| class map | ordered labels | shape은 같지만 class 의미가 뒤바뀜 |

이 사양 $S$에서 feature 길이를 계산하는 함수를 $g$라 두면 다음과 같다.

$$
F=g(S;\theta_{\mathrm{backbone}})
$$

여기서 $g$는 단순히 $CHW$를 입력에서 읽는 함수가 아니다. convolution, pooling, padding, stride가 만드는 모든 중간 shape을 거친 결과다. backbone topology나 전처리 crop이 바뀌어도 $F$가 바뀔 수 있다.

## 5. tensor shape을 build 로그로 남기기

오늘 사용할 모델은 다음 경로를 갖는다.

| 단계 | 연산 | 출력 shape |
| --- | --- | --- |
| 입력 | RGB NCHW | `(N, 3, 32, 40)` |
| block 1 | `Conv2d(3, 8, 3, stride=2, padding=1)` | `(N, 8, 16, 20)` |
| block 2 | `Conv2d(8, 16, 3, stride=2, padding=1)` | `(N, 16, 8, 10)` |
| flatten | `flatten(1)` | `(N, 1280)` |
| lazy head | `LazyLinear(3)` | `(N, 3)` |

convolution 출력 높이는 다음 식으로 추적한다.

$$
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+2p-d(k-1)-1}{s}+1
\right\rfloor
$$

너비도 같은 식을 쓴다. 두 block 모두 $k=3$, $s=2$, $p=1$, $d=1$이므로 $32 \times 40$은 $16 \times 20$, 다시 $8 \times 10$이 된다.

$$
F=16 \times 8 \times 10=1280
$$

따라서 head weight와 bias shape은 각각 `(3, 1280)`, `(3)`이다. build 로그에는 입력만 기록하지 말고 각 stage의 output shape과 최종 parameter shape도 남긴다. 이 로그는 “어느 연산부터 달라졌는가”를 찾는 가장 싼 장애 자료다.

## 6. 분산 materialization: seed보다 weight 동기화를 신뢰한다

Lazy weight 초기화 범위는 fan-in $F$에 의존한다. 단순화한 균등분포는 다음과 같다.

$$
W_{ij} \sim \mathcal U\left(-\frac{1}{\sqrt F},\frac{1}{\sqrt F}\right)
$$

replica $r$가 서로 다른 RNG 상태로 materialize하면 다음이 된다.

$$
W^{(0)} \ne W^{(1)} \ne \cdots \ne W^{(R-1)}
$$

DDP가 parameter를 동기화하는 정상 경로를 정확히 사용하면 초기 parameter broadcast가 이를 정렬할 수 있다. 그러나 wrapper 이전에 optimizer state를 만들거나, rank별 warm-up에서 다른 graph·shape을 사용하거나, custom sharding을 잘못 구성하면 진단이 어려워진다. 운영 순서는 다음처럼 보수적으로 잡는다.

1. 모든 rank가 동일한 topology와 `InputSpec` hash를 확인한다.
2. CPU 또는 target device에서 representative input으로 모든 lazy parameter를 materialize한다.
3. `UninitializedParameter`가 0개인지 collective 전 검사한다.
4. rank 0의 model state를 broadcast하거나 DDP/FSDP가 보장하는 초기 동기화를 사용한다.
5. parameter 이름·shape·dtype·checksum을 rank 간 비교한다.
6. 그 뒤 optimizer, scheduler, scaler를 만들거나 checkpoint state를 복원한다.

같은 seed는 재현성에 유용하지만 동기화 프로토콜을 대신하지 않는다. 라이브러리 버전, 생성 순서, rank별 선행 RNG 소비가 달라지면 같은 “설정 seed”만으로 동일 weight를 보장할 수 없기 때문이다.

## 7. model registry에 올릴 release bundle

최소 bundle은 다음 파일로 구성한다.

```text
lazy-head-v3/
  model.pt 또는 model.onnx
  manifest.json
  classes.json
  golden-input.bin
  golden-output.json
  build-report.json
```

`manifest.json`의 핵심 필드는 다음과 같다.

| 범주 | 필드 |
| --- | --- |
| identity | `model_name`, `semantic_version`, `git_sha`, `build_id` |
| input | layout, $C,H,W$, dtype, resize, normalization |
| output | class order, logits dtype, output shape |
| materialization | `in_features`, weight shape, build seed |
| artifact | byte size, SHA-256, exporter·opset |
| compatibility | minimum runtime, target providers |
| validation | golden tolerance, test report hash |

artifact hash는 weight 내용의 동일성을 확인하고, spec hash는 입출력 의미의 동일성을 확인한다. 둘 중 하나만 같아서는 같은 release가 아니다.

$$
\mathrm{release\_id}
=
H(\mathrm{artifact\ bytes})
\mathbin\Vert
H(\mathrm{canonical\ manifest})
$$

`classes.json`도 hash 범위에 포함한다. class 개수가 같아도 순서가 바뀌면 출력 의미가 달라진다.

## 8. 실행 가능한 Python 운영 계약

다음 코드는 **독립 실행 가능한 Python·PyTorch 예제**다. build, replica drift와 동기화, manifest hash, checkpoint round trip, 잘못된 입력 거부, finite backward를 한 번에 검증한다.

```python
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parameter import UninitializedParameter


@dataclass(frozen=True)
class InputSpec:
    layout: str = "NCHW"
    channels: int = 3
    height: int = 32
    width: int = 40
    dtype: str = "float32"
    color_order: str = "RGB"
    version: int = 3

    def validate(self) -> None:
        if self.layout != "NCHW" or self.dtype != "float32":
            raise ValueError("this release accepts NCHW float32 only")
        if (self.channels, self.height, self.width) != (3, 32, 40):
            raise ValueError("input shape contract mismatch")

    def canonical_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class LazyServiceModel(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.LazyLinear(num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 40):
            raise ValueError("expected [N,3,32,40]")
        return self.head(torch.flatten(self.features(x), 1))


def uninitialized(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ]


@torch.no_grad()
def build(spec: InputSpec, seed: int) -> LazyServiceModel:
    spec.validate()
    torch.manual_seed(seed)
    model = LazyServiceModel().eval()
    dummy = torch.zeros(2, spec.channels, spec.height, spec.width)
    output = model(dummy)
    if uninitialized(model):
        raise RuntimeError("lazy parameter survived build")
    if tuple(model.head.weight.shape) != (3, 1280):
        raise RuntimeError("unexpected head shape")
    if tuple(output.shape) != (2, 3):
        raise RuntimeError("unexpected output shape")
    return model


def state_bytes(model: nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def load_verified(payload: bytes, spec: InputSpec) -> LazyServiceModel:
    candidate = build(spec, seed=0)
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    candidate.load_state_dict(state, strict=True)
    candidate.eval()
    with torch.no_grad():
        candidate(torch.zeros(1, 3, 32, 40))
    return candidate


spec = InputSpec()
replica_a = build(spec, seed=101)
replica_b = build(spec, seed=202)

# 서로 다른 build seed는 실제로 다른 초기 weight를 만든다.
assert not torch.equal(replica_a.head.weight, replica_b.head.weight)

# rank 0 state broadcast를 직렬화·로드로 모사한다.
payload = state_bytes(replica_a)
replica_b.load_state_dict(replica_a.state_dict(), strict=True)
assert all(
    torch.equal(left, right)
    for left, right in zip(replica_a.parameters(), replica_b.parameters())
)

# 배포 golden은 결정적인 입력과 출력으로 고정한다.
golden = torch.linspace(-1.0, 1.0, 3 * 32 * 40).reshape(1, 3, 32, 40)
with torch.no_grad():
    expected = replica_a(golden)
restored = load_verified(payload, spec)
with torch.no_grad():
    actual = restored(golden)
torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

# affine head를 NumPy oracle로 독립 검산한다.
with torch.no_grad():
    feature = torch.flatten(restored.features(golden), 1)
np_output = (
    feature.detach().numpy().astype(np.float64)
    @ restored.head.weight.detach().numpy().astype(np.float64).T
    + restored.head.bias.detach().numpy().astype(np.float64)
)
np.testing.assert_allclose(np_output, actual.detach().numpy(), rtol=1e-5, atol=1e-6)

# 학습 가능성과 gradient 유한성을 확인한다.
restored.train()
optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
labels = torch.tensor([1, 2])
batch = torch.stack([golden[0], -golden[0]])
loss = nn.functional.cross_entropy(restored(batch), labels)
optimizer.zero_grad(set_to_none=True)
loss.backward()
assert all(
    parameter.grad is None or torch.isfinite(parameter.grad).all()
    for parameter in restored.parameters()
)
optimizer.step()

# 다른 해상도는 materialize 재시도가 아니라 입구에서 거부한다.
try:
    restored(torch.zeros(1, 3, 64, 64))
except ValueError as error:
    assert "[N,3,32,40]" in str(error)
else:
    raise AssertionError("wrong resolution was accepted")

manifest = {
    "input_spec_sha256": spec.canonical_hash(),
    "artifact_sha256": hashlib.sha256(payload).hexdigest(),
    "head_weight_shape": list(replica_a.head.weight.shape),
    "output_shape": list(expected.shape),
    "class_order": ["rock", "paper", "scissors"],
}
assert manifest["head_weight_shape"] == [3, 1280]
print("manifest:", json.dumps(manifest, sort_keys=True))
print("golden logits:", [round(value, 6) for value in expected[0].tolist()])
print("loss:", round(float(loss.detach()), 6))
```

### 8.1 이 코드가 의도적으로 하지 않는 것

- serving 중 새 resolution을 보고 head를 다시 만들지 않는다.
- manifest 검증 없이 checkpoint를 로드하지 않는다.
- materialize 이전에 optimizer를 만들어 동작 여부를 버전 우연에 맡기지 않는다.
- `torch.inference_mode()` 안에서 학습용 parameter를 처음 만들지 않는다. 새 parameter가 inference tensor가 되는 위험을 피하려고 build에는 `torch.no_grad()`를 쓴다.
- model hash만 보고 class 의미까지 같다고 가정하지 않는다.

### 8.2 실제 분산 환경에 추가할 검사

위 코드는 두 replica의 state broadcast를 직렬화·로드로 모사한다. 실제 `torch.distributed` 환경에서는 각 rank가 다음 벡터를 `all_gather`해 비교한다.

```python
# 설명용 조각: 실제 실행에는 초기화된 process group이 필요하다.
signature = [
    (name, tuple(parameter.shape), str(parameter.dtype))
    for name, parameter in model.named_parameters()
]
# canonical JSON hash를 all_gather하고 모든 rank가 같은지 검사한다.
```

parameter 값 checksum도 비교하되, 부동소수점 합 하나만 쓰면 상쇄 충돌이 생길 수 있다. 작은 모델은 CPU byte hash, 큰 sharded model은 여러 통계와 shard별 cryptographic hash를 build report에 남긴다.

## 9. checkpoint와 optimizer의 순서

신규 학습과 resume의 안전한 순서는 다르다.

### 9.1 신규 학습

```text
topology 생성
-> InputSpec 검증
-> representative input으로 materialize
-> replica parameter 동기화
-> optimizer / scheduler / scaler 생성
-> 첫 train step
```

### 9.2 resume

```text
checkpoint metadata만 먼저 읽기
-> topology·InputSpec·class map 호환성 검사
-> 동일 spec으로 materialize
-> model state strict load
-> smoke forward와 golden 비교
-> optimizer / scheduler / scaler 생성
-> 각 state strict restore
-> global step·RNG·sampler position 복원
```

optimizer는 현재 PyTorch에서 materialize 전 parameter identity를 유지해 동작할 수 있는 경우가 있다. 그러나 이를 운영 계약으로 채택하면 optimizer state shape, wrapper flattening, compile capture와의 조합을 모두 버전별로 보증해야 한다. concrete model 뒤에 optimizer를 만드는 순서는 더 좁고 감사 가능한 계약이다.

## 10. ONNX와 정적 배포 경계

ONNX로 export할 때 lazy 상태는 release artifact에 남아서는 안 된다. 권장 gate는 다음과 같다.

1. `UninitializedParameter`가 0개인지 확인한다.
2. representative input으로 eager golden logits를 저장한다.
3. batch 축만 dynamic인지, 높이·너비도 dynamic인지 명시한다.
4. `onnx.checker`로 graph를 검사한다.
5. target ONNX Runtime provider에서 golden input을 실행한다.
6. output shape, class 순서, finite 값, 오차 tolerance를 비교한다.
7. graph input과 initializer의 shape를 manifest에 기록한다.

flatten head는 $F=C_LH_LW_L$이므로 높이·너비를 dynamic axis로 선언해도 weight의 안쪽 차원은 바뀌지 않는다. 즉 export annotation만으로 진짜 가변 해상도 모델이 되지 않는다. 가변 해상도가 요구사항이라면 GAP 또는 다른 공간 불변 head로 architecture를 바꾸고 모든 지원 shape를 다시 검증해야 한다.

```python
# 설명용 export gate: 현재 검증 환경에는 onnx·onnxruntime이 없다.
assert not uninitialized(model)
torch.onnx.export(
    model.eval(),
    torch.zeros(1, 3, 32, 40),
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
)
```

현재 로컬 환경에는 `onnx`와 `onnxruntime`이 설치되어 있지 않아 실제 export·target runtime parity는 실행하지 못했다. 이를 숨기지 않고 release 미검증 항목으로 남긴다. Python eager, NumPy oracle, C++·C# golden은 아래에서 별도로 검증한다.

## 11. C++17 정적 serving head

다음 코드는 **독립 실행 가능한 C++17 예제**다. 서비스는 lazy initialization을 재현하지 않고 manifest에서 확정한 $F=4$, $K=3$ affine head와 shape guard를 읽는다는 축소 예제다.

```cpp
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main() {
    constexpr std::size_t F = 4;
    constexpr std::size_t K = 3;
    const std::array<float, F> x{1.0F, -2.0F, 0.5F, 3.0F};
    const std::array<float, K * F> weight{
        0.25F, -0.50F, 1.00F, 0.75F,
        -1.00F, 0.50F, 0.25F, -0.50F,
        0.10F, 0.20F, 0.30F, 0.40F,
    };
    const std::array<float, K> bias{0.10F, -0.20F, 0.30F};
    std::array<float, K> y{};

    if (x.size() != F || weight.size() != K * F) {
        throw std::runtime_error("release manifest shape mismatch");
    }
    for (std::size_t row = 0; row < K; ++row) {
        float value = bias[row];
        for (std::size_t col = 0; col < F; ++col) {
            value += weight[row * F + col] * x[col];
        }
        y[row] = value;
    }

    const std::array<float, K> expected{4.10F, -3.575F, 1.35F};
    for (std::size_t index = 0; index < K; ++index) {
        if (std::abs(y[index] - expected[index]) > 1e-6F) {
            throw std::runtime_error("golden output mismatch");
        }
    }
    std::cout << std::fixed << std::setprecision(6)
              << y[0] << " " << y[1] << " " << y[2] << "\n";
}
```

C++ array의 weight는 논리적으로 `(K,F)`이고 row-major로 평탄화했다. 실제 ONNX Runtime tensor가 이 물리 배열을 직접 요구한다는 뜻은 아니다. exporter가 `Gemm`의 `transB`를 어떻게 설정했는지 graph와 runtime API로 확인해야 한다.

## 12. C# 정적 serving head

다음 코드는 **독립 실행 가능한 C# 예제**이며 C++과 같은 배열, 누적 순서, tolerance를 사용한다.

```csharp
using System;

public static class Program
{
    public static void Main()
    {
        const int F = 4;
        const int K = 3;
        float[] x = { 1.0f, -2.0f, 0.5f, 3.0f };
        float[] weight = {
            0.25f, -0.50f, 1.00f, 0.75f,
            -1.00f, 0.50f, 0.25f, -0.50f,
            0.10f, 0.20f, 0.30f, 0.40f
        };
        float[] bias = { 0.10f, -0.20f, 0.30f };
        float[] output = new float[K];

        if (x.Length != F || weight.Length != K * F)
            throw new InvalidOperationException("release manifest shape mismatch");

        for (int row = 0; row < K; row++)
        {
            float value = bias[row];
            for (int col = 0; col < F; col++)
                value += weight[row * F + col] * x[col];
            output[row] = value;
        }

        float[] expected = { 4.10f, -3.575f, 1.35f };
        for (int index = 0; index < K; index++)
            if (Math.Abs(output[index] - expected[index]) > 1e-6f)
                throw new InvalidOperationException("golden output mismatch");

        Console.WriteLine(
            $"{output[0]:F6} {output[1]:F6} {output[2]:F6}"
        );
    }
}
```

C# 서비스도 Python model을 다시 lazy initialize하지 않는다. registry에서 승인된 ONNX와 manifest를 받고, session 생성 직후 golden vector를 한 번 실행한 뒤 ready 상태로 전환한다.

## 13. 프레임워크 간 shape·layout·dtype 대응

| 경계 | PyTorch 학습 | C++ 서비스 | C# 서비스 |
| --- | --- | --- | --- |
| 이미지 논리 shape | `(N,C,H,W)` | runtime tensor `{N,C,H,W}` | runtime tensor `{N,C,H,W}` |
| 메모리 layout | contiguous NCHW 확인 | stride·contiguous 확인 | DenseTensor 차원 순서 확인 |
| head 입력 | `(N,F)` | `(N,F)` | `(N,F)` |
| weight 논리 shape | `(K,F)` | graph initializer 기준 | graph initializer 기준 |
| 계산 | $ZW^{\mathsf T}+b$ | `Gemm`/MatMul | `Gemm`/MatMul |
| 입력 dtype | 보통 `float32` | `float` | `System.Single` |
| output | logits `(N,K)` | logits `(N,K)` | logits `(N,K)` |

동등성 테스트는 세 층으로 나눈다.

1. preprocessed tensor checksum이 같은가?
2. backbone 중간 feature 또는 flatten vector가 같은가?
3. logits와 최종 class가 tolerance 안에서 같은가?

logits만 다르다고 바로 head transpose를 의심하지 않는다. RGB/BGR, resize kernel, integer division, normalization, NHWC/NCHW가 먼저 달라졌을 수 있다.

## 14. 성능과 메모리 예산

### 14.1 parameter 수와 training memory

bias를 포함한 head parameter 수는 다음과 같다.

$$
P=KF+K=K(F+1)
$$

오늘의 $K=3$, $F=1280$이면 다음과 같다.

$$
P=3(1280+1)=3843
$$

FP32 parameter, gradient, Adam의 두 FP32 moment만 단순 합산하면 대략 parameter당 16 bytes다.

$$
M_{\mathrm{head}}
\approx
P(4_{\mathrm{param}}+4_{\mathrm{grad}}+8_{\mathrm{Adam}})
=61{,}488\ \mathrm{bytes}
$$

약 60.05 KiB다. allocator 정렬, activation, temporary workspace, mixed-precision master copy는 포함하지 않았다. class 수가 1000이고 $F=100{,}352$라면 head만 약 100.35M parameter이므로 Adam 단순 예산이 약 1.50 GiB까지 커진다. lazy는 이 비용을 줄이지 않는다.

### 14.2 materialization peak

build 순간에는 다음 메모리가 겹칠 수 있다.

- dummy input과 backbone activation
- 새 parameter storage
- checkpoint를 메모리에 읽은 임시 byte buffer
- device copy 또는 shard용 임시 buffer
- export graph capture buffer

따라서 production image와 같은 큰 dummy를 모든 worker가 동시에 실행하면 시작 시 메모리 spike가 생긴다. 별도 build job에서 한 번 materialize·export하고 serving worker는 concrete artifact만 읽는 이유다.

### 14.3 cold-start와 ready probe

cold-start를 다음처럼 분해해 기록한다.

$$
T_{\mathrm{ready}}
=T_{\mathrm{download}}
+T_{\mathrm{hash}}
+T_{\mathrm{session}}
+T_{\mathrm{golden}}
+T_{\mathrm{warmup}}
$$

평균 하나보다 각 구간의 p50·p95와 artifact size를 함께 본다. lazy materialization이 serving에 남아 있으면 `session` 또는 첫 `warmup`에 숨어 release 간 비교가 어려워진다.

### 14.4 처리량 측정 규칙

- build 시간과 steady-state inference 시간을 분리한다.
- warm-up 횟수와 측정 batch를 고정한다.
- sync가 필요한 device는 측정 전후 synchronize한다.
- batch별 latency뿐 아니라 sample/s와 peak RSS·device memory를 함께 기록한다.
- flatten head와 GAP head를 비교할 때 backbone, input, class 수, precision을 같게 둔다.

## 15. 수치 안정성과 재현성

### 15.1 build dtype

mixed precision으로 학습하더라도 build는 보통 FP32 parameter로 수행하고 autocast는 train·inference step 경계에 둔다. FP16 dummy로 처음 materialize하면 parameter dtype과 초기화 rounding이 의도치 않게 release 계약이 될 수 있다.

### 15.2 누적 순서와 tolerance

C++·C# 축소 예제는 같은 순서로 scalar를 누적해 exact한 golden에 가깝다. 실제 BLAS, fused kernel, GPU에서는 reduction 순서가 달라질 수 있으므로 보통 다음 혼합 tolerance를 사용한다.

$$
|y-\hat y| \le a_{\mathrm{tol}}+r_{\mathrm{tol}}|y|
$$

FP32와 FP16에 같은 tolerance를 기계적으로 쓰지 않는다. class가 같은지만 검사하면 경계 근처 logit drift를 놓치므로 logits 오차와 top-k를 함께 본다.

### 15.3 exact replay의 범위

다음이 모두 고정되어야 exact replay를 기대할 수 있다.

- code·framework·driver version
- topology와 materialization 순서
- seed와 rank별 RNG state
- data sampler position과 augmentation RNG
- optimizer·scheduler·scaler state
- deterministic algorithm 설정과 hardware

서로 다른 runtime의 목표는 보통 bitwise identity가 아니라 명시한 tolerance의 semantic parity다.

## 16. 테스트·디버깅 순서

장애 때는 다음 순서로 범위를 줄인다.

1. 요청이 올바른 bundle version으로 routing되었는지 확인한다.
2. artifact SHA-256과 canonical manifest hash를 다시 계산한다.
3. 입력 rank, shape, layout, dtype, color order를 확인한다.
4. resize·normalization 뒤 checksum을 Python golden과 비교한다.
5. model 안에 uninitialized parameter가 없는지 확인한다.
6. head의 concrete weight shape `(K,F)`와 class map 길이를 비교한다.
7. golden input을 eager와 target runtime에서 실행한다.
8. 첫 번째로 달라지는 중간 tensor를 찾는다.
9. NaN·Inf, max absolute·relative error, top-k 차이를 기록한다.
10. canary만 실패하면 traffic shape·hardware provider·concurrency 차이를 비교한다.

테스트 pyramid는 다음처럼 구성한다.

| 층 | 검사 | 실행 시점 |
| --- | --- | --- |
| unit | shape 식, manifest parser, hash canonicalization | 모든 commit |
| contract | materialize 후 lazy 0개, wrong shape 거부 | 모든 commit |
| numerical | NumPy affine, finite gradient, golden logits | 모든 commit |
| distributed | rank signature·weight hash 일치 | training image 변경 시 |
| export | ONNX checker, eager-runtime parity | release candidate |
| serving | cold-start, batch, concurrency, memory | canary 전 |
| monitoring | version별 error·latency·drift | 배포 후 지속 |

## 17. 실무 실패 사례

### 17.1 readiness probe가 model shape을 결정했다

서버가 topology만 생성한 뒤 `/ready` probe에서 비용을 아끼려고 `(1,3,16,16)`을 넣었다. head는 작은 $F$로 고정되었고 첫 정상 `(1,3,32,40)` 요청이 행렬 곱에서 실패했다.

해결은 probe shape을 고치는 것만이 아니다. CI build에서 materialize·export한 concrete artifact를 배포하고, ready probe는 승인된 golden input만 실행해야 한다.

### 17.2 두 rank가 같은 topology인데 첫 loss가 달랐다

rank 0은 validation warm-up 뒤 materialize했고 rank 1은 곧바로 materialize했다. 설정 seed는 같았지만 선행 RNG 소비량이 달라 head 초기값이 달라졌다. wrapper 밖의 custom optimizer가 이미 state를 만들고 있어 초기 동기화 뒤에도 state가 어긋났다.

해결은 build RNG stream을 분리하고, materialize 뒤 weight hash 동기화가 끝난 다음 optimizer를 만드는 것이다.

### 17.3 새 checkpoint가 로드됐지만 class 의미가 뒤집혔다

이전과 새 모델 모두 output shape이 `(N,3)`이라 strict load와 smoke shape test가 통과했다. 그러나 class order가 `[rock, paper, scissors]`에서 `[paper, rock, scissors]`로 바뀌었다.

class map hash와 golden label까지 release identity에 넣어야 한다. tensor shape parity는 semantic parity가 아니다.

### 17.4 dynamic axis를 선언했는데 큰 이미지가 실패했다

ONNX export에서 height·width를 dynamic으로 표시했지만 flatten head의 weight는 $F=1280$으로 고정되어 있었다. runtime은 더 큰 flatten vector와 weight를 곱할 수 없었다.

export option이 architecture 제약을 없애지는 않는다. 공간 축을 지원하려면 GAP 등으로 head를 다시 설계하고 shape matrix 전체를 검증한다.

### 17.5 model file만 hot swap한 뒤 일부 pod만 실패했다

manifest와 session cache key는 옛 버전인데 shared volume의 `model.onnx`만 바뀌었다. 새 pod는 새 weight, 기존 pod는 old session을 사용했다.

bundle을 immutable directory와 content-addressed URI로 배포하고, artifact·manifest hash가 모두 맞을 때만 traffic을 받는다. hot swap 대신 새 deployment revision을 만든다.

### 17.6 평균 latency는 정상인데 시작 직후 timeout이 났다

steady-state 평균에는 download, hash, session 생성, warm-up이 제외되어 있었다. autoscaling으로 새 pod가 늘어날 때만 cold-start timeout이 발생했다.

`T_ready` 구간별 histogram을 기록하고 pre-warmed capacity, artifact 크기, session option을 함께 조정한다.

## 18. 배포·모니터링·rollback

### 18.1 배포 게이트

- artifact와 manifest를 immutable storage에 업로드한다.
- staging에서 target CPU/GPU provider별 golden parity를 통과시킨다.
- canary는 적은 traffic과 대표 input shape·batch를 받는다.
- error rate, p95·p99 latency, memory, logits drift를 stable과 비교한다.
- 기준을 통과한 동일 bundle hash만 점진 확대한다.

### 18.2 모니터링 필드

각 inference span 또는 집계 metric에 다음을 연결한다.

- `release_id`, artifact hash prefix, spec version
- runtime·provider·hardware class
- batch와 input shape
- preprocessing version
- queue·preprocess·inference·postprocess latency
- NaN·Inf count와 logits norm
- predicted class distribution과 reject rate
- shape rejection·golden self-test failure count

원본 이미지나 raw feature를 무단 저장하지 않는다. 개인정보·보존 정책에 맞춰 aggregate 또는 승인된 sampled telemetry만 사용한다.

### 18.3 rollback 단위

rollback은 model file 하나가 아니라 다음 원자적 bundle 전체다.

```text
model + manifest + class map + preprocessing + runtime config
```

database schema처럼 input spec version에 호환 범위를 두되, 모호한 fallback으로 잘못된 shape를 pad·crop하지 않는다. 계약 위반은 명시적으로 거부하고 metric을 올린다.

## 19. 실무 체크리스트

### build

- [ ] 소스 topology의 lazy module 목록을 기록했는가?
- [ ] versioned `InputSpec`에서 representative input을 만들었는가?
- [ ] build 뒤 `UninitializedParameter`가 0개인가?
- [ ] 각 stage shape와 final `(K,F)`를 기록했는가?
- [ ] replica parameter shape·dtype·hash가 일치하는가?
- [ ] optimizer는 동기화된 concrete model 뒤에 생성했는가?

### release

- [ ] artifact와 canonical manifest SHA-256이 있는가?
- [ ] class order와 preprocessing version이 bundle에 포함되는가?
- [ ] eager golden과 target runtime golden이 tolerance 안에서 같은가?
- [ ] supported batch·shape matrix를 실제 실행했는가?
- [ ] ONNX checker와 target provider test를 통과했는가?
- [ ] cold-start와 peak memory를 release 기준으로 측정했는가?

### serving

- [ ] serving binary가 lazy parameter를 새로 만들지 않는가?
- [ ] 요청 입구에서 rank·shape·layout·dtype을 검사하는가?
- [ ] ready probe가 승인된 golden vector를 쓰는가?
- [ ] bundle hash가 cache와 deployment revision의 key인가?
- [ ] canary 자동 중단과 이전 bundle rollback이 가능한가?
- [ ] telemetry에 release·spec version이 포함되는가?

## 20. 연습문제

### 문제 1

backbone 출력이 `(N,64,14,10)`이고 class가 12개다. flatten head의 $F$, parameter 수, FP32 parameter byte 수를 구하라.

### 문제 2

8개 rank가 같은 seed를 설정했다. 그러면 materialized weight가 같다는 별도 검사를 생략해도 되는가? 이유를 설명하라.

### 문제 3

batch만 dynamic이고 공간 크기는 고정인 ONNX 입력 shape를 기호로 쓰고, flatten head에서 높이·너비를 함부로 dynamic으로 만들 수 없는 이유를 설명하라.

### 문제 4

model artifact SHA는 같지만 `classes.json` 순서가 다르다. 같은 release로 취급할 수 있는가?

### 문제 5

서비스의 eager Python과 C# ONNX logits가 다르다. 조사 순서를 4단계 이상 제시하라.

### 문제 6

GAP head로 바꾸면 모든 resolution을 무조건 지원한다는 주장에 반례를 두 가지 들어라.

## 21. 연습문제 해답

### 해답 1

$$
F=64 \times 14 \times 10=8960
$$

$$
P=12(8960+1)=107{,}532
$$

FP32 parameter만 저장하면 $107{,}532 \times 4=430{,}128$ bytes, 약 420.05 KiB다. gradient와 optimizer state는 포함하지 않았다.

### 해답 2

생략할 수 없다. rank마다 materialize 전 RNG 소비량, module 생성 순서, library·device 경로가 다를 수 있다. 동일 spec과 seed를 확인한 뒤에도 concrete parameter signature와 동기화 결과를 검사해야 한다.

### 해답 3

입력은 `(N,3,32,40)`에서 $N$만 symbolic이다. flatten head weight의 안쪽 차원은 $F=1280$으로 고정되어 있으므로 $H,W$가 바뀌어 다른 $F$가 되면 `Gemm`의 안쪽 차원이 맞지 않는다.

### 해답 4

아니다. 동일 logits의 index가 다른 class 의미를 가리킬 수 있다. artifact hash와 spec·class map hash를 함께 release identity로 사용해야 한다.

### 해답 5

bundle과 manifest hash, input shape·dtype·layout, 전처리 tensor checksum, 중간 feature, head weight shape·transpose, logits tolerance 순서로 좁힌다. 첫 번째로 달라지는 경계를 찾아야 한다.

### 해답 6

작은 입력이 여러 stride를 거치며 0 크기 feature를 만들 수 있다. backbone이 고정 positional embedding 또는 고정 token grid를 요구할 수도 있다. 전처리 crop이나 최소 object scale 같은 품질 계약도 여전히 resolution 범위를 제한한다.

## 22. 핵심 요약

1. `LazyLinear`는 shape 결정을 늦추지만 materialize 뒤의 $F$는 고정된다.
2. lazy 상태는 build 단계에서만 허용하고 release bundle과 serving에는 남기지 않는다.
3. input shape뿐 아니라 layout, dtype, 전처리, class order를 versioned ABI로 관리한다.
4. 같은 seed보다 concrete weight 동기화와 rank별 signature 검사를 신뢰한다.
5. materialize·동기화 뒤 optimizer와 distributed wrapper의 상태를 구성한다.
6. artifact hash와 manifest hash가 함께 같아야 같은 release다.
7. dynamic axis 선언은 flatten head의 architecture 제약을 없애지 않는다.
8. lazy는 parameter·FLOPs를 줄이지 않으며 cold-start와 build peak를 추가할 수 있다.
9. Python·NumPy·C++·C# golden을 전처리, feature, logits 경계로 나누어 비교한다.
10. rollback 단위는 model, manifest, class map, preprocessing, runtime config의 불변 bundle이다.

## 23. 다음 학습 예고

다음은 `02-02.AdaptiveAvgPooll2d.md`를 3회차 실무 엔지니어 수준으로 다룬다. 가변 해상도 요청을 bucket으로 운영하고, adaptive pooling의 bin 경계를 Python·C++·C#에서 맞추며, ONNX runtime·quantization·성능 프로파일·shape drift 모니터링까지 연결한다.
