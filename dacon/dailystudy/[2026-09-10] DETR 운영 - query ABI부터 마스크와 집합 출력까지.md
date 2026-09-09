<!-- curriculum: cycle=3; level=production-engineering; source_index=7/18; source=02-06.FCOS_DETR.md; part=2/2 -->

# DETR 운영: query ABI부터 마스크와 집합 출력까지

## 학습 진도

| 항목 | 내용 |
| --- | --- |
| 날짜 | 2026-09-10 |
| 회차·수준 | 3회차 · 실무 엔지니어 (`production-engineering`) |
| 현재 소스 | 7/18 · `02-06.FCOS_DETR.md` |
| Part | 2/2 · DETR serving·운영 계약 |
| 이전 소스 | 7/18 · `02-06.FCOS_DETR.md` Part 1/2 · FCOS point ABI와 후보 예산 |
| 다음 소스 | 8/18 · `02-07.ViT.md` · ViT 동적 해상도와 배포 운영 |

## 오늘의 운영 질문

같은 DETR 가중치를 Python과 C# 서비스에 올렸는데 Python은 객체 3개, C#은 객체 97개를 반환했다. raw tensor의 shape도 checksum도 같다. 무엇이 달랐을까?

```text
image -> resize/pad -> backbone -> flatten + position -> encoder memory
      -> object query -> decoder layers -> class logits + normalized CXCYWH
      -> no-object filtering -> coordinate restore -> response set
```

DETR는 NMS를 기본 후처리에서 없앴지만 후처리 자체를 없애지는 않았다. `no-object` class 위치, softmax 축, query 수, 좌표 형식, padding attention mask, decoder layer 선택, threshold와 정렬 규칙이 모두 릴리스 ABI다. 이 글은 그 경계를 하나의 재현 가능한 운영 계약으로 묶는다.

## 학습 목표

이 글을 마치면 다음을 할 수 있다.

1. object query와 영상 token을 서로 다른 ABI로 정의한다.
2. 가변 크기 batch의 image mask를 feature attention mask까지 추적한다.
3. 마지막 decoder 출력과 보조 출력을 이름·shape·dtype으로 고정한다.
4. `CXCYWH` 정규화 box를 원본 영상 `XYXY`로 복원한다.
5. `no-object` 확률과 foreground score로 NMS 없는 결과 집합을 만든다.
6. Python, C++17, C#에서 같은 golden 결과를 재현한다.
7. ONNX graph 경계, 동적 축, FP16 구간과 메모리 예산을 정한다.
8. query 포화, padding 누출, 중복 예측, decoder drift를 관측하고 rollback한다.

## 선수 지식과 기호

- CNN feature map과 Transformer encoder·decoder
- softmax, cross entropy, `no-object` class
- `CXCYWH`, `XYXY`, IoU와 GIoU
- Hungarian matching과 일대일 집합 손실
- ONNX Runtime, mixed precision, p50·p95·p99

| 기호 | 뜻 |
| --- | --- |
| $N$ | batch 크기 |
| $C$ | foreground class 수 |
| $Q$ | 고정 object query 수 |
| $D$ | Transformer embedding 차원 |
| $H_f,W_f$ | backbone feature 높이와 너비 |
| $S=H_fW_f$ | encoder token 수 |
| $L_d$ | decoder layer 수 |
| $Z^{(l)}$ | decoder layer $l$의 class logit |
| $B^{(l)}$ | decoder layer $l$의 정규화된 `CXCYWH` box |
| $V$ | 유효한 image 또는 feature 위치 mask |
| $K$ | attention에서 무시할 위치 mask |
| $M_n$ | image $n$의 실제 객체 수 |

이 글은 image와 feature를 `NCHW`, Transformer token을 `NSD`, class logit을 `(N,Q,C+1)`, box를 `(N,Q,4)`로 둔다. 마지막 class index $C$가 `no-object`이며, box는 padded model input을 기준으로 정규화된 `CXCYWH`다.

## 1. 원본과 앞선 회차에서 이번에 확장하는 것

원본 [02-06.FCOS_DETR.md](../05.ImageClassification/02-06.FCOS_DETR.md)는 고정 query, 이분 매칭, 집합 예측의 전환점을 소개한다. 1회차는 matching cost와 `no-object`를 수식으로 풀었고, 2회차는 matcher·set criterion·보조 손실을 실행 코드로 연결했다. 이번 글은 학습 결과가 실제 서비스에서 같은 의미를 유지하게 만든다.

| 과거 초점 | 오늘 추가하는 운영 계약 |
| --- | --- |
| object query와 set prediction 직관 | query 수·순서·embedding 차원·class map의 versioned schema |
| class·L1·GIoU matching | matcher는 학습 graph 밖, serving graph에는 포함하지 않는 경계 |
| padding mask 단위 테스트 | resize/pad metadata부터 feature mask까지 이어지는 계보 |
| 마지막 layer와 auxiliary loss | raw output 이름과 decoder layer 선택을 고정 |
| `no-object` target 학습 | softmax 축·배경 index·threshold·deterministic sort 계약 |
| 단일 PyTorch 구현 | Python·C++·C#·ONNX provider의 단계별 parity gate |

원문의 표현은 다음처럼 교정해서 읽어야 한다.

1. self-attention만으로 중복이 사라지는 것이 아니다. 일대일 matching을 사용하는 set loss가 직접적인 학습 제약이고 decoder attention은 slot 간 조정을 돕는다.
2. matched query만 학습되는 것이 아니다. box loss는 matched query에만 적용하지만 unmatched query도 `no-object` classification loss를 받는다.
3. 원형 DETR matcher는 class와 L1뿐 아니라 GIoU 비용도 결합한다.
4. NMS가 없다는 말은 response 조립이 없다는 뜻이 아니다. `no-object` 제거, score 선택, 좌표 복원, clip, 정렬이 남는다.
5. query가 100개라는 값은 보편 상수가 아니다. 모델 artifact의 capacity와 output shape를 결정하는 계약이다.
6. crowd 장면에서 NMS 오류를 피할 가능성은 있지만 query capacity와 feature 해상도 때문에 자동으로 우수해지는 것은 아니다.

## 2. 직관: DETR 서비스는 고정 slot을 의미 있는 집합으로 바꾼다

모델은 매 요청마다 정확히 $Q$개의 slot을 출력한다. 각 slot에는 $C+1$개 class logit과 box 네 값이 있다. 그러나 API response는 보통 가변 길이다.

```text
fixed model output                         variable response
(Q, C+1), (Q, 4)  -> background filter -> list[detection]
```

slot 7이 오늘은 사람이고 다음 영상에서는 자동차여도 정상이다. query index를 class나 객체 ID처럼 외부 API에 노출하면 잘못된 상태성이 생긴다. 반대로 모든 query를 무조건 detection으로 내보내면 `no-object` slot이 수십 개의 가짜 객체가 된다.

NMS 없는 serving의 핵심은 “아무것도 하지 않음”이 아니라, 학습 때 정의한 집합 의미를 손상하지 않는 최소 변환만 수행하는 것이다.

## 3. 릴리스 manifest와 raw tensor ABI

### 3.1 최소 manifest

모델 파일 옆 manifest에는 적어도 다음 정보가 있어야 한다.

```json
{
  "schema_version": 1,
  "model_family": "detr",
  "input_layout": "NCHW",
  "input_dtype": "float32",
  "query_count": 6,
  "embedding_dim": 32,
  "foreground_classes": ["person", "bicycle", "car"],
  "no_object_index": 3,
  "box_format": "normalized_cxcywh",
  "box_reference": "padded_model_input",
  "final_logits_name": "pred_logits",
  "final_boxes_name": "pred_boxes",
  "aux_outputs_exported": false,
  "softmax_in_graph": false,
  "score_rule": "max_foreground_probability",
  "threshold": 0.7,
  "sort_rule": "score_desc_query_index_asc"
}
```

가중치 hash만 같아도 manifest가 다르면 같은 detector가 아니다. 특히 `no_object_index`, `softmax_in_graph`, `box_format`, `box_reference`는 shape가 맞아도 조용히 결과를 망가뜨린다.

### 3.2 최종 출력

마지막 decoder layer의 raw 출력은 다음과 같다.

$$
Z=Z^{(L_d)}\in\mathbb{R}^{N\times Q\times(C+1)}
$$

$$
B=B^{(L_d)}\in[0,1]^{N\times Q\times4}
$$

box channel 순서는 다음으로 고정한다.

$$
B_{nq}=(c_x,c_y,w,h)
$$

`(N,Q,C+1)`을 `(N,C+1,Q)`로 잘못 읽어도 전체 원소 수와 checksum은 같다. 비정사각 shape와 index-coded fixture를 사용해야 transpose 오류를 찾을 수 있다.

### 3.3 보조 decoder 출력

학습 graph는 중간 layer 출력을 사용할 수 있다.

$$
Z_{aux}\in\mathbb{R}^{(L_d-1)\times N\times Q\times(C+1)}
$$

$$
B_{aux}\in\mathbb{R}^{(L_d-1)\times N\times Q\times4}
$$

production inference는 일반적으로 마지막 layer만 반환한다. exporter가 list를 여러 ONNX output으로 펼치면 소비자가 첫 layer를 final로 오인할 수 있다. 출력 이름을 `pred_logits`, `pred_boxes`로 고정하고 auxiliary output은 `aux_00_logits`처럼 별도 namespace에 둔다.

## 4. image mask에서 attention mask까지

### 4.1 mask polarity

전처리기는 실제 image content를 `True`로 표시하는 mask를 만들기 쉽다.

$$
V^{img}_{nyx}=1\quad\text{if pixel }(y,x)\text{ belongs to content}
$$

PyTorch `key_padding_mask`는 반대 의미다.

$$
K_{ns}=1\quad\text{if token }s\text{ must be ignored}
$$

따라서 feature 크기로 축소하고 같은 flatten 순서를 적용한 뒤 한 번 부정한다.

$$
K=\neg\operatorname{flatten}(V^{feat})
$$

`True=valid`를 그대로 전달하면 실제 영상 token을 가리고 padding만 읽는다.

### 4.2 mask downsampling

원본 content 크기가 $(H_r,W_r)$이고 오른쪽·아래로 padding했다고 하자. backbone feature의 각 위치가 content와 겹치는지 정하는 규칙을 모델과 함께 고정해야 한다.

가장 단순한 nearest 규칙은 다음과 같다.

$$
V^{feat}=\operatorname{InterpolateNearest}(V^{img},H_f,W_f)
$$

그러나 convolution receptive field는 padding 경계를 넘어설 수 있다. 정확한 목표는 “순수 content receptive field”가 아니라 학습 때 사용한 mask semantics의 재현이다. export 전 PyTorch와 ONNX에서 동일 image·mask fixture의 memory와 final logit을 비교한다.

### 4.3 shape 추적

입력이 `(N,3,96,128)`, backbone stride가 16, $D=32$, $Q=6$, $C=3$, $L_d=3$이면 다음과 같다.

```text
image                    (N, 3, 96, 128)
image valid mask         (N, 96, 128)
backbone feature         (N, 32, 6, 8)
feature valid mask       (N, 6, 8)
flattened source         (N, 48, 32)
key padding mask         (N, 48)
encoder memory           (N, 48, 32)
query embedding          (6, 32)
decoder hidden layers    (3, N, 6, 32)
class logits             (3, N, 6, 4)
normalized boxes         (3, N, 6, 4)
serving final logits     (N, 6, 4)
serving final boxes      (N, 6, 4)
```

### 4.4 padding metamorphic test

content와 mask는 유지하고 padding 값만 바꾼다.

$$
X_1=X\odot V+p_1(1-V)
$$

$$
X_2=X\odot V+p_2(1-V)
$$

올바른 mask 계약이면 허용 오차 안에서 결과가 같아야 한다.

$$
f(X_1,V)\approx f(X_2,V)
$$

이 테스트는 polarity 반전, mask 누락, flatten 순서 불일치를 잘 잡는다. 단, image padding 값이 backbone convolution을 통해 content feature에 영향을 주는 설계라면 feature-level fixture와 end-to-end fixture를 나누어 원인을 격리한다.

## 5. object query와 capacity 계약

query embedding은 영상 patch가 아니라 학습되는 출력 slot의 초기 상태다.

$$
E_q\in\mathbb{R}^{Q\times D}
$$

batch마다 같은 parameter를 복제하지만 decoder가 서로 다른 encoder memory를 읽는다.

$$
H^{(l)}\in\mathbb{R}^{N\times Q\times D}
$$

$$
Z^{(l)}=W_cH^{(l)}+b_c
$$

$$
B^{(l)}=\sigma(\operatorname{MLP}(H^{(l)}))
$$

$Q$는 다음 세 가지를 동시에 정한다.

- 한 영상에서 표현할 수 있는 detection 수의 상한
- decoder self-attention의 대략적인 $O(Q^2D)$ 비용
- raw output과 response filtering의 메모리·latency

$M_n>Q$인 학습 sample을 조용히 자르면 capacity 문제가 데이터 손실로 숨는다. ingest 단계에서 `gt_count_gt_query_count`를 계측하고 crop·tiling 또는 query 수 변경을 설계한다.

query 수를 늘리는 변경은 단순 config 수정이 아니다. query embedding parameter shape, 학습 분포, ONNX output shape, downstream buffer 크기, latency budget이 함께 바뀐다.

## 6. 학습과 serving의 경계

### 6.1 Hungarian matching은 학습 경로다

image $n$의 비용 행렬은 다음 shape다.

$$
\mathcal{C}^{(n)}\in\mathbb{R}^{Q\times M_n}
$$

대표 비용은 다음과 같다.

$$
\mathcal{C}_{qm}
=
-\lambda_{cls}p_q(c_m)
+\lambda_{L1}\lVert\hat b_q-b_m\rVert_1
-\lambda_{giou}\operatorname{GIoU}(\hat b_q,b_m)
$$

Hungarian solver는 `detach`된 비용으로 integer index를 고른다. serving에는 GT가 없으므로 matcher도 없다.

```text
training: prediction -> detached cost -> CPU assignment -> set loss
serving:  prediction -> foreground/no-object probabilities -> response set
```

matcher를 ONNX graph에 넣으려다 custom op를 늘리는 것은 inference 의미에 필요하지 않다.

### 6.2 unmatched query도 학습된다

matched query는 foreground class와 box loss를 받고, unmatched query는 `no-object` class loss를 받는다.

$$
t_q=
\begin{cases}
c_m,&q\leftrightarrow m\\
\varnothing,&\text{otherwise}
\end{cases}
$$

serving에서 `no-object` logit을 버리고 foreground끼리만 softmax하면 이 학습 의미가 사라진다.

## 7. stable softmax와 response 집합

### 7.1 softmax 축과 안정화

query $q$의 class 확률은 마지막 축에서 계산한다.

$$
p_{qk}=\frac{\exp(z_{qk}-m_q)}{\sum_{j=0}^{C}\exp(z_{qj}-m_q)}
$$

$$
m_q=\max_j z_{qj}
$$

query 축으로 softmax하면 $Q$개 slot이 class 확률을 나눠 갖게 되어 전혀 다른 모델이 된다. FP16 logit도 softmax와 threshold 비교 구간에서는 FP32 승격을 권장한다.

### 7.2 foreground score

각 query의 foreground class와 score는 다음과 같다.

$$
\hat c_q=\operatorname*{arg\,max}_{0\le k<C}p_{qk}
$$

$$
s_q=\max_{0\le k<C}p_{qk}
$$

기본 response rule은 $s_q\ge\tau$인 query를 남기는 것이다. `no-object`가 argmax인지도 함께 요구하는 정책을 쓸 수 있지만, 기존 threshold와 동일하지 않으므로 별도 version으로 관리한다.

### 7.3 결정적 정렬

NMS는 없지만 API 순서를 안정화할 필요는 있다. 다음 key를 사용한다.

```text
(-score, query_index)
```

score가 같은 경우 query index 오름차순으로 정한다. query index는 정렬 tie-break일 뿐 영속적인 object ID는 아니다.

### 7.4 중복은 관측 대상이다

원형 DETR의 기본 serving에 class별 NMS를 갑자기 추가하면 학습된 set semantics와 validation metric이 바뀐다. 대신 같은 class의 고점수 box pair가 IoU 임계값을 넘는 비율을 관측한다.

$$
r_{dup}=\frac{\#\{(i,j):c_i=c_j,\ s_i,s_j\ge\tau,\ \operatorname{IoU}(b_i,b_j)>u\}}{\max(1,\#\text{detections})}
$$

중복률이 상승하면 decoder layer 선택, checkpoint, class map, threshold, fine-tuning 상태를 먼저 조사한다.

## 8. 정규화 `CXCYWH`에서 원본 `XYXY`로

### 8.1 padded input pixel 좌표

model input 크기를 $(H_{in},W_{in})$이라 하자.

$$
x_0=(c_x-w/2)W_{in}
$$

$$
y_0=(c_y-h/2)H_{in}
$$

$$
x_1=(c_x+w/2)W_{in}
$$

$$
y_1=(c_y+h/2)H_{in}
$$

width에 $H_{in}$을 곱하는 축 교환은 정사각 입력 테스트에서 드러나지 않는다. 반드시 비정사각 fixture를 둔다.

### 8.2 resize·padding 역변환

원본 크기를 $(H_0,W_0)$, 실제 정수 resize 결과를 $(H_r,W_r)$, 왼쪽·위 padding을 $(p_x,p_y)$라 한다.

$$
a_x=\frac{W_r}{W_0},\qquad a_y=\frac{H_r}{H_0}
$$

각 x, y 좌표는 다음처럼 복원한다.

$$
x^{orig}=\frac{x^{pad}-p_x}{a_x}
$$

$$
y^{orig}=\frac{y^{pad}-p_y}{a_y}
$$

마지막에 x는 $[0,W_0]$, y는 $[0,H_0]$로 clip한다. 반열린 continuous box를 중간에 정수화하지 않는다.

### 8.3 padding 중심 box

DETR에는 FCOS처럼 point valid mask로 후보를 바로 제거하는 단계가 없다. decoder query가 padding을 보지 못하도록 attention mask를 적용하고, 최종 box가 content rectangle 밖에 많이 놓이는 현상은 metric으로 감시한다. box center가 content 밖이라고 무조건 제거하는 정책은 학습·평가와 다를 수 있으므로 별도 postprocess revision이 필요하다.

## 9. NumPy 수작업 검증

다음은 실행 가능한 독립 예제다. 비정사각 input, 큰 logit, `no-object`, score tie를 모두 포함한다.

```python
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cxcywh_to_original(boxes, input_hw, original_hw, resized_hw, pad_xy):
    h_in, w_in = input_hw
    h0, w0 = original_hw
    hr, wr = resized_hw
    px, py = pad_xy
    cx, cy, w, h = np.moveaxis(boxes.astype(np.float64), -1, 0)
    padded = np.stack([
        (cx - 0.5 * w) * w_in,
        (cy - 0.5 * h) * h_in,
        (cx + 0.5 * w) * w_in,
        (cy + 0.5 * h) * h_in,
    ], axis=-1)
    padded[..., [0, 2]] = (padded[..., [0, 2]] - px) / (wr / w0)
    padded[..., [1, 3]] = (padded[..., [1, 3]] - py) / (hr / h0)
    padded[..., [0, 2]] = np.clip(padded[..., [0, 2]], 0.0, w0)
    padded[..., [1, 3]] = np.clip(padded[..., [1, 3]], 0.0, h0)
    return padded


# foreground 3 classes + 마지막 no-object
logits = np.array([
    [4.0, 1.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 5.0],
    [1.0, 4.0, 0.0, -1.0],
    [4.0, 1.0, 0.0, -1.0],
], dtype=np.float32)
boxes = np.array([
    [0.50, 0.50, 0.50, 0.50],
    [0.20, 0.20, 0.10, 0.10],
    [0.25, 0.50, 0.25, 0.50],
    [0.75, 0.50, 0.25, 0.50],
], dtype=np.float32)

prob = softmax(logits)
fg_class = prob[:, :-1].argmax(axis=-1)
fg_score = prob[:, :-1].max(axis=-1)
keep = np.flatnonzero(fg_score >= 0.90)
order = sorted(keep.tolist(), key=lambda q: (-fg_score[q], q))
xyxy = cxcywh_to_original(
    boxes, input_hw=(80, 120), original_hw=(60, 100),
    resized_hw=(60, 100), pad_xy=(10, 10),
)

assert order == [0, 2, 3]
assert fg_class[order].tolist() == [0, 1, 0]
np.testing.assert_allclose(
    xyxy[0], [20.0, 10.0, 80.0, 50.0], atol=1e-12
)
np.testing.assert_allclose(
    xyxy[2], [5.0, 10.0, 35.0, 50.0], atol=1e-12
)
assert prob[1, 3] > 0.97
print("order", order)
print("classes", fg_class[order].tolist())
print("scores", np.round(fg_score[order], 6).tolist())
print("first_box", xyxy[0].tolist())
```

예상 결과는 query 1이 높은 `no-object` 확률 때문에 탈락하고, 같은 score의 query 0과 3은 query index 순서로 정렬되는 것이다.

## 10. PyTorch Python 구현

다음은 실행 가능한 교육용 serving contract 예제다. 완전한 DETR backbone 대신 masked memory summary와 작은 decoder head를 사용해 mask polarity, layer 선택, shape, finite backward, padding 불변성을 검증한다.

```python
import torch
from torch import nn
from torch.nn import functional as F


class MiniDetrContract(nn.Module):
    def __init__(self, channels=8, dim=16, queries=6, classes=3, layers=3):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, 1)
        self.query = nn.Parameter(torch.randn(queries, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
            for _ in range(layers)
        ])
        self.class_head = nn.Linear(dim, classes + 1)
        self.box_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4)
        )

    def forward(self, feature, valid_mask):
        # feature: NCHW, valid_mask: NHW and True=valid
        if feature.shape[0] != valid_mask.shape[0] or feature.shape[-2:] != valid_mask.shape[-2:]:
            raise ValueError("feature/mask shape mismatch")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be bool")
        src = self.proj(feature).flatten(2).transpose(1, 2)  # NSD
        valid = valid_mask.flatten(1)                         # NS
        if not valid.any(dim=1).all():
            raise ValueError("each image needs at least one valid token")
        weight = valid.to(src.dtype).unsqueeze(-1)
        memory = (src * weight).sum(1) / weight.sum(1)
        hidden = self.query.unsqueeze(0) + memory.unsqueeze(1)
        logits_layers, boxes_layers = [], []
        for block in self.blocks:
            hidden = hidden + block(hidden)
            logits_layers.append(self.class_head(hidden))
            boxes_layers.append(self.box_head(hidden).sigmoid())
        return torch.stack(logits_layers), torch.stack(boxes_layers)


torch.manual_seed(20260910)
model = MiniDetrContract()
model.eval()
content = torch.randn(2, 8, 3, 4)
valid = torch.tensor([
    [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
    [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
], dtype=torch.bool)

feature_a = content.clone()
feature_b = content.clone()
feature_a.masked_fill_(~valid.unsqueeze(1), 0.0)
feature_b.masked_fill_(~valid.unsqueeze(1), 1000.0)
with torch.no_grad():
    logits_a, boxes_a = model(feature_a, valid)
    logits_b, boxes_b = model(feature_b, valid)

assert logits_a.shape == (3, 2, 6, 4)
assert boxes_a.shape == (3, 2, 6, 4)
torch.testing.assert_close(logits_a, logits_b, rtol=0.0, atol=0.0)
torch.testing.assert_close(boxes_a, boxes_b, rtol=0.0, atol=0.0)

# 학습 경로에서는 모든 query의 class loss와 matched query의 box loss를 분리한다.
model.train()
logits_layers, boxes_layers = model(feature_a, valid)
target_class = torch.full((2, 6), 3, dtype=torch.long)
target_class[0, 0] = 1
target_class[1, 2] = 0
class_loss = F.cross_entropy(logits_layers[-1].reshape(-1, 4), target_class.reshape(-1))
matched_pred = torch.stack([boxes_layers[-1, 0, 0], boxes_layers[-1, 1, 2]])
matched_target = torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]])
box_loss = F.l1_loss(matched_pred, matched_target)
aux_loss = sum(x.square().mean() for x in logits_layers[:-1]) * 0.01
loss = class_loss + 5.0 * box_loss + aux_loss
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
print("layer_shapes", tuple(logits_layers.shape), tuple(boxes_layers.shape))
print("padding_invariant", True)
print("loss", round(float(loss.detach()), 6))
```

이 코드는 설명용 축소 모델이지만 실행 가능하다. 실제 DETR에서는 masked mean 대신 positional encoding이 더해진 encoder memory와 decoder self/cross-attention을 사용한다.

## 11. C++17 후처리 예제

다음은 표준 라이브러리만 사용하는 실행 가능한 golden 예제다. production에서는 ONNX Runtime tensor에서 `logits`와 `boxes`를 읽되 같은 함수를 적용한다.

```cpp
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

struct Detection {
    int query;
    int label;
    double score;
    std::array<double, 4> box;
};

std::array<double, 4> softmax(const std::array<double, 4>& z) {
    const double m = *std::max_element(z.begin(), z.end());
    std::array<double, 4> p{};
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        p[i] = std::exp(z[i] - m);
        sum += p[i];
    }
    for (double& value : p) value /= sum;
    return p;
}

std::array<double, 4> restore(const std::array<double, 4>& b) {
    const double h_in = 80.0, w_in = 120.0;
    const double h0 = 60.0, w0 = 100.0;
    const double hr = 60.0, wr = 100.0;
    const double px = 10.0, py = 10.0;
    const double ax = wr / w0, ay = hr / h0;
    std::array<double, 4> out{
        ((b[0] - 0.5 * b[2]) * w_in - px) / ax,
        ((b[1] - 0.5 * b[3]) * h_in - py) / ay,
        ((b[0] + 0.5 * b[2]) * w_in - px) / ax,
        ((b[1] + 0.5 * b[3]) * h_in - py) / ay,
    };
    out[0] = std::clamp(out[0], 0.0, w0);
    out[2] = std::clamp(out[2], 0.0, w0);
    out[1] = std::clamp(out[1], 0.0, h0);
    out[3] = std::clamp(out[3], 0.0, h0);
    return out;
}

int main() {
    const std::array<std::array<double, 4>, 4> logits{{
        {{4, 1, 0, -1}}, {{0, 0, 0, 5}},
        {{1, 4, 0, -1}}, {{4, 1, 0, -1}}
    }};
    const std::array<std::array<double, 4>, 4> boxes{{
        {{0.50, 0.50, 0.50, 0.50}}, {{0.20, 0.20, 0.10, 0.10}},
        {{0.25, 0.50, 0.25, 0.50}}, {{0.75, 0.50, 0.25, 0.50}}
    }};
    std::vector<Detection> result;
    for (int q = 0; q < 4; ++q) {
        auto p = softmax(logits[q]);
        int label = int(std::max_element(p.begin(), p.begin() + 3) - p.begin());
        double score = p[label];
        if (score >= 0.90) result.push_back({q, label, score, restore(boxes[q])});
    }
    std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.query < b.query;
    });
    assert((std::vector<int>{result[0].query, result[1].query, result[2].query}
            == std::vector<int>{0, 2, 3}));
    assert(std::abs(result[0].box[0] - 20.0) < 1e-12);
    std::cout << std::fixed << std::setprecision(6)
              << result[0].box[0] << ' ' << result[0].box[1] << ' '
              << result[0].box[2] << ' ' << result[0].box[3] << '\n';
}
```

예상 출력은 `20.000000 10.000000 80.000000 50.000000`이다.

## 12. C# 후처리 예제

다음도 외부 패키지 없이 실행 가능한 같은 golden 계약이다.

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

sealed class Detection
{
    public int Query;
    public int Label;
    public double Score;
    public double[] Box = Array.Empty<double>();
}

static class Program
{
    static double[] Softmax(double[] z)
    {
        double m = z.Max();
        double[] p = z.Select(v => Math.Exp(v - m)).ToArray();
        double sum = p.Sum();
        return p.Select(v => v / sum).ToArray();
    }

    static double Clamp(double x, double lo, double hi)
    {
        return Math.Max(lo, Math.Min(hi, x));
    }

    static double[] Restore(double[] b)
    {
        const double hIn = 80.0, wIn = 120.0;
        const double h0 = 60.0, w0 = 100.0;
        const double hr = 60.0, wr = 100.0;
        const double px = 10.0, py = 10.0;
        double ax = wr / w0, ay = hr / h0;
        return new[] {
            Clamp(((b[0] - 0.5 * b[2]) * wIn - px) / ax, 0.0, w0),
            Clamp(((b[1] - 0.5 * b[3]) * hIn - py) / ay, 0.0, h0),
            Clamp(((b[0] + 0.5 * b[2]) * wIn - px) / ax, 0.0, w0),
            Clamp(((b[1] + 0.5 * b[3]) * hIn - py) / ay, 0.0, h0),
        };
    }

    static void Main()
    {
        double[][] logits = {
            new double[] {4, 1, 0, -1}, new double[] {0, 0, 0, 5},
            new double[] {1, 4, 0, -1}, new double[] {4, 1, 0, -1}
        };
        double[][] boxes = {
            new double[] {0.50, 0.50, 0.50, 0.50},
            new double[] {0.20, 0.20, 0.10, 0.10},
            new double[] {0.25, 0.50, 0.25, 0.50},
            new double[] {0.75, 0.50, 0.25, 0.50}
        };
        var result = new List<Detection>();
        for (int q = 0; q < logits.Length; ++q)
        {
            double[] p = Softmax(logits[q]);
            int label = Enumerable.Range(0, 3).OrderByDescending(k => p[k]).First();
            if (p[label] >= 0.90)
                result.Add(new Detection { Query = q, Label = label,
                                           Score = p[label], Box = Restore(boxes[q]) });
        }
        result = result.OrderByDescending(x => x.Score).ThenBy(x => x.Query).ToList();
        if (!result.Select(x => x.Query).SequenceEqual(new[] {0, 2, 3}))
            throw new Exception("query order mismatch");
        if (Math.Abs(result[0].Box[0] - 20.0) > 1e-12)
            throw new Exception("box mismatch");
        Console.WriteLine(string.Join(" ", result[0].Box.Select(x => x.ToString("F6"))));
    }
}
```

Python, C++17, C# 예제는 모두 `float64` 후처리 golden을 사용한다. 실제 서비스가 `float32`라면 상대·절대 허용 오차와 threshold 경계 fixture를 별도로 둔다.

## 13. 프레임워크 간 shape·layout·dtype 대응

| 의미 | PyTorch | ONNX | C++ Runtime | C# Runtime |
| --- | --- | --- | --- | --- |
| image | `NCHW`, `float32` | `[N,3,H,W]` | contiguous `float` | `DenseTensor<float>` |
| image mask | `NHW`, `bool` | `[N,H,W]`, `bool` | `bool` 대신 `uint8_t` 변환 주의 | `DenseTensor<bool>` 지원 확인 |
| final logits | `(N,Q,C+1)` | `pred_logits` | row-major `[n,q,k]` | row-major `[n,q,k]` |
| final boxes | `(N,Q,4)` | `pred_boxes` | `CXCYWH`, normalized | `CXCYWH`, normalized |
| softmax accumulator | 보통 FP32 | graph 또는 consumer | `double` golden, `float` production | `double` golden, `float` production |
| response box | tensor | graph 밖 권장 | `XYXY`, original pixel | `XYXY`, original pixel |

### 13.1 layout checksum의 한계

transpose는 원소 집합을 바꾸지 않으므로 sum·mean checksum이 같다. 다음 fixture가 더 강하다.

$$
z_{nqk}=10000n+100q+k
$$

consumer가 임의의 `(n,q,k)`를 읽어 예상 값을 확인하면 stride와 layout 오류를 잡을 수 있다.

### 13.2 dtype 경계

- backbone과 attention은 검증 후 FP16 또는 BF16을 사용할 수 있다.
- layer normalization, attention softmax, class softmax는 provider별 오차를 측정한다.
- box sigmoid 출력은 FP32로 승격해 좌표 복원과 clip을 수행한다.
- threshold 바로 근처 score는 작은 오차로 admission이 뒤집힌다.
- INT8 DETR는 calibration dataset에 aspect ratio, padding 비율, 객체 수 분포를 포함한다.

## 14. 테스트와 디버깅

### 14.1 단위 테스트 행렬

| 테스트 | 잡는 오류 |
| --- | --- |
| 비정사각 input | width·height 축 교환 |
| index-coded logits | `(N,Q,C+1)` transpose |
| 마지막 class만 큰 query | `no-object` index 누락 |
| 큰 양·음 logit | softmax overflow |
| score tie | runtime별 비결정적 순서 |
| padding 값 변경 | mask 누락·polarity 반전 |
| 모든 pixel padding | zero denominator와 invalid request |
| $M_n=0$ training batch | empty GT matcher·box loss |
| $M_n=Q$와 $M_n>Q$ | query capacity 경계 |
| box가 content 밖 | inverse geometry·clip 정책 |
| auxiliary output 존재 | 첫 layer를 final로 읽는 오류 |

### 14.2 단계별 parity

최종 detection만 비교하면 원인을 찾기 어렵다. 다음 순서로 좁힌다.

```text
preprocessed image/mask
-> backbone feature
-> encoder memory sample
-> final raw logits/boxes
-> probabilities/admitted query indices
-> padded XYXY
-> original XYXY
-> response ordering
```

각 단계에서 shape, dtype, min/max/mean, finite count와 고정 index sample을 저장한다. production 개인정보를 담은 원본 tensor 전체를 log로 남기지는 않는다.

### 14.3 threshold flip test

provider A와 B의 score가 각각 `0.69998`, `0.70002`이고 threshold가 `0.7`이면 raw 오차는 작지만 API 결과가 다르다. parity report는 다음 두 지표를 함께 낸다.

- raw score 최대·평균 절대 오차
- threshold admission 일치율과 뒤집힌 query 수

### 14.4 empty response

모든 query가 threshold 아래인 결과는 정상 값이다. downstream schema가 빈 배열을 허용하고, fallback으로 가장 높은 query 하나를 강제로 내보내지 않는지 테스트한다.

## 15. 성능·메모리·수치 안정성

### 15.1 주요 복잡도

encoder self-attention의 score memory는 대략 다음에 비례한다.

$$
O(NhS^2)
$$

decoder self-attention은 다음과 같다.

$$
O(NhQ^2)
$$

decoder cross-attention은 다음과 같다.

$$
O(NhQS)
$$

$h$는 attention head 수다. 입력 해상도가 커지면 $S=H_fW_f$가 증가해 encoder 비용이 급격히 커진다. query 수만 보고 capacity를 늘리면 decoder와 output 비용도 커진다.

### 15.2 score tensor 메모리 예산

attention score만 단순 계산하면 element 수는 다음과 같다.

$$
E_{enc}=NhS^2
$$

$$
E_{cross}=NL_dhQS
$$

예를 들어 $N=4$, $h=8$, $S=1200$, $Q=300$, $L_d=6$, FP16 2 byte라면 encoder layer 하나의 score만 약 92.16 MB다.

$$
4\times8\times1200^2\times2=92{,}160{,}000\text{ bytes}
$$

실제 peak는 QKV, activation, allocator workspace, residual과 provider fusion에 따라 더 크다. 공식은 capacity planning의 하한 추정이지 profiler를 대체하지 않는다.

### 15.3 batching

서로 다른 aspect ratio를 큰 rectangle로 padding해 batch하면 유효 token 비율이 낮아진다. attention mask가 정확해도 padded token의 메모리와 일부 연산은 남는다. aspect bucket을 사용하고 다음 값을 관측한다.

$$
r_{valid}=\frac{\sum V^{img}}{NH_{in}W_{in}}
$$

낮은 `valid_ratio` bucket의 p99와 OOM을 따로 본다.

### 15.4 안정화 체크

- softmax 전에 max logit을 뺀다.
- logits, boxes, attention output의 NaN·Inf를 release gate에서 차단한다.
- FP16과 FP32의 admitted query index를 비교한다.
- box width·height가 음수가 되지는 않지만 0 근처 collapse 비율을 본다.
- threshold와 class map은 model revision과 함께 고정한다.

## 16. ONNX·배포 설계

### 16.1 graph 경계 선택

권장 기본 경계는 다음과 같다.

```text
ONNX graph: image + mask -> pred_logits + pred_boxes
consumer: stable softmax -> filter -> restore -> deterministic sort
```

장점은 Python, C++, C# consumer의 의미를 작은 golden으로 검증하기 쉽다는 것이다. 단점은 후처리 구현이 여러 언어에 중복된다는 점이다.

softmax와 좌표 변환까지 graph에 넣을 수도 있다. 이 경우 manifest가 `probability` output임을 명시하고 consumer가 softmax를 다시 적용하지 않게 한다. dynamic ragged detection list를 graph 안에서 만드는 것은 provider 호환성과 디버깅 비용을 높일 수 있다.

### 16.2 동적 축

- batch 축 $N$은 허용 profile 안에서 dynamic으로 둘 수 있다.
- height·width가 dynamic이면 feature token $S$와 memory plan도 dynamic이다.
- query $Q$와 class $C+1$은 artifact마다 static으로 고정하는 편이 안전하다.
- mask shape는 image spatial shape와 함께 검증한다.
- output buffer를 이전 artifact의 $Q$로 고정 할당하지 않는다.

### 16.3 release gate

```text
artifact hash + manifest schema
-> input/output introspection
-> synthetic golden
-> real golden batch
-> provider parity
-> load/soak test
-> shadow traffic
-> canary
-> gradual rollout
```

provider parity는 최소한 다음을 포함한다.

- final logits·boxes의 shape와 dtype
- raw 최대·평균 절대 오차
- foreground label·admitted query 일치율
- original `XYXY` 오차와 empty response 일치
- 중복률, detection count 분포
- input profile별 peak memory와 p50·p95·p99

### 16.4 immutable bundle

rollback 단위는 다음 전체다.

```text
preprocess + mask semantics + model + class map + query schema
+ output names + decoder layer policy + softmax/filter rule
+ box transform + threshold + runtime/provider + golden fingerprints
```

model만 rollback하면 새 class map이나 box consumer와 옛 output이 섞일 수 있다.

## 17. 실무 실패 사례: NMS 없는 모델에 97개 객체가 나타났다

### 증상

- Python baseline은 image당 detection p50이 3개였다.
- 새 C# service는 p50이 97개, 거의 query 수와 같았다.
- raw `pred_logits` checksum과 `pred_boxes` checksum은 일치했다.
- latency는 낮았지만 downstream tracker의 CPU가 포화됐다.

### 조사

1. input과 raw output이 같으므로 backbone·runtime parity 문제를 제외했다.
2. C# consumer는 마지막 logit을 제거한 뒤 foreground 3개에만 softmax를 적용했다.
3. 따라서 모든 query의 foreground 확률 합이 1이 되었고 적어도 한 class가 높은 값을 받았다.
4. Python은 foreground와 `no-object`를 포함한 $C+1$개 전체에 softmax를 적용했다.
5. 별도로 C#은 score tie에서 불안정 정렬을 사용해 response 순서도 흔들렸다.

### 근본 원인

`no_object_index`와 `softmax_axis`가 model manifest에 없었고 consumer 테스트는 foreground 객체가 있는 query만 포함했다. 높은 `no-object` logit fixture가 없었다.

### 수정

1. `no_object_index=C`, `softmax_axis=-1`, `softmax_in_graph=false`를 schema에 추가했다.
2. 마지막 class logit이 5이고 foreground가 0인 golden query를 넣었다.
3. threshold admission index까지 Python·C++·C#에서 비교했다.
4. response를 `(-score, query_index)`로 안정 정렬했다.
5. `detections_per_image`, `background_argmax_ratio`, `admission_flip_count`에 canary 경보를 걸었다.

이 장애는 raw tensor checksum만으로 의미적 parity를 보장할 수 없음을 보여 준다.

## 18. 운영 모니터링과 장애 격리

| 단계 | metric | 이상 신호 |
| --- | --- | --- |
| preprocess | aspect bucket, valid ratio | 새 bucket·padding 급증 |
| feature mask | valid token count | 0 또는 profile band 이탈 |
| raw class | foreground/no-object logit quantile | 배경 logit 급락 |
| raw box | center·size quantile, finite count | width·height collapse |
| filter | admitted query count | p99가 $Q$에 근접 |
| geometry | outside-content·clipped box ratio | padding 계보 오류 |
| set quality | duplicate pair ratio | decoder/checkpoint drift |
| capacity | detections near $Q$, GT overflow proxy | query 포화 |
| runtime | p50·p95·p99, peak memory | profile별 SLO 초과 |

metric label에는 bundle ID, model revision, runtime provider, input profile을 넣는다. query index와 request ID를 label로 쓰면 cardinality가 폭발하므로 trace sample에만 남긴다.

장애 격리 순서는 다음이 실용적이다.

```text
detection count drift
-> raw logits drift인가?
-> softmax/no-object/filter drift인가?
-> box restore drift인가?
-> 특정 aspect bucket인가?
-> 특정 provider·precision인가?
```

## 19. 운영 체크리스트

### 모델·출력

- [ ] $Q$, $D$, $C+1$, class map과 `no-object` index가 manifest에 있다.
- [ ] final decoder output 이름을 introspection으로 확인한다.
- [ ] auxiliary output을 final로 읽지 않는다.
- [ ] box format과 normalization reference가 명시됐다.
- [ ] query index를 영속 object ID로 사용하지 않는다.

### mask·기하

- [ ] image valid mask와 attention ignore mask의 polarity가 구분됐다.
- [ ] mask downsampling과 flatten 순서가 학습·export·serving에서 같다.
- [ ] padding metamorphic test가 provider마다 통과한다.
- [ ] 실제 정수 resize 비율과 padding origin을 보존한다.
- [ ] 비정사각 input으로 x·y 축과 `CXCYWH` 변환을 검사한다.

### 수치·response

- [ ] $C+1$ 전체에 마지막 축 stable softmax를 적용한다.
- [ ] graph 안팎에서 softmax를 중복 적용하지 않는다.
- [ ] threshold, score rule, tie-break가 versioning됐다.
- [ ] empty response와 query 포화가 정상 처리된다.
- [ ] 기본 DETR에 임의 NMS를 추가하지 않는다.

### 릴리스·운영

- [ ] Python·C++·C# golden 결과가 일치한다.
- [ ] ONNX provider별 raw·admission·geometry parity를 확인한다.
- [ ] aspect bucket별 load·soak test와 memory profile이 있다.
- [ ] detection count, no-object ratio, duplicate ratio를 canary에서 본다.
- [ ] 전체 bundle의 atomic rollback이 준비됐다.

## 20. 연습문제

### 문제 1

$N=2$, $Q=100$, foreground class가 20개일 때 final logit과 box shape를 쓰고, FP32 raw output의 byte 수를 구하라.

### 문제 2

valid mask가 `True=content`이고 flatten 결과가 `[True, True, False, False]`다. PyTorch ignore mask를 쓰라. 부정을 빠뜨리면 무엇을 읽는가?

### 문제 3

한 query의 logit이 `(0,0,0,5)`이고 마지막 index가 `no-object`다. foreground만 잘라 softmax할 때와 네 class 전체에 softmax할 때의 최대 foreground 확률을 비교하라.

### 문제 4

model input은 `(H,W)=(80,120)`이고 box가 `(0.5,0.5,0.5,0.5)`다. padded input의 `XYXY`를 구하라. 원본은 `(60,100)`, resize도 `(60,100)`, padding은 `(p_x,p_y)=(10,10)`일 때 원본 box를 구하라.

### 문제 5

$N=4$, $h=8$, $S=1200$인 encoder attention score를 FP16으로 저장할 때 layer 하나의 byte 수와 decimal MB를 구하라.

### 문제 6

두 runtime의 raw score 최대 오차는 $4\times10^{-5}$로 작지만 threshold admission 결과가 다르다. 추가해야 할 parity 지표와 fixture를 설명하라.

### 문제 7

DETR response에서 같은 class의 고점수 중복 box가 증가했다. 바로 NMS를 넣는 대신 먼저 확인할 항목 네 가지를 쓰라.

## 21. 해답

### 해답 1

class logit은 `(2,100,21)`, box는 `(2,100,4)`다. element 수는 다음과 같다.

$$
2\times100\times(21+4)=5000
$$

FP32는 4 byte이므로 20,000 byte다.

### 해답 2

ignore mask는 `[False, False, True, True]`다. 부정을 빠뜨리면 content token을 무시하고 padding token을 attention memory로 읽는다.

### 해답 3

foreground 세 개만 softmax하면 각각 $1/3$이므로 최대값은 약 0.3333이다. 전체 네 class에 softmax하면 최대 foreground 확률은 다음과 같다.

$$
\frac{1}{3+e^5}\approx0.00660
$$

작은 threshold에서도 admission 결과가 달라질 수 있다.

### 해답 4

padded input에서는 다음과 같다.

$$
(x_0,y_0,x_1,y_1)=(30,20,90,60)
$$

resize 비율은 두 축 모두 1이고 padding을 빼면 원본 box는 다음과 같다.

$$
(20,10,80,50)
$$

### 해답 5

$$
4\times8\times1200^2\times2=92{,}160{,}000\text{ bytes}
$$

decimal 단위로 92.16 MB다. QKV·activation·workspace는 포함하지 않은 attention score만의 값이다.

### 해답 6

raw 최대·평균 오차 외에 threshold admission index 일치율과 flip count를 비교해야 한다. threshold 바로 아래와 위의 score를 만드는 fixture, 큰 양·음 logit, `no-object` 우세 query를 포함한다.

### 해답 7

final decoder layer 선택, checkpoint·bundle ID, class map과 `no-object` 처리, threshold·softmax 책임을 먼저 확인한다. padding mask polarity와 provider precision도 이어서 점검한다. NMS 추가는 response semantics를 바꾸는 별도 모델 변경으로 평가해야 한다.

## 22. 핵심 요약

1. DETR의 고정 $Q$개 output은 variable response가 아니라 raw slot ABI다.
2. query index는 class나 추적 ID가 아니며 query 수 변경은 모델·buffer·latency를 함께 바꾼다.
3. `True=valid` image mask를 feature 크기로 맞춘 뒤 `True=ignore` attention mask로 부정한다.
4. serving은 마지막 decoder의 `(N,Q,C+1)` logit과 정규화 `CXCYWH`를 명시적으로 선택한다.
5. `no-object`를 포함한 전체 class 축에 stable softmax를 적용해야 set loss의 의미가 유지된다.
6. NMS는 없지만 threshold, 좌표 복원, clip, 결정적 정렬은 남는다.
7. 비정사각·index-coded·padding metamorphic fixture가 shape와 mask 오류를 드러낸다.
8. Python·C++·C#·ONNX parity는 raw, admission, geometry, response 네 단계에서 확인한다.
9. input 해상도와 query 수는 attention memory·latency·capacity의 trade-off다.
10. preprocess, mask, model, class map, decoder policy, 후처리는 하나의 immutable rollback bundle이다.

## 다음 학습 예고

다음은 3회차 8/18 `02-07.ViT.md`로 이동한다. patchification과 positional embedding을 동적 해상도 ABI로 만들고, padding·interpolation·layout, attention memory, Python·C++·C#·ONNX 동등성, latency와 drift monitoring을 ViT 분류 서비스의 운영 계약으로 연결한다.
