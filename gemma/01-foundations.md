# 01. LLM·SFT·LoRA 기초

## 1. 언어 모델이 실제로 학습하는 것

언어 모델은 문장을 통째로 외우는 프로그램이 아니라, 앞의 토큰이 주어졌을 때 다음 토큰의 확률 분포를 계산하는 함수입니다.

```text
"배송이" → {"시작": 0.42, "완료": 0.18, "지연": 0.11, ...}
```

학습 데이터의 정답 토큰 확률이 높아지도록 cross-entropy loss를 줄입니다. 토큰은 단어와 같지 않습니다. 한국어 한 어절이 여러 토큰으로 나뉠 수 있으므로 “문자 수”와 “시퀀스 길이”는 다릅니다.

### 실습: 토큰 수 측정

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
samples = [
    "반품하고 싶어요.",
    "상품 수령 후 7일 이내에 주문 내역에서 반품을 신청하세요.",
]
for text in samples:
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(text, len(ids), ids)
```

과제:

1. 한국어, 영어, 코드, 이모지 문장을 각 10개 준비합니다.
2. 문자 수와 토큰 수의 비율을 비교합니다.
3. 가장 긴 샘플이 `max_length`에서 잘리는지 확인합니다.

## 2. 사전학습, 지시 튜닝, 내 데이터의 차이

- 사전학습: 대규모 텍스트로 언어와 지식을 학습합니다.
- 지시 튜닝: 사용자의 요청에 응답하는 형식을 학습합니다.
- 도메인 SFT: 특정 업무의 답변 방식과 제한을 학습합니다.
- 선호도 학습: 여러 답변 중 무엇이 더 좋은지 학습합니다.

처음 프로젝트에서는 지시 튜닝 모델(`-it`)에 지도 미세 조정(SFT)을 적용합니다. 소량의 데이터로 새로운 사실을 완벽히 주입하는 것보다, 이미 아는 지식을 원하는 형식과 정책으로 답하게 만드는 데 더 적합합니다. 자주 바뀌는 사실은 RAG나 도구 호출을 먼저 검토하세요.

## 3. SFT 데이터와 loss mask

다음 대화가 있다고 합시다.

```json
{
  "messages": [
    {"role": "user", "content": "반품 기간은?"},
    {"role": "assistant", "content": "수령 후 7일 이내입니다."}
  ]
}
```

일반적인 completion-only SFT는 사용자 토큰을 문맥으로 사용하지만 assistant 응답 토큰에만 손실을 계산합니다.

```text
user tokens       assistant tokens
[loss 제외]        [loss 계산]
```

전체 문장에 손실을 계산하면 사용자의 질문까지 생성하도록 학습되어 불필요한 동작이 생길 수 있습니다. 프레임워크마다 `loss_mask`, `completion_only_loss`, data collator 등 표현이 다르므로 실제 배치의 label에서 마스킹 값(`-100`)을 확인해야 합니다.

## 4. 전체 미세 조정과 PEFT

### 전체 미세 조정

모든 가중치를 업데이트합니다.

- 장점: 변화 용량이 가장 큼
- 단점: 많은 VRAM과 체크포인트 공간, 기존 능력 손상 위험
- 사용: 충분한 데이터·예산이 있고 큰 행동 변화를 검증할 때

### LoRA

기본 가중치 `W`는 고정하고 작은 저랭크 행렬 `A`, `B`만 학습합니다.

```text
W' = W + scale × B × A
scale ≈ alpha / rank
```

- `rank(r)`: 어댑터 표현 용량. 높을수록 학습 파라미터와 메모리 증가
- `alpha`: LoRA 업데이트 크기 조절
- `target_modules`: 어댑터를 삽입할 선형 계층
- `dropout`: 작은 데이터의 과적합을 줄일 수 있지만 항상 이득은 아님

### QLoRA

고정된 기본 모델을 보통 4비트로 로드하고 LoRA 어댑터는 더 높은 정밀도로 학습합니다.

- 장점: VRAM 크게 절감
- 단점: 하드웨어·커널·패키지 호환성 증가, 약간의 속도/품질 절충
- 주의: “4비트 모델의 가중치를 직접 학습”하는 것과 같지 않습니다.

## 5. 배치 크기와 그래디언트 누적

```text
effective batch
= micro batch
× gradient accumulation
× data parallel device 수
```

예:

```text
micro batch 1 × accumulation 8 × GPU 2 = effective batch 16
```

모델 병렬화는 모델 하나를 여러 장치에 나누므로 위의 데이터 병렬 장치 수와 구분해야 합니다.

## 6. 핵심 하이퍼파라미터

| 항목 | 작은 시작값 | 너무 작을 때 | 너무 클 때 |
|---|---:|---|---|
| learning rate | LoRA `1e-4`~`2e-4` | 거의 학습 안 됨 | 불안정·망각 |
| rank | 8 | 표현력 부족 | 메모리·과적합 |
| sequence length | 256~512 | 중요한 문맥 잘림 | VRAM 급증 |
| epochs | 1~3 | 미학습 | 암기·일반화 저하 |
| warmup ratio | 0.03~0.1 | 초반 불안정 | 유효 학습 구간 감소 |
| weight decay | 0.0~0.1 | 과적합 가능 | 학습 억제 |

이 값은 정답이 아니라 실험 시작점입니다. 한 실험에서 여러 변수를 동시에 바꾸지 마세요.

## 7. 추론 설정을 학습 품질과 혼동하지 않기

같은 모델도 생성 파라미터에 따라 출력이 달라집니다.

- `do_sample=False`: 재현 가능한 greedy 기준 평가
- `temperature`: 분포의 날카로움
- `top_p`, `top_k`: 후보 토큰 제한
- `max_new_tokens`: 생성 길이 상한

튜닝 전후 비교에서는 동일한 프롬프트와 생성 설정을 사용합니다. 먼저 greedy로 회귀 테스트하고, 창작 품질은 별도 sampling 평가로 다룹니다.

## 8. 가장 흔한 실패

### 손실은 감소하지만 답변이 나쁨

- 학습과 추론 템플릿 불일치
- assistant가 아닌 전체 토큰에 loss 적용
- 데이터 중복 또는 평가 누수
- 답변 품질이 낮거나 서로 모순됨
- 너무 높은 학습률/epoch

### OOM

다음 순서로 하나씩 적용합니다.

1. `micro_batch_size=1`
2. 시퀀스 길이 축소
3. QLoRA/4비트
4. gradient checkpointing
5. optimizer/attention 최적화
6. 모델 크기 축소
7. 모델·데이터 병렬화

### 학습이 너무 느림

- CPU 전처리 병목과 GPU 병목을 구분
- padding 비율 측정
- 길이가 비슷한 샘플 묶기 또는 packing 검토
- GPU가 지원하면 BF16, Flash Attention 검토
- 첫 실행의 컴파일/JIT 시간을 반복 실행과 구분

## 9. 개념 확인 실습

다음 질문에 코드와 로그를 근거로 답하세요.

1. 현재 실험의 학습 가능한 파라미터 비율은 몇 %인가?
2. assistant 토큰만 loss에 포함되는가?
3. 시퀀스 길이를 256→512로 바꾸면 피크 VRAM은 얼마나 변하는가?
4. rank 8→16에서 품질 향상이 반복 실험의 변동보다 큰가?
5. 어댑터만 저장한 결과와 병합 모델의 디스크 크기는 왜 다른가?

## 통과 기준

- [ ] SFT와 사전학습의 목적 차이를 설명한다.
- [ ] LoRA와 QLoRA의 메모리 절감 위치를 설명한다.
- [ ] effective batch size를 계산한다.
- [ ] loss mask를 실제 토큰 단위로 확인한다.
- [ ] OOM 대응 순서를 자신의 환경에 맞게 적었다.

다음: [데이터 설계와 평가](./02-data-and-evaluation.md)

