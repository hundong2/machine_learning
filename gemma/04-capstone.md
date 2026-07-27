# 04. 전문가 캡스톤: 재현 가능한 Gemma 업무 어시스턴트

## 프로젝트 목표

작은 고객지원 예제를 자신의 도메인으로 교체하고, 두 개 이상의 프레임워크에서 Gemma를 튜닝한 뒤 품질·비용·운영성을 근거로 하나를 선택합니다.

완료 산출물:

```text
capstone/
├── problem-statement.md
├── data-card.md
├── dataset/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── test.jsonl
│   └── challenge.jsonl
├── experiments/
│   ├── baseline/
│   ├── framework-a/
│   └── framework-b/
├── evaluation/
│   ├── rubric.md
│   ├── predictions.jsonl
│   └── report.md
├── model-card.md
├── deployment-runbook.md
└── decision-record.md
```

## 1단계: 문제 정의

다음을 한 페이지로 씁니다.

- 사용자는 누구인가?
- 모델이 받아야 할 입력과 반환할 출력은?
- 답하면 안 되는 요청은?
- RAG/규칙/작은 분류기로 해결할 수 없는 이유는?
- 품질·latency·비용 목표는?
- 실패했을 때 피해와 완화책은?

튜닝이 필요 없다는 결론도 유효합니다. 그 판단을 데이터로 설명해야 합니다.

## 2단계: 데이터 카드

최소 항목:

- 출처와 수집 권한
- 라이선스
- 개인정보 처리
- 생성/정제 방법
- 언어·도메인·기간 분포
- train/validation/test 분할 기준
- 알려진 편향과 누락
- 삭제/정정 절차
- dataset hash

권장 규모는 문제에 따라 다릅니다. 먼저 50~200개 고품질 예제로 행동 변화가 보이는지 확인하고, 오류 분석을 근거로 확장합니다.

## 3단계: 베이스라인

튜닝 전에 세 기준을 비교합니다.

1. zero-shot
2. few-shot prompt
3. 가능하면 RAG 또는 규칙

튜닝 모델은 단순 zero-shot만 이기는 것이 아니라, 더 저렴한 대안보다 의미 있게 좋아야 합니다.

## 4단계: 프레임워크 A

추천:

- Hugging Face 또는 Keras로 해석 가능한 기준 구현
- 20-step smoke
- 1 epoch
- rank/lr 작은 탐색
- seed 3개

모든 run에서:

- resolved config
- Git SHA
- environment
- dataset hash
- train/eval curve
- peak VRAM/tokens/s
- predictions

을 저장합니다.

## 5단계: 프레임워크 B

목적에 맞춰 선택:

- Unsloth: 단일 GPU 효율
- LLaMA-Factory: 설정/UI
- Axolotl: 운영/멀티 GPU
- Gemma JAX/Keras Distributed: TPU·연구
- Vertex/GKE: 클라우드 운영

A와 모델, 데이터, step, effective batch, target modules, generation 설정을 최대한 동일하게 맞춥니다.

## 6단계: 실험 설계

한 번에 한 변수만:

```text
E0: baseline
E1: LoRA r=8, lr=1e-4
E2: LoRA r=16, lr=1e-4
E3: LoRA r=8, lr=2e-4
E4: QLoRA r=8, lr=1e-4
```

최종 test는 선택이 끝난 뒤 한 번만 사용합니다. 반복해서 test를 보고 튜닝하면 test가 validation이 됩니다.

## 7단계: 오류 분류

각 실패를 하나 이상으로 태깅:

- 사실 오류
- 정책 위반
- 형식 오류
- 누락
- 환각
- 과도한 거절
- 장황함
- 언어/문체
- template/EOS
- 인프라/timeout

오류 유형별 빈도와 심각도를 전후 비교합니다.

## 8단계: 견고성

- 질문 표현 변화
- 오탈자
- 매우 짧거나 긴 입력
- 모순된 정보
- 정책 경계
- prompt injection
- 개인정보 요청
- 다국어/코드 스위칭
- 동시 요청과 긴 context

## 9단계: 배포 기술 검증

하나를 선택:

- adapter + Transformers
- merged Safetensors + vLLM
- GGUF + llama.cpp
- Vertex endpoint
- GKE Deployment

오프라인 품질, cold start, tokens/s, p95 latency, peak VRAM, 시간당 비용을 기록합니다.

## 10단계: 의사결정 기록

예시 가중치:

| 기준 | 가중치 | A | B |
|---|---:|---:|---:|
| 품질 | 35 |  |  |
| 안전/견고성 | 20 |  |  |
| 학습 비용 | 10 |  |  |
| 추론 비용/latency | 15 |  |  |
| 운영·재현성 | 15 |  |  |
| 개발 경험 | 5 |  |  |

점수뿐 아니라 근거 artifact 링크를 붙입니다.

## 최종 심사 체크리스트

- [ ] 문제와 비목표가 명확하다.
- [ ] 데이터 권리·개인정보·분할이 문서화됐다.
- [ ] prompt/RAG 기준선이 있다.
- [ ] 두 프레임워크를 공정 비교했다.
- [ ] seed 반복과 test 격리가 지켜졌다.
- [ ] 자동·사람·안전 평가가 있다.
- [ ] artifact를 새 환경에서 재현했다.
- [ ] 비용과 리소스 정리 절차가 있다.
- [ ] model card와 deployment runbook이 있다.
- [ ] 왜 이 프레임워크를 선택했는지 증거로 설명한다.

이 체크리스트를 모두 통과하면 단순 튜토리얼 실행을 넘어, Gemma 튜닝 실험을 설계·검증·운영할 수 있는 수준에 도달한 것입니다.

