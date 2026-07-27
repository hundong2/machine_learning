# Gemma 파인튜닝 프레임워크 실전 학습 로드맵

Google의 [Gemma 모델 미세 조정](https://ai.google.dev/gemma/docs/tune?hl=ko) 페이지에 연결된 프레임워크와 플랫폼을, 처음 시작하는 사람이 실제 실험을 설계하고 운영할 수 있는 수준까지 단계적으로 학습하는 자료입니다.

> 기준일: 2026-07-27
>
> 기본 실습 모델: `google/gemma-3-1b-it`
>
> 권장 환경: Linux/WSL2 또는 Google Colab, Python 3.11~3.12, NVIDIA GPU
>
> 주의: 모델 다운로드 전 Hugging Face 또는 Kaggle에서 Gemma 이용약관에 동의해야 합니다.

## 이 자료를 마치면

- Transformer, 토큰, 손실, SFT, LoRA, QLoRA를 설명할 수 있습니다.
- 목적과 하드웨어에 맞는 튜닝 프레임워크를 선택할 수 있습니다.
- 동일한 데이터로 Keras, JAX, Hugging Face, 설정 기반 도구를 비교할 수 있습니다.
- 학습/검증 누수 없이 데이터셋을 만들고 기준 모델과 튜닝 모델을 평가할 수 있습니다.
- OOM, 잘못된 채팅 템플릿, 과적합, 체크포인트 불일치를 진단할 수 있습니다.
- 단일 GPU 실험을 멀티 GPU, GKE, Vertex AI로 확장하고 비용과 자원을 통제할 수 있습니다.
- 재현 가능한 실험 보고서와 모델 카드를 작성할 수 있습니다.

## 0. 가장 먼저 읽기

1. [학습 환경과 첫 실행](./00-start-here.md)
2. [LLM·SFT·LoRA 기초](./01-foundations.md)
3. [데이터 설계와 평가](./02-data-and-evaluation.md)
4. [`labs/` 실행 안내](./labs/README.md)

처음이라면 위 순서를 건너뛰지 마세요. 특히 채팅 템플릿과 데이터 분할을 이해하지 않고 튜닝하면 손실은 내려가도 실제 품질은 나빠질 수 있습니다.

## 1. 프레임워크별 실습

원문 페이지가 안내하는 항목을 빠짐없이 다룹니다.

| 순서 | 프레임워크 | 핵심 실습 | 추천 대상 |
|---:|---|---|---|
| 1 | [Keras + KerasHub LoRA](./frameworks/01-keras-lora.md) | 가장 짧은 코드로 LoRA, 백엔드 교체 | 딥러닝 입문자 |
| 2 | [Gemma 라이브러리 + JAX/Kauldron](./frameworks/02-gemma-jax.md) | 데이터 변환, 명시적 손실, 체크포인트 | 연구·JAX 사용자 |
| 3 | [Hugging Face Transformers + TRL + PEFT](./frameworks/03-huggingface-peft.md) | 재현 가능한 LoRA/QLoRA 기준 실험 | 대부분의 사용자 |
| 4 | [LLaMA-Factory](./frameworks/04-llamafactory.md) | YAML/WebUI, 학습·병합·서빙 | 빠른 반복 실험 |
| 5 | [XTuner](./frameworks/05-xtuner.md) | 보관 예제 재현과 최신 V1 구분 | 기존 XTuner 프로젝트 |
| 6 | [Unsloth](./frameworks/06-unsloth.md) | 저메모리 QLoRA, Colab, 내보내기 | 1 GPU·Colab 사용자 |
| 7 | [Axolotl](./frameworks/07-axolotl.md) | YAML 기반 학습, 멀티 GPU 확장 | 팀 단위 운영 |
| 8 | [Keras 분산 조정](./frameworks/08-keras-distributed.md) | DeviceMesh, LayoutMap, 모델 병렬화 | TPU/JAX·분산 학습 |

## 2. 클라우드·운영 실습

원문은 아래 두 항목도 “프레임워크 선택” 목록에 포함하지만, 정확히는 학습 실행·관리 플랫폼입니다.

1. [Google Cloud GKE: 컨테이너·Job·멀티 GPU](./cloud/01-gke.md)
2. [Google Cloud Vertex AI: 관리형 튜닝](./cloud/02-vertex-ai.md)
3. [모델 형식·병합·배포·운영](./03-deployment-and-operations.md)
4. [전문가 캡스톤 프로젝트](./04-capstone.md)

## 3. 추천 학습 일정

### 2주 입문 트랙

- 1~2일: 시작하기, LLM/LoRA 기초
- 3~4일: 데이터 검증과 베이스라인 평가
- 5~7일: Hugging Face LoRA 20-step 스모크 테스트
- 8~10일: Keras 또는 Unsloth로 같은 데이터 재실험
- 11~12일: 결과 비교, 오류 분석
- 13~14일: 모델 카드와 실험 보고서 작성

### 6주 실무 트랙

- 1주: 공통 기초와 데이터 파이프라인
- 2주: Keras, Hugging Face, Unsloth
- 3주: LLaMA-Factory, Axolotl, XTuner 비교
- 4주: JAX/Kauldron과 Keras 분산 학습
- 5주: GKE 또는 Vertex AI 중 하나
- 6주: 캡스톤, 회귀 평가, 배포·비용·안전 점검

## 4. 어떤 도구를 먼저 선택할까?

| 상황 | 첫 선택 | 이유 |
|---|---|---|
| 파이썬·딥러닝이 처음 | Keras | API가 짧고 개념을 눈으로 확인하기 쉽습니다. |
| 자료와 생태계가 중요 | Hugging Face | Transformers/TRL/PEFT 중심의 폭넓은 예제가 있습니다. |
| VRAM 12~24GB 한 장 | Unsloth 또는 HF QLoRA | 4비트 기반 저메모리 실습이 쉽습니다. |
| 코드를 최소화하고 싶음 | LLaMA-Factory | YAML과 WebUI로 학습부터 병합까지 이어집니다. |
| 팀 표준 YAML과 대규모 확장 | Axolotl | 단일 설정을 멀티 GPU·클라우드로 확장하기 좋습니다. |
| JAX 내부 구조를 배우고 싶음 | Gemma 라이브러리 | 데이터·모델·손실·샤딩을 명시적으로 다룹니다. |
| TPU에서 모델 병렬화 | Keras 분산 또는 JAX | DeviceMesh/LayoutMap 또는 JAX 샤딩을 다룹니다. |
| Kubernetes 운영이 필요 | GKE | 컨테이너, Job, Secret, 관측, 서빙을 직접 통제합니다. |
| 인프라 운영 없이 작업 제출 | Vertex AI | 데이터 업로드와 관리형 튜닝 작업에 집중합니다. |

## 5. 버전과 링크에 관한 중요한 메모

- Google 원문의 Keras 예제는 `keras_nlp`와 과거 Gemma 체크포인트를 사용합니다. 이 자료의 새 실습은 후속 패키지인 `keras_hub`를 우선 사용하고, 원문 코드는 비교 대상으로 남깁니다.
- LLaMA-Factory, XTuner, Unsloth, Axolotl 링크는 Google Gemma cookbook의 `.archive/` 예제로 연결됩니다. 최신 설치에 그대로 붙여 넣지 말고 각 장의 최신 공식 문법을 사용하세요.
- XTuner는 2025년 V1에서 초대형 MoE 중심으로 크게 바뀌었습니다. Gemma 1 보관 예제와 최신 V1을 같은 환경에 섞지 않습니다.
- 모델·CUDA·PyTorch·Transformers 조합은 빠르게 변합니다. 실제 프로젝트는 실행에 성공한 버전을 잠그고 환경 정보를 결과물과 함께 저장하세요.
- Gemma 4는 모든 프레임워크에서 같은 속도로 지원되지 않습니다. 먼저 Gemma 3 1B로 파이프라인을 검증한 뒤 모델 지원표를 재확인해 교체하세요.

## 6. 완료 체크리스트

- [ ] `python labs/validate_dataset.py`가 모든 샘플을 통과한다.
- [ ] 학습/검증/테스트 데이터가 목적별로 분리되어 있다.
- [ ] 튜닝 전 출력과 평가값을 저장했다.
- [ ] 최소 한 프레임워크에서 LoRA 스모크 테스트를 완료했다.
- [ ] 다른 프레임워크에서 같은 데이터로 재현했다.
- [ ] 학습 가능한 파라미터 수와 피크 VRAM을 기록했다.
- [ ] 정량 지표뿐 아니라 실패·경계·안전 사례를 검토했다.
- [ ] 어댑터와 병합 모델의 차이를 설명하고 각각 추론했다.
- [ ] 비밀키가 Git에 포함되지 않았음을 확인했다.
- [ ] 캡스톤의 실험 보고서와 모델 카드를 완성했다.

## 공식 출발점

- [Gemma 모델 미세 조정](https://ai.google.dev/gemma/docs/tune?hl=ko)
- [Gemma 설정 및 이용약관](https://ai.google.dev/gemma/docs/setup)
- [Gemma 모델 카드](https://ai.google.dev/gemma/docs/model_card)
- [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
