==============================
오픈소스 AI 프로젝트 Top 리스트
==============================

설명
----
구글, 마이크로소프트, 메타 등에서 최근 GitHub에서 높은 스타를 받은 대표적인 오픈소스 AI 프로젝트들을 순위별로 소개합니다. (실무 활용도, 커뮤니티 활동, 문서 품질 고려)

1) Meta — LLaMA / Llama Ecosystem
---------------------------------
- 개요: 대규모 언어 모델(LLM)과 파생 생태계. 연구·서비스에서 폭넓게 활용.
- 주요 리포지토리:
  - LLaMA 모델 허브: https://github.com/meta-llama/llama
  - Llama Recipes(파인튜닝/추론): https://github.com/facebookresearch/llama-recipes
- 실무 포인트: 사내 데이터로 파인튜닝, 온프레미스 추론/프라이버시 통제.

2) Meta — Segment Anything (SAM)
--------------------------------
- 개요: 범용 이미지 세분화 모델. 프롬프트(포인트/박스) 기반 즉시 세그멘테이션.
- 리포지토리: https://github.com/facebookresearch/segment-anything
- 실무 포인트: 라벨링 비용 절감, 비전 파이프라인 전처리, 의료/지도/제조 등 응용.

3) Google — MediaPipe / Mediapipe Tasks
---------------------------------------
- 개요: 멀티모달(비전·오디오) ML 파이프라인 라이브러리. 크로스플랫폼 지원.
- 리포지토리:
  - MediaPipe: https://github.com/google/mediapipe
  - MediaPipe Tasks: https://github.com/google/mediapipe/tree/master/mediapipe/tasks
- 실무 포인트: 모바일/엣지 디바이스에서 실시간 추론(손/얼굴/포즈), 프로토타이핑 신속.

4) Microsoft — Semantic Kernel
------------------------------
- 개요: 에이전트/플러그인 구성, 메모리·플래너 등 LLM 앱 프레임워크.
- 리포지토리: https://github.com/microsoft/semantic-kernel
- 실무 포인트: 엔터프라이즈 워크플로우, 작업 분해·도구호출, .NET/Java/Python 통합.

5) Microsoft — DeepSpeed
------------------------
- 개요: 대규모 모델 학습/추론 최적화(ZeRO, 모듈 병렬화, 혼합정밀).
- 리포지토리: https://github.com/microsoft/DeepSpeed
- 실무 포인트: 수십억~수천억 파라미터 모델 학습 비용 절감, 멀티GPU/클러스터 확장.

6) Google — JAX / Flax
-----------------------
- 개요: XLA 기반 고성능 수치 연산 및 딥러닝 라이브러리(함수형 스타일).
- 리포지토리:
  - JAX: https://github.com/google/jax
  - Flax: https://github.com/google/flax
- 실무 포인트: TPU/GPU 최적화 연구, 미분/벡터화/자동 병렬화로 연구 생산성 향상.

7) Meta — PyTorch (AI 생태계 핵심)
---------------------------------
- 개요: 딥러닝 표준 프레임워크. 풍부한 생태계(TorchVision, Lightning 등).
- 리포지토리: https://github.com/pytorch/pytorch
- 실무 포인트: 연구→프로덕션 전환 용이, 커뮤니티/모델/튜토리얼 자원 풍부.

추가 트렌드 참고
----------------
- OpenAI — Whisper(음성 인식): https://github.com/openai/whisper
- Hugging Face — Transformers(LLM 허브): https://github.com/huggingface/transformers
- LangChain — 에이전트/툴콜링 프레임워크: https://github.com/langchain-ai/langchain
- vLLM — 고성능 LLM 서빙: https://github.com/vllm-project/vllm

비고
----
- 순위는 스타 수, 최근 커밋 활동, 산업 활용 레퍼런스 등을 종합한 실무 체감 기준입니다.
- 실제 도입 시 라이선스(상업 이용 제한 여부)와 하드웨어 요구사항을 확인하세요.