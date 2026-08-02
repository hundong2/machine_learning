# 12. 공식 자료 지도와 용어집

## 조사 원칙

아래는 2026-08-02에 확인한 Google·Google DeepMind·공식 GitHub와 원 논문입니다. Preview API와 모델 이름은 변경될 수 있으므로 링크의 최신 상태를 우선합니다.

## 핵심 페이지

- [Gemini Robotics 1.5 VLA 랜딩 페이지](https://deepmind.google/models/gemini-robotics/vla/)
- [Gemini Robotics ER 2 개발자 개요](https://ai.google.dev/gemini-api/docs/robotics-overview)
- [Gemini Robotics ER 2 모델 카드](https://deepmind.google/models/model-cards/gemini-robotics-er-2/)
- [Gemini Robotics On-Device 2](https://deepmind.google/models/gemini-robotics/on-device/)
- [Gemini Robotics 안전 개요](https://deepmind.google/models/gemini-robotics/responsibly-advancing-ai-and-robotics/)

## ER 2 기능별 개발 문서

- [Spatial reasoning](https://ai.google.dev/gemini-api/docs/robotics-spatial)
- [Agentic vision](https://ai.google.dev/gemini-api/docs/robotics-agentic)
- [Task orchestration](https://ai.google.dev/gemini-api/docs/robotics-orchestration)
- [Robotics with streaming](https://ai.google.dev/gemini-api/docs/robotics-streaming)
- [Video understanding](https://ai.google.dev/gemini-api/docs/robotics-video-progress)
- [Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [API key 보안](https://ai.google.dev/gemini-api/docs/api-key)

## 공식 코드

- [google-gemini/robotics-samples](https://github.com/google-gemini/robotics-samples)
- [Getting Started notebook](https://github.com/google-gemini/robotics-samples/blob/main/Getting%20Started/gemini_robotics_er.ipynb)
- [Live API robot examples](https://github.com/google-gemini/robotics-samples/tree/main/live-api)
- [Google Gen AI Python SDK](https://github.com/googleapis/python-genai)
- [공식 SDK 문서](https://googleapis.github.io/python-genai/)

과거 `robotics-pointing-sample`은 2026-05-13 archive됐으므로 구조 학습용으로만 보고 새 프로젝트는 현재 `robotics-samples`와 ER 2 문서를 사용합니다.

## 발표와 기술 보고서

- [Introducing Gemini Robotics ER 2](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/)
- [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020)
- [Gemini Robotics 1.5 technical report](https://arxiv.org/abs/2510.03342)
- [Gemini Robotics 2: Safety Evaluations](https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-2-Safety.pdf)
- [Evaluating Gemini Robotics Policies in a Veo World Simulator](https://arxiv.org/abs/2512.10675)
- [ASIMOV-Agentic dataset](https://huggingface.co/datasets/google/asimov_agentic)
- [ASIMOV semantic safety benchmark](https://asimov-benchmark.github.io/)

## 용어집

| 용어 | 뜻 |
| --- | --- |
| embodiment | 로봇의 몸체, 관절, gripper, sensor 구성 |
| embodied reasoning | 물리 공간·시간·가능 행동에 관한 추론 |
| VLM | vision-language model, 시각·언어를 주로 텍스트로 출력 |
| VLA | vision-language-action, 관찰·명령에서 행동 생성 |
| affordance | 물체가 허용하는 행동 가능성 |
| grounding | 언어 대상을 이미지·공간·센서와 연결 |
| pointing | 대상 위치를 대표하는 한 점 예측 |
| trajectory | 시간 순서가 있는 pose/waypoint 경로 |
| proprioception | 로봇 자신의 관절·속도·힘 상태 |
| embodiment adaptation | 다른 로봇에 policy를 맞추는 과정 |
| motion transfer | 서로 다른 embodiment 사이 행동 지식 전이 |
| function calling | 모델이 실행할 함수 이름·인자를 구조적으로 제안 |
| orchestration | 여러 tool/skill의 순서·상태·복구를 관리 |
| moment finding | 영상에서 핵심 사건이 발생한 시각 탐색 |
| safety envelope | 허용 workspace·속도·힘 등의 독립 제약 |
| functional safety | 고장 시 위험을 제어하는 인증 가능한 시스템 안전 |
| HIL | hardware-in-the-loop, 일부 실제 하드웨어를 포함한 시험 |

## 번역 시 주의

- “thinking”은 사람과 같은 내면을 의미하지 않고 모델의 추론 계산·출력 메커니즘을 가리킵니다.
- “direct control”은 safety certification을 의미하지 않습니다.
- “generalist”는 모든 로봇·환경에서 보장된다는 뜻이 아닙니다.
- “on-device”는 모든 SBC에서 실행 가능하다는 뜻이 아닙니다.
- “real-time”은 endpoint latency 맥락이며 hard real-time control 보장이 아닙니다.

