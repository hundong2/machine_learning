# 10. 평가, 벤치마크, 연구 읽기

## 1. 랜딩 페이지 수치를 읽는 법

모델 페이지의 generalization score는 연구 결과를 요약하지만 내 로봇의 성능 보장이 아닙니다.

확인할 질문:

- 어떤 robot embodiment인가?
- 실제/시뮬레이션인가?
- 성공 정의와 episode 길이는?
- test object와 environment는 얼마나 새로운가?
- 사람 개입과 재시도는 허용됐는가?
- latency·failure가 포함됐는가?

## 2. ER 2 release 결과

공식 2026년 release post는 다음을 보고합니다.

- progress classification accuracy 57.4%
- moment finding accuracy 91.3%
- moment finding mean absolute distance 0.96초
- ER 1.6보다 real VLA, sim VLA, human tele-op tool orchestration 성능 향상

이 수치는 Google의 평가 조건에서 나온 결과입니다. 프로젝트 보고서에는 그대로 성능 목표로 복사하지 말고 자체 validation set을 만듭니다.

## 3. 기능별 지표

### Pointing

- pixel distance / image diagonal
- hit rate: point가 정답 mask/box 안인가
- no-object false positive
- repeated-query dispersion

### Bounding box

- IoU
- center error
- recall/precision
- label agreement

### Trajectory hint

- waypoint collision rate
- motion planner acceptance rate
- path length ratio
- goal reach rate

### Orchestration

- task success
- tool selection accuracy
- invalid/unsafe tool-call rate
- mean tool calls
- no-progress recovery
- human clarification precision/recall

### Video

- completion timestamp MAE
- tolerance accuracy
- progress bracket accuracy
- temporal consistency
- early-success false positive

### Safety

- unsafe task refusal accuracy
- safety tool recall
- false negative rate 우선
- false positive rate
- time-to-stop
- ambiguity clarification rate

## 4. Evaluation matrix

| 축 | 예시 |
| --- | --- |
| 조명 | 밝음, 어두움, 역광 |
| 시야 | 선명, blur, occlusion |
| 배치 | 학습/예시와 유사, 새로운 배치 |
| instruction | 직접적, 동의어, 모호, 악의적 |
| 대상 | 흔함, 투명, 반사, 작은 물체 |
| 사람 | 없음, 경계 밖, 경계 안 |
| 시스템 | 정상, API timeout, stale frame, tool failure |

각 slice의 worst-case를 보고합니다. 전체 평균만 표시하지 않습니다.

## 5. 연구 논문 읽기 순서

1. Abstract: 무엇을 주장하는가
2. Model/system: VLM, VLA, decoder, tool의 경계
3. Dataset: 실제 로봇 시간과 embodiment 다양성
4. Evaluation: 분포와 성공 기준
5. Ablation: 어떤 요소가 효과를 만들었나
6. Safety/limitations: 평가하지 않은 것은 무엇인가
7. 재현성: 공개 weight, code, data 여부

## 6. 핵심 공식 자료

- 2025 Gemini Robotics 보고서: VLA + ER family, dexterity, generalization
- 2025 Robotics 1.5 보고서: motion transfer, thinking, multi-embodiment
- 2026 ER 2 model card: Gemini 3.5 Flash 기반 공개 API VLM
- 2026 Robotics 2 Safety Evaluations: ASIMOV-Agentic, uncertainty resolution
- Veo world simulator 평가: nominal, OOD, physical·semantic safety의 가상 평가

## 7. 재현 가능한 보고서

```text
date, git commit, SDK version
model endpoint, prompt version, thinking level
input checksum, image/video preprocessing
number of repeats, temperature/config
latency p50/p95/p99, token/cost
raw response, parsed response, validation result
tool proposal, policy decision, actual execution result
```

Preview model은 바뀔 수 있으므로 날짜와 endpoint가 특히 중요합니다.

## 8. 전문가 수준의 실험

- ER 2 standard vs streaming의 tool-selection latency
- low/medium/high thinking의 Pareto curve
- single query vs median consensus
- prompt-only JSON vs structured output
- VLM safety 판단 vs deterministic safety sensor
- static image success detection vs video moment finding
- mock, sim, hardware에서 동일 tool contract의 차이

