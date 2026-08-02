# 07. Live API 스트리밍과 영상 진행도 추론

## 1. 두 endpoint의 차이

| 항목 | ER 2 standard | ER 2 streaming |
| --- | --- | --- |
| 모델 ID | `gemini-robotics-er-2-preview` | `gemini-robotics-er-2-streaming-preview` |
| 대표 용도 | 이미지·영상 분석, structured output, code execution | 낮은 지연의 연속 session과 tool calling |
| Live API | 미지원 | 지원 |
| 입력 | text/image/video/audio | text, JPEG image ≤ 1 FPS, PCM audio |
| protocol | 요청/응답 | stateful WSS |
| function call | 지원 | blocking call만 지원 |

Preview 사양은 바뀔 수 있으므로 실행 전 공식 문서를 확인합니다.

## 2. Streaming agent 구조

공식 guide는 세 단계를 제시합니다.

1. robot capability를 function tool로 선언
2. session 동안 audio/image/text를 계속 전송
3. receive loop에서 tool call을 실행하고 tool response 반환

물리 행동 tool은 `behavior: BLOCKING`을 사용해 완료 전에 다음 행동을 선택하지 않게 합니다.

## 3. 이미지 1 FPS의 의미

streaming endpoint의 JPEG 입력은 최대 1 FPS입니다. 이는 joint servo나 collision avoidance 주기가 아닙니다.

권장 다중 속도 구조:

```text
ER 2 semantic loop: ≤ 1 Hz image reasoning
local perception: 10~60 Hz
skill/controller: 50~1000 Hz
safety monitor: 독립 실시간
```

ER 2는 “무엇을 할지, 진행됐는지”를 판단하고 local controller가 “어떻게 안정적으로 움직일지”를 담당합니다.

## 4. Heartbeat

공식 문서에 따르면 frame만 보내서는 새 reasoning turn이 시작되지 않습니다. 최신 frame과 짧은 text/audio prompt를 주기적으로 보내는 heartbeat가 필요할 수 있습니다.

주의:

- heartbeat가 진행 중 generation을 interrupt할 수 있음
- 중복 tool call 방지
- 최신 frame만 유지
- task가 없을 때는 `ack` 같은 no-op
- 1 FPS와 API 비용 제한

## 5. Moment finding

standard ER 2는 영상에서 중요한 사건의 timestamp를 찾습니다.

```json
{"completion_time_seconds": 12.4}
```

완료되지 않았으면 `null`을 요구합니다. 평가할 때는 정답 timestamp와의 absolute distance, 일정 tolerance 안의 accuracy를 함께 봅니다.

## 6. Progress classification

공식 출력은 다섯 bracket입니다.

```text
0-20, 20-40, 40-60, 60-80, 80-100
```

이 값은 정밀 연속 progress가 아닙니다. 단조 증가도 보장되지 않으므로 상태기계를 둡니다.

```text
새 bracket이 이전보다 낮음
→ 한 번 더 관찰
→ 실패 탐지 또는 occlusion 확인
→ 필요하면 재계획
```

## 7. Backpressure와 freshness

카메라 frame을 모두 queue에 쌓지 않습니다.

- capacity 1의 latest-frame buffer
- capture timestamp 보존
- max frame age
- tool 실행 중에도 안전 sensor는 계속 처리
- session reconnect 시 old command 폐기

## 8. Streaming skeleton

`05_streaming_skeleton.py`는 API를 호출하지 않는 dry-run으로 다음을 설명합니다.

- bounded latest-frame slot
- tool allowlist
- blocking tool lifecycle
- heartbeat payload
- stop event와 reconnect 경계

공식 Live API는 빠르게 변하는 preview이므로 실제 연결 부분은 최신 공식 sample과 버전을 맞춘 뒤 adapter로 구현합니다.

## 완료 조건

- cloud streaming을 control loop로 사용하지 않는다.
- frame age와 tool command age를 제한한다.
- reconnect 후 이전 action을 재실행하지 않는다.
- moment finding과 progress classification을 별도 지표로 평가한다.

