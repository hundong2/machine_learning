# 02. VLM, VLA, 행동 표현의 기초

## 1. VLM에서 VLA로

### VLM

Vision-Language Model은 이미지·영상과 언어를 입력받아 텍스트, 좌표, 계획을 생성합니다.

```text
image + "파란 블록은 어디인가?" → [y, x]
```

### VLA

Vision-Language-Action model은 관찰과 명령을 받아 로봇 행동을 생성합니다.

```text
camera + proprioception + instruction → action chunk
```

`proprioception`은 관절각, 속도, gripper 상태처럼 로봇 자신의 상태입니다.

## 2. 행동을 표현하는 방법

| 표현 | 예 | 장점 | 위험·한계 |
| --- | --- | --- | --- |
| joint position | 각 관절 목표각 | 명확함 | embodiment 종속 |
| end-effector delta | 손끝 Δx, Δy, Δz | 작업 중심 | IK·특이점 필요 |
| velocity | 선속도·각속도 | 반응성 | 적분 drift, 안전 제한 필요 |
| action token | 이산 토큰 | 언어 모델과 통합 | 양자화·해석 문제 |
| skill call | `pick(object)` | 고수준·안전 검증 용이 | skill 라이브러리 필요 |

ER 2 공개 API는 텍스트와 tool call을 출력합니다. 이를 joint command와 혼동하지 마세요.

## 3. Action chunking

한 시점의 행동 하나가 아니라 짧은 미래 구간을 한 번에 예측하면 지연을 숨기고 부드러운 동작을 만들 수 있습니다. 그러나 환경이 변하면 남은 chunk가 오래된 계획이 됩니다.

전문 구현은 다음을 고려합니다.

- receding horizon: 일부만 실행하고 다시 관찰
- interruption: 사람·장애물 등장 시 chunk 폐기
- temporal ensemble: 여러 예측을 시간적으로 결합
- freshness: observation age와 action age 제한

## 4. Generalization의 종류

Gemini Robotics 랜딩 페이지가 말하는 generality를 평가하려면 범위를 나눠야 합니다.

- instruction: 같은 작업의 새로운 표현
- visual: 색, 배경, 조명, 카메라 변화
- object: 새로운 모양·재질
- task: 학습하지 않은 작업 조합
- embodiment: 다른 팔, gripper, 센서
- environment: 새로운 방과 배치

in-distribution 평균 점수 하나로 모두를 주장하면 안 됩니다.

## 5. Imitation learning의 최소 개념

VLA는 일반적으로 demonstration의 관찰·명령·행동을 이용해 행동을 모방합니다.

```text
dataset = {(observation_t, instruction, action_t:t+k)}
loss = distance(predicted_action, expert_action)
```

문제점:

- expert가 방문하지 않은 상태에서 오류가 누적됨
- 동일 상황의 올바른 행동이 여러 개일 수 있음
- embodiment마다 action space가 다름
- 성공 demonstration만으로 실패 복구를 배우기 어려움

## 6. Motion transfer의 의미

Robotics 1.5 보고서의 motion transfer는 서로 다른 로봇 데이터에서 공통 기술을 학습하고 새 embodiment로 옮기려는 메커니즘입니다. 이것이 calibration 없는 즉시 호환을 뜻하지는 않습니다.

새 robot에 여전히 필요한 것:

- action·state adapter
- camera와 robot frame calibration
- joint/velocity/force limit
- demonstration과 validation
- 실제 hardware safety 검토

## 실습 설계

이 저장소에서는 VLA weight 대신 skill call을 행동 표현으로 사용합니다.

```json
{"name": "move", "arguments": {"x": 0.12, "y": -0.04, "z": 0.20}}
```

이 호출은 다음 검사를 통과한 뒤에만 mock robot이 실행합니다.

- 알려진 tool인가
- 숫자와 단위가 맞는가
- workspace 안인가
- 금지 구역과 겹치지 않는가
- 한 번에 이동 가능한 거리인가
- 사람 확인이 필요한가

