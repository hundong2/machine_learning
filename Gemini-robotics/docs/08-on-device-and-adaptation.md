# 08. On-Device와 embodiment adaptation

## 공식 정보 요약

Gemini Robotics On-Device 2는 네트워크 latency나 연결 없이 로봇 장치에서 실행하도록 최적화된 VLA입니다. 공식 페이지는 새로운 로봇·센서에 몇 시간의 학습과 200개 미만 예제로 빠르게 적응할 수 있다고 소개합니다. 현재 trusted tester 대상입니다.

## 1. On-device가 필요한 이유

- network outage에서도 동작
- 낮고 예측 가능한 inference latency
- 영상의 cloud 전송 최소화
- 고주기 반응
- 장치별 최적화

대신 compute, memory, thermal, battery 제한이 생깁니다.

## 2. Cloud ER + Local VLA

```text
Cloud ER 2
  - 목표 분해
  - 공간·시간 추론
  - tool 선택
  - 성공·실패 판정

Local VLA / skill
  - 짧은 행동 실행
  - 빠른 관찰-행동 loop
  - network 단절 시 안전 종료

Deterministic controller
  - joint·velocity·force 제한
```

## 3. 새 embodiment 적응 데이터

episode에는 보통 다음이 필요합니다.

- 여러 camera observation
- robot proprioception
- 자연어 instruction
- timestamp가 정렬된 action
- success/failure label
- calibration과 robot configuration

데이터 품질 체크:

- timestamp drift
- action/state unit
- reset 상태
- demonstration success
- 사람 개인정보
- rare failure와 recovery

## 4. 적은 예제의 함정

“200개 미만”은 모든 작업에서 보장되는 수치가 아닙니다.

- 짧고 반복적인 skill과 긴 작업은 난이도가 다름
- gripper·camera 변화가 클수록 adapter가 어려움
- demonstration 다양성이 개수보다 중요할 수 있음
- safety validation set은 train example과 별도로 필요

## 5. On-device 평가

- cold/warm latency p50, p95, p99
- observation-to-action age
- action frequency
- memory와 thermal throttling
- battery consumption
- network disconnected success rate
- OOD refusal/stop rate
- physical safety violation rate

## 6. 공개 대안으로 학습하기

On-Device 2 접근이 없다면 다음 개념을 공개 도구로 연습할 수 있습니다.

- LeRobot: demonstration 수집과 policy training
- Open X-Embodiment: multi-embodiment dataset 구조 연구
- MuJoCo / Isaac Sim / Gazebo: simulation validation
- ROS 2: sensor, action, safety adapter
- 공개 VLA 논문: RT-1/RT-2, OpenVLA, π0 계열 비교

이 대안이 Gemini Robotics On-Device 2의 내부 구조와 동일하다는 뜻은 아닙니다. transferable engineering concept를 연습하는 경로입니다.

## 7. 전문 프로젝트 아이디어

같은 `RobotSkill` interface에 세 backend를 구현합니다.

```text
MockSkillBackend
SimulationSkillBackend
HardwareSkillBackend
```

ER 2 agent는 backend 차이를 모르며, safety gate와 tool result 계약은 동일하게 유지합니다.

