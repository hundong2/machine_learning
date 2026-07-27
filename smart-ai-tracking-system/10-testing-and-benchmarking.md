# 10. 시험, 벤치마크, 관측성

## 목표

데모가 한 번 성공한 것을 완성으로 보지 않고, 정확도·지연·제어·고장 복구를 반복 가능한 시험으로 증명합니다.

## 시험 피라미드

### 단위 시험

- letterbox scale과 padding
- 좌표 복원
- confidence 계산
- NMS
- target score와 hysteresis
- PID, deadband, anti-windup
- 각도·속도 clamp
- 상태 전이

### 계약 시험

- 모델 checksum
- tensor 이름·dtype·shape
- Python/C++ 골든 parity
- 설정 스키마
- ROS message와 frame 의미
- ServoDriver 구현의 공통 동작

### 통합 시험

- MP4 → C++ 탐지 JSON
- rosbag → perception → controller
- mock driver → command CSV
- Pi 카메라 reconnect
- systemd restart와 graceful shutdown

### 시스템·하드웨어 시험

- 조도와 거리
- 대상 유실과 재획득
- 30분 soak
- 전원·열 조건
- 실제 정착 시간과 오버슈트
- 케이블 분리와 명령 timeout

## 1단계: 골든 데이터

작고 안정적인 골든 세트를 버전 관리합니다.

```text
tests/golden/
├─ images/
├─ preprocess/
├─ raw_outputs/
├─ detections/
└─ manifest.yaml
```

`manifest.yaml`에는 원본 checksum, 라이선스, 모델 checksum, 생성 코드 commit을 적습니다. 모델을 의도적으로 변경할 때만 검토 후 골든 결과를 갱신합니다.

## 2단계: 동등성 기준

탐지 순서가 다를 수 있으므로 배열 index를 직접 비교하지 않습니다.

1. class별 후보 분리
2. IoU matrix 계산
3. greedy 또는 Hungarian matching
4. 매칭/미매칭 집계
5. box IoU와 confidence delta 분포

보고:

```text
class_agreement
matched_box_iou_p50/p05
confidence_abs_delta_p50/p95/max
missing_detections
extra_detections
```

## 3단계: 지연 벤치마크

측정 원칙:

- Release 빌드
- warm-up 제외
- 최소 1,000프레임 또는 충분한 시간
- 고정 입력과 설정
- CPU governor, thread 수, 온도 기록
- UI on/off 분리
- 평균뿐 아니라 p50/p95/p99

종단 지연:

```text
commanded_at - frame.captured_at
```

추론 함수 시간만 “실시간 지연”이라고 부르지 않습니다. decode, queue, 전처리, 후처리, 제어까지 분해합니다.

## 4단계: 프레임 신선도

다음을 동시에 그립니다.

- 입력 FPS
- 처리 FPS
- drop ratio
- frame age
- queue wait

두 구현이 모두 10 FPS라도:

- A: 최신 frame, age 90ms, 60% drop
- B: FIFO 누적, age 2.5s, drop 0%

로봇 제어에는 A가 훨씬 낫습니다. drop을 무조건 실패로 해석하지 않습니다.

## 5단계: 제어 벤치마크

각 궤적을 최소 5회 반복합니다.

| 지표 | 의미 |
| --- | --- |
| rise time | 목표의 정해진 비율에 도달하는 시간 |
| settling time | 허용 오차 안에 들어와 유지되는 시간 |
| overshoot | 목표를 지나친 최대 비율 |
| steady-state error | 안정 후 남는 오차 |
| lost recovery | 유실 후 재획득까지 시간 |
| saturation ratio | 제한에 걸린 명령 비율 |

시뮬레이션과 실제 하드웨어를 같은 표에 두되 환경 열을 명확히 분리합니다.

## 6단계: fault injection

| 장애 | 주입 방법 | 기대 동작 |
| --- | --- | --- |
| 카메라 단절 | mock 예외 또는 케이블 분리 | FAULT, PWM 안전 상태, 제한적 재시도 |
| 탐지 유실 | 빈 결과 N프레임 | LOST, 정지/중립 정책 |
| 느린 추론 | 인위적 delay | queue 제한, frame drop, stale 차단 |
| NaN output | mock tensor | 결과 거부, crash 없음 |
| 잘못된 model | shape 변경 artifact | 시작 시 계약 실패 |
| controller 종료 | process kill | driver watchdog timeout |
| 디스크 부족 | 로그 sink 실패 mock | 제어 유지 또는 안전 종료 정책 |
| 과열 | 제한된 환경 관찰 | 경고·성능 저하 기록 |

고장 주입은 실제 전원선을 무작정 뽑는 것으로 시작하지 않습니다. mock과 software fault부터 단계적으로 수행합니다.

## 7단계: 구조화 로그

프레임 로그 예:

```json
{
  "event": "frame_processed",
  "frame_id": 1842,
  "frame_age_ms": 87.2,
  "infer_ms": 61.3,
  "end_to_end_ms": 91.7,
  "detections": 2,
  "target_state": "TRACKING",
  "dropped_total": 938,
  "cpu_temp_c": 67.4
}
```

로그에는 원본 이미지나 개인정보를 기본 저장하지 않습니다. 디버그 이미지 저장은 명시적 옵션과 보존 기간을 둡니다.

## 8단계: 벤치마크 표

| 장치 | 모델/정밀도 | 입력 | provider | p50 | p95 | fresh FPS | drop | 온도 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dev PC | 기록 | 416 | CPU | 측정 | 측정 | 측정 | 측정 | 측정 |
| Pi 5 | 기록 | 416 | CPU | 측정 | 측정 | 측정 | 측정 | 측정 |

숫자 옆에 반드시 commit, 모델 checksum, 실행 명령, 전원·냉각 조건을 연결합니다.

## 회귀 게이트

- 모델 계약 불일치: 빌드/실행 실패
- 골든 class agreement 기준 미달: CI 실패
- p95 지연 10% 이상 악화: 경고 또는 승인 필요
- 메모리 지속 증가: 릴리스 차단
- timeout 안전 시험 실패: 릴리스 차단

## 완료 조건

- 단위·계약·통합·시스템 시험이 구분되어 있다.
- 같은 명령으로 벤치마크를 재실행할 수 있다.
- 평균뿐 아니라 p95/p99와 frame age를 보고한다.
- 최소 6개의 장애 시나리오가 자동 또는 반자동 검증된다.
- 모든 결과가 commit과 모델 checksum으로 역추적된다.

