# 06. 추적, PID, 안전 상태기계

## 목표

탐지 박스를 모터 각도로 바로 변환하지 않고, 대상 선택·시간 추적·오차 필터·제어기·제한기·watchdog를 분리해 흔들림과 위험 동작을 줄입니다.

## 1단계: 대상 선택

후보가 여러 개일 때 정책이 없으면 가장 높은 confidence 대상이 프레임마다 바뀔 수 있습니다.

초기 정책:

1. 설정 클래스와 threshold로 필터링
2. 기존 대상과 IoU가 큰 후보 우선
3. 없으면 화면 중심과 가까운 후보
4. 너무 작은 박스는 제외
5. 새 대상은 연속 N회 확인 후 확정

점수 예시:

```text
score =
  0.50 * detector_confidence
+ 0.30 * IoU_with_previous
+ 0.20 * center_proximity
```

가중치는 정답이 아니라 명시적 출발점입니다. target switch 횟수와 유실 복구 시간으로 비교합니다.

## 2단계: 정규화 오차

박스 중심을 `(cx, cy)`, 영상 크기를 `(W, H)`라 하면:

```text
ex = (cx - W/2) / (W/2)
ey = (cy - H/2) / (H/2)
```

`ex`, `ey`는 대략 `[-1, 1]` 범위입니다. 카메라 설치 방향에 따라 팬·틸트 부호가 달라지므로 설정 파일로 캘리브레이션합니다.

```yaml
control:
  pan_sign: -1
  tilt_sign: 1
  deadband_x: 0.04
  deadband_y: 0.05
```

## 3단계: 필터링

가장 단순한 지수 이동 평균:

```text
filtered = alpha * measurement + (1 - alpha) * previous
```

- alpha가 크면 빠르지만 흔들림이 큼
- alpha가 작으면 부드럽지만 지연이 큼

추적기가 필요하면 detector를 매 N프레임 실행하고 중간 프레임은 IoU/Kalman 기반 상태 추정으로 연결합니다. 먼저 detector-every-frame 기준선을 측정한 뒤 추가합니다.

## 4단계: PID

각 축을 독립적으로 시작합니다.

```text
u(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt
```

실전 순서:

1. `Ki=Kd=0`으로 두고 작은 `Kp`부터 시작
2. 반응이 충분히 빠를 때까지 `Kp` 증가
3. 오버슈트와 노이즈가 크면 `Kd`와 저역 통과 필터 검토
4. 정상 오차가 반복될 때만 작은 `Ki` 추가
5. 출력 포화 중에는 적분을 멈추는 anti-windup 적용

입문 프로젝트에서는 PI 또는 PD가 전체 PID보다 안정적으로 시작될 수 있습니다.

## 5단계: 제한기

제어기 출력 뒤에 항상 다음 순서의 보호를 둡니다.

1. deadband
2. 유효한 timestamp 검사
3. 각도 clamp
4. 속도/slew-rate clamp
5. 선택적 가속도 제한
6. 장치 명령

```cpp
if (observation_age > max_observation_age) {
  state = State::Lost;
  driver.disable();
  return;
}
```

## 6단계: 서로 다른 주기 분리

- 카메라: 예를 들어 30 Hz
- 탐지: 장치가 가능한 5~30 Hz
- 제어: 일정한 20~50 Hz
- PWM 출력: 드라이버가 요구하는 주기

탐지가 새로 오지 않아도 제어 loop는 일정 주기로 watchdog를 확인합니다. 반대로 오래된 관측을 계속 재사용해서는 안 됩니다.

## 7단계: 하드웨어 전 합성 시험

목표 궤적:

- step: 중앙 → 오른쪽 0.5
- ramp: 왼쪽에서 오른쪽으로 일정 속도
- sine: 부드러운 왕복
- dropout: 300ms, 1s, 3s 관측 유실
- noise: Gaussian jitter
- delay: 50~300ms 지연

측정:

- rise time
- settling time
- percent overshoot
- steady-state error
- command saturation ratio
- target switch count
- lost→reacquired 시간

CSV 예시:

```text
ts_ms,state,target_ex,filtered_ex,pan_cmd,pan_deg,age_ms,saturated
```

## 8단계: 실제 서보 캘리브레이션

1. 기구와 분리한 서보를 중립 펄스로 이동합니다.
2. 브래킷을 중립 방향에 조립합니다.
3. 제조사 최대 범위를 그대로 쓰지 말고 작은 범위부터 늘립니다.
4. 윙윙거리거나 기구 끝에 닿기 전의 안전 범위를 저장합니다.
5. pan과 tilt를 한 축씩 시험합니다.
6. 전원 재부팅 후 갑작스러운 점프가 없는지 확인합니다.

설정 예시:

```yaml
servo:
  pan:
    min_deg: -70
    neutral_deg: 0
    max_deg: 70
    max_speed_deg_s: 90
  tilt:
    min_deg: -35
    neutral_deg: 0
    max_deg: 45
    max_speed_deg_s: 60
  command_timeout_ms: 300
```

값은 기구마다 직접 캘리브레이션합니다.

## 완료 조건

- 대상 선택 정책과 전환 조건이 테스트된다.
- 합성 입력에서 정착 시간·오버슈트·정상 오차를 계산한다.
- stale observation과 명령 timeout이 안전 상태로 전이한다.
- 모터 출력에 각도·속도 제한이 있다.
- 실제 서보 없이 동일 제어 로직을 재현할 수 있다.

## 면접 포인트

> “탐지 confidence가 가장 높은 대상을 매 프레임 선택하면 대상이 튀는 문제가 있어 이전 대상 IoU와 hysteresis를 추가했습니다. 제어기는 고정 주기로 실행하고, 관측 age가 제한을 넘으면 PID 입력으로 재사용하지 않고 LOST 상태로 전환합니다.”

