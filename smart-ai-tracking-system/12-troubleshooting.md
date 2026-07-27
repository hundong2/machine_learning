# 12. 장애 진단 플레이북

문제를 한 번에 여러 계층에서 고치지 않습니다. **입력 → 전처리 → 모델 → 후처리 → 추적 → 제어 → 드라이버 → 전원·기구** 순서로 경계를 확인합니다.

## 진단 공통 절차

1. 실패를 고정 영상 또는 mock으로 재현합니다.
2. 마지막 정상 경계와 첫 비정상 경계를 찾습니다.
3. 모델 checksum, 설정, commit, 실행 환경을 기록합니다.
4. 한 변수만 바꿉니다.
5. 수정 후 골든·성능·fault 회귀를 다시 실행합니다.

## 모델이 로드되지 않음

확인:

- 파일 존재와 읽기 권한
- SHA-256 일치
- ONNX checker
- runtime/opset 호환
- ARM64 라이브러리인지
- provider가 실제 포함되었는지

조치:

- metadata 출력 전용 CLI로 애플리케이션과 분리
- CPU provider에서 먼저 검증
- 더 낮은 opset로 무작정 바꾸기 전에 unsupported operator 확인

## Python과 C++ 결과가 다름

우선순위:

1. 같은 원본 이미지와 모델 checksum인가
2. BGR/RGB
3. float range
4. HWC/NCHW
5. resize/letterbox와 padding 반올림
6. output 축 순서
7. objectness와 class score 계산
8. confidence threshold
9. class-aware NMS
10. 원본 좌표 복원

중간 tensor를 저장해 첫 번째로 달라지는 지점을 찾습니다.

## 박스가 밀리거나 크기가 틀림

- stretch resize와 letterbox 혼용
- 좌우/상하 padding 값이 다름
- integer rounding 순서
- `xywh`와 `xyxy` 혼동
- network pixel과 normalized coordinate 혼동
- padding을 빼기 전에 scale을 나눔

정답 박스 모서리에 색 점을 찍는 시각 테스트와 수치 단위 테스트를 함께 사용합니다.

## FPS는 괜찮은데 반응이 늦음

- FIFO queue에 오래된 frame이 쌓임
- capture backend 내부 buffer
- display의 `waitKey` 또는 영상 encoding
- ROS reliable queue가 누적
- 처리 완료 timestamp를 capture timestamp로 오인

`frame_age_ms`, queue wait, sequence gap을 먼저 확인합니다. latest-frame 정책과 작은 QoS depth를 적용합니다.

## 메모리가 계속 증가함

- frame clone 누적
- 결과 vector/history 무제한 보관
- logger buffer
- display/encoder queue
- ORT tensor 수명
- thread가 소비하지 않는 queue

ASan, heap profiler, RSS 시계열로 조사합니다. soak test를 최소 30분 실행합니다.

## 서보가 떨림

- 탐지 박스 jitter
- target switching
- deadband 없음
- `Kp` 또는 `Kd` 과다
- derivative에 필터 없음
- PWM/전원 불안정
- 기구 backlash
- 제어 주기 불규칙

모터를 분리하고 mock output 그래프부터 확인합니다. 필터와 deadband 뒤에도 명령이 안정적인데 실제 서보만 떨리면 전원·PWM·기구를 조사합니다.

## 서보 때문에 Pi가 재부팅됨

즉시 전원을 끄고 다음을 확인합니다.

- 서보를 Pi 5V/GPIO에서 직접 공급했는지
- 별도 5V 전원의 전류 용량
- 공통 GND
- 배선 극성
- 급격한 동시 기동
- 전압 강하

소프트웨어 튜닝으로 전원 문제를 숨기지 않습니다. PCA9685, 별도 전원, 적절한 배선을 구성한 뒤 재시험합니다.

## 반대 방향으로 움직임

- 카메라 영상 좌표의 y축은 아래가 양수
- servo 설치 방향에 따른 부호
- pan/tilt joint axis
- optical frame 규약

코드를 뒤집어 고치기보다 `pan_sign`, `tilt_sign` 캘리브레이션으로 관리하고 합성 좌표 테스트를 둡니다.

## 대상을 자주 바꿈

- confidence 최고 후보만 선택
- 이전 target 연계 없음
- 새 target 확인 횟수 없음
- detector threshold가 너무 낮음

IoU 연계, hysteresis, 최소 유지 시간, center/size prior를 적용하고 switch count를 측정합니다.

## ROS 2 메시지가 보이지 않음

- topic 이름과 namespace
- message type
- publisher/subscriber QoS 호환
- domain ID
- source timestamp
- lifecycle node active 상태
- Gazebo bridge 방향과 type

`ros2 topic info --verbose`, `ros2 node info`, diagnostics를 사용하고 최소 publisher/subscriber로 축소합니다.

## Pi에서만 느림

- Debug 빌드
- 열 스로틀링
- 저전압
- 과한 thread 수
- 입력 decode/색 변환 비용
- UI와 영상 저장
- 메모리 swap
- CPU용으로 너무 큰 모델/해상도

모델만 측정한 기준선부터 구성 요소를 하나씩 더합니다. 냉각·전원을 고정한 뒤 해상도와 모델 크기를 비교합니다.

## 고장 보고 템플릿

```text
증상:
최초 발생 시각:
재현 명령:
입력 checksum:
모델 checksum:
commit:
환경:
기대 결과:
실제 결과:
마지막 정상 경계:
첫 비정상 경계:
첨부 로그/그래프:
시도한 변경:
```

