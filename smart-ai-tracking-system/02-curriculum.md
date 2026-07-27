# 02. 12주 실무 커리큘럼

## 운영 방식

매주 다음 순서를 반복합니다.

1. 30분: 이번 주 가설과 측정 기준 작성
2. 4~6시간: 가장 작은 수직 기능 구현
3. 1~2시간: 자동 시험과 벤치마크
4. 1시간: 실패 사례와 결정 기록
5. 30분: 3분 데모와 회고

코드를 많이 작성하는 것보다, 한 기능을 **재현·검증·설명**할 수 있게 끝내는 것이 중요합니다.

## Week 1 — 제품 요구사항과 기초 도구

### 배울 것

- Git 브랜치와 작은 커밋
- Python 가상 환경
- C++ RAII, 스마트 포인터, `std::chrono`, 인터페이스
- CMake target 중심 구성
- 카메라 좌표계와 텐서 shape

### 실습

- 프로젝트 헌장과 상태기계를 작성합니다.
- `tracker_cli --source sample.mp4 --dry-run` 형태의 CLI 계약을 정의합니다.
- 샘플 영상의 checksum을 기록합니다.
- 빈 C++ 라이브러리와 테스트 실행 파일을 빌드합니다.

### 산출물

`PROJECT_SPEC.md`, 아키텍처 그림, 빌드 로그, ADR-001.

## Week 2 — Python 사전학습 기준선

### 배울 것

- 객체 탐지의 박스, confidence, class
- resize/letterbox, RGB/BGR, NCHW
- precision, recall, IoU, mAP의 역할
- 결정적 입력과 결과 직렬화

### 실습

- 소형 사전학습 모델로 `person`을 탐지합니다.
- 동일 영상에서 프레임 번호별 결과를 JSONL로 저장합니다.
- 오탐·미탐 20개를 수동 분류합니다.

### 산출물

기준 결과 JSONL, 오류 갤러리, 기준 벤치마크.

## Week 3 — 데이터와 선택적 파인튜닝

### 배울 것

- train/validation/test 누수
- 클래스 불균형과 어려운 음성 샘플
- 데이터 라이선스와 개인정보
- 실험 추적과 seed

### 실습

- 기준 모델이 요구사항을 만족하지 못할 때만 커스텀 데이터를 만듭니다.
- 조도·거리·가림·배경별로 데이터 slice를 나눕니다.
- 전체 점수와 slice별 점수를 비교합니다.

### 산출물

데이터 카드, 모델 카드, 실험 표, 실패 사례.

## Week 4 — ONNX 계약

### 배울 것

- opset, dynamic shape, graph validation
- 모델 세대별 raw/end-to-end 출력 차이
- 전처리·후처리 소유권
- 수치 오차와 의미 동등성

### 실습

- 고정 shape FP32 ONNX부터 내보냅니다.
- 입력·출력 이름, dtype, shape, 좌표 정의를 계약서에 적습니다.
- Python 원본과 ONNX Runtime 결과를 50개 골든 입력으로 비교합니다.

### 산출물

`model-contract.yaml`, checksum, parity 보고서.

## Week 5 — C++ 단일 이미지 추론

### 배울 것

- ONNX Runtime session과 tensor 수명
- OpenCV 메모리 배치
- 예외와 오류 반환
- CMake imported target

### 실습

- C++에서 모델 메타데이터를 출력합니다.
- 이미지 하나를 전처리·추론·후처리합니다.
- Python 결과와 박스·점수를 비교하는 테스트를 작성합니다.

### 산출물

재현 가능한 CMake 빌드, 단일 이미지 결과, 계약 테스트.

## Week 6 — 실시간 C++ 파이프라인

### 배울 것

- producer/consumer와 bounded queue
- warm-up, 단계별 지연, p95
- 메모리 재사용과 불필요한 복사
- graceful shutdown

### 실습

- 캡처와 추론을 분리합니다.
- 큐 크기를 1~2로 제한하고 drop 수를 기록합니다.
- capture→preprocess→infer→postprocess→display 시간을 측정합니다.

### 산출물

지연 분포, flame graph 또는 profiler 캡처, 병목 ADR.

## Week 7 — 대상 추적과 모의 제어

### 배울 것

- 대상 선택, hysteresis, IoU 연계
- low-pass/Kalman filter
- deadband, PID, anti-windup
- watchdog와 상태기계

### 실습

- 합성된 목표 궤적에 제어기를 연결합니다.
- 지연·잡음·탐지 유실을 주입합니다.
- 정착 시간, 오버슈트, 정상 오차를 측정합니다.

### 산출물

제어 그래프, 자동 테스트, 튜닝 근거.

## Week 8 — Raspberry Pi 배포

### 배울 것

- ARM64 패키징
- systemd와 환경 파일
- 온도·전압·메모리 관측
- 장치 권한과 로그

### 실습

- 네이티브 Release 빌드를 만들고 고정 영상을 실행합니다.
- 30분 soak test를 합니다.
- 온도와 스로틀링 상태를 성능과 함께 기록합니다.

### 산출물

설치 스크립트, 서비스 파일, Pi 벤치마크.

## Week 9 — 실제 팬·틸트

### 배울 것

- PWM 캘리브레이션
- 기구 한계와 좌표 부호
- slew-rate와 명령 제한
- 비상 정지

### 실습

- 카메라 없이 작은 각도 명령부터 검증합니다.
- 한 축씩 닫힌 고리를 구성한 뒤 2축으로 확장합니다.
- 케이블 분리와 대상 유실 시험을 수행합니다.

### 산출물

배선도, 캘리브레이션 파일, 안전 시험 영상.

## Week 10 — ROS 2 Jazzy 통합

### 배울 것

- topic/service/action 선택
- QoS와 sensor data
- launch, parameter, lifecycle
- rosbag2 기반 재현

### 실습

- perception, tracking, control 노드로 분리합니다.
- 녹화 bag을 재생해 하드웨어 없이 회귀 시험합니다.
- 노드 정지와 늦은 메시지를 주입합니다.

### 산출물

인터페이스 표, launch, bag 기반 통합 테스트.

## Week 11 — 시뮬레이션·품질·최적화

### 배울 것

- Gazebo Harmonic과 `ros_gz_bridge`
- HIL 경계
- 성능 회귀와 fault injection
- FP16/INT8의 정확도 비용

### 실습

- 모의 팬·틸트 joint와 카메라를 연결합니다.
- CPU 기준선을 먼저 고정한 뒤 필요할 때만 가속기를 실험합니다.
- 최적화 전후를 같은 입력과 조건으로 비교합니다.

### 산출물

시뮬레이션 영상, 비교 벤치마크, 최적화 ADR.

## Week 12 — 포트폴리오 릴리스

### 배울 것

- 기술 README
- 재현 가능한 릴리스
- 데모 스토리텔링
- 장애와 트레이드오프 설명

### 실습

- 깨끗한 환경에서 문서만 보고 재설치합니다.
- 90초 데모와 5분 기술 설명을 녹화합니다.
- 이력서 문장을 수치로 작성합니다.

### 산출물

릴리스 태그, 데모 영상, 모델 카드, 벤치마크 보고서, 이력서 문장.

## 역량 매핑

| 채용 신호 | 프로젝트 증거 |
| --- | --- |
| ML 엔지니어링 | 데이터 slice, 모델 카드, ONNX parity |
| C++ 시스템 | RAII, bounded queue, CMake, sanitizer |
| 엣지 최적화 | ARM64 배포, 지연 분해, 온도·메모리 |
| 로보틱스 | 상태기계, PID, watchdog, ROS 2 QoS |
| 품질 엔지니어링 | 골든 테스트, fault injection, soak test |
| 기술 소통 | ADR, 벤치마크 조건, 실패 분석, 데모 |

