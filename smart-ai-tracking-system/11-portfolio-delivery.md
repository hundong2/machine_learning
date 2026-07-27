# 11. 포트폴리오 패키징과 면접 준비

## 목표

프로젝트를 “코드가 있는 저장소”가 아니라 문제 정의, 설계 판단, 수치, 실패와 개선을 짧은 시간에 검증할 수 있는 채용 자료로 만듭니다.

## 채용 담당자가 3분 안에 볼 것

최상단 README 첫 화면에 다음이 보여야 합니다.

1. 한 문장 문제 정의
2. 20초 GIF 또는 데모 이미지
3. 아키텍처 그림
4. 핵심 수치 3~5개
5. 빠른 실행 명령
6. 기술 선택과 한계

기술 배지 20개보다 “Pi 5에서 416 입력, ONNX Runtime CPU, p95 118ms, drop 42%, frame age p95 131ms” 같은 조건 있는 숫자가 강합니다.

## 필수 공개 산출물

| 산출물 | 증명하는 역량 |
| --- | --- |
| `PROJECT_SPEC.md` | 요구사항과 범위 관리 |
| 아키텍처·상태기계 | 시스템 설계 |
| 모델·데이터 카드 | ML 책임성과 재현성 |
| ONNX 계약·parity | Python→C++ 배포 |
| CMake·테스트 | C++ 품질 |
| ROS interface·launch | 로봇 통합 |
| 벤치마크 보고서 | 성능 엔지니어링 |
| fault test | 운영·안전 사고 |
| ADR 3개 이상 | 트레이드오프 판단 |
| 90초 데모 | 결과 전달 |

## README 권장 구조

```text
# Project name + one-line outcome
## Demo
## Measured results
## Architecture
## Safety behavior
## Quick start
## Model contract
## ROS 2 interfaces
## Tests and benchmarks
## Hardware and wiring
## Limitations
## Roadmap
## License and data/model attribution
```

## 핵심 ADR 주제

최소 3개를 작성합니다.

- ADR-001: FIFO 대신 latest-frame queue
- ADR-002: Raspberry Pi CPU 기준선에 ONNX Runtime 선택
- ADR-003: 직접 GPIO PWM 대신 PCA9685
- ADR-004: 단일 프로세스와 ROS 노드 경계
- ADR-005: 고정 shape FP32를 초기 계약으로 선택

결론만 쓰지 말고 맥락, 후보, 결정, 장단점, 재검토 조건을 남깁니다.

## 90초 데모 대본

- 0~10초: 문제와 장치
- 10~25초: Python→ONNX→C++ 구조
- 25~50초: 실제 추적과 overlay
- 50~65초: 대상 유실·재획득 또는 카메라 fault
- 65~80초: p95 지연, FPS, 제어 그래프
- 80~90초: 한계와 다음 개선

편집으로 실패를 숨기기보다 정상과 고장 복구를 같이 보여줍니다.

## 이력서 문장 공식

```text
[문제/규모]를 위해 [설계/구현]하여
[측정 조건]에서 [결과]를 달성했고,
[품질/안전 장치]로 [위험]을 관리했다.
```

초안 예시:

> Python 객체 탐지 모델을 ONNX로 내보내고 C++20/ONNX Runtime 기반 최신 프레임 파이프라인과 ROS 2 Jazzy 팬·틸트 제어를 구현해 Raspberry Pi 5에서 p95 종단 지연 **측정값**과 신선한 추론 **측정 FPS**를 달성했으며, 골든 parity·bounded queue·watchdog·fault injection으로 출력 동등성과 안전 정지를 검증했습니다.

측정 전에는 숫자를 채우지 않습니다. “30 FPS” 같은 희망값을 실적으로 쓰지 마세요.

## 5분 기술 설명

1. 왜 이 문제를 골랐는가
2. Python과 C++의 책임을 어떻게 나눴는가
3. ONNX 계약에서 가장 어려웠던 차이는 무엇인가
4. 왜 평균 FPS 대신 frame age와 p95를 봤는가
5. target switching과 servo jitter를 어떻게 줄였는가
6. 고장 시 어떤 계층이 안전을 보장하는가
7. 다음 병목을 어떻게 찾고 개선할 것인가

## 면접 예상 질문

### 모델·배포

- letterbox 좌표를 원본 이미지로 어떻게 복원했나요?
- raw YOLO output과 end-to-end output 차이는 무엇인가요?
- FP16/INT8 적용 전후 정확도를 어떻게 검증했나요?
- ONNX Runtime과 OpenCV DNN 중 무엇을 기준으로 선택했나요?

### C++

- `cv::Mat`과 ORT tensor의 수명을 어떻게 관리했나요?
- queue가 가득 차면 왜 drop하나요?
- thread 종료와 camera reconnect는 어떻게 처리했나요?
- 메모리 복사를 어디서 줄였나요?

### 로봇·제어

- PID gain은 어떻게 정했나요?
- 탐지 주기와 제어 주기가 다른 이유는 무엇인가요?
- target이 사라졌을 때 무엇이 일어나나요?
- servo driver가 controller crash를 어떻게 감지하나요?

### ROS 2

- 각 topic의 QoS를 왜 그렇게 골랐나요?
- timestamp를 왜 원본 카메라에서 유지하나요?
- node를 더 합치거나 나눌 기준은 무엇인가요?
- rosbag으로 무엇을 회귀 시험하나요?

## 정직하게 공개할 한계

- 단안 2D 박스이므로 실제 3D 거리와 자세를 알 수 없음
- 저가 서보의 backlash와 위치 feedback 부재
- Pi CPU에서 작은 모델·해상도 타협
- 조도·가림·모션 블러에 대한 취약성
- 얼굴 식별을 하지 않으며 개인정보 영상을 기본 저장하지 않음
- 산업 안전 인증 시스템이 아님

한계를 공개하면 프로젝트가 약해지는 것이 아니라, 시스템 경계를 이해한다는 신호가 됩니다.

## 릴리스 체크리스트

- [ ] 공개 가능한 데이터와 영상만 포함했다.
- [ ] 모델·데이터 라이선스와 출처를 표시했다.
- [ ] 모델 파일 checksum과 다운로드 방법이 있다.
- [ ] 깨끗한 환경에서 Quick Start를 재현했다.
- [ ] 하드웨어 없이 실행할 demo mode가 있다.
- [ ] 모든 벤치마크에 조건과 commit이 있다.
- [ ] 배선도와 전기 안전 경고가 있다.
- [ ] fault demo가 있다.
- [ ] 알려진 한계가 있다.
- [ ] 90초 영상과 5분 설명을 준비했다.

