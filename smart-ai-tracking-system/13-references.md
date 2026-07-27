# 13. 공식 자료와 다음 학습 경로

빠르게 변하는 라이브러리는 블로그 코드보다 공식 문서의 현재 버전을 먼저 확인합니다. 프로젝트 문서에는 참고한 날짜, 실제 고정 버전, 모델 checksum을 함께 남기세요.

## 모델 학습과 내보내기

- [PyTorch 튜토리얼](https://pytorch.org/tutorials/) — tensor, dataset, model, optimization 기초
- [PyTorch ONNX 공식 문서](https://docs.pytorch.org/docs/stable/onnx.html) — `torch.export` 기반 exporter와 dynamic shape
- [Ultralytics Train 모드](https://docs.ultralytics.com/modes/train/) — 사전학습 모델의 커스텀 데이터 학습
- [Ultralytics Export 모드](https://docs.ultralytics.com/modes/export/) — ONNX, TensorRT 등 export 옵션과 출력 형식
- [ONNX 공식 문서](https://onnx.ai/onnx/intro/) — graph, operator, checker의 기본 개념

### 학습 순서

1. PyTorch tensor와 inference
2. 객체 탐지 지표와 오류 분석
3. 사전학습 모델 기준선
4. 필요한 경우에만 파인튜닝
5. 고정 FP32 ONNX
6. 골든 parity
7. dynamic/quantization은 별도 실험

## C++ 추론

- [ONNX Runtime C++ 시작하기](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [ONNX Runtime API 기초](https://onnxruntime.ai/docs/tutorials/api-basics.html)
- [ONNX Runtime 실행 provider](https://onnxruntime.ai/docs/execution-providers/)
- [OpenCV DNN 모듈](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
- [CMake 튜토리얼](https://cmake.org/cmake/help/latest/guide/tutorial/)

ONNX Runtime C++를 기본 backend로 두고 OpenCV는 카메라·전처리·시각화에 사용합니다. OpenCV DNN도 작은 대안 실험으로 비교할 수 있지만, 두 backend의 결과와 성능을 같은 계약으로 검증합니다.

## NVIDIA·Intel 최적화

- [NVIDIA TensorRT 설치](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [TensorRT C++ API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)
- [TensorRT 아키텍처](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html)
- [OpenVINO 문서](https://docs.openvino.ai/)

TensorRT는 NVIDIA GPU/Jetson, OpenVINO는 Intel 하드웨어를 주 대상으로 합니다. Raspberry Pi CPU 프로젝트에 이름만 추가하지 말고 장치와 목표가 맞을 때 backend 실험으로 분리합니다.

## Raspberry Pi

- [Raspberry Pi 카메라 소프트웨어](https://www.raspberrypi.com/documentation/computers/camera_software.html) — Bookworm의 `rpicam-*`, libcamera, Picamera2
- [Raspberry Pi 하드웨어와 GPIO](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) — GPIO 전압과 모터 연결 주의
- [Raspberry Pi AI 소프트웨어](https://www.raspberrypi.com/documentation/computers/ai.html) — Pi 5용 선택적 AI 가속 경로

과거 `raspistill`, `raspivid`, 원본 Picamera, wiringPi 중심 튜토리얼은 현재 시스템과 맞지 않을 수 있습니다. 카메라는 `rpicam-*`/libcamera, GPIO는 현재 유지되는 라이브러리 또는 전용 PWM 드라이버를 기준으로 확인합니다.

## ROS 2와 Gazebo

- [ROS 2 Jazzy 문서](https://docs.ros.org/en/jazzy/)
- [ROS 2 Topics, Services, Actions](https://docs.ros.org/en/jazzy/How-To-Guides/Topics-Services-Actions.html)
- [ROS 2 QoS 개념](https://docs.ros.org/en/jazzy/Concepts/Intermediate/About-Quality-of-Service-Settings.html)
- [ROS 2 Jazzy 배포판 정보](https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html)
- [Gazebo ROS 설치와 호환 조합](https://gazebosim.org/docs/jetty/ros_installation/)
- [Gazebo Harmonic ROS 2 통합](https://gazebosim.org/docs/harmonic/ros2_integration/)
- [Navigation2 시작하기](https://docs.nav2.org/getting_started/index.html) — 이동 로봇 확장 시
- [MoveIt 2 문서](https://moveit.picknik.ai/main/index.html) — 로봇 팔 확장 시

## 선택적 시뮬레이션

- [Unity ML-Agents 공식 저장소](https://github.com/Unity-Technologies/ml-agents) — C#·Unity 강화학습 실험
- [NVIDIA Isaac Sim 문서](https://docs.isaacsim.omniverse.nvidia.com/latest/) — NVIDIA 기반 고충실도 로봇 시뮬레이션

현재 팬·틸트 프로젝트는 Gazebo와 mock driver로 충분합니다. Unity와 Isaac Sim은 명확한 2차 목표가 생길 때 선택합니다.

## 전문가 수준으로 확장하는 순서

1. **품질:** property-based test, fuzzing, sanitizer, CI 성능 회귀
2. **비전:** tracker 비교, occlusion 처리, calibration, depth
3. **최적화:** graph profiling, zero-copy, quantization, accelerator backend
4. **제어:** system identification, feed-forward, cascaded control
5. **ROS:** lifecycle, composition, real-time executor, tracing
6. **운영:** OTA, signed model artifact, health monitoring, rollback
7. **제품화:** 개인정보, 보안 위협 모델, 기구·전기 안전, 현장 시험

한 영역의 최신 API를 외우는 것보다 입력부터 물리 출력까지의 계약과 실패를 측정하는 능력이 오래 유지되는 전문성입니다.

