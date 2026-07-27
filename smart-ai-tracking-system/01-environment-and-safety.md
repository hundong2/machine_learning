# 01. 개발 환경, 부품, 전기 안전

## 권장 환경

### 개발 PC

- Ubuntu 24.04 LTS 권장
- Python 3.11 또는 프로젝트가 검증한 버전
- C++20, CMake 3.24 이상, Ninja
- OpenCV, ONNX Runtime C++
- Git, pre-commit, clang-format, clang-tidy
- ROS 2 Jazzy와 Gazebo Harmonic은 ROS 단계에서 설치

Windows나 macOS에서도 1~7주차의 핵심 코드는 만들 수 있습니다. ROS 2·Gazebo·Raspberry Pi와 환경 차이를 줄이려면 Ubuntu 24.04 개발 환경이 가장 단순합니다. 운영체제와 패키지 버전은 `docs/environment.md`에 기록합니다.

### Python 가상 환경 예시

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision ultralytics onnx onnxruntime opencv-python pytest
pip freeze > requirements-lock.txt
```

운영체제, CPU/GPU, Python, PyTorch, Ultralytics, ONNX, ONNX Runtime 버전을 실행 로그 첫 부분에 출력하세요. 단순한 `requirements.txt`보다 실제 검증 버전과 모델 checksum이 재현성에 더 중요합니다.

### C++ 빌드 기준

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

경고를 가능한 한 오류로 취급하고, AddressSanitizer·UndefinedBehaviorSanitizer용 개발 프리셋과 Release 벤치마크 프리셋을 분리합니다.

## 하드웨어 BOM

| 부품 | 권장 | 이유 |
| --- | --- | --- |
| SBC | Raspberry Pi 5 8GB 또는 Pi 4 | 64비트 OS와 충분한 메모리 |
| 전원 | 보드 공식 권장 전원 | 저전압·스로틀링 방지 |
| 냉각 | Pi 5용 능동 냉각 | 긴 추론 부하에서 클럭 유지 |
| 카메라 | USB UVC 웹캠 또는 Camera Module | USB는 입문이 쉽고 CSI는 지연·통합 학습에 유리 |
| 팬·틸트 | SG90 2개 + 기구 브래킷 | 저비용 2축 실험 |
| PWM | PCA9685 I2C 드라이버 | 안정적인 다채널 PWM |
| 서보 전원 | 별도 5V, 충분한 전류 용량 | 보드 전원 강하·리셋 방지 |
| 배선 | 점퍼선, 공통 GND, 필요 시 퓨즈 | 안전한 기준 전위와 보호 |

초기 예산에는 microSD, 전원, 냉각, PWM 드라이버, 브래킷을 포함해야 합니다. “Pi와 서보 2개만 사면 끝”이라고 계산하면 전원 불안정과 기구 간섭 때문에 일정이 흔들립니다.

## 전기 안전 원칙

1. Raspberry Pi GPIO는 3.3V 논리입니다.
2. 모터나 서보의 전원을 GPIO 핀에서 직접 공급하지 않습니다.
3. 서보는 별도 5V 전원을 사용하고 Raspberry Pi와 **GND를 공통 연결**합니다.
4. PCA9685의 논리 전원과 서보 전원 단자를 구분합니다.
5. 전원을 넣기 전에 멀티미터로 극성과 전압을 확인합니다.
6. 팬·틸트 기구를 분리한 상태에서 작은 각도 범위로 먼저 시험합니다.
7. 각도 제한, 속도 제한, timeout을 소프트웨어에 둡니다.
8. 사람이 손으로 즉시 전원을 끌 수 있는 위치에서 첫 시험을 합니다.

Raspberry Pi 공식 문서도 모터를 GPIO에 직접 연결하지 말고 적절한 모터 컨트롤러를 사용하라고 안내합니다. 서보는 별도 전원과 PWM 드라이버를 쓰는 구성이 안전하고 재현성도 좋습니다.

## 카메라 스택 선택

Raspberry Pi OS Bookworm의 공식 카메라 도구는 `rpicam-*`이며 `libcamera`를 기반으로 합니다. 과거 튜토리얼의 `raspistill`, `raspivid`, 원본 Picamera 스택은 레거시이므로 새 프로젝트의 기본으로 사용하지 않습니다.

- USB UVC: OpenCV `VideoCapture`로 빠르게 시작
- CSI Camera: 먼저 `rpicam-hello`로 하드웨어 확인
- C++ 제품 경로: libcamera 또는 rpicam-apps 구조를 검토
- Python 진단 경로: Picamera2 사용 가능

카메라 API를 추론 엔진에 직접 묶지 말고 다음 인터페이스를 둡니다.

```cpp
struct Frame {
  cv::Mat image;
  std::uint64_t sequence;
  std::chrono::steady_clock::time_point captured_at;
};

class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual std::optional<Frame> next() = 0;
};
```

그러면 `VideoFileSource`, `UsbCameraSource`, `LibcameraSource`, `RosImageSource`를 교체할 수 있습니다.

## 하드웨어 없이 시작하는 모의 장치

실제 서보보다 먼저 `MockServoDriver`를 작성합니다.

```cpp
class ServoDriver {
 public:
  virtual ~ServoDriver() = default;
  virtual void set_degrees(float pan, float tilt) = 0;
  virtual void disable() noexcept = 0;
};
```

모의 드라이버는 명령 시간, 제한 전·후 각도, 상태를 CSV로 남깁니다. Python으로 각도·오차 그래프를 그려 PID 튜닝과 상태 전이를 검증합니다.

## 실습

1. PC 환경 정보와 설치 버전을 `docs/environment.md`에 저장합니다.
2. 카메라로 30초 MP4를 만들고 해상도·FPS·코덱을 기록합니다.
3. `FrameSource`와 `MockServoDriver`의 최소 인터페이스를 설계합니다.
4. `MockServoDriver`가 허용 범위를 넘는 명령을 거부하는 테스트를 작성합니다.
5. 하드웨어를 구매한다면 배선도를 먼저 그리고 리뷰 체크리스트를 작성합니다.

## 완료 조건

- 개발 PC에서 Python과 C++ 샘플을 각각 실행할 수 있다.
- 동일한 30초 입력 영상이 모든 성능 비교에 사용된다.
- 실제 서보 없이 모의 드라이버를 테스트할 수 있다.
- 전원, 공통 GND, 각도 제한, 비상 정지 계획이 문서화되어 있다.

