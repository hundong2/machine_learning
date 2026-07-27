# 07. Raspberry Pi 배포와 운영

## 목표

개발 PC에서 검증한 C++ 추론기를 Raspberry Pi ARM64 환경에 재현 가능하게 설치하고, 전원·열·장치 단절을 포함한 운영 조건에서 안정성을 측정합니다.

## 1단계: 기준 환경

권장 기준:

- Raspberry Pi 5, 64-bit Raspberry Pi OS
- 공식 권장 전원과 능동 냉각
- USB UVC 카메라로 첫 통합
- ONNX Runtime CPU Execution Provider
- OpenCV C++ 최소 모듈
- PCA9685 + 외부 서보 전원

Pi Camera는 `rpicam-hello`로 먼저 독립 검증합니다. 카메라 문제와 모델 문제를 동시에 디버깅하지 않습니다.

## 2단계: 네이티브 빌드와 교차 빌드

처음에는 Pi에서 직접 Release 빌드하는 것이 단순합니다.

```bash
cmake -S . -B build-pi -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRACKER_BUILD_TESTS=ON
cmake --build build-pi
ctest --test-dir build-pi --output-on-failure
```

빌드 시간이 문제가 된 뒤에만 Ubuntu PC의 `aarch64-linux-gnu` toolchain이나 CI 교차 빌드를 추가합니다. 교차 빌드에서는 다음을 고정합니다.

- target architecture와 ABI
- sysroot
- OpenCV와 ONNX Runtime의 target용 라이브러리
- RPATH 또는 설치 경로
- 실제 Pi에서 실행하는 smoke test

## 3단계: ONNX Runtime 배포

공식 C++ 배포 패키지 또는 검증된 ARM64 빌드를 사용하고 출처, 버전, checksum을 기록합니다. 무작위 블로그의 오래된 바이너리를 프로젝트 핵심 의존성으로 사용하지 않습니다.

실행 시작 로그:

```text
app_version
git_commit
model_sha256
contract_version
onnxruntime_version
execution_provider
opencv_version
cpu_model
os_release
camera_backend
input_resolution
```

## 4단계: 성능 기준선

동일한 고정 영상을 사용해 다음 순서로 측정합니다.

1. 카메라·UI 없이 모델만
2. 파일 decode + 모델
3. 카메라 + 모델
4. 카메라 + 모델 + 시각화
5. 전체 제어 loop

각 단계에서:

- warm-up 프레임 수
- 실행 시간
- p50/p95/p99
- 처리 FPS와 입력 FPS
- drop ratio
- RSS memory
- CPU 사용률
- 온도와 스로틀링

해상도와 thread 수를 바꿀 때는 한 항목만 변경합니다.

## 5단계: 카메라 운영

USB 카메라 진단 순서:

1. OS에서 장치 노드 확인
2. 지원 해상도/FPS/포맷 확인
3. 모델 없이 10분 캡처
4. timestamp 단조 증가 확인
5. 케이블 분리·재연결 시험
6. 장치 번호 대신 안정적인 장치 식별 경로 검토

CSI 카메라 진단 순서:

1. `rpicam-hello`
2. `rpicam-vid`로 파일 기록
3. libcamera 파이프라인 단독 검증
4. 애플리케이션 adapter에 연결

## 6단계: 서비스 운영

systemd 서비스 개념 예시:

```ini
[Unit]
Description=Smart AI Tracker
After=network.target

[Service]
Type=simple
User=tracker
WorkingDirectory=/opt/smart-tracker
EnvironmentFile=/etc/smart-tracker.env
ExecStart=/opt/smart-tracker/bin/tracker_cli --config /etc/smart-tracker.yaml
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

실제 서비스에는 다음이 필요합니다.

- 전용 비루트 사용자
- 카메라/I2C 최소 권한
- 읽기 전용 모델 경로
- 로그 크기 제한
- 종료 신호에서 PWM disable
- 연속 restart 제한
- health 상태와 마지막 오류

## 7단계: 설정 분리

소스에 하드코딩하지 않습니다.

```yaml
model:
  path: /opt/smart-tracker/models/person.onnx
  contract: /opt/smart-tracker/models/person.contract.yaml
camera:
  backend: v4l2
  device: /dev/video0
  width: 640
  height: 480
control:
  enabled: false
  dry_run: true
telemetry:
  jsonl_path: /var/log/smart-tracker/metrics.jsonl
```

첫 부팅은 반드시 `control.enabled=false`, `dry_run=true`로 시작합니다.

## 8단계: 30분 soak test

시험 중 주기적으로 기록:

- 메모리 증가
- 온도와 CPU frequency
- 카메라 오류
- 프레임 drop
- ORT 오류
- 평균·p95 지연 변화
- 제어 timeout 횟수

종료 후 메모리와 지연을 시간축으로 그립니다. 마지막 5분의 지연이 첫 5분보다 나빠졌다면 열, 메모리, 로그 I/O를 조사합니다.

## 선택적 가속

- Pi CPU 기준선을 먼저 완성합니다.
- OpenVINO는 Intel 장치 경로입니다.
- TensorRT는 NVIDIA GPU/Jetson 경로입니다.
- Pi 5의 별도 Hailo AI HAT은 공식 Raspberry Pi AI 문서를 따라 별도 backend로 추가할 수 있습니다.

가속기 교체가 `TargetSelector`나 `Controller`에 영향을 주지 않도록 `InferenceBackend` 뒤로 숨깁니다.

## 완료 조건

- 새 Pi에서 문서대로 설치·빌드할 수 있다.
- 30분 soak test 동안 crash와 무한 queue 증가가 없다.
- 온도·스로틀링 조건이 벤치마크에 포함된다.
- 카메라 분리와 프로세스 종료 시 서보가 안전 상태가 된다.
- 서비스와 모델의 버전을 로그로 역추적할 수 있다.

