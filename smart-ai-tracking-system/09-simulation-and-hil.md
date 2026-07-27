# 09. Gazebo 시뮬레이션과 HIL

## 목표

실제 서보와 카메라 없이 ROS 2 인터페이스, 좌표계, 제어 방향, timeout을 검증한 뒤 같은 controller를 하드웨어에 연결합니다.

## 1단계: 시뮬레이션 범위

시뮬레이터에서 검증하기 좋은 것:

- URDF/SDF joint 축과 한계
- pan/tilt 부호
- TF tree
- 카메라 topic
- target 이동 시 제어 안정성
- ROS message·QoS·launch
- sensor dropout과 지연

실제 장치에서만 검증할 것:

- 서보 dead zone, backlash, 기구 마찰
- 전원 강하와 노이즈
- 실제 카메라 노출과 rolling shutter
- 온도와 ARM 성능
- 케이블 간섭

시뮬레이션 성능을 실제 장치 수치로 제시하지 않습니다.

## 2단계: 모델 구조

```text
base_link
└─ pan_joint (revolute)
   └─ pan_link
      └─ tilt_joint (revolute)
         └─ tilt_link
            └─ camera_link
               └─ camera_optical_frame
```

각 joint에:

- axis
- lower/upper limit
- velocity limit
- effort limit
- damping

을 명시합니다. 실제 캘리브레이션 범위보다 넓게 설정하지 않습니다.

## 3단계: Jazzy + Harmonic

공식 권장 조합인 ROS 2 Jazzy와 Gazebo Harmonic을 사용합니다. ROS와 Gazebo transport 사이에는 `ros_gz_bridge`를 둡니다.

브리지 설정은 launch 안에 긴 문자열로 숨기지 말고 YAML로 관리합니다.

```yaml
- ros_topic_name: /camera/image_raw
  gz_topic_name: /world/tracker/model/camera/link/sensor/image
  ros_type_name: sensor_msgs/msg/Image
  gz_type_name: gz.msgs.Image
  direction: GZ_TO_ROS
```

실제 topic 이름과 message type은 생성한 world와 현재 공식 문서에서 확인합니다.

## 4단계: 시뮬레이션 실습

1. 정지된 팬·틸트 모델을 spawn합니다.
2. joint command를 수동으로 보내 축과 부호를 확인합니다.
3. camera image를 ROS 2로 bridge합니다.
4. 움직이는 target을 world에 추가합니다.
5. perception 대신 완벽한 synthetic target observation을 controller에 보냅니다.
6. controller가 안정되면 실제 perception node를 연결합니다.
7. 100ms 지연, 10% drop, 1초 단절을 주입합니다.

완벽한 관측으로 control부터 검증하고, 이후 vision noise를 더하면 문제 원인을 분리하기 쉽습니다.

## 5단계: Software-in-the-loop

모든 구성 요소가 PC에서 실행됩니다.

```text
Gazebo camera → ROS image → C++ perception
→ target observation → controller → simulated joints
```

검사:

- target 이동 방향과 joint 방향
- 정착 시간과 오버슈트
- 메시지 age
- target 유실 상태
- 재시작 후 상태

## 6단계: Hardware-in-the-loop

단계적으로 실제 요소를 교체합니다.

| 단계 | 카메라 | 추론 | 제어 대상 |
| --- | --- | --- | --- |
| SIL | Gazebo | PC | Gazebo joint |
| HIL-A | 녹화 bag | Pi | mock driver |
| HIL-B | 실제 카메라 | Pi | mock driver |
| HIL-C | 실제 카메라 | Pi | 실제 PCA9685·서보 |

각 단계가 이전 단계의 동일 테스트를 통과해야 다음으로 이동합니다.

## 7단계: 선택 확장

### Unity와 C#

Unity 기반 디지털 트윈, 산업 시각화, 사용자 인터페이스가 목표일 때 C# 경로를 추가합니다. 단순 팬·틸트 제어 때문에 Unity ML-Agents를 도입하지 않습니다. 강화학습으로 탐색 정책을 연구할 명확한 문제가 있을 때만 별도 실험으로 둡니다.

### NVIDIA Isaac Sim

Jetson/NVIDIA GPU, 합성 데이터, 고충실도 센서, 대규모 로봇 시뮬레이션이 필요할 때 고려합니다. 현재 프로젝트의 저비용 CPU 기준선보다 시스템 요구사항과 복잡도가 높으므로 2차 프로젝트가 적합합니다.

## 완료 조건

- TF와 joint 축을 그림과 자동 검사로 설명할 수 있다.
- 완벽한 synthetic observation에서 controller가 목표를 만족한다.
- 지연·drop·단절 fault를 재현한다.
- 같은 controller 코드가 mock, Gazebo, 실제 driver에 연결된다.
- SIL 수치와 실제 하드웨어 수치를 분리 보고한다.

