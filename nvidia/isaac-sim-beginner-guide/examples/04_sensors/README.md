# 📡 Phase 4: 센서

로봇에 달리는 다양한 센서를 시뮬레이션합니다.

## 📋 예제 목록

### 01_camera.py
**목표**: RGB, Depth, Segmentation 데이터 획득
```bash
python 01_camera.py
```

### 02_lidar.py
**목표**: 360도 LiDAR 스캔
```bash
python 02_lidar.py
```

### 03_imu.py
**목표**: 가속도 / 각속도 / 방향 측정
```bash
python 03_imu.py
```

## 🎥 카메라 종류

| 종류 | 용도 |
|------|------|
| RGB | 컴퓨터 비전, SLAM |
| Depth | 물체까지 거리, 3D 재구성 |
| Stereo | 스테레오 매칭 |
| Fisheye | 광각 촬영 (어안) |

## 📊 센서 데이터 형식

### RGB
```python
# shape: (H, W, 3), uint8
rgb = camera.get_rgba()[:, :, :3]
```

### Depth
```python
# shape: (H, W), float32, 단위: meter
depth = camera.get_depth()
```

### LiDAR
```python
# shape: (N, 3), float32, 단위: meter
point_cloud = lidar.get_current_frame()["data"]
```

### IMU
```python
imu_data = imu.get_current_frame()
lin_acc = imu_data["lin_acc"]     # (3,) m/s²
ang_vel = imu_data["ang_vel"]     # (3,) rad/s
orient = imu_data["orientation"]   # (4,) 쿼터니언
```

## 💡 성능 팁

- **해상도**: 높을수록 GPU 메모리 많이 씀. 학습용은 64x64 ~ 256x256도 충분
- **Frequency**: 센서 주기가 physics_dt보다 빠르면 중복 계산
- **Annotation 끄기**: 안 쓰는 annotation(bbox, seg)은 끄면 빨라짐

## ➡️ 다음 단계

[Phase 5: 강화학습](../05_reinforcement_learning/)
