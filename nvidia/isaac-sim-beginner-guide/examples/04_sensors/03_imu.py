"""
03_imu.py
=========
IMU(Inertial Measurement Unit) 센서

학습 목표:
    1. IMU 센서 생성
    2. 선가속도 / 각속도 / 방향 데이터 획득
    3. 실제 로봇의 자세 추정에 IMU가 어떻게 쓰이는지 이해

IMU 구성:
    - Accelerometer: 선가속도 측정 (중력 포함)
    - Gyroscope: 각속도 측정
    - Magnetometer (option): 자기장 측정 (북쪽 방향)
    
IMU로 알 수 있는 것:
    - 로봇의 기울기 (중력 가속도의 방향)
    - 회전 속도
    - 움직임의 가속/감속
    
IMU로 알 수 없는 것:
    - 절대 위치 (이중 적분은 드리프트 때문에 부정확)
    - 절대 방향 (자기 센서 없이는)
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.physics import IMUSensor
import numpy as np


def main():
    world = World(physics_dt=1.0/200.0, rendering_dt=1.0/60.0)  # 200Hz IMU
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # IMU를 장착할 큐브 (로봇을 흉내낸 것)
    # ============================================================
    cube = DynamicCuboid(
        prim_path="/World/IMUCube",
        name="imu_cube",
        position=np.array([0, 0, 2.0]),  # 2m 높이에서 낙하 시작
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([0.5, 0.5, 1.0]),
        mass=1.0,
    )
    
    # ============================================================
    # IMU 센서 생성 (큐브 내부에 장착)
    # ============================================================
    imu = IMUSensor(
        prim_path="/World/IMUCube/IMU",
        name="imu_sensor",
        frequency=200,  # 200Hz 샘플링
        translation=np.array([0, 0, 0]),  # 큐브 중앙
        orientation=np.array([1, 0, 0, 0]),  # 큐브와 같은 방향
    )
    
    world.reset()
    imu.initialize()
    
    print("✅ IMU 센서 초기화 완료 (200Hz)")
    
    # ============================================================
    # 초기 각속도 부여 (회전하며 떨어지도록)
    # ============================================================
    cube.set_angular_velocity(np.array([1.0, 0.5, 2.0]))
    
    # 데이터 저장용
    times = []
    linear_accels = []
    angular_vels = []
    orientations = []
    
    dt = world.get_physics_dt()
    
    print("\n📡 IMU 데이터 수집 시작 (5초)")
    print("-" * 70)
    
    for step in range(1000):  # 5초 @ 200Hz
        world.step(render=True)
        sim_time = step * dt
        
        # ============================================================
        # IMU 데이터 읽기
        # ============================================================
        imu_data = imu.get_current_frame()
        
        if imu_data:
            lin_acc = imu_data.get("lin_acc", np.zeros(3))
            ang_vel = imu_data.get("ang_vel", np.zeros(3))
            orientation = imu_data.get("orientation", np.array([1, 0, 0, 0]))
            
            times.append(sim_time)
            linear_accels.append(lin_acc)
            angular_vels.append(ang_vel)
            orientations.append(orientation)
            
            # 0.25초마다 출력
            if step % 50 == 0:
                print(f"[{sim_time:.3f}s] "
                      f"선가속도: [{lin_acc[0]:+6.2f}, {lin_acc[1]:+6.2f}, {lin_acc[2]:+6.2f}] m/s² | "
                      f"각속도: [{ang_vel[0]:+5.2f}, {ang_vel[1]:+5.2f}, {ang_vel[2]:+5.2f}] rad/s")
    
    # ============================================================
    # 분석: IMU가 측정한 중력 가속도 확인
    # ============================================================
    linear_accels = np.array(linear_accels)
    
    print("\n" + "=" * 70)
    print("📊 IMU 데이터 분석")
    print("=" * 70)
    
    # 바닥에 떨어진 후 정지 상태의 평균 가속도 계산
    if len(linear_accels) > 500:
        static_accel = linear_accels[-200:].mean(axis=0)
        gravity_magnitude = np.linalg.norm(static_accel)
        
        print(f"\n정지 상태 평균 가속도: {static_accel.round(3)}")
        print(f"가속도 크기: {gravity_magnitude:.3f} m/s² (이론: 9.81 m/s²)")
        print(f"→ 중력 벡터: 크기 ~9.81, 위쪽(Z) 방향")
        print("\n💡 설명: 물체가 정지하면 IMU는 중력에 의한 '가짜 가속도'를 읽습니다.")
        print("   이건 로봇의 기울기를 알아내는 데 사용돼요.")
    
    # ============================================================
    # 데이터를 CSV로 저장
    # ============================================================
    import csv
    with open("./imu_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "ax", "ay", "az", "wx", "wy", "wz",
                         "qw", "qx", "qy", "qz"])
        for t, a, w, q in zip(times, linear_accels, angular_vels, orientations):
            writer.writerow([t, *a, *w, *q])
    
    print(f"\n💾 IMU 데이터 저장: ./imu_data.csv ({len(times)} samples)")
    
    # ============================================================
    # 플롯 (matplotlib 필요)
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        la = np.array(linear_accels)
        av = np.array(angular_vels)
        
        ax1.plot(times, la[:, 0], label='ax', color='red')
        ax1.plot(times, la[:, 1], label='ay', color='green')
        ax1.plot(times, la[:, 2], label='az', color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Linear Acceleration (m/s²)')
        ax1.set_title('IMU Accelerometer')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(times, av[:, 0], label='wx', color='red')
        ax2.plot(times, av[:, 1], label='wy', color='green')
        ax2.plot(times, av[:, 2], label='wz', color='blue')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('IMU Gyroscope')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./imu_data.png', dpi=100)
        print(f"📊 플롯 저장: ./imu_data.png")
    except ImportError:
        pass
    
    simulation_app.close()


if __name__ == "__main__":
    main()
