"""
02_lidar.py
===========
LiDAR 센서 시뮬레이션

학습 목표:
    1. Rotating LiDAR 센서 생성
    2. 포인트 클라우드 데이터 획득
    3. LiDAR 포인트 시각화
    4. 거리 기반 장애물 감지

LiDAR(Light Detection And Ranging):
    레이저를 쏴서 물체까지의 거리를 측정하는 센서.
    자율주행, 로봇 내비게이션의 핵심 센서.

Isaac Sim의 LiDAR 종류:
    - Rotating LiDAR: 360도 스캐닝 (Velodyne 등)
    - Solid State LiDAR: 고정 시야각
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
import numpy as np
import omni.kit.commands
from pxr import Gf


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 주변에 장애물 배치 (LiDAR가 감지할 대상)
    # ============================================================
    # 다양한 거리와 각도에 물체 배치
    obstacles = [
        {"pos": [3.0, 0, 0.5], "scale": [0.5, 0.5, 1.0], "color": [1, 0, 0]},
        {"pos": [0, 3.0, 0.5], "scale": [1.0, 0.5, 1.0], "color": [0, 1, 0]},
        {"pos": [-3.0, 0, 0.5], "scale": [0.5, 1.0, 1.0], "color": [0, 0, 1]},
        {"pos": [0, -3.0, 0.5], "scale": [1.0, 1.0, 1.0], "color": [1, 1, 0]},
        {"pos": [2.0, 2.0, 0.5], "scale": [0.3, 0.3, 1.0], "color": [1, 0, 1]},
        {"pos": [-2.0, -2.0, 0.5], "scale": [0.3, 0.3, 1.0], "color": [0, 1, 1]},
    ]
    
    for i, obs in enumerate(obstacles):
        DynamicCuboid(
            prim_path=f"/World/Obstacle_{i}",
            name=f"obstacle_{i}",
            position=np.array(obs["pos"]),
            scale=np.array(obs["scale"]),
            color=np.array(obs["color"]),
            mass=10.0,  # 무거워서 충돌해도 거의 안 움직임
        )
    
    # ============================================================
    # LiDAR 센서 생성 (RTX LiDAR 사용)
    # ============================================================
    # LiDAR를 위한 빈 부모 생성
    lidar_path = "/World/Lidar"
    
    # RTX LiDAR 생성 (Isaac Sim 5.0+ 방식)
    # 지원 프로파일: Example_Rotary, Velodyne_VLS128 등
    _, lidar_prim = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=lidar_path,
        parent=None,
        config="Example_Rotary",
        translation=(0, 0, 1.0),  # 지면 위 1m 높이에 설치
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # 정면 향함
    )
    
    # LiDAR 데이터 획득 준비
    hydra_texture = None
    try:
        from isaacsim.sensors.rtx import LidarRtx
        
        lidar = LidarRtx(
            prim_path=lidar_path,
            name="rotating_lidar",
            rotation_frequency=10.0,  # 10 Hz 회전
            position=np.array([0, 0, 1.0]),
        )
        lidar.initialize()
        print("✅ LiDAR 초기화 완료")
    except Exception as e:
        print(f"⚠️ LiDAR 초기화 이슈: {e}")
        print("   Isaac Sim 버전에 따라 API가 다를 수 있습니다.")
    
    world.reset()
    
    # ============================================================
    # 시뮬레이션 + LiDAR 데이터 처리
    # ============================================================
    print("\n📡 LiDAR 스캔 시작 (10초)")
    
    for step in range(600):
        world.step(render=True)
        
        # 1초마다 LiDAR 데이터 읽기
        if step % 60 == 0 and step > 0:
            try:
                current_frame = lidar.get_current_frame()
                
                if "data" in current_frame:
                    point_cloud = current_frame["data"]  # (N, 3)
                    
                    if point_cloud is not None and len(point_cloud) > 0:
                        # 통계 출력
                        num_points = len(point_cloud)
                        distances = np.linalg.norm(point_cloud, axis=1)
                        
                        print(f"\n[{step/60:.0f}s] LiDAR 스캔 결과:")
                        print(f"   포인트 수: {num_points}")
                        print(f"   최소 거리: {distances.min():.2f}m")
                        print(f"   최대 거리: {distances.max():.2f}m")
                        print(f"   평균 거리: {distances.mean():.2f}m")
                        
                        # 장애물 감지 (2m 이내)
                        close_points = point_cloud[distances < 2.0]
                        print(f"   2m 이내 포인트: {len(close_points)}개")
                        
                        # 포인트 클라우드를 PLY 파일로 저장
                        if step == 120:  # 2초 시점 데이터만 저장
                            save_point_cloud(point_cloud, f"./lidar_pc_{step:04d}.ply")
            except Exception as e:
                print(f"   데이터 읽기 오류: {e}")
    
    simulation_app.close()


def save_point_cloud(points, filename):
    """포인트 클라우드를 PLY 파일로 저장 (MeshLab 등에서 열람 가능)"""
    try:
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        print(f"   💾 포인트 클라우드 저장: {filename}")
    except Exception as e:
        print(f"   저장 실패: {e}")


if __name__ == "__main__":
    main()
