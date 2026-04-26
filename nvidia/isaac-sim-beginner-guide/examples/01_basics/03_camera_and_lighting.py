"""
03_camera_and_lighting.py
=========================
카메라와 조명 설정

학습 목표:
    1. 조명의 종류 (Distant, Sphere, Rect, Dome)
    2. 카메라 생성 및 위치 설정
    3. 카메라 관점 변경

조명 종류:
    - DistantLight: 태양광 같은 평행광 (방향만 중요)
    - SphereLight: 전구 같은 점광원
    - RectLight: 직사각형 면광원 (LED 패널)
    - DomeLight: 환경 조명 (하늘, HDRI)
    - CylinderLight: 형광등 같은 선광원
    - DiskLight: 원형 면광원
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from pxr import UsdLux, Sdf


def add_distant_light(stage, prim_path="/World/DistantLight", intensity=3000.0):
    """태양광 같은 평행광 추가"""
    light = UsdLux.DistantLight.Define(stage, Sdf.Path(prim_path))
    light.CreateIntensityAttr(intensity)
    light.CreateAngleAttr(0.53)  # 태양의 각도 크기
    return light


def add_dome_light(stage, prim_path="/World/DomeLight", intensity=500.0):
    """환경 조명(하늘) 추가"""
    light = UsdLux.DomeLight.Define(stage, Sdf.Path(prim_path))
    light.CreateIntensityAttr(intensity)
    return light


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 1. 조명 추가
    # ============================================================
    stage = world.stage
    add_distant_light(stage, intensity=3000.0)
    add_dome_light(stage, intensity=500.0)
    
    # ============================================================
    # 2. 보이는 물체 추가
    # ============================================================
    for i in range(5):
        DynamicCuboid(
            prim_path=f"/World/Cube_{i}",
            name=f"cube_{i}",
            position=np.array([i * 0.5 - 1.0, 0.0, 0.2 + i * 0.2]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([i/5.0, 0.5, 1.0 - i/5.0]),
            mass=0.5,
        )
    
    # ============================================================
    # 3. 카메라 추가 (로봇에 장착할 수도, 고정 관점으로도 사용)
    # ============================================================
    # 카메라 위치에서 원점을 바라보도록
    camera_position = np.array([3.0, 3.0, 2.0])
    
    # 카메라가 원점을 향하는 회전 계산
    # look_at 벡터
    target = np.array([0, 0, 0.5])
    direction = target - camera_position
    direction = direction / np.linalg.norm(direction)
    
    # 카메라는 기본적으로 -Z 방향을 바라봄
    # 간단한 예제를 위해 오일러 각으로 지정
    camera = Camera(
        prim_path="/World/Camera",
        position=camera_position,
        frequency=20,                  # 20Hz로 이미지 캡처
        resolution=(640, 480),         # 해상도
        orientation=rot_utils.euler_angles_to_quats(
            np.array([0, -30, 135]), degrees=True  # Roll, Pitch, Yaw
        ),
    )
    
    world.reset()
    camera.initialize()
    
    # ============================================================
    # 4. 시뮬레이션 + 카메라 이미지 저장
    # ============================================================
    import os
    os.makedirs("./output_images", exist_ok=True)
    
    for step in range(300):
        world.step(render=True)
        
        # 60 스텝(1초)마다 이미지 캡처
        if step % 60 == 0 and step > 0:
            rgb_data = camera.get_rgba()[:, :, :3]  # (H, W, 3)
            
            if rgb_data is not None and rgb_data.size > 0:
                try:
                    from PIL import Image
                    img = Image.fromarray(rgb_data.astype(np.uint8))
                    save_path = f"./output_images/frame_{step:04d}.png"
                    img.save(save_path)
                    print(f"💾 이미지 저장: {save_path}")
                except ImportError:
                    print("⚠️ PIL이 없습니다. pip install pillow")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
