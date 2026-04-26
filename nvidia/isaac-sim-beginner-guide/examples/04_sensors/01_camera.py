"""
01_camera.py
============
카메라 센서: RGB, Depth, Segmentation

학습 목표:
    1. 카메라 센서 생성 및 장착
    2. RGB 이미지 획득
    3. Depth map 획득
    4. Instance Segmentation 마스크 획득
    5. 이미지를 파일로 저장

카메라의 종류 (Isaac Sim):
    - 일반 RGB 카메라
    - Stereo 카메라 (좌/우 쌍)
    - Fisheye (어안)
    - Depth 카메라 (RealSense 등을 시뮬레이션)
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import os


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 씬에 다양한 물체 배치 (카메라로 촬영할 대상)
    # ============================================================
    colors = [
        [1, 0, 0],  # 빨강
        [0, 1, 0],  # 초록
        [0, 0, 1],  # 파랑
        [1, 1, 0],  # 노랑
        [1, 0, 1],  # 마젠타
    ]
    
    for i, color in enumerate(colors):
        angle = i * 2 * np.pi / len(colors)
        DynamicCuboid(
            prim_path=f"/World/Cube_{i}",
            name=f"cube_{i}",
            position=np.array([np.cos(angle) * 1.0, np.sin(angle) * 1.0, 0.3]),
            scale=np.array([0.25, 0.25, 0.25]),
            color=np.array(color),
            mass=0.5,
        )
    
    # 중앙에 구
    DynamicSphere(
        prim_path="/World/CenterSphere",
        name="center_sphere",
        position=np.array([0, 0, 0.5]),
        radius=0.2,
        color=np.array([0.8, 0.8, 0.8]),
        mass=1.0,
    )
    
    # ============================================================
    # 카메라 생성
    # ============================================================
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([2.5, 0, 1.5]),
        frequency=30,
        resolution=(1280, 720),  # HD 해상도
        orientation=rot_utils.euler_angles_to_quats(
            np.array([0, -20, 180]), degrees=True
        ),
    )
    
    world.reset()
    camera.initialize()
    
    # ============================================================
    # 여러 종류의 이미지 출력 활성화
    # ============================================================
    # RGB는 기본 활성화
    
    # Depth (거리) 정보 활성화
    camera.add_distance_to_image_plane_to_frame()
    
    # Instance Segmentation (각 객체에 다른 ID)
    camera.add_instance_segmentation_to_frame()
    
    # Semantic Segmentation (객체 클래스)
    camera.add_semantic_segmentation_to_frame()
    
    # 바운딩 박스
    camera.add_bounding_box_2d_tight_to_frame()
    
    # 출력 폴더
    output_dir = "./camera_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"✅ 카메라 초기화 완료 (해상도: 1280x720)")
    print(f"   이미지 저장 경로: {output_dir}")
    
    # ============================================================
    # 시뮬레이션 + 이미지 캡처
    # ============================================================
    for step in range(300):  # 5초
        world.step(render=True)
        
        # 1초마다 캡처
        if step % 60 == 0 and step >= 60:  # 시뮬레이션 안정화 후
            frame_idx = step // 60
            
            # --- RGB ---
            rgba = camera.get_rgba()
            if rgba is not None and rgba.size > 0:
                rgb = rgba[:, :, :3].astype(np.uint8)
                save_image(rgb, f"{output_dir}/rgb_{frame_idx:03d}.png")
            
            # --- Depth ---
            depth = camera.get_depth()
            if depth is not None and depth.size > 0:
                # 깊이 값을 시각화 가능한 형태로 변환
                depth_vis = normalize_depth(depth)
                save_image(depth_vis, f"{output_dir}/depth_{frame_idx:03d}.png", grayscale=True)
                print(f"   📏 Depth 범위: {depth.min():.2f}m ~ {depth.max():.2f}m")
            
            # --- Segmentation ---
            current_frame = camera.get_current_frame()
            if "instance_segmentation" in current_frame:
                seg = current_frame["instance_segmentation"]["data"]
                if seg is not None and seg.size > 0:
                    # 각 인스턴스 ID에 다른 색상
                    seg_vis = colorize_segmentation(seg)
                    save_image(seg_vis, f"{output_dir}/seg_{frame_idx:03d}.png")
            
            print(f"💾 Frame {frame_idx} 저장 완료 (sim_time={step/60:.1f}s)")
    
    print(f"\n✅ 완료. {output_dir} 폴더를 확인하세요.")
    simulation_app.close()


def save_image(arr, path, grayscale=False):
    """numpy 배열을 PNG로 저장"""
    try:
        from PIL import Image
        if grayscale:
            img = Image.fromarray(arr.astype(np.uint8), mode='L')
        else:
            img = Image.fromarray(arr.astype(np.uint8))
        img.save(path)
    except ImportError:
        # PIL 없으면 numpy로 raw 저장
        np.save(path.replace('.png', '.npy'), arr)


def normalize_depth(depth, max_depth=10.0):
    """Depth 맵을 0~255 범위로 정규화 (무한대는 255로)"""
    depth_clipped = np.clip(depth, 0, max_depth)
    depth_vis = (depth_clipped / max_depth * 255).astype(np.uint8)
    return depth_vis


def colorize_segmentation(seg):
    """Segmentation 마스크를 컬러 이미지로 변환"""
    h, w = seg.shape[:2] if len(seg.shape) >= 2 else (seg.shape[0], 1)
    
    # 각 고유 ID에 랜덤 색상 할당
    unique_ids = np.unique(seg)
    np.random.seed(42)
    
    colored = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for uid in unique_ids:
        if uid == 0:  # 배경
            continue
        color = np.random.randint(50, 255, size=3)
        mask = (seg == uid)
        colored[mask] = color
    
    return colored


if __name__ == "__main__":
    main()
