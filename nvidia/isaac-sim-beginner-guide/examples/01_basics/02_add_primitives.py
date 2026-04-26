"""
02_add_primitives.py
====================
기본 도형(Primitives) 추가하기: 큐브, 구, 실린더, 원뿔

학습 목표:
    1. VisualXxx vs DynamicXxx의 차이 (시각 vs 물리)
    2. 위치, 회전, 크기, 색상 설정
    3. Prim Path 구조 이해 (USD 계층 구조)
    4. 쿼터니언으로 회전 지정하기

Prim Path 설명:
    "/World/Objects/Cube1" 처럼 파일 시스템 경로와 비슷한 구조
    - "/" 로 시작
    - 각 prim은 고유한 경로를 가짐
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import (
    VisualCuboid,     # 시각 전용 (물리 없음)
    DynamicCuboid,    # 물리 적용
    DynamicSphere,
    DynamicCylinder,
    VisualCone,
)
import numpy as np


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 1. 시각 전용 큐브 (Visual Cuboid) - 물리 없음, 떨어지지 않음
    # ============================================================
    visual_cube = VisualCuboid(
        prim_path="/World/VisualCube",
        name="visual_cube",
        position=np.array([0.0, -1.0, 0.5]),  # x, y, z (단위: meter)
        size=0.3,
        color=np.array([1.0, 1.0, 0.0]),  # RGB (0~1) - 노란색
    )
    
    # ============================================================
    # 2. 물리 적용 큐브 (Dynamic Cuboid) - 중력 받음
    # ============================================================
    # 쿼터니언 순서: [w, x, y, z] - Isaac Sim 표준
    # Z축 기준 45도 회전
    angle = np.pi / 4
    quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    
    dynamic_cube = DynamicCuboid(
        prim_path="/World/DynamicCube",
        name="dynamic_cube",
        position=np.array([0.0, 0.0, 2.0]),  # 2m 높이에서 시작
        orientation=quat,                     # 회전 적용
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([1.0, 0.0, 0.0]),     # 빨간색
        mass=1.0,                             # 1kg
    )
    
    # ============================================================
    # 3. 물리 적용 구 (Dynamic Sphere) - 굴러갈 수 있음
    # ============================================================
    sphere = DynamicSphere(
        prim_path="/World/Sphere",
        name="sphere",
        position=np.array([1.0, 0.0, 3.0]),
        radius=0.2,
        color=np.array([0.0, 1.0, 0.0]),     # 초록색
        mass=0.5,
    )
    
    # ============================================================
    # 4. 실린더 (병, 파이프 등 원통형 물체)
    # ============================================================
    cylinder = DynamicCylinder(
        prim_path="/World/Cylinder",
        name="cylinder",
        position=np.array([-1.0, 0.0, 2.5]),
        radius=0.15,
        height=0.4,
        color=np.array([0.0, 0.0, 1.0]),     # 파란색
        mass=0.8,
    )
    
    # ============================================================
    # 5. 시각 전용 원뿔 (장식용)
    # ============================================================
    cone = VisualCone(
        prim_path="/World/Cone",
        name="cone",
        position=np.array([0.0, 1.0, 0.3]),
        radius=0.15,
        height=0.3,
        color=np.array([1.0, 0.0, 1.0]),     # 마젠타
    )
    
    # 월드 초기화
    world.reset()
    
    print("✅ 5개의 객체 생성 완료")
    print("   - 노란 큐브: 시각만 (떨어지지 않음)")
    print("   - 빨간 큐브: 물리 적용 (중력에 낙하)")
    print("   - 초록 구: 물리 적용")
    print("   - 파란 실린더: 물리 적용")
    print("   - 분홍 원뿔: 시각만")
    
    # ============================================================
    # 시뮬레이션 루프: 1초마다 동적 객체의 위치 출력
    # ============================================================
    for step in range(600):
        world.step(render=True)
        
        if step % 60 == 0:  # 1초마다
            # World에서 객체 가져와서 현재 위치 조회
            pos, _ = dynamic_cube.get_world_pose()
            linear_vel = dynamic_cube.get_linear_velocity()
            print(f"[{step/60:.1f}s] 빨간 큐브 위치: {pos}, 속도: {linear_vel}")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
