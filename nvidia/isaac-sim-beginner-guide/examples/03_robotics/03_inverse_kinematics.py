"""
03_inverse_kinematics.py
========================
역기구학(IK)을 이용한 엔드이펙터 제어

학습 목표:
    1. Lula Kinematics Solver 사용
    2. 엔드이펙터 목표 위치 설정
    3. IK 해를 관절 각도로 변환
    4. 직선 궤적 그리기

시나리오:
    Franka의 그리퍼 끝을 공중에서 원을 그리도록 이동시킵니다.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects import VisualSphere
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
import numpy as np


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # Franka 로봇 생성 (Isaac Sim의 고수준 API 사용)
    # ============================================================
    franka = Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0]),
    )
    world.scene.add(franka)
    
    # ============================================================
    # 목표 지점 시각화 (작은 빨간 구)
    # ============================================================
    target_marker = VisualSphere(
        prim_path="/World/TargetMarker",
        name="target_marker",
        position=np.array([0.5, 0, 0.5]),
        radius=0.02,
        color=np.array([1, 0, 0]),
    )
    
    # ============================================================
    # 월드 초기화 (articulation 초기화 필요)
    # ============================================================
    world.reset()
    
    # ============================================================
    # RMPFlow Controller: 모션 플래닝 + IK 한꺼번에
    # ============================================================
    # RMPFlow는 IK보다 더 고급: 장애물 회피도 가능
    controller = RMPFlowController(
        name="franka_controller",
        robot_articulation=franka,
    )
    
    print("✅ Franka + RMPFlow Controller 준비 완료")
    print("   엔드이펙터가 원 궤적을 그리도록 합니다.")
    
    # ============================================================
    # 원 궤적 정의: 반지름 0.2m, 중심 (0.5, 0, 0.5)
    # ============================================================
    radius = 0.2
    center = np.array([0.5, 0.0, 0.5])
    
    dt = world.get_physics_dt()
    
    for step in range(1800):  # 30초
        sim_time = step * dt
        
        # ============================================================
        # 원 궤적 위의 점 계산
        # ============================================================
        angle = sim_time * 0.5  # 느린 회전 (0.5 rad/s)
        target_position = center + np.array([
            0,
            radius * np.cos(angle),
            radius * np.sin(angle),
        ])
        
        # 그리퍼 방향: 항상 아래를 향하도록
        # 쿼터니언 [w, x, y, z]: x축 기준 180도 회전 (아래 방향)
        target_orientation = np.array([0, 1, 0, 0])
        
        # 목표 마커 업데이트
        target_marker.set_world_pose(position=target_position)
        
        # ============================================================
        # 컨트롤러에 목표 전달
        # ============================================================
        action = controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=target_orientation,
        )
        
        # 로봇에 명령 적용
        franka.apply_action(action)
        
        world.step(render=True)
        
        # 1초마다 현재 엔드이펙터 위치 출력
        if step % 60 == 0:
            # Franka의 엔드이펙터는 panda_hand 또는 panda_rightfinger
            ee_pose = franka.end_effector.get_world_pose()
            ee_pos = ee_pose[0]
            error = np.linalg.norm(target_position - ee_pos)
            print(f"[{sim_time:5.2f}s] 목표: {target_position.round(3)} | "
                  f"EE: {ee_pos.round(3)} | 오차: {error*1000:.1f}mm")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
