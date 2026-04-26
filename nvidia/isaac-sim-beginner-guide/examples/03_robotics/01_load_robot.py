"""
01_load_robot.py
================
미리 만들어진 로봇 로딩하기 (Franka Panda, Jetbot)

학습 목표:
    1. Isaac Sim의 내장 로봇 사용
    2. USD 파일 경로 구조 이해
    3. Articulation API로 로봇 정보 조회
    4. 초기 관절 값 설정

Isaac Sim에 포함된 주요 로봇:
    - Franka Emika Panda (7-DOF 팔 + 그리퍼)
    - UR10/UR5/UR3 (유니버설 로봇)
    - Jetbot (2-바퀴 이동 로봇)
    - Carter (4-바퀴 이동 로봇)
    - Humanoid (휴머노이드)
    - Ant (네 다리 강화학습용)
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.storage.native import get_assets_root_path
import numpy as np


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/30.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # Isaac Sim 에셋 경로 가져오기
    # ============================================================
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("❌ Isaac Sim 에셋 서버에 접근할 수 없습니다.")
        print("   Nucleus 서버가 실행 중인지 확인하세요.")
        simulation_app.close()
        return
    
    print(f"📁 에셋 루트 경로: {assets_root_path}")
    
    # ============================================================
    # 방법 1: Franka Panda 로봇 로딩
    # ============================================================
    # Franka USD 파일 경로
    franka_usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
    
    # Stage에 참조 추가 (USD파일을 씬에 로드)
    add_reference_to_stage(
        usd_path=franka_usd_path,
        prim_path="/World/Franka"
    )
    
    # Articulation 객체로 래핑
    franka = Articulation(
        prim_paths_expr="/World/Franka",
        name="franka",
        positions=np.array([[0.0, 0.0, 0.0]]),  # 월드에서의 위치
    )
    
    # ============================================================
    # 방법 2: Jetbot (이동 로봇) 로딩
    # ============================================================
    jetbot_usd_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
    add_reference_to_stage(
        usd_path=jetbot_usd_path,
        prim_path="/World/Jetbot"
    )
    jetbot = Articulation(
        prim_paths_expr="/World/Jetbot",
        name="jetbot",
        positions=np.array([[1.5, 0.0, 0.03]]),  # franka 옆에 배치
    )
    
    # World에 등록
    world.scene.add(franka)
    world.scene.add(jetbot)
    
    # 초기화
    world.reset()
    
    # ============================================================
    # 로봇 정보 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("🦾 Franka Panda 정보")
    print("=" * 60)
    print(f"   DOF 개수: {franka.num_dof}")
    print(f"   관절 이름: {franka.dof_names}")
    
    joint_positions = franka.get_joint_positions()
    print(f"   현재 관절 위치 (rad): {joint_positions[0].round(3)}")
    
    # 조인트 한계
    dof_props = franka.get_dof_properties()
    if dof_props is not None:
        print(f"   조인트 한계 (rad):")
        for i, name in enumerate(franka.dof_names):
            lower = dof_props['lower'][i] if 'lower' in dof_props else -np.pi
            upper = dof_props['upper'][i] if 'upper' in dof_props else np.pi
            print(f"      {name}: [{lower:.2f}, {upper:.2f}]")
    
    print("\n" + "=" * 60)
    print("🚗 Jetbot 정보")
    print("=" * 60)
    print(f"   DOF 개수: {jetbot.num_dof}")
    print(f"   관절 이름: {jetbot.dof_names}")
    
    # ============================================================
    # Franka를 "준비 자세"로 이동
    # ============================================================
    # 일반적인 준비 자세 (home position)
    home_position = np.array([[
        0.0,       # panda_joint1 (base rotation)
        -np.pi/4,  # panda_joint2 (shoulder)
        0.0,       # panda_joint3
        -3*np.pi/4,# panda_joint4 (elbow)
        0.0,       # panda_joint5
        np.pi/2,   # panda_joint6 (wrist)
        np.pi/4,   # panda_joint7
        0.04,      # panda_finger_joint1 (gripper)
        0.04,      # panda_finger_joint2 (gripper)
    ]])
    
    # DOF 수가 맞는지 확인 후 설정
    if franka.num_dof == home_position.shape[1]:
        franka.set_joint_positions(home_position)
        print(f"\n✅ Franka를 홈 포지션으로 이동")
    
    # ============================================================
    # 시뮬레이션: 로봇이 가만히 서 있는 상태 관찰
    # ============================================================
    print("\n🎬 시뮬레이션 시작... (10초)")
    
    for step in range(600):
        world.step(render=True)
    
    print("✅ 완료")
    simulation_app.close()


if __name__ == "__main__":
    main()
