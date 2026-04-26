"""
02_joint_control.py
===================
로봇 관절 제어: Position, Velocity, Effort

학습 목표:
    1. 3가지 제어 모드 (Position, Velocity, Effort)
    2. PD Controller 이해 (Stiffness, Damping)
    3. 사인파 궤적 추종
    4. 제어 이득(gain) 튜닝의 영향

제어 모드:
    1. Position Control: 목표 각도 지정 → 내부 PD 제어기가 토크 계산
    2. Velocity Control: 목표 각속도 지정 → 내부 제어기가 토크 계산  
    3. Effort Control: 토크 직접 지정 (가장 낮은 수준)

PD Controller:
    τ = K_p · (q_target - q) + K_d · (q̇_target - q̇)
    
    - K_p (stiffness): 위치 오차를 줄이려는 힘. 크면 빠르지만 진동.
    - K_d (damping): 속도 오차를 줄이려는 힘. 진동을 억제.
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
    
    # Franka 로딩
    assets_root_path = get_assets_root_path()
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Robots/Franka/franka.usd",
        prim_path="/World/Franka"
    )
    
    franka = Articulation(
        prim_paths_expr="/World/Franka",
        name="franka",
    )
    world.scene.add(franka)
    world.reset()
    
    num_dof = franka.num_dof
    print(f"Franka DOF: {num_dof}")
    
    # ============================================================
    # 실험 1: Position Control - 사인파 궤적 추종
    # ============================================================
    # Joint 1만 움직이고, 나머지는 홈 포지션 유지
    home_position = np.array([
        0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.04, 0.04
    ])
    
    # 초기 자세로 설정
    franka.set_joint_positions(home_position.reshape(1, -1))
    
    # 기록용 데이터
    times = []
    target_angles = []
    actual_angles = []
    
    dt = world.get_physics_dt()
    
    print("\n🔵 실험 1: Position Control (사인파 추종)")
    print("-" * 60)
    
    for step in range(600):  # 10초
        sim_time = step * dt
        
        # Joint 1만 사인파로 움직임
        target = home_position.copy()
        amplitude = np.pi / 4  # ±45도 진동
        frequency = 0.5  # Hz
        target[0] = amplitude * np.sin(2 * np.pi * frequency * sim_time)
        
        # 위치 목표 설정
        franka.set_joint_position_targets(target.reshape(1, -1))
        
        world.step(render=True)
        
        # 데이터 기록
        current = franka.get_joint_positions()[0]
        times.append(sim_time)
        target_angles.append(target[0])
        actual_angles.append(current[0])
        
        if step % 60 == 0:
            error = abs(target[0] - current[0])
            print(f"[{sim_time:.2f}s] 목표: {target[0]:+.3f} / 실제: {current[0]:+.3f} "
                  f"/ 오차: {error:.4f}")
    
    # ============================================================
    # 실험 2: 단위 스텝 응답 (Step Response)
    # ============================================================
    print("\n🔴 실험 2: Step Response (급격한 변화에 대한 반응)")
    print("-" * 60)
    
    # 홈 포지션으로 돌아가기
    franka.set_joint_positions(home_position.reshape(1, -1))
    
    for step in range(60):
        world.step(render=True)
    
    # 갑자기 90도 목표 설정
    step_target = home_position.copy()
    step_target[0] = np.pi / 2
    
    print(f"   현재: {franka.get_joint_positions()[0][0]:.3f}")
    print(f"   목표로 변경: {step_target[0]:.3f}")
    
    # 목표 설정 후 응답 관찰
    franka.set_joint_position_targets(step_target.reshape(1, -1))
    
    settling_time = None
    threshold = 0.02  # 2% 오차
    
    for step in range(300):  # 5초 관찰
        world.step(render=True)
        sim_time = step * dt
        
        current = franka.get_joint_positions()[0][0]
        error = abs(step_target[0] - current)
        
        # Settling time 찾기
        if settling_time is None and error < threshold:
            settling_time = sim_time
            print(f"   ⏱️ Settling time: {settling_time:.3f}s")
        
        if step % 30 == 0:
            print(f"[{sim_time:.2f}s] 현재: {current:.4f} / 오차: {error:.4f}")
    
    # ============================================================
    # 결과 플롯 저장 (matplotlib 필요)
    # ============================================================
    try:
        import matplotlib
        matplotlib.use('Agg')  # GUI 없이 파일로만 저장
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, target_angles, 'b--', label='Target', linewidth=2)
        ax.plot(times, actual_angles, 'r-', label='Actual', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Joint Angle (rad)')
        ax.set_title('Franka Joint 1 - Sinusoidal Trajectory Tracking')
        ax.legend()
        ax.grid(True)
        
        import os
        os.makedirs("./plots", exist_ok=True)
        plt.savefig('./plots/joint_tracking.png', dpi=100, bbox_inches='tight')
        print(f"\n📊 플롯 저장: ./plots/joint_tracking.png")
    except ImportError:
        print("\n⚠️ matplotlib이 없어서 플롯을 저장할 수 없습니다.")
        print("   pip install matplotlib")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
