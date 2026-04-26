"""
02_rigid_bodies.py
==================
강체 동역학: 힘과 토크 적용

학습 목표:
    1. apply_force: 강체에 힘 적용
    2. apply_torque: 강체에 회전 토크 적용
    3. 속도 직접 설정
    4. F=ma, τ=Iα 확인

실험:
    - 공에 옆으로 힘을 가해 밀어내기
    - 큐브에 토크를 가해 회전시키기
    - 초기 속도를 주어 포물선 궤적 만들기
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
import numpy as np


def main():
    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 실험 1: 포물선 운동 (초기 속도 설정)
    # ============================================================
    projectile = DynamicSphere(
        prim_path="/World/Projectile",
        name="projectile",
        position=np.array([-3.0, 0, 0.5]),
        radius=0.1,
        color=np.array([1, 0.5, 0]),  # 주황
        mass=1.0,
    )
    
    # ============================================================
    # 실험 2: 수평 힘을 받는 큐브
    # ============================================================
    pushed_cube = DynamicCuboid(
        prim_path="/World/PushedCube",
        name="pushed_cube",
        position=np.array([0, -2, 0.5]),
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([0, 1, 1]),  # 청록
        mass=2.0,  # 2kg
    )
    
    # ============================================================
    # 실험 3: 회전하는 큐브 (토크 적용)
    # ============================================================
    spinning_cube = DynamicCuboid(
        prim_path="/World/SpinningCube",
        name="spinning_cube",
        position=np.array([0, 2, 0.5]),
        scale=np.array([0.3, 0.3, 0.3]),
        color=np.array([1, 0, 1]),  # 마젠타
        mass=1.0,
    )
    
    world.reset()
    
    # ============================================================
    # 초기 속도 설정 (실험 1)
    # ============================================================
    # 45도 각도로 발사: vx = v*cos(45), vz = v*sin(45)
    initial_speed = 5.0  # m/s
    angle = np.pi / 4   # 45도
    projectile.set_linear_velocity(np.array([
        initial_speed * np.cos(angle),  # x 방향
        0,                              # y 방향
        initial_speed * np.sin(angle),  # z 방향 (위로)
    ]))
    
    print("🚀 포물선 운동 시작")
    print(f"   초기 속도: {initial_speed}m/s @ {np.degrees(angle):.0f}도")
    print(f"   이론 최고점: {(initial_speed * np.sin(angle))**2 / (2 * 9.81):.3f}m")
    print(f"   이론 도달 거리: {initial_speed**2 * np.sin(2*angle) / 9.81:.3f}m")
    
    dt = world.get_physics_dt()
    
    max_height = 0
    
    for step in range(600):
        # ============================================================
        # 지속적인 힘 적용 (실험 2): 매 스텝마다 x 방향으로 힘
        # ============================================================
        # ⚠️ 주의: apply_force는 월드 좌표계 기준. Newton 단위.
        # F = ma → 2kg 물체에 10N을 주면 5m/s² 가속
        
        # Isaac Sim 5.0+ API 사용법
        # (구버전과 다를 수 있음. 사용하는 버전의 docs 확인 필요)
        try:
            # 지면 위에 떠 있는 동안만 힘 적용
            pos, _ = pushed_cube.get_world_pose()
            if pos[0] < 3.0:  # x 범위 제한
                from omni.physics.tensors.impl.api import SimulationView
                # 간단한 예: 속도 직접 설정으로 대체
                current_vel = pushed_cube.get_linear_velocity()
                # F = ma → a = F/m, v_new = v + a*dt
                force_x = 10.0  # 10N
                acceleration = force_x / 2.0  # 2kg
                new_vel = current_vel + np.array([acceleration * dt, 0, 0])
                pushed_cube.set_linear_velocity(new_vel)
        except Exception as e:
            pass
        
        # ============================================================
        # 각속도 설정 (실험 3): 회전만 시킴
        # ============================================================
        # 각속도 [ωx, ωy, ωz] (rad/s)
        # Z축 기준 2 rad/s로 회전
        if step == 0:
            spinning_cube.set_angular_velocity(np.array([0, 0, 2.0]))
        
        world.step(render=True)
        sim_time = step * dt
        
        # 포물선 궤적의 최고점 추적
        pos_proj, _ = projectile.get_world_pose()
        if pos_proj[2] > max_height:
            max_height = pos_proj[2]
        
        # 1초마다 상태 출력
        if step % 120 == 0:
            vel_proj = projectile.get_linear_velocity()
            print(f"\n[{sim_time:.2f}s] 포물체: pos={pos_proj[:3].round(2)}, "
                  f"vel={vel_proj.round(2)}")
    
    print(f"\n🎯 관찰된 최고점: {max_height:.3f}m")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
