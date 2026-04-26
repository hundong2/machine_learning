"""
01_gravity_collision.py
=======================
중력 시뮬레이션과 자유낙하 검증

학습 목표:
    1. 중력에 의한 자유낙하 관찰
    2. 이론값과 시뮬레이션 결과 비교
    3. 복원 계수(restitution)의 영향 관찰
    4. Physics Material 설정

이론:
    자유낙하: h = (1/2) * g * t²
    낙하 시간: t = sqrt(2*h/g)
    최종 속도: v = g * t = sqrt(2*g*h)
    
    g = 9.81 m/s² (지구 중력)
    h = 2.0 m 일 때 → t ≈ 0.639s, v ≈ 6.26 m/s
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicSphere
from isaacsim.core.api.materials import PhysicsMaterial
import numpy as np


def main():
    world = World(physics_dt=1.0/240.0, rendering_dt=1.0/60.0)  # 고정밀 물리
    world.scene.add_default_ground_plane()
    
    # ============================================================
    # 서로 다른 복원 계수를 가진 3개의 공
    # ============================================================
    
    # 공 1: 복원 계수 0 (완전 비탄성 - 튕기지 않음)
    material_0 = PhysicsMaterial(
        prim_path="/World/Mat_0",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    ball_0 = DynamicSphere(
        prim_path="/World/Ball_Inelastic",
        name="ball_inelastic",
        position=np.array([-1.0, 0, 2.0]),
        radius=0.1,
        color=np.array([1, 0, 0]),  # 빨강
        mass=1.0,
        physics_material=material_0,
    )
    
    # 공 2: 복원 계수 0.5 (적당히 튕김)
    material_05 = PhysicsMaterial(
        prim_path="/World/Mat_05",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.5,
    )
    ball_05 = DynamicSphere(
        prim_path="/World/Ball_Medium",
        name="ball_medium",
        position=np.array([0.0, 0, 2.0]),
        radius=0.1,
        color=np.array([0, 1, 0]),  # 초록
        mass=1.0,
        physics_material=material_05,
    )
    
    # 공 3: 복원 계수 0.95 (매우 탄성 - 슈퍼볼)
    material_09 = PhysicsMaterial(
        prim_path="/World/Mat_09",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.95,
    )
    ball_09 = DynamicSphere(
        prim_path="/World/Ball_Elastic",
        name="ball_elastic",
        position=np.array([1.0, 0, 2.0]),
        radius=0.1,
        color=np.array([0, 0, 1]),  # 파랑
        mass=1.0,
        physics_material=material_09,
    )
    
    world.reset()
    
    # ============================================================
    # 이론값 계산
    # ============================================================
    h = 2.0 - 0.1  # 시작 높이 - 공 반지름 = 중심이 이동할 거리
    g = 9.81
    t_theory = np.sqrt(2 * h / g)
    v_theory = np.sqrt(2 * g * h)
    
    print("=" * 60)
    print(f"📚 이론 계산 (h={h:.2f}m, g={g}m/s²)")
    print(f"   - 예상 낙하 시간: {t_theory:.4f}s")
    print(f"   - 예상 충돌 속도: {v_theory:.4f}m/s")
    print("=" * 60)
    
    # ============================================================
    # 시뮬레이션: 공이 처음 바닥에 닿는 순간 탐지
    # ============================================================
    first_contact_time = None
    first_contact_velocity = None
    start_time = 0.0
    
    max_height_after_bounce = {'inelastic': 0, 'medium': 0, 'elastic': 0}
    
    dt = world.get_physics_dt()
    
    for step in range(2000):  # 충분한 시간
        world.step(render=True)
        sim_time = step * dt
        
        # 빨간 공(비탄성)으로 낙하 시간 측정
        pos_0, _ = ball_0.get_world_pose()
        vel_0 = ball_0.get_linear_velocity()
        
        # 바닥 근처(z < 0.11) 도달 감지
        if first_contact_time is None and pos_0[2] < 0.11:
            first_contact_time = sim_time
            first_contact_velocity = abs(vel_0[2])
            print(f"\n🎯 빨간 공 바닥 도달!")
            print(f"   - 시뮬레이션 시간: {first_contact_time:.4f}s (이론: {t_theory:.4f}s)")
            print(f"   - 낙하 속도: {first_contact_velocity:.4f}m/s (이론: {v_theory:.4f}m/s)")
            print(f"   - 시간 오차: {abs(first_contact_time - t_theory)*1000:.2f}ms")
        
        # 튕긴 후 최대 높이 추적 (1초 이후부터)
        if sim_time > 0.8:
            pos_05, _ = ball_05.get_world_pose()
            pos_09, _ = ball_09.get_world_pose()
            
            max_height_after_bounce['medium'] = max(
                max_height_after_bounce['medium'], pos_05[2]
            )
            max_height_after_bounce['elastic'] = max(
                max_height_after_bounce['elastic'], pos_09[2]
            )
    
    # ============================================================
    # 최종 결과 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("📊 시뮬레이션 결과 요약")
    print("=" * 60)
    print(f"   복원계수 0.5 공의 최대 튕김 높이: {max_height_after_bounce['medium']:.4f}m")
    print(f"      (이론: 0.5² * {h:.2f} = {0.5**2 * h:.4f}m)")
    print(f"   복원계수 0.95 공의 최대 튕김 높이: {max_height_after_bounce['elastic']:.4f}m")
    print(f"      (이론: 0.95² * {h:.2f} = {0.95**2 * h:.4f}m)")
    print("\n   💡 참고: 실제 시뮬레이션은 수치 오차와 에너지 손실이 있어")
    print("      이론값과 완벽히 일치하지 않습니다.")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
