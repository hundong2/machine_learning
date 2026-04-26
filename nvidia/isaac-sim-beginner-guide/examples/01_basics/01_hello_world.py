"""
01_hello_world.py
=================
Isaac Sim 첫 예제: 빈 씬을 띄우고 바닥과 조명만 추가합니다.

실행 방법:
    # 방법 1: pip 설치 후
    python 01_hello_world.py
    
    # 방법 2: Isaac Sim 설치 폴더에서
    ./python.sh path/to/01_hello_world.py
    
학습 목표:
    1. SimulationApp의 역할 이해
    2. World 객체와 physics_dt의 의미
    3. Ground plane과 조명 추가 방법
    4. 시뮬레이션 루프 구조 파악
"""

# ============================================================
# ⚠️ 중요: SimulationApp은 다른 isaacsim 모듈 import 전에 먼저 생성해야 함
# ============================================================
from isaacsim import SimulationApp

# headless=False이면 GUI가 뜨고, True이면 화면 없이 백그라운드 실행
simulation_app = SimulationApp({"headless": False})

# ============================================================
# 이제부터 Isaac Sim API를 import할 수 있음
# ============================================================
from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
import numpy as np


def main():
    # 1. World 객체 생성 (시뮬레이션의 최상위 컨테이너)
    world = World(
        physics_dt=1.0 / 60.0,    # 물리 시뮬레이션: 60Hz (16.67ms마다 계산)
        rendering_dt=1.0 / 30.0,  # 렌더링: 30Hz
        stage_units_in_meters=1.0 # 단위: 미터
    )
    
    # 2. 바닥 추가 (로봇이 서 있을 지면)
    world.scene.add_default_ground_plane(
        z_position=0,        # Isaac Sim은 Z축이 위쪽
        name="default_ground",
        prim_path="/World/defaultGroundPlane",
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0      # 반발 계수 (0이면 통통 튀지 않음)
    )
    
    # 3. 월드 초기화 (physics scene 생성 등)
    world.reset()
    
    print("✅ Isaac Sim 시작 성공!")
    print(f"   - 물리 주기: {world.get_physics_dt()}s")
    print(f"   - 렌더링 주기: {world.get_rendering_dt()}s")
    print("   - ESC 키를 눌러 종료하세요")
    
    # 4. 메인 시뮬레이션 루프
    for step in range(600):  # 600 스텝 (10초 @ 60Hz)
        world.step(render=True)  # 물리 한 스텝 진행 + 렌더링
        
        # 2초마다 한 번 메시지 출력
        if step % 120 == 0:
            sim_time = step * world.get_physics_dt()
            print(f"   시뮬레이션 시간: {sim_time:.2f}s")
    
    # 5. 정리
    simulation_app.close()
    print("👋 시뮬레이션 종료")


if __name__ == "__main__":
    main()
