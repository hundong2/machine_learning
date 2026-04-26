"""
01_gym_wrapper.py
=================
Isaac Sim + Gymnasium 호환 환경 만들기

학습 목표:
    1. RL에서 쓰는 Gymnasium 인터페이스 이해
    2. Observation / Action 공간 정의
    3. Reward 설계
    4. Reset / Step 함수 구현

Gymnasium의 표준 인터페이스:
    env.reset() → obs, info
    env.step(action) → obs, reward, terminated, truncated, info
    
📝 참고:
    실제 대규모 강화학습은 Isaac Lab을 사용하는 것이 훨씬 편리합니다.
    Isaac Lab은 이 래퍼 로직을 이미 구현해놨고, GPU 병렬화도 지원합니다.
    
    이 예제는 "RL 환경이 어떻게 만들어지는지" 이해하기 위한 학습용입니다.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("⚠️ gymnasium이 필요합니다: pip install gymnasium")
    simulation_app.close()
    exit()


class ReachTargetEnv(gym.Env):
    """
    간단한 Reach 태스크:
    - 큐브가 에이전트 역할
    - 목표 지점(빨간 구)까지 이동하면 보상
    - action: 큐브에 가할 속도 [vx, vy]
    - observation: [큐브 위치, 목표 위치, 상대 거리]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps=200):
        super().__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # ============================================================
        # Action space: 2D 속도 제어 (vx, vy), 범위 [-1, 1]
        # ============================================================
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # ============================================================
        # Observation space: [agent_x, agent_y, target_x, target_y, dx, dy]
        # ============================================================
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        
        # Isaac Sim 월드
        self.world = World(
            physics_dt=1.0/60.0,
            rendering_dt=1.0/30.0,
            stage_units_in_meters=1.0
        )
        self.world.scene.add_default_ground_plane()
        
        # 에이전트 (파란 큐브)
        from isaacsim.core.api.objects import VisualSphere
        self.agent = DynamicCuboid(
            prim_path="/World/Agent",
            name="agent",
            position=np.array([0, 0, 0.2]),
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0, 0, 1]),
            mass=1.0,
        )
        
        # 목표 마커 (빨간 구)
        self.target_marker = VisualSphere(
            prim_path="/World/Target",
            name="target",
            position=np.array([2.0, 0, 0.1]),
            radius=0.15,
            color=np.array([1, 0, 0]),
        )
        
        # 목표 위치 (x, y)
        self.target_position = np.array([2.0, 0.0])
        
        self.world.reset()
    
    def _get_observation(self):
        """현재 상태 관측값 반환"""
        agent_pos, _ = self.agent.get_world_pose()
        
        obs = np.array([
            agent_pos[0],           # agent x
            agent_pos[1],           # agent y
            self.target_position[0], # target x
            self.target_position[1], # target y
            self.target_position[0] - agent_pos[0],  # dx
            self.target_position[1] - agent_pos[1],  # dy
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, obs):
        """보상 함수 설계"""
        agent_pos = obs[:2]
        target_pos = obs[2:4]
        distance = np.linalg.norm(target_pos - agent_pos)
        
        # 거리가 가까울수록 +, 멀수록 -
        reward = -distance
        
        # 목표 도달 보너스
        if distance < 0.3:
            reward += 100.0  # 큰 보상
        
        return reward, distance < 0.3  # (reward, success)
    
    def reset(self, seed=None, options=None):
        """에피소드 시작 시 호출"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # 에이전트를 원점 근처로 리셋
        random_offset = self.np_random.uniform(-0.5, 0.5, size=2)
        start_pos = np.array([random_offset[0], random_offset[1], 0.2])
        
        self.agent.set_world_pose(position=start_pos)
        self.agent.set_linear_velocity(np.array([0, 0, 0]))
        self.agent.set_angular_velocity(np.array([0, 0, 0]))
        
        # 목표 위치도 랜덤하게
        target_x = self.np_random.uniform(1.5, 3.0)
        target_y = self.np_random.uniform(-1.5, 1.5)
        self.target_position = np.array([target_x, target_y])
        self.target_marker.set_world_pose(
            position=np.array([target_x, target_y, 0.1])
        )
        
        # 물리 안정화
        for _ in range(5):
            self.world.step(render=False)
        
        obs = self._get_observation()
        info = {"target": self.target_position}
        
        return obs, info
    
    def step(self, action):
        """한 스텝 진행"""
        self.current_step += 1
        
        # 액션 해석: [vx, vy] 속도
        # -1 ~ 1 범위를 -2 ~ 2 m/s로 스케일링
        velocity = np.array([
            action[0] * 2.0,
            action[1] * 2.0,
            0.0
        ], dtype=np.float32)
        
        self.agent.set_linear_velocity(velocity)
        
        # 물리 진행 (5 스텝 = 약 0.083초)
        for _ in range(5):
            self.world.step(render=True)
        
        # 관측 및 보상 계산
        obs = self._get_observation()
        reward, reached = self._compute_reward(obs)
        
        terminated = reached  # 목표 도달 시 종료
        truncated = self.current_step >= self.max_steps  # 시간 초과
        
        info = {
            "distance": np.linalg.norm(obs[4:6]),
            "reached": reached,
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.world.stop()


def main():
    """랜덤 정책으로 환경 테스트"""
    env = ReachTargetEnv(max_steps=100)
    
    num_episodes = 3
    for ep in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        print(f"시작 위치: {obs[:2].round(2)}, 목표: {obs[2:4].round(2)}")
        
        total_reward = 0
        
        for step in range(100):
            # 랜덤 액션 (실제로는 학습된 정책을 사용)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 10 == 0:
                print(f"  Step {step}: 거리={info['distance']:.3f}m, 보상={reward:.2f}")
            
            if terminated:
                print(f"  🎯 목표 도달! (step {step})")
                break
            if truncated:
                print(f"  ⏱️ 시간 초과")
                break
        
        print(f"Episode {ep+1} 총 보상: {total_reward:.2f}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
