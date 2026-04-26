"""
02_cartpole_ppo.py
==================
CartPole(역진자) PPO 강화학습

학습 목표:
    1. 고전적인 RL 벤치마크: CartPole 이해
    2. Stable-Baselines3의 PPO로 학습
    3. 학습된 정책 평가 및 가시화
    
CartPole 문제:
    카트 위에 서 있는 막대를 넘어지지 않도록 좌우로 움직이며 균형 잡기.
    
State (obs):
    - 카트 위치 (x)
    - 카트 속도 (v)
    - 막대 각도 (θ)
    - 막대 각속도 (ω)
    
Action:
    - 카트에 가할 힘 방향 (-1 또는 +1 또는 연속값)
    
Reward:
    - 매 스텝 +1 (오래 서 있을수록 좋음)
    - 막대가 쓰러지면 에피소드 종료

설치:
    pip install stable-baselines3[extra]
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
    print("⚠️ pip install gymnasium")
    simulation_app.close()
    exit()


class CartPoleEnv(gym.Env):
    """
    Isaac Sim 기반 CartPole 환경
    
    ⚠️ 이 예제는 단순화를 위해 자체 구현된 카트-폴을 사용합니다.
    실제 Isaac Sim에는 미리 만들어진 CartPole USD가 있습니다:
    /Isaac/Robots/CartPole/cartpole.usd
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, max_steps=500):
        super().__init__()
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action: 카트에 가할 수평 힘
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: [cart_x, cart_v, pole_theta, pole_omega]
        obs_limit = np.array([10.0, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_limit, high=obs_limit, dtype=np.float32)
        
        # 실패 조건
        self.pole_angle_threshold = np.pi / 4  # ±45도 이상 기울면 실패
        self.cart_position_threshold = 2.5     # ±2.5m 이상 벗어나면 실패
        
        # Isaac Sim 설정
        self.world = World(
            physics_dt=1.0/60.0,
            rendering_dt=1.0/30.0,
        )
        self.world.scene.add_default_ground_plane()
        
        # ============================================================
        # 간단한 카트-폴 구성 (물리 특성은 이상적이지 않음)
        # 진지한 학습은 Isaac Lab의 CartPole을 사용하세요.
        # ============================================================
        self._setup_cartpole()
        
        self.world.reset()
    
    def _setup_cartpole(self):
        """간단한 카트-폴 구조 생성"""
        # 카트 (큐브)
        self.cart = DynamicCuboid(
            prim_path="/World/Cart",
            name="cart",
            position=np.array([0, 0, 0.1]),
            scale=np.array([0.4, 0.3, 0.2]),
            color=np.array([0.3, 0.3, 0.3]),
            mass=1.0,
        )
        
        # 폴 (긴 막대)
        self.pole = DynamicCuboid(
            prim_path="/World/Pole",
            name="pole",
            position=np.array([0, 0, 0.7]),
            scale=np.array([0.05, 0.05, 1.0]),
            color=np.array([0.8, 0.4, 0.1]),
            mass=0.1,
        )
        # 실제로는 Revolute Joint로 카트와 폴을 연결해야 함
        # 이 예제는 시뮬레이션 구조보다 RL 흐름에 집중
    
    def _get_observation(self):
        cart_pos, _ = self.cart.get_world_pose()
        cart_vel = self.cart.get_linear_velocity()
        pole_pos, pole_quat = self.pole.get_world_pose()
        pole_ang_vel = self.pole.get_angular_velocity()
        
        # 폴의 각도 계산 (Y축 기준 기울어진 정도)
        # 쿼터니언에서 Y축 회전 각도 추출
        w, x, y, z = pole_quat
        pole_angle = 2 * np.arctan2(np.sqrt(x*x + z*z), w) * np.sign(x if abs(x) > abs(z) else z)
        pole_omega = pole_ang_vel[1]  # Y축 각속도
        
        obs = np.array([
            cart_pos[0],
            cart_vel[0],
            pole_angle,
            pole_omega,
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 리셋 (약간의 노이즈 추가)
        noise = self.np_random.uniform(-0.05, 0.05, size=4)
        
        self.cart.set_world_pose(position=np.array([noise[0], 0, 0.1]))
        self.cart.set_linear_velocity(np.array([noise[1], 0, 0]))
        
        self.pole.set_world_pose(
            position=np.array([noise[0], 0, 0.7]),
        )
        self.pole.set_angular_velocity(np.array([0, noise[3], 0]))
        
        for _ in range(3):
            self.world.step(render=False)
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        # 액션: 카트에 힘 적용 (간단화: 속도 직접 변경)
        force = float(action[0]) * 10.0
        current_vel = self.cart.get_linear_velocity()
        new_vel = current_vel + np.array([force * self.world.get_physics_dt(), 0, 0])
        self.cart.set_linear_velocity(new_vel)
        
        # 시뮬레이션 진행
        self.world.step(render=True)
        
        obs = self._get_observation()
        cart_x, _, pole_theta, _ = obs
        
        # 보상 계산
        reward = 1.0  # 서 있을 때마다 +1
        
        # 실패 조건 체크
        terminated = False
        if abs(pole_theta) > self.pole_angle_threshold:
            terminated = True
        if abs(cart_x) > self.cart_position_threshold:
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        info = {"cart_x": cart_x, "pole_theta": pole_theta}
        
        return obs, reward, terminated, truncated, info


def train_ppo():
    """PPO 알고리즘으로 학습"""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("⚠️ stable-baselines3가 필요합니다:")
        print("   pip install stable-baselines3[extra]")
        return
    
    print("🧠 PPO 학습 시작")
    print("=" * 60)
    
    env = CartPoleEnv(max_steps=200)
    env = Monitor(env)
    
    # PPO 에이전트 생성
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )
    
    # 학습 (짧게 - 실제로는 100,000 스텝 이상 필요)
    print("\n⏳ 학습 중... (5,000 스텝 - 실험용. 실제는 100k+ 필요)")
    model.learn(total_timesteps=5000)
    
    # 모델 저장
    model.save("./ppo_cartpole")
    print("💾 모델 저장: ./ppo_cartpole.zip")
    
    # ============================================================
    # 학습된 모델 평가
    # ============================================================
    print("\n🎯 학습된 정책 평가 (3 에피소드)")
    print("=" * 60)
    
    for ep in range(3):
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {ep+1}: {step+1} steps, total reward = {total_reward:.1f}")
    
    env.close()


if __name__ == "__main__":
    train_ppo()
    simulation_app.close()
