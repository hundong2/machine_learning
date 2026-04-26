"""
예제 3: 커스텀 RL 태스크 만들기 - Reacher

학습 목표:
- 로봇팔이 목표 지점에 end-effector를 도달시키는 태스크 구현
- Observation, Action, Reward 설계 방법 학습
- Gymnasium 인터페이스에 맞는 커스텀 환경 구축

사전 요구사항:
- 02_cartpole_ppo.py 완료
- examples/03_robotics 전체 이해

실행 방법:
    python 03_custom_task.py --train       # 학습
    python 03_custom_task.py --eval        # 평가

태스크 설명:
- 2-DOF 로봇팔 (간단한 구현)
- 목표: 무작위 위치의 타겟에 end-effector 도달
- 관측(obs): [joint_positions, joint_velocities, target_position, distance_vector]
- 행동(action): [joint_1_torque, joint_2_torque]
- 보상(reward): -distance_to_target + bonus_if_reached
"""
import argparse
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="학습 모드")
parser.add_argument("--eval", action="store_true", help="평가 모드")
parser.add_argument("--headless", action="store_true", help="GUI 없이 실행")
parser.add_argument("--timesteps", type=int, default=200_000)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless or args.train})

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere, DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage


class ReacherEnv(gym.Env):
    """
    2-DOF Reacher 태스크.

    목표: end-effector를 타겟 위치에 도달시키기
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_episode_steps: int = 200):
        super().__init__()

        # Isaac Sim World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 타겟 (시각화용 빨간 구)
        self.target = self.world.scene.add(
            VisualSphere(
                prim_path="/World/Target",
                name="target",
                position=np.array([0.3, 0.0, 0.2]),
                radius=0.03,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

        # 간단한 로봇팔 대신 Cuboid 2개로 2-DOF 근사
        # (실제로는 URDF 또는 USD에서 Articulation을 로드)
        self.base = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Base",
                name="base",
                position=np.array([0.0, 0.0, 0.05]),
                scale=np.array([0.08, 0.08, 0.1]),
                color=np.array([0.2, 0.2, 0.8]),
                mass=1.0,
            )
        )

        self.world.reset()

        # 관측/행동 공간
        # obs: [joint1, joint2, vel1, vel2, target_x, target_y, target_z, dx, dy, dz]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        # action: [torque1, torque2] 정규화된 값 [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 내부 상태 (간단한 2-DOF 시뮬레이션)
        self.joint_positions = np.zeros(2, dtype=np.float32)
        self.joint_velocities = np.zeros(2, dtype=np.float32)
        self.link_length = 0.2
        self.dt = 1.0 / 60.0

        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.target_position = np.array([0.3, 0.0, 0.2], dtype=np.float32)

    def _forward_kinematics(self) -> np.ndarray:
        """2-DOF planar 팔의 end-effector 위치 계산."""
        q1, q2 = self.joint_positions
        l = self.link_length
        x = l * np.cos(q1) + l * np.cos(q1 + q2)
        y = l * np.sin(q1) + l * np.sin(q1 + q2)
        z = 0.2  # 고정 높이
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._forward_kinematics()
        diff = self.target_position - ee_pos
        return np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.target_position,
            diff,
        ]).astype(np.float32)

    def _sample_target(self) -> np.ndarray:
        """작업 공간 내 무작위 타겟."""
        r = np.random.uniform(0.1, 2 * self.link_length * 0.9)
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta), 0.2],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()

        # 랜덤 초기 관절 각도
        self.joint_positions = np.random.uniform(-0.3, 0.3, size=2).astype(np.float32)
        self.joint_velocities = np.zeros(2, dtype=np.float32)

        # 새 타겟 샘플링
        self.target_position = self._sample_target()
        self.target.set_world_pose(position=self.target_position)

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # 간단한 관절 동역학: torque → 가속도 → 속도 → 위치
        torque = np.clip(action, -1.0, 1.0) * 2.0  # 스케일링
        damping = 0.1
        accel = torque - damping * self.joint_velocities
        self.joint_velocities += accel * self.dt
        self.joint_positions += self.joint_velocities * self.dt

        # 관절 제한
        self.joint_positions = np.clip(self.joint_positions, -np.pi, np.pi)

        # Isaac Sim 물리 한 스텝 (시각화용)
        self.world.step(render=True)

        obs = self._get_obs()
        ee_pos = self._forward_kinematics()
        distance = np.linalg.norm(self.target_position - ee_pos)

        # 보상: 거리 기반 + 도달 보너스
        reward = -distance
        reached = distance < 0.03
        if reached:
            reward += 10.0

        self.step_count += 1
        terminated = reached
        truncated = self.step_count >= self.max_episode_steps

        info = {"distance": float(distance), "reached": bool(reached)}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        simulation_app.close()


def train():
    """PPO로 Reacher 학습."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(ReacherEnv())
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./runs/reacher/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )
    print(f"학습 시작: {args.timesteps} 스텝")
    model.learn(total_timesteps=args.timesteps)
    model.save("reacher_ppo")
    print("✅ 모델 저장: reacher_ppo.zip")
    env.close()


def evaluate():
    """학습된 모델 평가."""
    from stable_baselines3 import PPO

    env = ReacherEnv()
    try:
        model = PPO.load("reacher_ppo")
    except FileNotFoundError:
        print("❌ reacher_ppo.zip이 없습니다. 먼저 --train을 실행하세요.")
        env.close()
        return

    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print(f"  Episode {episode+1}: 타겟 도달! ({step+1} 스텝, "
                      f"거리={info['distance']:.4f})")
                break
            if truncated:
                print(f"  Episode {episode+1}: 시간 초과 "
                      f"(거리={info['distance']:.4f})")
                break
        print(f"  총 보상: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        print("사용법: python 03_custom_task.py --train 또는 --eval")
        print("       --headless 플래그로 GUI 없이 실행 가능")
        simulation_app.close()
