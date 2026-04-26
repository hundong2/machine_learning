# 🧠 Phase 5: 강화학습

Isaac Sim을 사용한 강화학습 기초.

## 📋 예제 목록

### 01_gym_wrapper.py
**목표**: Gymnasium 인터페이스로 Isaac Sim 환경 감싸기
```bash
python 01_gym_wrapper.py
```

### 02_cartpole_ppo.py
**목표**: PPO로 CartPole 학습
```bash
pip install stable-baselines3[extra]
python 02_cartpole_ppo.py
```

## 🎯 RL 학습의 핵심 요소

1. **State (Observation)**: 에이전트가 보는 세계
2. **Action**: 에이전트가 할 수 있는 행동
3. **Reward**: 행동이 얼마나 좋은지
4. **Policy π(a|s)**: 상태→행동 매핑
5. **Value function**: 미래 보상 기대치

## 🚀 Isaac Lab 추천

**대규모 RL 학습은 Isaac Lab을 쓰세요!**

Isaac Lab은 이 폴더의 예제들을 훨씬 더 잘 만든 버전입니다:
- **GPU 병렬화**: 수천 개 환경 동시 실행 (수백 배 빠름)
- **RSL-RL, SKRL, RL-Games 통합**
- **수십 개의 완성된 태스크**

### Isaac Lab 설치
```bash
# Isaac Sim 설치 후
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### 예제 실행
```bash
# Franka reaching task with PPO
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Reach-Franka-v0 --num_envs 4096 --headless

# 시각화
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Reach-Franka-v0 --num_envs 32
```

## 📊 알고리즘 선택 가이드

| 상황 | 추천 알고리즘 |
|------|-------------|
| 연속 액션, 일반적 | **PPO** (가장 안정적) |
| 이산 액션 | DQN, Double DQN |
| 탐색 중요 | SAC (entropy bonus) |
| Off-policy 필요 | SAC, TD3 |
| 다중 에이전트 | MAPPO |

## 💡 보상 설계 팁

- **Dense reward**: 매 스텝 보상 (빠르게 학습, 편향 위험)
- **Sparse reward**: 성공시에만 보상 (느리지만 정직)
- **Shaped reward**: 사람의 직관을 반영 (가이드)
- **Curriculum**: 쉬운 태스크 → 어려운 태스크 점진적으로

## 📚 참고 자료

- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab/)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)
