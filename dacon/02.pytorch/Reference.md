# 목차

- [노름(Norm)](#1노름norm은-벡터의-크기를-재는-함수)
- [Tensorboard](#2tensorboard)

# 1.노름(norm)은 벡터의 크기를 재는 함수

- 조건: 항상 0 이상, 0은 영벡터에서만, 스칼라배에 비례, 삼각부등식 만족
- 가장 많이 쓰는 것:
  - $L1$ 노름: $\|x\|_1=\sum_i |x_i|$
  - $L2$ 노름(유클리드): $\|x\|_2=\sqrt{\sum_i x_i^2}$
  - $L_\infty$ 노름: $\|x\|_\infty=\max_i |x_i|$

행렬에도 노름이 있고, 딥러닝에서는 가중치 크기 제어(정규화/규제), 거리 측정, 손실 계산에 자주 씁니다.

# 2.Tensorboard 

- [Tensorboard](./15.Configuration.ipynb)
