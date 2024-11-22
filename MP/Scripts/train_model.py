import gymnasium as gym
from environment import *

env = gym.make('QuantumRepeaterGrid-v0', n = 4, pgen = 0.9, pswap = 0.7)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action_idx = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action_idx)

    if done:
        observation, info = env.reset(return_info=True)

env.close()