import gymnasium as gym
from environment import *

pgen = 1
pswap = 1

env = gym.make('QuantumRepeaterGrid-v0', n = 5, pgen = pgen, pswap = pswap)

observation, info = env.reset(seed=42)

for _ in range(1000):
    action_idx = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action_idx)

    if done:
        observation, info = env.reset()

env.close()