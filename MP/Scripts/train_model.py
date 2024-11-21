import gymnasium as gym
from environment import *

env = gym.make('QuantumRepeaterGrid-v0', n = 4, pgen = 0.9, pswap = 0.7)


action = env.action_space.sample()
obs, reward, done, _ = env.step(action)