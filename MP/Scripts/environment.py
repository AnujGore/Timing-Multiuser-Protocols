import gymnasium as gym
from utils import generate_initial_state, generate_all_actions, take_action
import numpy as np

class GridTopologyEnv(gym.Env):
    def __init__(self, n: int, pgen, pswap):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap

        self.all_actions_dict = {}
        for index, element in enumerate(generate_all_actions(self.n)):
            self.all_actions_dict[index] = element

        self.action_space = gym.spaces.Discrete(len(self.all_actions_dict))  #Only upper (or lower) diagonal elements of the n x n matrix
        self.observation_space = gym.spaces.Box(low = -1.0, )

        self.agent_state = generate_initial_state(self.n, self.pgen)
        self.current_state = self.agent_state
        

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.agent_state = generate_initial_state(self.n, self.pgen)

    def step(self, action):
        self.current_state = take_action(action)

        terminated = self._is_final_state(self.agent_state)
        reward = 1 if terminated else 0
        observation = self.agent_state

        return observation, reward, terminated, {}

    def _is_final_state(self, arr):
        if arr[0][-1] != 0 and arr[-1][0] != 0:
            return True
        return False


gym.register(
    id ='QuantumRepeaterGrid-v0',
    entry_point=GridTopologyEnv
    )