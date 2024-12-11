import gymnasium as gym
import numpy as np

from utils.general_utils import generate_initial_ind, find_neighbours, correct_state

import json
env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json"))

class GridTopologyEnv():#gym.Env):
    def __init__(self, n: int, pgen, pswap, init_entangled, action_space):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap

        self.init_entangled = init_entangled

        self.ent_procedure = env_parameters["Entanglement_procedure"]

        self.user_loc = np.array([0, n-1, 2*n-1, n**2 - 1])
        self.cn_loc = int(n*np.floor(n/2) + np.floor(n/2))

        self.action_space = gym.spaces.Discrete(action_space)
        self.observation_space = gym.spaces.Box(low = -2.0, high = np.inf, shape = (3, n**2, n**2), dtype = np.float32)

    def reset(self):
        self.agent_state = np.zeros(shape = (4, self.n**2, self.n**2))

        if self.init_entangled:
            self.agent_state[0] = generate_initial_ind(self.n)
            idx = self.agent_state[0].nonzero()
            self.agent_state[1][idx[0], idx[1]] = np.random.geometric(self.pgen, size = idx[0].shape)
            rng = np.arange(self.n**2)
            self.agent_state[2][rng, rng] = self.agent_state[1].max(axis =1)
            self.agent_state[:, self.user_loc, self.user_loc] = 0
            self.agent_state[:, self.cn_loc, self.cn_loc] = 0

        self.agent_state = correct_state(self.agent_state)
        
        return self.agent_state, {}


    def act(self, action):
        for node in range(len(self.n**2)):
            if node in self.user_loc:
                pass
            elif node == self.cn_loc:
                pass
            else:
                node_action = action[node]

                neighbouring_nodes = np.array(find_neighbours(node, self.n))

                for i, neighbour in enumerate(neighbouring_nodes):
                    action_to_do = node_action[i]
                    if action_to_do == 1:
                        self.agent_state[2][node, node] += 0.5                                # change for sender receiver
                        if np.random.rand() < self.pgen:
                            self.agent_state[:2][node, neighbour] = 1
                            self.agent_state[:2][neighbour, node] = 1

                    if action_to_do == 2:
                        
                        self.agent_state[2][node, node] += 1

                         





# gym.register(
#     id ='QuantumRepeatersGrid-v0',
#     entry_point=GridTopologyEnv
#     )

thisGrid = GridTopologyEnv(3, 0.9, 0.6, True, 10)
# thisGrid.reset()