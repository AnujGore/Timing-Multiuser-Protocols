import gymnasium as gym
import numpy as np
from utils import generate_initial_adjanency, label_to_coor, hamming_distance

import json
simulation_parameters = json.load(open("MP/SP/Scripts/config_files/simulation_config.json"))

class GridTopologyEnv(gym.Env):
    def __init__(self, n: int, pgen, pswap, lifetime, action_space):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap
        self.lifetime = lifetime

        self.ent_procedure = simulation_parameters["Entanglement_procedure"]

        self.user_loc = np.array([0, n-1, 2*n-1, n**2 - 1])
        self.cn_loc = int(n*np.floor(n/2) + np.floor(n/2))

        self.action_space = gym.spaces.Discrete(action_space)
        self.observation_space = gym.spaces.Box(low = -2.0, high = self.lifetime+1, shape = (2, n**2, n**2), dtype = np.float32)

    def reset(self):
        self.agent_state = generate_initial_adjanency(self.n, self.pgen, self.lifetime)
        return self.agent_state, {}
    
    def act(self, action):

        action_to_do = action.data
        action_coors = [(action.row[i], action.col[i]) for i in range(action_to_do.size)]

        candidate_state = np.copy(self.agent_state[0]) #Store entanglement matrix
        swap_matrix = np.copy(self.agent_state[1])
        ent_request_matrix = np.copy(self.agent_state[2])

        for idx, a in enumerate(action_to_do):
            #Simulate swaps
            if a == 1:
                swap_matrix[action_coors[idx]] += 1
                if np.random.rand() < self.pswap:
                    connected_nodes = np.where(candidate_state[action_coors[idx][0]] > 0)[0]
                    candidate_state[connected_nodes[0], connected_nodes[1]] = max(candidate_state[action_coors[idx][0]])
                    candidate_state[connected_nodes[1], connected_nodes[0]] = max(candidate_state[action_coors[idx][0]])
                    
                    #Lose swapped links
                    candidate_state[action_coors[idx][0]][candidate_state[action_coors[idx][0]] > 0] = 0
                    candidate_state[:, action_coors[idx][0]][candidate_state[:, action_coors[idx][0]] > 0] = 0

            #Simulate entanglement
            if a == 2:
                if self.ent_procedure == "SenderReceiver":
                    sender_node = np.argmin(np.array([hamming_distance(label_to_coor(action_coors[idx][0], self.n), (int(np.floor(self.n/2)), int(np.floor(self.n/2)))), 
                                                     hamming_distance(label_to_coor(action_coors[idx][1], self.n), (int(np.floor(self.n/2)), int(np.floor(self.n/2))))]))
                    ent_request_matrix[action_coors[idx][sender_node], action_coors[idx][sender_node]] += 1
                    
                if self.ent_procedure == "MeetInTheMiddle":
                    ent_request_matrix[action_coors[idx]] += 0.5
                    
                if np.random.rand() < self.pgen:
                    nodes_to_connect = action_coors[idx]
                    candidate_state[nodes_to_connect[0], nodes_to_connect[1]] = 1.
                    candidate_state[nodes_to_connect[1], nodes_to_connect[0]] = 1. 


        #Kill old edges
        candidate_state[candidate_state > self.lifetime] = 0. 

        self.agent_state = np.stack([candidate_state, swap_matrix, ent_request_matrix])

        observation = self.agent_state
        reward = 1. if self._is_final_state() else 0.
        terminated = self._is_final_state()
        truncated = self._is_bad_state()
        return observation, reward, terminated, truncated, {}


    def _is_final_state(self):
        final_state = True
        for u in self.user_loc:
            if self.agent_state[0, u, self.cn_loc] == 0.:
                final_state = False

        return final_state
    
    def _is_bad_state(self):
        bad_state = False
        for n in range(self.n):
            if n == self.cn_loc:
                if np.sum(self.agent_state[0, n]) > 4:
                    bad_state = True
            else:
                if np.sum(self.agent_state[0, n]) > 2:
                    bad_state = True
        return bad_state

gym.register(
    id ='QuantumRepeatersGrid-v0',
    entry_point=GridTopologyEnv
    )