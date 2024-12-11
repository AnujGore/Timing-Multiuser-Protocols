import gymnasium as gym
import numpy as np

from scipy import sparse

from utils.general_utils import label_to_coor, hamming_distance
from utils.model_utils import reward_function

import json
env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json"))

class GridTopologyEnv(gym.Env):
    def __init__(self, n: int, pgen, pswap, action_space):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap

        self.ent_procedure = env_parameters["Entanglement_procedure"]

        self.user_loc = np.array([0, n-1, 2*n-1, n**2 - 1])
        self.cn_loc = int(n*np.floor(n/2) + np.floor(n/2))

        self.action_space = gym.spaces.Discrete(action_space)
        self.observation_space = gym.spaces.Box(low = -2.0, high = np.inf, shape = (3, n**2, n**2), dtype = np.float32)

    def reset(self):
        #Initialize adjancency and life matrices
        adjacency = np.zeros(shape = (3, self.n**2, self.n**2))
        self.agent_state = adjacency
        return self.agent_state, {}
    
    def act(self, action):
        action = sparse.coo_matrix(action)
        action_to_do = action.data
        action_coors = [(action.row[i], action.col[i]) for i in range(action_to_do.size)]

        adjacency_matrix = np.copy(self.agent_state[0]) #Store entanglement matrix
        entanglement_matrix = np.copy(self.agent_state[1])
        swap_matrix = np.copy(self.agent_state[2])

        for idx, a in enumerate(action_to_do):
            #Simulate entanglement
            if a == 1:

                nodes_to_entangle = action_coors[idx]

                #Regardless of success, the node has attempted entanglement
                if self.ent_procedure == "SenderReceiver":
                    node1_coor = label_to_coor(nodes_to_entangle[0], self.n)
                    node2_coor = label_to_coor(nodes_to_entangle[1], self.n)
                    cn_coor = label_to_coor(self.cn_loc, self.n)
                    sender_node_id = np.argmin([
                        hamming_distance(node1_coor, cn_coor),
                        hamming_distance(node2_coor, cn_coor)
                    ])
                    sender_node = nodes_to_entangle[sender_node_id]

                    entanglement_matrix[sender_node, sender_node] += 1
                
                if self.ent_procedure == "MeetInTheMiddle":
                    entanglement_matrix[nodes_to_entangle[0], nodes_to_entangle[0]] += 0.5
                    entanglement_matrix[nodes_to_entangle[1], nodes_to_entangle[1]] += 0.5
                
                #We can establish a link if the attempt is successful
                if np.random.rand() < self.pgen:
                    adjacency_matrix[nodes_to_entangle[0], nodes_to_entangle[1]] = 1.
                    adjacency_matrix[nodes_to_entangle[1], nodes_to_entangle[0]] = 1.

            #Simulate swap
            if a == 2:

                node_to_swap = action_coors[idx]

                #First we ensure that there are two links for the swap to occur
                if np.sum(adjacency_matrix[node_to_swap[0]] > 0) == 2:
                    #Same as above, if the swap is attempted we count the attempt to the latency
                    swap_matrix[node_to_swap[0], node_to_swap[0]] += 1

                    nodes_to_connect = np.where(adjacency_matrix[node_to_swap[0]] == 1.)[0]

                    #We connect the nodes if the swap is successful
                    if np.random.rand() < self.pswap:
                        adjacency_matrix[nodes_to_connect[0], nodes_to_connect[1]] = 1.
                        adjacency_matrix[nodes_to_connect[1], nodes_to_connect[0]] = 1.
                    
                    #Regardless of swap or not, we remove the connected links
                    adjacency_matrix[node_to_swap[0], nodes_to_connect[0]] = 0.
                    adjacency_matrix[node_to_swap[0], nodes_to_connect[1]] = 0.
                    adjacency_matrix[nodes_to_connect[0], node_to_swap[0]] = 0.
                    adjacency_matrix[nodes_to_connect[1], node_to_swap[0]] = 0.

        self.agent_state = np.stack([adjacency_matrix, entanglement_matrix, swap_matrix])

        observation = self.agent_state
        # reward = 1. if self._is_final_state() else 0.
        terminated = self._is_final_state()
        truncated = self._is_bad_state()
        reward = reward_function(terminated, truncated, self.agent_state, self.pgen, self.pswap)
        
        return observation, reward, terminated, truncated, {}


    def _is_final_state(self):
        final_state = True
        for u in self.user_loc:
            if self.agent_state[0, u, self.cn_loc] != 1.:
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