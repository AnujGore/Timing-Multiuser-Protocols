import gymnasium as gym
from utils import generate_initial_state, generate_all_actions
import numpy as np

class GridTopologyEnv(gym.Env):
    def __init__(self, n: int, pgen, pswap):
        """
        Environment class for a chain of repeaters. 

        Args:
            n (int): Number of repeaters (including Alice and Bob)
            pgen (float): Probability of generating a link
            pswap (float): Probability of swapping
        
        """
        self.n = n
        self.pgen = pgen
        self.pswap = pswap

        self.all_actions_dict = {}
        for index, element in enumerate(generate_all_actions(self.n)):
            self.all_actions_dict[index] = element

        self.action_space = gym.spaces.Discrete(len(self.all_actions_dict))  #Only upper (or lower) diagonal elements of the n x n matrix
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (n, n))

        self.agent_state = generate_initial_state(self.n, self.pgen)
        self.current_state = self.agent_state
        

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.agent_state = generate_initial_state(self.n, self.pgen)

        return self.agent_state, {}

    def step(self, action_idx):
        action_matrix = self.all_actions_dict[action_idx]

        # #Find which nodes need to swap
        nodes_to_swap = []
        swap_requests = np.array([action_matrix[j, j+2] for j in range(len(action_matrix)-2)])
        for pos, node in enumerate(swap_requests):
            if node == 1:
                nodes_to_swap.append(pos+1)
        
        nodes_to_swap = np.array(nodes_to_swap)
        
        #Find which nodes need to entangle
        nodes_to_entangle = []
        ent_requests = np.array([action_matrix[i, i+1] for i in range(len(action_matrix)-1)])
        for pos, node in enumerate(ent_requests):
            if node == 1:
                nodes_to_entangle.append(pos)
        
        nodes_to_entangle = np.array(nodes_to_entangle)

        #Find which edges need to wait
        all_edges = [i for i in range(0, self.n-1)]
        edges_to_wait = [val for val in all_edges if val not in nodes_to_entangle]



        candidate_state = np.copy(self.agent_state)

        #Swap first
        for i, node_pos in enumerate(nodes_to_swap):
            if np.random.rand() < self.pswap:
                #Get nodes connected to the node
                input_nodes = np.where(candidate_state[node_pos] > 0.)[0]    

                #Make it's edge alive (set to 1.)
                candidate_state[input_nodes[0], input_nodes[1]] = np.max(candidate_state[node_pos][input_nodes])+1     
                candidate_state[input_nodes[1], input_nodes[0]] = np.max(candidate_state[node_pos][input_nodes])+1

                #Remove edges from the initial edges
                candidate_state[node_pos, input_nodes[0]] = 0.             
                candidate_state[node_pos, input_nodes[1]] = 0.
                candidate_state[input_nodes[0], node_pos] = 0.
                candidate_state[input_nodes[1], node_pos] = 0.
                
        for i, e in enumerate(nodes_to_entangle):
            if np.random.rand() < self.pgen:
                candidate_state[e, e+1] = 1.
            
        for i, w in enumerate(edges_to_wait):
            candidate_state[w, w+1] += 1.
                

        terminated = self._is_final_state(self.agent_state)
        reward = 1 if terminated else 0
        observation = self.agent_state

        return observation, reward, terminated, False, {}

    def _is_final_state(self, arr):
        if arr[0][-1] != 0 and arr[-1][0] != 0:
            return True
        return False

gym.register(
    id ='QuantumRepeaterGrid-v0',
    entry_point=GridTopologyEnv
    )