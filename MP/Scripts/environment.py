import gymnasium as gym
from utils import generate_initial_state, generate_all_actions, _correct_state
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
        self.observation_space = gym.spaces.Box(low=-2.0, high=np.inf,shape=(n, n),dtype=np.float32)

        self.agent_state = generate_initial_state(self.n, self.pgen)
        self.current_state = self.agent_state
        

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.agent_state = generate_initial_state(self.n, self.pgen).astype(np.float32)

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


        candidate_state = np.copy(self.agent_state)

        #Swap first
        for i, node_pos in enumerate(nodes_to_swap):
            if np.random.rand() < self.pswap:
                #Get nodes connected to the node
                input_nodes = np.where(candidate_state[node_pos] > 0.)[0]

                if input_nodes.size < 2:
                    break     

                #Make it's edge alive (set to 1.)
                candidate_state[input_nodes[0], input_nodes[1]] = np.max(candidate_state[node_pos][input_nodes])+1     

                #Remove edges from the initial edges
                candidate_state[input_nodes[0], node_pos] = 0.             
                candidate_state[node_pos, input_nodes[1]] = 0.


        #Find which edges need to wait
        edges_to_wait = []
        relevant_state_mask = np.flip(np.tri(self.n, self.n, k = -1))
        relevant_states = np.multiply(relevant_state_mask, candidate_state)

        for row in range(self.n):
            for col in range(self.n):
                if relevant_states[row, col] > 0.:
                    edges_to_wait.append((row, col))

        #Make unused edges wait
        for i, w in enumerate(edges_to_wait):
            candidate_state[w[0], w[1]] += 1.
                
        #Entangled needed nodes
        for i, e in enumerate(nodes_to_entangle):
            if np.random.rand() < self.pgen:
                candidate_state[e, e+1] = 1.

        self.agent_state = _correct_state(candidate_state).astype(np.float32)

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