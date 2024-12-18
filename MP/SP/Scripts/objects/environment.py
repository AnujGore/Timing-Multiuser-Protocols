import gymnasium as gym
import numpy as np
from scipy import sparse
from utils.env_utils import label_to_coor, hamming_distance

class GridTopologyEnv(gym.Env):
    def __init__(self, n: int, pgen, pswap, **kwargs):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap

        self.ent_procedure = kwargs["ent_procedure"]

        self.age_limit = kwargs["age_limit"]

        self.user_loc = np.array([0, n-1, 2*n, n**2 - 1])
        self.cn_loc = int(n*np.floor(n/2) + np.floor(n/2))

        self.action_space = gym.spaces.Box(low = 0.0, high = 2.0, shape = (n**2, n**2), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low = -2.0, high = np.inf, shape = (3, n**2, n**2), dtype = np.float32)

    def reset(self):
        #Initialize adjancency and life matrices
        self.agent_state  = np.zeros(shape = (4, self.n**2, self.n**2))
        return self.agent_state, {}
    
    def act(self, action):
        action = sparse.coo_matrix(action)
        action_to_do = action.data
        action_coors = [(action.row[i], action.col[i]) for i in range(action_to_do.size)]

        adjacency_matrix = np.copy(self.agent_state[0]) #Store entanglement matrix
        age_matrix = np.copy(self.agent_state[1])
        entanglement_matrix = np.copy(self.agent_state[2])
        swap_matrix = np.copy(self.agent_state[3])

        new_edges = []

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

                    age_matrix[nodes_to_entangle[0], nodes_to_entangle[1]] = 1
                    age_matrix[nodes_to_entangle[1], nodes_to_entangle[0]] = 1
                    new_edges.append((nodes_to_entangle[0], nodes_to_entangle[1]))

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

                        age_matrix[nodes_to_connect[0], nodes_to_connect[1]] = max(
                            age_matrix[nodes_to_connect[0], node_to_swap[0]],
                            age_matrix[nodes_to_connect[1], node_to_swap[0]]
                        ) - 1

                        age_matrix[nodes_to_connect[1], nodes_to_connect[0]] = max(
                            age_matrix[nodes_to_connect[0], node_to_swap[0]],
                            age_matrix[nodes_to_connect[1], node_to_swap[0]]
                        ) - 1
                    
                    #Regardless of swap or not, we remove the connected links
                    adjacency_matrix[node_to_swap[0], nodes_to_connect[0]] = 0.
                    adjacency_matrix[node_to_swap[0], nodes_to_connect[1]] = 0.
                    adjacency_matrix[nodes_to_connect[0], node_to_swap[0]] = 0.
                    adjacency_matrix[nodes_to_connect[1], node_to_swap[0]] = 0.

        new_edges = np.array(new_edges, dtype=np.int64)
        non_zero_edges = np.array(np.triu(age_matrix).nonzero()).T

        new_edges_set = set([tuple(i) for i in new_edges.tolist()])
        non_zero_edges_set = set([tuple(i) for i in non_zero_edges.tolist()])
        waited_edges = np.array(list(non_zero_edges_set-new_edges_set))

        if waited_edges.size != 0:
            age_matrix[waited_edges[:, 0], waited_edges[:, 1]] +=1
            age_matrix[waited_edges[:, 1], waited_edges[:, 0]] +=1


        #Kill old edges
        if self.age_limit is not None:
            dead_links_rows = np.where(age_matrix > self.age_limit)[0]
            dead_links_cols = np.where(age_matrix > self.age_limit)[1]

            adjacency_matrix[dead_links_rows, dead_links_cols] = 0
            age_matrix[dead_links_rows, dead_links_cols] = 0

        self.agent_state = np.stack([adjacency_matrix, age_matrix, entanglement_matrix, swap_matrix])

        observation = self.agent_state
        terminated = self._is_final_state()
        truncated = self._is_bad_state()
        reward = 1 if terminated else 0
        
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