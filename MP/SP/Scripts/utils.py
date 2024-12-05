import numpy as np
from itertools import product
from scipy import sparse
from collections import deque
import torch

def map_to_routing(listOfRepeaters, n):
    node_labels = []
    for r in listOfRepeaters:
        node_labels.append(r[0]*n + r[1])
    
    routing_labels = []
    for n in range(len(node_labels)-1):
        routing_labels.append((node_labels[n], node_labels[n+1]))

    return routing_labels    

def generate_involved_repeaters(n):
    listOfRepeaters = [
        [(i, int(np.floor(n/2))) for i in range(n)], #Vertical line
        [(int(np.floor(n/2)), i) for i in range(n)], #Horizontal line
        [(i, 0) for i in range(int(np.ceil(n/2)))], #First quadrant line
        [(i, n-1) for i in range(int(np.floor(n/2)), n)], #Second quadrant line
        [(0, i) for i in range(int(np.floor(n/2)), n)], #Third quadrant line
        [(n-1, i) for i in range(int(np.ceil(n/2)))] #Fourth quadrant line
    ]
    
    listOfRepeaters_unnested = []
    for listR in listOfRepeaters:
        labeled_r = map_to_routing(listR, n)
        for ele in labeled_r:
            listOfRepeaters_unnested.append(ele)

    return np.array(listOfRepeaters_unnested)        

def generate_all_actions(n):
    all_actions = []
    swap_combinations = list(product(range(2), repeat=n**2))
    user_loc = np.array([0, n-1, (n**2)-n, (n**2)-1])
    cn_loc = int(n*np.floor(n/2) + np.floor(n/2))
    involved_edges = generate_involved_repeaters(n)
    ent_combinations = list(product([0, 2], repeat=len(involved_edges)))
    for c in swap_combinations:
        arr = np.diag(c)
        for e in ent_combinations:
            arr[involved_edges[:, 0], involved_edges[:, 1]] = e

            #users and cn cannot swap
            arr[user_loc, user_loc] = 0
            arr[cn_loc, cn_loc] = 0.
            
            #Repeaters cannot swap and entangle
            for node in range(n**2 - 1):
                if arr[node, node] == 1:
                    arr[node, :] = 0.
                    arr[:, node] = 0.
                    arr[node, node] = 1.

            all_actions.append(arr)

    all_actions = np.array(all_actions)
    all_actions_flattened = all_actions.reshape(all_actions.shape[0], -1)
    all_actions_flattened = np.unique(all_actions_flattened, axis = 0)
    all_actions = all_actions_flattened.reshape(-1, all_actions.shape[1], all_actions.shape[2])
    # all_actions = list(map(_correct_state, all_actions))
    
    return np.array(list(map(sparse.coo_matrix, all_actions)))


def generate_initial_adjanency(n, pgen, lifetime):
    adjacency = np.zeros(shape = (3, n**2, n**2))
    rng = np.arange(n**2)
    adjacency[:, rng, rng] = -1
    edge_paths = generate_involved_repeaters(n)
    edge_lifes = np.random.geometric(pgen, max(edge_paths.shape))
    old = True
    while old:
        old_edges = np.where(edge_lifes > lifetime)[0]
        if old_edges.size == 0:
            old = False
        new_edges = np.random.geometric(pgen, old_edges.size)
        edge_lifes[old_edges] = new_edges
    adjacency[0, edge_paths[:, 0], edge_paths[:, 1]] = edge_lifes
    return _correct_state(adjacency)

def _correct_state(state):
    """
    Corrects a state matrix to ensure it is symmetric.

    Parameters:
    state (np.ndarray): The input matrix to correct.

    Returns:
    np.ndarray: A symmetric version of the input matrix.
    """

    for i, m in enumerate(state):
        state[i, :, :] += np.rot90(np.fliplr(m))
    rng = np.arange(max(state.shape))
    state[:, rng, rng] = -1
    return state


def label_to_coor(label:int, n:int):
    row = label//n
    col = label%n
    return (row, col)


def hamming_distance(node1, node2):
    return int(np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1]))


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_experience(self, state, action, reward, new_state, terminated, truncated):
        experience = (state, action, reward, new_state, terminated, truncated)
        self.buffer.append(experience)

    def sample_experience(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        return len(self.buffer)
    
def find_coo_matrix(target, total):
    for i, m in enumerate(total):
        if (np.array_equal(target.row, m.row) and
            np.array_equal(target.col, m.col) and
            np.array_equal(target.data, m.data)):
            return i
    