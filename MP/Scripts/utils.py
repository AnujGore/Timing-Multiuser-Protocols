import numpy as np 
from itertools import product


def generate_initial_state(n, pgen):
    arr = np.zeros(shape = (n, n))
    for i in range(n): arr[i, i] = -0.5
    for e in range(1, n): arr[e-1, e] = np.random.geometric(pgen)
    arr = np.array(arr, dtype = np.float32)
    return _correct_state(arr)

def _correct_state(state):
    flipped = np.flip(state)
    state += flipped
    return state


def generate_all_actions(n):
    repeat_val = (n - 2 + n - 1)
    all_actions_as_array = list(product(range(2), repeat=repeat_val))
    
    all_actions_as_matrix = []

    for arr in all_actions_as_array:
        action_matrix = np.zeros(shape = (n, n))
        for j, swap_request in enumerate(arr[:n-2]):
            action_matrix[j, j+2] = swap_request

        for i, ent_request in enumerate(arr[n-2:]):
            action_matrix[i, i+1] = ent_request

        all_actions_as_matrix.append(action_matrix)
        
    return np.array(all_actions_as_matrix)