import numpy as np 
from itertools import product

def generate_initial_state(n, pgen):
    arr = np.zeros(shape = (n, n))
    for i in range(n): arr[i, i] = np.NaN 
    for e in range(1, n): arr[e-1, e] = np.random.geometric(pgen)
    return _correct_state(arr)

def _correct_state(state):
    flipped = np.flip(state)
    state += flipped
    return state


def take_action(action, state):
    print(action)
    print(state)
    return


def generate_all_actions(n):
    repeat_val = (n + n-1)
    all_actions_as_array = list(product(range(2), repeat=repeat_val))
    
    all_actions_as_matrix = []
    for arr in all_actions_as_array:
        action_matrix = np.diag(arr[:n])
        for i, ele in enumerate(arr[n:]):
            action_matrix[i, i+1] = ele

        all_actions_as_matrix.append(action_matrix)
        
    return np.array(all_actions_as_matrix)    