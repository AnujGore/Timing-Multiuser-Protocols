import numpy as np
from itertools import product
from .env_utils import generate_involved_repeaters

def correct_state(state):
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


def generate_all_actions(n):
    all_actions = []
    swap_combinations = list(product([0, 2], repeat=n**2))
    user_loc = np.array([0, n-1, (n**2)-n, (n**2)-1])
    cn_loc = int(n*np.floor(n/2) + np.floor(n/2))
    involved_edges = generate_involved_repeaters(n)
    ent_combinations = list(product(range(2), repeat=len(involved_edges)))
    for c in swap_combinations:
        arr = np.diag(c)
        for e in ent_combinations:
            arr[involved_edges[:, 0], involved_edges[:, 1]] = e

            #users and cn cannot swap
            arr[user_loc, user_loc] = 0
            arr[cn_loc, cn_loc] = 0.
            
            #Repeaters cannot swap and entangle
            for node in range(n**2 - 1):
                if arr[node, node] == 2:
                    arr[node, :] = 0.
                    arr[:, node] = 0.
                    arr[node, node] = 2.

            all_actions.append(arr)

    all_actions = np.array(all_actions)
    all_actions_flattened = all_actions.reshape(all_actions.shape[0], -1)
    all_actions_flattened = np.unique(all_actions_flattened, axis = 0)
    all_actions = all_actions_flattened.reshape(-1, all_actions.shape[1], all_actions.shape[2])
    
    return all_actions