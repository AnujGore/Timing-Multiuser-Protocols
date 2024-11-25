import numpy as np
from itertools import product
import torch

def generate_initial_state(n, pgen, init_entangled):
    """
    Generates an initial state matrix of size `n x n`.

    The diagonal is set to -1, and off-diagonal elements are initialized 
    with values sampled from a geometric distribution with parameter `pgen`.

    Parameters:
    n (int): Size of the matrix.
    pgen (float): Probability parameter for the geometric distribution.

    Returns:
    np.ndarray: A symmetric `n x n` state matrix.
    """
    arr = np.zeros(shape=(n, n))
    for i in range(n):
        arr[i, i] = -1.  # Set diagonal elements to -1

    if init_entangled:
        for e in range(1, n):
            arr[e-1, e] = np.random.geometric(pgen)  # Populate off-diagonal elements
            
    arr = np.array(arr, dtype=np.float32)
    return _correct_state(arr)

def _correct_state(state):
    """
    Corrects a state matrix to ensure it is symmetric.

    Parameters:
    state (np.ndarray): The input matrix to correct.

    Returns:
    np.ndarray: A symmetric version of the input matrix.
    """
    n = state.shape[0]
    mask_offset_1 = np.flip(np.tri(n, n, k=-1))
    mask = np.flip(np.tri(n, n))
    lower_tri = np.flip(np.multiply(mask_offset_1, state))
    lower_tri = np.rot90(np.flipud(lower_tri))
    upper_diagonal_tri = np.multiply(mask, state)
    return upper_diagonal_tri + lower_tri

def generate_all_actions(n):
    """
    Generates all possible action matrices for a system of size `n`.

    Action matrices encode binary choices for "swap requests" and "entanglement requests."

    Parameters:
    n (int): Size of the system.

    Returns:
    np.ndarray: An array of `(n x n)` matrices, each representing a unique action configuration.
    """
    repeat_val = (n - 2 + n - 1)
    all_actions_as_array = list(product(range(2), repeat=repeat_val))
    all_actions_as_matrix = []

    for arr in all_actions_as_array:
        action_matrix = np.zeros(shape=(n, n))
        for j, swap_request in enumerate(arr[:n-2]):
            action_matrix[j, j+2] = swap_request
        for i, ent_request in enumerate(arr[n-2:]):
            action_matrix[i, i+1] = ent_request
        all_actions_as_matrix.append(action_matrix)

    return np.array(all_actions_as_matrix)


def generate_all_states(n, lifetime):
    """
    Generates all possible states matrices for a system of size `n` for a lifetime `lifetime.

    Parameters:
    n (int): Size of the system.

    Returns:
    np.ndarray: An array of `(n x n)` matrices, each representing a unique action configuration.
    """
    repeat_val = sum([i for i in range(1, n)])
    all_states_as_array = list(product(range(lifetime+1), repeat = repeat_val))
    
    all_states_as_matrix = []
    for arr in all_states_as_array:
        state_matrix = -1*np.ones(shape = (n, n))
        triu_indices = np.triu_indices(n, 1)
        state_matrix[triu_indices] = np.array(arr)
        all_states_as_matrix.append(_correct_state(state_matrix))
    
    return np.array(all_states_as_matrix)

def find_tensor(tensor, target):
    """
    Finds 2d tensor (`target`) among 3d tensor (`tensor`).

    Parameters:
    tensor (tensor): All the tensors to search from
    target (tensor): Target tensor

    Returns:
    int: index of tensors
    
    """
    matches_mask = (tensor == torch.Tensor(target)).all(dim=(1, 2))
    matching_indices = torch.where(matches_mask)[0]
    return matching_indices[0].item()