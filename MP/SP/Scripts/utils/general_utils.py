import numpy as np

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

def label_to_coor(label:int, n:int):
    row = label//n
    col = label%n
    return (row, col)


def hamming_distance(node1, node2):
    return int(np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1]))
