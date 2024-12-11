from itertools import product
import numpy as np

def generate_all_actions(n):

    number_of_repeaters = n**2 - 5
    actions_per_repeater = 4

    all_action_list = product(range(2), number_of_repeaters*actions_per_repeater)
    all_actions = []
    for action in all_action_list:
        all_actions.append(np.reshape(np.array(action), newshape = (actions_per_repeater, number_of_repeaters)))
    
    return np.array(all_actions)

def generate_initial_ind(n):

    matrix = np.zeros(shape = (n**2, n**2))
    off_one_arr = np.tile(np.array([1., 1., 0.]), int(np.ceil(n**2/3)))[:n**2-1]
    matrix += np.diag(off_one_arr, k = 1)
    off_three_arr = np.eye(n**2, n**2, 3)
    matrix += off_three_arr

    return matrix


def find_neighbours(idx, n):
    x, y = divmod(idx, n)
    neighbours = []

    if x > 0:
        neighbours.append((x-1, y))
    if x < n - 1:  # Bottom neighbor
        neighbours.append((x + 1, y))
    if y > 0:  # Left neighbor
        neighbours.append((x, y - 1))
    if y < n - 1:  # Right neighbor
        neighbours.append((x, y + 1))
    
    return [n*r + c for (r, c) in neighbours]


def correct_state(state):
    for i, m in enumerate(state):
        state[i, :, :] += np.rot90(np.fliplr(m))
    return state