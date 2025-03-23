import torch
from itertools import product

def generate_all_actions(n):

    num_free_diagonal = n - 2        
    num_one_offset = n - 1  

    combinations = list(product([0, 1], repeat=num_free_diagonal + num_one_offset))

    matrices = torch.zeros((len(combinations), n, n), dtype=torch.float32)

    for idx, comb in enumerate(combinations):
        diag_elements = [0] + list(comb[:num_free_diagonal]) + [0]  # Force first and last diagonal to be 0
        one_offset_elements = comb[num_free_diagonal:]  # One-offset diagonal  

        # Convert to tensors
        diag_tensor = torch.tensor(diag_elements, dtype=torch.float32)
        one_offset_tensor = torch.tensor(one_offset_elements, dtype=torch.float32)

        matrices[idx].diagonal().copy_(diag_tensor)  # Set diagonal
        matrices[idx].diagonal(1).copy_(one_offset_tensor)  # Set one-offset diagonal
        matrices[idx].diagonal(-1).copy_(one_offset_tensor)  # Mirror for symmetry

    return matrices

def generate_all_valid_actions(n):

    matrices = generate_all_actions(n)

    valid_matrices = []

    for matrix in matrices:
        valid = True
        n = matrix.shape[0]

        for i in range(n):
            if matrix[i, i] == 1:
                # If diagonal is 1, row i and column i (excluding M[i, i]) must be zero
                if torch.any(matrix[i, :i]) or torch.any(matrix[i, i+1:]) or \
                   torch.any(matrix[:i, i]) or torch.any(matrix[i+1:, i]):
                    valid = False
                    break

        if valid:
            valid_matrices.append(matrix)

    return torch.stack(valid_matrices) if valid_matrices else torch.empty((0, n, n), dtype=torch.float32)


def generate_all_states(n):
    total_edges = int(((n)*(n-1))/2)

    combinations = list(product(range(2), repeat=total_edges))

    matrices = torch.zeros((len(combinations), n, n), dtype=torch.float32)

    for idx, comb in enumerate(combinations):
        this_matrix = torch.zeros(size = (n, n))
        indexes_upper = torch.triu_indices(n, n, 1)
        this_matrix[indexes_upper[0], indexes_upper[1]] = torch.Tensor(comb)
        
        this_matrix += torch.rot90(torch.fliplr(this_matrix))

        matrices[idx] = this_matrix

    return matrices


def generate_all_valid_states(n):
    all_states = generate_all_states(n)

    all_valid_states = []
    for state in all_states:
        good_state = True
        for node in range(n):
            if (node == 0 or node == n-1):
                if len(state[node].nonzero()) > 1:
                    good_state = False
            else:
                if len(state[node].nonzero()) > 2:
                    good_state = False

        if good_state:
            all_valid_states.append(state)

    return torch.stack(all_valid_states)

def check_if_final_state(state) -> bool:
    chain_state = state[0]
    if (chain_state[0, -1] != 0 and chain_state[-1, 0] != 0):
        return True
    else:
        return False
    
def check_if_bad_state(state: torch.Tensor) -> bool:
    chain_state = state[0]
    if (torch.count_nonzero(chain_state[-1]).item() > 1 or 
        torch.count_nonzero(chain_state[0]).item() > 1 or
        (torch.count_nonzero(chain_state[1:-1], dim = 1) > 2).any()):
        return True
    else:
        return False

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def reward_shape(chain_state, terminated, truncated):
    if terminated:
        return 1000/torch.amax(chain_state).item()
    elif truncated:
        return -100
    
    else:
        return -1


def get_operation_count(states):
    ent_count = []; swap_count = []
    for state in states:
        ent_count.append(torch.amax(state[1]).item())
        swap_count.append(torch.amax(state[2]).item())

    return ent_count, swap_count

def get_state_id(all_states, state):
    matches = torch.all(torch.isclose(all_states, state), dim=(1, 2))
    matching_indices = torch.nonzero(matches, as_tuple=True)[0]
    return matching_indices
        
def get_action_id(all_actions, action):
    matches = torch.all(torch.isclose(all_actions, action), dim=(1, 2))
    matching_indices = torch.nonzero(matches, as_tuple=True)[0]
    return matching_indices