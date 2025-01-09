import torch
import numpy as np
import networkx as nx

def sample_from_prob(tensor, cn_loc, user_loc):
    action_tensor = torch.zeros_like(tensor)

    for i in range(tensor.size(0)):
        diagonal_value = tensor[i, i]
        off_diagonal_values = tensor[i, :].clone()
        off_diagonal_values[i] = float('-inf')

        max_off_diagonal_value = off_diagonal_values.max()
        max_off_diagonal_index = off_diagonal_values.argmax()

        # To repeat or not to repeat; that is the question
        if diagonal_value >= max_off_diagonal_value:
            if i == cn_loc:
                pass
            elif i in user_loc:
                pass
            else:
                action_tensor[i, :] = 0  
                action_tensor[:, i] = 0  
                action_tensor[i, i] = 2 
        else:
            action_tensor[i, max_off_diagonal_index] = 1
            action_tensor[max_off_diagonal_index, i] = 1  
            action_tensor[i, i] = 0  

    return action_tensor

def shortest_path(n):
    system_matrix = np.zeros((n**2, n**2))
    users = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
    cn_loc = (int(np.floor(n/2)), int(np.floor(n/2)))

    this_graph = nx.grid_graph(dim = (n, n))
    paths = []
    this_graph = this_graph.copy()
    for u in users:
        path = nx.dijkstra_path(this_graph, cn_loc, u)
        this_graph.remove_nodes_from(path[1:])
        paths.append(path)

    def to_index(x, y, n):
        return (x*n + y)

    for branch in paths:
        for e in range(len(branch)-1):
            system_matrix[to_index(branch[e][0], branch[e][1], n), to_index(branch[e+1][0], branch[e+1][1], n)] = 1
            system_matrix[to_index(branch[e+1][0], branch[e+1][1], n), to_index(branch[e][0], branch[e][1], n)] = 1

    return system_matrix

def generate_random_action(n, all_actions_list, **kwargs):
    action_adjacency = np.zeros(shape = (n**2, n**2))
    if np.size(kwargs["user_loc"][0]) != 1:
        users = [n*r[0] + r[1] for r in kwargs["user_loc"]]
    else:
        users = kwargs["user_loc"]
    if type(kwargs["cn_loc"]) == tuple:
        cn_loc = n*kwargs["cn_loc"][0] + kwargs["cn_loc"][1]
    else:
        cn_loc = kwargs["cn_loc"]
    entanglement_adjacency = shortest_path(n)
    swap_diag = np.random.choice(np.array([0, 2]), n**2)
    action_adjacency += np.diag(swap_diag)
    entanglement_adjacency[entanglement_adjacency == 1] = np.random.choice(np.arange(2), len(entanglement_adjacency[entanglement_adjacency == 1]))
    entanglement_adjacency += np.rot90(np.fliplr(entanglement_adjacency))
    entanglement_adjacency[entanglement_adjacency > 1] = 1
    action_adjacency += entanglement_adjacency
    action_adjacency[users, users] = 0
    action_adjacency[cn_loc, cn_loc] = 0
    for node in np.arange(n**2):
        if np.sum(np.triu(action_adjacency)[node] > 0) > 1:
            operation_to_do = np.random.choice([1, 2])
            if operation_to_do == 1:
                action_adjacency[node][action_adjacency[node] == 2] = 0
            elif operation_to_do == 2:
                action_adjacency[node][action_adjacency[node] == 1] = 0
        
        if np.sum(np.triu(action_adjacency)[:, node] > 0) != 1:
            operation_to_do = np.random.choice([1, 2])
            if operation_to_do == 1:
                action_adjacency[:, node][action_adjacency[:, node] == 2] = 0
            elif operation_to_do == 2:
                action_adjacency[:, node][action_adjacency[:, node] == 1] = 0
    final_action_adjacency = np.triu(action_adjacency)
    final_action_adjacency += np.rot90(np.fliplr(final_action_adjacency))
    rng = np.arange(n**2)
    final_action_adjacency[rng, rng] /= 2
        
    if any(np.array_equal(final_action_adjacency, arr) for arr in all_actions_list):
        generate_random_action(n, all_actions_list, kwargs)
    else:
        all_actions_list.append(final_action_adjacency)
        return final_action_adjacency, all_actions_list