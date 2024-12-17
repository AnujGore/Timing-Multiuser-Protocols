import numpy as np
import networkx as nx

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



def shortest_path(n, users, cn_loc, system_matrix):
    thisG = nx.grid_graph(dim = (n, n))
    thisAdj = nx.adjacency_matrix(thisG)

    this_graph = nx.from_numpy_array(thisAdj)
    paths = []
    this_graph = this_graph.copy()
    for u in users:
        path = nx.dijkstra_path(this_graph, cn_loc, u)
        this_graph.remove_nodes_from(path[1:])
        paths.append(path)

    for branch in paths:
        for n in range(len(branch)-1):
            system_matrix[branch[n], branch[n+1]] = 1
            system_matrix[branch[n+1], branch[n]] = 1


    return system_matrix

def generate_random_action(n, all_actions_list, **kwargs):
    action_adjacency = np.zeros(shape = (n**2, n**2))
    entanglement_adjacency = np.zeros_like(action_adjacency)

    if np.size(kwargs["users"][0]) != 1:
        users = [n*r[0] + r[1] for r in kwargs["users"]]
    else:
        users = kwargs["users"]
    if type(kwargs["cn_loc"]) == tuple:
        cn_loc = n*kwargs["cn_loc"][0] + kwargs["cn_loc"][1]
    else:
        cn_loc = kwargs["cn_loc"]

    entanglement_adjacency = shortest_path(n, users, cn_loc, entanglement_adjacency)

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
    


