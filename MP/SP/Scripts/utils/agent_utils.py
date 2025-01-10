import torch
import numpy as np
import networkx as nx

def sample_from_prob(tensor):
    #First we sample the upper triangular action matrix
    action_tensor_triu = torch.zeros_like(tensor)
    tensor_triu = tensor.clone()

    rng = torch.arange(0, tensor.size(0))

    for i in range(tensor_triu.size(0)):
        if sum(tensor_triu[i]) != 0:
            index = torch.multinomial(tensor_triu[i], 1).item()         #This ensures we sample from the elements rather than a coin toss
            if index == i:
                action_tensor_triu[i, i] = 2
            else:
                action_tensor_triu[i] = torch.ones(action_tensor_triu[i].size(0))
                action_tensor_triu[i][tensor_triu[i] == 0] = 0
                action_tensor_triu[i, i] = 0

    #Then the lower triangular
    action_tensor_tril = torch.zeros_like(tensor)
    tensor_tril = tensor.t()
    for i in range(tensor_tril.size(0)):
        if sum(tensor_tril[i]) != 0:
            index = torch.multinomial(tensor_tril[i], 1).item()         #This ensures we sample from the elements rather than a coin toss
            if index == i:
                action_tensor_tril[i, i] = 2
            else:
                action_tensor_tril[i] = torch.ones(action_tensor_tril[i].size(0))
                action_tensor_tril[i][tensor_tril[i] == 0] = 0
                action_tensor_tril[i, i] = 0

    #Mask with a logical and to ensure that the action rules are followed
    mask = torch.logical_and(action_tensor_triu, action_tensor_tril.t())
    action_tensor = torch.mul(mask, action_tensor_triu)
    action_tensor = action_tensor + action_tensor.t() 
    
    #To get rid of double added (2 + 2 = 4)
    action_tensor[rng, rng] = action_tensor[rng, rng]/2

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
    

def softmax_with_zero(tensor):
    softmax_result = torch.nn.functional.softmax(tensor, dim = 1)
    softmax_result[tensor == 0] = 0
    return softmax_result