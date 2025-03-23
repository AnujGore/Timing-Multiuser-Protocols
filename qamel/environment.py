import torch

class RepeaterChain:
    def __init__(self, n, pgen, pswap, device):
        self.n = n
        self.pgen = pgen
        self.pswap = pswap
        self.device = device

    def reset(self):
        self.chain_state = torch.zeros(size = (3, self.n, self.n)).to(self.device)
        return self.chain_state

    def step(self, this_state, action):
        if action.dim() == 3: action = action[0]

        this_state_copy = this_state.clone()

        edge_idx = torch.triu_indices(self.n, self.n, 1)
        
        for i in range(edge_idx.size(1)):
            edge_action_coor = edge_idx[:, i]
            edge_action = action[edge_action_coor[0], edge_action_coor[1]]

            #Generate new links
            if edge_action == 1:
                this_state_copy[1][edge_action_coor[0], edge_action_coor[0]] += 1
                this_state_copy[1][edge_action_coor[1], edge_action_coor[1]] += 1

                this_state_copy[0][edge_action_coor[0], edge_action_coor[1]] = 0
                this_state_copy[0][edge_action_coor[1], edge_action_coor[0]] = 0

                if torch.rand(1) < self.pgen:
                    this_state_copy[0][edge_action_coor[0], edge_action_coor[1]] = 1
                    this_state_copy[0][edge_action_coor[1], edge_action_coor[0]] = 1

        nodes_to_swap = torch.diagonal(action, 0)[1:-1]

        #Swap edges
        node_idx_to_swap = (nodes_to_swap == 1).nonzero(as_tuple=True)[0]
        for node_idx in node_idx_to_swap:
            connected_nodes = (this_state_copy[0][node_idx+1].to(int) > 0).nonzero(as_tuple=True)[0]
            
            if len(connected_nodes) > 2:
                #Truncate the episode
                return -100, this_state_copy
            elif len(connected_nodes) == 2:
                this_state_copy[2][(node_idx+1).to(int), (node_idx+1).to(int)] += 1

                if torch.rand(1) < self.pswap:
                    this_state_copy[0][connected_nodes[0], connected_nodes[1]] = max(
                        this_state_copy[0][connected_nodes[0], (node_idx+1).to(int)],
                        this_state_copy[0][connected_nodes[1], (node_idx+1).to(int)]
                    )

                    this_state_copy[0][connected_nodes[1], connected_nodes[0]] = max(
                        this_state_copy[0][connected_nodes[0], (node_idx+1).to(int)],
                        this_state_copy[0][connected_nodes[1], (node_idx+1).to(int)]
                    )
                
                this_state_copy[0][connected_nodes[0], (node_idx+1).to(int)] = 0
                this_state_copy[0][connected_nodes[1], (node_idx+1).to(int)] = 0
                this_state_copy[0][(node_idx+1).to(int), connected_nodes[0]] = 0
                this_state_copy[0][(node_idx+1).to(int), connected_nodes[1]] = 0

        return this_state_copy