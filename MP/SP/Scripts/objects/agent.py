import networkx as nx
import torch
from torch.nn import init
import math
import numpy as np


class DualWeightsNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.stack = torch.nn.Sequential(
            DualLinear(input_dim, hidden_dims[0]),
            torch.nn.ReLU(),
            DualLinear(hidden_dims[0], hidden_dims[1]),
            torch.nn.ReLU(),
            DualLinear(hidden_dims[1], hidden_dims[2]),
            torch.nn.ReLU(),
            DualLinear(hidden_dims[2], output_dim),
        )

    def forward(self, x):
        action_mat = self.stack(x)
        return action_mat


class DualLinear(torch.nn.Module):
    r"""
    This class creates the matrix operation as:
         X' = A X A' + B

    where X' is then the input to the activation function.

    Args:
     - in_dimensions (`tuple`) =  Dimensions of input matrix
     - out_dimensions (`tuple`) = Dimensions of output matrix

    """

    def __init__(self, in_dimensions, out_dimensions, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_dimensions 
        self.out_features = out_dimensions
        self.weight_front = torch.nn.parameter.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs), requires_grad = True)
        self.weight_back = torch.nn.parameter.Parameter(torch.empty((self.in_features, self.out_features), **factory_kwargs), requires_grad = True)
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.empty((out_dimensions, out_dimensions), **factory_kwargs), requires_grad = True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight_back, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_front, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1. / self.in_features
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        Ax = torch.mm(self.weight_front, input)
        return torch.mm(Ax, self.weight_back) + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
   

class Agent:
    def __init__(self, n, **kwargs):
        self.n = n
        this_lattice = nx.grid_graph((n, n))
        self.lattice_adj = torch.tensor(nx.adjacency_matrix(this_lattice).todense())

        self.edge_action_model = DualWeightsNet(n**2, kwargs["hidden_dims"], n**2)
        self.node_action_model = DualWeightsNet(n**2, kwargs["hidden_dims"], n**2)

        self.rng = torch.arange(0, n**2)

        self.softmaxer = torch.nn.Softmax(dim = 0)

        self.cn_loc = int(np.floor(n**2/2))
        self.user_loc = np.array([0, n-1, n**2-1, n**2 - n - 1])


    def predict_action(self, state):
        edge_action_tensor = torch.abs(torch.mul(self.edge_action_model(state), self.lattice_adj)) #To get rid of edge interactions which do not exist
        node_action_tensor = torch.diag(self.node_action_model(state)[self.rng, self.rng]) #To get rid of edge interactions which do not exist

        edge_loc = torch.where(self.lattice_adj == 1)

        all_probabilities = torch.cat([node_action_tensor[self.rng, self.rng], edge_action_tensor[edge_loc[0], edge_loc[1]]])
        all_probabilities = self.softmaxer(all_probabilities)
        
        prob_swap = torch.diag(all_probabilities[:self.n**2])

        prob_entangle_tensor = torch.zeros(size=(self.n**2, self.n**2))
        prob_entangle_tensor[edge_loc[0], edge_loc[1]] = all_probabilities[self.n**2:]
        prob_entangle_tensor = torch.triu(prob_entangle_tensor)

        prob_action_tensor = prob_swap + prob_entangle_tensor

        action_tensor = torch.zeros_like(prob_action_tensor)

        for i in range(prob_action_tensor.size(0)):
            diagonal_value = prob_action_tensor[i, i]
            off_diagonal_values = prob_action_tensor[i, :].clone()
            off_diagonal_values[i] = float('-inf')

            max_off_diagonal_value = off_diagonal_values.max()
            max_off_diagonal_index = off_diagonal_values.argmax()

            # To repeat or not to repeat; that is the question
            if diagonal_value >= max_off_diagonal_value:
                if i == self.cn_loc:
                    pass
                elif i in self.user_loc:
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
