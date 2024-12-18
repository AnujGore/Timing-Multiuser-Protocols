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
        self.repeater_decision_tensor = torch.nn.Parameter(torch.rand(n**2))
        self.cn_decision_tensor = torch.nn.Parameter(torch.rand(1))

        a = torch.rand(n**2, requires_grad=True)
        self.swap_parameter = torch.diag(a)

        self.rng = torch.arange(0, n**2)

        self.softmaxer_col = torch.nn.Softmax(dim = 1)
        self.softmaxer_row = torch.nn.Softmax(dim = 0)
        self.softmaxer = torch.nn.Softmax()

        self.cn_loc = int(np.floor(n**2/2))
        self.user_loc = np.array([0, n-1, n**2-1, n**2 - n - 1])


    def predict_action(self, state):
        edge_actions = torch.abs(torch.mul(self.edge_action_model(state), self.lattice_adj)) #To get rid of edge interactions which do not exist
        node_actions = torch.diag(self.node_action_model(state)[self.rng, self.rng]) #To get rid of edge interactions which do not exist

        action_mat = edge_actions + node_actions
        action_mat = correct_action(action_mat, self.n)
        
        for node in self.rng:
            if node in self.user_loc:
                action_mat[node, torch.argmax(action_mat[node])] = 1.
            elif node == self.cn_loc:
                if self.cn_decision_tensor < 0.2:
                    action_mat[node, :] = 0
                    action_mat[:, node] = 0
                elif 0.2 <= self.cn_decision_tensor < 0.4:
                    action_mat[node, torch.argmax(action_mat[node])] = 1. 


def correct_action(action, n):
    cn_loc = int(np.floor(n**2/2))
    user_loc = np.array([0, n-1, n**2-1, n**2 - n - 1])

    action[cn_loc, cn_loc] = 0
    action[user_loc, user_loc] = 0

    return action



n = 3
test_state = torch.zeros(size=(n**2, n**2))
myAgent = Agent(n, hidden_dims = [64, 32, 16])
myAgent.predict_action(test_state)