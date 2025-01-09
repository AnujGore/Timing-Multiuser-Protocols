import torch
from torch.nn import init
import math
import numpy as np
from utils.agent_utils import sample_from_prob, shortest_path

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
        Ax = torch.matmul(self.weight_front, input)
        return torch.matmul(Ax, self.weight_back) + self.bias

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
   

class Agent:
    def __init__(self, n, **kwargs):
        self.n = n
        self.path_adj = torch.Tensor(shortest_path(n)).to(torch.float32)

        self.edge_action_model = DualWeightsNet(n**2, kwargs["hidden_dims"], n**2)
        self.node_action_model = DualWeightsNet(n**2, kwargs["hidden_dims"], n**2)

        self.rng = torch.arange(0, n**2)

        self.softmaxer = torch.nn.Softmax(dim = 0)

        self.cn_loc = int(np.floor(n**2/2))
        self.user_loc = np.array([0, n-1, n**2-1, n**2 - n - 1])

        self.this_parameters = list(self.edge_action_model.parameters()) + list(self.node_action_model.parameters())


    def predict_action(self, state):
        #Check if state is a tensor
        if type(state) != 'torch.Tensor':
            state = torch.Tensor(state)

        #Unpack state if necessary
        if state.shape[0] == 4:
            state = state[1, :, :] #Training on age matrix only

        #Setting dtype
        state = state.to(torch.float32)

        edge_action_tensor = torch.abs(torch.mul(self.edge_action_model(state), self.path_adj)) #To get rid of edge interactions which do not exist
        node_action_tensor = torch.diag(self.node_action_model(state)[self.rng, self.rng]) #For swaps only

        edge_loc = torch.where(self.path_adj == 1)

        all_probabilities = torch.cat([node_action_tensor[self.rng, self.rng], edge_action_tensor[edge_loc[0], edge_loc[1]]])
        all_probabilities = self.softmaxer(all_probabilities)
        
        prob_swap = torch.diag(all_probabilities[:self.n**2])

        prob_entangle_tensor = torch.zeros(size=(self.n**2, self.n**2))
        prob_entangle_tensor[edge_loc[0], edge_loc[1]] = all_probabilities[self.n**2:]
        prob_entangle_tensor = torch.triu(prob_entangle_tensor)

        prob_action_tensor_upper = prob_swap + prob_entangle_tensor
        prob_action_tensor_lower = prob_action_tensor_upper.t()
        prob_action_tensor = prob_action_tensor_upper + prob_action_tensor_lower - torch.diag(torch.diag(prob_action_tensor_lower))

        #User and CN do not swap
        prob_action_tensor[self.user_loc, self.user_loc] = 0
        prob_action_tensor[self.cn_loc, self.cn_loc] = 0

        #Make accessible to other methods
        self.prob_action_tensor = prob_action_tensor

        return sample_from_prob(prob_action_tensor, self.cn_loc, self.user_loc)
    
    def log_prob(self, action):
        self.prob_action_tensor = torch.clamp(self.prob_action_tensor, min=1e-10, max=1 - 1e-10)
        log_prob = torch.sum(action*torch.log(self.prob_action_tensor) + (1-action)*torch.log(1-self.prob_action_tensor))
        return log_prob
    
    def entropy(self):
        entropy = -torch.sum(self.prob_action_tensor*torch.log(self.prob_action_tensor))
        return entropy
