import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, num_actions, num_features = 32, dropout_rate = 0.6, hidden_layers = [256, 128, 64]):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.dropout_rate = dropout_rate
        self.fc_layers_list = [(torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]), torch.nn.ReLU(), torch.nn.Dropout(dropout_rate)) for i in range(len(hidden_layers)-1)]
        self.fc_layers = [item for t in self.fc_layers_list for item in t]

        self.fc_in = nn.Linear(num_features*3*3, hidden_layers[0])
        self.fc = nn.Sequential(*self.fc_layers)
        self.fc_out = nn.Linear(hidden_layers[-1], num_actions)


    def forward(self, x, batches):
        #Consider only the entanglement layer
        if batches:
            x = x[:, 0, :, :]
        else:
            x = x[0, :, :]

        #Convert to tensor
        if torch.is_tensor(x):
            pass
        else:
            x = torch.tensor(x)

            
        x = x.to(torch.float32).unsqueeze(0)
        if batches:
            x = x.reshape((x.shape[1], x.shape[0], x.shape[2], x.shape[3]))
        else:
            x = x.to(torch.float32).unsqueeze(0)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_in(x))

        x = self.fc(x)
        x = self.fc_out(x)
        return x
    


class AgentSP:
    def __init__(self, num_features, num_actions):
        self.num_features = num_features
        self.num_actions = num_actions
        self.qnn = DQN(num_actions, num_features)
        self.target_nn = DQN(num_actions, num_features)
        self.target_nn.load_state_dict(self.qnn.state_dict())