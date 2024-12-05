import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, num_actions, num_features = 32):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(num_features*3*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)


    def forward(self, x):
        #Consider only the entanglement layer
        x = x[:, 0, :, :]

        #Convert to tensor
        if torch.is_tensor(x):
            pass
        else:
            x = torch.tensor(x)

            
        x = x.to(torch.float32).unsqueeze(0)
        x = x.reshape((x.shape[1], x.shape[0], x.shape[2], x.shape[3]))

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
    


class AgentSP:
    def __init__(self, num_features, num_actions, epsilon):
        self.num_features = num_features
        self.num_actions = num_actions
        self.qnn = DQN(num_actions, num_features)
        self.target_nn = DQN(num_actions, num_features)
        self.target_nn.load_state_dict(self.qnn.state_dict())
        self.exploration_rate = epsilon

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = torch.argmax(self.qnn(state)).numpy()
        else:
            action_idx = np.random.randint(0, self.num_actions)

        return action_idx