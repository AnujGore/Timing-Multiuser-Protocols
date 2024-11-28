import torch
import numpy as np
from utils import find_tensor

import json
hyperparameter_dict = json.load(open("MP/Scripts/config_files/model_config.json"))

class Agent:
    def __init__(self, n, lifetime, all_actions_array, all_states_array, exploration_rate = 1., learning_rate = 0.1, discount_factor = 0.95):
        self.n = n
        self.lifetime = lifetime

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.all_states_tensor = torch.Tensor(all_states_array).to(self.device)
        self.all_actions_tensor = torch.Tensor(all_actions_array).to(self.device)

        self.q_table = torch.zeros(
            (self.all_states_tensor.shape[0], self.all_actions_tensor.shape[0]),
            dtype=torch.float32,
            device=self.device
            ).to(device = self.device)

        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate_decay = 1.

    def act(self, state, learn: bool):
        if learn and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(0, self.all_actions_tensor.shape[0])
        else:
            state_idx = find_tensor(self.all_states_tensor, state)
            action_idx = torch.argmax(self.q_table[state_idx]).item()

            # self.exploration_rate *= self.exploration_rate_decay
            # self.exploration_rate = max(0.1, self.exploration_rate)

        return action_idx
    
    def update(self, state, new_state, reward, action_idx, truncation):
        if truncation:
            state_idx = find_tensor(self.all_states_tensor, state)
            self.q_table[state_idx, action_idx] += self.learning_rate*[reward - self.q_table[state_idx, action_idx]][0].item()
        else:
            state_idx = find_tensor(self.all_states_tensor, state)
            new_state_idx = find_tensor(self.all_states_tensor, new_state)
            self.q_table[state_idx, action_idx] += self.learning_rate*[reward + 
                                                                       self.discount_factor*max(self.q_table[new_state_idx]) -
                                                                       self.q_table[state_idx, action_idx]][0].item()