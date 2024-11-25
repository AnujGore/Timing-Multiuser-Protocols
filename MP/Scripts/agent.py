import torch
import numpy as np
from utils import generate_all_actions, generate_all_states, find_tensor

import json
hyperparameter_dict = json.load(open("MP/Scripts/model_config.json"))

class Agent:
    def __init__(self, n, lifetime):
        self.n = n
        self.lifetime = lifetime

        self.all_states_tensor = torch.Tensor(generate_all_states(self.n, self.lifetime))

        self.all_actions_tensor = torch.Tensor(generate_all_actions(self.n))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.q_table = torch.zeros(
            (self.all_states_tensor.shape[0], self.all_actions_tensor.shape[0]),
            dtype=torch.float32,
            device=self.device
            )


        self.exploration_rate = hyperparameter_dict["exploration_rate"]

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(0, self.all_actions_tensor.shape[0])
        else:
            state_idx = find_tensor(self.all_states_tensor, state)
            action_idx = torch.argmax(self.q_table[state_idx]).item()

        self.exploration_rate *= hyperparameter_dict["exploration_rate_decay"]
        self.exploration_rate = max(self.exploration_rate, hyperparameter_dict["exploration_rate_min"]) #clip the exploration rate

        return action_idx
    
    def update(self, state, new_state, reward, action_idx):
        state_idx = find_tensor(self.all_states_tensor, state)
        # state_idx = [key for key, arr in self.all_states_dict.items() if np.array_equal(state, arr)][0]
        new_state_idx = find_tensor(self.all_states_tensor, new_state)
        # new_state_idx = [key for key, arr in self.all_states_dict.items() if np.array_equal(new_state, arr)][0]
        self.q_table[state_idx, action_idx] += hyperparameter_dict["learning_rate"]*[reward + 
                                                                                     hyperparameter_dict["discount_factor"]*
                                                                                     max(self.q_table[new_state_idx])-
                                                                                     self.q_table[state_idx, action_idx]
                                                                                     ][0].item()