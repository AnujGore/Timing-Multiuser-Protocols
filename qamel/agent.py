import torch
import numpy as np
import os
from .utils import generate_all_valid_actions, generate_all_valid_states, get_action_id, get_state_id

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs/logs")

class Agent:
    def __init__(self, n, **kwargs):
        """
        n = number of nodes (including end nodes)
        """

        self.n = n

        self.device = kwargs["device"]

        states_dir = os.path.join(OUTPUT_DIR, "states")
        actions_dir = os.path.join(OUTPUT_DIR, "actions")

        if not os.path.exists(f"quamel/outputs/logs/states/{n}_nodes.npy"):
            this_all_states = generate_all_valid_states(self.n).to(self.device)
            this_all_states = this_all_states.cpu().numpy()

            this_all_states_filename = os.path.join(states_dir, f"{n}_nodes.npy")
            np.save(this_all_states_filename, this_all_states)
        else:
            this_all_states = np.load(os.path.join(states_dir, f"{n}_nodes.npy"))
        
        self.all_states = torch.Tensor(this_all_states).to(self.device)

        if not os.path.exists(f"quamel/outputs/logs/actions/{n}_nodes.npy"):
            this_all_actions = generate_all_valid_actions(self.n).to(self.device)
            this_all_actions = this_all_actions.cpu().numpy()

            this_all_actions_filename = os.path.join(actions_dir, f"{n}_nodes.npy")
            np.save(this_all_actions_filename, this_all_actions)
        else:
            this_all_actions = np.load(os.path.join(actions_dir, f"{n}_nodes.npy"))

        self.all_actions = torch.Tensor(this_all_actions).to(self.device)

        self.q_table = torch.zeros(size = (self.all_states.size(0), self.all_actions.size(0))).to(self.device)

        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"]

    def predict_action(self, state, epsilon):
        #Epsilon greedy
        explore = True if torch.randn(1) < epsilon else False
        if explore:
            action_idx = torch.randint(self.all_actions.size(0), (1,))
        else:
            action_idx = torch.argmax(self.q_table[get_state_id(self.all_states, state)])

        return self.all_actions[action_idx]

    def update_q_table(self, state, action, reward, new_state, truncated):
        state_idx = get_state_id(self.all_states, state)
        action_idx = get_action_id(self.all_actions, action)
        new_state_idx = get_state_id(self.all_states, new_state)

        if truncated:
            self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.alpha * (reward + self.q_table[state_idx, action_idx])
        else:
            self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + self.alpha * (reward + self.gamma * self.q_table[new_state_idx][0].max() - self.q_table[state_idx, action_idx])


