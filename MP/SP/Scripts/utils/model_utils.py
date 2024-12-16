import numpy as np
import torch
from collections import deque
import torch

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def find_coo_matrix(target, total):
    for i, m in enumerate(total):
        if (np.array_equal(target.row, m.row) and
            np.array_equal(target.col, m.col) and
            np.array_equal(target.data, m.data)):
            return i
        

def evaluate(model_path, env, model, model_config, device, all_actions, max_actions_taken):

    eval_model = model(env.n, len(all_actions), model_config.num_features, dropout_rate = model_config.dropout_rate, hidden_layers = model_config.hidden_layers)
    eval_model.load_state_dict(torch.load(model_path, weights_only = False))
    eval_model.to(device = device)
    eval_model.eval()

    state, _ = env.reset()

    done = False; truncated = False; elapsed = False
    actions_taken = 0
    while not(done or truncated or elapsed):
        q_values = eval_model(torch.tensor(state[1]).unsqueeze(0).unsqueeze(0).to(torch.float32))
        action_id = torch.argmax(q_values).numpy()
        
        action = all_actions[action_id]
        next_state, _, done, truncated, _ = env.act(action)
        reward = reward_function(done, truncated, actions_taken)

        actions_taken +=1

        if actions_taken >= max_actions_taken:
            elapsed = True

        state=  next_state

    return actions_taken, reward


class SampleData:
    def __init__(self, batch_data):
        self.state = np.array([row[0] for row in batch_data])
        self.action = np.array([row[1] for row in batch_data])
        self.reward = np.array([row[2] for row in batch_data])
        self.next_state = np.array([row[3] for row in batch_data])
        self.terminated = np.array([row[4] for row in batch_data])
        self.truncated = np.array([row[5] for row in batch_data])

        self.state = torch.tensor(self.state)
        self.action = torch.tensor(self.action)
        self.reward = torch.FloatTensor(self.reward)
        self.next_state = torch.tensor(self.next_state)
        self.terminated = torch.BoolTensor(self.terminated)
        self.truncated = torch.BoolTensor(self.truncated)

    def to_action_idx(self, all_actions):
        idx = []
        all_actions = torch.Tensor(all_actions)
        for action in self.action:
            for i, possible_action in enumerate(all_actions):
                if (torch.equal(action, possible_action)):
                    idx.append(i)

        return torch.tensor(idx)



class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_experience(self, state, action, reward, new_state, terminated, truncated):
        experience = (state, action, reward, new_state, terminated, truncated)
        self.buffer.append(experience)

    def sample_experience(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        return SampleData(batch)

    def __len__(self):
        return len(self.buffer)
     

def reward_function(done: bool, truncated: bool, actions_taken):
    if done:
        return 100 - actions_taken
    if truncated:
        return -100
    else:
        return -1