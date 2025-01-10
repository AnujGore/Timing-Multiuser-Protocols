import torch
import numpy as np

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


def reward_function(next_state, previous_state, path_matrix, truncated, done, actions_taken):
    reward = 0
    new_edges = torch.Tensor(next_state[0] - previous_state[0])

    n = np.sqrt(len(next_state[0]))
    user_loc = np.array([0, n-1, n**2-1, n**2 - n], dtype = int)
    cn_loc = int(n*np.floor(n/2) + np.floor(n/2))

    #Check if new entanglement has been generated
    new_edges_ent = new_edges.clone()
    new_edges_ent = torch.mul(path_matrix, new_edges_ent)

    if (new_edges_ent == 1.).any():
        reward += 5

    #Check if swap has been performed
    new_edges_swap = new_edges.clone()
    mask = torch.logical_not(path_matrix).to(torch.int)
    new_edges_swap = torch.mul(mask, new_edges_swap)

    if (next_state[0][cn_loc, user_loc] == 1.).any():
        reward += 50

    if (new_edges_swap == 1.).any():
        reward += 10
    if done:
        reward += 100 - actions_taken
    if truncated:
        return -100

    return reward


