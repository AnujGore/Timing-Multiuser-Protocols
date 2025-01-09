import torch

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


def reward_function(done: bool, truncated: bool, actions_taken):
    if done:
        return 100 - actions_taken
    if truncated:
        return -100
    else:
        return -1
