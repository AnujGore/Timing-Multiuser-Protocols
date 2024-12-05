import json
dqn_hyperparameter_dict = json.load(open("MP/Scripts/config_files/neural_net_config.json"))


def train(env, agent, num_episodes, threshold_action):
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False; truncated = False; elapsed = False
        actions_taken = 0; cumulative_reward = 0

        while not (done or truncated or elapsed):
            action_idx = agent.act(state, learn = True)
            new_state, reward, done, truncated, info = env.step(action_idx)
            actions_taken += 1
            cumulative_reward+= reward
            agent.update(state, new_state, reward, action_idx, truncated)
            state = new_state
            if actions_taken >= threshold_action:
                elapsed = True