import numpy as np

def test(env, agent, num_episodes, threshold_action):
    num_actions_arr = np.zeros(shape = num_episodes)
    reward_arr = np.zeros(shape = num_episodes)

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False; truncated = False; elapsed = False
        actions_taken = 0; cumulative_reward = 0

        while not (done or truncated or elapsed):
            action_idx = agent.act(state, learn = False)
            new_state, reward, done, truncated, info = env.step(action_idx)
            actions_taken += 1
            cumulative_reward+= reward
            state = new_state
            if actions_taken >= threshold_action:
                elapsed = True

        num_actions_arr[episode] = actions_taken
        reward_arr[episode] = cumulative_reward
    return num_actions_arr.mean()