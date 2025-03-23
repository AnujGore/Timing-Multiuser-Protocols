import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.agent import Agent
from qamel.environment import RepeaterChain
from qamel.utils import check_if_bad_state, check_if_final_state, reward_shape

import numpy as np
import torch
import argparse
import time
from rich.progress import track

def train_q_agent(env_vars, hyperparameter_configs, **kwargs):

    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    training_episodes = kwargs["training_episodes"]
    max_actions = kwargs["max_actions"]

    alpha = hyperparameter_configs.alpha
    gamma = hyperparameter_configs.gamma
    epsilon = hyperparameter_configs.epsilon

    torch_device = kwargs["torch_device"]

    this_QAlgorithm = Agent(n, alpha = alpha, gamma = gamma, device = torch_device)
    this_RepeaterChain = RepeaterChain(n, pgen, pswap, torch_device)

    cumulative_reward_per_episode = np.zeros(training_episodes)

    for episode in track(range(training_episodes), description="Training q-learning agent..."):
        cumulative_reward = 0; step = 0
        done = False
        current_state = this_RepeaterChain.reset()

        while not done:
            action = this_QAlgorithm.predict_action(current_state[0], epsilon)
            new_state = this_RepeaterChain.step(current_state, action)

            bad_state = check_if_bad_state(new_state)
            final_state = check_if_final_state(new_state)
            
            reward = reward_shape(new_state, final_state, bad_state)

            this_QAlgorithm.update_q_table(current_state[0], action, reward, new_state[0], bad_state)

            current_state = new_state

            step += 1
            cumulative_reward += reward

            if (step >= max_actions or check_if_final_state(current_state) or check_if_bad_state(current_state)):
                done = True
                cumulative_reward_per_episode[episode] = cumulative_reward

    return this_QAlgorithm.q_table, cumulative_reward_per_episode


class hyperparameters:
    alpha = 0.0755410045582013
    epsilon = 0.10210439965486162
    gamma = 0.027548998387277125
        
max_actions = 100
training_episodes = 10000


if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", torch_device)

    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type = int)
    parser.add_argument("--pgen", type = float)
    parser.add_argument("--pswap", type = float)

    env_vars_class = parser.parse_args()
    env_vars = env_vars_class.__dict__

    this_hyperparameters = hyperparameters

    if os.path.exists(f"qamel/q_table_storage/{env_vars["n"]}_nodes.txt"):
        print(f"An agent for {env_vars["n"]} nodes has been trained.")

    else:
        start_time = time.time()
        q_table, training_rewards = train_q_agent(env_vars, this_hyperparameters, max_actions = max_actions, training_episodes = training_episodes, torch_device = torch_device)

        np.savetxt(f"qamel/q_table_storage/{env_vars["n"]}_nodes.txt", q_table.cpu().numpy())

        end_time = time.time()
        print(f"Took {end_time - start_time} seconds to train.")