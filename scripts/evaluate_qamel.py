import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qamel.environment import RepeaterChain
from qamel.utils import check_if_bad_state, check_if_final_state, reward_shape, get_state_id

import numpy as np
import torch
import argparse
from rich.progress import track

def evaluate_q_table(env_vars, **kwargs):

    n = env_vars["n"]
    pgen = env_vars["pgen"]
    pswap = env_vars["pswap"]

    eval_episodes = kwargs["eval_episodes"]

    torch_device = kwargs["torch_device"]

    eval_repeater_chain = RepeaterChain(n, pgen, pswap, torch_device)

    eval_actions_taken = np.zeros(eval_episodes)
    eval_reward_per_episode = np.zeros(eval_episodes)

    eval_ent_count = np.zeros(eval_episodes)
    eval_swap_count = np.zeros(eval_episodes)

    q_table = np.loadtxt(f"qamel/q_table_storage/{n}_nodes.txt")

    all_states = np.load(f"qamel/outputs/logs/states/{n}_nodes.npy")
    all_states = torch.Tensor(all_states).to(torch_device)

    all_actions = np.load(f"qamel/outputs/logs/actions/{n}_nodes.npy")
    all_actions = torch.Tensor(all_actions).to(torch_device)

    if torch.is_tensor(q_table):
        q_table = q_table.cpu().numpy()

    for eval_episode in track(range(eval_episodes), description=f"Evaluating for node {env_vars["n"]}"):
        cumulative_reward = 0
        done = False
        current_state = eval_repeater_chain.reset()

        while not done:
            action_id = np.argmax(q_table[get_state_id(all_states, current_state[0])])
            new_state = eval_repeater_chain.step(current_state, all_actions[action_id])

            bad_state = check_if_bad_state(new_state)
            final_state = check_if_final_state(new_state)

            reward = reward_shape(new_state, final_state, bad_state)
            cumulative_reward += reward

            current_state = new_state 

            if (check_if_final_state(current_state) or check_if_bad_state(current_state)):
                eval_reward_per_episode[eval_episode] = cumulative_reward
                eval_ent_count[eval_episode] = torch.amax(current_state[1]).item()
                eval_swap_count[eval_episode] = torch.amax(current_state[2]).item()
                done = True

        
    return eval_actions_taken, eval_reward_per_episode, eval_ent_count, eval_swap_count


def run_evaluation(n, pgen, pswap, eval_episodes):
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_vars = {
        "n": n,
        "pgen": pgen,
        "pswap": pswap,
        "eval_episodes": eval_episodes
    }

    _, _, ent_counts, swap_counts = evaluate_q_table(env_vars, eval_episodes=eval_episodes, torch_device=torch_device)

    # Save results
    np.savetxt(f"qamel/outputs/results/ent_counts/{n}_nodes.txt", ent_counts)
    np.savetxt(f"qamel/outputs/results/swap_counts/{n}_nodes.txt", swap_counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--pgen", type=float, required=True)
    parser.add_argument("--pswap", type=float, required=True)
    parser.add_argument("--eval_episodes", type=int, required=True)

    args = parser.parse_args()

    # Call function with parsed arguments
    run_evaluation(args.n, args.pgen, args.pswap, args.eval_episodes)
