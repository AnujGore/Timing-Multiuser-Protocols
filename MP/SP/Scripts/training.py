import numpy as np
import tyro
import tyro
import torch

from utils.model_utils import linear_schedule, evaluate, ExperienceReplay
from utils.SP_path_utils import generate_all_actions

from objects.agent import DQN

from utils.model_utils import linear_schedule, evaluate, ExperienceReplay
from utils.SP_path_utils import generate_all_actions

from objects.agent import DQN

import json
env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json")) 
env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json")) 
simulation_parameters = json.load(open("MP/SP/Scripts/config_files/simulation_config.json"))
model_hyperparameters = json.load(open("MP/SP/Scripts/config_files/model_config.json"))

from rich.progress import Progress

n = env_parameters["n"]
from rich.progress import track

n = env_parameters["n"]

from pathlib import Path

all_actions_file = Path(f"MP/SP/Outputs/action_files/all_actions_3_nodes.npy")
if all_actions_file.is_file():
    all_actions = np.load(all_actions_file, allow_pickle=True)
else:
    all_actions = generate_all_actions(n)
    np.save(all_actions_file, all_actions)



class Args:
    total_timesteps: int = simulation_parameters["total_timesteps"]
    buffer_size: int = simulation_parameters["buffer_size"]
    gamma: float =  simulation_parameters["gamma"] 
    copy_to_target: int =  simulation_parameters["target_network_update_freq"]
    batch_size: int = simulation_parameters["batch_size"]
    start_epsilon: float = simulation_parameters["start_e"]
    end_epsilon: float = simulation_parameters["end_e"]
    epsilon_fraction: float = simulation_parameters["e_fraction"]
    min_training_steps: int = simulation_parameters["training_starts"]
    training_freq: int = simulation_parameters["training_frequency"]
    eval_freq: int = simulation_parameters["evaluate_after"]
    eval_episodes: int = simulation_parameters["eval_episodes"]
    max_actions_taken: int = simulation_parameters["max_actions_taken"]

    hidden_layers: list = model_hyperparameters["hidden_layers"]
    dropout_rate: float = model_hyperparameters["dropout_rate"]
    learning_rate: float = model_hyperparameters["learning_rate"]
    num_features: float = model_hyperparameters["num_features"]



def train_agent(env, agent, Args):
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    rb = ExperienceReplay(args.buffer_size)

    total_rewards = []
    total_rewards_std = []
    total_actions_needed = []
    total_actions_needed_std = []
    
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(agent.qnn.parameters(), lr = args.learning_rate)
    actions_taken = 0
    state, _ = env.reset()

    with Progress() as progress:
        task1 = progress.add_task("[cyan]Running timesteps...", total = args.total_timesteps)
        for step in range(args.total_timesteps):
            progress.update(task1, advance=1)
            epsilon = linear_schedule(args.start_epsilon, args.end_epsilon, args.epsilon_fraction*args.total_timesteps, step)
            if np.random.rand() < epsilon:
                action_id = np.random.randint(0, len(all_actions))
            else:
                q_values = agent.qnn(state, batches = False)
                action_id = torch.argmax(q_values).numpy()
            
            action = all_actions[action_id]

            next_state, reward, terminated, truncated, _ = env.act(action)
            actions_taken += 1

            if truncated or terminated:
                state, _ = env.reset()

            if actions_taken > args.max_actions_taken:
                state, _ = env.reset()
                actions_taken = 0

            rb.store_experience(state, action, reward, next_state, terminated, truncated)

            state = next_state

            if step > args.min_training_steps:
                if step % args.training_freq == 0:
                    data = rb.sample_experience(args.batch_size)
                    training_q_values = agent.qnn(data.state, batches = True).gather(1, data.to_action_idx(all_actions).reshape(1, -1)).flatten()
                    with torch.no_grad():
                        target_q_values = agent.target_nn(data.next_state, batches = True).max(dim = 1)[0]
                        training_target = data.reward.flatten() + args.gamma * (target_q_values * (1 -  data.terminated.to(dtype=torch.float32).flatten()))

                    this_loss = loss(training_target, training_q_values)
                    optimizer.zero_grad()
                    this_loss.backward()
                    optimizer.step()

                if step % args.copy_to_target == 0:
                    agent.target_nn.load_state_dict(agent.qnn.state_dict())

            if step % args.eval_freq == 0:
                torch.save(agent.qnn.state_dict(), f"MP/SP/Outputs/model_paths/qNet_{n}_nodes.pth")
                eval_rewards = np.zeros(shape = args.eval_episodes)
                eval_actions_taken = np.zeros(shape = args.eval_episodes)
                for e in range(args.eval_episodes):
                    actions_taken, reward = evaluate(
                        model_path = f"MP/SP/Outputs/model_paths/qNet_{n}_nodes.pth",
                        env = env,
                        model = DQN,
                        args = args,
                        device = device,
                        epsilon = epsilon, 
                        all_actions= all_actions
                    )
                    eval_actions_taken[e] = actions_taken
                    eval_rewards[e] = reward

                total_rewards.append(np.mean(eval_rewards))
                total_rewards_std.append(np.std(eval_rewards))

                total_actions_needed.append(np.mean(eval_actions_taken))
                total_actions_needed_std.append(np.std(eval_actions_taken))

    return total_rewards, total_rewards_std, total_actions_needed, total_actions_needed_std

from objects.environment import GridTopologyEnv
from objects.agent import AgentSP

pgen = env_parameters["pgen"]
pswap = env_parameters["pswap"]
num_features = model_hyperparameters["num_features"]

myEnv = GridTopologyEnv(n, pgen, pswap, len(all_actions))

myAgent = AgentSP(num_features, len(all_actions))

myArgs = Args

r, r_std, a, a_std = train_agent(myEnv, myAgent, myArgs)

np.savetxt(f"MP/SP/Outputs/{n}_nodes_rewards.txt", r)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_rewards_std.txt", r_std)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_actions.txt", a)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_actions_std.txt", a_std)