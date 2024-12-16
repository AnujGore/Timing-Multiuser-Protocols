import numpy as np
import torch

from utils.model_utils import linear_schedule, evaluate, ExperienceReplay, reward_function
from utils.general_utils import generate_all_actions

from objects.agent import DQN

import wandb
# wandb.login(key = "763741f56851a90ea63e9cb916910a6d9eecc66a")

import json
env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json")) 
simulation_parameters = json.load(open("MP/SP/Scripts/config_files/simulation_config.json"))
model_hyperparameters = json.load(open("MP/SP/Scripts/config_files/model_config.json"))

from rich.progress import Progress

class model_config:
    num_features = model_hyperparameters["num_features"]
    dropout_rate = model_hyperparameters["dropout_rate"]
    hidden_layers = model_hyperparameters["hidden_layers"]
    learning_rate = model_hyperparameters["learning_rate"]

class rl_config:
    buffer_size = simulation_parameters["buffer_size"]
    total_timesteps = simulation_parameters["total_timesteps"]
    min_training_steps = simulation_parameters["training_starts"]
    training_freq = simulation_parameters["training_frequency"]
    batch_size = simulation_parameters["batch_size"]
    gamma = simulation_parameters["gamma"]
    copy_to_target = simulation_parameters["copy_to_target"]
    eval_episodes = simulation_parameters["eval_episodes"]
    eval_freq = simulation_parameters["eval_freq"]
    eval_start = simulation_parameters["evaluate_after"]
    epsilon_decay = simulation_parameters["eps_decay"]

def train_agent(env, agent, this_model_config, this_rl_config, max_actions_taken):
    wandb.init(
        project="sp-dqn",
        dir = "/rdata/ong/Anuj/Timing-Multiuser-Protocols/MP/SP/"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    rb = ExperienceReplay(this_rl_config.buffer_size)

    total_rewards = []
    total_rewards_std = []
    total_actions_needed = []
    total_actions_needed_std = []
    total_rewards_mean = []
    total_actions_needed_mean = []
    
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(agent.qnn.parameters(), lr = this_model_config.learning_rate)
    actions_taken = 0
    state, _ = env.reset()

    with Progress() as progress:
        task1 = progress.add_task("[cyan]Running timesteps...", total = this_rl_config.total_timesteps)
        for step in range(this_rl_config.total_timesteps):
            progress.update(task1, advance=1)
            epsilon = 0.05 + (0.99 - 0.05)*torch.math.exp(-1*(step/this_rl_config.epsilon_decay))
            if np.random.rand() < epsilon:
                action_id = np.random.randint(0, len(all_actions))
            else:
                q_values = agent.qnn(torch.tensor(state[1]).unsqueeze(0).unsqueeze(0).to(torch.float32))
                action_id = torch.argmax(q_values).numpy()
            
            action = all_actions[action_id]

            actions_taken += 1
            next_state, _, terminated, truncated, _ = env.act(action)
            reward = reward_function(terminated, truncated, actions_taken)

            if truncated or terminated:
                state, _ = env.reset()

            if actions_taken >= max_actions_taken:
                state, _ = env.reset()
                actions_taken = 0

            rb.store_experience(state, action, reward, next_state, terminated, truncated)

            state = next_state

            if step == this_rl_config.min_training_steps+1: print(f"Training started after {this_rl_config.min_training_steps} episodes.")
            if step > this_rl_config.min_training_steps:
                if step % this_rl_config.training_freq == 0:
                    data = rb.sample_experience(this_rl_config.batch_size)
                    training_q_values = agent.qnn(data.state[:, 1, :, :].unsqueeze(1).to(torch.float32)).gather(1, data.to_action_idx(all_actions).reshape(1, -1)).flatten()
                    with torch.no_grad():
                        target_q_values = agent.target_nn(data.next_state[:, 1, :, :].unsqueeze(1).to(torch.float32)).max(dim = 1)[0]
                        training_target = data.reward.flatten() + (this_rl_config.gamma * target_q_values)

                    this_loss = loss(training_target, training_q_values)
                    wandb.log({"Target Rewards": torch.mean(target_q_values)})
                    wandb.log({"Q-Network Rewards": torch.mean(training_q_values)})
                    wandb.log({"Loss": this_loss})
                    optimizer.zero_grad()
                    this_loss.backward()
                    optimizer.step()

                if step % this_rl_config.copy_to_target == 0:
                    agent.target_nn.load_state_dict(agent.qnn.state_dict())

                if step > this_rl_config.eval_start and step % this_rl_config.eval_freq == 0:
                    task2 = progress.add_task("[green]Evaluating...", total = this_rl_config.eval_episodes)
                    torch.save(agent.qnn.state_dict(), f"MP/SP/Outputs/model_paths/qNet_{n}_nodes_{"entangled" if env.init_ent else "not_entangled"}.pth")
                    eval_rewards = np.zeros(shape = this_rl_config.eval_episodes)
                    eval_actions_taken = np.zeros(shape = this_rl_config.eval_episodes)
                    for e in range(this_rl_config.eval_episodes):
                        actions_taken, reward = evaluate(
                            model_path = f"MP/SP/Outputs/model_paths/qNet_{n}_nodes_{"entangled" if env.init_ent else "not_entangled"}.pth",
                            env = env,
                            model = DQN,
                            model_config = this_model_config, 
                            device = device, 
                            all_actions= all_actions,
                            max_actions_taken = max_actions_taken
                        )
                        eval_actions_taken[e] = actions_taken
                        eval_rewards[e] = reward
                        progress.update(task2, advance = 1)
                    
                    wandb.log({"Mean Rewards": np.mean(eval_rewards),
                               "Mean Actions": np.mean(eval_actions_taken)})
                    wandb.log({"Total Rewards": np.sum(eval_rewards)})

                    total_rewards.append(eval_rewards)
                    total_actions_needed.append(eval_actions_taken)

                    total_rewards_mean.append(np.mean(eval_rewards))
                    total_rewards_std.append(np.std(eval_rewards))

                    total_actions_needed_mean.append(np.mean(eval_actions_taken))
                    total_actions_needed_std.append(np.std(eval_actions_taken))
                    progress.remove_task(task2)

    return total_rewards_mean, total_rewards_std, total_actions_needed_mean, total_actions_needed_std, total_rewards, total_actions_needed

from objects.environment import GridTopologyEnv
from objects.agent import AgentSP

pgen = env_parameters["pgen"]
pswap = env_parameters["pswap"]
num_features = model_hyperparameters["num_features"]

myEnv = GridTopologyEnv(n, pgen, pswap, len(all_actions))

myAgent = AgentSP(n, model_config, len(all_actions))

r, r_std, a, a_std = train_agent(myEnv, myAgent, model_config, rl_config, max_actions)

np.savetxt(f"MP/SP/Outputs/{n}_nodes_rewards.txt", r)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_rewards_std.txt", r_std)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_actions.txt", a)
np.savetxt(f"MP/SP/Outputs/{n}_nodes_actions_std.txt", a_std)