import wandb
import pprint

sweep_config = {
    "name": "hyperparameter_swap",
    "method": "random",
    "metric": {"name": "reward", "goal": "maximize"},
    "parameters": {
        "hidden_layers": {
            "values": [[512, 256, 128],
                       [256, 128, 64],
                       [128, 64, 32]]
                       },

            "num_features": {
                "distribution": "q_log_uniform_values",
                "q": 8,
                "min": 4,
                "max": 64
                },
            
            "learning_rate": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.1
                },

            "dropout_rate": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.6
                },
            
            "epsilon": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 1.0
                },
            
            "gamma": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.99
                },

            "training_freq": {
                "distribution": "uniform",
                "min": 1,
                "max": 10
                },

            "copy_to_target":{
                "distribution": "uniform",
                "min": 10,
                "max": 50
                }
            }
    }

sweep_id = wandb.sweep(sweep_config, project="sp-rl-sweep")

from utils.model_utils import evaluate, ExperienceReplay
from objects.agent import AgentSP, DQN
from objects.environment import GridTopologyEnv
from utils.general_utils import generate_all_actions
import torch
import json
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_parameters = json.load(open("MP/SP/Scripts/config_files/env_config.json"))
simulation_parameters = json.load(open("MP/SP/Scripts/config_files/simulation_config.json"))

pgen = env_parameters["pgen"]
pswap = env_parameters["pswap"]
n = env_parameters["n"]

from pathlib import Path

all_actions_file = Path(f"MP/SP/Outputs/action_files/all_actions_{n}_nodes.npy")
if all_actions_file.is_file():
    all_actions = np.load(all_actions_file, allow_pickle=True)
else:
    all_actions = generate_all_actions(n)
    np.save(all_actions_file, all_actions)


class Args:
    total_timesteps: int = simulation_parameters["total_timesteps"]
    buffer_size: int = simulation_parameters["buffer_size"]
    copy_to_target: int =  simulation_parameters["target_network_update_freq"]
    batch_size: int = simulation_parameters["batch_size"]
    min_training_steps: int = simulation_parameters["training_starts"]
    training_freq: int = simulation_parameters["training_frequency"]
    eval_freq: int = simulation_parameters["evaluate_after"]
    eval_episodes: int = simulation_parameters["eval_episodes"]
    max_actions_taken: int = simulation_parameters["max_actions_taken"]

this_args = Args

def sweep_params(config = None):
    
    with wandb.init(
        dir = "/rdata/ong/Anuj/Timing-Multiuser-Protocols/wandb",
        config = config
        ):
        config = wandb.config
        args = this_args
        rb = ExperienceReplay(args.buffer_size)
        total_rewards = []
        myAgent = AgentSP(config.num_features, len(all_actions), config.dropout_rate, config.hidden_layers)
        myEnv = GridTopologyEnv(n, pgen, pswap, len(all_actions))

        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(myAgent.qnn.parameters(), lr = config.learning_rate)
        actions_taken = 0
        state, _ = myEnv.reset()

        for step in range(args.total_timesteps):
            epsilon = config.epsilon
            if np.random.rand() < epsilon:
                action_id = np.random.randint(0, len(all_actions))
            else:
                q_values = myAgent.qnn(state, batches = False)
                action_id = torch.argmax(q_values).numpy()
            
            action = all_actions[action_id]

            next_state, reward, terminated, truncated, _ = myEnv.act(action)
            actions_taken += 1

            if truncated or terminated:
                state, _ = myEnv.reset()

            if actions_taken > args.max_actions_taken:
                state, _ = myEnv.reset()
                actions_taken = 0

            rb.store_experience(state, action, reward, next_state, terminated, truncated)

            state = next_state

            if step > args.min_training_steps:
                if step % args.training_freq == 0:
                    data = rb.sample_experience(args.batch_size)
                    training_q_values = myAgent.qnn(data.state, batches = True).gather(1, data.to_action_idx(all_actions).reshape(1, -1)).flatten()
                    with torch.no_grad():
                        target_q_values = myAgent.target_nn(data.next_state, batches = True).max(dim = 1)[0]
                        training_target = data.reward.flatten() + config.gamma * (target_q_values * (1 -  data.terminated.to(dtype=torch.float32).flatten()))

                    this_loss = loss(training_target, training_q_values)
                    optimizer.zero_grad()
                    this_loss.backward()
                    optimizer.step()

                if step % args.copy_to_target == 0:
                    myAgent.target_nn.load_state_dict(myAgent.qnn.state_dict())

                if step % args.eval_freq == 0:
                    torch.save(myAgent.qnn.state_dict(), f"MP/SP/Outputs/model_paths/qNet_{n}_nodes.pth")
                    eval_rewards = np.zeros(shape = args.eval_episodes)
                    eval_actions_taken = np.zeros(shape = args.eval_episodes)
                    for e in range(args.eval_episodes):
                        actions_taken, reward = evaluate(
                            model_path = f"MP/SP/Outputs/model_paths/qNet_{n}_nodes.pth",
                            env = myEnv,
                            model = DQN,
                            args = args,
                            config = config,
                            device = device,
                            all_actions= all_actions
                        )
                        eval_actions_taken[e] = actions_taken
                        eval_rewards[e] = reward

                    total_rewards.append(np.mean(eval_rewards))
            
        wandb.log({"reward": total_rewards[-1]})


wandb.agent(sweep_id, sweep_params, count = 200)