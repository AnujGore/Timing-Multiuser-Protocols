import numpy as np
from objects.environment import GridTopologyEnv
from utils import generate_all_actions, ExperienceReplay, find_coo_matrix
from objects.agent import AgentSP
import torch

import json
simulation_parameters = json.load(open("MP/SP/Scripts/config_files/simulation_config.json"))
model_hyperparameters = json.load(open("MP/SP/Scripts/config_files/model_config.json"))

n = simulation_parameters["n"]

#Load actions
from pathlib import Path

all_actions_file = Path(f"MP/SP/Outputs/action_files/all_actions_3_nodes.npy")
if all_actions_file.is_file():
    all_actions = np.load(all_actions_file, allow_pickle=True)
else:
    all_actions = generate_all_actions(n)
    np.save(all_actions_file, all_actions)

pgen = simulation_parameters["pgen"]
pswap = simulation_parameters["pswap"]
lifetime = simulation_parameters["lifetime"]

loss = torch.nn.MSELoss()
epsilon = model_hyperparameters["epsilon"]
gamma = model_hyperparameters["gamma"]
lr = model_hyperparameters["learning_rate"]
to_target_nn = model_hyperparameters["transfer_after"]

def train_agent(env, agent, experiences):
    
    optimizer = torch.optim.SGD(agent.qnn.parameters(), lr = lr)

    #Initialize experience replay
    for s in range(model_hyperparameters["experience_replay_size"]):
        state, _ = env.reset()
        action_idx = np.random.randint(0, len(all_actions))
        action = all_actions[action_idx]
        new_state, reward, terminated, truncated, _ = env.act(action)
        experiences.store_experience(state, action, reward, new_state, terminated, truncated)


    #Apply Q network
    training_episodes = 10; training_batch_size = 5
    for t in range(training_episodes):
        training_batch = experiences.sample_experience(training_batch_size)
        states = torch.tensor(np.array([b[0] for b in training_batch]))
        actions = np.array([b[1] for b in training_batch])
        rewards = torch.tensor(np.array([b[2] for b in training_batch]))
        new_states = torch.tensor(np.array([b[3] for b in training_batch]))
        dones = torch.tensor(np.array([b[4] for b in training_batch]))
        exits = torch.tensor(np.array([b[5] for b in training_batch]))

        q_values_all = agent.qnn(states)
        batch_action_idx = torch.tensor(np.array([find_coo_matrix(act, all_actions) for act in actions]))
        q_values = q_values_all[np.arange(len(batch_action_idx)), batch_action_idx]

        with torch.no_grad():
            target_next_q_values = agent.target_nn(new_states).max(1)[0]
            target_values = rewards + gamma * target_next_q_values
            target_values = target_values.to(torch.float32)
        
        losses = loss(target_values, q_values)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if t%to_target_nn == 0:
            agent.target_nn.load_state_dict(agent.qnn.state_dict())

        print(f"Step {t}, Loss: {losses.item()}")


myEnv = GridTopologyEnv(n, pgen, pswap, lifetime, len(all_actions))
agent = AgentSP(10, len(all_actions), epsilon)
experiences = ExperienceReplay(model_hyperparameters["experience_replay_size"])
train_agent(myEnv, agent, experiences)