import numpy as np
from objects.environment import GridTopologyEnv
from utils import generate_all_actions
from objects.agent import AgentSP

import json
simulation_parameters = json.load(open("RepeaterGrid/SP/Scripts/config_files/simulation_config.json"))
model_hyperparameters = json.load(open("RepeaterGrid/SP/Scripts/config_files/model_config.json"))

n = simulation_parameters["n"]

#Load actions
from pathlib import Path

all_actions_file = Path(f"RepeaterGrid/SP/Outputs/action_files/all_actions_{n}_nodes.npy")
if all_actions_file.is_file():
    all_actions = np.load(all_actions_file, allow_pickle=True)
else:
    all_actions = generate_all_actions(n)
    np.save(all_actions_file, all_actions)


pgen = simulation_parameters["pgen"]
pswap = simulation_parameters["pswap"]
lifetime = simulation_parameters["lifetime"]

def train_agent(env, agent):

    #Initialize experience replay
    experience_replay_size = model_hyperparameters["experience_replay_size"]
    experience_replay = np.zeros(shape = (experience_replay_size, 4))

    #Apply Q network
    num_timesteps = 10
    for t in range(num_timesteps):
        for s in range(experience_replay_size):
            state, _ = env.reset()
            action_idx = agent.act(state)
            action = all_actions[action_idx]
            new_state, reward, _, _, _ = env.act(action)
            experience_replay[s] = np.array([state, action, reward, new_state])



myEnv = GridTopologyEnv(n, pgen, pswap, lifetime, len(all_actions))
agent = AgentSP(10, len(all_actions), 1)
train_agent(myEnv, agent)