import gymnasium as gym
from environment import *
from agent import *
from rich.progress import Progress

import numba

import json
simulation_parameter_dict = json.load(open("MP/Scripts/simulation_config.json"))

num_episodes = simulation_parameter_dict["num_episodes"]

print("Making environment...")

env = gym.make('QuantumRepeaterGrid-v0', n = simulation_parameter_dict["n"], 
               pgen = simulation_parameter_dict["pgen"], 
               pswap = simulation_parameter_dict["pswap"], 
               lifetime = simulation_parameter_dict["lifetime"], 
               init_entangled = True if simulation_parameter_dict["init_entangled"] == 1 else False)
env.reset()

print("Making agent...")
agent = Agent(n = simulation_parameter_dict["n"], 
              lifetime = simulation_parameter_dict["lifetime"])

res_list = np.zeros(shape = num_episodes)

with Progress() as progress:
    task1 = progress.add_task("[cyan]Training RL model...", total=num_episodes)
    for episode in numba.prange(num_episodes):
        state, info = env.reset()
        done = False
        latency = 0

        while not done:
            action_idx = agent.act(state)
            new_state, reward, done, truncated, info = env.step(action_idx)
            latency += 1
            agent.update(state, new_state, reward, action_idx)
            state = new_state
            print(state)

        res_list[episode] = latency
    
        progress.update(task1, advance=1)

np.savetxt(f"MP/Outputs/N_{simulation_parameter_dict["n"]}_lifetime_{simulation_parameter_dict["lifetime"]}.txt", res_list)

env.close()