import gymnasium as gym
from environment import *
from agent import *
from rich.progress import Progress

import numba

import pathlib

import json
simulation_parameter_dict = json.load(open("MP/Scripts/simulation_config.json"))

num_episodes = simulation_parameter_dict["num_episodes"]

n = simulation_parameter_dict["n"]
lifetime = simulation_parameter_dict["lifetime"]

all_states_file = pathlib.Path(f"MP/Outputs/{n}_nodes_{lifetime}_lifetime_states.npy")
if all_states_file.is_file():
    print("Loading pregenerated states...")
    all_states_array = np.load(all_states_file)

else:
    print("Generating states...")
    all_states_array = generate_all_states(n, lifetime)
    np.save(all_states_file, all_states_array)

all_actions_file = pathlib.Path(f"MP/Outputs/{n}_nodes_actions.npy")
if all_actions_file.is_file():
    print("Loading pregenerated actions...")
    all_actions_array = np.load(all_actions_file)
else:
    print("Generating actions...")
    all_actions_array = generate_all_actions(n)
    np.save(all_actions_file, all_actions_array)

print("Loading environment...")

env = gym.make('QuantumRepeaterGrid-v0', n = simulation_parameter_dict["n"], 
               pgen = simulation_parameter_dict["pgen"], 
               pswap = simulation_parameter_dict["pswap"], 
               all_actions_array = all_actions_array, 
               all_states_array = all_states_array,
               wait = False,
               lifetime = simulation_parameter_dict["lifetime"], 
               init_entangled = True if simulation_parameter_dict["init_entangled"] == 1 else False)
env.reset()

print("Loading agent...")
agent = Agent(n = simulation_parameter_dict["n"], 
              lifetime = simulation_parameter_dict["lifetime"],
              all_actions_array=all_actions_array,
              all_states_array=all_states_array)

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


        res_list[episode] = latency
    
        progress.update(task1, advance=1)

np.savetxt(f"MP/Outputs/N_{n}_lifetime_{lifetime}.txt", res_list)

env.close()