import numpy as np
import matplotlib.pyplot as plt

import json
simulation_parameter_dict = json.load(open("MP/Scripts/simulation_config.json"))

n = simulation_parameter_dict["n"]
lifetime = simulation_parameter_dict["lifetime"]

# Load the text file data
actions = np.loadtxt(f'MP/Outputs/N_{n}_lifetime_{lifetime}.txt')[0]
rewards = np.loadtxt(f'MP/Outputs/N_{n}_lifetime_{lifetime}.txt')[1]

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(range(len(actions)), actions, marker='o', color = 'b', label='Actions taken')
plt.plot(range(len(actions)), rewards, marker='x', color = 'r', label='Rewards')
plt.xlabel('Episodes')  # Replace with appropriate label
plt.ylabel('Waiting time')  # Replace with appropriate label
plt.title('Convergence of q-table')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
