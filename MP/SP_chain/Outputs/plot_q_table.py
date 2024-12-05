import numpy as np
import matplotlib.pyplot as plt

import json
simulation_parameter_dict = json.load(open("MP/Scripts/simulation_config.json"))

n = simulation_parameter_dict["n"]
lifetime = simulation_parameter_dict["lifetime"]

# Load the text file data
q_table = np.loadtxt(f'MP/Outputs/N_{n}_lifetime_{lifetime}_qtable.txt')

plt.imshow(q_table, aspect='auto', cmap='viridis')

# Add a color bar to show the scale
plt.colorbar()

# Add labels to the axes
plt.xlabel("Actions")
plt.ylabel("States")

plt.xticks([], [])
plt.yticks([], [])


# Display the heatmap
plt.tight_layout()
plt.show()