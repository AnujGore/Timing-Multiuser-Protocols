import numpy as np
import matplotlib.pyplot as plt

nodes = 3
entangled = True

y1 = np.loadtxt(f'MP/SP/Outputs/results/{nodes}_nodes_actions_{"entangled" if entangled else "not_entangled"}.txt')
err1 = np.loadtxt(f'MP/SP/Outputs/results/{nodes}_nodes_actions_{"entangled" if entangled else "not_entangled"}_std.txt')

y2 = np.loadtxt(f'MP/SP/Outputs/results/{nodes}_nodes_rewards_{"entangled" if entangled else "not_entangled"}.txt')
err2 = np.loadtxt(f'MP/SP/Outputs/results/{nodes}_nodes_rewards_{"entangled" if entangled else "not_entangled"}_std.txt')

# Create the plot
fig, ax1 = plt.subplots()

x = np.arange(1, len(y2)+1)

# First dataset (left y-axis)
ax1.scatter(x, y2, marker = 'x', label='Actions')
ax1.set_xlabel('Timesteps')  # Change as needed
ax1.set_ylabel('Actions')  # Change as needed


# # Add the second y-axis
# ax2 = ax1.twinx()
# ax2.errorbar(x, y2, yerr=err2, fmt='s-', label='Rewards', color = 'blue')
# ax2.set_ylabel('Rewards')  # Change as needed

# # Add legends for clarity
# fig.tight_layout()  # Adjust spacing to prevent overlap
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
plt.grid(True)
plt.show()
