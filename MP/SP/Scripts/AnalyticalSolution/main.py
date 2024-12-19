import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Load matrices from text files
matrix1 = np.loadtxt(r'MP\SP\Scripts\AnalyticalSolution\different_matrix_2_seg.txt')
matrix2 = np.loadtxt(r'MP\SP\Scripts\AnalyticalSolution\different_matrix_3_seg.txt')
matrix3 = np.loadtxt(r'MP\SP\Scripts\AnalyticalSolution\different_matrix_4_seg.txt')

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

vmin = min(matrix1.min(), matrix2.min(), matrix3.min())
vmax = max(matrix1.max(), matrix2.max(), matrix3.max())

# Plot each matrix
cax1 = axes[0].imshow(matrix1, cmap='GnBu', aspect='auto', vmin = vmin, vmax = vmax)
axes[0].set_title('2 segments')
cax2 = axes[1].imshow(matrix2, cmap='GnBu', aspect='auto', vmin = vmin, vmax = vmax)
axes[1].set_title('3 segments')
cax3 = axes[2].imshow(matrix3, cmap='GnBu', aspect='auto', vmin = vmin, vmax = vmax)
axes[2].set_title('4 segments')

# Add a common colorbar

swap_loc = np.where(np.round(np.linspace(0.1, 0.9, 25), 2) == 0.6)

x_ticks = np.arange(0, 25, step=1)
y_ticks = np.arange(0, 25, step=1)

# Apply the common ticks
for ax in axes:
    ax.plot([-1, 25], [swap_loc[0]-1, swap_loc[0]-1], 'r--', alpha=0.75)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks(x_ticks, np.round(np.linspace(0.1, 0.9, 25), 2), rotation=90, fontsize=6)
    ax.set_yticks(y_ticks, np.round(np.linspace(0.1, 0.9, 25), 2), fontsize=6)

fig.supxlabel(r'$p_{gen}$')
fig.supylabel(r'$p_{SWAP}$', x = 0.01)
fig.suptitle("Difference between published latency and analytical solution for different segments")
fig.tight_layout()


cbar = fig.colorbar(cax1, ax=axes, orientation='vertical', pad = 0.01)

plt.show()

