import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scripts.evaluate_qamel import run_evaluation 

p = 0.4

nodes = np.arange(3, 6, 1)

eval_eps = 100

q = 0.7
k = 0.5

def p_l(L):
    p_0 = 0.0018
    return p_0*10**((-0.2*L)/10)

qamel_results = np.zeros(shape = (2, len(nodes)))

for j, vertices in enumerate(nodes):
    run_evaluation(vertices, p, q, eval_eps)
    ent_count = np.loadtxt(f"qamel/outputs/results/ent_counts/{vertices}_nodes.txt")
    swap_count = np.loadtxt(f"qamel/outputs/results/swap_counts/{vertices}_nodes.txt")
    fusion_count = np.random.geometric(k, size = eval_eps)

    data = np.array([ent_count, swap_count, fusion_count]).mean(axis = 1)
    data_std = np.array([ent_count, swap_count, fusion_count]).std(axis = 1)

    qamel_results[0, j] = np.sum(data)
    qamel_results[1, j] = np.sum(data_std)

np.savetxt(f"qamel/outputs/results/qamel_results_for_different_nodes.txt", qamel_results)