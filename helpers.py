import numpy as np
from tqdm import tqdm

def monte_carlo_sim(protocol, sim_count, **kwargs):
    waiting_times = np.zeros((4, sim_count), dtype = np.float64)
    for i in tqdm(range(sim_count), desc=f"Simulating {protocol.__name__} protocol", unit="simulation"):
        waiting_times[:, i] = protocol(**kwargs)

    return waiting_times

def manhattan_distance(vertices):
    md = 0
    users = [(0, 0), (0, vertices-1), (vertices-1, 0), (vertices-1, vertices-1)]
    central_node = (int(vertices/2), int(vertices/2))
    for u in users:
        new_md = int(np.abs(u[0] - central_node[0]) + np.abs(u[1] - central_node[1]))
        if new_md > md:
            md = new_md

    return md

def even_factors(n):
    i = 1
    while 2**(i+1) <= n:
        i+=1
    return 2**i

def good_factors(n):
    diff = 10000
    good_factor_list = []
    while diff > 0:
        good_factor_list.append(even_factors(n))
        n = n - even_factors(n)
        diff = n
    
    return good_factor_list