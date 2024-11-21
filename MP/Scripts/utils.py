import numpy as np

def generate_initial_state(n, pgen):
    arr = np.zeros(shape = (n, n))
    for i in range(n): arr[i, i] = np.NaN 
    for e in range(1, n): arr[e-1, e] = np.random.geometric(pgen)
    flipped_array = np.flip(arr)
    arr += flipped_array
    return arr