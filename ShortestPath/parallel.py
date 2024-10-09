import numba as nb
import numpy as np

@nb.njit(parallel = True)
def probability_success(p_array, p):
    for i in nb.prange(len(p_array)):
        p_array[i] = 1 - (1-p)**i

@nb.njit(parallel = True)
def filter_thres(p_array, thres):
    for i in nb.prange(len(p_array)):
        if p_array[i] >= thres:
            p_array[i] = p_array[i]
        else:
            p_array[i] = np.NaN