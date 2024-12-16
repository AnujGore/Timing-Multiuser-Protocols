import sys
sys.path.append(".")

import numpy as np
from ShortestPath.Scripts.analytical_solutions import generate_tpm

def two_segment_solution(pgen, pswap, metric):
    system_matrix = np.zeros(shape = (4, 4))
    system_matrix[0] = np.array([1 - (1-pgen)**2, -2*pgen*(1-pgen), -pgen**2, 0])
    system_matrix[1] = np.array([0, 1 - (1-pgen), -pgen, 0])
    system_matrix[2] = np.array([-(1 - pswap), 0, 1, -pswap])
    system_matrix[3] = np.array([0, 0, 0, 1])

    coefficient_vec = np.array([1, 1, 0, 0])
    if metric == "entanglement":
        pass
    if metric == "swap":
        coefficient_vec = np.logical_not(coefficient_vec).astype(int)

    solution = np.linalg.solve(system_matrix, coefficient_vec)
    return solution


def three_segment_solution(pgen, pswap, metric):
    system_matrix = np.zeros(shape = (9, 9))
    q = 1-pgen
    system_matrix[0] = np.array([(1- q**3), -2*pgen*q**2, -pgen*q**2, -(pgen**2)*2*q, -q*pgen**2, -pgen**3, 0, 0, 0])
    system_matrix[1] = np.array([0, 1 - q**2, 0, -pgen*q, -pgen*q, -pgen**2, 0, 0, 0])
    system_matrix[2] = np.array([0, 0, 1 - q**2, -2*pgen*q, 0, -pgen**2, 0, 0, 0])
    system_matrix[3] = np.array([-(1-pswap), 0, 0, 1, 0, 0, -pswap, 0, 0])
    system_matrix[4] = np.array([0, 0, 0, 0, 1 - q, -pgen, 0, 0, 0])
    system_matrix[5] = np.array([0, -(1-pswap), 0, 0, 0, 1, 0, -pswap, 0])
    system_matrix[6] = np.array([0, 0, 0, 0, 0, 0, (1 - q), -pgen, 0])
    system_matrix[7] = np.array([-(1-pswap), 0, 0, 0, 0, 0, 0, 1, 0])
    system_matrix[8] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])


    coefficient_vec = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0])
    if metric == "entanglement":
        pass
    if metric == "swap":
        coefficient_vec = np.logical_not(coefficient_vec).astype(int)

    solution = np.linalg.solve(system_matrix, coefficient_vec)
    return solution


def n_segment_solution(n, pgen, pswap):
    transition_matrix = generate_tpm(n, pgen, pswap)
    Q = transition_matrix[:-1, :-1]
    R = np.linalg.inv(np.eye(len(Q), len(Q)) - Q)
    K_bar = R@np.ones(len(R))

    return K_bar