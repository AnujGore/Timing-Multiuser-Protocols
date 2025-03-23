import numpy as np
import math

def generate_tpm(N, p, a):
    """
    Generates the probability transition matrix for a Markov chain with N segments based on Shchukin et al.

    Args:
         - N: number of segments on branch
         - p: probability of entanglement
         - s: probability of swapping
         - f: probability of fusion

    Returns: Transition probability matrix.    
    """
    tpm_1 = np.array([[1-p, p],[0, 1]], dtype = np.float64)
    mats= []
    for _ in range(N):
        mats.append(tpm_1)

    while True:
        temp_mats = []
        for l in range(0, len(mats), 2):
            tpm = np.kron(mats[l], mats[l+1])
            tpm[:-1, 0] += (1-a)*tpm[:-1, -1]
            tpm[:-1, -1] *= a
            temp_mats.append(tpm)
        
        mats = temp_mats
        if len(mats) == 1:
            break

    return mats[0]

def shchukin_eq(n, p, a):
    """
    Note that n = number of segments
    """
    tpm = generate_tpm(n, p, a)
    Q = tpm[:-1, :-1]
    u = tpm[:-1, -1]

    R = np.linalg.inv(np.eye(len(Q), len(Q)) - Q)
    K_bar = R@np.ones(len(R))
    return K_bar[0]


def bernardes_eq(n_segments, p):
    """
    This only applies for non-deterministic generation of entanglement
    """

    val = 0
    for j in range(1, n_segments+1):
        val += ((-1)**(j+1))*(math.comb(n_segments, j))*((1)/(1- (1 - p)**j))
    
    return val


def markov_approach(vertices, p, a):

    list_of_distances = []
    
    users = [(0, 0), (0, vertices-1), (vertices-1, 0), (vertices-1, vertices-1)]
    central_node = (int(vertices/2), int(vertices/2))

    for u in users:
        list_of_distances.append(int(np.abs(u[0] - central_node[0]) + np.abs(u[1] - central_node[1])))


    return shchukin_eq(max(list_of_distances), p, a)


def including_fusion(vertices, p, a, f):

    list_of_distances = []
    
    users = [(0, 0), (0, vertices-1), (vertices-1, 0), (vertices-1, vertices-1)]
    central_node = (int(vertices/2), int(vertices/2))

    for u in users:
        list_of_distances.append(int(np.abs(u[0] - central_node[0]) + np.abs(u[1] - central_node[1])))

    tpm = generate_tpm(max(list_of_distances), p, a)
    tpm[:-1, 0] += (1-f)*tpm[:-1, -1]
    tpm[:-1, -1] *= f

    Q = tpm[:-1, :-1]
    R = np.linalg.inv(np.eye(len(Q), len(Q)) - Q)
    K_bar = R@np.ones(len(R))
    return K_bar[0]


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