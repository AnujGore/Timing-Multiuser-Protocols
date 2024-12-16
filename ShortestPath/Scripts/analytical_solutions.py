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