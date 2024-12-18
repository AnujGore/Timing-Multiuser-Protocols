from analytical import *
from adjacency_analytical import MonteCarloSP

def match_analytical(n, diff_mat):
    p = np.linspace(0.1, 0.9, 100)
    q = np.linspace(0.1, 0.9, 100)
    for i, pgen in enumerate(p):
        for j, pswap in enumerate(q):
            #published solution
            if n == 3:
                pub_sol = two_segment_solution(pgen, pswap, "entanglement")
            elif n == 4:
                pub_sol = three_segment_solution(pgen, pswap, "entanglement")
            elif n == 5:
                pub_sol = n_segment_solution(n-1, pgen, pswap)

            #our solution
            print(f"pgen: {pgen}; pswap: {pswap}; segments: {n-1}")
            our_sol = MonteCarloSP(3, pgen, pswap, "SenderReceiver")

            diff_mat[i, j] = np.abs(pub_sol[0] - our_sol[0].numpy())

    return diff_mat


for n in [3, 4, 5]:
    diff_mat = np.zeros(shape = (100, 100))
    diff_mat = match_analytical(n, diff_mat)
    np.savetxt(f"different_matrix_{n-1}_seg.txt", diff_mat)