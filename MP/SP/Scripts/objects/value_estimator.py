import numpy as np

def value_estimator(matrix, n):
    final_state = np.zeros(shape = (n*n, n*n))
    cn_loc = int(np.floor(n**2/2))
    final_state[cn_loc, n-1] = 1
    final_state[n-1, cn_loc] = 1
    final_state[cn_loc, 0] = 1
    final_state[0, cn_loc] = 1
    final_state[cn_loc, n**2-1] = 1
    final_state[n**2-1, cn_loc] = 1
    final_state[cn_loc, n*(n-1)] = 1
    final_state[n*(n-1), cn_loc] = 1

    final_state_eigs = np.linalg.eigvals(final_state)
    final_state_eigvecs = np.linalg.eig(final_state)[1]
    final_state_eigs_max_ind = np.argmax(final_state_eigs)
    final_state_value = np.abs(final_state_eigs[final_state_eigs_max_ind]*final_state_eigvecs[int(np.floor(n**2/2)), final_state_eigs_max_ind])

    eigvals = np.linalg.eigvals(matrix)
    eigvecs = np.linalg.eig(matrix)[1]

    max_ind = np.argmax(eigvals)

    this_matrix_val = eigvals[max_ind]*eigvecs[int(np.floor(n**2/2)), max_ind]

    final_val = this_matrix_val/final_state_value

    return final_val

