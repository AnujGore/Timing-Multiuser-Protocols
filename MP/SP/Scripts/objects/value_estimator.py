import numpy as np

def value_estimator(matrix, n):
    final_state = np.zeros(shape = (9, 9))
    final_state[0, 4] = 1
    final_state[4, 0] = 1
    final_state[2, 4] = 1
    final_state[4, 2] = 1
    final_state[4, 6] = 1 
    final_state[6, 4] = 1
    final_state[4, 8] = 1
    final_state[8, 4] = 1

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
