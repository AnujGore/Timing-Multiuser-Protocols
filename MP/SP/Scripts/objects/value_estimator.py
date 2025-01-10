import numpy as np

type1 = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [1, 0, 0, 1, 0, 0, 1, 0, 1,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,]
])


type2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 1, 0, 1, 0, 1, 0, 1, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
])


type3 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0,],
    [1, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 1, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 1, 0, 0,],
    [0, 1, 0, 1, 0, 1, 0, 1, 0,],
    [0, 0, 1, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 1, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 1,],
    [0, 0, 0, 0, 0, 0, 0, 1, 0,],
])

type4 = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [1, 0, 1, 0, 0, 0, 1, 1, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 1,],
    [0, 0, 0, 0, 0, 0, 0, 1, 0,],
])

eigenvalues_1, eigenvectors_1 = np.linalg.eig(type1)
eigenvalues_2, eigenvectors_2 = np.linalg.eig(type2)
eigenvalues_3, eigenvectors_3 = np.linalg.eig(type3)
eigenvalues_4, eigenvectors_4 = np.linalg.eig(type4)

# Print the results
print("Eigenvalues for Type 1 (Cross Grid):", eigenvalues_1**2)
# print("Eigenvectors for Type 1 (Cross Grid):", eigenvectors_1[np.argmax(eigenvalues_1)])
print("Eigenvalues for Type 2 (Plus Sign Structure):", eigenvalues_2**2)
# print("Eigenvectors for Type 2 (Plus Sign Structure):", eigenvectors_2[np.argmax(eigenvalues_2)])
print("Eigenvalues for Type 3 (Hitler symbol):", eigenvalues_3**2)
# print("Eigenvectors for Type 3 (Hitler symbol):", eigenvectors_3[np.argmax(eigenvalues_3)])
print("Eigenvalues for Type 4:", eigenvalues_4**2)
# print("Eigenvectors for Type 4:", eigenvectors_4[np.argmax(eigenvalues_4)])