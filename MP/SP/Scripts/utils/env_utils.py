import numpy as np

def label_to_coor(label:int, n:int):
    row = label//n
    col = label%n
    return (row, col)


def hamming_distance(node1, node2):
    return int(np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1]))