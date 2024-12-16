import numpy as np
import networkx as nx

def map_to_routing(listOfRepeaters, n):
    node_labels = []
    for r in listOfRepeaters:
        node_labels.append(r[0]*n + r[1])
    
    routing_labels = []
    for n in range(len(node_labels)-1):
        routing_labels.append((node_labels[n], node_labels[n+1]))

    return routing_labels


def generate_involved_repeaters(n):
    listOfRepeaters = [
        [(i, int(np.floor(n/2))) for i in range(n)], #Vertical line
        [(int(np.floor(n/2)), i) for i in range(n)], #Horizontal line
        [(i, 0) for i in range(int(np.ceil(n/2)))], #First quadrant line
        [(i, n-1) for i in range(int(np.floor(n/2)), n)], #Second quadrant line
        [(0, i) for i in range(int(np.floor(n/2)), n)], #Third quadrant line
        [(n-1, i) for i in range(int(np.ceil(n/2)))] #Fourth quadrant line
    ]
    
    listOfRepeaters_unnested = []
    for listR in listOfRepeaters:
        labeled_r = map_to_routing(listR, n)
        for ele in labeled_r:
            listOfRepeaters_unnested.append(ele)

    return np.array(listOfRepeaters_unnested)

def shortest_path(n, users, cn_loc, pgen, system_matrix):
    thisG = nx.grid_graph(dim = (n, n))
    thisAdj = nx.adjacency_matrix(thisG)

    this_graph = nx.from_numpy_array(thisAdj)
    paths = []
    this_graph = this_graph.copy()
    for u in users:
        path = nx.dijkstra_path(this_graph, cn_loc, u)
        this_graph.remove_nodes_from(path[1:])
        paths.append(path)

    for branch in paths:
        for n in range(len(branch)-1):
            system_matrix[0][branch[n], branch[n+1]] = 1
            system_matrix[0][branch[n+1], branch[n]] = 1
            system_matrix[1][branch[n], branch[n+1]] = np.random.geometric(pgen)
            system_matrix[1][branch[n+1], branch[n]] = system_matrix[1][branch[n], branch[n+1]]


    return system_matrix


def label_to_coor(label:int, n:int):
    row = label//n
    col = label%n
    return (row, col)


def hamming_distance(node1, node2):
    return int(np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1]))