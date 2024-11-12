from math import sqrt


def compute_distance_euc_2d(node_i, node_j,nodes_coord):
    coord_i = nodes_coord[node_i]
    coord_j = nodes_coord[node_j]
    return sqrt((coord_j[0]-coord_i[0])**2 + (coord_j[1]-coord_i[1])**2)