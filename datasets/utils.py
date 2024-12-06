from math import sqrt


def compute_distance_euc_2d(node_i, node_j,nodes_coord):
    coord_i = nodes_coord[node_i]
    coord_j = nodes_coord[node_j]
    return sqrt((coord_j[0]-coord_i[0])**2 + (coord_j[1]-coord_i[1])**2)


def compute_distance_wit_lat_lon(node_i, node_j,nodes_coord):
    # Calcul de la distance entre deux noeuds en utilisant la formule de Haversine
    # code inspiré de stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    from math import sin, cos, sqrt, atan2, radians
    # Approximation du rayon de la terre en km
    rayon = 6373.0

    lat1 = radians(nodes_coord[node_i][0])
    lon1 = radians(nodes_coord[node_i][1])
    lat2 = radians(nodes_coord[node_j][0])
    lon2 = radians(nodes_coord[node_j][1])
    diff_lon = lon2 - lon1
    diff_lat = lat2 - lat1
    a = sin(diff_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(diff_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return rayon * c