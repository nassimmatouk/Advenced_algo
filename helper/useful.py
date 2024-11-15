from math import sqrt
import random
import networkx as nx  # Pour la génération et la vérification des graphes


def compute_distance_euc_2d(node_i, node_j,nodes_coord):
    coord_i = nodes_coord[node_i]
    coord_j = nodes_coord[node_j]
    return sqrt((coord_j[0]-coord_i[0])**2 + (coord_j[1]-coord_i[1])**2)

def generate_random_matrix(num_cities, symmetric=True, max_distance=100):
    """
    Génère une matrice de distances aléatoire
    """
    distances = [[0 if i == j else random.randint(1, max_distance) for j in range(num_cities)] for i in range(num_cities)]
    
    if symmetric:
        # Rendre la matrice symétrique
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distances[j][i] = distances[i][j]
    
    return distances

def generate_erdos_renyi_graph(num_cities, prob=0.5, max_distance=100):
    """
    Génère un graphe TSP basé sur le modèle Erdos-Renyi.
    """
    G = nx.erdos_renyi_graph(num_cities, prob)
    while not nx.is_connected(G):  # Assurer que le graphe est connecté
        G = nx.erdos_renyi_graph(num_cities, prob)
    
    # Générer la matrice des distances en fonction du graphe généré
    distances = [[0 if i == j else (random.randint(1, max_distance) if G.has_edge(i, j) else float('inf')) for j in range(num_cities)] for i in range(num_cities)]
    
    return distances

def generate_watts_strogatz_graph(num_cities, k=2, prob=0.3, max_distance=100):
    """
    Génère un graphe TSP basé sur le modèle Watts-Strogatz.
    """
    G = nx.watts_strogatz_graph(num_cities, k, prob)
    while not nx.is_connected(G):  # Assurer que le graphe est connecté
        G = nx.watts_strogatz_graph(num_cities, k, prob)
    
    # Générer la matrice des distances en fonction du graphe généré
    distances = [[0 if i == j else (random.randint(1, max_distance) if G.has_edge(i, j) else float('inf')) for j in range(num_cities)] for i in range(num_cities)]
    
    return distances
