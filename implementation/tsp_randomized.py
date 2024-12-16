import sys
import os

import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import random
import time
from itertools import permutations

# Ajouter le répertoire parent au PYTHONPATH (pour pouvoir exécuté directement le fichier)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

### Algorithme randomisé pour le TSP ###
def tsp_randomized(distances, iterations=1000):
    """
    Résout le problème TSP en utilisant une approche randomisée.
    :param distances: Matrice de distances NxN.
    :param iterations: Nombre d'itérations pour générer des chemins aléatoires.
    :return: Meilleur chemin trouvé et distance totale.
    """
    num_cities = len(distances)
    best_route = None
    min_distance = float('inf')

    for _ in range(iterations):
        # Générer une permutation aléatoire des villes
        route = list(range(num_cities))
        random.shuffle(route)

        # Calculer la distance totale pour cette permutation
        current_distance = sum(distances[route[i]][route[i + 1]] for i in range(num_cities - 1))
        current_distance += distances[route[-1]][route[0]]  # Retour à la ville de départ

        # Mettre à jour la meilleure solution si nécessaire
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = route

    # Ajouter la ville de départ à la fin pour boucler le chemin
    if best_route:
        best_route.append(best_route[0])

    return best_route, min_distance

# Création d'une matrice de distances depuis TSPLIB
def create_distance_matrix(problem, max_nodes=8):
    nodes = list(problem.get_nodes())[:max_nodes]
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distances, nodes

def measure_execution_time_randomized(distances):
    start_time = time.time()
    best_route, min_distance = tsp_randomized(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

def plot_graph(cities, distances=None, labels=True):
    G = nx.Graph()

    # Ajout des nœuds au graphe
    for i, city in enumerate(cities):
        G.add_node(i, pos=city)

    # Ajout des arêtes au graphe (si les distances sont fournies)
    if distances is not None and isinstance(distances, np.ndarray) and distances.size > 0:
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j and distances[i, j] < float('inf'):
                    G.add_edge(i, j, weight=distances[i, j])

    # Récupération des positions pour l'affichage
    pos = {i: city for i, city in enumerate(cities)}
    nx.draw(G, pos, with_labels=labels, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title("Graphe des villes")
    plt.show()

# Visualisation de la solution du TSP
def plot_solution(cities, best_route):
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    plt.plot(x, y, 'bo-', label="Chemin optimal")
    plt.plot(x[0], y[0], 'go', label="Départ/Arrivée")
    plt.title("Solution TSP (Brute Force)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()



### Analyse et visualisation de la complexité pour TSP Randomized ###
def analyze_complexity_randomized(max_cities=10, iterations=1000):
    """
    Analyse la complexité en temps de l'algorithme randomisé TSP.
    :param max_cities: Nombre maximum de villes pour les tests.
    :param iterations: Nombre d'itérations aléatoires pour chaque test.
    """
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        # Générer une matrice de distances aléatoire
        distances = generate_random_matrix(num_cities, symmetric=True)

        # Mesurer le temps d'exécution de l'algorithme randomisé
        start_time = time.time()
        tsp_randomized(distances, iterations=iterations)
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)

        print(f"{num_cities} villes : temps d'exécution = {execution_time:.6f} secondes")

    # Visualiser la complexité
    plt.plot(city_counts, times, 'o-', color='purple', label="TSP Randomized")
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (Randomized)")
    plt.grid()
    plt.legend()
    plt.show()


### Tests par cas pour l'approche randomisée ###

# Cas 1 : Matrice de distances simple
def test_simple_matrix_randomized():
    print("\n=== Test avec une matrice simple (Randomized) ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Positions fictives des villes

    plot_graph(cities, distances)
    best_route, min_distance = tsp_randomized(distances, iterations=1000)
    print("Meilleur chemin (Randomized) :", best_route)
    print("Distance minimale (Randomized) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour un maximum de 5 villes
    print("\n=== Analyse de la complexité pour une matrice simple ===")
    analyze_complexity_randomized(max_cities=5, iterations=1000)


# Cas 2 : Matrice de distances générée aléatoirement
def test_random_matrix_randomized(num_cities=5, iterations=1000):
    print(f"\n=== Test avec une matrice aléatoire ({num_cities} villes) (Randomized) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]  # Générer des positions aléatoires

    plot_graph(cities, distances)
    best_route, min_distance = tsp_randomized(distances, iterations=iterations)
    print("Meilleur chemin (Randomized) :", best_route)
    print("Distance minimale (Randomized) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour le nombre de villes spécifié
    print("\n=== Analyse de la complexité pour une matrice aléatoire ===")
    analyze_complexity_randomized(max_cities=num_cities, iterations=iterations)


# Cas 3 : Instance TSPLIB
def test_tsplib_instance_randomized(tsp_file, max_nodes=8, iterations=1000):
    print(f"\n=== Test avec une instance TSPLIB ({max_nodes} villes) (Randomized) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)

    # Extraire les coordonnées des villes
    cities = [problem.node_coords[node] for node in selected_nodes]
    plot_graph(cities, distances)

    best_route, min_distance = tsp_randomized(distances, iterations=iterations)
    print("Meilleur chemin (Randomized) :", best_route)
    print("Distance minimale (Randomized) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour les nœuds sélectionnés
    print("\n=== Analyse de la complexité pour une instance TSPLIB ===")
    analyze_complexity_randomized(max_cities=max_nodes, iterations=iterations)


### Main avec analyse de la complexité ###

if __name__ == "__main__":
    # Test 1 : Matrice simple
    test_simple_matrix_randomized()

    # Test 2 : Matrice aléatoire
    # test_random_matrix_randomized(num_cities=6, iterations=5000)

    # Test 3 : Instance TSPLIB
    # tsp_file = "C:/Users/userlocal/Desktop/M1/AA/theProject/berlin52.tsp"
    # test_tsplib_instance_randomized(tsp_file, max_nodes=8, iterations=5000)
