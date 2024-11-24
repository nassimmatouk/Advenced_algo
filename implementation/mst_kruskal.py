import math
import time
from matplotlib import pyplot as plt
from helper.useful import generate_random_matrix


# Fonction pour trouver le parent dans l'ensemble des sous-arbres
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# Fonction pour unir deux ensembles
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)

    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    elif rank[root_x] > rank[root_y]:
        parent[root_y] = root_x
    else:
        parent[root_y] = root_x
        rank[root_x] += 1


# Algorithme de Kruskal
def kruskal_minimum_spanning_tree(cities):
    N = len(cities)
    edges = []
    mst_edges = []
    cost = 0

    # Générer toutes les arêtes avec leurs poids
    for i in range(N):
        for j in range(i + 1, N):
            dist = math.dist(cities[i], cities[j])
            edges.append((dist, i, j))

    # Trier les arêtes par poids croissant
    edges.sort()

    # Initialiser les sous-arbres
    parent = list(range(N))
    rank = [0] * N

    for edge in edges:
        weight, u, v = edge
        root_u = find(parent, u)
        root_v = find(parent, v)

        # Ajouter l'arête si elle ne forme pas un cycle
        if root_u != root_v:
            mst_edges.append((u, v))
            cost += weight
            union(parent, rank, root_u, root_v)

            # Arrêter si toutes les arêtes nécessaires sont ajoutées
            if len(mst_edges) == N - 1:
                break

    return mst_edges, cost


# Mesurer le temps d'exécution pour Kruskal
def measure_execution_time(cities):
    start_time = time.time()
    mst_edges, cost = kruskal_minimum_spanning_tree(cities)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time


# Analyser la complexité et tracer le graphe
def analyze_and_plot(max_cities=10):
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        # Générer une matrice aléatoire de distances
        distances = generate_random_matrix(num_cities, symmetric=True)

        # Mesurer le temps d'exécution
        execution_time = measure_execution_time(distances)
        times.append(execution_time)

        print(f"{num_cities} villes : temps d'exécution = {execution_time:.4f} secondes")

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.plot(city_counts, times, marker='o', color='b')
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Temps d'exécution de l'algorithme de Kruskal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Lancer l'analyse et tracer le graphe
    analyze_and_plot(max_cities=50)
