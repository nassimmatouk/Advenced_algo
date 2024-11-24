import time
import numpy as np
from heapq import heappop, heappush

from helper.useful import generate_random_matrix


def prim_mst(adj):
    N = len(adj)
    selected = [False] * N
    mst_edges = []
    min_heap = [(0, 0, -1)]  # (cost, current_node, parent_node)

    while min_heap:
        cost, u, parent = heappop(min_heap)
        if selected[u]:
            continue
        selected[u] = True
        if parent != -1:
            mst_edges.append((parent, u, cost))

        for v in range(N):
            if not selected[v] and adj[u][v] != 0:
                heappush(min_heap, (adj[u][v], v, u))

    return mst_edges


def preorder_traversal(mst, start):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    for u, v, cost in mst:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return path


def tsp_mst_approximation(adj):
    mst = prim_mst(adj)
    path = preorder_traversal(mst, 0)
    path.append(path[0])  # return to the starting point
    return path


# Fonction pour mesurer le temps d'exécution d'une fonction
def measure_execution_time_mst_prim(func, *args, **kwargs):
    """
    Mesure le temps d'exécution d'une fonction donnée.
    - func: fonction à exécuter.
    - args, kwargs: arguments passés à la fonction.
    """
    start_time = time.time()  # Temps de départ
    result = func(*args, **kwargs)  # Appel de la fonction
    end_time = time.time()  # Temps de fin
    print(f"Temps d'exécution de {func.__name__}: {end_time - start_time:.6f} secondes")
    return result


# Code principal
if __name__ == "__main__":

    # Boucle pour tester avec différentes tailles de graphes (nombre de villes)
    for number_cities in range(3, 100):
        print("\n******************** Nombre de villes :", number_cities, "********************")

        # Génère une matrice d'adjacence aléatoire pour le graphe
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)

        # Mesure le temps d'exécution de l'approximation TSP avec Prim
        path = measure_execution_time_mst_prim(tsp_mst_approximation, adj)

        # Calcul du coût total du chemin
        cost = sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1))

        # Affichage des résultats
        print("Chemin approximatif du TSP (MST avec Prim) :", path)
        print("Nombre de villes :", number_cities)
        print("Distance minimale :", cost)

