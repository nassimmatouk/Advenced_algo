import time
import numpy as np
from helper.useful import generate_random_matrix

# Classe pour implémenter les ensembles disjoints (Union-Find)
class DisjointSet:
    def __init__(self, n):
        # Initialisation : chaque élément est son propre parent
        self.parent = list(range(n))
        self.rank = [0] * n  # Suivi des rangs pour optimiser les unions

    def find(self, u):
        # Trouve le représentant de l'ensemble contenant u (avec compression de chemin)
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        # Réalise l'union des ensembles contenant u et v
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # Attache l'arbre de plus faible rang sous l'arbre de plus grand rang
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# Fonction pour construire un arbre couvrant minimum (MST) avec l'algorithme de Kruskal
def kruskal_mst(adj):
    N = len(adj)
    edges = []
    # Collecte toutes les arêtes avec leurs poids
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != 0:
                edges.append((adj[i][j], i, j))
    edges.sort()  # Trie les arêtes par poids croissant

    ds = DisjointSet(N)  # Initialisation des ensembles disjoints
    mst_edges = []

    # Parcours des arêtes triées
    for edge in edges:
        weight, u, v = edge
        if ds.find(u) != ds.find(v):  # Si u et v ne sont pas dans le même ensemble
            ds.union(u, v)  # Union des ensembles
            mst_edges.append((u, v, weight))  # Ajout de l'arête au MST

    return mst_edges

# Fonction pour effectuer un parcours préfixe (preorder traversal) d'un MST
def preorder_traversal(mst, start):
    from collections import defaultdict

    # Construction d'un graphe non orienté à partir du MST
    graph = defaultdict(list)
    for u, v, cost in mst:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()  # Ensemble des nœuds visités
    path = []  # Chemin construit

    def dfs(node):
        # Parcours en profondeur (DFS)
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)  # Démarrage du DFS depuis le nœud donné
    return path

# Approximation du problème du TSP avec l'arbre couvrant minimum
def tsp_mst_kruskal(adj):
    mst = kruskal_mst(adj)  # Construction du MST
    path = preorder_traversal(mst, 0)  # Parcours préfixe du MST
    path.append(path[0])  # Retour au point de départ pour former un cycle
    cost = sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1))
    return cost, path

# Fonction pour mesurer le temps d'exécution d'une autre fonction


def measure_execution_time_mst_kruskal(distances):
    start_time = time.time()
    cost, path = tsp_mst_kruskal(distances)
    end_time = time.time()
    execution_time = end_time - start_time
    return cost, path, execution_time



# Code principal
if __name__ == "__main__":
    # Boucle pour tester avec différentes tailles de graphes (nombre de villes)
    for number_cities in range(3, 100):
        print("\n******************** Nombre de villes :", number_cities, "********************")

        # Génère une matrice d'adjacence aléatoire pour le graphe
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)

        # Mesure le temps d'exécution de l'approximation TSP
        cost, path, execution_time = measure_execution_time_mst_kruskal(adj)

        # Affichage des résultats
        print("Chemin approximatif du TSP (MST avec Kruskal) :", path)
        print("Nombre de villes :", number_cities)
        print("Distance minimale :", cost)

