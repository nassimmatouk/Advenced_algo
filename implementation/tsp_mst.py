import math
import time
from matplotlib import pyplot as plt
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, compute_distance_euc_2d


def minimun_spanning_trees(cities):
    INF = float('inf')
    N = len(cities)
    selected_node = [False] * N
    no_edge = 0
    selected_node[0] = True
    cost = 0
    mst_edges = []
    path = [0]  # Chemin des villes dans l'ordre du MST

    while no_edge < N - 1:
        minimum = INF
        a, b = -1, -1

        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if not selected_node[n]:
                        dist = math.dist(cities[m], cities[n])
                        if dist < minimum:
                            minimum = dist
                            a, b = m, n

        cost += minimum
        selected_node[b] = True
        no_edge += 1
        mst_edges.append((a, b))
        path.append(b)  # Ajouter la ville visitée au chemin

    return mst_edges, cost, path

def measure_execution_time(cities):
    start_time = time.time()
    mst_edges, cost, path = minimun_spanning_trees(cities)
    end_time = time.time()
    execution_time = end_time - start_time
    return mst_edges, cost, path, execution_time


def analyze_complexity(max_cities=10):
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        distances = generate_random_matrix(num_cities, symmetric=True)

        mst_edges, cost, path, execution_time = measure_execution_time(distances)
        times.append(execution_time)
        print(f"{num_cities} villes: temps d'exécution = {execution_time:.4f} secondes")

    plt.figure(figsize=(10, 6))
    plt.plot(city_counts, times, 'o-', color='b')
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (Minimun Spanning Trees)")
    plt.show()


if __name__ == "__main__":
    # Exemple : générer une matrice aléatoire de distances

    #analyze_complexity(200)
    #exit()

    for number_cities in range(5,200):
        distances = generate_random_matrix(num_cities=number_cities, symmetric=True)
        # Tester la résolution minimum spanning tree
        mst_edges, cost, path, execution_time = measure_execution_time(distances)
        print("nombres de villes : ", number_cities)
        print("Temps d'éxecution :", execution_time)
        print("Distance minimale :", cost)
        print("Meilleur chemin :", path)

