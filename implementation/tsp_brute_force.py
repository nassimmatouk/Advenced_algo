import sys
import os

from itertools import permutations
import matplotlib.pyplot as plt
import time

# Ajouter le répertoire parent au PYTHONPATH (pour pouvoir exécuté directement le fichier)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

def calculate_total_distance(route, distances):
    total_distance = 0
    num_cities = len(route)
    for i in range(num_cities - 1):
        total_distance += distances[route[i]][route[i + 1]]
    total_distance += distances[route[-1]][route[0]]  # Retour à la ville de départ
    return total_distance

def tsp_brute_force(distances):
    num_cities = len(distances)
    cities = list(range(num_cities))
    
    min_distance = float('inf')
    best_route = None
    
    for perm in permutations(cities):
        current_distance = calculate_total_distance(perm, distances)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    return best_route, min_distance

def plot_tsp_solution(cities, best_route):
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-', label="Chemin")
    plt.plot(x[0], y[0], 'go', label="Départ")
    plt.title("Solution TSP - Brute Force")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def measure_execution_time(distances):
    start_time = time.time()
    _, min_distance = tsp_brute_force(distances)
    end_time = time.time()
    return end_time - start_time

def analyze_complexity(max_cities=10):
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        distances = generate_random_matrix(num_cities, symmetric=True)
        
        execution_time = measure_execution_time(distances)
        times.append(execution_time)
        print(f"{num_cities} villes: temps d'exécution = {execution_time:.4f} secondes")
    
    plt.figure(figsize=(10, 6))
    plt.plot(city_counts, times, 'o-', color='b')
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (force brute)")
    plt.show()

if __name__ == "__main__":
    # Exemple : générer une matrice aléatoire de distances
    distances = generate_random_matrix(num_cities=5, symmetric=True)
    
    # Tester la résolution brute-force
    best_route, min_distance = tsp_brute_force(distances)
    print("Meilleur chemin :", best_route)
    print("Distance minimale :", min_distance)
    
    # Analyser la complexité avec des matrices aléatoires
    analyze_complexity(max_cities=10)
