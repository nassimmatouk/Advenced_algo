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


def measure_execution_time_brute_force(distances):
    start_time = time.time()
    path, cost = tsp_brute_force(distances)
    end_time = time.time()
    execution_time = end_time - start_time
    return cost, path, execution_time
