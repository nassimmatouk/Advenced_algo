import sys
import time
from helper.useful import generate_random_matrix


class NearestNeighborTSPSolution:
    def __init__(self, adj):
        self.N = len(adj)  # Nombre de villes
        self.adj = adj  # Matrice des distances
        self.final_path = []  # Stocke le chemin final
        self.cost = 0  # Coût total du chemin

    def nearest_neighbor(self, start):
        visited = [False] * self.N  # Suivi des villes visitées
        path = [start]  # Chemin initialisé avec la ville de départ
        visited[start] = True
        current_city = start
        total_cost = 0

        for _ in range(self.N - 1):
            nearest_city = None
            min_distance = sys.maxsize

            # Trouver la ville la plus proche non visitée
            for city in range(self.N):
                if not visited[city] and self.adj[current_city][city] < min_distance:
                    nearest_city = city
                    min_distance = self.adj[current_city][city]

            # Ajouter cette ville au chemin
            path.append(nearest_city)
            visited[nearest_city] = True
            total_cost += min_distance
            current_city = nearest_city

        # Retourner à la ville de départ
        total_cost += self.adj[current_city][start]
        path.append(start)

        self.final_path = path
        self.cost = total_cost

    def solve(self, start=0):
        self.nearest_neighbor(start)
        return self.cost, self.final_path


# Fonction pour mesurer le temps d'exécution
def measure_execution_time_nearest_neighbor(distances):
    start_time = time.time()
    solver = NearestNeighborTSPSolution(distances)
    cost, path = solver.solve(start=0)
    end_time = time.time()
    execution_time = end_time - start_time
    return cost, path, execution_time


# Code principal
if __name__ == "__main__":
    for number_cities in range(3, 100):
        print("******************** Nombre de villes :", number_cities, "********************")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)
        cost, final_path, execution_time = measure_execution_time_nearest_neighbor(adj)
        print("Coût total :", cost)
        print("Chemin emprunté :", end=' ')
        for city in final_path:
            print(city, end=' ')
        print("\nTemps d'exécution :", execution_time)
