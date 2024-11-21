import math


# Fonction pour calculer la distance Euclidienne
def distance(city1, city2):
    return math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)


# Fonction de réduction de la matrice : réduit les coûts pour chaque ligne et chaque colonne
def reduce_matrix(dist_matrix):
    n = len(dist_matrix)
    row_reduction = [float('inf')] * n
    col_reduction = [float('inf')] * n

    # Réduction des lignes
    for i in range(n):
        row_min = min(dist_matrix[i])
        if row_min != float('inf'):
            row_reduction[i] = row_min
            for j in range(n):
                if dist_matrix[i][j] != float('inf'):
                    dist_matrix[i][j] -= row_min

    # Réduction des colonnes
    for j in range(n):
        col_min = min(dist_matrix[i][j] for i in range(n))
        if col_min != float('inf'):
            col_reduction[j] = col_min
            for i in range(n):
                if dist_matrix[i][j] != float('inf'):
                    dist_matrix[i][j] -= col_min

    return row_reduction, col_reduction


# Fonction de calcul de la borne inférieure : somme des réductions de lignes et de colonnes
def calculate_bound(dist_matrix):
    row_reduction, col_reduction = reduce_matrix([row[:] for row in dist_matrix])
    bound = sum(row_reduction) + sum(col_reduction)
    return bound


# Fonction de recherche Branch-and-Bound pour le TSP
def branch_and_bound_tsp(dist_matrix):
    n = len(dist_matrix)
    best_bound = float('inf')
    best_solution = None

    # Fonction récursive pour l'exploration de l'arbre
    def search(current_path, visited, current_bound, current_cost):
        nonlocal best_bound, best_solution

        # Si toutes les villes sont visitées, vérifier la solution
        if len(current_path) == n:
            # Ajouter le coût de retour à la ville de départ
            current_cost += dist_matrix[current_path[-1]][current_path[0]]
            if current_cost < best_bound:
                best_bound = current_cost
                best_solution = current_path + [current_path[0]]
            return

        # Pruner si le coût courant + la borne est supérieure à la meilleure solution
        if current_cost + current_bound >= best_bound:
            return

        # Explorer toutes les villes non visitées
        for i in range(n):
            if not visited[i]:
                # Réduire la matrice en fonction de la ville i
                new_visited = visited[:]
                new_visited[i] = True
                new_path = current_path + [i]
                new_cost = current_cost + dist_matrix[current_path[-1]][i] if current_path else 0

                # Calculer la nouvelle borne inférieure
                new_bound = calculate_bound([dist_matrix[j][:] for j in range(n)])
                search(new_path, new_visited, new_bound, new_cost)

    # Initialiser la recherche
    visited = [False] * n
    search([0], visited, calculate_bound(dist_matrix), 0)

    return best_solution, best_bound


# Exemple d'utilisation

# Exemple de matrice de distances (coordonnées des villes)
cities = [
    (288, 149), (288, 129), (270, 133), (256, 141), (256, 157),
    (246, 157), (236, 169), (228, 169), (228, 161), (220, 169)
]

# Construction de la matrice des distances
dist_matrix = [[distance(city1, city2) for city2 in cities] for city1 in cities]

# Résolution du problème avec Branch-and-Bound
best_solution, best_bound = branch_and_bound_tsp(dist_matrix)

print("Meilleure solution:", best_solution)
print("Coût total:", best_bound)
