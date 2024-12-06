import random


def generate_random_matrix(num_cities, symmetric=True, max_distance=100):
    """
    Génère une matrice de distances aléatoire
    """
    distances = [[0 if i == j else random.randint(1, max_distance) for j in range(num_cities)] for i in
                 range(num_cities)]

    if symmetric:
        # Rendre la matrice symétrique
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distances[j][i] = distances[i][j]

    return distances
