# TSP Solver



## Description


## Getting Started

### Dependencies

[README.md](README.md)
### Installing



#### Commande line to built  the code

```console

```



### Executing program


### How to test 



## Authors



ex. SANOU Salimata Ana 
ex. [@githubaccount](https://github.com/salimataana)

## Version History



## Acknowledgments

### References


import math
from turtle import distance


def compute_distance_euc_2d(city1, city2):
    return math.dist(city1, city2)

def MST_heuristic(cities):
    INF = float('inf')
    N = len(cities)
    selected_node = [False] * N
    no_edge = 0
    selected_node[0] = True
    cost = 0

    while no_edge < N - 1:
        minimum = INF
        a, b = -1, -1

        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if not selected_node[n]:
                        dist = distance(cities[m], cities[n])
                        if dist < minimum:
                            minimum = dist
                            a, b = m, n

        cost += minimum
        selected_node[b] = True
        no_edge += 1

    return cost

cities = [(288, 149), (288, 129), (270, 133), (256, 141), (256, 157)]
cost = MST_heuristic(cities)
print(f"Le coût total du Minimum Spanning Tree est : {cost}")












import math

def compute_distance_euc_2d(city1, city2):
    return math.dist(city1, city2)

def MST_heuristic(cities):
    INF = float('inf')
    N = len(cities)
    selected_node = [False] * N
    no_edge = 0
    selected_node[0] = True
    cost = 0
    visited_cities = [cities[0]]  # Villes visitées au fur et à mesure

    while no_edge < N - 1:
        minimum = INF
        a, b = -1, -1

        # Affichage des villes restantes non visitées
        remaining_cities = [i for i in range(N) if not selected_node[i]]
        print(f"Villes restantes à visiter : {remaining_cities}")

        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if not selected_node[n]:
                        dist = compute_distance_euc_2d(cities[m], cities[n])
                        if dist < minimum:
                            minimum = dist
                            a, b = m, n

        # Ajouter la ville à l'arbre couvrant
        cost += minimum
        selected_node[b] = True
        visited_cities.append(cities[b])  # Simuler la visite de la ville
        no_edge += 1

        # Affichage de la ville ajoutée à l'arbre
        print(f"Ville {b} ajoutée à l'arbre avec une distance de {minimum:.2f}")
        print(f"Villes visitées jusqu'à maintenant : {visited_cities}")

    return cost, visited_cities

# Exemple avec un petit ensemble de villes
cities = [(288, 149), (288, 129), (270, 133), (256, 141), (256, 157)]
cost, visited_cities = MST_heuristic(cities)
print(f"\nLe coût total du Minimum Spanning Tree est : {cost:.2f}")
print(f"Villes visitées dans l'ordre : {visited_cities}")


