from typing import List

import matplotlib.pyplot as plt
from jinja2.nodes import Tuple


def show_graph(cities, mst_edges):
    # Affichage du graphe
    fig, ax = plt.subplots()

    # Affichage des villes
    for i, city in enumerate(cities):
        ax.scatter(city[0], city[1], label=f'Ville {i + 1}', s=100)

    # Affichage des arêtes du MST
    for edge in mst_edges:
        city1, city2 = edge
        x_values = [cities[city1][0], cities[city2][0]]
        y_values = [cities[city1][1], cities[city2][1]]
        ax.plot(x_values, y_values, 'b-')

    # Ajouter les étiquettes des villes
    for i, city in enumerate(cities):
        ax.text(city[0] + 5, city[1] + 5, f'{i + 1}', fontsize=12)

    # Ajouter des légendes et des titres
    ax.set_title("Minimum Spanning Tree (MST)")
    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")
    plt.legend()

    # Afficher le graphe
    plt.show()


def analizer_complexity(complexity_data: List[Tuple(List[int], List[float])]):
    plt.figure(figsize=(10, 6))
    for data in complexity_data:
        plt.plot(data[0], data[1], 'o-', color='b')
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (Minimum Spanning Trees)")
    plt.show()