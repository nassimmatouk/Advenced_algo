import matplotlib.pyplot as plt
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
