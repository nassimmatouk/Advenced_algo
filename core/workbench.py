
##  Auteur: Ana SANOU
##  Date de création : 24/11/2024
##
##
from enum import Enum
from functools import lru_cache
from typing import List, Tuple

from matplotlib import pyplot as plt

from helper.useful import generate_random_matrix
from implementation.tsp_branch_and_bound import measure_execution_time_branch_bound
from implementation.tsp_brute_force_copy import measure_execution_time_brute_force
from implementation.tsp_mst_approach_krusKal import measure_execution_time_mst_kruskal
from implementation.tsp_mst_approach_prim import measure_execution_time_mst_prim


class Algorithm(Enum):
    """
    Enumération des algorithmes à comparer
    NOM_ALGO = (fonction_de_mesure_de_temps, couleur_sur_le_graphique)
    """
    BRANCH_BOUND = measure_execution_time_branch_bound, "b"
    MST_KRUSKAL = measure_execution_time_mst_kruskal, "r"
    MST_PRIM = measure_execution_time_mst_prim, "o"
    BRUTE_FORCE = measure_execution_time_brute_force, "g"

class Worbench:
    def __init__(self, algo_to_compare: List[Algorithm],
                                interval_numbers_nodes: Tuple[int, int]= (3, 10),
                                symmetric: bool = True):
        self. interval_numbers_nodes =interval_numbers_nodes
        self.algo_to_compare = algo_to_compare
        self.data_to_plot = {} # {algo_name: (number_nodes, execution_times, color)}
        self.data_to_store = {} # {algo_name: (number_nodes, execution_times)}
        self.symmetric = symmetric


    def add_algo(self, algo):
        """
        ajout un algorithme à comparer
        """
        self.algo_to_compare.append(algo)

    @lru_cache
    def get_adjency_matrix(self, numbers_node):
        """
        Génère une matrice d'adjacence aléatoire
        lrucache permet de stocker les valeurs de la fonction pour ne pas les recalculer (pour ne pas regenerer la matrice)
        """
        adj = generate_random_matrix(num_cities=numbers_node, symmetric=self.symmetric)
        return adj

    def run_all_algos(self):
        """
        Exécute tous les algorithmes à comparer
        Stocker les données pour les graphiques et les fichiers CSV
        """

        for algo in self.algo_to_compare:
            print(f"*************Start ALGO: {algo}***************")
            number_nodes = []
            execution_times = []
            #memory_consommings = []
            costs = []
            paths = []
            for number_node in range(*self.interval_numbers_nodes):
                matrix_adj = self.get_adjency_matrix(number_node)
                # On exécute chaque algo ici
                cost, path, execution_time=algo.value[0](matrix_adj)
                print(f"************************ALGO {algo}****NODES : {number_node}, TIMES, {execution_time},DISTANCES : {cost}, OPTIMAL PATH : {path}*************")

                number_nodes.append(number_node)
                execution_times.append(execution_time)
                costs.append(cost)
                paths.append(path)
            self.data_to_plot[algo.name] = (number_nodes, execution_times, algo.value[1])
            self.data_to_store[algo.name] = (number_nodes, execution_times, costs, paths)
            print(f"*************END ALGO: {algo}***************")



    def compare(self):
        """
        Exécute tous les algorithmes, sauvegarde les données et trace les graphiques
        """
        self.run_all_algos()
        self.save_to_csv()
        self.plot()

    def save_to_csv(self):
        """
        Sauvegarde les données dans des fichiers CSV
        """
        DELIMITER ="|"
        for algo_name, data in self.data_to_store.items():
            number_nodes, execution_times, costs, final_paths = data
            with open(f"data/workbenchdata/{algo_name}.csv", "w") as f:
                f.write(f"number_nodes{DELIMITER}execution_times{DELIMITER}distances{DELIMITER}optimal_path\n")
                for i in range(len(number_nodes)):
                    f.write(f"{number_nodes[i]}{DELIMITER}{execution_times[i]}{DELIMITER}{costs[i]}{DELIMITER}{final_paths[i]}\n")

    def plot(self):
        """
        Trace les graphiques
        """
        for algo_name, data in self.data_to_plot.items():
            plt.plot(data[0], data[1], 'o-', label=algo_name)
        plt.xlabel("Number of nodes")
        plt.ylabel("Execution time (s)")
        plt.title("Complexity in time of the TSP algorithm")
        plt.legend()
        plt.show()