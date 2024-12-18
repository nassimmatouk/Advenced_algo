
from core.workbench import Worbench, Algorithm, Dataset


def main():
    """
    Main function

    """
    workbench = Worbench(algo_to_compare=[
                                         #Algorithm.BRUTE_FORCE,
                                         Algorithm.MST_KRUSKAL,
                                         #Algorithm.BRANCH_BOUND,
                                         Algorithm.MST_PRIM,
                                         Algorithm.NEAREST_NEIGHBOR,
                                         Algorithm.GREEDY,
                                         Algorithm.RANDOMIZED,
                                         #Algorithm.BRANCH_BOUND_CLASSIC

                                               ],
                                                interval_numbers_nodes=(3, 500),
                                                symmetric=True,
                                                #dataset=Dataset.RANDOM,
                                                dataset=Dataset.A280,
                                                draw_one_graph=False
                                                )
    workbench.compare()

main()