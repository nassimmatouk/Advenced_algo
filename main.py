
from core.workbench import Worbench, Algorithm


def main():
    """
    Main function

    """
    workbench = Worbench(algo_to_compare=[
                                         #Algorithm.BRANCH_BOUND ,
                                         #Algorithm.MST_KRUSKAL,
                                         #Algorithm.MST_PRIM,
                                         Algorithm.BRUTE_FORCE
                                               ],
                                                interval_numbers_nodes=(3, 10),
                                                symmetric=True
                                                )
    workbench.compare()

main()