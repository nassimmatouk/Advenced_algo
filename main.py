
from core.workbench import Worbench, Algorithm


def main():
    """
    Main function

    """
    workbench = Worbench(algo_to_compare=[
                                         Algorithm.BRANCH_BOUND ,
                                         Algorithm.MST_KRUSKAL,
                                         Algorithm.MST_PRIM
                                               ],
                                                interval_numbers_nodes=(3, 17),
                                                symmetric=True
                                                )
    workbench.compare()

main()