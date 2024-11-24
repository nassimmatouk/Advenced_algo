
from core.workbench import Worbench, Algorithm


def main():
    """
    Main function

    """
    workbench = Worbench(algo_to_compare=[
                                                                                    Algorithm.MST,
                                                                                    Algorithm.BRANCH_BOUND
                                                                                    ],
                                                interval_numbers_nodes=(3, 17),
                                                symmetric=True
                                                )
    workbench.compare()

main()