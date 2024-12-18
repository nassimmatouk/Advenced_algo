# Comparison of TSP Solving Algorithms

## Description

This project aims to compare different algorithms for solving the Traveling Salesman Problem (TSP). TSP is a combinatorial optimization problem where the goal is to find the shortest possible route that visits a set of nodes and returns to the starting node. We analyze and compare the efficiency and performance of several algorithms, such as the greedy algorithm, tabu search, genetic algorithms, and many others.

## Installation

To install this project, follow the steps below:

1. Clone the repository:
	```bash
	git clone https://github.com/nassimmatouk/Advenced_algo
	cd Advenced_algo
	```

2. Create a virtual environment and activate it:
	```bash
	python -m venv env
	source env/bin/activate  # On Windows: env\Scripts\activate
	```

3. Install the necessary dependencies:
	```bash
	pip install -r requirements.txt
	```

4. To run the project:
	```bash
	python main.py
	```

## Configuring the `main.py` File

The `main.py` file is the main entry point of the project. Here is an example of how to configure and use it:

```python
from core.workbench import Worbench, Algorithm, Dataset

""" Main function """
def main():
	workbench = Worbench(algo_to_compare=[
                                       	Algorithm.BRUTE_FORCE,
                                       	Algorithm.MST_KRUSKAL,
                                       	Algorithm.BRANCH_BOUND,
                                       	Algorithm.MST_PRIM,
                                       	Algorithm.NEAREST_NEIGHBOR,
                                       	Algorithm.GREEDY,
                                       	Algorithm.RANDOMIZED,
                                       	Algorithm.BRANCH_BOUND_CLASSIC ],
                   	interval_numbers_nodes=(3, 500),
                   	symmetric=True,
                   	dataset=Dataset.A280,
                   	draw_one_graph=False )
	workbench.compare()

main()
```

5. run a single algorithm
 
 You can also test some algorithms directly in their file include the main function
