from datasets.utils import compute_distance_euc_2d
from parser.parser import parse_tsp_file


def get_data_a280(num_cities=280,symmetric=True):
    if not symmetric:
        raise ValueError("The A280 dataset is symmetric")
    data = parse_tsp_file('datasets/data/a280.tsp')
    distances = [[compute_distance_euc_2d(i,j,data["NODE_COORDS"]) for i in data["NODE_COORDS"]] for j in data["NODE_COORDS"]]
    distances = distances[:num_cities][:num_cities]
    distances_new = []
    for index,distance in enumerate(distances):
        if index == num_cities:
            break
        distances_new.append(distance[:num_cities])

    return distances_new