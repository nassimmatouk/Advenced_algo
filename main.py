from helper.distance import compute_distance_euc_2d
from parser.parser import parse_tsp_file


def main():
    file_path = 'C:\\Users\\userlocal\\PycharmProjects\\algoMST\\data\\a280.tsp'  # Remplace par le chemin vers ton fichier
    parsed_data = parse_tsp_file(file_path)
    distance = compute_distance_euc_2d(12, 100, parsed_data['NODE_COORDS'])
    print(distance)

main()