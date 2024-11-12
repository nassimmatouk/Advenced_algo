def parse_tsp_file(file_path):
    # Initialisation du dictionnaire data
    data = {
        'NAME': '',
        'COMMENT': '',
        'TYPE': '',
        'DIMENSION': 0,
        'EDGE_WEIGHT_TYPE': '',
        'NODE_COORDS': {}
    }

    #Lecture du fichier
    with open(file_path, 'r') as file:
        lines = file.readlines()

    #Parcours des lignes du fichier
    for line in lines:
        line = line.strip()  #Enlever les espaces blancs au début et à la fin de la ligne

        #Extraction des métadonnées
        if line.startswith('NAME'):
            data['NAME'] = line.split(':')[1].strip()
        elif line.startswith('COMMENT'):
            data['COMMENT'] = line.split(':')[1].strip()
        elif line.startswith('TYPE'):
            data['TYPE'] = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            data['DIMENSION'] = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            data['EDGE_WEIGHT_TYPE'] = line.split(':')[1].strip()

        #Section des coordonnées des nœuds
        elif line.startswith('NODE_COORD_SECTION'):
            #On commence à lire les coordonnées des nœuds après cette ligne
            continue
        elif line == 'EOF':
            break  ## Fin du fichier, on arrête le traitement

        #Lecture des coordonnées des nœuds
        elif line:
            parts = line.split()
            if len(parts) == 3:
                #
                node_id = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                data['NODE_COORDS'][node_id]=(x, y)

   # Retour des données extraites
    return data


#Un exemple d'utilisation

"""
file_path = 'C:\\Users\\userlocal\\PycharmProjects\\algoMST\\data\\a280.tsp'  # Remplace par le chemin vers ton fichier
parsed_data = parse_tsp_file(file_path)

#Afficher les données extraites
print(parsed_data['NAME'])  #
print(parsed_data['COMMENT'])  #
print(parsed_data['TYPE'])  #
print(parsed_data['DIMENSION'])  #
print(parsed_data['EDGE_WEIGHT_TYPE'])  #
# print(list(parsed_data['NODE_COORDS'].items())[:10])
print(parsed_data['NODE_COORDS'])

"""