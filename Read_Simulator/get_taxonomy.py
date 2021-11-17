from utils import get_gtdb_info
import json
import sys
import os

def main():
    input_dir = sys.argv[1]
    gtdb_path = sys.argv[2]
    # load dictionary mapping labels to species
    f = open(os.path.join(input_dir, 'class_mapping.json'))
    class_mapping = json.load(f)
    # get gtdb taxonomy
    _, _, _, _, gtdb_taxonomy = get_gtdb_info(gtdb_path)
    ranks = ['genus', 'family', 'order', 'class']
    for i in range(len(ranks)):
        # get taxa
        list_taxa = []
        species_labels = []
        for j in range(len(class_mapping)):
            if class_mapping[str(j)] in gtdb_taxonomy:
                list_taxa.append(gtdb_taxonomy[class_mapping[str(j)]][i].split('__')[1])
                species_labels.append(j)
        # create dictionary mapping each taxon at a taxonomic level to a label
        list_unique_taxa = list(set(list_taxa))
        label_dict = dict(zip(list(range(len(list_unique_taxa))), list_unique_taxa))
        rev_label_dict = {value: key for key, value in label_dict.items()}
        with open(os.path.join(input_dir, f'{ranks[i]}_mapping_dict.json'), "w") as f:
            json.dump(label_dict, f)
        # create dictionary mapping labels at the species level to labels at given rank
        rank_species_dict = {str(k): rev_label_dict[list_taxa[k]] for k in range(len(species_labels))}
        with open(os.path.join(input_dir, f'{ranks[i]}_species_mapping_dict.json'), "w") as f:
            json.dump(rank_species_dict, f)

if __name__ == "__main__":
    main()
