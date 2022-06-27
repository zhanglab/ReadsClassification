import json
import pandas as pd
import os
from utils import load_json_dict
from collections import defaultdict

def get_gtdb_info(args):
    # load gtdb info
    gtdb_df = pd.read_csv(args.gtdb_info, delimiter='\t', low_memory=False)
    gtdb_taxonomy = list(gtdb_df['gtdb_taxonomy'])
    ncbi_taxonomy = list(gtdb_df['ncbi_taxonomy_unfiltered'])
    accession = list(gtdb_df['accession'])
    taxids = list(gtdb_df['ncbi_taxid'])

    return gtdb_taxonomy, ncbi_taxonomy, accession, taxids, gtdb_df

def get_rank_taxa(args, dl_toda_taxonomy):
    args.labels_mapping_dict = {}
    for r_name, r_index in {'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'class': 5, 'phylum': 6}.items():
        args.labels_mapping_dict[r_name] = {}
        for label, taxonomy in dl_toda_taxonomy.items():
            args.labels_mapping_dict[r_name][label] = taxonomy.split(';')[r_index]
    # create dictionary mapping species names to species labels
    args.inv_labels_mapping_dict = {v: k for k, v in args.labels_mapping_dict['species'].items()}

def load_mapping_dict(args, input_dir):
    args.labels_mapping_dict = {}
    args.labels_mapping_dict['species'] = load_json_dict(os.path.join(input_dir, 'species_labels.json'))
    args.labels_mapping_dict['genus'] = load_json_dict(os.path.join(input_dir, 'genus_labels.json'))
    args.labels_mapping_dict['family'] = load_json_dict(os.path.join(input_dir, 'family_labels.json'))
    args.labels_mapping_dict['order'] = load_json_dict(os.path.join(input_dir, 'order_labels.json'))
    args.labels_mapping_dict['class'] = load_json_dict(os.path.join(input_dir, 'class_labels.json'))
    args.labels_mapping_dict['phylum'] = load_json_dict(os.path.join(input_dir, 'phylum_labels.json'))
    args.inv_labels_mapping_dict = {v: k for k, v in args.labels_mapping_dict['species'].items()}
    args.rank_species_mapping = {}
    args.rank_species_mapping['genus'] = load_json_dict(os.path.join(input_dir, 'genus_species_labels.json'))
    args.rank_species_mapping['family'] = load_json_dict(os.path.join(input_dir, 'family_species_labels.json'))
    args.rank_species_mapping['order'] = load_json_dict(os.path.join(input_dir, 'order_species_labels.json'))
    args.rank_species_mapping['class'] = load_json_dict(os.path.join(input_dir, 'class_species_labels.json'))
    args.rank_species_mapping['phylum'] = load_json_dict(os.path.join(input_dir, 'phylum_species_labels.json'))


def get_dl_toda_taxonomy(args, species):
    list_ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum']
    if species.isnumeric():
        species_id = species
        species = args.labels_mapping_dict['species'][species_id]
    else:
        species_id = str(args.inv_labels_mapping_dict[species])

    if args.to_ncbi:
        return args.dl_toda_taxonomy[species_id]
    else:
        list_taxa = [species, species]
        list_taxids = [species_id, species_id]
        for i in range(1, len(list_ranks), 1):
            rank_taxid = str(args.rank_species_mapping[list_ranks[i]][str(species_id)])
            rank_taxon = args.labels_mapping_dict[list_ranks[i]][str(rank_taxid)]
            list_taxids.append(rank_taxid)
            list_taxa.append(rank_taxon)

        return '|'.join(list_taxids), ';'.join(list_taxa), ';'.join(list_ranks)
