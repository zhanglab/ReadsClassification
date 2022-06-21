import sys
import os
import pandas as pd
# from Bio import SeqIO
from collections import defaultdict
import json

# def get_sequences(fasta):
#     records = list(SeqIO.parse(fasta, 'fasta'))
#     return [i.id for i in records]

# def get_ncbi_ground_truth(kraken_db_dir, training_genomes_info):
#
#     # load kraken taxids assigned to every sequence in training genomes
#     ncbi_db_df = pd.read_csv(os.path.join(kraken_db_dir, 'seqid2taxid.map'), sep='\t', header=None)
#     ncbi_db_dict = dict(zip(ncbi_db_df.iloc[:,0].tolist(), ncbi_db_df.iloc[:,1].tolist()))
#
#     # get ncbi taxonomy of every sequence in training genomes
#     ncbi_tax_dict, ncbi_taxids_dict  = create_ncbi_taxonomy(list(set(ncbi_db_df.iloc[:, 1].tolist())), kraken_db_dir, kraken_db_dir)
#
#     ncbi_taxonomy = defaultdict(set)
#     ranks = ['species', 'genus', 'family', 'order', 'class', 'phylum']
#     missing_labels = set()
#     # get sequences
#     with open(training_genomes_info, 'r') as f:
#         for line in f:
#             line = line.rstrip().split('\t')
#             seq = get_sequences(line[2])
#             for s in seq:
#                 if s in ncbi_db_dict:
#                     ncbi_tax = ';'.join([str(ncbi_tax_dict[ncbi_db_dict[s]][r]) for r in ranks])
#                     ncbi_taxonomy[line[1]].add(ncbi_tax)
#                 else:
#                     print(line[2], line[1], s)
#                     missing_labels.add(line[1])
#
#     # for k, v in ncbi_taxonomy.items():
#     #     print(k, len(v), len(set(v)), set(v))
#     print(len(ncbi_taxonomy))
#     print(len(missing_labels), len(missing_labels.difference(set(list(ncbi_taxonomy.keys())))))
#
#     for k in ncbi_taxonomy.keys():
#         ncbi_taxonomy[k] = list(ncbi_taxonomy[k])
#
#     with open(os.path.join(kraken_db_dir, 'ncbi_dl_toda_tax.json'), 'w') as f:
#         json.dump(ncbi_taxonomy, f)
#
#     return ncbi_taxonomy, ncbi_tax_dict


def parse_nodes_file(nodes_file):
    data = {}
    df = pd.read_csv(nodes_file, delimiter='\t', low_memory=False, header=None)
    list_taxids = df.iloc[:, 0].tolist()
    list_parents = df.iloc[:, 2].tolist()
    list_ranks = df.iloc[:, 4].tolist()
    for i in range(len(list_taxids)):
        data[str(list_taxids[i])] = [str(list_parents[i]), list_ranks[i]]
    return data

def parse_names_file(names_file):
    data = defaultdict(dict)
    df = pd.read_csv(names_file, delimiter='\t', low_memory=False, header=None)
    list_taxids = df.iloc[:, 0].tolist()
    list_names = df.iloc[:, 2].tolist()
    list_name_types = df.iloc[:, 6].tolist()
    for i in range(len(list_taxids)):
        if list_name_types[i] == 'scientific name' or list_name_types[i] == 'includes' or list_name_types[i] == 'synonym':
            if list_name_types[i] not in data[str(list_taxids[i])]:
                data[str(list_taxids[i])][list_name_types[i]] = {list_names[i]}
            else:
                data[str(list_taxids[i])][list_name_types[i]].add(list_names[i])

    return data

def correct_gtdb_info(gtdb_df, d_names, n_taxon, genome_id):
    print(f'taxid incorrect: {genome_id}\t{n_taxon}')
    new_taxid = str()
    for k, v in d_names.items():
        if v == n_taxon:
            new_taxid = k
    index = gtdb_df.index[gtdb_df['accession']==genome_id].tolist()[0]
    gtdb_df.loc[index, 'ncbi_taxid'] = new_taxid

    return new_taxid

def get_ncbi_taxonomy(current_taxid, d_nodes, d_names, gtdb_df=None, n_taxon=None, genome_id=None):
    ranks = {'strain': 0, 'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'class': 5, 'phylum': 6}

    list_taxids = ['na']*len(ranks)
    list_taxa = ['na']*len(ranks)
    list_ranks = ['na']*len(ranks)

    parent = str(current_taxid)

    while parent != '1' and parent != '0':
        if parent not in d_nodes:
            if gtdb_df:
                # correct gtdb info file
                parent = str(correct_gtdb_info(gtdb_df, d_names, n_taxon, genome_id))
            else:
                break

        t_rank = d_nodes[parent][1]

        if t_rank in ranks:
            # get name of taxon
            if parent in d_names:
                if 'scientific name' in d_names[parent].keys():
                    if len(d_names[parent]['scientific name']) > 1:
                        sys.exit(f'more than one scientific names: {parent}\t{current_taxid}\t{d_names[parent]["scientific name"]}')
                    else:
                        t_name = list(d_names[parent]['scientific name'])[0]
                else:
                    sys.exit(f'scientific name not in names type: {parent}\t{current_taxid}')
                    # t_name = d_names[parent]['includes'] if 'includes' in d_names[parent] else d_names[parent]['synonym']
            else:
                print(f'{parent} not in ncbi tax db')
                t_name = 'na'
            if t_name == '':
                t_name = 'na'

            list_taxa[ranks[t_rank]] = t_name
            list_taxids[ranks[t_rank]] = parent
            list_ranks[ranks[t_rank]] = t_rank

        parent = d_nodes[parent][0]

    return '|'.join(list_taxids), ';'.join(list_taxa), ';'.join(list_ranks)

def create_ncbi_taxonomy(taxid, ncbi_db_dir, output_dir=None):
    # get ncbi taxids info
    d_nodes = parse_nodes_file(os.path.join(kraken_db_dir, 'taxonomy', 'nodes.dmp'))
    d_names = parse_names_file(os.path.join(kraken_db_dir, 'taxonomy', 'names.dmp'))

    # remove taxid 0 corresponding to unclassified sequences by kraken
    if '0' in taxids:
        taxids.remove('0')
    taxids = list(taxids)
    # define list of taxa of interest
    ranks = ['strain', 'species', 'genus', 'family', 'order', 'class', 'phylum']
    # create table with the reconstructed ncbi taxonomy for each taxid in the given list
    table_taxa = pd.DataFrame(columns = ranks, index = taxids)
    table_taxids = pd.DataFrame(columns = ranks, index = taxids)
    # fill out table with default values
    for r in ranks:
        table_taxa[r] = 'na'
        table_taxids[r] = 0
    # get taxa at every rank of interest (if taxid starts at higher taxonomic rank, then all ranks
    # below will be misclassified, all ranks that we are not interested in will be ignored and cases where
    # species are unknown will also be misclassified)
    # ranks_count = {'strain': 0, 'species': 0, 'genus': 0, 'family': 0, 'order': 0, 'class': 0, 'phylum': 0}
    for i in range(len(taxids)):
        list_taxids, list_taxa, list_ranks = get_ncbi_taxids(str(taxids[i]), d_nodes, d_names)
        if len(list_taxids) > 0:
            for j in range(len(list_ranks)):
                table_taxa.loc[taxids[i], list_ranks[j]] = list_taxa[j]
                table_taxids.loc[taxids[i], list_ranks[j]] = list_taxids[j]
        else:
            print(f'missing taxid: {taxids[i]}')

    if output_dir:
        # write table to file
        table_taxa.to_csv(os.path.join(output_dir, 'table-ncbi-taxonomy.csv'))
        table_taxids.to_csv(os.path.join(output_dir, 'table-ncbi-taxids.csv'))

    table_taxa = table_taxa.to_dict(orient='index')
    # print(table_taxa)
    return table_taxa, table_taxids
