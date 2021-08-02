from Functions_For_Read_Sim import *  # used for all of the functions in Functions_For_Read_Sim
import pandas as pd  # used to manage tsv files

codon_table = 'codon_list.csv'  # name of the codon_list.csv file
species_file_name = 'species.tsv'  # name of the species.tsv file
gtdb_database = 'bac120_metadata_r95.tsv'  # file name of the gtdb_database

species_df = pd.read_csv(species_file_name, delimiter='\t', header=None)  # dataframe holding the species.tsv file
database_df = pd.read_csv(gtdb_database, delimiter='\t')  # dataframe holding the gtdb_database

# this dictionary has the species as keys and the labels as values: labels are integers

label_dictionary = dict(zip((range(len(list(species_df[0])))), list(species_df[0])))
# These lists are needed to assemble the genome dictionary


# this dictionary has accession ids as values and the species will be the keys

genome_dict = parse_dataframe(database_df)

json_dict(label_dictionary, 'class_mapping.json')  # takes the label dictionary and turns it into a .json file

# used for the codon table dictionaries

df = pd.read_csv('codon_list.csv', delimiter='\t')  # dataframe containing the proteins and codons
amino_list = df.amino  # list of amino acids
codon_list = df.codons  # list of codons
codon_amino = dict(zip(codon_list, amino_list))  # maps codon to amino acid
amino_codon = list_dict(amino_list, codon_list)

# generates the genome datasets


fasta_file = 'GCF_900660545.1_genomic.fna'


genome = exclude_plasmid(fasta_file)

mut_string, mut_stats, fw_read, reverse_read = mutate(genome['NZ_LR214974.1'], 0,
                                                      'NZ_LR214974.1', codon_amino, amino_codon, 0, False)

