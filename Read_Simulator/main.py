from Functions_For_Read_Sim import *  # used for all of the functions in Functions_For_Read_Sim
import pandas as pd  # used to manage tsv files

codon_table = 'codon_list.csv'  # name of the codon_list.csv file
species_file_name = 'species.tsv'  # name of the species.tsv file
gtdb_database = 'bac120_metadata_r95.tsv'  # file name of the gtdb_database

species_df = pd.read_csv(species_file_name, delimiter='\t')  # dataframe holding the species.tsv file
database_df = pd.read_csv(gtdb_database, delimiter='\t')  # dataframe holding the gtdb_database

# this dictionary has the species as keys and the labels as values: labels are integers

label_dictionary = dict(zip(list(species_df['Formosa sp007197735']), list(range(len(
    list(species_df['Formosa sp007197735']))))))

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
json_dict(codon_amino, 'codon_to_amino.json')
json_dict(amino_codon, 'amino_to_codon.json')

# generates the genome datasets


fasta_file = 'GCF_900660685.1_genomic.fna'

genome = {rec.id: list(rec.seq) for rec in SeqIO.parse(fasta_file, 'fasta')}

mut_string, mut_stats = mutate(genome['NZ_LR215041.1'], 0, 'NZ_LR215041.1', codon_amino, amino_codon)

print(mut_string)
