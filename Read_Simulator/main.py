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

species_in_database = [name[(name.find('s__') + 3):] for name in list(database_df['gtdb_taxonomy'])]
accession_id_list = list(database_df.accession)

# this dictionary has accession ids as values and the species will be the keys

genome_dict = {species_name: accession_id for species_name, accession_id in zip(species_in_database, accession_id) if }

json_dict(label_dictionary)  # takes the label dictionary and turns it into a .json file

print(genome_dict)
generate_datasets(genome_dict, label_dictionary,
                  'GCF_900101915.1_IMG-taxon_2599185216_annotated_assembly_genomic.fna')
