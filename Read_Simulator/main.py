from Functions_For_Read_Sim import *  # used for all of the functions in Functions_For_Read_Sim
import pandas as pd  # used to manage tsv files
import sys


# These lists are needed to assemble the genome dictionary


# this dictionary has accession ids as values and the species will be the keys


# used for the codon table dictionaries


# generates the genome datasets


# fasta_file = 'GCF_900660545.1_genomic.fna'

# genome = exclude_plasmid(fasta_file)


# This area is being used as a test. Should be deleted later

# test_mapping = {'0': 'Mycoplasma cynos', '1': 'Mycoplasma columbinum'}
# test_accession_ids = {'Mycoplasma cynos': exclude_plasmid('GCF_900660545.1_genomic.fna'),
# 'Mycoplasma columbinum': exclude_plasmid('GCF_900660685.1_genomic.fna')}


def main():
    species_file_name = sys.argv[1]  # name of the species.tsv file
    gtdb_database = sys.argv[2]  # file name of the gtdb_database
    codon_list = sys.argv[3]
    path = sys.argv[4]

    # load genetic code

    df = pd.read_csv(codon_list, delimiter='\t')  # dataframe containing the proteins and codons
    amino_list = df.amino  # list of amino acids
    codon_list = df.codons  # list of codons
    codon_amino = dict(zip(codon_list, amino_list))  # maps codon to amino acid
    amino_codon = list_dict(amino_list, codon_list)  # maps amino acids to codons

    # loads species in dataset

    species_df = pd.read_csv(species_file_name, sep='\t', header=None)  # dataframe holding the species.tsv file

    # load gtdb database info

    database_df = pd.read_csv(gtdb_database, delimiter='\t')  # dataframe holding the gtdb_database
    genome_dict = parse_dataframe(database_df, species_df, path)  # gets the genome dictionary

    # this dictionary has the species as keys and the labels as values: labels are integers

    label_dictionary = dict(zip((range(len(list(species_df[0])))), list(species_df[0])))

    json_dict(label_dictionary, 'class_mapping.json')  # takes the label dictionary and turns it into a .json file

    # calls the generate dataframe function

    generate_datasets(genome_dict, label_dictionary, codon_amino, amino_codon)

    print(genome_dict)


if __name__ == '__main__':
    main()
