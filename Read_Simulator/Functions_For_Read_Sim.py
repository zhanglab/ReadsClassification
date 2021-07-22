import json  # used to generate json
import random  # needed to generate random numbers
import re  # used to check characters
from Bio import SeqIO  # used to parse fasta file
import pandas as pd
from collections import defaultdict  # used to create genome dictionary


# This function takes a dictionary and makes it into json format


def json_dict(dictionary):
    with open("class_mapping.json", "w") as f:
        json.dump(dictionary, f)


# This function will create a file to hold the number of mutations per ORF and number of mutations per genome


def mutation_statistics(filename, string_to_file):
    with open(filename, 'w') as f:
        f.write(string_to_file)
    filename.close()

# This function will combine the species names in species.tsv with its accession_id


def generate_datasets(genome_dict, label_dict, fastafile):
    for species, label in label_dict.items():
        # get accession ids for the species in label_dict
        list_genomes = genome_dict[species]
        # load the genome fasta file (key = sequence id, value = sequence info)
        genome = {rec.id: list(rec.seq) for rec in SeqIO.parse(fastafile, 'fasta')}
        # create dictionaries to store reverse and forward reads
        fw_read_dict = {}  # key = read id, value = read sequence
        rv_read_dict = {}  # key = read id, value = read sequence
        # create dictionaries to store fasta sequences of mutated sequences
        dict_mutated_sequences = {}  # key = sequence id, value = mutated sequence
        for sequence_id, sequence in genome.items():
            # call mutate function to mutate the sequence and generate reads
            # (add reads to dict_rev_reads and dict_fw_reads)
            mutate(sequence, label, sequence_id)
        print(genome)

    # generate fasta file with mutated sequence
    # generate fastq file for forward and reverse reads (separately)
    # add information about percentage of mutations to file mentioned above


# The function below produces the complement of a sequence


def complement(seq):
    str_comp = ''
    nucleotide_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    for i in seq:
        try:
            str_comp += nucleotide_dict[i]
        except KeyError:
            str_comp += i
    return str_comp


# This function breaks up a dict of sequences into 250 nucleotide segments later held in a list of strings

def generate_reads(sequence_id, label, positive_strand, negative_strand, dict_forward_reads, dict_reverse_reads):
    insert_size = 600
    read_length = 250
    num_reads = 0
    for i in range(0, len(positive_strand) - insert_size + read_length, read_length):
        fw_read = positive_strand[i:i + read_length]
        rv_read = negative_strand[i + read_length + insert_size:i + read_length + insert_size + read_length]
        dict_forward_reads[f'{sequence_id}-{label}-{num_reads}-1'] = fw_read
        dict_reverse_reads[f'{sequence_id}-{label}-{num_reads}-2'] = rv_read
        # print(f'This is the forward read {fw_read} {len(fw_read)}')
        # print(f'This is the reverse read {rv_read} {len(rv_read)}')
    return dict_forward_reads, dict_reverse_reads


# reads dictionary of codons and mutates the open reading frames


def mutate(seq, label, seq_id):
    df = pd.read_csv('codon_list.csv', delimiter='\t')  # dataframe containing the proteins and codons
    amino_list = df.amino  # list of amino acids
    codon_list = df.codons  # list of codons
    codon_amino = dict(zip(codon_list, amino_list))  # maps codon to amino acid
    amino_codon = list_dict(amino_list, codon_list)

    forward_dict = {}
    reverse_dict = {}
    counter = 0
    mutated_sequence = []
    list_stop_codons = ['TAA', 'TAG', 'TGA']
    total = len(seq)
    i = 0
    while i < len(seq) - 3:
        codon = seq[i:i + 3]
        if bool(re.match('^[ACTG]+$', ''.join(codon))):
            if ''.join(codon) == 'ATG':
                orf = []
                j = i
                while ''.join(seq[j:j + 3]) not in list_stop_codons and j < len(seq) - 3:
                    if bool(re.match('^[ACTG]+$', ''.join(seq[j:j + 3]))):
                        mutated_codon = random_select(seq[j:j + 3], codon_amino, amino_codon)
                        orf += mutated_codon
                        counter += mut_counter(mutated_codon, seq[j:j + 3])
                    else:
                        orf += seq[j:j + 3]
                    j += 3
                orf += list_stop_codons[random.randint(0, 2)]
                mutated_sequence += orf
                i = j + 3

            else:
                mutated_sequence += codon
                i += 3

        else:
            mutated_sequence += codon
            i += 3

        positive_strand = ''.join(mutated_sequence)
        negative_strand = complement(positive_strand)

        # TODO: add generate read function
        forward_dict, reverse_dict = generate_reads(seq_id, label, positive_strand, negative_strand, forward_dict,
                                                    reverse_dict)
    return forward_dict, reverse_dict, ((counter / total) * 100)


def mut_counter(mutated_codon, original_codon):
    counter = 0
    for i in range(len(mutated_codon)):
        if mutated_codon[i] != original_codon[i]:
            counter += 1
    return counter


def random_select(list_codon, codon_amino, protein_keys):
    codon = ''.join(list_codon)
    amino = codon_amino[codon]
    codon_changed = protein_keys[amino][random.randint(0, (len(protein_keys[amino]) - 1))]
    return list(codon_changed)


# This function will make a dictionary have values of lists


def list_dict(amino_list, codon_list):
    temp_dict = {}
    for i in range(len(amino_list)):
        if amino_list[i] in temp_dict:
            temp_dict[amino_list[i]].append(codon_list[i])

        else:
            temp_dict[amino_list[i]] = [codon_list[i]]
    return temp_dict


def write_to_fasta(seq_dict, description):
    file = open('modifiedDNA.fna', 'w')  # opens the fasta file that will hold the mutated seq
    for seq_id, seq in seq_dict.items():  # iterates through the mutated dict
        sequence = ''
        count = 0  # iterating int to go through the seq in each dict entry
        while count < len(seq):
            sequence = sequence + '\n' + ''.join(seq[count: count + 60])
            count += 60
        full_str = '> ' + description[seq_id] + ' ' + sequence + '\n'  # the full str that will be be written to fasta
        file.write(full_str)  # writes to fasta
    file.close()  # closes file


def create_genome_dict(species_in_database, accession_id_list, database_df):
    genome_dict = defaultdict(list)
    for i in range(len(database_df)):
        genome_dict[species_in_database[i]].append(accession_id_list[i])
    return genome_dict



