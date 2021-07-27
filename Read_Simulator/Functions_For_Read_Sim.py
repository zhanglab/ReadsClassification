import json  # used to generate json
import random  # needed to generate random numbers
import re  # used to check characters
from Bio import SeqIO  # used to parse fasta file
from collections import defaultdict  # used to create genome dictionary
import pandas


# This function takes a dictionary and makes it into json format


def json_dict(dictionary, filename):
    with open(filename, "w") as f:
        json.dump(dictionary, f)
    f.close()


# This function will create a file to hold the number of mutations per ORF and number of mutations per genome


def mutation_statistics(filename, string_to_file):
    with open(filename, 'w') as f:
        f.write(string_to_file)


# This function will combine the species names in species.tsv with its accession_id


def generate_datasets(genome_dict, label_dict, fastafile, codon_amino, amino_codon):
    # load the genome fasta file (key = sequence id, value = sequence info
    genome = {rec.id: list(rec.seq) for rec in SeqIO.parse(fastafile, 'fasta')}

    for species, label in label_dict.items():
        # TODO: get accession ids for the species in label_dict

        # create dictionaries to store reverse and forward reads
        fw_read_dict = {}  # key = read id, value = read sequence
        rv_read_dict = {}  # key = read id, value = read sequence
        # create dictionaries to store fasta sequences of mutated sequences
        dict_mutated_sequences = defaultdict(list)  # key = sequence id, value = mutated sequence
        for sequence_id, sequence in genome.items():
            # call mutate function to mutate the sequence and generate reads
            # (add reads to dict_rev_reads and dict_fw_reads)
            print(f'progress {len(sequence)} {sequence_id}')
            mut_seq, mut_stats = mutate(sequence, label, sequence_id, codon_amino, amino_codon)
            dict_mutated_sequences[sequence_id] = mut_seq

        print(dict_mutated_sequences)
    # generate fasta file with mutated sequence
    # file = open('modified_DNA_sequence.fna', 'w')
    # for i in seq_list:

    # generate fastq file for forward and reverse reads (separately)
    # add information about percentage of mutations to file mentioned above
    return 'check to see if it worked'


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


# TODO: Might be broken, This function breaks up a dict of sequences into 250 nucleotide segments
#  later held in a list of strings

def generate_reads(sequence_id, label, positive_strand, negative_strand, dict_forward_reads, dict_reverse_reads):
    insert_size = 600
    read_length = 250
    num_reads = 0
    for i in range(0, len(positive_strand) - insert_size + read_length, read_length):
        fw_read = positive_strand[i:i + read_length]
        rv_read = negative_strand[i + read_length + insert_size:i + read_length + insert_size + read_length]
        dict_forward_reads[f'{sequence_id}-{label}-{num_reads}-1'] = fw_read
        dict_reverse_reads[f'{sequence_id}-{label}-{num_reads}-2'] = rv_read
        # print(f'This is the forward read {fw_read} {len(fw_read)} \n this is the index: {i}')
        # print(f'This is the reverse read {rv_read} {len(rv_read)}')
    return dict_forward_reads, dict_reverse_reads


# reads dictionary of codons and mutates the open reading frames


def mutate(seq, label, seq_id, codon_amino, amino_codon):
    forward_dict = {}
    reverse_dict = {}
    counter = 0
    mutated_sequence = []
    list_stop_codons = ['TAA', 'TAG', 'TGA']
    total = len(seq)
    i = 0
    while i < len(seq) - 3:
        if i % 100 == 0:
            print(f'running {i}')
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
        # print(positive_strand)
        # print(negative_strand)
        # TODO: add generate read function
        forward_dict, reverse_dict = generate_reads(seq_id, label, positive_strand, negative_strand, forward_dict,
                                                    reverse_dict)
    # print(f'{mutated_sequence}')

    return ''.join(mutated_sequence), ((counter / total) * 100)


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


# parses the genome file into a dictionary: keys are the species names and the values are a list of accession ids

# this function takes in a dataframe and will parse it into pieces that will later be used for other functions


def parse_dataframe(dataframe):
    # species = key and value = list of species

    dataframe_dict = defaultdict(list)
    ncbi_assembly_level_list = list(dataframe.ncbi_assembly_level)
    ncbi_genome_category_list = list(dataframe.ncbi_genome_category)
    species_in_database = [name[(name.find('s__') + 3):] for name in list(dataframe['gtdb_taxonomy'])]
    accession_id_list = list(dataframe.accession)

    for i in range(len(dataframe)):
        if ncbi_assembly_level_list[i] == "Complete Genome" and ncbi_genome_category_list[i] \
                != ("derived from metagenome" or "derived from environmental_sample"):
            dataframe_dict[species_in_database[i]].append(accession_id_list[i][3:])

    return dataframe_dict
