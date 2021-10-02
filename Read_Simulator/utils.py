import os
import gzip
import json
import pandas as pd
import random
import string
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict

def get_genetic_code(args):
    """ returns dictionaries mapping codons to amino acids and inverse dictionary  """
    genetic_code_df = pd.read_csv(args.genetic_code, delimiter='\t')  # dataframe containing the codons and their respective amino acids
    codon_aa = dict(zip(genetic_code_df.codons, genetic_code_df.amino))  # maps codon to amino acid
    aa_codon = defaultdict(list)
    for codon, aa in codon_aa.items():
        aa_codon[aa].append(codon)
    return codon_aa, aa_codon

def get_species(args):
    """ returns list of target species  """
    species_df = pd.read_csv(os.path.join(args.input_path, 'species.tsv'), sep='\t', header=None)
    return species_df[0].tolist()

def get_dataset_info(args, genome_dict, list_species):
    """ returns dictionary mapping labels to species  """
    # report missing species
    with open(os.path.join(args.input_path, 'missing-species.txt'), 'w') as f:
        for s in list_species:
            if s not in genome_dict.keys():
                f.write(f'{s}\n')
    # update list of species if necessary
    list_species = list(genome_dict.keys())
    # create dictionary mapping labels to species
    label_dict = dict(zip(list(range(len(list_species))), list_species))
    # create json file
    with open(os.path.join(args.input_path, 'class_mapping.json'), "w") as f:
        json.dump(label_dict, f)
    # report number of genomes per species
    with open(os.path.join(args.input_path, 'num_genomes_report.txt'), 'w') as f:
        for key, value in label_dict.items():
            f.write(f'{key}\t{value}\t{len(genome_dict[value])}\n')

    return label_dict

def load_class_mapping(filename):
    """ returns dictionary mapping labels to species stored in json file  """
    with open(filename) as f:
        class_mapping = json.load(f)
    return class_mapping

def find_largest_genome_set(args, genome_dict):
    """ returns species with largest number of genomes  """
    max_num = 0
    largest_species = ''
    for species, accession_list in genome_dict.items():
        if len(accession_list) > max_num:
            max_num = len(accession_list)
            largest_species = species
    # report species with largest number of genomes
    with open(os.path.join(args.input_path, 'species-max-num-genomes'), 'w') as f:
        f.write(f'{largest_species}\t{max_num}\n')

    return max_num

def get_sequences(fastafile):
    """ returns the list of all sequences in the fasta file except plasmids """
    fasta_list = []
    with gzip.open(fastafile, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if (rec.description.find('plasmid') or rec.description.find('Plasmid')) == -1:
                fasta_list.append(rec)
    return fasta_list

def complement(seq):
    """ returns the complement strand of a dna sequence """
    str_comp = ''
    nucleotide_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    for i in seq:
        try:
            str_comp += nucleotide_dict[i]
        except KeyError:
            str_comp += i
    return str_comp

def create_fastq_record(read_seq, read_id, list_records):
    """ generates fastq records """
    read_qual = ''.join([random.choice(string.ascii_uppercase) for _ in range(len(read_seq))])
    record_str = f'{"@" + read_id}\n{read_seq}\n+\n{read_qual}\n'
    list_records.append(record_str)

# this function looks for a open reading frame and if one exists returns true
def find_orf(seq, i):
    orf = ''
    list_stop_codons = ['TAA', 'TAG', 'TGA']
    j = i
    while j <= (len(seq) - 3):
        orf += seq[j:j+3]
        j += 3
        # add stop codon to ORF
        if seq[j:j+3] in list_stop_codons:
            orf += seq[j:j+3]
            break
        # if no stop codons have been found when reaching the end of the sequence
        # return an empty string
        if j == (len(seq) - 3) and seq[j:j+3] not in list_stop_codons:
            orf = ''
    return orf

def select_codon(codon, codon_amino, amino_codon):
    """ returns a synonymous codon """
    amino = codon_amino[codon]
    # randomly select one of the synonymous codons
    codon_changed = amino_codon[amino][random.randint(0, (len(amino_codon[amino]) - 1))]
    return codon_changed

def mut_counter(mutated_codon, original_codon):
    """ returns the number of substitutions or point mutations """
    counter = 0
    for i in range(len(mutated_codon)):
        if mutated_codon[i] != original_codon[i]:
            counter += 1
    return counter

def get_tetra_nt_fqcy(TETRA_nt, sequence):
    """ gets the frequency of all tetranucleotides in the sequence """
    for i in range(0, len(sequence)-3, 1):
        tetra_nt = sequence[i:i+4]
        TETRA_nt[tetra_nt] += 1
    return
