import os
import gzip
import json
import pandas as pd
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

def get_label_dict(args, genome_dict):
    """ returns dictionary mapping labels to species  """
    # update list of species if necessary
    args.list_species = list(genome_dict.keys())
    # create dictionary mapping labels to species
    label_dict = dict(zip(list(range(len(args.list_species))), args.list_species))
    # create json file
    with open(os.path.join(args.input_path, 'class_mapping.json'), "w") as f:
        json.dump(label_dict, f)
    return label_dict

def load_class_mapping(filename):
    """ returns dictionary mapping labels to species stored in json file  """
    with open(filename) as f:
        class_mapping = json.load(f)
    return class_mapping

def find_largest_genome_set(genome_dict):
    """ returns species with largest number of genomes  """
    largest = 0
    largest_species = ''
    for species, accession_list in genome_dict.items():
        if len(accession_list) > largest:
            largest = len(accession_list)
            largest_species = species
    return largest, largest_species

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
    read_qual = [random.choice(string.ascii_uppercase) for _ in range(len(read_seq))]
    record = SeqRecord(Seq(read_seq), id=read_id)
    quality_scores = [ord(i) for i in read_qual]
    record.letter_annotations["phred_quality"] = quality_scores
    record.format('fastq')
    list_records.append(record)

# this function looks for a open reading frame and if one exists returns true
def find_orf(seq, i, list_stop_codons):
    while seq[i:i + 3] not in list_stop_codons:
        if i >= len(seq):
            is_mut = -1
            old_stop_codon = []
            return is_mut, old_stop_codon
        i += 3
        is_mut = 0
    old_stop_codon = seq[i:i + 3]
    return is_mut, old_stop_codon

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
