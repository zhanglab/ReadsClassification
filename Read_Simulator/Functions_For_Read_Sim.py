import json  # used to generate json
import random  # needed to generate random numbers
import re  # used to check characters
import string
import glob
from collections import defaultdict  # used to create genome dictionary
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import os
import math
from utils import *

def mutate_genomes(args, species, label, needed_iterations):
    """ returns 2 list of genomes (original and mutated versions), one for training and one for testing  """
    # create empty dictionary to store the genomes (key = genome id, value = list of sequences)
    dict_sequences = defaultdict(list)
    # create file to store mutation mutation report
    mut_f = open(os.path.join(args.input_path, f'{label}_mutation_report.txt'), 'w')
    # create dictionary to keep track of the count per genomes
    genomes_count = defaultdict(int)

    # define the list of genomes to be mutated and the list of genomes not to be mutated
    if args.mutate:
        # get average GC content and average tetranucleotide frequencies per species
        args.GC_content, args.TETRA_nt = get_genomes_info(args, species, label)
        # get list of randomly selected genomes that will be mutated
        list_mutate = random.choices(args.genome_dict[species], k=(needed_iterations - len(args.genome_dict[species])))
        # define list of genomes that won't be mutated
        list_genomes = args.genome_dict[species]
    else:
        list_mutate = []
        # randomly select genomes to add to the list of genomes that won't be mutated
        list_genomes = args.genome_dict[species] + random.choices(args.genome_dict[species], k=(needed_iterations - len(args.genome_dict[species])))

    # get sequences from unmutated genomes
    for fasta_file in list_genomes:
        # get genome id from fasta filename
        genome_id = '_'.join(fasta_file.split('/')[-1].split('_')[:2])
        genomes_count[genome_id] += 1
        # get sequences
        rec_list = get_sequences(fasta_file)
        for rec in rec_list:
            dict_sequences[f'{genome_id}-{genomes_count[genome_id]}'].append(str(rec.seq))

    # mutate genomes
    if len(list_mutate) != 0:
        for fasta_file in list_mutate:
            # get genome id from fasta filename
            genome_id = '_'.join(fasta_file.split('/')[-1].split('_')[:2])
            genomes_count[genome_id] += 1
            # get sequences
            seq_list = get_sequences(fasta_file)
            for rec in seq_list:
                # call mutate function to mutate the sequence and generate reads
                mut_seq, mut_stats = \
                    mutate(str(rec.seq), label, rec.id, args.codon_amino, args.amino_codon)
                mut_f.write(f'{genome_id}\t{rec.id}\t{100 - mut_stats}\n')
                dict_sequences[f'{genome_id}-{genomes_count[genome_id]}'].append(mut_seq)

    # split genomes between training and testing sets
    total_genomes = list(dict_sequences.keys())
    num_train_genomes = math.ceil(0.7*len(total_genomes))
    random.shuffle(total_genomes)
    train_genomes = total_genomes[:num_train_genomes]
    test_genomes = total_genomes[num_train_genomes:]

    # report genomes in testing and training/validation sets
    with open(os.path.join(args.input_path, f'{label}-train-genomes'), 'w') as f:
        for g in train_genomes:
            f.write(f'{label}\t{g}\n')
    with open(os.path.join(args.input_path, f'{label}-test-genomes'), 'w') as f:
        for g in test_genomes:
            f.write(f'{label}\t{g}\n')

    # create training, validation and testing sets and simulate reads
    create_train_val_sets(args, label, train_genomes, dict_sequences)
    create_test_set(args, label, test_genomes, dict_sequences)

    return

def create_train_val_sets(args, label, list_genomes, dict_sequences):
    for genome in list_genomes:
        rec_fw_reads = []
        rec_rv_reads = []
        for seq in dict_sequences[genome]:
            # forward reads are simulated from the positive strand and reverse reads from the negative strands
            simulate_reads(label, genome, seq, complement(seq), rec_fw_reads, rec_rv_reads)
            # positive and negative strands are inversed
            simulate_reads(label, genome, complement(seq)[::-1], seq[::-1], rec_fw_reads, rec_rv_reads)
            print(len(rec_fw_reads), len(rec_rv_reads))
        # split reads into training and validation sets if constraints are met
        random.shuffle(rec_fw_reads)
        random.shuffle(rec_rv_reads)
        num_train_reads = math.ceil(0.7*len(rec_fw_reads))
        with open(os.path.join(args.input_path, f'{label}-train-reads.fq'), 'a') as outfile:
            outfile.write(''.join(rec_fw_reads[:num_train_reads]))
            outfile.write(''.join(rec_rv_reads[:num_train_reads]))
        with open(os.path.join(args.input_path, f'{label}-val-reads.fq'), 'a') as outfile:
            outfile.write(''.join(rec_fw_reads[num_train_reads:]))
            outfile.write(''.join(rec_rv_reads[num_train_reads:]))

    return

def create_test_set(args, label, list_genomes, dict_sequences):
    # test_reads = []
    for genome in list_genomes:
        rec_fw_reads = []
        rec_rv_reads = []
        for seq in dict_sequences[genome]:
            # forward reads are simulated from the positive strand and reverse reads from the negative strands
            simulate_reads(label, genome, seq, complement(seq), rec_fw_reads, rec_rv_reads)
            # positive and negative strands are inversed
            simulate_reads(label, genome, complement(seq)[::-1], seq[::-1], rec_fw_reads, rec_rv_reads)
        # test_reads += rec_fw_reads + rec_rv_reads
        with open(os.path.join(args.input_path, f'{label}-test-reads.fq'), 'a') as outfile:
            outfile.write(''.join(rec_fw_reads))
            outfile.write(''.join(rec_rv_reads))

    return


# This function breaks up a dict of sequences into 250 nucleotide segments
# later held in a list of strings
# Option is what reading frame you would like the reading frame to be
# Option 0 - 2 shifts the seq over respectively
# is_comp is a boolean variable that indicates if the user wants the positive and negative strings to be reversed
def simulate_reads(label, sequence_id, positive_strand, negative_strand, rec_forward_reads, rec_reverse_reads):
    """ simulate forward and reverse reads """
    # set distance between forward and reverse reads
    inner_distance = -100
    # set length of reads
    read_length = 250
    # simulate forward and reverse reads
    for i in range(0, len(positive_strand) - (inner_distance + read_length), read_length):
        fw_read = positive_strand[i:i + read_length]
        rv_read = negative_strand[i + read_length + inner_distance:i + read_length + inner_distance + read_length]
        if len(rv_read) == read_length and len(fw_read) == read_length:
            create_fastq_record(fw_read, f'{sequence_id}-{label}-{len(rec_forward_reads)}-1', rec_forward_reads)
            create_fastq_record(rv_read, f'{sequence_id}-{label}-{len(rec_reverse_reads)}-2', rec_reverse_reads)
    return

def mutate(seq, label, seq_id, codon_amino, amino_codon):
    """ returns a mutated sequence with synonymous mutations randomly added to every ORF """
    # randomly select one of the 3 reading frames
    rf_option = random.choice([0, 1, 2])
    # keep track of the number of point mutations
    counter = 0
    # create variable to store the mutated sequence
    mutated_sequence = ''
    # define a list with STOP codons
    list_stop_codons = ['TAA', 'TAG', 'TGA']
    last_add = len(seq) - (len(seq) % 3)
    i = 0
    while i <= (len(seq) - 3):
        codon = seq[i:i + 3]
        if codon == 'ATG':
            # sees if a stop codon exists
            is_orf, old_stop_codon = find_orf(seq, i, list_stop_codons)
            j = i
            if is_orf == 0:  # the loop for when a stop codon exists
                orf = ''
                while j < (len(seq) - 3) and seq[j:j + 3] not in list_stop_codons:
                    if bool(re.match('^[ACTG]+$', seq[j:j + 3])):
                        # replace by a synonymous codon
                        mutated_codon = select_codon(seq[j:j + 3], codon_amino, amino_codon)
                        orf += mutated_codon
                        # keep track of the number of point mutations
                        counter += mut_counter(mutated_codon, seq[j:j + 3])
                    # add original codon to mutated sequence if presence of non universal base
                    else:
                        orf += seq[j:j + 3]
                    j += 3
                # add randomly selected stop codons
                new_stop_codon = list_stop_codons[random.randint(0, 2)]
                orf += new_stop_codon
                counter += mut_counter(new_stop_codon, old_stop_codon)
                mutated_sequence += orf
            # when there is no stop codon
            elif is_orf == -1:
                while j < (len(seq) - 3):
                    mutated_sequence += seq[j:j + 3]
                    j += 3
            i = j + 3
        else:
            mutated_sequence += seq[i:i + 3]
            i += 3
    # adds on the last characters of the sequence if its length is not a multiple of 3
    mutated_sequence += seq[last_add:]

    return mutated_sequence, ((counter / len(seq)) * 100)


def select_genomes(args):
    # parses the genome file into a dictionary: keys are the species names and the values are a list of accession ids
    """ returns a dictionary with a list of genomes for each species """
    # load gtdb information
    gtdb_df = pd.read_csv(args.gtdb_info, delimiter='\t', low_memory=False)

    # get species in database
    species_in_database = [name[(name.find('s__') + 3):] for name in list(gtdb_df['gtdb_taxonomy'])]

    # retrieve information on genomes
    ncbi_assembly_level_list = list(gtdb_df.ncbi_assembly_level)
    ncbi_genome_category_list = list(gtdb_df.ncbi_genome_category)
    accession_id_list = list(gtdb_df.accession)

    # get genomes available in local NCBI and GTDB databases
    gtdb_genomes_avail = {'_'.join(i.split('/')[-1].split('_')[:2]) : i for i in glob.glob(os.path.join(args.gtdb_path, '*.fna.gz'))}
    ncbi_genomes_avail = {'_'.join(i.split('/')[-1].split('_')[:2]) : i for i in glob.glob(os.path.join(args.ncbi_path, '*.fna.gz'))}

    # filter genomes based on species in dataset and genomes type
    genome_dict = defaultdict(list)
    for i in range(len(gtdb_df)):
        if species_in_database[i] in args.list_species:
            if ncbi_assembly_level_list[i] == "Complete Genome" and ncbi_genome_category_list[i] \
                    != ("derived from metagenome" or "derived from environmental_sample"):
                # verify if genome fasta file is available in the local gtdb database
                if accession_id_list[i][3:] in gtdb_genomes_avail:
                    genome_dict[species_in_database[i]].append(os.path.join(args.gtdb_path, gtdb_genomes_avail[accession_id_list[i][3:]]))
                # verify if genome fasta file is available in the local ncbi database
                elif accession_id_list[i][3:] in ncbi_genomes_avail:
                    genome_dict[species_in_database[i]].append(os.path.join(args.ncbi_path, ncbi_genomes_avail[accession_id_list[i][3:]]))

    return genome_dict

def get_genomes_info(args, species, label):
    total_GC_content = float()
    TETRA_nt = defaultdict(int)
    for genome in args.genome_dict[species]:
        rec_list = get_sequences(fastafile)
        for rec in rec_list:
            """ compute the genome GC content: Count(G + C)/Count(A + T + G + C) * 100% """
            total_GC_content += (float((str(rec.seq).count('C') + str(rec.seq).count('G'))) / (str(rec.seq).count('C') + str(rec.seq).count('G') + str(rec.seq).count('A') + str(rec.seq).count('T'))) * 100
            get_tetra_nt_fqcy(TETRA_nt, str(rec.seq))
            print(total_GC_content, len(TETRA_nt))
            
    return total_GC_content/len(args.genome_dict[species]),
