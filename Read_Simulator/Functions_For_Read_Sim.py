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

def generate_dataset(args, label, needed_iterations):
    """ central function that calls the functions necessary to mutate genomes and generate reads  """
    # create file to store mutation mutation report
    mut_f = open(os.path.join(args.input_path, f'{label}_mutation_report.txt'), 'w')
    # create file to store number of read_seq
    reads_f = open(os.path.join(args.input_path, f'{label}_summary_reads_report.txt'), 'w')
    # create dictionaries to store reverse and forward reads
    rec_fw_read = []  # key = read id, value = read sequence
    rec_rv_read = []  # key = read id, value = read sequence

    # generate reads from original non mutated genomes
    for fastafile in args.genome_dict[species]:
        genome_id = '_'.join(fastafile.split('/')[-1].split('_')[:2])
        # get sequences
        seq_list = get_sequences(fastafile)
        for rec in seq_list:
            # this portion is iterating through the fasta file and creating reads
            # forward reads are simulated from the positive strand and reverse reads from the negative strands
            generate_reads(rec.id, label, rec.seq, complement(rec.seq), rec_fw_read, rec_rv_read)
            # positive and negative strands are inversed
            generate_reads(rec.id, label, complement(rec.seq)[::-1], rec.seq[::-1], rec_fw_read, rec_rv_read)
            # writes a mutation report and since no mutations are done puts the mut_stats at 100
            mut_f.write(f'{genome_id}\t{rec.id}\t{100}\n')

    # get list of randomly selected genomes that will be mutated
    mutate_list = random.choices(args.genome_dict[species], k=(needed_iterations - len(args.genome_dict[species])))
    # generate reads from mutated genomes
    if len(mutate_list) != 0:
        mut_records = []
        for count, fasta_file in enumerate(mutate_list):
            # get sequences
            seq_list = get_sequences(fastafile)
            for rec in seq_list:
                # call mutate function to mutate the sequence and generate reads
                mut_seq, mut_stats = \
                    mutate(rec.seq, label, rec.id, args.codon_amino, args.amino_codon, rec_fw_read, rec_rv_read)
                # forward reads are simulated from the positive strand and reverse reads from the negative strands
                generate_reads(rec.id, label, mut_seq, complement(mut_seq), rec_fw_read, rec_rv_read)
                # positive and negative strands are inversed
                generate_reads(rec.id, label, complement(mut_seq)[::-1], mut_seq[::-1], rec_fw_read, rec_rv_read)
                # create record for fasta file
                mut_rec = SeqRecord(Seq(mut_seq), id=f'{rec.id}-mutated', name=rec.name, description=rec.description)
                mut_records.append(mut_rec)
                mut_f.write(f'{genome_id}\t{rec.id}\t{100 - mut_stats}\n')

        # write mutated genome to fasta file
        SeqIO.write(mut_records, os.path.join(args.input_path, f'{label}-{genome_id}-mutated-{count}.fna'), "fasta")

    # write reads to fastq file
    SeqIO.write(rec_fw_read, os.path.join(args.input_path, f'{label}-fw-read.fq'), "fastq")
    SeqIO.write(rec_rv_read, os.path.join(args.input_path, f'{label}-rv-read.fq'), "fastq")
    reads_f.write(f'{species}\t{len(rec_fw_read)}\t{len(rec_rv_read)}\t{len(rec_fw_read) + len(rec_rv_read)}\n')


# This function breaks up a dict of sequences into 250 nucleotide segments
# later held in a list of strings
# Option is what reading frame you would like the reading frame to be
# Option 0 - 2 shifts the seq over respectively
# is_comp is a boolean variable that indicates if the user wants the positive and negative strings to be reversed
def generate_reads(sequence_id, label, positive_strand, negative_strand, rec_forward_reads, rec_reverse_reads):
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


def mutate(seq, label, seq_id, codon_amino, amino_codon, rec_fw_reads, rec_rv_reads):
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
                orf = []
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
                    genome_dict[species_in_database[i]].append(os.path.join(args.ncbi_path, gtdb_genomes_avail[accession_id_list[i][3:]]))

    return genome_dict
