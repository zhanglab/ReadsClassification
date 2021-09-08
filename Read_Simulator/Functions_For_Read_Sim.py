import json  # used to generate json
import random  # needed to generate random numbers
import re  # used to check characters
import string
from collections import defaultdict  # used to create genome dictionary
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import os

# This function takes a dictionary and makes it into json format
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def missing_genome(missing_set):
    with open('missing_genomes.txt', "a") as f:
        for i in missing_set:
            f.write(i)
            f.write('\n')


def json_dict(dictionary, filename):
    with open(filename, "w") as f:
        json.dump(dictionary, f)
    f.close()


# finds the number of iterations that need to be done for a species

def num_iterations(filename, string_to_file):
    with open(filename, 'a') as f:
        f.write(string_to_file)
        f.write('\n')



def get_seq(fasta_file, genome_id, NCBI_path):

    try:
        seq_list = exclude_plasmid(fasta_file)  # accession ids as keys and sequences as values
    except FileNotFoundError:
        seq_list = exclude_plasmid(os.path.join(NCBI_path, f'{genome_id}_*_genomic.fna.gz'))
    return seq_list






def generate_datasets(genome_dict, label_dict, codon_amino, amino_codon, NCBI_path):
    # load the genome fasta file (key = species_name, value = list of the fasta paths on aimos

    needed_iterations, largest_species = find_largest_genome_set(genome_dict)

    missing_set = set()

    reads_total = 0

    for label, species in label_dict.items():
        print(f'this is species: {species}')
        with open('mutation_report.txt', 'a') as f:
            f.write(f'{species}\t{label}\t')
        # create dictionaries to store reverse and forward reads
        rec_fw_read = []  # key = read id, value = read sequence
        rec_rv_read = []  # key = read id, value = read sequence
        # create dictionaries to store fasta sequences of mutated sequences

        # TODO: for AIMOS change k to 430

        # this is the portion that starts creating reads for the unmutated genomes

        working_genome_ids = []

        for fastafile in genome_dict[species]:
            # genome_id = fastafile[fastafile.find('GCF_'):fastafile.find('_g')]
            genome_id = '_'.join([fastafile.split('/')[-1].split('_')[0], fastafile.split('/')[-1].split('_')[1]])

            seq_list = get_seq(fastafile, genome_id)
            for rec in seq_list:
                # this portion is iterating through the fasta file and creating reads
                generate_reads(rec.id, label, rec.seq, complement(rec.seq), rec_fw_read, rec_rv_read)
                # writes a mutation report and since no mutations are done puts the mut_stats at 100
            with open('mutation_report.txt', 'a') as f:
                f.write(f'{genome_id}\t{rec.id}\t{100}\n')

                except FileNotFoundError:
                    missing_set.add(genome_id)
                    continue

        # TODO this does not work for the species that hase 430 genomes

        mutate_list = random.choices(working_genome_ids, k=(430 - len(working_genome_ids)))

        for fastafile in mutate_list:
            seq_list = exclude_plasmid(fastafile)  # accession ids as keys and sequences as values

            # TODO for amios change this
            # genome_id = fastafile[fastafile.find('GCF_'):fastafile.find('_g')]
            genome_id = '_'.join([fastafile.split('/')[-1].split('_')[0], fastafile.split('/')[-1].split('_')[1]])

            mut_records = []
            for rec in seq_list:
                # call mutate function to mutate the sequence and generate reads
                # (add reads to dict_rev_reads and dict_fw_reads)
                mut_seq, mut_stats = \
                    mutate(rec.seq, label, rec.id, codon_amino, amino_codon, rec_fw_read, rec_rv_read)
                mut_rec = SeqRecord(Seq(mut_seq), id=f'{rec.id}-mutated', name=rec.name, description=rec.description)
                mut_records.append(mut_rec)
                with open('mutation_report.txt', 'a') as f:
                    f.write(f'{genome_id}\t{rec.id}\t{100 - mut_stats}\n')

            SeqIO.write(mut_records, f'{genome_id}-mutated.fna', "fasta")
        SeqIO.write(rec_fw_read, f'{label}-fw-read.fq', "fastq")
        SeqIO.write(rec_rv_read, f'{label}-rv-read.fq', "fastq")
        missing_genome(missing_set)
        reads_total += len(rec_fw_read) + len(rec_rv_read)
        with open('summary_reads_report.txt', 'a') as f:
            f.write(f'{species}\t{len(rec_fw_read)}\t{len(rec_rv_read)}\t{len(rec_fw_read) + len(rec_rv_read)}\n')
    # generate fastq file for forward and reverse reads (separately)
    # add information about percentage of mutations to file mentioned above
    with open('summary_reads_report.txt', 'a') as f:
        f.write(f'{reads_total}')


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


# This function breaks up a dict of sequences into 250 nucleotide segments
# later held in a list of strings
# Option is what reading frame you would like the reading frame to be
# Option 0 - 2 shifts the seq over respectively
# is_comp is a boolean variable that indicates if the user wants the positive and negative strings to be reversed


def generate_reads(sequence_id, label, positive_strand, negative_strand, rec_forward_reads, rec_reverse_reads):
    inner_distance = -100
    read_length = 250

    for i in range(0, len(positive_strand) - (inner_distance + read_length), read_length):
        fw_read = positive_strand[i:i + read_length]
        rv_read = negative_strand[i + read_length + inner_distance:i + read_length + inner_distance + read_length]
        if len(rv_read) == read_length and len(fw_read) == read_length:
            create_fastq_record(fw_read, f'{sequence_id}-{label}-{len(rec_forward_reads)}-1', rec_forward_reads)
            create_fastq_record(rv_read, f'{sequence_id}-{label}-{len(rec_reverse_reads)}-2', rec_reverse_reads)


# reads dictionary of codons and mutates the open reading frames


def mutate(seq, label, seq_id, codon_amino, amino_codon, rec_fw_reads, rec_rv_reads):
    rf_option = random.choice([0, 1, 2])

    seq = list(seq)
    counter = 0
    mutated_sequence = []
    list_stop_codons = ['TAA', 'TAG', 'TGA']
    total = len(seq)
    last_add = total - (total % 3)
    i = 0
    while i <= (len(seq) - 3):
        codon = seq[i:i + 3]
        if ''.join(codon) == 'ATG':
            is_orf, old_stop_codon = find_orf(seq, i)  # sees if a stop codon exists
            j = i
            if is_orf == 0:  # the loop for when a stop codon exists
                orf = []
                while j < (len(seq) - 3) and (''.join(seq[j:j + 3]) not in list_stop_codons):
                    if bool(re.match('^[ACTG]+$', ''.join(seq[j:j + 3]))):
                        mutated_codon = random_select(seq[j:j + 3], codon_amino, amino_codon)
                        orf += mutated_codon
                        counter += mut_counter(mutated_codon, seq[j:j + 3])
                    else:
                        orf += seq[j:j + 3]
                    j += 3
                new_stop_codon = list_stop_codons[random.randint(0, 2)]
                orf += new_stop_codon
                counter += mut_counter(new_stop_codon, old_stop_codon)
                mutated_sequence += orf

            elif is_orf == -1:  # the loop for when there is no stop codon
                while j < (len(seq) - 3):
                    mutated_sequence += seq[j:j + 3]
                    j += 3
            i = j + 3
        else:
            mutated_sequence += seq[i:i + 3]
            i += 3
    mutated_sequence += seq[last_add:]  # adds on the last characters of the sequence
    positive_strand = ''.join(mutated_sequence)
    negative_strand = complement(positive_strand)

    # TODO: add generate read function

    is_comp = random.getrandbits(1)

    if is_comp is True:
        positive_strand = complement(positive_strand)
        negative_strand = complement(negative_strand)

    generate_reads(seq_id, label, positive_strand, negative_strand, rec_fw_reads, rec_rv_reads)
    # generate_reads(seq_id, label, negative_strand[::-1], positive_strand[::-1], rec_fw_reads, rec_rv_reads,
    # len(rec_fw_reads))

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


# TODO: see if this function is necessary, if so this function will have to be changed
#  to take a string rather than a dict


def write_to_fasta_mut_str(seq_dict, description):
    file = open('modifiedDNA.fna', 'a')  # opens the fasta file that will hold the mutated seq
    for seq_id, seq in seq_dict.items():  # iterates through the mutated dict
        sequence = ''
        count = 0  # iterating int to go through the seq in each dict entry
        while count < len(seq):
            sequence = sequence + '\n' + ''.join(seq[count: count + 60])
            count += 60
        full_str = '> ' + description[seq_id] + ' ' + sequence + '\n'  # the full str that will be be written to fasta
        file.write(full_str)  # writes to fasta
    file.close()  # closes file


# this function will be used to take the fw and rv dictionaries and write them to fasta files


def write_dict_to_fasta(fastqfile, read_dict):
    for label, read in read_dict.items():
        saveFasta = open(fastqfile, 'a')
        saveFasta.write('\n')
        saveFasta.write('>' + label)
        for i in range(0, (len(read) - 60), 60):
            saveFasta.write('\n')
            saveFasta.write(read[i:i + 60])
        saveFasta.close()


# parses the genome file into a dictionary: keys are the species names and the values are a list of accession ids

# TODO: change the way the path is coded
def parse_dataframe(genome_dataframe, species_dataframe, GTDB_path, NCBI_path):
    # species = key and value = list of species
    dataframe_dict = defaultdict(list)
    ncbi_assembly_level_list = list(genome_dataframe.ncbi_assembly_level)
    ncbi_genome_category_list = list(genome_dataframe.ncbi_genome_category)
    species_in_database = [name[(name.find('s__') + 3):] for name in list(genome_dataframe['gtdb_taxonomy'])]

    # TODO: for AiMOS the direction of the \ need to be changed to /

    accession_id_list = [(GTDB_path + genome_dataframe.accession + '_genomic.fna.gz')]

    accession_id_list = [path.replace('RS_', '') for path in accession_id_list]

    species_list = list(species_dataframe[0])

    for i in range(len(genome_dataframe)):
        if ncbi_assembly_level_list[i] == "Complete Genome" and ncbi_genome_category_list[i] \
                != ("derived from metagenome" or "derived from environmental_sample"):
            dataframe_dict[species_in_database[i]].append(accession_id_list[i])

    complete_species_dict = defaultdict(list)

    for j in species_list:
        if j in dataframe_dict:
            complete_species_dict[j] = dataframe_dict[j]

    print(complete_species_dict)

    return complete_species_dict


# this function finds the species with the most accession ids


def find_largest_genome_set(genome_dict):
    largest = 0
    largest_species = ''
    for species, accession_list in genome_dict.items():
        if len(accession_list) > largest:
            largest = len(accession_list)
            largest_species = species
    return largest, largest_species

def exclude_plasmid(fastafile):
    fasta_list = []
    with gzip.open(fastafile, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if (rec.description.find('plasmid') or rec.description.find('Plasmid')) == -1:
                fasta_list.append(rec)

    return fasta_list


# this function looks for a open reading frame and if one exists returns true
def find_orf(seq, i):
    stop_codon_list = ['TAA', 'TAG', 'TGA']
    while ''.join(seq[i:i + 3]) not in stop_codon_list:
        if i >= len(seq):
            is_mut = -1
            old_stop_codon = []
            return is_mut, old_stop_codon
        i += 3
        is_mut = 0
    old_stop_codon = seq[i:i + 3]
    return is_mut, old_stop_codon


# this function will get a dictionary of descriptions from a fastafile

def get_descriptions_fasta(fastafile):
    description_dict = {rec.id: rec.description for rec in SeqIO.parse(fastafile, 'fasta')
                        if (rec.description.find('plasmid') or rec.description.find('Plasmid')) == -1}
    return description_dict


def create_fastq_record(read_seq, read_id, list_records):
    read_qual = [random.choice(string.ascii_uppercase) for _ in range(len(read_seq))]
    record = SeqRecord(Seq(read_seq), id=read_id)
    quality_scores = [ord(i) for i in read_qual]
    record.letter_annotations["phred_quality"] = quality_scores
    record.format('fastq')
    list_records.append(record)
