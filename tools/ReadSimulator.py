import os
import gzip
from .TaxonomyGraph import *
import multiprocess as mp
import json
import re
import random
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from .utils import load_json_dictionary
from .kmer import parse_seq
from .TFRecords import write_TFRecord

# load codons and amino acids info
CODONS = load_json_dictionary('CodonsDict.json', '/'.join([os.getcwd(), 'ReadsClassification/tools']))
AAS = load_json_dictionary('AasDict.json', '/'.join([os.getcwd(), 'ReadsClassification/tools']))

def GetAntisenseStrand(DNAsequence):
    # Reverse string
    antisense = DNAsequence[::-1]
    # Make a mapping table for complementary bases
    complement = str.maketrans('ATCG', 'TAGC')
    return antisense.translate(complement)


def ModifyDNA(DNAsequence, start, end):
    newORF = str()
    for i in range(start + 3, end, 3):
        codon = DNAsequence[i:i + 3]
        # check codon for other characters than a, c, t or g
        if not bool(re.match('^[ACTG]+$', codon)):
            newORF += codon
            continue
        else:
            # Get corresponding aminoacid
            aa = CODONS[codon]
            # Get list of synonymous codons
            syncodons = AAS[aa]
            # shuffle list of synonymous codons and chose one codon
            random.shuffle(syncodons)
            randomNum = random.choice(range(len(syncodons)))
            ChosenCodon = syncodons[randomNum]
            # add codon to new orf
            newORF += ChosenCodon
    # replace orf in sequence
    DNAsequence = DNAsequence[0:start + 3] + newORF + DNAsequence[end:len(DNAsequence)]
    return DNAsequence


def FindORFs(DNAsequence):
    new_DNA = DNAsequence
    orf = str()
    start = int()
    for i in range(0, len(DNAsequence) - 3, 3):
        codon = DNAsequence[i:i + 3]
        # check codon for other characters than a, c, t or g
        if bool(re.match('^[ACTG]+$', codon)):
            continue
        # check if codon is start codon
        if CODONS[codon] == 'Met' and len(orf) == 0:
            start = i
            orf += codon
        # modify any orf in DNA sequence
        elif CODONS[codon] == 'Stop' and len(orf) >= 3:
            end = i
            orf += codon
            new_DNA = ModifyDNA(new_DNA, start, end)
            orf = str()
            start = int()
            # start reading after the stop codon
            i += 3
        # add next codons to orf
        if len(orf) >= 3:
            orf += codon
    return new_DNA

def GetPathToFastqFile(genomeID):
    genomeID = genomeID[3:]
    for root, dirs, files in os.walk('/project/projectdirs/m3513/srp/ncbi/genomes/all/'):
        for file in files:
            if file == '_'.join([genomeID, 'genomic.fna.gz']):
                return os.path.join(root, file)


def ParseFastaFile(fasta_file):
    with gzip.open(fasta_file, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            if 'plasmid' not in str(record.id) or 'Plasmid' not in str(record.id):
                return str(record.seq)


def ProcessDNAseq(data_dict_train, data_dict_val, data_dict_test, class_mapping, label, dict_genomes, args):
    print('Label processed: {}'.format(label))
    # list to store the reads
    data = []
    # get full path to genome fasta file
    fasta_file = GetPathToFastqFile(dict_genomes[class_mapping[label]][0])
    # get concatenated DNA sequences in fasta file
    sense_strand = ParseFastaFile(fasta_file)
    # DNA sequences
    templates = [sense_strand]
    # get antisense strand
    #antisense_strand = GetAntisenseStrand(sense_strand)
    # generate reads from sense and antisense strands
    #templates = [sense_strand, antisense_strand]

    # Calculate the number of copies of genomes to get in order to have a number of reads >= tio the number of reads desired
    num_reads_estimated = len(sense_strand) // 150
    while num_reads_estimated <= args.readsnum:
        # generate data if not enough reads
        newDNA = FindORFs(sense_strand)
        templates.append(newDNA)
        num_reads_estimated += len(sense_strand) // 150

    # simulate reads
    for seq in templates:
        for n in range(0, len(seq) - 150, 150):
            read = seq[n:n + 150]
            if re.match('^[ATCG]+$', read):
                integer_encoded = parse_seq(read, args)
                if len(integer_encoded) == args.length:
                    read_data = [integer_encoded, label]
                    data.append(read_data)

    # split reads into train, val and test sets
    random.shuffle(data)
    num_reads_test = int(0.3 * len(data))
    data_dict_test[label] = data[:num_reads_test]
    train_data = data[num_reads_test:]
    num_reads_train = int(0.7 * len(train_data))
    data_dict_val[label] = train_data[num_reads_train:]
    data_dict_train[label] = train_data[:num_reads_train]

def MP_DistributeGenomes(class_mapping, dict_genomes, args):
    with mp.Manager() as manager:
        # Create a dict in server process memory
        data_dict_train = manager.dict()
        data_dict_test = manager.dict()
        data_dict_val = manager.dict()
        # Create new processes
        processes = [mp.Process(target=ProcessDNAseq, args=(data_dict_train, data_dict_val, data_dict_test, class_mapping, label, dict_genomes, args)) for label in class_mapping]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # combine data
        train_reads, train_labels = combine_data(data_dict_train, 'train')
        test_reads, test_labels = combine_data(data_dict_test, 'test')
        val_reads, val_labels = combine_data(data_dict_val, 'val')

        # create TFRecords
        write_TFRecord(train_reads, train_labels, args, 'train', len(class_mapping))
        write_TFRecord(test_reads, test_labels, args, 'test', len(class_mapping))
        write_TFRecord(val_reads, val_labels, args, 'val', len(class_mapping))


def combine_data(dict_data, set_type):
    dataset = []
    with open(os.path.join(args.output, 'reads.txt'), 'a') as f:
        f.write('Set: {}'.format(set_type))
        for label in dict_data.keys():
            dataset += dict_data[label]
            f.write('Label: {} - number of reads: {}'.format(label, len(dict_data[label])))
        f.write('Total number of reads: {}'.format(len(dataset)))
    # separate reads and labels
    random.shuffle(dataset)
    reads = np.asarray([i[0] for i in dataset], dtype=np.int32)
    labels = np.asarray([int(i[1]) for i in dataset], dtype=np.int32)
    return reads, labels
