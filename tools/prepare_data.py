import numpy as np
import os
import sys
import multiprocess as mp
import random
import math
from Bio import SeqIO
import re
import json

# Creates a json file of the dictionaries
def create_json(dict, output):
    # Creation of the class_mapping file
    filename = os.path.join(output, 'class_mapping.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

# Function that parse the Fastq File and process the reads
def parse_fastq(train_data_dict, test_data_dict, fastq_files, label, args):
    data = []
    # Create a dictionary like object to store information about the reads
    reads_dict = SeqIO.index(fastq_files, 'fastq')
    # Get reads IDs in a list
    listReadIDs = list(reads_dict.keys())
    random.shuffle(listReadIDs)

    def unidirectional(listReadIDs):
        if args.model == 'kmer':
            from .kmer import parse_seq
        else:
            from .baseemb import parse_seq

        listReadIDs = listReadIDs[:50000]
        # Check read sequence and get one-hot encoded sequence
        for i in range(len(listReadIDs)):
            read_id = listReadIDs[i]
            seq_record = reads_dict[read_id]
            if re.match('^[ATCG]+$', str(seq_record.seq)):
                integer_encoded = parse_seq(str(seq_record.seq), args)
                if len(integer_encoded) == args.length:
                    data.append([integer_encoded, label])

    def bidirectional(listReadIDs):
        from .kmer import parse_seq
        for i in range(0, len(listReadIDs), 2):
            fw_read_id = listReadIDs[i]
            rv_read_id = listReadIDs[i + 1]
            fw_seq_record = reads_dict[fw_read_id]
            rv_seq_record = reads_dict[rv_read_id]
            fw_KmerVector = parse_seq(str(fw_seq_record.seq), args)
            rv_KmerVector = parse_seq(str(rv_seq_record.seq), args)
            if len(fw_KmerVector) == args.length and len(rv_KmerVector) == args.length:
                data.append([fw_KmerVector, rv_KmerVector, label])

    bidirectional(listReadIDs) if args.model == 'bidirectional' else unidirectional(listReadIDs)

    # Split data into train and test sets
    NumReadsTrain = int(math.ceil(0.7 * len(data)))
    training_data = data[:NumReadsTrain]
    testing_data = data[NumReadsTrain:]

    # add data to dictionaries
    train_data_dict[label] = training_data
    test_data_dict[label] = testing_data

# Function that gets the full path to the fq file
def path_to_fq_file(genomeID, args):
    currentdir = args.input
    for root, dirs, files in os.walk('/'.join([currentdir, genomeID])):
        for file in files:
            if file == 'anonymous_reads.fq':
                return os.path.join(root, file)

# Function that creates the dataset of simulated reads from all the fastq files available
def get_info(args):
    fastq_files = {}
    class_mapping = {}
    with open(args.input + '/Species.tsv', 'r') as info:
        for class_num, line in enumerate(info):
            line = line.strip('\n')
            columns = line.split('\t')
            species = columns[0]
            genomeID = columns[2]
            # Add Class to class_mapping dictionary
            class_mapping[class_num] = species
            fastq_files[class_num] = path_to_fq_file(genomeID, args)
    print('Dictionary mapping Classes to integers: {}'.format(class_mapping))
    # Create json file of class_mapping
    create_json(class_mapping, args.output)
    return fastq_files

def multiprocesses(fastq_files, args):
    with mp.Manager() as manager:
        # Create a list in server process memory
        test_data_dict = manager.dict()
        train_data_dict = manager.dict()
        # Create new processes
        processes = [mp.Process(target=parse_fastq, args=(
        train_data_dict, test_data_dict, fastq_files[x], x, args)) for x in range(len(fastq_files))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        create_npy(train_data_dict, 'train', args)
        create_npy(test_data_dict, 'test', args)

# Function that creates npy files
def create_npy(dict, set_type, args):
    data = np.asarray(dict[0])
    numberReads = len(dict[0])
    for i in range(1, len(dict)):
        numberReads += len(dict[i])
        data = np.concatenate((data, np.asarray(dict[i])), axis=0)

    # save data
    if set_type == 'train':
        np.random.shuffle(data)
        num_reads_train = int(0.7 * len(data))
        traindata = data[:num_reads_train]
        valdata = data[num_reads_train:]
        print('Number of reads in whole training dataset: {}'.format(len(data)), file=sys.stderr)
        print('Number of reads in training set: {}'.format(len(traindata)), file=sys.stderr)
        print('Number of reads in validation set: {}'.format(len(valdata)), file=sys.stderr)
        np.save(args.output + '/train_data_{0}.npy'.format(args.model), traindata)
        np.save(args.output + '/val_data_{0}.npy'.format(args.model), valdata)

    elif set_type == 'test':
        print('Number of reads in testing set: {}'.format(len(data)), file=sys.stderr)
        np.save(args.output + '/test_data_{0}.npy'.format(args.model), data)