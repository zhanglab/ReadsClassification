import numpy as np
import os
import sys
import multiprocessing as mp
import random
import math
from Bio import SeqIO
import re
import json
import csv
import h5py

# Creates a json file of the dictionaries
def create_class_mapping(class_dict, output, hdf5):
    # Open HDF5 file
    with h5py.File(os.path.join(output, hdf5), 'a') as hf:
        hf.create_dataset('class_mapping', data=str(class_dict))

# Function that parse the Fastq File and process the reads
def parse_fastq(train_data_dict, test_data_dict, fastq_files, label, args):
    forward, reverse = [], []
    # Create a dictionary like object to store information about the reads
    reads_dict = SeqIO.index(fastq_files, 'fastq')
    # Get reads IDs in a list
    listReadIDs = list(reads_dict.keys())
    random.shuffle(listReadIDs)

    def unpaired():
        if args.model == 'baseemb':
            from .baseemb import parse_seq
        else:
            from .kmer import parse_seq

        # Check read sequence and get one-hot encoded sequence
        for i in range(len(listReadIDs)):
            read_id = listReadIDs[i]
            seq_record = reads_dict[read_id]
            if re.match('^[ATCG]+$', str(seq_record.seq)):
                integer_encoded = parse_seq(str(seq_record.seq), args)
                if len(integer_encoded) == args.length:
                    forward.append(integer_encoded)

    def paired():
        from .kmer import parse_seq
        for i in range(0, len(listReadIDs), 2):
            fw_read_id = listReadIDs[i]
            rv_read_id = listReadIDs[i + 1]
            fw_seq_record = reads_dict[fw_read_id]
            rv_seq_record = reads_dict[rv_read_id]
            if re.match('^[ATCG]+$', str(fw_seq_record.seq)) and re.match('^[ATCG]+$', str(rv_seq_record.seq)):
                fw_KmerVector = parse_seq(str(fw_seq_record.seq), args)
                rv_KmerVector = parse_seq(str(rv_seq_record.seq), args)
                if len(fw_KmerVector) == args.length and len(rv_KmerVector) == args.length:
                    forward.append(fw_KmerVector)
                    reverse.append(rv_KmerVector)

    paired() if args.reads == 'paired' else unpaired()

    # Split data into train and test sets
    NumReadsTrain = int(math.ceil(0.7 * len(forward)))

    # add data to dictionaries
    train_data_dict['labels'] = train_data_dict['labels'] + ([label] * NumReadsTrain)
    train_data_dict['forward'] = train_data_dict['forward'] + forward[:NumReadsTrain]
    test_data_dict['labels'] = test_data_dict['labels'] + ([label] * (len(forward) - NumReadsTrain))
    test_data_dict['forward'] = test_data_dict['forward'] + forward[NumReadsTrain:]

    if args.reads == 'paired':
        train_data_dict['reverse'] = train_data_dict['reverse'] + reverse[:NumReadsTrain]
        test_data_dict['reverse'] = test_data_dict['reverse'] + reverse[NumReadsTrain:]

# Function that gets the full path to the fq file
def path_to_fq_file(genomeID, currentdir):
    for root, dirs, files in os.walk(os.path.join(currentdir, genomeID)):
        if os.path.exists(os.path.join(root, 'anonymous_reads.fq')):
            return os.path.join(root, 'anonymous_reads.fq')

# Function that creates the dataset of simulated reads from all the fastq files available
def get_info(args):
    fastq_files = {}
    class_mapping = {}
    with open(os.path.join(args.input, 'Species.tsv')) as info:
        reader = csv.reader(info, delimiter='\t')
        for class_num, row in enumerate(reader):
            species, genomeID = row[0], row[2]
            # Add Class to class_mapping dictionary
            class_mapping[class_num] = species
            fastq_files[class_num] = path_to_fq_file(genomeID, args.input)

    with open(os.path.join(args.output, 'reads.txt'), 'w+') as f:
        f.write('Dictionary mapping Classes to integers: {}\n'.format(class_mapping))

    # Create hdf5 dataset of class_mapping
    create_class_mapping(class_mapping, args.output, args.hdf5)
    return fastq_files, len(class_mapping)

def multiprocesses(fastq_files, args):
    with mp.Manager() as manager:
        # Create a dict in server process memory
        test_data_dict = manager.dict()
        train_data_dict = manager.dict()

        # Initialize the dict values as lists
        keys = ['labels', 'forward']
        for key in keys:
            test_data_dict[key] = []
            train_data_dict[key] = []
        if args.reads == 'paired':
            test_data_dict['reverse'] = []
            train_data_dict['reverse'] = []

        # Create new processes
        processes = [mp.Process(target=parse_fastq, args=(
        train_data_dict, test_data_dict, fastq_files[x], x, args)) for x in range(len(fastq_files))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Shuffle, separate and create hdf5 files
        train_data_dict, val_data_dict = shuffle_data(train_data_dict, args.output. args.reads)
        create_hdf5(train_data_dict, 'train', args)
        create_hdf5(val_data_dict, 'val', args)

        with open(os.path.join(args.output, 'reads.txt'), 'a+') as f:
            f.write('Number of reads in testing set: {}'.format(len(test_data_dict['labels'])))
        create_hdf5(test_data_dict, 'test', args)

def shuffle_data(data, output, reads):
    # Zip the dictionary lists together.
    # Ex: {'labels': [0,0], 'forward': [34,56], 'reverse': [90,45]} --> [[0,34,90], [0,56,45], [0,10,1]]
    zipped_data = [list(value) for value in zip(*data.values())]
    random.shuffle(zipped_data)
    num_reads_train = int(0.7 * len(zipped_data))
    traindata = zipped_data[:num_reads_train]
    valdata = zipped_data[num_reads_train:]
    with open(os.path.join(output, 'reads.txt'), 'a+') as f:
        f.write('Number of reads in whole training dataset: {}\n'.format(len(zipped_data)))
        f.write('Number of reads in training set: {}\n'.format(len(traindata)))
        f.write('Number of reads in validation set: {}\n'.format(len(valdata)))

    train_data_dict = {}
    val_data_dict = {}
    separated_train_data = map(list, zip(*traindata))
    separated_val_data = map(list, zip(*valdata))
    if reads == 'paired':
        train_data_dict['labels'], train_data_dict['forward'], train_data_dict['reverse'] = separated_train_data
        val_data_dict['labels'], val_data_dict['forward'], val_data_dict['reverse'] = separated_val_data
    else:
        train_data_dict['labels'], train_data_dict['forward'] = separated_train_data
        val_data_dict['labels'], val_data_dict['forward'] = separated_val_data

    return train_data_dict, val_data_dict

def create_hdf5(data, set_type, args):
    # Open HDF5 file
    with h5py.File(os.path.join(args.output, args.hdf5), 'a') as hf:
        hf.create_dataset(set_type + "_labels", data=np.asarray(data['labels'], dtype=np.int32),
                          dtype=np.int32, chunks=args.chunks)
        hf.create_dataset(set_type, data=np.asarray(data['forward'], dtype=np.int32),
                          dtype=np.int32, chunks=args.chunks)
        # Add training and validation paired reads
        if args.reads == 'paired' and set_type != 'testing':
            hf.create_dataset(set_type + "_paired", data=np.asarray(data['reverse'], dtype=np.int32),
                              dtype=np.int32, chunks=args.chunks)