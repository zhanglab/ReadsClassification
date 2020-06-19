#!/usr/bin/env python
# coding: utf-8

# Import all necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import sys
import multiprocess as mp
import random
import math
from Bio import SeqIO
import re
import json
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

base_dict = {'A':0, 'C':1, 'T':2, 'G':3}
# Creates a json file of the dictionaries
def CreateJSON(dict):
    # Creation of the class_mapping file 
    filename = 'class_mapping.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

# Function that looks for any characters different
# from A, C, T or G and converts the DNA sequence into one-hot encoded sequence
def ParseSeq(DNAsequence):
    # Drop the read if it contains letters other than A, C, T or G
    if not re.match('^[ATCG]+$', DNAsequence):
        return None
    integer_encoded = [base_dict[base] for base in DNAsequence]
    return integer_encoded

# Function that parse the Fastq File and process the reads
def ParseFastq(DictTrainData, DictTestData, FastqFile, label):
    data = []
    # Create a dictionary like object to store information about the reads
    Reads_dict = SeqIO.index(FastqFile, 'fastq')
    # Get reads IDs in a list
    listReadIDs = list(Reads_dict.keys())
    print(listReadIDs[0])
    random.shuffle(listReadIDs)
    print(listReadIDs[0])
    listReadIDs = listReadIDs[:50000]
    # Check read sequence and get one-hot encoded sequence
    for i in range(len(listReadIDs)):
        read_id = listReadIDs[i]
        seq_record = Reads_dict[read_id]
        integer_encoded = ParseSeq(str(seq_record.seq))
        if len(integer_encoded) == 150:
            data.append([integer_encoded, label])

    print('Number of reads in fastq file: {0} - {1}'.format(FastqFile, len(listReadIDs)))
    # Split data into train and test sets
    NumReadsTrain = int(math.ceil(0.7 * len(data)))
    training_data = data[:NumReadsTrain]
    testing_data = data[NumReadsTrain:]
    print('Reads for training: {}'.format(len(training_data)))
    print('Reads for testing: {}'.format(len(testing_data)))
    # add data to dictionaries
    DictTrainData[label] = training_data
    DictTestData[label] = testing_data

# Function that gets the full path to the fq file
def GetPathToFqFile(genomeID):
    currentdir = os.getcwd()
    for root, dirs, files in os.walk('/'.join([currentdir, genomeID])):
        for file in files:
            if file == 'anonymous_reads.fq':
                return os.path.join(root, file)


# Function that creates the dataset of simulated reads from all the fastq files available
def GetSetInfo():
    DictFastqFiles = {}
    class_mapping = {}
    with open(os.getcwd() + '/Species.tsv', 'r') as info:
        for numClasses, line in enumerate(info):
            line = line.strip('\n')
            columns = line.split('\t')
            species = columns[0]
            genomeID = columns[2]
            # Add Class to class_mapping dictionary
            class_mapping[numClasses] = species
            DictFastqFiles[numClasses] = GetPathToFqFile(genomeID)
    print('DictFastqFiles: {}'.format(DictFastqFiles))
    print('Dictionary mapping Classes to integers: {}'.format(class_mapping))
    # Create json file of class_mapping
    CreateJSON(class_mapping)
    return DictFastqFiles, class_mapping


def MultiProcesses(DictFastqFiles, class_mapping, num_classes):
    with mp.Manager() as manager:
        # Create a list in server process memory
        DictTestData = manager.dict()
        DictTrainData = manager.dict()
        # Create new processes
        processes = [mp.Process(target=ParseFastq, args=(
        DictTrainData, DictTestData, DictFastqFiles[x], x)) for x in range(len(DictFastqFiles))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print('Number of reads for each species and set: ')
        for key, value in DictTestData.items():
            print(key, len(value))
        for key, value in DictTrainData.items():
            print(key, len(value))
        CreateFiles(DictTrainData, 'train', class_mapping, num_classes)
        CreateFiles(DictTestData, 'test', class_mapping, num_classes)


# Function that creates npy files
def CreateFiles(Dict, SetType, class_mapping, num_classes):
    data = np.asarray(Dict[0])
    print('Label 0 - {0} - number of reads: {1} - Label in: {2}'.format(class_mapping[0], len(Dict[0]), Dict[0][1]))
    numberReads = len(Dict[0])
    for i in range(1, len(Dict)):
        print('Label {0} - {1} - number of reads: {2} - Label in: {3}'.format(i, class_mapping[i], len(Dict[i]), Dict[i][0][1]))
        numberReads += len(Dict[i])
        data = np.concatenate((data, np.asarray(Dict[i])), axis=0)
    print('number of reads: {0} - Set type: {1}'.format(numberReads, SetType))
    # save data
    if SetType == 'train':
        # np.random.shuffle(data)
        NumReadsTrain = int(0.7 * len(data))
        traindata = data[:NumReadsTrain]
        valdata = data[NumReadsTrain:]
        print('Number of reads in whole training dataset: {}'.format(len(data)), file=sys.stderr)
        print('Number of reads in training set: {}'.format(len(traindata)), file=sys.stderr)
        print('Number of reads in validation set: {}'.format(len(valdata)), file=sys.stderr)
        np.save(os.getcwd() + '/data/train_data_50000_{0}classes.npy'.format(num_classes), traindata)
        np.save(os.getcwd() + '/data/val_data_50000_{0}classes.npy'.format(num_classes), valdata)
    elif SetType == 'test':
        print('Number of reads in testing set: {}'.format(len(data)), file=sys.stderr)
        np.save(os.getcwd() + '/data/test_data_50000_{0}classes.npy'.format(num_classes), data)

def parse_args():
    #### SET MODEL PARAMETERS #####
    parser = argparse.ArgumentParser()
    parser.add_argument("-classes", type=int, help="number of classes", required=True)
    args = parser.parse_args()
    return args.classes

def main():
    num_classes = parse_args()
    # get list of species in file directory to reads
    DictFastqFiles, class_mapping = GetSetInfo()
    # Create dataset
    MultiProcesses(DictFastqFiles, class_mapping, num_classes)

if __name__ == "__main__":
    main()