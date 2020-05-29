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
from operator import add
import datetime
from Bio import SeqIO
import gzip
import re
import collections
from collections import defaultdict
from contextlib import redirect_stdout
import csv
import json
import itertools

# Creates a json file of the dictionaries
def CreateJSON(dict, k_value):
    # Creation of the class_mapping file or the kmer file
    filename = '{0}mer_dict.json'.format(k_value)
    if k_value == None:
        filename = 'class_mapping.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

# Function that returns a dictionary of all possible kmers with 
# each kmer associated with an integer
def GetKmersDictionary(k_value):
    list_nt = ['A', 'T', 'C', 'G']
    # Get list of all possible 4**k_value kmers
    list_kmers = [''.join(p) for p in itertools.product(list_nt, repeat=k_value)]
    # Assign an integer to each kmer
    kmers_dict = dict(zip(list_kmers, range(len(list_kmers))))
    # Create a json file of the dictionary
    CreateJSON(kmers_dict, k_value)
    print('Size of kmers dictionary: {}'.format(len(kmers_dict)))
    return kmers_dict

# Function that looks for any characters different 
# from A, C, T or G and converts the DNA sequence into a vector of 10-mers 
def ParseSeq(DNAsequence, kmers_dict, k_value, seq_length):
    # create empty list to store all the kmers
    listKmers = []
    # Drop the read if it contains letters other than A, C, T or G
    if not re.match('^[ATCG]+$', DNAsequence):
        return listKmers

    # Creates a sliding window of width 10
    # 141 kmers
    #    for n in range(len(DNAsequence) - 9):
    #        kmer = DNAsequence[n:n+10]
    # 29 kmers --> 5
    # 15 kmers --> 10
    # 71 kmers --> 2
    numkmers = 0
    for n in range(0, len(DNAsequence) - (k_value - 1), k_value):
        if numkmers < seq_length:
            kmer = DNAsequence[n:n+k_value]
            # Lookup integer mapped to the kmer
            kmer_Integer = kmers_dict[kmer]
            # Add kmer to vector of kmer
            listKmers.append(kmer_Integer)
            numkmers += 1
    print('Number of kmers: {0} - {1}'.format(len(listKmers), numkmers))
    # Pad or truncate list to 141 kmers
    #    listKmers = listKmers[:141] + [0]*(141 - len(listKmers))
    # transform list into numpy array
    array_int = np.asarray(listKmers)
    # Flip array
    array_int = np.flip(array_int, axis=0)
    return array_int

# Function that parse the Fastq File and process the reads 
def ParseFastq(DictTrainData, DictTestData, FastqFile, label, class_mapping, kmers_dict, k_value, seq_length):
    data = []
    # Create a dictionary like object to store information about the reads
    Reads_dict = SeqIO.index(FastqFile, 'fastq')
    # Get reads IDs in a list
    listReadIDs = list(Reads_dict.keys())
    print(listReadIDs[0])
    random.shuffle(listReadIDs)
    print(listReadIDs[0])
    listReadIDs = listReadIDs[:50000]
    # Check read sequence and get vector of kmers
    for i in range(len(listReadIDs)):
         read_id = listReadIDs[i]
         seq_record = Reads_dict[read_id]
         KmerVector = ParseSeq(str(seq_record.seq), kmers_dict, k_value, seq_length)
         if len(KmerVector) == seq_length:
             # Get onehot encoding --> not to use for stratified cross validation
             #   onehotvector = OneHotEncoding(label, len(class_mapping))
             # add read to data
             #   data.append([KmerVector, onehotvector])
             data.append([KmerVector, label]) 

    print('Number of reads in fastq file: {0} - {1}'.format(FastqFile,len(listReadIDs))
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
    for root, dirs, files in os.walk('/'.join([currentdir,genomeID])):
        for file in files:
            if file == 'anonymous_reads.fq':
                print(os.path.join(root,file))
                return os.path.join(root,file)

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
    listspecies = list(class_mapping.values())
    print('DictFastqFiles: {}'.format(DictFastqFiles))
    print('Dictionary mapping Classes to integers: {}'.format(class_mapping))
    # Create json file of class_mapping
    CreateJSON(class_mapping, None)
    return DictFastqFiles, class_mapping, listspecies

def MultiProcesses(DictFastqFiles, class_mapping, kmers_dict, k_value, seq_length, num_classes):
    with mp.Manager() as manager:
        # Create a list in server process memory
        DictTestData = manager.dict()
        DictTrainData = manager.dict()
        # Create new processes
        processes = [mp.Process(target=ParseFastq,args=(DictTrainData, DictTestData, DictFastqFiles[x], x, class_mapping, kmers_dict, k_value, seq_length)) for x in range(len(DictFastqFiles))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print('Number of reads for each species and set: ')
        for key, value in DictTestData.items():
            print(key, len(value))
        for key, value in DictTrainData.items():
            print(key, len(value))
        CreateFiles(DictTrainData,'train',class_mapping,k_value,seq_length,num_classes)
        CreateFiles(DictTestData,'test',class_mapping,k_value,seq_length,num_classes)
        return DictTestData, DictTrainData

# Function that creates npy files
def CreateFiles(Dict,SetType,class_mapping,k_value,seq_length,num_classes):
    data = np.asarray(Dict[0])
    print('Label 0 - {0} - number of reads: {1} - Label in: {2}'.format(class_mapping[0],len(Dict[0]),Dict[0][1]))
    numberReads = len(Dict[0])
    for i in range(1,len(Dict)):
        print('Label {0} - {1} - number of reads: {2} - Label in: {3}'.format(i,class_mapping[i],len(Dict[i]),Dict[i][0][1]))
        numberReads += len(Dict[i])
        data = np.concatenate((data,np.asarray(Dict[i])),axis=0)
    print(len(data))
    print('number of reads: {0} - Set type: {1}'.format(numberReads,SetType))    
    # save data
    if SetType == 'train':
        #np.random.shuffle(data)
        NumReadsTrain = int(0.7*len(data))
        traindata = data[:NumReadsTrain]
        valdata = data[NumReadsTrain:]
        print('Number of reads in whole training dataset: {}'.format(len(data)), file=sys.stderr)
        print('Number of reads in training set: {}'.format(len(traindata)), file=sys.stderr)
        print('Number of reads in validation set: {}'.format(len(valdata)), file=sys.stderr)
        np.save(os.getcwd() + '/data/train_data_50000_{0}kmers_k{1}-{2}classes.npy'.format(seq_length,k_value,num_classes), traindata)
        np.save(os.getcwd() + '/data/val_data_50000_{0}kmers_k{1}-{2}classes.npy'.format(seq_length,k_value,num_classes), valdata)
    elif SetType == 'test':
        print('Number of reads in testing set: {}'.format(len(data)), file=sys.stderr)
        np.save(os.getcwd() + '/data/test_data_50000_{0}kmers_k{1}-{2}classes.npy'.format(seq_length,k_value,num_classes), data) 

def main():
    #### SET MODEL PARAMETERS #####
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, help="size of kmer", required=True)
    parser.add_argument("-classes", type=int, help="number of classes", required=True)
    parser.add_argument("-len", type=int, help="size of vector", required=True)
    args = parser.parse_args()
    
    k_value = args.k
    num_classes = args.classes
    seq_length = args.len
        
    # Get dictionary mapping all possible 10-mers to integers
    kmers_dict = GetKmersDictionary(k_value)
    # get list of species in file directory to reads
    DictFastqFiles, class_mapping, listspecies = GetSetInfo()
    # Create dataset
    DictTestData, DictTrainData = MultiProcesses(DictFastqFiles, class_mapping, kmers_dict, k_value, seq_length, num_classes)
    # Get number of kmers in whole set
    print('Number of {0}-mers: {1}'.format(k_value, len(kmers_dict)))

if __name__ == "__main__":
    main()
