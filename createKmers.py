# Implementation of an artificial neural network
# Load all necessary libraries
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import gzip
import shutil
import re

# Set model parameters
k_value = 10
sequence_length = int(150 - k_value + 1)

# Create matrix
def createMatrix(DictVectors):
    sequences = np.full((len(DictVectors), sequence_length), list(DictVectors.values()))
    return sequences

# Get list of all possible 4**k_value kmers
def get_all_kmers(list_nt, prefix, n, k, list_kmers):
    if k == 0:
        return list_kmers.append(prefix)
    for i in range(n):
        newPrefix = prefix + list_nt[i]
        get_all_kmers(list_nt, newPrefix, n, k-1, list_kmers)

# Function that looks for any characters different 
# from A, C, T or G and converts the DNA sequence into a vector of 10-mers 
def ParseSeq(DNAsequence,kmers_dict):
    # create empty list to store all the kmers
    listKmers = []

    # Drop the read if it contains letters other than A, C, T or G
    if not re.match('^[ATCG]+$', DNAsequence):
        return listKmers

    # Creates a sliding window of width 10
    for n in range(len(DNAsequence) - 9):
        kmer = DNAsequence[n:n+10]
        # Lookup integer mapped to the kmer
        kmer_Integer = kmers_dict[kmer]
        # Add kmer to vector of kmer
        listKmers.append(kmer_Integer)

    # Pad or truncate list to 141 kmers
    listKmers = listKmers[:141] + [0]*(141 - len(listKmers))

    # transform list into numpy array
    array_int = np.asarray(listKmers)
    # Flip array
    array_int = np.flip(array_int, axis=0)
    return array_int

def GetKmersDictionary(k_value=10):
    # Create empty list to store all possible kmers
    list_kmers = []
    list_nt = ['A', 'T', 'C', 'G']
    # Get list of all possible 4**k_value kmers
    get_all_kmers(list_nt, "", len(list_nt), k_value, list_kmers)
    # generate a list of integers from 0 to the number of kmers
    list_num = list(range(0,len(list_kmers)))
    # Assign an integer to each kmer
    kmers_dict = dict(zip(list_kmers, list_num))
    return kmers_dict


# Goes through the fastq file to create the dictionary
def ParseFastq(Reads_dict, kmers_dict):
    DictVectors = {} # keys = read_id, value = array of integer
    for record, seq_record in Reads_dict.items():
        # Check read sequence
        KmerVector = ParseSeq(str(seq_record.seq), kmers_dict)
        if len(KmerVector) == 141:
            DictVectors[record] = KmerVector
    print(f'Total number of reads: {len(Reads_dict)}')
    print(f'Number of reads after processing: {len(DictVectors)}')
    return DictVectors

def unzipFile(FastqFile, FileName):
    with gzip.open(FastqFile, 'rb') as zippedFastq:
        with open(FileName, 'wb') as unzippedFastq:
            shutil.copyfileobj(zippedFastq, unzippedFastq)
