import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys

def get_reverse_seq(read):
    """ Converts an k-mer to its reverse complement. All ambiguous bases are treated as Ns. """
    translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                        "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                        "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                        "X": "N"}
    list_bases = list(read)
    list_bases = [translation_dict[base] for base in list_bases]
    return ''.join(list_bases)[::-1]


def vocab_dict(filename):
    """ Turns the vocabulary into a dict={kmer: id} """
    kmer_to_id = {}
    with open(filename) as handle:
        for line in handle:
            kmer = line.rstrip().split('\t')[0]
            idx = line.rstrip().split('\t')[1]
            kmer_to_id[kmer] = int(idx)
    return kmer_to_id

def get_kmer_index(kmer, dict_kmers):
    """Convert kmers into their corresponding index"""
    if kmer in dict_kmers:
        idx = dict_kmers[kmer]
    elif get_reverse_seq(kmer) in dict_kmers:
        idx = dict_kmers[get_reverse_seq(kmer)]
    else:
        idx = dict_kmers['unknown']

    return idx

def get_kmer_arr(read, k_value, dict_kmers, kmer_vector_length):
    """ Converts a DNA sequence split into a list of k-mers """
    list_kmers = []
    for i in range(len(read)):
        if i + k_value >= len(read) + 1:
            break
        kmer = seq[i:i + k_value]
        idx = get_kmer_index(kmer, dict_kmers)
        list_kmers.append(idx)

    if len(list_kmers) < kmer_vector_length:
        # pad list of kmers with 0s to the right
        list_kmers = list_kmers + [0] * (kmer_vector_length - len(list_kmers))

    return np.array(list_kmers)
