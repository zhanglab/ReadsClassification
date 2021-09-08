import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys
import glob
from tfrecords_utils import vocab_dict, seq2kmer
import random

def get_reverse(read):
    """
    Converts a k-mer to its reverse complement.
    All ambiguous bases are treated as Ns.
    """
    translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                        "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                        "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                        "X": "N"}
    list_bases = list(read)
    list_bases = [translation_dict[base] for base in list_bases]
    return ''.join(list_bases)[::-1]


def vocab_dict(filename):
    """
    Turns the vocabulary into a dict={kmer: id}.
    """
    kmer_to_id = {}
    num_kmer = 1
    with open(filename) as handle:
        for line in handle:
            kmer = line.rstrip()
            kmer_to_id[kmer] = num_kmer
            num_kmer += 1
    return kmer_to_id

def get_indexes(kmer, dict_kmers):
    """Convert kmers into their corresponding index"""
    if kmer in dict_kmers:
        idx = dict_kmers[kmer]
    elif get_reverse(kmer) in dict_kmers:
        idx = dict_kmers[get_reverse(kmer)]
    else:
        idx = dict_kmers['unknown']

    return idx

def get_kmer_vector(read, k_value, dict_kmers, kmer_vector_length):
    """
    Converts a DNA sequence split into a list of k-mers.
    Returns:
         kmer_array: a numpy array of corresponding k-mer indexes.
    """
    list_kmers = []
    for i in range(len(read)):
        if i + k_value >= len(read) + 1:
            break
        kmer = read[i:i + k_value]
        idx = get_indexes(kmer, dict_kmers)
        list_kmers.append(idx)

    if len(list_kmers) < kmer_vector_length:
        # pad list of kmers with 0s to the right
        list_kmers = list_kmers + [0] * (kmer_vector_length - len(list_kmers))

    return np.array(list_kmers)

def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(args, list_reads, type_set):
    """
    Converts reads to tfrecord, and saves to output file
    """
    # define output tfrecords filename
    output_tfrec = os.path.join(args.input_dir, 'tfrecords', f'{args.label}-{type_set}.tfrec')
    # iterate over the list of reads and convert them to vector of k-mers
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        for record in list_reads:
            read_seq = str(record.split('\n')[1])
            label = int(record.split('\n')[0].split('-')[1])
            kmer_array = get_kmer_vector(read_seq, args.k_value, args.dict_kmers, args.kmer_vector_length)
            data = \
                {
                    'read': wrap_read(kmer_array),
                    'label': wrap_label(label)
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)

def get_reads(args, type_set):
   with open(os.path.join(args.input_dir, f'{args.label}-sets', f'{args.label}-{type_set}.fq'), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        random.shuffle(reads)
        return reads

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str, help="Path to directories with datasets")
    parser.add_argument('-label', type=str, help="Label species ID")
    parser.add_argument('-vocab', help="Path to the vocabulary file")
    parser.add_argument('-k_value', default=12, type=int, help="The size of k for reads splitting (default 12)")
    parser.add_argument('-read_length', default=250, type=int, help="The length of simulated reads")
    args = parser.parse_args()
    # define size of vector of kmers
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)
    # generate records for training, validation adn testing sets
    for type_set in ['training', 'validation', 'testing']:
        # get and shuffle reads
        reads = get_reads(args, type_set)
        create_tfrecord(args, reads, type_set)

if __name__ == "__main__":
    main()
