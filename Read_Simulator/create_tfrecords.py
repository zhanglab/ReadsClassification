import os
import numpy as np
import tensorflow as tf
import multiprocess as mp
import argparse
import sys
import random

def get_rev_complement(read):
    """ Converts a k-mer to its reverse complement """
    translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                        "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                        "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                        "X": "N"}
    rev_complement = [translation_dict[nt] for nt in read]
    return ''.join(rev_complement)[::-1]

def get_voc_dict(filename):
    """ Returns a dictionary mapping the kmers to their corresponding integers """
    kmer_to_id = {}
    num_kmer = 1
    with open(filename) as handle:
        for line in handle:
            kmer = line.rstrip()
            kmer_to_id[kmer] = num_kmer
            num_kmer += 1
    return kmer_to_id

def convert_kmer_to_int(kmer, dict_kmers):
    """ Converts kmers into their corresponding integer """
    if kmer in dict_kmers:
        idx = dict_kmers[kmer]
    elif get_rev_complement(kmer) in dict_kmers:
        idx = dict_kmers[get_rev_complement(kmer)]
    else:
        idx = dict_kmers['unknown']

    return idx

def get_kmer_vector(read, k_value, dict_kmers, kmer_vector_length):
    """ Returns a numpy array of k-mers as integers """
    list_kmers = []
    for i in range(len(read)):
        if i + k_value >= len(read) + 1:
            break
        kmer = read[i:i + k_value]
        idx = convert_kmer_to_int(kmer, dict_kmers)
        list_kmers.append(idx)
    if len(list_kmers) < kmer_vector_length:
        # pad list of kmers with 0s to the right
        list_kmers = list_kmers + [0] * (kmer_vector_length - len(list_kmers))
    return np.array(list_kmers)

def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def shuffle_reads(args, label):
    """ Loads and shuffle reads """
    with open(os.path.join(ags.input_path, f'{label}-{args.dataset_type}-reads.fq'), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        random.shuffle(reads)
        return reads

def get_tfrecords(args, label):
    """ Converts reads to tfrecords """
    # get reads
    list_reads = shuffle_reads(args, label)
    # define tfrecords filename
    output_tfrec = f'{label}-{args.dataset_type}-reads.tfrec'
    with tf.compat.v1.python_io.TFRecordWriter(os.path.join(args.input_path, output_tfrec)) as writer:
        for read in list_reads:
            rec = read.split('\n')
            read_seq = str(rec[1])
            label = int(rec[0].split('|')[1])
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to fastq files')
    parser.add_argument('--voc', type=str, help='path to file containing vocabulary (list of kmers)')
    parser.add_argument('--dataset_type', type=str, help='type of dataset used training, evaluating and testing', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    # load class_mapping dictionary
    args.class_mapping = load_class_mapping(os.path.join(args.input_path, 'class_mapping.json'))
    # define the length of kmer vectors
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = get_voc_dict(args.voc)
    # generate tfrecords for each species in parallel
    with mp.Manager() as manager:
        # create new processes
        processes = [mp.Process(target=get_tfrecords, args=(args, label)) for label in args.class_mapping.keys()]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

if __name__ == '__main__':
    main()