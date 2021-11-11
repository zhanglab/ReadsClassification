import os
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import argparse
import sys
import random
import glob
import math
from utils import *

# create a communicator consisting of all the processors
comm = MPI.COMM_WORLD
# get the number of processors
size = comm.Get_size()
# get the rank of each processor
rank = comm.Get_rank()
print(comm, size, rank)

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

def get_reads(args, fq_file):
    with open(fq_file, 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        return reads

def get_tfrecords(args, tfrec, shuffle=False):
    # define tfrecords filename
    output_tfrec_filename = tfrec.split('/')[-1] + '.tfrec'
    output_tfrec = os.path.join(args.output_path, output_tfrec_filename)
    print(output_tfrec)
    # get list of fastq files
    list_fq_files = sorted(glob.glob(os.path.join(tfrec, f'*-{args.dataset}-reads.fq')))
    # get reads
    list_reads = []
    # report the number of reads
    output_num_reads = os.path.join(tfrec, 'num-reads'])
    for fq_file in list_fq_files:
        label_reads = get_reads(args, fq_file)
        label = fq_file.split('/')[-1].split('-')[0]
        list_reads += label_reads
        with open(output_num_reads, 'a') as f:
            f.write(f'{label}\t{len(label_reads)}\n')
    if shuffle:
        random.shuffle(list_reads)
    # converts reads to tfrecords
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        for read in list_reads:
            rec = read.split('\n')
            read_seq = str(rec[1])
            label = int(rec[0].split('-')[2])
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
    # report total number of reads
    with open(output_num_reads, 'a') as f:
        f.write(f'{len(list_reads)}\n')

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to fastq files')
    parser.add_argument('--dataset', type=str, help='type of dataset', choices=['train', 'val'])
    parser.add_argument('--read_length', type=int, help='length of reads in bp')
    parser.add_argument('--k_value', type=int, help='length of kmers', default=12)
    parser.add_argument('--voc', type=str, help='path to file containing vocabulary (list of kmers)')
    args = parser.parse_args()
    # define the length of kmer vectors
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = get_voc_dict(args.voc)
    # define output directory
    args.output_path = os.path.join(args.input_path, 'tfrecords')
    if rank == 0:
        # create directory to store tfrecords
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)
        # get the list of tfrecords directories
        list_tfrecords = sorted(glob.glob(os.path.join(args.input_path, 'fq_files', f'{args.dataset}-tfrec-*')))
        # generate lists to store tfrecords filenames
        tfrec_files_per_processes = [[] for i in range(size)]
        # divide tfrec files into number of processes available
        group_size = len(list_tfrecords)//size
        print(f'group size: {group_size}')
        print(list_tfrecords)
        num_process = 0
        for i in range(len(list_tfrecords)):
            tfrec_files_per_processes[num_process].append(list_tfrecords[i])
            num_process += 1
            if num_process == size:
                num_process = 0
    else:
        tfrec_files_per_processes = None
    # scatter list of fastq files to all processes
    tfrec_files_per_processes = comm.scatter(tfrec_files_per_processes, root=0)
    print(f'Rank: {rank}\n{tfrec_files_per_processes}\n')
    # create
    for tfrec in tfrec_files_per_processes:
        get_tfrecords(args, tfrec, True)


if __name__ == '__main__':
    main()
