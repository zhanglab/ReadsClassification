import os
import numpy as np
import tensorflow as tf
from mpi4py import MPI
import argparse
import sys
import random
import glob
from utils import *

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
    with open(os.path.join(args.input_path, f'{label}-{args.dataset_type}-reads.fq'), 'r') as infile:
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
    with tf.compat.v1.python_io.TFRecordWriter(os.path.join(args.output_path, output_tfrec)) as writer:
        for read in list_reads:
            rec = read.split('\n')
            read_seq = str(rec[1])
            label = int(rec[0].split('-')[1])
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
    # report the number of reads
    with open(os.path.join(args.output_path, f'{label}-{args.dataset_type}-num-reads'), 'w') as f:
        f.write(f'{label}\t{len(list_reads)}\n')

def main():
    # create a communicator consisting of all the processors
    comm = MPI.COMM_WORLD
    # get the number of processors
    size = comm.Get_size()
    # get the rank of each processor
    rank = comm.Get_rank()
    print(comm, size, rank)
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path to fastq files')
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
        # get list of fastq files to convert
        fq_files = ['-'.join(i.split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.input, f'*.fq')))]
        # resume converting reads to tfrecords if any were previously created
        if len(os.listdir(args.output_path)) != 0:
            # get tfrecords done
            tfrec_done = ['-'.join(i.split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.output_path, f'*-reads.tfrec')))]
            num_reads_files = ['-'.join(i.split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.output_path, f'*-num-reads')))]
            # find tfrecords missing
            diff = set(tfrec_done).difference(set(num_reads_files))
            # define final list of fastq files to convert
            final_fq_files = ['-'.join([i, 'reads.fq']) for i in set(fq_files).difference(diff)]
            print(f'number of fq files to convert: {len(diff)}\t{len(final_fq_files)}')
        # load class_mapping dictionary
        #class_mapping = load_class_mapping(os.path.join(args.input_path, 'class_mapping.json'))
        # split dictionary into N lists of dictionaries with N equal to the number of processes
    #     list_dict = [{} for i in range(size)]
    #     l_pos = 0
    #     for i in range(len(class_mapping)):
    #         list_dict[l_pos][i] = class_mapping[str(i)]
    #         l_pos += 1
    #         if l_pos == size:
    #             l_pos = 0
    #     print(f'Rank: {rank}\n{list_dict}\n{len(list_dict)}')
    #     print(f'Rank: {rank}\t{args.kmer_vector_length}')
    # else:
    #     class_mapping = None
    #     list_dict = None
    # # scatter dictionary to all processes
    # list_dict = comm.scatter(list_dict, root=0)
    # print(f'Rank: {rank}\n{list_dict}\n')
    # # generate tfrecords for each species in parallel
    # for label in list_dict.keys():
    #     get_tfrecords(args, label)
    # with mp.Manager() as manager:
    #     # create new processes
    #     processes = [mp.Process(target=get_tfrecords, args=(args, label)) for label in args.class_mapping.keys()]
    #     for p in processes:
    #         p.start()
    #     for p in processes:
    #         p.join()

if __name__ == '__main__':
    main()
