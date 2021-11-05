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

def get_reads(args, fq_file, fq_type=None, count=False):
    """ Loads and shuffle reads """
    with open(os.path.join(args.input_path, fq_file), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        if fq_type == 'train' or fq_type == 'val':
            random.shuffle(reads)
        if count == True:
            if len(reads) > 25000000:
                list_new_fq_files = []
                # split fastq file into smaller fastq files
                num_new_fq = math.ceil(len(reads)/25000000)
                start = 0
                for i in range(1, num_new_fq+1, 1):
                    new_fq_filename = os.path.join(args.input_path, '-'.join([fq_file.split('-')[0], fq_file.split('-')[1], f'{i}-reads.fq']))
                    end = start + 25000000 if i < num_new_fq else start + (len(reads) - (i-1)*25000000)
                    with open(new_fq_filename) as outfile:
                        outfile.write(''.join(reads[start:end]))
                    start = end
                    list_new_fq_files.append(new_fq_filename.split('/')[-1])
                return list_new_fq_files
            else:
                return [fq_file]
        return reads

def get_tfrecords(args, fq_file):
    """ Converts reads to tfrecords """
    # get type of fastq file
    fq_type = fq_file.split('-')[1]
    # get reads
    list_reads = get_reads(args, fq_file, fq_type)
    # define tfrecords filename
    output_tfrec = os.path.join(args.output_path, '.'.join([fq_file.split('.')[0], 'tfrec']))
    print(output_tfrec)
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
    # report the number of reads
    output_num_reads = os.path.join(args.output_path, '-'.join([fq_file.split('-')[0], fq_file.split('-')[1], 'num-reads']))
    with open(output_num_reads, 'w') as f:
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
        list_fq_files = ['-'.join(i.split('/')[-1].split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.input_path, f'*.fq')))]
        # resume converting reads to tfrecords if any were previously created
        if len(os.listdir(args.output_path)) != 0:
            # get tfrecords done
            tfrec_present = ['-'.join(i.split('/')[-1].split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.output_path, f'*-reads.tfrec')))]
            num_reads_files = ['-'.join(i.split('/')[-1].split('-')[:2]) for i in sorted(glob.glob(os.path.join(args.output_path, f'*-num-reads')))]
            print(len(tfrec_present), len(num_reads_files), len(list_fq_files))
            print(tfrec_present[0], num_reads_files[0], list_fq_files[0])
            # find tfrecords missing
            tfrec_completed = set(tfrec_present).intersection(set(num_reads_files))
            # define final list of fastq files to convert
            pre_final_fq_files = ['-'.join([i, 'reads.fq']) for i in set(list_fq_files).difference(tfrec_completed)]
            # count number of reads per file missing (in case there are too many reads to convert in the time limit given)
            final_fq_files = []
            for fq_file in pre_final_fq_files:
                final_fq_files += get_reads(args, fq_file, fq_type=None, count=True)
            print(f'number of fq files to convert: {len(tfrec_completed)}\t{len(final_fq_files)}')
        else:
            final_fq_files = ['-'.join([i, 'reads.fq']) for i in list_fq_files]
        group_size = len(final_fq_files)//size
        print(f'group size: {group_size}')
        print(final_fq_files)
        fq_files_per_processes = [[] for i in range(size)]
        print(len(fq_files_per_processes))
        num_process = 0
        for i in range(len(final_fq_files)):
            fq_files_per_processes[num_process].append(final_fq_files[i])
            num_process += 1
            if num_process == size:
                num_process = 0
        # fq_files_per_processes = [final_fq_files[i:i+group_size] for i in range(0, len(final_fq_files), group_size)]
        print(f'Rank: {rank}\n{fq_files_per_processes}\n{len(fq_files_per_processes)}')

    else:
        class_mapping = None
        fq_files_per_processes = None
    # scatter list of fastq files to all processes
    fq_files_per_processes = comm.scatter(fq_files_per_processes, root=0)
    print(f'Rank: {rank}\n{fq_files_per_processes}\n')
    # generate tfrecords for each species in parallel
    for fq_file in fq_files_per_processes:
        get_tfrecords(args, fq_file)

if __name__ == '__main__':
    main()
