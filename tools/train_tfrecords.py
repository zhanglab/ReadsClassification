import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys
from tempfile import TemporaryFile
from tfrecords_utils import vocab_dict, seq2kmer
import random

def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_ID(value):
    print(value, len(value))
    """ returns a bytes_list from a string """
#    print(f'before isinstance: {value}')
#    if isinstance(value, type(tf.constant(0))):
#        value = value.numpy()
#        print(f'inside isinstance: {value}')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_read_id(args):
    read_id = int(''.join([args.input_fastq.split('_')[3][1], args.input_fastq.split('_')[2][:1], args.input_fastq.split('_')[1][1], args.input_fastq.split('_')[0][3:]]))
    return read_id


def convert2tfrecord(args, list_reads):
    """
    Converts reads to tfrecord, and saves to output file
    """
    #with open(shuffled_reads_file) as handle:
    #for rec in SeqIO.parse(handle, 'fastq'):
#    output_tfrec = args.input_fastq[:-3] + '.tfrec'
    output_tfrec = os.path.join(args.input_fastq.split('/')[-2], args.input_fastq.split('/')[-1].split('.')[0] + '.tfrec')
    print(output_tfrec)
#    read_id_start = get_read_id(args)
    with tf.compat.v1.python_io.TFRecordWriter(os.path.join(os.getcwd(), output_tfrec)) as writer:
        for count, read in enumerate(list_reads):    
            #read = str(rec.seq)
            rec = read.split('\n')
            read_seq = str(rec[1])
            label = int(rec[0].split('|')[1])
#            read_id = args.num_reads_per_file * args.fq_num + count
#            if count == 0 or count == len(list_reads) - 1:
#                print(f'fastq file number: {args.fq_num} - {read_id}')
#            read_id = str(rec[0])
            # convert read id string to int
#            read_id_int = [ord(i) for i in read_id]
#            if len(read_id) < max_size:
#                read_id_int += [ord('$')]*(max_size - len(read_id)) + [len(read_id)]
#            else:
#                read_id_int += [len(read_id)]
            #label = int(read_id.split('|')[1])
            kmer_array = seq2kmer(read_seq, args.k_value, args.dict_kmers, args.kmer_vector_length)
#            print(f'{read_seq}\n{label}\n{read_id}\n{kmer_array}\n{len(kmer_array)}\n{read_id_int}')
#            print(type(read_id_int))
            data = \
                { 
                    'read': wrap_read(kmer_array),
                    'label': wrap_label(label)
#                    'read_id': wrap_label(read_id)
#                    'read_id': wrap_ID(bytes(read_id, 'utf-8'))
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)

def create_npz_files(args):
    with open(args.input_fastq) as handle:
        n_reads = 45000
        n_shard = 0
        list_reads = []
        list_labels = []
        for rec in SeqIO.parse(handle, 'fastq'):
            read = str(rec.seq)
            read_id = rec.id
            label = int(read_id.split('|')[1])
            kmer_array = seq2kmer(read, args.k_value, args.dict_kmers, args.kmer_vector_length) 
            if len(list_reads) < n_reads:
                list_reads.append(kmer_array)
                list_labels.append(label)
            else:
                output_npz = os.path.join(os.getcwd(), args.input_fastq.split('.')[0] + f'_{n_shard}.npz')
                np.savez(output_npz, x=np.asarray(list_reads, dtype=np.uint8), y=np.asarray(list_labels, dtype=np.uint8))
                n_shard += 1
                list_reads = []
                list_labels = []
                #if n_shard == 1:
                #break

def shuffle_reads(args):
   with open(os.path.join(os.getcwd(), args.input_fastq), 'r') as infile:
        content = infile.readlines()
        reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        if args.type_dataset != 'test': 
            random.shuffle(reads)
        return reads

def get_max_size(args):
    max_size = 0
    with open(args.input_fastq, "r") as handle:
        for rec in SeqIO.parse(handle, 'fastq'):
            if len(str(rec.description)) > max_size:
                max_size = len(str(rec.description))
    return max_size

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--k_value', default=12, type=int, help="The size of k for reads splitting (default 12)")
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--type_dataset', type=str, help="Type of dataset", choices=['train', 'val', 'test'])
#    parser.add_argument('--num_reads_per_file', type=int, help='number of reads per fastq file')
    args = parser.parse_args()
    # retrieve the fastq file number
#    args.fq_num = int(args.input_fastq.split('.')[0].split('_')[3])
    #args.output_tfrec = args.input_fastq.split('.')[0] + '.tfrec'
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)
    # get maximum size of read ids (strings)
#    max_size = get_max_size(args)
    # shuffle reads
    reads = shuffle_reads(args)
    # convert reads to tfrecords
    convert2tfrecord(args, reads)
#    create_npz_files(args)
    # create file to store max size
#    with open(os.path.join(os.getcwd(), f'{args.input_fastq[:-3]}-maxsize'), 'w') as f:
#        f.write(f'{args.input_fastq[:-4]}\t{max_size}')

if __name__ == "__main__":
    main()

