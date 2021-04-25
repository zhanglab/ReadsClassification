import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys

from tfrecords_utils import vocab_dict, seq2kmer


def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_ID(value):
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert2tfrecord(args, max_size):
    """
    Converts reads to tfrecord, and saves to output file
    """
    with tf.compat.v1.python_io.TFRecordWriter(args.output_tfrec) as writer:
        with open(args.input_fastq) as handle:
            for rec in SeqIO.parse(handle, 'fastq'):
                read = str(rec.seq)
                read_id = str(rec.id)
                # convert string to int
                read_id_int = [ord(i) for i in read_id]
                if len(read_id) < max_size:
                    read_id_int += [ord('$')]*(max_size - len(read_id)) + [len(read_id)]
                else:
                    read_id_int += [len(read_id)]
                print(read_id, read_id_int, len(read_id_int))
                # get array of kmers
                kmer_array = seq2kmer(read, args.k_value, args.dict_kmers, args.kmer_vector_length)
                data = \
                        {
                            'read': wrap_read(kmer_array),
                            'read_id': wrap_ID(np.asarray(read_id_int))
                        }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)

def get_max_size(args):
    max_size = 0
    with open(args.input_fastq) as handle:
        for rec in SeqIO.parse(handle, 'fastq'):
            if len(str(rec.id)) > max_size:
                max_size = len(str(rec.id))
    return max_size

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--k_value', default=12, type=int, help="The size of k for reads splitting (default 12)")
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")

    args = parser.parse_args()
    args.output_tfrec = args.input_fastq.split('.')[0] + '.tfrec'
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)
    # get maximum size of read ids (strings)
    max_size = get_max_size(args)
    print(f'read id max size: {max_size}')
    # convert reads to tfrecords
    convert2tfrecord(args, max_size)

if __name__ == "__main__":
    main()

