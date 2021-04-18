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
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert2tfrecord(args):
    """
    Converts reads to tfrecord, and saves to output file
    """
    with tf.compat.v1.python_io.TFRecordWriter(args.output_tfrec) as writer:
        with open(args.input_fastq) as handle:
            for rec in SeqIO.parse(handle, 'fastq'):
                read = str(rec.seq)
                read_id = str(rec.id)
                kmer_array = seq2kmer(read, args.k_value, args.dict_kmers, args.kmer_vector_length)
                data = \
                        {
                            'read': wrap_read(kmer_array),
                            'read_id': wrap_ID(read_id)
                        }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)

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

    convert2tfrecord(args)

if __name__ == "__main__":
    main()

