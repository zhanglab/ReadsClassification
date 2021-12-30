import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys

from tfrecords_utils import vocab_dict, seq2kmer


def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_ID(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecords(args):
    """ Converts reads to tfrecord, and saves to output file """
    outfile = open(os.path.join(['/'.join(os.path.abspath(args.input_fastq).split('/')[0:-1], 'read_ids.tsv']))
    with tf.compat.v1.python_io.TFRecordWriter(args.output_tfrec) as writer:
        # define read ID number
        read_id_prefix = ''.join([os.path.abspath(args.input_fastq).split('/')[-2].split('-')[1], args.input_fastq.split('-'
        with open(args.input_fastq) as handle:
            for count, rec in enumerate(SeqIO.parse(handle, 'fastq')):
                read = str(rec.seq)
                read_id = rec.id
                read_id_int = int(f'{read_id_prefix}{count}')
                outfile.write(f'{read_id_int}\t{read_id}\n')
                label = int(read_id.split('|')[1])
                kmer_array = seq2kmer(read, args.k_value, args.dict_kmers, args.kmer_vector_length)
                data = \
                    {
                        'read': wrap_read(kmer_array),
                        'label': wrap_label(label),
                        'read_id': wrap_ID(read_id_int)
                    }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)
    outfile.close()

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

    create_tfrecords(args)

if __name__ == "__main__":
    main()
