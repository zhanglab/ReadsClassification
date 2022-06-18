import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import random
import gzip
import sys
import math
import statistics
from tfrecords_utils import vocab_dict, get_kmer_arr, get_flipped_reads, cut_read

def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_meta_tfrecords(args):
    """ Converts metagenomic reads to tfrecords """
    output_tfrec = os.path.join(args.output_dir, args.output_prefix + '.tfrec')
    outfile = open('/'.join([args.output_dir, args.output_prefix + f'-read_ids.tsv']), 'w')
    num_reads = 0
    length_reads = set()
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        with gzip.open(args.input_fastq, 'rt') as handle:
            for count, rec in enumerate(SeqIO.parse(handle, 'fastq'), 1):
                read = str(rec.seq)
                length_reads.add(len(read))
                read_id = rec.description
                outfile.write(f'{read_id}\t{count}\n')
                list_reads = []
                if len(read) > args.read_length:
                    list_reads = cut_read(args, read)
                    print(read)
                    print(list_reads)
                    break
                else:
                    list_reads = [read]
                for r in list_reads:
                    kmer_array = get_kmer_arr(r, args.k_value, args.dict_kmers, args.kmer_vector_length)
                    data = \
                        {
                            'read': wrap_read(kmer_array),
                            'label': wrap_label(count)
                        }
                    feature = tf.train.Features(feature=data)
                    example = tf.train.Example(features=feature)
                    serialized = example.SerializeToString()
                    writer.write(serialized)
                    num_reads += 1

            with open(os.path.join(args.output_dir, args.output_prefix + '-read_count'), 'w') as f:
                f.write(f'{num_reads}')

    outfile.close()
    print(f'{args.input_fastq}\t{statistics.mean(length_reads)}\t{min(length_reads)}\t{max(length_reads)}')


def create_tfrecords(args):
    """ Converts simulated reads to tfrecord """
    output_tfrec = os.path.join(args.output_dir, args.output_prefix + '.tfrec')
    outfile = open('/'.join([args.output_dir, args.output_prefix + f'-read_ids.tsv']), 'w')
    records = [rec.format('fastq') for rec in list(SeqIO.parse(args.input_fastq, "fastq"))]
    if args.flipped:
        records += get_flipped_reads(args, records)
        print(f'with flipped reads: {len(records)}')
    if args.dataset_type in ['training', 'validation']:
        random.shuffle(records)
    with tf.compat.v1.python_io.TFRecordWriter(output_tfrec) as writer:
        for count, rec in enumerate(records, 1):
            label = int(rec.split('\n')[0].split('|')[1])
            read_id = rec.split("\n")[0]
            outfile.write(f'{read_id}\t{count}\n')
            kmer_array = get_kmer_arr(rec, args.k_value, args.dict_kmers, args.kmer_vector_length, args.read_length)
            data = \
                {
                    'read': wrap_read(kmer_array),
                    'label': wrap_label(label),
                    'read_id': wrap_label(count)
                }
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()
            writer.write(serialized)

        with open(os.path.join(args.output_dir, args.output_prefix + '-read_count'), 'w') as f:
            f.write(f'{count}')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
    parser.add_argument('--output_dir', help="Path to the output directory")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--k_value', default=12, type=int, help="Size of k-mers")
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    parser.add_argument('--dataset_type', type=str, help="Type of dataset", choices=['testing', 'training', 'validation', 'meta'])
    parser.add_argument('--flipped', help='Use to add flipped versions of reads into tfrecords', action='store_true')

    args = parser.parse_args()
    args.output_prefix = args.input_fastq.split('/')[-1].split('.')[0]
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)

    if args.dataset_type in ['testing', 'training', 'validation']:
        create_tfrecords(args)
    elif args.dataset_type == 'meta':
        create_meta_tfrecords(args)

if __name__ == "__main__":
    main()
