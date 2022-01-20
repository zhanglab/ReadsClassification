import os
import numpy as np
from Bio import SeqIO
import tensorflow as tf
import argparse
import sys
import gzip
from tfrecords_utils import vocab_dict, seq2kmer


def wrap_read(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_ID(value):
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_read_id(args):
    read_id = int(''.join([args.input_fastq.split('/')[-1].split('_')[3][1], args.input_fastq.split('/')[-1].split('_')[2][:1], args.input_fastq.split('/')[-1].split('_')[1][1], args.input_fastq.split('/')[-1].split('_')[0][3:], '0']))
    
    return read_id


def convert2tfrecord(args):
    """
    Converts reads to tfrecord, and saves to output file
    """
#    n_reads = 150000
#    n_shards = round(args.num_reads / n_reads)
#    print('number of shards {}'.format(n_shards))
#    print('number of reads {}'.format(args.num_reads))
#    list_records = list(SeqIO.parse(args.input_fastq, 'fastq'))
#    print(f'number of records: {len(list_records)}')
#    start = 0
#    for i in range(1, n_shards+1, 1):
#        tfrecord_shard_path = os.path.join(args.output, 'testing-reads-{}-of-{}.tfrec'.format(i, n_shards))
#        end = start + n_reads if i < n_shards else start + (len(list_records) - (i-1)*n_reads)
#        rec_shard = list_records[start:end]
    output_tfrec = '-'.join(args.input_fastq.split('.')[:-1]) + '.tfrec'
    print(output_tfrec)
    read_id = get_read_id(args)
    with tf.compat.v1.python_io.TFRecordWriter(os.path.join(os.getcwd(), 'tfrecords', output_tfrec)) as writer:
        with gzip.open(args.input_fastq, 'rt') as handle:
            for rec in SeqIO.parse(handle, 'fastq'):
                read = str(rec.seq)
                print(read_id)
                read_id = read_id + 1
#                # convert string to int
#                read_id_int = [ord(i) for i in read_id]
#                if len(read_id) < max_size:
#                    read_id_int += [ord('$')]*(max_size - len(read_id)) + [len(read_id)]
#                else:
#                    read_id_int += [len(read_id)]
                # get array of kmers
#                kmer_array = seq2kmer(read, args.k_value, args.dict_kmers, args.kmer_vector_length)
#                print(kmer_array)
#                data = \
#                        {
#                            'read': wrap_read(kmer_array),
#                            'read_id': wrap_ID(np.asarray(read_id))
#                        }
#                feature = tf.train.Features(feature=data)
#                example = tf.train.Example(features=feature)
#                serialized = example.SerializeToString()
#                writer.write(serialized)

def get_max_size(args):
    max_size = 0
    with open(args.input_fastq, "r") as handle:
        for rec in SeqIO.parse(handle, 'fastq'):
            if len(str(rec.id)) > max_size:
                max_size = len(str(rec.id))
    return max_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fastq', help="Path to the input fastq file")
#    parser.add_argument('--output', help="Path to store output files")
    parser.add_argument('--vocab', help="Path to the vocabulary file")
    parser.add_argument('--k_value', default=12, type=int, help="The size of k for reads splitting (default 12)")
    parser.add_argument('--read_length', default=250, type=int, help="The length of simulated reads")
    args = parser.parse_args()
    # create output filename
#    args.output_tfrec = os.path.join(args.output, 'tfrecords', f'{args.input_fastq[:-3]}.tfrec')
    # define size of kmer vectors
    args.kmer_vector_length = args.read_length - args.k_value + 1
    # get dictionary mapping kmers to indexes
    args.dict_kmers = vocab_dict(args.vocab)
    # get maximum size of read ids (strings)
#    max_size = get_max_size(args)
    # convert reads to tfrecords
    convert2tfrecord(args)
    # create file to store max size
#    with open(os.path.join(args.output, 'tfrecords', f'{args.input_fastq[:-3]}-maxsize'), 'w') as f:
#        f.write(f'{args.input_fastq[:-4]}\t{max_size}') 

if __name__ == "__main__":
    main()

