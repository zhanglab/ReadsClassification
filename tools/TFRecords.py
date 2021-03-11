import tensorflow as tf
import numpy as np
import os

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(kmer_vector, label):
    data_description = {
        'kmers': _int64_feature(kmer_vector),
        'label': _float_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=data_description))
    return example_proto.SerializeToString()

def write_TFRecord(reads, labels, args, set_type, num_classes):
    # each TFRecord represent one read and its label
    # reads are stored in files/shards of 100-200MB--> 150000 reads per shard (169.2MB)
    n_reads = 150000
    n_shards = round(len(reads) / n_reads)
    print('number of shards {}'.format(n_shards))
    print('number of reads {}'.format(len(reads)))
    start = 0
    for i in range(1, n_shards+1, 1):
        tfrecord_shard_path = os.path.join(args.output, 'tfrecords/{}/{}-reads-{}-of-{}'.format(set_type, set_type, i, n_shards))
        end = start + n_reads if i < n_shards else start + (len(reads) - (i-1)*n_reads)
        reads_list = reads[start:end]
        labels_list = labels[start:end]
        with tf.io.TFRecordWriter(tfrecord_shard_path) as writer:
            for read, label in zip(reads_list, labels_list):
                one_hot = tf.keras.utils.to_categorical(label, num_classes=num_classes)
                example = create_example(read, one_hot)
                writer.write(example)
        start = end

