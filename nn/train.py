from . import models
from .models.NN import AbstractNN
from .utils import check_argv, process_folder, process_model

import argparse
import sys
import os
import tensorflow as tf
from datetime import datetime
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def parse_args(*addl_args, argv=None):
    """
    Get parameters for the model
    """
    req_k = ['bidirectional', 'bidirectional-3', 'multilstm', 'multilstm-3', 'kmer', 'cnn', 'gru']
    format = ['hdf5', 'tfrecords']

    argv = check_argv(argv)
    parser = argparse.ArgumentParser(description="Run network training")
    parser.add_argument('model', type=str, help='model to run', choices=models._models.keys())

    parser.add_argument('-input', type=str, help='directory containing data set', default=os.getcwd())
    parser.add_argument('-output', type=str, help='directory to save model outputs to', default=os.getcwd())

    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=10)
    parser.add_argument('-d', '--dropout_rate', type=float, help='dropout rate to use', default=0.5)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate to use', default=0.0001)
    parser.add_argument('-emb', '--embedding_size', type=int, help='embedding size to use', default=60)
    parser.add_argument('-hs', '--hidden_size', type=int, help='#LSTM memory units to use', default=40)

    # Specify a specific HDF5 file
    parser.add_argument('--hdf5', type=str, help='the HDF5 file containing the dataset')

    # Specify a specific TFRecords folder
    parser.add_argument('--tfrecords', type=str, help='the TFRecords containing the dataset')

    # Specify the dataset format
    parser.add_argument('--format', type=str, help='the format of the dataset', choices=format, required=True)

    # Kmer/Bidirectional model requirements
    parser.add_argument('-k', '--kvalue', type=int, help="size of kmers",
                        required=(model in sys.argv for model in req_k))
    # Base embedding model requirements
    parser.add_argument('-l', '--length', type=int, help="sequence length", default=150)
    # CNN model requirements
    parser.add_argument('-f', '--filter', type=int, help="filter number", default=10)
    parser.add_argument('-kernel', type=int, help="kernel size", default=5)
    # CNN and GRU support paired and unpaired reads
    parser.add_argument('-reads', help="Specify if unpaired or paired reads",
                        required=('cnn' in sys.argv or 'gru' in sys.argv),
                        choices=['paired', 'unpaired'])

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    return args

def process_args(num_gpus, args=None):
    """
    Process arguments for training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    if args.kvalue is not None:
        args.num_kmers = 4**args.kvalue
        args.vector_size = 150 - args.kvalue + 1

    args.dataset = process_folder(args)

    # Set read type if not using CNN or GRU
    if args.reads is None:
        args.reads = 'unpaired' if (args.model in ['kmer', 'base-emb']) else 'paired'

    # Set batch size
    if args.batch_size is None:
        args.batch_size = 32 * num_gpus

    # Load class_mapping
    f = open(os.path.join(args.input, 'species-integers.json'))
    args.class_mapping = json.load(f)

    model = process_model(args)
    return model, args

def run_model():
    """
    Run the training, testing, and validation for the model
    """
    # Create a strategy to distribute the variables and the graph across multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Build model
        model, args = process_args(strategy.num_replicas_in_sync, parse_args())

        # Train model
        start = datetime.now()
        model.call(strategy)
        end = datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        with open(os.path.join(args.output, 'metrics.txt'), 'a+') as f:
            f.write("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))

if __name__ == "__main__":
    run_model()
