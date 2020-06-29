from . import models
from .loader import read_dataset
from .models.LSTM import AbstractLSTM
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
    argv = check_argv(argv)
    parser = argparse.ArgumentParser(description="Run network training")
    parser.add_argument('model', type=str, help='model to run', choices=models._models.keys())

    parser.add_argument('-input', type=str, help='directory containing data set', default=os.getcwd())
    parser.add_argument('-output', type=str, help='directory to save model outputs to', default=os.getcwd())

    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=10)
    parser.add_argument('-d', '--dropout_rate', type=float, help='dropout rate to use', default=0.5)
    parser.add_argument('-emb', '--embedding_size', type=float, help='embedding size to use', default=60)
    parser.add_argument('-hs', '--hidden_size', type=float, help='#LSTM memory units to use', default=40)

    # Kmer/Bidirectional model requirements
    parser.add_argument('-vs', '--vector_size', type=int, help="size of vectors", required=('kmer' or 'bidirectional') in sys.argv)
    parser.add_argument('-k', '--kvalue', type=int, help="size of kmers", required=('kmer' or 'bidirectional') in sys.argv)
    # Base embedding model requirements
    parser.add_argument("-l", "--length", type=int, help="sequence length", required='base-emb' in sys.argv)

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    return args

def process_args(args=None):
    """
    Process arguments for training
    """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)
    args.input, args.output = process_folder(args)
    args.class_mapping = read_dataset(args.input)
    if args.model == 'kmer' or args.model == 'bidirectional':
        args.num_kmers = 4**args.kvalue
        del args.kvalue

    model = process_model(args)
    return model

def run_model():
    """
    Run the training, testing, and validation for the model
    """
    # Create a strategy to distribute the variables and the graph across multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Build model
        model = process_args(parse_args())
        # Train model
        start = datetime.now()
        print("\nStart training: {}".format(start), file=sys.stderr)
        model.call(strategy)
        end = datetime.now()
        print("\nEnd training: {}".format(end), file=sys.stderr)
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        print("Took %02d:%02d:%02d.%d" % (hours, minutes, seconds, total_time.microseconds))


if __name__ == "__main__":
    run_model()