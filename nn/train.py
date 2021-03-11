import argparse
import sys
import os
import json
import tensorflow as tf
import datetime
from tensorflow.python.client import device_lib

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f'TF version: {tf.__version__}')
print(f'python version: {sys.version}')
sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(f'CUDA version: {cuda_version}')
cudnn_version = sys_details["cudnn_version"]
print(f'CUDNN version: {cudnn_version}')
print(sys_details)
print("NCCL DEBUG SET")
os.environ["NCCL_DEBUG"] = "WARN"

device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# set random seed
tf.compat.v1.set_random_seed(1234)

from . import models
from .models.NN import AbstractNN
from .utils import check_argv, process_folder, process_model, process_checkpoint, get_rank_mapping_dict

def parse_args(*addl_args, argv=None):
    """
    Get parameters for the model
    """
    req_k = ['bidirectional', 'bidirectional-3', 'multilstm', 'multilstm-3', 'kmer', 'cnn-1D', 'cnn-2D', 'gru',
             'kmer-multiple-output']

    argv = check_argv(argv)
    parser = argparse.ArgumentParser(description="Run network training")
    parser.add_argument('model', type=str, help='model to run', choices=models._models.keys())

    parser.add_argument('-input', type=str, help='directory containing data set', default=os.getcwd())
    parser.add_argument('-output', type=str, help='directory to save model outputs to', default=os.getcwd())
    parser.add_argument("-k", "--kvalue", type=int, help="size of kmer", default=10)
    parser.add_argument('-c', '--checkpoint', type=str, help='resume training from file')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to use', default=1)
    parser.add_argument('-d', '--dropout_rate', type=float, help='dropout rate to use', default=0.5)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate to use', default=0.0001)
    parser.add_argument('-emb', '--embedding_size', type=int, help='embedding size to use', default=60)
    parser.add_argument('-hs', '--hidden_size', type=int, help='#LSTM memory units to use', default=40)
    parser.add_argument("-rl", "--readlength", type=int, help="length of reads", default=150)
    parser.add_argument("-m", "--mode", type=str, help="mode of running",
                        choices=['training', 'testing', 'inference'], required=True)
    parser.add_argument("-r", "--rank", type=str, help="taxonomic rank at which we classify the reads", default='species')

    # Specify a specific HDF5 file
    #    parser.add_argument('--hdf5', type=str, help='the HDF5 file containing the dataset')

    # Kmer/Bidirectional model requirements
    # parser.add_argument('-k', '--kvalue', type=int, help="size of kmers",
    #                    required=(model in sys.argv for model in req_k))
    # Base embedding model requirements
    # parser.add_argument('-l', '--length', type=int, help="sequence length", default=150)
    # CNN model requirements
    #parser.add_argument('-f', '--filter', type=int, help="filter number", default=10)
    #parser.add_argument('-ker', '--kernel', type=int, help="kernel size", default=5)
    # CNN and GRU support paired and unpaired reads
    #parser.add_argument('-reads', help="Specify if unpaired or paired reads",
    #                    required=('cnn-1D' in sys.argv or 'cnn-2D' in sys.argv or 'gru' in sys.argv),
    #                    choices=['paired', 'unpaired'])
    parser.add_argument('-reads', help="Specify if unpaired or paired reads", choices=['paired', 'unpaired'])

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
        kmers = {'10': 524800, '12': 8390658}
        args.num_kmers = kmers[str(args.kvalue)]
        args.vector_size = args.readlength - args.kvalue + 1
        print(f'#kmers: {args.num_kmers} - vector size: {args.vector_size}')

    #    args.hdf5 = process_folder(args)
    # Set read type if not using CNN or GRU
    if args.reads is None:
        args.reads = 'unpaired' if (args.model in ['kmer', 'base-emb']) else 'paired'

    # check output and logs directories
    process_folder(args)

    # Set batch size
    if args.batch_size is None:
        args.batch_size = 32 * num_gpus

    # get class_mapping dictionary
    f = open(os.path.join(args.input, 'class-mapping.json'))
    args.class_mapping = json.load(f)
    # get model architecture
    model = process_model(args)
    args.checkpoint = process_checkpoint(args)

    if args.mode == 'inference':
        get_rank_mapping_dict(args)

    return model, args

def run_model():
    """
    Run the training, testing or inference of the model
    """
    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        # Build model
        model, args = process_args(strategy.num_replicas_in_sync, parse_args())
        # Train model
        start = datetime.datetime.now()
        model.call(strategy)
        end = datetime.datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
        with open(os.path.join(args.output, 'metrics.txt'), 'a+') as f:
            f.write("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
            f.write(
                f'learning rate: {args.learning_rate}\tnumber of kmers: {args.num_kmers}\tbatch size per gpu: {args.batch_size}\n')


if __name__ == "__main__":
    run_model()
