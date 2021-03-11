import argparse
import tensorflow as tf
import math
import datetime
import sys
import os
import json

from training import run_training
from testing import run_testing

# show which devices operations and tensors are assigned to
#tf.debugging.set_log_device_placement(True)

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

def check_argv(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()
    return argv

def parse_args(*addl_args, argv=None):
    """ get command line arguments """
    argv = check_argv(argv)
    parser = argparse.ArgumentParser(description="Create train/test datasets")
    parser.add_argument('-input', type=str, help='directory containing input files', required=True)
    parser.add_argument('-output', type=str, help='directory with output files', default=os.getcwd())
    parser.add_argument('-mode', type=str, help='mode for running program', required=True, choices=['training', 'testing'])
    parser.add_argument('-read_length', type=int, help='read length', default=250)
    parser.add_argument('-batch_size', type=int, help='batch size per gpu', default=125)
    parser.add_argument('-num_gpu', type=int, help='number of gpus to use', default=4, choices=[2, 3, 4])
    parser.add_argument('-epochs', type=int, help='number of epochs', default=1000)
    parser.add_argument('-embedding_size', type=int, help='size of embeddings', default=60)
    parser.add_argument('-dropout_rate', type=float, help='dropout rate', default=0.5)
    parser.add_argument('-train_size', type=int, help='size of training set', required=('training' in sys.argv))
    parser.add_argument('-val_size', type=int, help='size of validation set', required=('training' in sys.argv))
    parser.add_argument('-test_size', type=int, help='size of testing set', required=('testing' in sys.argv))
    parser.add_argument('-model_name', type=str, help='name of model', default='model')

    for a in addl_args:
        parser.add_argument(*a[0], **a[1])

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    return args

def process_args(args=None):
    """ process arguments """
    if not isinstance(args, argparse.Namespace):
        args = parse_args(args)

    # verify that input directory contains tfrecord indexes
    for set_type in ['train', 'val', 'test']:
        if not os.path.isdir(os.path.join(args.input, 'tfrecords', set_type, 'idx_files')):
            sys.exit(f'prepare tfrecord indexes for {set_type} set with dali_tf2idx.sh')

    # define data
    args.train_tfrecord = os.path.join(args.input,'tfrecords', 'train', 'training_all_subsets.tfrec')
    args.train_tfrecord_idx = os.path.join(args.input, 'tfrecords', 'train', 'idx_files/training_all_subsets.tfrec.idx')
    args.val_tfrecord = os.path.join(args.input, 'tfrecords', 'val', 'validation_all_subsets.tfrec')
    args.val_tfrecord_idx = os.path.join(args.input, 'tfrecords', 'val', 'idx_files/validation_all_subsets.tfrec.idx')
    args.test_tfrecord = os.path.join(args.input, 'tfrecords', 'test', 'testing_all_subsets.tfrec')
    args.test_tfrecord_idx = os.path.join(args.input, 'tfrecords', 'test', 'idx_files/testing_all_subsets.tfrec.idx')

    # load taxa mapping dictionary
    f = open(os.path.join(args.input, 'class_mapping.json'))
    args.class_mapping = json.load(f)
    args.num_classes = len(args.class_mapping)

    # define global batch size
    args.global_batch_size = args.batch_size * args.num_gpu

    # define the k value
    args.k_value = 12

    # define the number of features
    args.vector_size = args.read_length - args.k_value + 1

    # define number of k-mers in the vocabulary
    args.num_kmers = 8390658 # number of canonical k-mers with k-value of 12

    # get list of all visible GPUs
    args.gpus = tf.config.experimental.list_physical_devices('GPU')
    # adjust number of gpus
    if args.num_gpu != len(args.gpus):
        args.gpus = args.gpus[:args.num_gpu]
    for gpu in args.gpus:
        # enable memory growth to prevent the runtime initialization to allocate all memory on the device
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.gpus:
        # specify which gpus are visible to the runtime
        tf.config.experimental.set_visible_devices(args.gpus, 'GPU')

    args.dict_gpus = {2: ['/gpu:0', '/gpu:1'], 3: ['/gpu:0', '/gpu:1', '/gpu:2'], 4: ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']}

    # create directory for model
    if not os.path.isdir(os.path.join(args.output, args.model_name)):
        os.makedirs(os.path.join(args.output, args.model_name))

    return args

def run_command():

    # get arguments
    args = process_args(parse_args())

    start = datetime.datetime.now()

    if args.mode == 'training':
        # define number of steps (iterations or global batches) per epoch in training set
        args.steps_per_epoch = math.ceil(args.train_size / args.global_batch_size)
        # define number of steps (iterations or global batches) in validation set
        args.validation_steps = math.ceil(args.val_size / args.global_batch_size)
        run_training(args)
    elif args.mode == 'testing':
        # define number of steps (iterations or global batches) in testing set
        args.testing_steps = math.ceil(args.test_size / args.global_batch_size)
        run_testing(args)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print("\nTook %02d:%02d:%02d.%d\n" % (hours, minutes, seconds, total_time.microseconds))
    with open(os.path.join(args.output, f'{args.mode}-info'), 'w+') as f:
        f.write(f'{args.mode} runtime: {hours}:{minutes}:{seconds}.{total_time.microseconds}')



if __name__ == "__main__":
    run_command()
