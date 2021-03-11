import glob
import sys
import os
import h5py
import itertools
import datetime
import tensorflow as tf
from . import models
from collections import defaultdict

def kmer_dictionary(k_value):
    list_nt = ['A', 'T', 'C', 'G']
    # Get list of all possible 4**k_value kmers
    list_kmers = [''.join(p) for p in itertools.product(list_nt, repeat=k_value)]
    # Assign an integer to each kmer
    kmers_dict = dict(zip(list_kmers, range(len(list_kmers))))
    return kmers_dict

def check_argv(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()
    return argv

def get_ckpts(args):
    ckpts = sort(os.listdir(os.path.join(args.input, 'ckpts')))
    return ckpts

def process_checkpoint(args):
    if args.checkpoint is None:
        args.num_epochs_done = 0
    else:
        if args.mode == 'training':
            args.num_epochs_done = int(args.checkpoint.split('-')[4])
            args.epochs = args.epochs + args.num_epochs_done
        args.checkpoint = os.path.join(args.input, 'ckpts', args.checkpoint)

def get_label(args, string):
    for label, species_name in args.class_mapping.items():
        if species_name == string:
            return label

def get_rank_mapping_dict(args):
    # create dictionary with training genomes as key and corresponding label as value
    dict_train_genomes = {}
    with open(os.path.join(args.input, 'list-genomes'), 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            dict_train_genomes[fields[1]] = get_label(args, fields[0])

    ranks = {'species': 6, 'genus': 5, 'family': 4, 'order': 3, 'class': 2, 'phylum': 1}
    args.rank_mapping_dict = defaultdict(list)
    with open(os.path.join(args.input, 'training-genomes'), 'r') as f:
        content = f.readlines()
        for line in content:
            fields = line.rstrip().split('\t')
            species = fields[len(fields)-1]
            species_label = get_label(args, species)
            # get species label if species not found in the mapping dictionary used for training
            if species_label == None:
                species_label = dict_train_genomes[fields[0]]
            # replace '' by unknown
            fields = ['unknown' if i == '' else i for i in fields]
            # get taxon at desired rank
            taxon_rank = fields[ranks[args.rank]]
            args.rank_mapping_dict[taxon_rank].append(int(species_label))
    # create dictionary mapping labels to taxa of interest
    train_labels = list(range(len(args.rank_mapping_dict)))
    train_taxa = [key for key in args.rank_mapping_dict.keys()]
    args.integer_mapping = {train_labels[i]: train_taxa[i] for i in range(len(train_taxa))}

def process_folder(args):
    """
    Check the output folder to see if it exists
    """

    def check_output():
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # create log directories
    def create_logs():
        if not os.path.isdir(os.path.join(args.output, 'logs')):
            # create logs directory
            os.makedirs(os.path.join(args.output, 'logs'))
            os.makedirs(os.path.join(args.output, 'logs', 'train'))
            os.makedirs(os.path.join(args.output, 'logs', 'test'))
            os.makedirs(os.path.join(args.output, 'logs', 'learning_rate'))
            os.makedirs( os.path.join(args.output, 'logs', 'profiler'))

        # create summary writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.train_summary_writer = tf.summary.create_file_writer(logdir=os.path.join(args.output, 'logs', 'train'), filename_suffix=current_time)
        args.test_summary_writer = tf.summary.create_file_writer(logdir=os.path.join(args.output, 'logs', 'test'), filename_suffix=current_time)
        args.learning_rate_summary_writer = tf.summary.create_file_writer(logdir=os.path.join(args.output, 'logs', 'learning_rate'), filename_suffix=current_time)
        args.profile_log_dir = os.path.join(args.output, 'logs', 'profiler')


    # create ckpt directory
    def create_ckpt_dir():
        if not os.path.isdir(os.path.join(args.output, 'ckpts')):
            os.makedirs(os.path.join(args.output, 'ckpts'))

    check_output(), create_ckpt_dir(), create_logs()

def process_model(args):
    """
    Build model object
    """
    model = models._models[args.model]
    model = model(args)
    return model

