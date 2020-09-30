from .utils import get_args, process_folder, check_h5_ext
from .prepare_data import *
from .CreateDataset import *
from .TaxonomyGraph import GetLabelWeights
from .ReadSimulator import MP_DistributeGenomes

import sys
import numpy as np
import itertools

# Function that returns a dictionary of all possible kmers with
# each kmer associated with an integer
def kmer_dictionary(k_value):
    list_nt = ['A', 'T', 'C', 'G']
    # Get list of all possible 4**k_value kmers
    list_kmers = [''.join(p) for p in itertools.product(list_nt, repeat=k_value)]
    # Assign an integer to each kmer
    kmers_dict = dict(zip(list_kmers, range(len(list_kmers))))
    return kmers_dict

# Converts the DNA sequence into a vector of k-mers
def parse_seq(sequence, args):
    # Map every k bases
    list_kmers = []

    for n in range(len(sequence) - (args.kvalue - 1)):
        kmer = sequence[n:n + args.kvalue]
        # Lookup integer mapped to the kmer
        kmer_integer = args.kmers_dict[kmer]
        # Add kmer to vector of kmer
        list_kmers.append(kmer_integer)

    # transform list into numpy array
    array_int = np.asarray(list_kmers, dtype=np.int32)
    # Flip array
    array_int = np.flip(array_int, axis=0)
    return array_int

def parse_args():
    #### SET MODEL PARAMETERS #####
    models = ['kmer', 'bidirectional', 'multilstm', 'cnn', 'gru']
    ranks = ['species', 'genus', 'family', 'order', 'class']
    simulators = ['in-house', 'standard']
    parser = get_args('Processing reads for models that use kmers')
    parser.add_argument('model', help='Model type that will be trained', choices=models)
    parser.add_argument("-k", "--kvalue", type=int, help="size of kmer", required=True)

    parser.add_argument("-r", "--rank", type=str, help="taxonomic rank", choices=ranks, default='species')
    parser.add_argument("-rn", "--readsnum", type=int, help="desired number of reads per label", default=200000)
    parser.add_argument("-s", "--simulator", type=str, help=" type of read simulator", choices=simulators, required=True)

    # CNN and GRU support paired and unpaired reads
    parser.add_argument("-reads", help="Specify if unpaired or paired reads",
                        required=('cnn' in sys.argv or 'gru' in sys.argv),
                        choices=['paired', 'unpaired'])


    args = parser.parse_args()
    args.length = 150 - args.kvalue + 1
    args.input, args.output = process_folder(args)
    args.hdf5 = check_h5_ext(args.hdf5)

    # Set read type if not using CNN or GRU
    if args.reads is None:
        args.reads = 'unpaired' if args.model == 'kmer' else 'paired'
    print(f'Processing {args.reads} reads for {args.model}')
    return args

if __name__ == "__main__":
    args = parse_args()
    args.kmers_dict = kmer_dictionary(args.kvalue)
    # process reads generated with standard tools
    if args.simulator == 'standard':
        fastq_files, args.class_num = get_info(args)
        multiprocesses(fastq_files, args)

    # use in-house read simulator
    elif args.simulator == 'in-house':
        # create a new instance of dataset
        dataset = Dataset()
        # create graph with taxonomic lineages
        graph = dataset.CreateDataset(args)
        # Get label_weights and class_mapping dictionaries for each taxonomic rank
        graph.GetLabelWeights(args)
        # get dictionary of genomes with desired taxonomic rank
        tax_rank_integer = [key for (key, value) in graph.ranks.items() if value == args.rank][0]
        # get class_mapping dictionary
        class_mapping = load_json_dictionary('{}-integers.json'.format(args.rank), args.output)
        # Launch simulation
        MP_DistributeGenomes(class_mapping, graph.records[tax_rank_integer], args)
