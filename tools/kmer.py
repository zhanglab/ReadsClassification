from .utils import get_args, process_folder
from .prepare_data import *

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
    array_int = np.asarray(list_kmers)
    # Flip array
    array_int = np.flip(array_int, axis=0)
    return array_int

def parse_args():
    #### SET MODEL PARAMETERS #####
    models = ['kmer', 'bidirectional', 'multilstm', 'CNN', 'GRU']
    parser = get_args('Processing reads for models that use kmers')
    parser.add_argument('model', help='Model type that will be trained', choices=models)
    parser.add_argument("-k", "--kvalue", type=int, help="size of kmer", required=True)

    # CNN and GRU support paired and unpaired reads
    parser.add_argument("-reads", help="Specify if unpaired or paired reads",
                        required=(model in ('CNN', 'GRU')))


    args = parser.parse_args()
    args.length = 150 - args.kvalue + 1
    args.input, args.output = process_folder(args)

    # Set read type if not using CNN or GRU
    if args.reads is None:
        args.reads = 'unpaired' if args.model == 'kmer' else 'paired'

    return args

if __name__ == "__main__":
    args = parse_args()
    fastq_files = get_info(args)
    args.kmers_dict = kmer_dictionary(args.kvalue)
    multiprocesses(fastq_files, args)