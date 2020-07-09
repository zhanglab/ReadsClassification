from .utils import get_args, process_folder
from .prepare_data import *

base_dict = {'A':0, 'C':1, 'T':2, 'G':3}

# Converts the DNA sequence into one-hot encoded sequence
def parse_seq(sequence, args):
    integer_encoded = [base_dict[base] for base in sequence]
    return integer_encoded

def parse_args():
    #### SET MODEL PARAMETERS #####
    parser = get_args()
    parser.add_argument("-l", "--length", type=int, help="sequence length", default=150)
    args = parser.parse_args()
    args.input, args.output = process_folder(args)
    args.reads = 'unpaired'
    args.model = 'base-emb'
    return args

if __name__ == '__main__':
    args = parse_args()
    fastq_files = get_info(args)
    multiprocesses(fastq_files, args)