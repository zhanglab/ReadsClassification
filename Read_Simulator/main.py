import argparse
import multiprocess as mp
from Functions_For_Read_Sim import *  # used for all of the functions in Functions_For_Read_Sim
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path containing species.tsv file')
    parser.add_argument('--gtdb_path', type=str, help='path to GTDB database')
    parser.add_argument('--ncbi_path', type=str, help='path to NCBI database')
    parser.add_argument('--gtdb_info', type=str, help='path to file with info on GTDB')
    parser.add_argument('--genetic_code', type=str, help='path to file containing the genetic code')
    parser.add_argument('--num_mutate', type=int, help='number of times a genome should be mutated')
    parser.add_argument('--mutate', action='store_true', default=False)
    args = parser.parse_args()

    # load genetic code
    args.codon_amino, args.amino_codon = get_genetic_code(args)
    # loads species in dataset
    args.list_species = get_species(args)
    # select genomes
    args.genome_dict = select_genomes(args)  # gets the genome dictionary
    # get species with largest number of genomes
    if args.num_mutate is None:
        needed_iterations = find_largest_genome_set(args)
    else:
        needed_iterations = args.num_mutate
    # create dictionary mapping labels to species
    args.label_dict = get_dataset_info(args)
    # generate reads for each species in parallel
    with mp.Manager() as manager:
        # create new processes
        processes = [mp.Process(target=mutate_genomes, args=(args, species, label, needed_iterations)) for label, species in args.label_dict.items()]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
