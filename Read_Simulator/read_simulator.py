import argparse
from mpi4py import MPI
from fn_for_read_simulator import *  # used for all of the functions in Functions_For_Read_Sim
from utils import *

def main():
    # create a communicator consisting of all the processors
    comm = MPI.COMM_WORLD
    # get the number of processors
    size = comm.Get_size()
    # get the rank of each processor
    rank = comm.Get_rank()
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='path containing species.tsv file')
    parser.add_argument('--gtdb_path', type=str, help='path to GTDB database')
    parser.add_argument('--ncbi_path', type=str, help='path to NCBI database')
    parser.add_argument('--gtdb_info', type=str, help='path to file with info on GTDB')
    parser.add_argument('--genetic_code', type=str, help='path to file containing the genetic code')
    parser.add_argument('--num_mutate', type=int, help='number of times a genome should be mutated')
    parser.add_argument('--train_genomes', type=int, help='number of training genomes')
    parser.add_argument('--mutate', action='store_true', default=False)
    parser.add_argument('--overlap', action='store_true', default=False)
    args = parser.parse_args()
    # load genetic code
    args.codon_amino, args.amino_codon = get_genetic_code(args)
    # select genomes only on processor with rank 0
    if rank == 0:
        # loads species in dataset
        list_species = get_species(args)
        # select genomes
        genome_dict, gtdb_taxonomy = select_genomes(args, list_species) # gets the genome dictionary
        # create dictionary mapping labels to species
        label_dict, needed_iterations = get_dataset_info(args, genome_dict, list_species, gtdb_taxonomy)
        if args.num_mutate is not None:
            needed_iterations = args.num_mutate
        # split dictionary into N lists of dictionaries with N equal to the number of processes
        list_dict = [{} for i in range(size)]
        l_pos = 0
        for i in range(len(label_dict)):
            list_dict[l_pos][i] = label_dict[i]
            l_pos += 1
            if l_pos == size:
                l_pos = 0

        num_items = 0
        for i in range(len(list_dict)):
            num_items += len(list_dict[i])
    else:
        needed_iterations = None
        list_dict = None
        genome_dict = None

    # broadcast the needed_iterations variable to all processes
    needed_iterations = comm.bcast(needed_iterations, root=0)
    # broadcast the dictionary of genomes to all processes
    genome_dict = comm.bcast(genome_dict, root=0)
    # scatter dictionary to all processes
    list_dict = comm.scatter(list_dict, root=0)
    # start read simulation
    for label, species in list_dict.items():
        mutate_genomes(args, species, label, needed_iterations, genome_dict)


if __name__ == '__main__':
    main()
