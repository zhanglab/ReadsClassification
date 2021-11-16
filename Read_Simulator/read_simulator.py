import argparse
from mpi4py import MPI
from Functions_For_Read_Sim import *  # used for all of the functions in Functions_For_Read_Sim
from utils import *

def main():
    # create a communicator consisting of all the processors
    comm = MPI.COMM_WORLD
    # get the number of processors
    size = comm.Get_size()
    # get the rank of each processor
    rank = comm.Get_rank()
    print(comm, size, rank)
    # parse command line arguments
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
    # select genomes only on processor with rank 0
    if rank == 0:
        # loads species in dataset
        list_species = get_species(args)
        # select genomes
        genome_dict = select_genomes(args, list_species)  # gets the genome dictionary
        # get species with largest number of genomes
        # if args.num_mutate is None:
        #     needed_iterations = find_largest_genome_set(args, genome_dict)
        # else:
        #     needed_iterations = args.num_mutate
        # create dictionary mapping labels to species
        label_dict, needed_iterations = get_dataset_info(args, genome_dict, list_species)
        # split dictionary into N lists of dictionaries with N equal to the number of processes
        list_dict = [{} for i in range(size)]
        l_pos = 0
        for i in range(len(label_dict)):
            list_dict[l_pos][i] = label_dict[i]
            l_pos += 1
            if l_pos == size:
                l_pos = 0
        print(f'Rank: {rank}\n{label_dict}\n')
        print(f'Rank: {rank}\n{list_dict}\n{len(list_dict)}')
        num_items = 0
        for i in range(len(list_dict)):
            num_items += len(list_dict[i])
        print(f'Rank: {rank}\n{num_items}\n')
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
    print(f'Rank: {rank}\n{list_dict}\n{needed_iterations}\n')
    # start read simulation
    for label, species in list_dict.items():
        mutate_genomes(args, species, label, needed_iterations, genome_dict)

    generate reads for each species in parallel
    with mp.Manager() as manager:
        # create new processes
        processes = [mp.Process(target=mutate_genomes, args=(args, species, label, needed_iterations)) for label, species in args.label_dict.items()]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
