import sys
import os
import glob
# import multiprocess as mp
# from multiprocess import Pool
from mpi4py import MPI
import pandas as pd
from collections import defaultdict

# create a communicator consisting of all the processors
comm = MPI.COMM_WORLD
# get the number of processors
size = comm.Get_size()
# get the rank of each processor
rank = comm.Get_rank()
print(comm, size, rank)

def get_taxonomy(gtdb_info, test_genomes, train_genomes):
    # load gtdb information
    gtdb_df = pd.read_csv(gtdb_info, delimiter='\t', low_memory=False)
    # get species in database
    taxonomy = [i.split(';')[::-1] for i in list(gtdb_df['gtdb_taxonomy'])]
    # get genome accession ids
    accession_id_list = [i[3:] for i in list(gtdb_df.accession)]
    # get taxonomy
    genomes_taxonomy = {accession_id_list[i]: taxonomy[i] for i in range(len(accession_id_list)) if accession_id_list[i] in test_genomes or accession_id_list[i] in train_genomes}
    print(f'{len(taxonomy)} - {len(accession_id_list)} - {len(genomes_taxonomy)} - {len(test_genomes)+len(train_genomes)}')
    return genomes_taxonomy

def get_genomes(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
        genomes = [i.rstrip().split('\t')[0] for i in content]
    return genomes

def get_avg_distance(test_genome, dict_mash_dist, genomes_taxonomy, list_train_genomes, r_index, mash_dir):
    # get train genomes with same taxon as test genome
    train_genomes = [i for i in list_train_genomes if genomes_taxonomy[i][int(r_index)].split('__')[1] == genomes_taxonomy[test_genome][int(r_index)].split('__')[1]]
    print(f'Number of train genomes associated with test genome {test_genome}: {len(train_genomes)} - taxon: {genomes_taxonomy[test_genome][int(r_index)].split("__")[1]}')
    avg_msh_dist = float()
    for train_g in train_genomes:
        avg_msh_dist += dict_mash_dist[test_genome][train_g]
    print(f'{mp.current_process()}\t{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}')
    with open(os.path.join(output_path, f'{test_genome}-avg-mash-dist'), 'w') as f:
        f.write(f'{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}\n')
        f.write(f'Species\t{genomes_taxonomy[test_genome]}\n')
        for train_g in train_genomes:
            f.write(f'Training genome\t{train_g}\n')

def get_mash_distances(mash_dir, test_genomes, train_genomes):
    dict_mash_dist = defaultdict(dict) # key = test genome, value = list of mash distances to training genomes
    for test_g in test_genomes:
        for train_g in train_genomes:
            with open(os.path.join(mash_dir, f'{test_g}-{train_g}-mash-dist'), 'r') as f:
                for line in f:
                    dict_mashs_dist[test_g][train_g] = float(line.rstrip().split('\t')[2])
    return dict_mash_dist

def main():
    mash_dir = sys.argv[1]
    list_train_genomes = sys.argv[2]
    list_test_genomes = sys.argv[3]
    gtdb_info = sys.argv[4]
    if rank == 0:
        # get training and testing genomes
        test_genomes = get_genomes(list_test_genomes)
        train_genomes = get_genomes(list_train_genomes)
        # get mash distances between testing and training genomes
        dict_mash_dist = get_mash_distances(mash_dir, test_genomes, train_genomes)
        genomes_taxonomy = get_taxonomy(gtdb_info, test_genomes, train_genomes)
    else:
        train_genomes = None
        dict_mash_dist = None
        genomes_taxonomy = None
    # broadcast list of training genomes to all processes
    train_genomes = comm.bcast(train_genomes, root=0)
    # broadcast dictionaru holding mash distances to all processes
    dict_mash_dist = comm.bcast(dict_mash_dist, root=0)
    # broadcast taxonomy information
    genomes_taxonomy = comm.bcast(genomes_taxonomy, root=0)
    # tax_ranks = {'0': 'species', '1': 'genus', '2': 'family', '3': 'order', '4': 'class'}
    tax_ranks = {'2': 'family', '3': 'order', '4': 'class'}
    for r_index, r_name in tax_ranks.items():
        output_path = os.path.join(mash_dir, f'mash-avg-dist-{r_name}')
        if rank == 0:
            # create directory to store mash average distances
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            else:
                # get list of testing genomes done
                test_genomes_done = ['-'.join(i.split('/')[-1].split('-')[0]) for i in sorted(glob.glob(os.path.join(mash_dir, f'mash-avg-dist-{r_name}', f'*-avg-mash-dist')))]
                # get list of testing genomes to do
                test_genomes_to_do = set(list(dict_mash_dict.keys())).intersection(set(test_genomes_done))
                print(f'{r_name} - genomes to do: {len(test_genomes_to_do)}')
                # assign test genomes to all processes
                test_genomes_per_processes = [[] for i in range(size)]
                print(len(test_genomes_per_processes))
                num_process = 0
                for i in range(len(test_genomes_to_do)):
                    test_genomes_per_processes[num_process].append(test_genomes_to_do[i])
                    num_process += 1
                    if num_process == size:
                        num_process = 0
        else:
            test_genomes_per_processes = None
        # scatter list of genomes to all processes
        test_genomes_per_processes = comm.scatter(test_genomes_per_processes, root=0)
        print(f'Rank: {rank}\n{test_genomes_per_processes}\n')
        # get average mash distance for each genome
        for test_g in test_genomes_per_processes:
            get_avg_distance(test_g, dict_mash_dist, genomes_taxonomy, train_genomes, r_index, mash_dir)




        # get average mash distance for each testing genome
        # pool = Pool(processes=mp.cpu_count())
        # for test_g in test_genomes:
        #     pool.apply_async(target=get_avg_distance, args=(test_g, train_genomes, mash_dir, output_path, genomes_taxonomy, r_index))
        # pool.close()
        # pool.join()
        # processes = [mp.Process(target=get_avg_distance, args=(test_g, train_genomes, mash_dir, output_path, genomes_taxonomy, r_index)) for test_g in test_genomes]
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()


if __name__ == "__main__":
    main()
