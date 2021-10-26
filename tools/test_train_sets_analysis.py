import sys
import os
import multiprocess as mp

def get_genomes(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
        genomes = [i.rstrip().split('\t')[0] for i in content]
    return genomes

def get_avg_distance(test_genome, train_genomes, mash_dir, output_path):
    avg_msh_dist = float()
    for train_g in train_genomes:
        with open(os.path.join(mash_dir, f'{test_genome}-{train_g}-mash-dist'), 'r') as f:
            for line in f:
                avg_msh_dist += float(line.rstrip().split('\t')[2])
    print(f'{mp.current_process()}\t{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}')
    with open(os.path.join(output_path, f'{test_genome}-avg-mash-dist'), 'w') as f:
        f.write(f'{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}\n')

def main():
    mash_dir = sys.argv[1]
    job_id = int(sys.argv[2])
    list_train_genomes = sys.argv[3]
    list_test_genomes = sys.argv[4]
    # get training and testing genomes
    train_genomes = get_genomes(list_train_genomes)
    test_genomes = get_genomes(list_test_genomes)
    print(f'number of processes: {len(test_genomes)}')
    # create directory to store mash average distances
    output_path = os.path.join(mash_dir, 'mash-avg-dist')
    if job_id == 0:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    # get average mash distance for each testing genome
    processes = [mp.Process(target=get_avg_distance, args=(test_g, train_genomes, mash_dir, output_path)) for test_g in test_genomes]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
