import sys
import os
import multiprocess as mp

def get_genomes(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
        genomes = [i.rstrip().split('\t')[0] for i in content]
    return genomes

def get_avg_distance(test_genome, train_genomes, mash_dir):
    avg_msh_dist = float()
    for train_g in train_genomes:
        with open(os.path.join(mash_dir, f'{test_genome}-{train_genome}-mash-dist'), 'r') as f:
            for line in f:
                avg_msh_dist += float(line.rstrip().split('\t')[2])
    print(f'{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}')
    with open(os.path.join(mash_dir, f'{test_genome}-avg-mash-dist'), 'w') as f:
        f.write(f'{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}\n')

def main():
    mash_dir = sys.argv[1]
    list_train_genomes = sys.argv[2]
    list_test_genomes = sys.argv[3]
    # get training and testing genomes
    train_genomes = get_genomes(list_train_genomes)
    test_genomes = get_genomes(list_test_genomes)
    # get average mash distance for each testing genome
    processes = [mp.Process(target=get_avg_distance, args=(test_g, train_genomes, mash_dir)) for test_g in test_genomes]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
