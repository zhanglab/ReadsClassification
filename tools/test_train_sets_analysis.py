import sys
import os
import multiprocess as mp
import pandas as pd

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

def get_avg_distance(test_genome, list_train_genomes, mash_dir, output_path, genomes_taxonomy, r_index):
    # get train genomes with same species as test genome
    train_genomes = [i for i in list_train_genomes if genomes_taxonomy[i][int(r_index)].split('__')[1] == genomes_taxonomy[test_genome][int(r_index)].split('__')[1]]
    print(f'Number of train genomes associated with test genome {test_genome}: {len(train_genomes)} - taxon: {genomes_taxonomy[test_genome][int(r_index)].split("__")[1]}')
    avg_msh_dist = float()
    for train_g in train_genomes:
        with open(os.path.join(mash_dir, f'{test_genome}-{train_g}-mash-dist'), 'r') as f:
            for line in f:
                avg_msh_dist += float(line.rstrip().split('\t')[2])
    print(f'{mp.current_process()}\t{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}')
    with open(os.path.join(output_path, f'{test_genome}-avg-mash-dist'), 'w') as f:
        f.write(f'{test_genome}\t{round(float(avg_msh_dist)/len(train_genomes), 5)}\n')
        f.write(f'Species\t{genomes_taxonomy[test_genome]}\n')
        for train_g in train_genomes:
            f.write(f'Training genome\t{train_g}\n')

def main():
    mash_dir = sys.argv[1]
    job_id = int(sys.argv[2])
    list_train_genomes = sys.argv[3]
    list_test_genomes = sys.argv[4]
    gtdb_info = sys.argv[5]
    # get training and testing genomes
    test_genomes = get_genomes(list_test_genomes)
    train_genomes = get_genomes(list_train_genomes)
    # ranks = {'0': 'species', '1': 'genus', '2': 'family', '3': 'order', '4': 'class'}
    ranks = {'1': 'genus', '2': 'family', '3': 'order', '4': 'class'}
    genomes_taxonomy = get_taxonomy(gtdb_info, test_genomes, train_genomes)
    for r_index, r_name in ranks.items():
        # create directory to store mash average distances
        output_path = os.path.join(mash_dir, f'mash-avg-dist-{r_name}')
        if job_id == 0:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        # get average mash distance for each testing genome
        processes = [mp.Process(target=get_avg_distance, args=(test_g, train_genomes, mash_dir, output_path, genomes_taxonomy, r_index)) for test_g in test_genomes]
        for p in processes:
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
