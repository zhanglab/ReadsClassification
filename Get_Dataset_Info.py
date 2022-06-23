from collections import defaultdict, Counter
import sys
import pandas as pd
import os
import glob
import multiprocessing as mp
import json
import math

# load json format dictionary
def load_json_dict(json_path):
    with open(json_path, 'r') as f:
        dict = json.load(f)

    return dict

# This function creates a dataframe from a tsv file. Then, we iterate through the dataframe to make a dictionary.
# The species labels are the keys and the values are a list filled with respective genome associations.
def create_df(path):
    count = 0  # variable used to iterate through the dataframe
    df = pd.read_csv(path, sep='\t', header=None)  # converts the tsv file to a dataframe
    NCBI_dict = defaultdict(list)  # makes the default dictionary value a list

    # iterates through the dataframe to create the dictionary
    for key in df[1]:
        NCBI_dict[key].append(df[0][count])
        count += 1

    return NCBI_dict

# Finds the length of each list in a dictionary and associates this new integer to the key of the previous list
def create_dict_count(dict_list):
    temp = {}
    for key in dict_list:
        temp[key] = len(dict_list[key])

    return temp

# Counts the number of reads associated with each species label and makes a new dictionary with this value
# def num_reads(path, genome_dict, species_dict, output_dir, process_id):
    # reads_dict = defaultdict(int)
    # with open(os.path.join(output_dir, f'results-process-id-{process_id}'), 'w') as out_f:
    #     for label, genome_list in genome_dict.items():
    #         out_f.write(f'{label}\t{species_dict[str(label)]}\t{len(genome_list)}\t')
    #         for genome_fq in genome_list:
    #             with open(os.path.join(path, (genome_fq + '.fq')), 'r') as f:
    #                 content = f.readlines()
    #             records = [''.join(content[i:i + 4]) for i in range(0, len(content), 4)]
    #             reads_dict[label] += math.ceil(len(records) * .70)  # Only 70% of the reads are used for training
    #         out_f.write(f'{reads_dict[label]}\n')


def num_reads(list_fq_files, dict_num_reads, process_id):
    reads_dict = defaultdict(int)
    for fq_file in list_fq_files:
        if process_id == 0:
            print(process_id, fq_file)
        with open(fq_file, 'r') as f:
            content = f.readlines()
        records = [''.join(content[i:i + 4]) for i in range(0, len(content), 4)]
        # labels = [r.split('\n')[0].split('|')[1] for r in records]
        # num_reads_in_file = Counter(labels)
        if process_id == 0:
            print(process_id, fq_file, len(records))
        # for k, v in num_reads_in_file.items():
        #     reads_dict[k] += v

    dict_num_reads[process_id] = reads_dict

# split dictionary into sub dictionaries with one dictionary per process
def split_data(input_list: list, num_parts: int) -> list:
    list_len: int = len(input_list)
    return [input_list[i * list_len // num_parts:(i + 1) * list_len // num_parts]
            for i in range(num_parts)]

# # split dictionary into sub dictionaries with one dictionary per process
# def split_dict(input_dict: dict, num_parts: int) -> list:
#     list_len: int = len(input_dict)
#     return [dict(list(input_dict.items())[i * list_len // num_parts:(i + 1) * list_len // num_parts])
#             for i in range(num_parts)]

def main():

    NCBI_info_path = sys.argv[1] # path to file containing list of genomes mapped to species labels
    fq_path = sys.argv[2] # path to directory containing fastq files in training or testing set
    json_path = sys.argv[3] # path to species_labels_full_taxonomy.json dictionary
    output_dir = sys.argv[4] # path to output directory
    dataset_type = sys.argv[5] # testing or training datasets
    level_analysis = sys.argv[6] # label or genome --> # reads are reported per genome or per label

    species_dict = load_json_dict(json_path)
    labels_dict = create_df(NCBI_info_path)
    n_genomes = sum([len(v) for v in labels_dict.values()])
    print(f'# genomes in {dataset_type} set: {n_genomes}')
    fq_files = sorted(glob.glob(os.path.join(fq_path, "*-reads.fq")))
    print(f'# fq files: {len(fq_files)}')
    fq_files_per_process = split_data(fq_files, mp.cpu_count())
    n_fq_files = sum([len(l) for l in fq_files_per_process])
    print(f'# fq files: {n_fq_files}\t# processes: {len(fq_files_per_process)}\t# processes: {mp.cpu_count()}')

    # get number of reads per label or sequence
    with mp.Manager() as manager:
        dict_num_reads = manager.dict()
        processes = [mp.Process(target=num_reads, args=(fq_files_per_process[i], dict_num_reads, i)) for i in
                     range(len(fq_files_per_process))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        n_reads = 0
        for process_id, data in dict_num_reads.items():
            n_reads += sum(data.values())

        print(f'# reads in {dataset_type}: {n_reads}')

    # processes = [mp.Process(target=num_reads, args=(fq_path, list_label_dict[i], species_dict, output_dir, i)) for i in
    #              range(len(list_label_dict))]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()


if __name__ == '__main__':
    main()
