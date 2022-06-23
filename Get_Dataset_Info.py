from collections import defaultdict
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
def create_df(path, level_analysis):
    key_index = 1 if level_analysis == 'label' else 0
    value_index = 0 if level_analysis == 'label' else 1
    count = 0  # variable used to iterate through the dataframe
    df = pd.read_csv(path, sep='\t', header=None)  # converts the tsv file to a dataframe
    NCBI_dict = defaultdict(list)  # makes the default dictionary value a list

    # iterates through the dataframe to create the dictionary
    for key in df[key_index]:
        NCBI_dict[key].append(df[value_index][count])
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


def num_reads(list_fq_files, dict_num_reads, process_id, level_analysis):
    reads_dict = defaultdict(int)
    for fq_file in list_fq_files:
        with open(fq_file, 'r') as f:
            content = f.readlines()
        if level_analysis == 'label':
            data = [''.join(content[i:i + 4]).split('\n')[0].split('|')[1] for i in range(0, len(content), 4)]
        elif level_analysis == 'genome':
            data = [''.join(content[i:i + 4]).split('\n')[0].split('-')[0][1:] for i in range(0, len(content), 4)]
        for d in data:
            reads_dict[d] += 1
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

def summary(dict_num_reads, data_dict, outfilename, level_analysis, species_dict=None):
    out_f = open(outfilename, 'w')

    n_reads = 0
    for x, y in data_dict.items():
        x_nreads = 0
        for process_id, data in dict_num_reads.items():
            if level_analysis == 'label':
                x_nreads += data[str(x)]
                n_reads += data[str(x)]
            elif level_analysis == 'genome':
                for s in y:
                    if s in data:
                        x_nreads += data[s]
                        n_reads += data[s]
        if level_analysis == 'label':
            out_f.write(f'{x}\t{len(y)}\t{x_nreads}\t{species_dict[str(x)]}\n')
        elif level_analysis == 'genome':
            out_f.write(f'{x}\t{len(y)}\t{x_nreads}\n')

    print(f'# reads: {n_reads}')


def main():

    data_info_path = sys.argv[1] # path to file containing list of genomes mapped to species labels and sequence ids
    fq_path = sys.argv[2] # path to directory containing fastq files in training or testing set
    json_path = sys.argv[3] # path to species_labels_full_taxonomy.json dictionary
    output_dir = sys.argv[4] # path to output directory
    dataset_type = sys.argv[5] # testing or training datasets
    level_analysis = sys.argv[6] # label or genome --> # reads are reported per genome or per label
    n_processes = int(sys.argv[7])

    species_dict = load_json_dict(json_path)
    data_dict = create_df(data_info_path, level_analysis)
    n_genomes = sum([len(v) for v in data_dict.values()])
    print(f'# genomes in {dataset_type} set: {n_genomes}')
    fq_files = sorted(glob.glob(os.path.join(fq_path, "*-reads.fq")))
    print(f'# fq files: {len(fq_files)}')
    fq_files_per_process = split_data(fq_files, n_processes)
    n_fq_files = sum([len(l) for l in fq_files_per_process])
    print(f'# fq files: {n_fq_files}\t# processes: {n_processes} - {len(fq_files_per_process)}\t# cpus on node: {mp.cpu_count()}')

    outfilename = os.path.join(output_dir, f'{level_analysis}-{dataset_type}-info.tsv')

    # get number of reads per label or sequence
    with mp.Manager() as manager:
        dict_num_reads = manager.dict()
        processes = [mp.Process(target=num_reads, args=(fq_files_per_process[i], dict_num_reads, i, level_analysis)) for i in
                     range(len(fq_files_per_process))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        if level_analysis == 'label':
            summary(dict_num_reads, data_dict, outfilename, level_analysis, species_dict)
        else:
            summary(dict_num_reads, data_dict, outfilename, level_analysis)


    # processes = [mp.Process(target=num_reads, args=(fq_path, list_label_dict[i], species_dict, output_dir, i)) for i in
    #              range(len(list_label_dict))]
    # for p in processes:
    #     p.start()
    # for p in processes:
    #     p.join()


if __name__ == '__main__':
    main()
