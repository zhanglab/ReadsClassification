import pandas as pd
import os
import sys
import glob
import numpy as np
import json
import argparse
from collections import defaultdict


def get_prob(probs, results, count_all, count_ab_thld):
    num_reads = 0
    num_reads_abv_thld = 0
    for r in results:
        df = pd.read_csv(r, delimiter='\t', header=None)
        list_probs = df.iloc[:,1].tolist()
        labels = df.iloc[:,0].tolist()
        print(r, len(list_probs))
        for i in range(len(list_probs)):
            probs[labels[i]].append(list_probs[i])
            count_all[labels[i]] += 1
            if list_probs[i] >= 0.5:
                count_ab_thld[labels[i]] += 1
                num_reads_abv_thld += 1
        num_reads += len(list_probs)

    print(f'#reads: {num_reads}\n{num_reads_abv_thld}')


def combine_data(args, results_prob):
    with open(os.path.join(args.output_dir, 'meta-all-probabilites'), 'w') as out_f:
        for r in results:
            with open(r, 'r') as in_f:
                out_f.write(in_f.read())

def create_summary(args, data, probs, species_mapping_dict, outfilename, condition=False):
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1])}
    with open(outfilename, 'w') as out_f:
        for key, value in sorted_data.items():
            if condition:
                new_probs = [i for i in probs[key] if i >= 0.5]
            else:
                new_probs = probs[key]
            out_f.write(f'{key}\t{species_mapping_dict[key]}\t{value}\t{min(new_probs)}\t{max(new_probs)}\t{np.median(new_probs)}\t{np.mean(new_probs)}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='directory containing json dictionary files mapping taxa to labels', required=True)
    parser.add_argument('--results_dir', type=str, help='directory containing dl toda results', required=True)
    parser.add_argument('--output_dir', type=str, help='output directory', required=True)
    args = parser.parse_args()

    # load dictionary mapping labels to species
    with open(os.path.join(args.input_dir, 'species_labels.json'), 'r') as f:
        species_mapping_dict = json.load(f)

    # get softmax layer output from each GPU
    results_prob = sorted(glob.glob(os.path.join(args.results_dir, '*-prob.tsv')))
    print(results_prob)

    # get probabilities
    probs = defaultdict(list) # key = species label, value = list of predicted probabilities
    count_all = defaultdict(int)
    count_ab_thld = defaultdict(int)
    get_prob(probs, results_prob, count_all, count_ab_thld)
    # create summary file
    create_summary(args, count_all, probs, species_mapping_dict, os.path.join(args.output_dir, 'meta-summary-report-all.tsv'))
    create_summary(args, count_ab_thld, probs, species_mapping_dict, os.path.join(args.output_dir, 'meta-summary-report-abv-0.5.tsv'), True)
    # create files combining all probabilites
    combine_data(args, results_prob)


if __name__ == "__main__":
    main()
