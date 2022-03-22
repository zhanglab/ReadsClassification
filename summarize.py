import sys
import os
import argparse
import itertools
import glob
import multiprocessing as mp
from summarize_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to temporary directory containing results files obtained from running read_classifier.py', required=True)
    parser.add_argument('--data_type', type=str, help='input data type', required=True, choices=['sim', 'meta'])
    parser.add_argument('--sample_size', type=int, help='size of sample for ROC curve analysis', required=('--roc' in sys.argv))
    parser.add_argument('--tpr', type=float, help='desired true positive rate (number between 0 and 1)', required=('--roc' in sys.argv), default=0.9)
    parser.add_argument('--rank_mapping_dir', type=str, help='path to json files containing dictionaries mapping taxa to labels', required=True)
    parser.add_argument('--thresholds_dir', type=str, help='path to directory containing decision thresholds at every taxonomic level', required=('--meta' in sys.argv))
    parser.add_argument('--roc', help='option to generate decision thresholds with ROC curves', action='store_true', required=('sim' in sys.argv))
    args = parser.parse_args()

    args.NUM_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK"))
    print(f'# cpus: {args.NUM_CPUS}')

    args.ranks = ['species', 'genus', 'family', 'order', 'class']

    if args.data_type == 'sim':
        if args.roc:
            # compute decision thresholds
            # load files of probabilities and ground truth
            list_prob_files = sorted(glob.glob(os.path.join(args.input_dir, '*-prob-out.npy')))
            list_labels_files = sorted(glob.glob(os.path.join(args.input_dir, '*-labels-out.npy')))
            print(len(list_prob_files), len(list_labels_files))
            probs, labels = get_results_from_npy(args.sample_size, list_prob_files, list_labels_files)
            print(len(probs), len(labels))
            # for r in args.ranks:
                # get_decision_thds(args, r, probs, labels)
        else:
            # compute accuracy, precision, recall
            # get predictions and ground truth at species level
            tsv_files = sorted(glob.glob(os.path.join(args.input_dir, '*.tsv')))
            pred_species, true_species, probs = get_results_from_tsv(args, tsv_files)
            pool = mp.pool.ThreadPool(args.NUM_CPUS)
            results = pool.starmap(get_metrics, zip(itertools.repeat(args, len(args.ranks)), itertools.repeat(true_species, len(args.ranks)),
            itertools.repeat(pred_species, len(args.ranks)), args.ranks))
            pool.close()
            pool.join()

    else:
        tsv_files = sorted(glob.glob(os.path.join(args.input_dir, '*.tsv')))
        load_decision_thds(args)
        load_mapping_dict(args)
        summary_dict = {}
        pool = mp.pool.ThreadPool(args.NUM_CPUS)
        results = pool.starmap(get_taxa_rel_abundance, zip(itertools.repeat(args, len(tsv_files)), tsv_files, itertools.repeat(summary_dict, len(tsv_files))))
        pool.close()
        pool.join()

        with open(os.path.join(args.input_dir, 'summary'), 'w') as out_f:
            out_f.write('sample\tspecies\tgenus\tfamily\torder\tclass\ttotal\n')
            for k, v in summary_dict.items():
                out_f.write(f'{k}\t')
                for r in args.ranks:
                    out_f.write(f'{v[r]}\t')
                out_f.write(f'{v["total"]}\n')


if __name__ == "__main__":
    main()
