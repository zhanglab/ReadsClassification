import os
import glob
from random import randint
from collections import defaultdict, Counter
import itertools
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def get_results_from_tsv(args, tsv_files):
    pred_species = []
    true_species = []
    probs = []
    for in_file in tsv_files:
        df = pd.read_csv(in_file, header=None, sep='\t')
        true_species += df.iloc[:,0].tolist()
        pred_species += df.iloc[:,1].tolist()
        probs += df.iloc[:,2].tolist()

    return pred_species, true_species, probs


def get_results_from_npy(sample_size, list_prob_files, list_labels_files):
    # load arrays of probabilities and labels
    probs = []
    labels = []
    for i in range(len(list_prob_files)):
        probs.append(np.load(list_prob_files[i]))
        labels += np.load(list_labels_files[i]).tolist()
        if len(labels) >= sample_size:
            break
    probs = np.concatenate(probs)
    return probs, labels

def load_json_dict(args, filename):
    with open(os.path.join(filename), 'r') as f:
        dict = json.load(f)
    return dict

def load_decision_thds(args):
    args.decision_thresholds_dict = {}
    args.decision_thresholds_dict['species'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'species-decision-thresholds.json'))
    args.decision_thresholds_dict['genus'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'genus-decision-thresholds.json'))
    args.decision_thresholds_dict['family'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'family-decision-thresholds.json'))
    args.decision_thresholds_dict['order'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'order-decision-thresholds.json'))
    args.decision_thresholds_dict['class'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'class-decision-thresholds.json'))

def load_mapping_dict(args):
    args.labels_mapping_dict = {}
    args.labels_mapping_dict['species'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'species_labels.json'))
    args.labels_mapping_dict['genus'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'genus_labels.json'))
    args.labels_mapping_dict['family'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'family_labels.json'))
    args.labels_mapping_dict['order'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'order_labels.json'))
    args.labels_mapping_dict['class'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'class_labels.json'))
    args.rank_species_mapping = {}
    args.rank_species_mapping['genus'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'genus_species_labels.json'))
    args.rank_species_mapping['family'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'family_species_labels.json'))
    args.rank_species_mapping['order'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'order_species_labels.json'))
    args.rank_species_mapping['class'] = load_json_dict(args, os.path.join(args.rank_mapping_dir, 'class_species_labels.json'))

def get_decision_thds(args, rank, probs, labels):
    # load species mapping labels dictionary
    labels_mapping_dict = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_labels.json'))
    if rank != 'species':
        # load dictionary mapping species to given rank labels`
        rank_species_mapping = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_species_labels.json'))
        labels = [rank_species_mapping[str(i)] for i in labels]

    labels_in_test_set = list(set(labels))
    print(len(labels_in_test_set))

    # get number of occurrences of each label
    counter = Counter(labels)
    with open(os.path.join(args.input_dir, f'counter_{rank}.json'), 'w') as out_f:
        json.dump(counter, out_f)

    # compute decision threshold for each label in the test set
    decision_thresholds = {}
    pool = mp.pool.ThreadPool(args.NUM_CPUS)
    results = pool.starmap(ROCcurve, zip(itertools.repeat(args, len(labels_in_test_set)), itertools.repeat(probs, len(labels_in_test_set)), itertools.repeat(labels, len(labels_in_test_set)), labels_in_test_set, itertools.repeat(counter, len(labels_in_test_set)), itertools.repeat(rank, len(labels_in_test_set)), itertools.repeat(decision_thresholds, len(labels_in_test_set))))
    pool.close()
    pool.join()
    # manager = mp.Manager()
    # decision_thresholds = manager.dict()
    # pool = mp.Pool(args.NUM_CPUS)
    # pool.starmap(ROCcurve, zip(itertools.repeat(args, len(labels_in_test_set)), itertools.repeat(probs, len(labels_in_test_set)), itertools.repeat(labels, len(labels_in_test_set)), labels_in_test_set, itertools.repeat(counter, len(labels_in_test_set)), itertools.repeat(rank, len(labels_in_test_set)), itertools.repeat(decision_thresholds, len(labels_in_test_set))))
    # pool.close()
    # pool.join()

    with open(os.path.join(args.input_dir, f'{rank}-decision-thresholds.json'), 'w') as f:
        json.dump(decision_thresholds, f)

def get_taxa_rel_abundance(args, tsv_file, summary_dict):
    pred_taxa, _, probs = get_results_from_tsv(args, [tsv_file])
    sample = tsv_file.split("/")[-1].split("-")[0].split("_")[-1]
    summary_dict[sample] = {}
    summary_dict[sample]['total'] = len(pred_taxa)
    outfile = os.path.join(args.input_dir, tsv_file.split("/")[-1].split('.')[0])
    for r in args.ranks:
        if r != 'species':
            pred_taxa = [args.rank_species_mapping[r][str(i)] for i in labels]

        rel_abundance = defaultdict(int)
        for i in range(len(pred_taxa)):
            if probs[i] >= args.decision_thresholds_dict[r][pred_taxa[i]]:
                rel_abundance[pred_taxa[i]] += 1

        with open(f'{outfile}-{r}-rel-abundance', 'w') as f:
            for k, v in rel_abundance.items():
                f.write(f'{args.labels_mapping_dict[r][str(k)]}\t{v}\t{len(pred_taxa)}\t{round(v/len(pred_taxa),5)}\n')
        summary_dict[sample][r] = sum(list(rel_abundance.values()))


def get_cm(true_taxa, predicted_taxa, rank_mapping_dict, rank):
    # create empty confusion matrix with rows = true classes and columns = predicted classes
    cm = np.zeros((len(rank_mapping_dict), len(rank_mapping_dict)))
    num_correct_pred = 0
    num_incorrect_pred = 0
    # fill out confusion matrix
    for i in range(len(true_taxa)):
        cm[true_taxa[i], predicted_taxa[i]] += 1
        if true_taxa[i] == predicted_taxa[i]:
            num_correct_pred += 1
        else:
            num_incorrect_pred += 1
    # accuracy = round(float(num_correct_pred)/len(true_taxa), 5)
    return cm

def get_metrics(args, true_taxa, predicted_taxa, rank):
    """ precision = True Positives / (True Positives + False Positives) """
    """ recall = True Positives / (True Positives + False Negatives) """
    """ accuracy = number of correct predictions / (number of reads in testing set) """

    labels_mapping_dict = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_labels.json'))

    if rank != 'species':
        # load dictionary mapping species labels to other ranks labels
        rank_species_mapping = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_species_labels.json'))
        # get vectors of predicted and true labels at given rank
        predicted_taxa = [rank_species_mapping[str(i)] for i in predicted_taxa]
        true_taxa = [rank_species_mapping[str(i)] for i in true_taxa]

    cm = get_cm(true_taxa, predicted_taxa, labels_mapping_dict, rank)

    f = open(os.path.join(args.input_dir, f'{rank}-metrics-classification-report.tsv'), 'w')
    f.write(f'{rank}\tprecision\trecall\tnumber\n')
    correct_predictions = 0
    total_num_reads = 0
    # get precision and recall for all species in testing set
    for i in set(true_taxa):
        true_positives = cm[i, i]
        other_labels = [j for j in range(len(labels_mapping_dict)) if j != i]
        false_positives = sum([cm[j, i] for j in other_labels])
        false_negatives = sum([cm[i, j] for j in other_labels])
        correct_predictions += true_positives
        num_testing_reads = sum([cm[i, j] for j in range(len(labels_mapping_dict))])
        total_num_reads += num_testing_reads
        precision = float(true_positives)/(true_positives+false_positives)
        recall = float(true_positives)/(true_positives+false_negatives)
        f.write(f'{labels_mapping_dict[str(i)]}\t{round(precision,5)}\t{round(recall,5)}\t{num_testing_reads}\n')

    accuracy =  round(float(correct_predictions)/total_num_reads, 5)
    f.write(f'{accuracy}')
    f.close()

def ROCcurve(args, probs, labels, label, counter, rank, decision_thresholds):
    print(f'{rank}\t{label}\t{len(labels)}\t{len(probs)}')
    # compute false positive ratea and true positive rate
    fpr, tpr, thresholds = roc_curve(np.asarray(labels), probs[:,label], pos_label=label)
    # Compute Youden's J statistics for each species: get optimal cut off corresponding to a high TPR and low FPR
    J_stats = tpr - fpr
    jstat_optimal_index = np.argmax(J_stats)
    opt_threshold = thresholds[jstat_optimal_index]
    target_tpr = [i for i in tpr if i <= args.tpr][-1]
    target_index = tpr.tolist().index(target_tpr)
    target_threshold = thresholds[target_index]
    target_fpr = fpr[target_index]
    j = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=j),'tpr' : pd.Series(tpr, index = j), 'J-stats' : pd.Series(J_stats, index = j), 'thresholds' : pd.Series(thresholds, index = j)})
    roc.to_csv(os.path.join(args.input_dir, f'{label}-{rank}-df.csv'))
    decision_thresholds[label] = [str(target_threshold), str(target_fpr), str(target_tpr)]
    # with open(os.path.join(args.input_dir, f'decision_threshold_{label}_{rank}.tsv'), 'w') as f:
    #     f.write(f'{label}\t{dict_labels[str(label)]}\t{opt_threshold}\t{fpr[jstat_optimal_index]}\t{tpr[jstat_optimal_index]}\t{jstat_optimal_index}\t{target_threshold}\t{target_fpr}\t{target_tpr}\t{target_index}\t{counter[label]}\t{len(labels)-counter[label]}\t{len(thresholds)}\n')
    # create roc curve
    # plt.clf()
    # plt.plot(fpr, tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(os.path.join(args.input_dir, f'{label}-{rank}-roc-curve.png'))
