import os
import glob
from random import randint
from collections import defaultdict, Counter
import itertools
import json
import pandas as pd
import numpy as np
from cami2_new import create_heatmap, norm_confusion_matrix
import multiprocessing as mp
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_json_dict
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

def create_confusion_matrices(args, mapping_dict):
    args.confusion_matrices = {}
    args.confusion_matrices['species'] = np.zeros((len(mapping_dict['species']), len(mapping_dict['species'])))
    args.confusion_matrices['genus'] = np.zeros((len(mapping_dict['genus']), len(mapping_dict['genus'])))
    args.confusion_matrices['family'] = np.zeros((len(mapping_dict['family']), len(mapping_dict['family'])))
    args.confusion_matrices['order'] = np.zeros((len(mapping_dict['order']), len(mapping_dict['order'])))
    args.confusion_matrices['class'] = np.zeros((len(mapping_dict['class']), len(mapping_dict['class'])))
    args.confusion_matrices['phylum'] = np.zeros((len(mapping_dict['phylum']), len(mapping_dict['phylum'])))


def get_ncbi_predicted_label(args, rank, predicted_taxon):
    predicted_labels = []
    for k, v in args.ncbi_labels_mapping_dict[rank].items():
        for i in range(len(v)):
            if v[i] == predicted_taxon:
                predicted_labels.append(k)
    return predicted_labels

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

def load_decision_thds(args):
    args.decision_thresholds_dict = {}
    args.decision_thresholds_dict['species'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'species-decision-thresholds.json'))
    args.decision_thresholds_dict['genus'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'genus-decision-thresholds.json'))
    args.decision_thresholds_dict['family'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'family-decision-thresholds.json'))
    args.decision_thresholds_dict['order'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'order-decision-thresholds.json'))
    args.decision_thresholds_dict['class'] = load_json_dict(args, os.path.join(args.thresholds_dir, 'class-decision-thresholds.json'))

def combine_probs(probs, new_probs, label, sp_labels, r_labels):
    # create list with all labels at species level
    sp_indices = [int(sp_labels[j]) for j in range(len(sp_labels)) if r_labels[j] == label]
    r_label_new_probs = np.asarray([j[sp_indices].sum() for j in probs])
    # insert new combined probs into new array
    new_probs[:,label] = r_label_new_probs

    return new_probs

def get_decision_thds(args, rank, probs, labels):
    # load species mapping labels dictionary
    labels_mapping_dict = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_labels.json'))
    if rank != 'species':
        # load dictionary mapping species to given rank labels
        rank_species_mapping = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_species_labels.json'))
        # convert labes at species level to its corresponding label at given rank
        labels = [rank_species_mapping[str(i)] for i in labels]
        sp_labels = list(rank_species_mapping.keys())
        r_labels = [int(rank_species_mapping[i]) for i in sp_labels]
        unique_r_labels = list(set(r_labels))
        print(f'# unique labels: {len(unique_r_labels)}')
        new_probs = np.zeros((len(probs), len(unique_r_labels)))
        pool = mp.pool.ThreadPool(args.NUM_CPUS)
        results = pool.starmap(combine_probs, zip(itertools.repeat(probs, len(unique_r_labels)), itertools.repeat(new_probs, len(unique_r_labels)),  unique_r_labels, itertools.repeat(sp_labels, len(unique_r_labels)), itertools.repeat(r_labels, len(unique_r_labels))))
        pool.close()
        pool.join()
        probs = new_probs

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
    inv_sp_labels_mapping_dict = {v: k for k, v in args.labels_mapping_dict['species'].items()}
    pred_sp_labels = [inv_sp_labels_mapping_dict[i] for i in pred_taxa]
    sample = tsv_file.split("/")[-1].split("-")[0].split("_")[-1]
    summary_dict[sample] = {}
    summary_dict[sample]['total'] = len(pred_taxa)
    outfile = os.path.join(args.input_dir, tsv_file.split("/")[-1].split('.')[0])
    print(f'{sample}\t{len(pred_taxa)}\t{len(probs)}\t{len(pred_sp_labels)}')
    for r in args.ranks:
        inv_labels_mapping_dict = {v: k for k, v in args.labels_mapping_dict[r].items()}
        if r != 'species':
            pred_labels = [args.rank_species_mapping[r][str(i)] for i in pred_sp_labels]
        else:
            pred_labels = pred_sp_labels
        print(pred_labels[0])
        rel_abundance = defaultdict(int)
        for i in range(len(pred_labels)):
            if str(pred_labels[i]) in args.decision_thresholds_dict[r]:
                if probs[i] >= float(args.decision_thresholds_dict[r][str(pred_labels[i])][0]):
                    rel_abundance[pred_labels[i]] += 1
            else:
                if probs[i] >= 0.5:
                    rel_abundance[pred_labels[i]] += 1

        with open(f'{outfile}-{r}-rel-abundance', 'w') as f:
            for k, v in rel_abundance.items():
                f.write(f'{args.labels_mapping_dict[r][str(k)]}\t{v}\t{len(pred_labels)}\t{round(v/len(pred_labels),5)}\n')
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

def get_metrics(args, true_taxa, predicted_taxa, confusion_matrices, rank):
    """ precision = True Positives / (True Positives + False Positives) """
    """ recall = True Positives / (True Positives + False Negatives) """
    """ accuracy = number of correct predictions / (number of reads in testing set) """
    """ f1-score = 2 * (precision * recall) / (precision + recall)  """

    labels_mapping_dict = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_labels.json'))

    if rank != 'species':
        # load dictionary mapping species labels to other ranks labels
        rank_species_mapping = load_json_dict(args, os.path.join(args.rank_mapping_dir, f'{rank}_species_labels.json'))
        # get vectors of predicted and true labels at given rank
        predicted_taxa = [rank_species_mapping[str(i)] for i in predicted_taxa]
        true_taxa = [rank_species_mapping[str(i)] for i in true_taxa]

    cm = get_cm(true_taxa, predicted_taxa, labels_mapping_dict, rank)
    create_heatmap(cm, os.path.join(args.input_dir, f'{rank}-cm-raw.png'), 'raw', rank)
    confusion_matrices[rank] = cm
    print(f'{rank}\t{cm.sum()}')
    cm_norm = norm_confusion_matrix(cm)
    create_heatmap(cm_norm, os.path.join(args.input_dir, f'{rank}-cm-norm.png'), 'norm', rank)

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
    decision_thresholds[label] = [str(opt_threshold), str(fpr[jstat_optimal_index]), str(tpr[jstat_optimal_index]),  str(target_threshold), str(target_fpr), str(target_tpr)]
    # with open(os.path.join(args.input_dir, f'decision_threshold_{label}_{rank}.tsv'), 'w') as f:
    #     f.write(f'{label}\t{dict_labels[str(label)]}\t{opt_threshold}\t{fpr[jstat_optimal_index]}\t{tpr[jstat_optimal_index]}\t{jstat_optimal_index}\t{target_threshold}\t{target_fpr}\t{target_tpr}\t{target_index}\t{counter[label]}\t{len(labels)-counter[label]}\t{len(thresholds)}\n')
    # create roc curve
    # plt.clf()
    # plt.plot(fpr, tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig(os.path.join(args.input_dir, f'{label}-{rank}-roc-curve.png'))
