import os
import json
import glob
from random import randint
import itertools
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def get_colors(tax_rank, list_taxa, input_path):
    colors = ['salmon', 'red', 'orange', 'limegreen', 'deepskyblue', 'dodgerblue',
    'slategrey', 'royalblue', 'darkorchid', 'violet', 'magenta', 'navy', 'green', 'gold',
    'chocolate', 'black', 'yellow', 'deeppink', 'blue', 'cyan']
    # select a number of colors equal to the number of taxa
    colors_s = random.choices(colors, k=(len(list_taxa)))
    dict_colors = dict(zip(list_taxa, colors_s))
    with open(os.path.join(input_path, f'{tax_rank}-colors.json'), "w") as f:
        json.dump(dict_colors, f)
    return dict_colors

# def create_barplot_training(train_data, val_data, filename, class_mapping):
#     list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
#     new_train_data = [train_data[i] for i in range(len(list_taxa))]
#     new_val_data = [val_data[i] for i in range(len(list_taxa))]
#     x_pos = np.arange(0, len(list_taxa)*2, 2)
#     width = 0.5
#
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(18, 15))
#     ax.bar(list_taxa, new_train_data, width, align='center', color='lightcoral', label='training')
#     ax.bar(list_taxa, new_val_data, width, align='center', color='dodgerblue', label='validation')
#     ax.legend(loc='upper right', fontsize=15)
#     ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
#     ax.set_ylabel('# reads', fontsize=15)
#     plt.savefig(f'{filename}', bbox_inches='tight')
#
#
# def create_barplot_testing(test_data, filename, class_mapping):
#     list_taxa = [class_mapping[str(i)] for i in range(len(class_mapping))]
#     new_test_data = [test_data[i] if i in test_data else 0 for i in range(len(list_taxa))]
#     x_pos = np.arange(0, len(list_taxa) * 2, 2)
#     width = 0.5
#
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(18, 15))
#     ax.bar(list_taxa, new_test_data, width, align='center', color='lightcoral', label='testing')
#     ax.legend(loc='upper right', fontsize=15)
#     ax.set_xticklabels(list_taxa, rotation=90, fontsize=7)
#     ax.set_ylabel('# reads', fontsize=15)
#     plt.savefig(f'{filename}', bbox_inches='tight')

# def learning_curves(history, filename, mode='std'):
#     # plot learning curves
#     plt.clf()
#     fig = plt.figure(dpi=500)
#     ax1 = fig.add_subplot(111)
#     ax2 = ax1.twinx()
#     # Add some extra space for the second axis at the bottom
#     fig.subplots_adjust(bottom=0.2)
#     fig.set_size_inches(7, 7)
#     # Set range of y axis to 0.0 - max of the four lists
#     if mode == 'std':
#         combined_lists = history.history['loss'] + history.history['val_loss'] + history.history['accuracy'] + \
#                      history.history['val_accuracy']
#     elif mode == 'hp':
#         combined_lists = history.history['loss'] + history.history['accuracy']
#
#     max_range = max(combined_lists)
#     ax2.set_ylim(0, 1)
#     ax1.set_ylim(0, max_range)
#     # Get x-axis values
#     list_num_epochs = list(range(1, len(history.history['loss']) + 1, 1))
#     x_coords = [i for i in range(1, len(history.history['loss']) + 1)]
#     if mode == 'std':
#         ax2.plot(list_num_epochs, history.history['val_accuracy'], color='salmon', linewidth=2.0,
#              label='Average Validation Accuracy')
#         ax1.plot(list_num_epochs, history.history['val_loss'], color='dodgerblue', linewidth=2.0, label='Validation Loss')
#     ax2.plot(list_num_epochs, history.history['accuracy'], color='lightsalmon', linewidth=2.0, label='Average Training Accuracy')
#     ax1.plot(list_num_epochs, history.history['loss'], color='skyblue', linewidth=2.0, label='Training Loss')
#     ax1.set_ylabel('Loss', fontsize=14, labelpad=12)
#     ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
#     ax2.set_ylabel('Average Accuracy', fontsize=14, labelpad=12)
#     ax2.ticklabel_format(useOffset=True, style='sci')
#     ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
#                fancybox=True, shadow=False, ncol=5)
#     ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
#                fancybox=True, shadow=False, ncol=5)
#     # Add vertical lines to represent each epoch
#     for xcoord in x_coords:
#         plt.axvline(x=xcoord, color='gray', linestyle='--')
#     plt.savefig(f'{filename}', bbox_inches='tight')

def ROCcurve(args, class_mapping, rank):
    # get arrays of predicted probabilities and true values
    list_pred_files = sorted(glob.glob(os.path.join(args.results_dir, 'pred-probs-*.npy')))
    list_true_files = sorted(glob.glob(os.path.join(args.results_dir, 'true-probs-*.npy')))

    # true_arr = np.asarray(trueclasses)
    # pred_arr = np.asarray(predictions)

    fpr = {}
    tpr = {}
    # roc_auc = {}
    thresholds = {}

    for i in range(len(list_pred_files)):
        pred_arr = np.load(list_pred_files[i])
        true_arr = np.load(list_true_files[i])
        print(pred_arr)
        print(true_arr)

        print(len(jaccard_score(true_arr, pred_arr)))

        for j in range(len(class_mapping)):
            fpr[j], tpr[j], thresholds[j] = roc_curve(true_arr[:, j], pred_arr[:, j])
        print(j, list_pred_files[j], fpr[0], tpr[0], thresholds[0])
            # roc_auc[j] = auc(fpr[j], tpr[j])

        J_stats = [None] * len(class_mapping)
        opt_thresholds = [None] * len(class_mapping)
        f = open(os.path.join(args.output_dir, f'decision_thresholds_{rank}'), 'w')
        # Compute Youden's J statistics for each taxon
        for j in range(len(class_mapping)):
            J_stats[i] = tpr[j] - fpr[j]
            jstat_max_index = np.argmax(J_stats[j])
            opt_thresholds[j] = thresholds[j][jstat_max_index]
            jstat_decision_threshold = round(J_stats[j][jstat_max_index], 2)
            f.write(f'{i}\t{class_mapping[str(j)]}\t{jstat_decision_threshold}\n')
        else:
            f.write(f'{i}\t{class_mapping[str(j)]}\t0.5\n')
        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(true_arr.ravel(), pred_arr.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # # Aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_mapping))]))
        # # Interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for j in range(len(class_mapping)):
        #     mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])
        #
        # # Finally average it and compute AUC
        # mean_tpr /= len(class_mapping)

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.clf()
    # ax = plt.gca()
    # plt.figure()
    # lw = 2
    # plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
    #
    #
    # for i, color in zip(range(len(class_mapping)), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(class_mapping[str(i)], roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # #plt.legend(loc=(0, -.6), prop=dict(size=9))
    # plt.savefig(os.path.join(output_path, f'ROCcurves_{tax_rank}.png'),bbox_inches='tight')
    # figlegend = plt.figure()
    # plt.figlegend(*ax.get_legend_handles_labels())
    # figlegend.savefig(os.path.join(output_path, f'ROCcurves_Legend_{tax_rank}.png'), bbox_inches='tight')
