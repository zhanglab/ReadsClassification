import os
import itertools
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy, epochs):
    plt.clf()
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(bottom=0.2)
    fig.set_size_inches(7, 7)
    # Set range of y axis to 0.0 - max of the four lists
    combined_lists = train_epoch_loss + test_epoch_loss
    max_range = max(combined_lists)
    ax2.set_ylim(0, 1)
    ax1.set_ylim(0, max_range)
    # Get x-axis values
    list_num_epochs = list(range(1, epochs + 1, 1))
    x_coords = [i for i in range(1, epochs + 1)]

    ax2.plot(list_num_epochs, test_epoch_accuracy, color='black', linewidth=2.0, label='Average Validation Accuracy')
    ax2.plot(list_num_epochs, train_epoch_accuracy, color='green', linewidth=2.0, label='Average Training Accuracy')
    ax1.plot(list_num_epochs, test_epoch_loss, color='red', linewidth=2.0, label='Validation Loss')
    ax1.plot(list_num_epochs, train_epoch_loss, color='blue', linewidth=2.0, label='Training Loss')
    ax1.set_ylabel('Loss', fontsize=14, labelpad=12)
    ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
    ax2.set_ylabel('Average Accuracy', fontsize=14, labelpad=12)
    ax2.ticklabel_format(useOffset=True, style='sci')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               fancybox=True, shadow=False, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               fancybox=True, shadow=False, ncol=5)
    # Add vertical lines to represent each epoch
    for xcoord in x_coords:
        plt.axvline(x=xcoord, color='gray', linestyle='--')
    plt.savefig(os.path.join(self.hparams.output, 'LearningCurves-{0}-model.png'.format(self.hparams.model)),
        bbox_inches='tight')

def ROCcurve(self):
    true_arr = np.asarray(self.trueclasses)
    pred_arr = np.asarray(self.predictions)
    fpr = {}
    tpr = {}
    roc_auc = {}
    thresholds = {}
    for i in range(len(self.hparams.class_mapping)):
        fpr[i], tpr[i], thresholds[i] = roc_curve(true_arr[:, i], pred_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    J_stats = [None] * len(self.hparams.class_mapping)
    opt_thresholds = [None] * len(self.hparams.class_mapping)
    # Compute Youden's J statistics for each species
    for i in range(len(self.hparams.class_mapping)):
        J_stats[i] = tpr[i] - fpr[i]
        opt_thresholds[i] = thresholds[i][np.argmax(J_stats[i])]
        print('Optimum threshold for class {0}: {1}'.format(self.hparams.class_mapping[i], opt_thresholds[i]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_arr.ravel(), pred_arr.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.hparams.class_mapping))]))
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(self.hparams.class_mapping)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(self.hparams.class_mapping)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.clf()
    plt.figure()
    # ax = plt.gca()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)

    colors = []
    for i in range(len(self.hparams.class_mapping)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    for i, color in zip(range(len(self.hparams.class_mapping)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(self.hparams.class_mapping[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=(0, -.6), prop=dict(size=9))
    plt.savefig(os.path.join(self.hparams.output, 'ROCcurves.png'),bbox_inches='tight')

def RecallPrecisionCurves(self):
    dict_metrics = metrics.classification_report(self.true_classes, self.predicted_classes,
                                                 zero_division=0, output_dict=True)
    print(dict_metrics)
    true_arr = np.asarray(self.trueclasses)
    pred_arr = np.asarray(self.predictions)
    precision = {}
    recall = {}
    average_precision = {}

    for i in range(len(self.hparams.class_mapping)):
        precision[i], recall[i], _ = precision_recall_curve(true_arr[:, i], pred_arr[:, i])
        average_precision[i] = average_precision_score(true_arr[:, i], pred_arr[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(true_arr.ravel(), pred_arr.ravel())
    average_precision["micro"] = average_precision_score(true_arr, pred_arr, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    # Plot the micro-averaged Precision-Recall curve
    plt.clf()
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.savefig(os.path.join(self.hparams.output, 'micro-averaged-PRC-all.png'), bbox_inches='tight')

    # Plot Precision-Recall curve for each class and iso-f1 curves
    plt.clf()
    colors = []
    for i in range(len(self.hparams.class_mapping)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    plt.figure()
    f_scores = np.linspace(0.2, 0.8, num=4)

    lines = []
    labels = []
#        for f_score in f_scores:
#            x = np.linspace(0.01, 1)
#            y = f_score * x / (2 * x - f_score)
#            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

#        lines.append(l)

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

    for i, color in zip(range(len(self.hparams.class_mapping)), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for species {0} (area = {1:0.2f})'.format(self.hparams.class_mapping[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labels, loc=(0, -.7), prop=dict(size=9))
    plt.savefig(os.path.join(self.hparams.output, 'RecallPrecisionCurves.png'), bbox_inches='tight')
