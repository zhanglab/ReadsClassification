import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd

def LearningCurvesPlot(self, train_epoch_loss, train_epoch_accuracy, test_epoch_loss, test_epoch_accuracy):
    plt.clf()
    fig = plt.figure(dpi=500)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(bottom=0.2)
    fig.set_size_inches(7, 7)
    list_num_epochs = list(range(1, self.hparams.epochs + 1, 1))

    # x_coords = [i*self.strategy.num_replicas_in_sync*num_train_batches for i in range(1,self.epochs+1)]
    x_coords = [i for i in range(1, self.hparams.epochs + 1)]

    ax2.plot(list_num_epochs, test_epoch_accuracy, color='black', linewidth=2.0, label='Average Validation Accuracy')
    ax2.plot(list_num_epochs, train_epoch_accuracy, color='green', linewidth=2.0, label='Average Training Accuracy')
    ax1.plot(list_num_epochs, test_epoch_loss, color='red', linewidth=2.0, label='Validation Loss')
    ax1.plot(list_num_epochs, train_epoch_loss, color='blue', linewidth=2.0, label='Training Loss')
    ax1.set_ylabel('Loss (%)', fontsize=14, labelpad=12)
    ax1.set_xlabel('Number of epochs', fontsize=14, labelpad=12)
    ax2.set_ylabel('Average Accuracy (%)', fontsize=14, labelpad=12)
    ax2.ticklabel_format(useOffset=True, style='sci')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               fancybox=True, shadow=False, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
               fancybox=True, shadow=False, ncol=5)
    # Add vertical lines to represent each epoch
    for xcoord in x_coords:
        plt.axvline(x=xcoord, color='gray', linestyle='--')
    plt.savefig(self.hparams.output + 'LearningCurves-{0}-model.png'.format(self.hparams.model),
        bbox_inches='tight')

def GetMetrics(cm, rank, listtaxa):
    print('Rank: {}'.format(rank))
    template = ('Taxon: {}, Recall: {}, Precision: {}')
    tp = float()
    fp = float()
    fn = float()
    tn = float()
    recall = float()
    precision = float()

    # Get the FP, TP, FN and TN for each class
    for j in range(cm.shape[0]):
        true_pos = cm.iloc[j, j]
        tp += cm.iloc[j, j]
        false_pos = cm.iloc[:, j].sum() - true_pos
        fp += false_pos
        false_neg = cm.iloc[j, :].sum() - true_pos
        fn += false_neg
        true_neg = cm.values.sum() - false_pos - false_neg - true_pos
        tn += true_neg
        recall += true_pos / (false_neg + true_pos)
        precision += true_pos / (true_pos + false_pos)
        print(template.format(listtaxa[j], round(float(true_pos) / (true_pos + false_neg), 3),
                              round(float(true_pos) / (true_pos + false_pos), 3)))

    recall_micro = tp / (tp + fn)
    precision_micro = tp / (tp + fp)
    recall_macro = recall / cm.shape[0]
    precision_macro = precision / cm.shape[0]
    print('Test Accuracy entire test set {0} level: {1}'.format(rank, round(float(tp + tn) / (tp + tn + fn + fp), 3)))
    print('Macro precision {0} level: {1}'.format(rank, precision_macro))
    print('Micro precision {0} level: {1}'.format(rank, precision_micro))
    print('Macro recall {0} level: {1}'.format(rank, recall_macro))
    print('Micro recall {0} level: {1}'.format(rank, recall_micro))

def GetFilteredResults(self):
    listspecies = list(self.hparams.class_mapping.values())
    # listgenus = ['Polaribacter', 'Reinekea', 'Colwellia', 'Paraglaciecola', 'Octadecabacter', 'Sulfitobacter', 'Halocynthiibacter', 'Pseudoalteromonas', 'Vibrio']
    listgenus = ['Polaribacter', 'Reinekea', 'Colwellia']
    filteredpred = list()
    filteredtrue = list()
    cm = pd.DataFrame(0, columns=listspecies, index=listspecies)
    cm_filtered = pd.DataFrame(0, columns=listspecies, index=listspecies)
    cm_genus = pd.DataFrame(0, columns=listgenus, index=listgenus)

    for i in range(len(self.true_classes)):
        true_class = self.true_classes[i]
        pred_class = self.predicted_classes[i]
        # Get species name
        predspecies = self.hparams.class_mapping[pred_class]
        truespecies = self.hparams.class_mapping[true_class]
        # Get genus name
        predgenus = self.dictgenus[predspecies]
        truegenus = self.dictgenus[truespecies]
        # Get probability
        prob = self.probabilities[i]
        # add entry into confusion matrix if probability above threshold
        if prob > self.probability_threshold:
            filteredpred.append(pred_class)
            filteredtrue.append(true_class)
            cm_filtered.iloc[cm_filtered.index.get_loc(truespecies), cm_filtered.columns.get_loc(predspecies)] += 1
            cm_genus.iloc[cm_genus.index.get_loc(truegenus), cm_genus.columns.get_loc(predgenus)] += 1
        else:
            self.unclassified += 1
        cm.iloc[cm.index.get_loc(truespecies), cm.columns.get_loc(predspecies)] += 1
    print('Test Accuracy entire test set species level - filtered predictions: %.2f%%' % (
                100 * np.sum(filteredpred == filteredtrue) / len(filteredpred)))
    print('Number of Unclassified reads: {}'.format(self.unclassified))
    print('Total number of reads tested 1: {}'.format(len(self.true_classes)))
    print('Total number of reads tested 2: {}'.format(len(self.predicted_classes)))
    print('Total number of reads tested 3: {}'.format(len(self.probabilities)))
    print('Results from unfiltered predictions at the species level:')
    GetMetrics(cm, 'Species', listspecies)
    print('Results from filtered predictions at the species level:')
    GetMetrics(cm_filtered, 'Species', listspecies)
    print('Results from filtered predictions at the genus level:')
    GetMetrics(cm_genus, 'Genus', listgenus)