import argparse
import os
import sys
import pandas as pd
import glob
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from dl_toda_tax_utils import load_mapping_dict, get_rank_taxa

def load_conf_scores(filename):
    c_scores = {}
    with open(f'{filename[:-10]}.tsv', 'r') as in_f:
        for line in in_f:
            read_id = line.rstrip().split('\t')[0]
            read_id = read_id[2:] if read_id[1] == '@' else read_id[1:]
            c_scores[read_id] = float(line.rstrip().split('\t')[3])

    return c_scores

def load_cnvd_data(filename):
    ground_truth = {}
    predictions = {}
    with open(filename, 'r') as in_f:
        content = in_f.readlines()
        for i in range(len(content)):
            read_id = content[i].rstrip().split('\t')[0]
            read_id = read_id[1:] if read_id[0] == '@' else read_id
            ground_truth[read_id] = content[i].rstrip().split('\t')[2]
            predictions[read_id] = content[i].rstrip().split('\t')[1]

    return ground_truth, predictions

def get_confusion_matrix(args, ground_truth, predictions, confidence_scores, r_name, r_index):
    ground_truth_taxa = set([v.split(';')[r_index] for v in ground_truth.values()])
    predictions_taxa = set([v.split(';')[r_index] for v in predictions.values()])
    predictions_taxa.add('unclassified')
    print(f'{r_name}\t# ground truth taxa: {len(ground_truth_taxa)}\t# predicted taxa: {len(predictions_taxa)}')
    # create output file to store results for confidence scores analysis
    out_f = open(f'{args.input[:-5]}-cutoff-{args.cutoff}-{r_name}-confidence-scores.tsv', 'w')
    # create empty confusion matrix
    cm = pd.DataFrame(columns = ground_truth_taxa, index = predictions_taxa)
    # fill out table with zeros
    for c in ground_truth_taxa:
        cm[c] = 0
    # fill out confusion matrix with number of reads
    total_n_reads = 0
    for read_id, read_tax in predictions.items():
        total_n_reads += 1
        # update read id if it is a flipped read
        if read_id[-1] == 'f':
            read_id = read_id[:-2]
        true_taxon = ground_truth[read_id].split(';')[r_index]
        if confidence_scores[read_id] > args.cutoff:
            tool_taxon = read_tax.split(';')[r_index]
        else:
            tool_taxon = 'unclassified'
        cm.loc[tool_taxon, true_taxon] += 1
        category_1 = 'correct' if tool_taxon == true_taxon else 'incorrect'
        category_2 = 'classified' if tool_taxon != 'unclassified' else 'unclassified'
        out_f.write(f'{true_taxon}\t{category_1}\t{category_2}\t{confidence_scores[read_id]}\n')

    print(f'{r_name}\t{cm.to_numpy().sum()}\t{len(ground_truth)}\t{len(predictions)}\t{total_n_reads}')

    return cm

def get_metrics(args, cm, r_name, output_file):
    taxa_in_dl_toda = set(list(args.labels_mapping_dict[r_name].values()))
    print(f'Taxa in DL-TODA: {r_name}\t{len(taxa_in_dl_toda)}')
    with open(f'{output_file}-{r_name}-metrics.tsv', 'w') as out_f:
        out_f.write('true taxon\tpredicted taxon\t#reads\tprecision\trecall\tF1\tTP\tFP\tFN\n')
        ground_truth = list(cm.columns)
        predicted_taxa = list(cm.index)
        print(f'Taxa tested: {r_name}\t{len(ground_truth)}\t{len(predicted_taxa)}')
        correct_predictions = 0
        classified_reads = 0
        unclassified_reads = 0
        problematic_reads = 0
        total_num_reads = 0
        for true_taxon in ground_truth:
            num_reads = sum([cm.loc[i, true_taxon] for i in predicted_taxa])
            if true_taxon != 'unclassified' and true_taxon != 'na':
                if true_taxon in taxa_in_dl_toda:
                    if true_taxon in predicted_taxa:
                        predicted_taxon = true_taxon
                        classified_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa if i not in ('unclassified', 'na')])
                        other_true_taxa = [i for i in ground_truth if i != true_taxon]
                        true_positives = cm.loc[predicted_taxon, true_taxon]
                        correct_predictions += true_positives
                        false_positives = sum([cm.loc[predicted_taxon, i] for i in other_true_taxa])
                        false_negatives = sum([cm.loc[i, true_taxon] for i in predicted_taxa if i != predicted_taxon])
                        precision = float(true_positives)/(true_positives+false_positives) if true_positives+false_positives > 0 else 0
                        recall = float(true_positives)/(true_positives+false_negatives) if true_positives+false_negatives > 0 else 0
                        f1_score = float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0
                        out_f.write(f'{true_taxon}\t{predicted_taxon}\t{num_reads}\t{precision}\t{recall}\t{f1_score}\t{true_positives}\t{false_positives}\t{false_negatives}\n')
                    else:
                        if args.zeros:
                            print(f'{true_taxon} has a precision/recall/F1 scores equal to 0')
                            out_f.write(f'{true_taxon}\tna\t{num_reads}\t0\t0\t0\t0\t0\t{num_reads}\n')
                        unclassified_reads += num_reads
                else:
                    print(f'{true_taxon} with {num_reads} reads is not in {args.tool} model')
                    problematic_reads += num_reads
            else:
                print(f'ground truth unknown: {true_taxon}\t{num_reads}')
                problematic_reads += num_reads

            total_num_reads += num_reads

        if 'unclassified' in predicted_taxa:
            unclassified_reads += sum([cm.loc['unclassified', i] for i in ground_truth])
        if 'na' in predicted_taxa:
            unclassified_reads += sum([cm.loc['na', i] for i in ground_truth])

        out_f.write(f'{correct_predictions}\t{cm.to_numpy().sum()}\t{classified_reads}\t{problematic_reads}\t{unclassified_reads}\t {problematic_reads+unclassified_reads+classified_reads}\t{total_num_reads}\n')

        accuracy_whole = round(correct_predictions/cm.to_numpy().sum(), 5) if cm.to_numpy().sum() > 0 else 0
        accuracy_classified = round(correct_predictions/classified_reads, 5) if classified_reads > 0 else 0
        accuracy_w_misclassified = round(correct_predictions/(classified_reads+unclassified_reads), 5) if (classified_reads+unclassified_reads) > 0 else 0
        out_f.write(f'Accuracy - whole dataset: {accuracy_whole}\n')
        out_f.write(f'Accuracy - classified reads only: {accuracy_classified}\n')
        out_f.write(f'Accuracy - classified and unclassified reads: {accuracy_w_misclassified}')


def merge_cm(args, all_cm, rank):
    if args.tool == 'dl-toda':
        excel_files = sorted(glob.glob(os.path.join(args.input_dir, f'*{args.tax_db}-cutoff-{args.cutoff}-confusion-matrix.xlsx')))
    else:
        excel_files = sorted(glob.glob(os.path.join(args.input_dir, f'*{args.tax_db}-confusion-matrix.xlsx')))
    print(rank, len(excel_files))
    df_list = []
    columns = []
    rows = []
    files = []
    for x in excel_files:
        df = pd.read_excel(x, index_col=0, sheet_name=None)
        if rank in df.keys():
            df_list.append(df[rank])
            columns += df[rank].columns.tolist()
            rows += df[rank].index.tolist()
            files.append(x)

    if len(df_list) != 0:
        predicted_taxa = set(rows)
        true_taxa = list(set(columns)).sort()
        # create combined table
        cm = pd.DataFrame(columns = true_taxa, index = predicted_taxa)
        for c in cm:
            cm[c] = 0
        for i in range(len(df_list)):
            if args.tool == "centrifuge" and files[i].split('-')[3] == 'paired':
                cm = cm.add(df_list[i]*2, fill_value=0)
            else:
                cm = cm.add(df_list[i], fill_value=0)
        cm = cm.fillna(0)
        print(cm)
        print(f'Sum of rank_table: {cm.to_numpy().sum()}')
        all_cm[rank] = cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='file with predicted and ground truth taxonomy')
    parser.add_argument('--tool', type=str, help='taxonomic classification tool', choices=['kraken', 'dl-toda', 'centrifuge'])
    parser.add_argument('--dataset', type=str, help='dataset ground truth', choices=['cami', 'testing'])
    parser.add_argument('--cutoff', type=float, help='decision thershold abover which reads are classified', default=0.0)
    parser.add_argument('--merge', help='summarized results from all samples combined', action='store_true')
    parser.add_argument('--metrics', help='get metrics from confusion matrix', action='store_true')
    parser.add_argument('--confusion_matrix', help='create confusion matrix', action='store_true')
    parser.add_argument('--zeros', help='add ground truth taxa with a null precision, recall and F1 metrics', action='store_true')
    parser.add_argument('--input_dir', type=str, help='path to input directory containing excel files to combine', default=os.getcwd())
    parser.add_argument('--dl_toda_tax', help='path to directory containing json directories with info on taxa present in dl-toda')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--to_ncbi', action='store_true', help='whether to analyze results with ncbi taxonomy', default=False)
    parser.add_argument('--compare', action='store_true', help='compare results files obtained with --metrics', default=False)
    parser.add_argument('--rank', type=str, help='taxonomic rank', choices=['species', 'genus', 'family', 'order', 'class', 'phylum'], required=('--compare' in sys.argv))
    parser.add_argument('--dl_toda_metrics', type=str, help='path to result file obtained from running --metrics', required=('--compare' in sys.argv))
    parser.add_argument('--tool_metrics', type=str, help='path to result file obtained from running --metrics with another tool than DL-TODA', required=('--compare' in sys.argv))
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())

    args = parser.parse_args()

    args.ranks = {'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'class': 5, 'phylum': 6}

    if args.dl_toda_tax:
        if args.to_ncbi:
            dl_toda_taxonomy = {}
            with open(os.path.join(args.dl_toda_tax, 'dl_toda_ncbi_taxonomy.tsv'), 'r') as in_f:
                content = in_f.readlines()
                for i in range(len(content)):
                    line = content[i].rstrip().split('\t')
                    dl_toda_taxonomy[line[0]] = line[1]
                get_rank_taxa(args, dl_toda_taxonomy)
            print(f'dl-toda ncbi taxonomy: {len(args.labels_mapping_dict["species"])}\t{len(args.labels_mapping_dict["genus"])}\t{len(args.labels_mapping_dict["family"])}\t{len(args.labels_mapping_dict["order"])}\t{len(args.labels_mapping_dict["class"])}\t{len(args.labels_mapping_dict["phylum"])}')
        else:
            load_mapping_dict(args, args.dl_toda_tax)

    if args.merge:
        with mp.Manager() as manager:
            all_cm = manager.dict()
            # Create new processes
            processes = [mp.Process(target=merge_cm, args=(args, all_cm, r)) for r in args.ranks.keys()]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            if args.tool == 'dl-toda':
                out_filename = os.path.join(args.input_dir, f'{args.tool}-cutoff-{args.cutoff}-all-reads-{args.tax_db}.xlsx')
            else:
                out_filename = os.path.join(args.input_dir, f'{args.tool}-all-reads-{args.tax_db}.xlsx')

            with pd.ExcelWriter(out_filename) as writer:
                for r_name, r_cm in all_cm.items():
                    r_cm.to_excel(writer, sheet_name=f'{r_name}')

    elif args.metrics:
        if args.tool == 'dl-toda':
            input_filename = os.path.join(args.input_dir, f'{args.tool}-cutoff-{args.cutoff}-all-reads-{args.tax_db}.xlsx')
        else:
            input_filename = os.path.join(args.input_dir, f'{args.tool}-all-reads-{args.tax_db}.xlsx')
        cm = pd.read_excel(input_filename, index_col=0, sheet_name=None)
        for r in args.ranks:
            if r in cm.keys():
                if args.tool == 'dl-toda':
                    out_filename = os.path.join(args.input_dir, f'{args.tool}-cutoff-{args.cutoff}-{args.tax_db}-w-zeros') if args.zeros else os.path.join(args.input_dir, f'{args.tool}-{args.tax_db}-wo-zeros')
                else:
                    out_filename = os.path.join(args.input_dir, f'{args.tool}-{args.tax_db}-w-zeros') if args.zeros else os.path.join(args.input_dir, f'{args.tool}-{args.tax_db}-wo-zeros')
                get_metrics(args, cm[r], r, out_filename)

    elif args.confusion_matrix:
        ground_truth, predictions = load_cnvd_data(args.input)
        confidence_scores = load_conf_scores(args.input)
        out_filename = f'{args.input[:-5]}-cutoff-{args.cutoff}-confusion-matrix.xlsx'

        all_cm = {}
        for r_name, r_index in args.ranks.items():
            cm = get_confusion_matrix(args, ground_truth, predictions, confidence_scores, r_name, r_index)
            all_cm[r_name] = cm

        with pd.ExcelWriter(out_filename) as writer:
            for r_name, r_cm in all_cm.items():
                r_cm.to_excel(writer, sheet_name=f'{r_name}')

    elif args.compare:
        dl_toda_res = pd.read_csv(args.dl_toda_metrics, sep='\t')
        tool_res = pd.read_csv(args.tool_metrics, sep='\t')
        dl_toda_true_taxa = dl_toda_res['true taxon'][:-4].tolist()

        # check data
        if dl_toda_true_taxa.sort() != tool_res['true taxon'][:-4].tolist().sort():
            sys.exit(f'Not the same taxa between DL-TODA and tool at rank: {args.rank}')
        if len(dl_toda_true_taxa) != len(tool_res['true taxon'][:-4].tolist()):
            sys.exit(f'Not the same number of taxa between DL-TODA and tool at rank: {args.rank}')
        if dl_toda_res['#reads'][:-4].tolist().sort() != tool_res['#reads'][:-4].tolist().sort():
            sys.exit(f'Not the same #reads column between DL-TODA and tool at rank: {args.rank}')

        # compute percentage difference between true positives for DL-TODA and the other tool
        diff = [((dl_toda_res['TP'][:-4][i]-tool_res['TP'][:-4][i])/dl_toda_res['#reads'][:-4][i])*100 for i in range(len(dl_toda_true_taxa))]
        if len(diff) != len(dl_toda_true_taxa):
            sys.exit(f'Issue with computing percentage difference at rank: {args.rank}')

        with open(os.path.join(args.output_dir, f'DL-TODA-vs-Kraken2-{args.dataset}-set-{args.rank}.tsv'), 'w') as out_f:
            for i in range(len(dl_toda_true_taxa)):
                out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["#reads"][:-4][i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\t{diff[i]}\n')

        low_out_f = open(os.path.join(args.output_dir, f'{args.rank}-lower-performance.tsv'), 'w')
        high_out_f = open(os.path.join(args.output_dir, f'{args.rank}-higher-performance.tsv'), 'w')

        for i in range(len(dl_toda_true_taxa)):
            if dl_toda_res['TP'][:-4][i] > tool_res['TP'][:-4][i]:
                high_out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\n')
            else:
                low_out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\n')





if __name__ == "__main__":
    main()
