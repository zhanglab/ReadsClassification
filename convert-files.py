import argparse
import sys
import os
import gzip
import math
from Bio import SeqIO
from collections import defaultdict
import multiprocessing as mp
from ncbi_tax_utils import get_ncbi_taxonomy, parse_nodes_file, parse_names_file
from dl_toda_tax_utils import get_dl_toda_taxonomy, load_mapping_dict, get_rank_taxa



def convert_kraken_output(args, data, process, d_nodes, d_names, results):
    process_results = []
    for line in data:
        line = line.rstrip().split('\t')
        read = line[1]
        if line[0] == 'U':
            process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{"|".join(["na"]*7)}\n')
        else:
            # get ncbi taxid of predicted taxon
            taxid_index = line[2].find('taxid')
            taxid = line[2][taxid_index+6:-1]
            # get ncbi taxonomy
            taxids, taxonomy, _ = get_ncbi_taxonomy(taxid, d_nodes, d_names)
            process_results.append(f'{read}\t{taxonomy}\t{taxids}\n')
    results[process] = process_results

def convert_dl_toda_output(args, data, process, results):
    process_results = []
    for line in data:
        line = line.rstrip().split('\t')
        read = line[0][1:] if line[0][0] =="@" else line[0]
        pred_species = line[2] if args.dataset == 'dl-toda' else line[1]
        true_species = read.split('|')[1]
        # get gtdb or ncbi taxonomy of predicted taxon
        if args.to_ncbi:
            pred_taxonomy = get_dl_toda_taxonomy(args, pred_species)
            true_taxonomy = get_dl_toda_taxonomy(args, true_species)
            process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
        else:
            taxids, taxonomy, _ = get_dl_toda_taxonomy(args, pred_species)
            process_results.append(f'{read}\t{taxonomy}\t{taxids}\n')

    results[process] = process_results


def convert_cami_dataset(args, data, process, d_nodes, d_names, results):
    process_results = []
    for line in data:
        line = line.decode("utf-8").rstrip().split('\t')
        read = line[0]
        # get ncbi taxid of true taxon
        taxid = line[2]
        # get ncbi taxonomy
        taxids, taxonomy, _ = get_ncbi_taxonomy(taxid, d_nodes, d_names)
        process_results.append(f'{read}\t{taxonomy}\t{taxids}\n')
    results[process] = process_results


def convert_centrifuge_output(args, data, process, d_nodes, d_names, results):
    # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
    process_results = []
    reads_seen = set()
    number_unclassified = 0
    for line in data:
        read = line.rstrip().split('\t')[0]
        if read not in reads_seen:
            taxid = line.rstrip().split('\t')[2]
            true_species = line.rstrip().split('\t')[0].split('|')[1] # change where it splits
            true_taxonomy = get_dl_toda_taxonomy(args, true_species)
            if taxid != '0':
                _, pred_taxonomy, _ = get_ncbi_taxonomy(taxid, d_nodes, d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
            else:
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
                number_unclassified += 1
            reads_seen.add(read)
    print(f'{process}\t{number_unclassified}\t{len(process_results)}')
    results[process] = process_results


def load_data(args):
    if args.dataset == 'cami':
        in_f = gzip.open(args.input_file, 'rb')
        content = in_f.readlines()
        content = content[1:]
    else:
        in_f = open(args.input_file, 'r')
        content = in_f.readlines()
        if args.dataset == "centrifuge":
            content = content[4: (len(content) - 2)]
    chunk_size = math.ceil(len(content)/mp.cpu_count())
    data = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    print(f'{chunk_size}\t{len(data)}\t{len(content)}')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to file to convert')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--fastq', action='store_true', help='type of data to parse', default=False)
    parser.add_argument('--dataset', type=str, help='type of dataset to convert', choices=['cami', 'kraken', 'dl-toda', 'centrifuge'])
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy database')
    parser.add_argument('--dl_toda_tax', help='path to directory containing json directories with info on taxa present in dl-toda', required=('dl-toda' in sys.argv))
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--to_ncbi', action='store_true', help='whether to analyze results with ncbi taxonomy', default=False)
    args = parser.parse_args()

    functions = {'kraken': convert_kraken_output, 'dl-toda': convert_dl_toda_output, 'cami': convert_cami_dataset, 'centrifuge': convert_centrifuge_output}

    if args.to_ncbi:
        # get dl_toda ncbi taxonomy
        with open(os.path.join(args.dl_toda_tax, 'dl_toda_ncbi_taxonomy.tsv'), 'r') as in_f:
            content = in_f.readlines()
            args.dl_toda_taxonomy = {}
            for i in range(len(content)):
                line = content[i].rstrip().split('\t')
                args.dl_toda_taxonomy[line[0]] = line[1]
        get_rank_taxa(args, args.dl_toda_taxonomy)

    if args.dataset == 'dl-toda':
        if args.fastq:
            records = list(SeqIO.parse(args.input_file, "fastq"))
            reads = [f'{i.id}\t\t{i.id.split("|")[1]}\n' for i in records]
            chunk_size = math.ceil(len(reads)/mp.cpu_count())
            data = [reads[i:i + chunk_size] for i in range(0, len(reads), chunk_size)]
        else:
            data = load_data(args)

        load_mapping_dict(args, args.dl_toda_tax)

        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=functions[args.dataset], args=(args, data[i], i, results)) for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            print(len(results))

            if args.output_dir is not None:
                out_filename = os.path.join(args.output_dir, f'{args.input_file.split("/")[-1][:-3]}-{args.tax_db}-cnvd') if args.fastq else os.path.join(args.output_dir, f'{args.input_file.split("/")[-1][:-4]}-{args.tax_db}-cnvd')
            else:
                out_filename = f'{args.input_file[:-3]}-{args.tax_db}-cnvd' if args.fastq else f'{args.input_file[:-4]}-{args.tax_db}-cnvd'

            out_f = open(out_filename, 'w')
            for p in results.keys():
                out_f.write(''.join(results[p]))
    else:
        data = load_data(args)
        # get ncbi taxids info
        d_nodes = parse_nodes_file(os.path.join(args.ncbi_db, 'taxonomy', 'nodes.dmp'))
        d_names = parse_names_file(os.path.join(args.ncbi_db, 'taxonomy', 'names.dmp'))

        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=functions[args.dataset], args=(args, data[i], i, d_nodes, d_names, results)) for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            print(len(results))

            if args.output_dir is not None:
                out_filename = os.path.join(args.output_dir, f'{args.input_file.split("/")[-2]}-{args.tax_db}-cnvd') if args.dataset == 'kraken' else os.path.join(args.output_dir, f'{args.input_file.split("/")[-1]}-{args.tax_db}-cnvd')
                out_f = open(out_filename, 'w')
            else:
                out_f = open(f'{args.input_file}-{args.tax_db}-cnvd', 'w')

            for p in results.keys():
                out_f.write(''.join(results[p]))






if __name__ == "__main__":
    main()
