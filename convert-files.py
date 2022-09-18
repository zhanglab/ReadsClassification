import argparse
import sys
import os
import gzip
import glob
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
        if args.dataset == 'meta':
            if line[0] == 'U':
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\n')
            else:
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\n')
        else:
            if args.dataset == 'cami':
                true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
            else:
                true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
            if line[0] == 'U':
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            else:
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')

    results[process] = process_results

def convert_dl_toda_output(args, data, process, d_nodes, d_names, results):
    process_results = []
    reads = [line.rstrip().split('\t')[0][1:] if line.rstrip().split('\t')[0][0] == "@" else line.rstrip().split('\t')[0] for line in data]
    pred_species = [line.rstrip().split('\t')[2] for line in data] if args.dataset == 'meta' else [line.rstrip().split('\t')[2] for line in data]

    if args.dataset == 'meta':
        probs = [float(line.rstrip().split('\t')[3]) for line in data]
        # print(len(data), len(probs), len(reads), len(pred_species))
        for i in range(len(pred_species)):
            if probs[i] > args.cutoff:
                pred_taxonomy = args.dl_toda_taxonomy[pred_species[i]]
                # pred_taxonomy = get_dl_toda_taxonomy(args, pred_species[i])
                process_results.append(f'{reads[i]}\t{pred_taxonomy}\n')
                # out_f.write(f'{reads[i]}\t{pred_taxonomy}\n')
            else:
                process_results.append(f'{reads[i]}\t{";".join(["unclassified"]*7)}\n')
                # out_f.write(f'{reads[i]}\t{";".join(["unclassified"]*7)}\n')
    else:
        if args.dataset == 'cami':
            true_taxonomy = [get_ncbi_taxonomy(args.cami_data[reads[i]], d_nodes, d_names) for i in range(len(reads))]
        else:
            true_taxonomy = [get_dl_toda_taxonomy(args, reads[i].split('|')[1]) for i in range(len(reads))]

        for i in range(len(pred_species)):
            # pred_taxonomy = get_dl_toda_taxonomy(args, pred_species[i])
            pred_taxonomy = args.dl_toda_taxonomy[pred_species[i]]
            process_results.append(f'{reads[i]}\t{pred_taxonomy}\t{true_taxonomy[i]}\n')
            # out_f.write(f'{reads[i]}\t{pred_taxonomy}\t{true_taxonomy[i]}\n')

    results[process] = process_results

def convert_centrifuge_output(args, data, process, d_nodes, d_names, results):
    # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
    process_results = []
    number_unclassified = 0
    for line in data:
        read = line.rstrip().split('\t')[0]
        taxid = line.rstrip().split('\t')[2]
        if args.dataset == 'cami':
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
        else:
            true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
        if taxid != '0':
            pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
            process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
        else:
            process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            number_unclassified += 1
    results[process] = process_results

def convert_diamond_output(args, data, process, d_nodes, d_names, results):
    # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
    process_results = []
    number_unclassified = 0
    for line in data:
        read = line.rstrip().split('\t')[0]
        taxid = line.rstrip().split('\t')[-1]
        if args.dataset == 'cami':
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
        else:
            true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
        if taxid != '0':
            pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
            process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
        else:
            process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            number_unclassified += 1
    results[process] = process_results

# returns a dictionary with gene ID as key and the taxonomy as the value
def parse_metaphlan_database(file_path):
    parse_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] == '>':
                temp = line.split("\t")
                if len(temp) < 2:
                    taxonomy = 'na'
                else:
                    temp[1] = taxonomy = taxonomy.split(';')
                    taxonomy = taxonomy[1][::-1]

                parse_dict[temp[0]] = 'na' + taxonomy
    return parse_dict

def convert_metaphlan_output(args, data, process, db_dict, results):
    process_results = []
    for line in data:
        read = line.rstrip().split('\t')[0]
        gene_id = line.rstrip().split('\t')[1]
        if args.dataset == 'cami':
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], db_dict)
        else:
            true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
        if taxid != '0':
            pred_taxonomy = db_dict[gene_id]
            process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
        else:
            process_results.append(f'{read}\t{";".join(["unclassified"] * 7)}\t{true_taxonomy}\n')
    results[process] = process_results



def load_cami_data(args):
    in_f = gzip.open(os.path.join(args.cami_path, 'reads_mapping.tsv.gz'), 'rb')
    content = in_f.readlines()
    # create dictionary mapping reads id to ncbi taxid
    data = {line.decode('utf8').rstrip().split('\t')[0]: line.decode('utf8').rstrip().split('\t')[2] for line in content[1:]}

    return data


def load_tool_output(args):
    # args is a generic variable allowing for multiple variables to be given to the function
    in_f = open(args.input_file, 'r')  # opens the file so we can read it
    content = in_f.readlines()         # sets an array named content to hold each line of the input file
    if args.tool == "centrifuge":      # checks if args.tool is equal to centrifuge
        # parse output of centrifuge to only take the first hit for each read
        content = content[1:]  # ignores the first line of the input file
        # reads_seen = set()     # creates an empty set named reads seen. Sets can not contain duplicate values
        parsed_content = defaultdict(list)    # creates an empty list named parsed_content
        for line in content:   # for loop that reads each line of the content list
            # first rstrip() gets rid of the trailing characters, and split('\t') creates a list based on the line variable
            # that is split by tabs. Then we choose the first word that is contained in this new array.
            read = line.rstrip().split('\t')[0]
            parsed_content[read].append(line)
            # if read not in reads_seen:   # checks to see if read is in the reads_seen array
                # parsed_content.append(line)  # adds the line variable to the parsed content array
                # reads_seen.add(read)     # adds the read variable to the reads_seen set.
        content = [v[0] if len(v) == 1 else '\t'.join([v[0].rstrip().split('\t')[0], '' ,'0']) for k, v in parsed_content.items()]   # sets the content array to the value of parsed content
    if args.tool == 'diamond':
        content = content[3:]
        reads_seen = set()
        parsed_content = []
        for line in content:
            read = line.rstrip().split('\t')[0]
            if read not in reads_seen:
                parsed_content.append(line)
                reads_seen.add(read)
        content = parsed_content

    # chunk_size is an integer used set the length of the sub-arrays
    # that will be used for multi-processing.
    chunk_size = math.ceil(len(content)/mp.cpu_count())
    # data is a list of lists. The lists in data are the sub arrays of content. This is necessary for multiprocessing.
    data = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    num_reads = [len(i) for i in data]
    print(f'{sum(num_reads)}\t{len(data)}\t{len(content)}\t{chunk_size}\t{len(data[-1])}\t{mp.cpu_count()}')

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='path to file to convert')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--fastq', action='store_true', help='type of data to parse', default=False)
    parser.add_argument('--tool', type=str, help='type of dataset to convert', choices=['kraken', 'dl-toda', 'centrifuge', 'metaphlan', 'diamond'])
    parser.add_argument('--dataset', type=str, help='type of dataset to convert', choices=['dl-toda', 'cami', 'meta'])
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy database')
    parser.add_argument('--cami_path', help='path to cami reads_mapping.tsv.gz file', required=('cami' in sys.argv))
    parser.add_argument('--cutoff', type=float, help='confidence score cutoff above which reads are considered as classified', default=0.0)
    parser.add_argument('--dl_toda_tax', help='path to directory containing json directories with info on taxa present in dl-toda', required=('dl-toda' in sys.argv))
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    # parser.add_argument('--to_ncbi', action='store_true', help='whether to analyze results with ncbi taxonomy', default=False)
    parser.add_argument('--metaphlan_db',
                        help='path to directory containing metaphlan database file called mpa_v30_CHOCOPhlAn_201901.fna',
                        required=('metaphlan' in sys.argv))
    args = parser.parse_args()

    functions = {'kraken': convert_kraken_output, 'dl-toda': convert_dl_toda_output, 'centrifuge': convert_centrifuge_output, 'metaphlan': convert_metaphlan_output, 'diamond': convert_diamond_output}

    # get ncbi taxids info
    d_nodes = parse_nodes_file(os.path.join(args.ncbi_db, 'nodes.dmp'))
    d_names = parse_names_file(os.path.join(args.ncbi_db, 'names.dmp'))


    if args.dataset == 'cami':
        args.cami_data = load_cami_data(args)

    if args.tax_db == 'ncbi':
        index = 2
    elif args.tax_db =='gtdb':
        index = 3

    # get dl_toda taxonomy
    with open(os.path.join(args.dl_toda_tax, 'dl_toda_taxonomy.tsv'), 'r') as in_f:
        content = in_f.readlines()
        args.dl_toda_taxonomy = {}
        for i in range(len(content)):
            line = content[i].rstrip().split('\t')
            args.dl_toda_taxonomy[str(line[1])] = line[index]
    # get_rank_taxa(args, args.dl_toda_taxonomy)

    if args.tool == 'dl-toda':
        if args.fastq:
            records = list(SeqIO.parse(args.input_file, "fastq"))
            reads = [f'{i.id}\t\t{i.id.split("|")[1]}\n' for i in records]
            chunk_size = math.ceil(len(reads)/mp.cpu_count())
            data = [reads[i:i + chunk_size] for i in range(0, len(reads), chunk_size)]
        else:
            data = load_tool_output(args)

        # load_mapping_dict(args, args.dl_toda_tax)

        # with mp.Manager() as manager:
        #     results = manager.dict()
        #     processes = [mp.Process(target=functions[args.tool], args=(args, data[i], i, d_nodes, d_names, results)) for i in range(len(data))]
        #     for p in processes:
        #         p.start()
        #     for p in processes:
        #         p.join()
        #
        #     if args.output_dir is not None:
        #         out_filename = os.path.join(args.output_dir, f'{args.input_file.split("/")[-1][:-3]}-{args.tax_db}-cnvd') if args.fastq else os.path.join(args.output_dir, f'{args.input_file.split("/")[-1][:-4]}-{args.tax_db}-cnvd')
        #     else:
        #         out_filename = f'{args.input_file[:-3]}-{args.tax_db}-cnvd' if args.fastq else f'{args.input_file[:-4]}-{args.tax_db}-cnvd'
        #
        #     out_f = open(out_filename, 'w')
        #     num_reads = 0
        #     print(f'{num_reads}\t{len(results)}')
        #     for p in results.keys():
        #         print(f'{p}\t{len(results[p])}')
        #         num_reads += len(results[p])
        #         out_f.write(''.join(results[p]))
        #     print(f'# reads: {num_reads}')
    else:
        data = load_tool_output(args)

        if args.output_dir is not None:
            out_filename = os.path.join(args.output_dir, f'{args.input_file.split("/")[-2]}-{args.tax_db}-cnvd') if args.tool == 'kraken' else os.path.join(args.output_dir, f'{args.input_file.split("/")[-1]}-{args.tax_db}-cnvd')
            # out_f = open(out_filename, 'w')
        else:
            # out_f = open(f'{args.input_file}-{args.tax_db}-cnvd', 'w')
            out_filename = f'{args.input_file}-{args.tax_db}-cnvd'

        with mp.Manager() as manager:
            results = manager.dict()
            if args.tool == "metaphlan":
                db_dict = parse_metaphlan_database()
                processes = [mp.Process(target=functions[args.tool], args=(args, data[i], i, db_dict, results)) for i in range(len(data))]
            else:
                processes = [mp.Process(target=functions[args.tool], args=(args, data[i], i, d_nodes, d_names, results))
                             for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()


            num_reads = 0
            print(f'{num_reads}\t{len(results)}')
            for p in results.keys():
                print(f'{p}\t{len(results[p])}')
                num_reads += len(results[p])
                out_f.write(''.join(results[p]))
            print(f'# reads: {num_reads}')

if __name__ == "__main__":
    main()
