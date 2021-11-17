import os

def count_GC(seq):
    num_GC = 0
    for nt in seq:
        if nt == 'G' or nt == 'C':
            num_GC += 1
    return num_GC

def get_GC_content(fasta_file):
    GC_count = 0
    genome_size = 0
    with open(fastq_file, 'r') as f:
        for line in f:
            if line[0] != '>':
                GC_count += count_GC(line.rstrip())
                genome_size += len(line.rstrip())
    return GC_count, genome_size

def get_read_num(file):
    dict_reads = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip()
            dict_reads[line.split('\t')[0][:-3]] = line.split('\t')[1:]
    return dict_reads

def main():
    file_w_genomes = sys.argv[1]
    file_w_read_count = sys.argv[2]
    outfilename = '-'.join(outfile.split('-')[:-1])
    outfile = open(f'{outfilename}-info', 'w')
    # get number of reads in training, validation and testing sets
    dict_num_reads = get_read_num(file_w_read_count)
    with open(file_w_genomes, 'r') as infile:
        for line in infile:
            genome = line.rstrip().split('\t')[0]
            GC_count, genome_size = get_GC_content(line.rstrip().split('\t')[1]))
            outfile.write(f'{line.rstrip()}\t{GC_count}\t{genome_size}\t{dict_num_reads[genome][0]}\t{dict_num_reads[genome][1]}\t{dict_num_reads[genome][2]}\n')




if __name__ == "__main__":
    main()
