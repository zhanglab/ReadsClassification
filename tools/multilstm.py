from .kmer import parse_args, get_info, kmer_dictionary, multiprocesses

if __name__ == '__main__':
    args = parse_args()
    args.reads = 'paired'
    args.model = 'multilstm'
    fastq_files = get_info(args)
    args.kmers_dict = kmer_dictionary(args.kvalue)
    multiprocesses(fastq_files, args)