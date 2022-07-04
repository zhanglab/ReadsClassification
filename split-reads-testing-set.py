import glob
import sys
import os
import multiprocessing as mp

def get_reads(fq_files, fw_reads, rv_reads):
    for fq in fq_files:
        with open(fq, 'r') as f:
            content = f.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            for r in reads:
                if r.split('\n')[0][-1] == '1':
                    fw_reads[r.split('\n')[0][:-2]] = r
                elif r.split('\n')[0][-1] == '2':
                    rv_reads[r.split('\n')[0][:-2]] = r

def write_reads_to_file(reads, fw_reads, rv_reads, type_of_reads):
    num_reads_per_set = 500000
    total_num_reads = 0
    for i in range(0, len(reads), num_reads_per_set):
        subset_fw_reads = [fw_reads[r] for r in reads[i:i+num_reads_per_set]]
        subset_rv_reads = [rv_reads[r] for r in reads[i:i+num_reads_per_set]]
        print(f'subset: {i}\t{len(subset_fw_reads)}\t{len(subset_rv_reads)}')
        total_num_reads += len(subset_fw_reads) + len(subset_rv_reads)
        with open(os.path.join(output_dir, f'fw-updated-testing-set-{i}-reads.fq'), 'w') as out_f:
            out_f.write(''.join(subset_fw_reads))
        with open(os.path.join(output_dir, f'rv-updated-testing-set-{i}-reads.fq'), 'w') as out_f:
            out_f.write(''.join(subset_rv_reads))
    print(f'total # reads: {total_num_reads}')


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    list_fq_files = sorted(glob.glob(os.path.join(input_dir, 'updated-testing-set-*-reads.fq')))
    chunk_size = 20
    set_fq_files = [list_fq_files[i:i+chunk_size] for i in range(0, len(list_fq_files), chunk_size)]
    print(f'{chunk_size}\t{len(set_fq_files)}\t{len(list_fq_files)}')
    with mp.Manager() as manager:
        fw_reads = manager.dict()
        rv_reads = manager.dict()
        processes = [mp.Process(target=get_reads, args=(set_fq_files[i], fw_reads, rv_reads)) for i in range(len(set_fq_files))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print(f'# fw reads: {len(fw_reads)}\t# rev reads: {len(rv_reads)}\t{len(fw_reads)+len(rv_reads)}')

        unpaired_reads = list(set(fw_reads.keys()).difference(set(rv_reads.keys()))) + list(set(rv_reads.keys()).difference(set(fw_reads.keys())))
        paired_reads = list(set(fw_reads.keys()).intersection(set(rv_reads.keys())))
        print(f'# unpaired reads: {len(unpaired_reads)}\t# paired reads: {len(paired_reads)}')

        write_reads_to_file(unpaired_reads, fw_reads, rv_reads, 'unpaired')
        write_reads_to_file(paired_reads, fw_reads, rv_reads, 'unpaired')
