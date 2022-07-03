import glob
import sys
import os

def get_reads(fq_files):
    fw_reads = {}
    rv_reads = {}
    for fq in fq_files:
        with open(fq, 'r') as f:
            content = f.readlines()
            reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
            for r in reads:
                if r.split('\n')[0][-1] == '1':
                    fw_reads[r.split('\n')[0]] = r
                elif r.split('\n')[0][-1] == '2':
                    rv_reads[r.split('\n')[0]] = r
    print(f'# fw reads: {len(fw_reads)}\t# rev reads: {len(rv_reads)}')
    return fw_reads, rv_reads


if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    list_fq_files = sorted(glob.glob(os.path.join(input_dir, 'updated-testing-set-*-reads.fq')))
    fw_reads, rv_reads = get_reads(list_fq_files)

    reads_id =[r[:-2] for r in list(fw_reads.keys())]
    print(f'# reads id: {len(reads_id)}')
    num_reads_per_set = 500000
    total_num_reads = 0
    for i in range(0, len(reads_id), num_reads_per_set):
        subset_fw_reads = [fw_reads[r] for r in reads_id[i:i+num_reads_per_set]]
        subset_rv_reads = [rv_reads[r] for r in reads_id[i:i+num_reads_per_set]]
        print(f'subset: {i}\t{len(subset_fw_reads)}\t{len(subset_rv_reads)}')
        total_num_reads += len(subset_fw_reads) + len(subset_rv_reads)
        with open(os.path.join(output_dir, f'fw-updated-testing-set-{i}-reads.fq'), 'w') as out_f:
            out_f.write(''.join(subset_fw_reads))
        with open(os.path.join(output_dir, f'rv-updated-testing-set-{i}-reads.fq'), 'w') as out_f:
            out_f.write(''.join(subset_rv_reads))
    print(f'total # reads: {total_num_reads}')
