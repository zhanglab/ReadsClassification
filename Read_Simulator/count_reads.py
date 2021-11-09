import os
import sys
import glob

def count_reads(input_path, type_reads):
    list_files = glob.glob(os.path.join(input_path, f'*-{type_reads}-num-reads'))
    total_num_reads = 0

    for nr_file in list_files:
        with open(nr_file, 'r') as f:
            content = f.readlines()
            total_num_reads += int(content[0].rstrip().split('\t')[1])

    with open(os.path.join(input_path, 'total-num-reads'), 'a') as f:
        f.write(f'{type_reads}\t{total_num_reads}\n')

def main():
    input_path = sys.argv[1]
    count_reads(input_path, 'train')
    count_reads(input_path, 'val')
    count_reads(input_path, 'test')


if __name__ == "__main__":
    main()
