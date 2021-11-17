import os
import sys
import glob

def count_reads_train_val(input_path):
    list_files = glob.glob(os.path.join(input_path, f'*-train-val-reads'))
    total_train_num_reads = 0
    total_val_num_reads = 0
    for nr_file in list_files:
        with open(nr_file, 'r') as f:
            content = f.readlines()
            total_train_num_reads += int(content[-2].rstrip().split('\t')[1])
            total_val_num_reads += int(content[-1].rstrip().split('\t')[1])
            print(int(content[-2].rstrip().split('\t')[1]), int(content[-1].rstrip().split('\t')[1]))

    with open(os.path.join(input_path, 'total-num-reads'), 'a') as f:
        f.write(f'train\t{total_train_num_reads}\nval\t{total_val_num_reads}\n')

def count_reads_test(input_path):
    list_files = glob.glob(os.path.join(input_path, f'*-test-reads'))
    total_test_num_reads = 0
    for nr_file in list_files:
        with open(nr_file, 'r') as f:
            content = f.readlines()
            total_val_num_reads += int(content[-1].rstrip().split('\t')[1])

    with open(os.path.join(input_path, 'total-num-reads'), 'a') as f:
        f.write(f'test\t{total_test_num_reads}\n')

def main():
    input_path = sys.argv[1]
    # count_reads_train_val(input_path)
    count_reads_test(input_path)

if __name__ == "__main__":
    main()
