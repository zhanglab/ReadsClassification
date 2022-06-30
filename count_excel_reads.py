import os
import glob
import pandas as pd
import numpy as np


def count_reads_excel(path):
    total_reads = 0
    excel_list = excel_files = sorted(glob.glob(os.path.join(f'*-confusion-matrix.xlsx')))
    for file in excel_list:
        df = pd.read_excel(file, index_col=0, sheet_name=None)
        total_reads += df['species'].to_numpy().sum()
    return total_reads


def main():
    path = "/data/zhanglab/eric_rangel/centrifuge_results"
    total = count_reads_excel(path)
    print(total)


if __name__ == "__main__":
    main()
