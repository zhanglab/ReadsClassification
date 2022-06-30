import os
import glob
import pandas as pd
import numpy as np


def count_reads_excel(path, rank):
    total_reads = 0
    excel_list = sorted(glob.glob(os.path.join(path, f'*-confusion-matrix.xlsx')))
    for file in excel_list:
        df = pd.read_excel(file, index_col=0, sheet_name=None)
        total_reads += df[rank].to_numpy().sum()
    return total_reads


def main():
    path = "/data/zhanglab/eric_rangel/centrifuge_results"
    ranks = ["species", "genus", "family", "order", "class", "phylum"]
    for r in ranks:
        total = count_reads_excel(path, r)
        print(f'total for {r} level: {total}')


if __name__ == "__main__":
    main()
