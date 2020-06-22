#!/usr/bin/env python
# coding: utf-8
# Import all necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals

from .prepare_data import *
import argparse

base_dict = {'A':0, 'C':1, 'T':2, 'G':3}

# Converts the DNA sequence into one-hot encoded sequence
def parse_seq(sequence, args):
    integer_encoded = [base_dict[base] for base in sequence]
    return integer_encoded

def parse_args():
    #### SET MODEL PARAMETERS #####
    parser = argparse.ArgumentParser()
    parser.add_argument("-classes", type=int, help="number of classes", required=True)
    parser.add_argument("-l", "--length", type=int, help="size of sequences", default=150)
    args = parser.parse_args()
    args.model = 'base-emb'
    return args

if __name__ == '__main__':
    args = parse_args()
    fastq_files = get_info()
    multiprocesses(fastq_files, args)
