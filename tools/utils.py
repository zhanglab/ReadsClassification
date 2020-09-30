import os
import sys
import argparse
import json

def get_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-input', type=str, help='folder containing data set', default=os.getcwd())
    parser.add_argument('-output', type=str, help='folder to output data', default=os.getcwd())

    parser.add_argument('-h5', '--hdf5', type=str, help='name to store data as an HDF5 file', default='NN-dataset.h5')
    parser.add_argument('--chunks', action='store_true', help='store HDF5 datasets as chunks', default=None)
    return parser

def process_folder(args):
    def check_input():
        if not os.path.exists(os.path.join(args.input, "Species.tsv")):
            print(f'{args.input} does not contain a Species.tsv file. Input a valid directory')
            sys.exit(1)
        return args.input

    def check_output():
        folders = [os.path.join(args.output, 'tfrecords-{}'.format(args.rn)),
                   os.path.join(args.output, 'tfrecords-{}/train'.format(args.rn)),
                   os.path.join(args.output, 'tfrecords-{}/test'.format(args.rn)),
                   os.path.join(args.output, 'tfrecords-{}/val'.format(args.rn))]

        if not os.path.isdir(args.output):
            os.makedirs(args.output)

        if not os.path.isdir(os.path.join(args.output, 'tfrecords-{}'.format(args.rn))):
            for folder in folders:
                os.makedirs(folder)

        return args.output

    return check_input(), check_output()

def check_h5_ext(h5):
    if h5.endswith('.h5'):
        return h5
    return h5 + ".h5"

def load_json_dictionary(input_file, dir):
    with open(os.path.join(dir, input_file)) as f:
        json_dict = json.load(f)
        return json_dict
