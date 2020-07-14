import os
import argparse

def get_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-input', type=str, help="folder containing data set", default=os.getcwd())
    parser.add_argument('-output', type=str, help="folder to output data", default=os.getcwd())
    parser.add_argument("-c", "--classes", type=int, help="number of classes", required=True)
    return parser

def process_folder(args):
    def check_input():
        if not os.path.exists(args.input + "/Species.tsv"):
            print(f'{args.input} does not contain a {file} file. Input a valid directory', file=sys.stderr)
            sys.exit(1)
        return args.input

    def check_output():
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        return args.output

    return check_input(), check_output()
