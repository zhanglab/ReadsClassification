import sys
import os

def check_argv(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()
    return argv

def process_folder(args):
    folder = args.folder
    if args.folder is None:
        folder = os.getcwd()
    if not os.path.exists(folder + "/Species.tsv"):
        print(f'{folder} does not contain a Species.tsv file. Input a valid directory', file=sys.stderr)
        sys.exit(1)
    return folder
