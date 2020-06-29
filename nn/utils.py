import sys
import os

from . import models

def check_argv(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()
    return argv

def process_folder(args):
    """
    Check the input folder to see if it has the necessary files
    Check the output folder to see if it exists
    """
    def check_input():
        req_files = ['/class_mapping.json', '/train_data_{0}.npy'.format(args.model),
                     '/val_data_{0}.npy'.format(args.model)]
        for file in req_files:
            if not os.path.exists(args.input + file):
                print(f'{args.input} does not contain a {file} file. Input a valid directory', file=sys.stderr)
                sys.exit(1)
        return args.input

    def check_output():
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        return args.output

    return check_input(), check_output()

def process_checkpoint(args):
    """
    Check if checkpoint exists
    """
    if not os.path.exists(f'{args.checkpoint}'):
        raise ValueError(f'{args.checkpoint} does not exist.')

def process_model(args):
    """
    Build model object
    """
    model = models._models[args.model]
    if args.checkpoint is not None:
        process_checkpoint(args)
        model = model.load_weights(args.checkpoint)
    else:
        model = model(args)
    return model
