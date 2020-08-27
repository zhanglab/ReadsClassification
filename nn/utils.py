import glob
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
    Check the input folder to see if it has the necessary hdf5 file
    Check the output folder to see if it exists
    """
    def check_input():
        # Check for h5 file if not specified
        if not args.hdf5:
            args.hdf5 = glob.glob(os.path.join(args.input, "*.h5"))
            if len(args.hdf5) > 1:
                sys.exit(f'{args.input} contains multiple .h5 files. '
                         f'Input a specific file using --hdf5.')
            args.hdf5 = args.hdf5[0]

        if not os.path.exists(args.hdf5):
            sys.exit(f'{args.hdf5} does not exist. Input a valid h5 file')
        return args.hdf5

    def check_output():
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    check_output()
    return check_input(), class_mapping

# def process_checkpoint(args):
#     """
#     Check if checkpoint exists
#     """
#     if not os.path.exists(f'{args.checkpoint}'):
#         raise ValueError(f'{args.checkpoint} does not exist.')

def process_model(args):
    """
    Build model object
    """
    model = models._models[args.model]
    if args.checkpoint is not None:
        #process_checkpoint(args)
        model = model.load_weights(args.checkpoint)
    else:
        model = model(args)
    return model
