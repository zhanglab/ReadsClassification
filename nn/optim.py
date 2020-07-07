from .train import parse_args
import os
import sys
import copy
import optuna
import argparse

class MetricsCallback():
    def __init__(self):
        super().__init__()
        self.metric = []

class Objective:
    """
    Helper class to build the objective function
    """
    def __init__(self, model_cls, hparams, dataset, monitor_metric='val_loss'):
        self.model_cls = model_cls
        self.hparams = hparams
        self.dataset = dataset
        self.monitor_metric = monitor_metric

    def get_hparams(self, trial):
        ret = dict()
        ret['dropout_rate'] = trial.suggest_uniform('dropout_rate', 0.0, 1.0, 0.1)
        ret['batch_size'] = trial.suggest_int('batch_size', 0, 100, 10)
        ret['hidden_size'] = trial.suggest_int('hidden_size', 0, 100, 10)
        ret['embedding_size'] = trial.suggest_int('embedding_size', 0, 100, 10)
        return ret

    def __call__(self, trial):
        # Set hyperparameters
        hparams = copy.copy(self.hparams)
        for key, value in self.get_hparams(trial).items():
            setattr(hparams, key, value)
        model = self.model_cls(hparams)
        model.set_dataset(self.dataset)

def parse_args(argv=None):
    """
    Parse arguments for optimization
    """
    if argv is None:
        argv = sys.argv[1:]
    if isinstance(argv, str):
        argv = argv.strip().split()

    desc = "Run network hyperparameter optimization"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('storage', help='Optuna storage database')
    parser.add_argument('study_name', help='the study to run trials for')
    parser.add_argument('output', type=str, help='file to write checkpoints to')
    parser.add_argument('-t', '--n_trials', type=int, default=100, help='number of trials to run')

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)

    args.parser.parse_args(argv)
    study = optuna.create_study(storage=args.storage,
                                study_name=args.study_name,
                                load_if_exists=True)
    n_trials = args.n_trials
    del args.storage
    del args.study_name
    del args.n_trials

    return study, n_trials, args

def optuna_run(train_cmdline, object_cls=Objective):
    from .train import parse_args as parse_train_args, process_args as process_train_args

    study, n_trials, args = parse_args()
    train_args = parse_train_args(argv=train_cmdline)
    for key, value in vars(args).items():
        setattr(train_args, key, value)

    model_cls = process_train_args(train_args)
    objective = object_cls(model_cls)
    study.optimize(objective, n_trials=n_trials)

if __name__ == '__main__':
    optuna_run()