import subprocess
import argparse
import json
import copy
import itertools
from multiprocessing import Pool


def run_experiment(experiment):
    cmd = ['python', 'main.py']
    for key, value in experiment.items():
        cmd.append(key)
        cmd.append(value)
    return subprocess.check_output(cmd)


if __name__ == '__main__':
    default = {'--exp_name': '1020_pop',
               '--template': 'pop'}

    dataset_types = ['eachmovie'] #, 'ml-20m', 'netflix']
    seeds = ['0', '1', '2', '3', '4']
    confidence_levels = ['0.4', '0.5']
    num_devices = 4
    num_exp_per_device = 2
    pool_size = num_devices * num_exp_per_device

    experiments = []
    device = 0
    for dataset, seed, confidence in itertools.product(*[dataset_types, seeds, confidence_levels]):
        exp = copy.deepcopy(default)
        exp['--dataset_type'] = dataset
        if dataset == 'eachmovie':
            exp['--min_rating'] = '5'
        elif dataset == 'netflix':
            exp['--num_epochs'] = '20'
        
        exp['--seed'] = seed
        exp['--confidence_level'] = confidence
        exp['--device'] = 'cuda:' + str(int(device % num_devices))
        experiments.append(exp)
        device += 1

    pool = Pool(pool_size)
    stdouts = pool.map(run_experiment, experiments, chunksize=1)
    pool.close()