import sys
import os
import warnings
from dotmap import DotMap
from typing import List

from src.arguments import Parser
from src.datasets import init_dataset
from src.dataloaders import init_dataloader
from src.models import init_model
from src.trainers import init_trainer
from src.envs import init_env

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def main(sys_argv: List[str] = None):
    # Parser
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    configs = Parser(sys_argv).parse()
    args = DotMap(configs, _dynamic=False)

    # Dataset
    dataset = init_dataset(args)
    dataset_path = dataset._get_preprocessed_folder_path()
    dataset = dataset.load_dataset()

    # DataLoader
    dataloader = init_dataloader(args, dataset, dataset_path)
    
    # Reward_model (used to evaluate the performance in offline manner)
    if args.reward_model_dir is not None:
        env = init_env(args, dataset)
    else:
        env = None
    
    # Model
    model= init_model(args)
    
    # Trainer
    trainer = init_trainer(args, dataset, dataloader, env, model)
    trainer.train()
    
    
    
if __name__ == '__main__':
    main()