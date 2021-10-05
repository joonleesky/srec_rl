import sys
import os
from dotmap import DotMap
from typing import List

from src.arguments import Parser
from src.datasets import init_dataset
from src.dataloaders import init_dataloader
from src.models import init_model
from src.trainers import init_trainer


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
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
    
    print('num_interactions:',args.num_interactions)
    print('num_user:', args.num_users)
    print('num_item:', args.num_items)
    print('num_ratings:', args.num_ratings)
    
    # Model
    model= init_model(args)

    # Trainer
    trainer = init_trainer(args, model, train_loader, val_loader, test_loader)
    trainer.train()
    
    
    
if __name__ == '__main__':
    main()