import sys
from dotmap import DotMap
from typing import List

from src.arguments import Parser
from src.datasets import init_dataset
from src.dataloaders import init_dataloader
from src.models import init_model


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
    print('num_interactions:',dataset['num_interactions'])
    print('num_user:', len(dataset['umap']))
    print('num_item:', len(dataset['smap']))
    
    import pdb
    pdb.set_trace()
    
    # DataLoader
    dataloader = init_dataloader(args, dataset, dataset_path)
    train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()
    
    # Reward Model

    # Behavior Model
    
    # RL Model
    model = init_model(args)
    
    
    
    # Train

if __name__ == '__main__':
    main()