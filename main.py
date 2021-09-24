import sys
from dotmap import DotMap
from typing import List

from src.arguments import Parser
from src.datasets import init_dataset

def main(sys_argv: List[str] = None):
    # Parser
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    configs = Parser(sys_argv).parse()
    args = DotMap(configs, _dynamic=False)

    # Dataset
    dataset = init_dataset(args)
    dataset = dataset.load_dataset()
    import pdb
    pdb.set_trace()

    # DataLoader




    # Train

if __name__ == '__main__':
    main()