import sys
from dotmap import DotMap
from typing import List
from parser import Parser


def main(sys_argv: List[str] = None):
    if sys_argv is None:
        sys_argv = sys.argv[1:]
    configs = Parser(sys_argv).parse()
    args = DotMap(configs, _dynamic=False)
    # Train

if __name__ == '__main__':
    main()