from .base import AbstractDataset
from ..utils import all_subclasses
from ..utils import import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractDataset)


DATASETS = {c.code():c
            for c in all_subclasses(AbstractDataset)
            if c.code() is not None}


def init_dataset(args):
    dataset = DATASETS[args.dataset_type]
    return dataset(args)