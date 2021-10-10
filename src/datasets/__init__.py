from .base import BaseDataset
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseDataset)


DATASETS = {c.code():c
            for c in all_subclasses(BaseDataset)
            if c.code() is not None}


def init_dataset(args):
    dataset = DATASETS[args.dataset_type]
    return dataset(args)