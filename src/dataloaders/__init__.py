from .base import AbstractDataloader
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractDataloader)


DATALOADERS = {c.code():c
               for c in all_subclasses(AbstractDataloader)
               if c.code() is not None}


def init_dataloader(args, dataset, dataset_path):
    dataloader = DATALOADERS[args.dataloader_type]
    return dataloader(args, dataset, dataset_path)