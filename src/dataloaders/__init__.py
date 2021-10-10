from .base import BaseDataloader
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseDataloader)


DATALOADERS = {c.code():c
               for c in all_subclasses(BaseDataloader)
               if c.code() is not None}


def init_dataloader(args, dataset, dataset_path):
    dataloader = DATALOADERS[args.dataloader_type]
    return dataloader(args, dataset, dataset_path)