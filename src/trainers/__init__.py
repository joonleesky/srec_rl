from .base import AbstractTrainer
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractTrainer)


TRAINERS = {c.code():c
            for c in all_subclasses(AbstractTrainer)
            if c.code() is not None}


def init_trainer(args, model, train_loader, val_loader, test_loader, local_exp_path):
    trainer = TRAINERS[args.trainer_type]
    return trainer(args, model, train_loader, val_loader, test_loader, local_exp_path)