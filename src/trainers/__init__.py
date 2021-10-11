from .base import BaseTrainer
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseTrainer)


TRAINERS = {c.code():c
            for c in all_subclasses(BaseTrainer)
            if c.code() is not None}


def init_trainer(args, dataset, dataloader, env, model):
    trainer = TRAINERS[args.trainer_type]
    return trainer(args, dataset, dataloader, env, model)