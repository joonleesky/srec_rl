from .base import AbstractModel
from ..utils import all_subclasses
from ..utils import import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractModel)

MODELS = {c.code():c
          for c in all_subclasses(AbstractModel)
          if c.code() is not None}


def init_model(args):
    model = MODELS[args.model_type]
    return model(args)
