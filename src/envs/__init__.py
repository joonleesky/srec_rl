from .base import BaseEnv
from ..common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, BaseEnv)


ENVS = {c.code():c
        for c in all_subclasses(BaseEnv)
        if c.code() is not None}


def init_env(args, dataset):
    env = ENVS[args.env_type]
    return env(args, dataset)