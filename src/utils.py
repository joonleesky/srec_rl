from pathlib import Path
from importlib import import_module
import zipfile
import wget
import inspect
import sys
import os
import pkgutil
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np


#############
## Dataset ##
#############
def download(url, savepath):
    wget.download(url, str(savepath))


def unzip(zippath, savepath):
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()

##############
## Settings ##
##############
def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def import_all_subclasses(_file, _name, _class):
    modules = get_all_submodules(_file, _name)
    for m in modules:
        for i in dir(m):
            attribute = getattr(m, i)
            if inspect.isclass(attribute) and issubclass(attribute, _class):
                setattr(sys.modules[_name], i, attribute)


def get_all_submodules(_file, _name):
    modules = []
    _dir = os.path.dirname(_file)
    for _, name, ispkg in pkgutil.iter_modules([_dir]):
        module = import_module('.' + name, package=_name)
        modules.append(module)
        if ispkg:
            modules.extend(get_all_submodules(module.__file__, module.__name__))
    return modules


def fix_random_seed(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
