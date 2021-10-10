import torch.nn as nn
from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def load(self, path):
        """
        chk_dict = torch.load(os.path.abspath(path))
        model_state_dict = chk_dict[STATE_DICT_KEY] if STATE_DICT_KEY in chk_dict else chk_dict['state_dict']
        d = {}
        # this is for stupid reason
        for k, v in model_state_dict.items():
            if k.startswith('model.'):
                d[k[6:]] = v
            else:
                d[k] = v
        model_state_dict = d
        model.load_state_dict(model_state_dict)
        """