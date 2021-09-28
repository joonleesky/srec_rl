from ..utils import fix_random_seed
import torch.nn as nn
from abc import *


class AbstractModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_init_seed = args.model_init_seed
        self.model_init_range = args.model_init_range

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def init_weights(self):
        fix_random_seed(self.args.model_init_seed)
        self.apply(self._init_weights)

    # override in each submodel if needed
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_init_range)
            module.bias.data.zero_()        
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_init_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        else:
            raise NotImplemented

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