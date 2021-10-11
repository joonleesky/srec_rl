from abc import *
from pathlib import Path
from dotmap import DotMap
from ..models import init_model
import json
import copy
import torch


class BaseEnv(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.device = args.device
        self.reward_model_dir = self._get_reward_model_dir()
        self.reward_model = self._init_reward_model().to(self.device)
        
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod    
    def step(self, item): 
        pass
    
    @abstractmethod
    def reset(self, user_ids, num_interactions): 
        pass

    def _init_reward_model(self):
        args = self.args
        config_path = self.reward_model_dir + '/config.json'
        model_path = self.reward_model_dir + '/model.pth'

        # check the model existence
        if Path(config_path).is_file() and Path(model_path).is_file():
            pass
        else:
            raise AssertionError("model weights or configuration does not exist for environment")
            
        with open(config_path, "r") as f:
            rm_config = json.load(f)
        rm_config = DotMap(rm_config, _dynamic=False)
        reward_model = init_model(rm_config)
        
        model_state = torch.load(model_path)['model_state_dict']
        reward_model.load(model_state, args.use_parallel)
        
        return reward_model
                
    def _get_reward_model_dir(self):
        args = self.args
        return 'simulator' + '/' + args.dataset_type + '/' + args.reward_model_dir
