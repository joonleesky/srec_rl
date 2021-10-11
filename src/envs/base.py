from abc import *
from pathlib import Path
from dotmap import DotMap
from ..models import init_model
import json
import copy
import torch


class BaseEnv(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.reward_model_dir = self._get_reward_model_dir()
        self.reward_model = self._init_reward_model().to(self.device)
        
    def step(self, action): 
        pass
    
    
    def reset(self, item_ids, rating_ids): 
        self.state = {'item_ids':item_ids, 
                      'rating_ids':rating_ids}
        # for safeness
        _state = copy.deepcopy(self.state)
        return _state

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
        reward_model = init_model(reward_model_config)
        
        return reward_model
                
    def _get_reward_model_dir(self):
        args = self.args
        return 'simulator' + '/' + args.dataset_type + '/' + args.reward_model_dir
