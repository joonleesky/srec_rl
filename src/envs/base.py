from abc import *
from pathlib import Path
from dotmap import DotMap
from ..models import init_model
import json


class BaseEnv(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.reward_model_dir = 'simulator' + '/' + args.dataset_type + '/' + args.reward_model_dir
        self.reward_model = self._init_reward_model()

    def _init_reward_model(self):
        args = self.args
        # check the environment validity
        reward_model_config_path = self.reward_model_dir + '/config.json'
        #reward_model_path = args.reward_model_dir
        with open(reward_model_config_path, "r") as f:
            reward_model_config = json.load(f)
        reward_model_config = DotMap(reward_model_config, _dynamic=False)
        reward_model = init_model(reward_model_config)
        
        raise NotImplementedError
