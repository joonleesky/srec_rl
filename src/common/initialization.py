import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
        

def fix_random_seed(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    
class NormInitializer():
    def __init__(self, init_range):
        self.init_range = init_range
        
    def __call__(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=self.init_range)
            m.bias.data.zero_()        
            
        elif isinstance(m, nn.Embedding):
            m.weight.data.normal_(mean=0.0, std=self.init_range)