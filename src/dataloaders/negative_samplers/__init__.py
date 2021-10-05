from .base import AbstractNegativeSampler
from ...common.class_utils import all_subclasses, import_all_subclasses
import_all_subclasses(__file__, __name__, AbstractNegativeSampler)


NEGATIVE_SAMPLERS = {c.code():c
                     for c in all_subclasses(AbstractNegativeSampler)
                     if c.code() is not None}


def init_negative_sampler(code, 
                          user2dict, 
                          num_users, 
                          num_items, 
                          negative_sample_size, 
                          negative_sampling_seed, 
                          save_folder):
    
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(user2dict, num_users, num_items, negative_sample_size, negative_sampling_seed, save_folder)
