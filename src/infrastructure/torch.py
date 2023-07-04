from os import environ
from typing import Dict, Union

from torch.cuda import memory_stats, is_available, get_device_name, current_device, empty_cache


def verify_torch_installation() -> Dict[str, Union[bool, str]]:
    return {'is_cuda_available': is_available(), 'cuda_device': get_device_name(current_device())}


def get_memory_stats():
    return memory_stats()


def manage_cuda_cache(optim_name: str):
    if optim_name == 'Lion 8-bit':
        empty_cache()


def set_cuda_config(optim_name: str):
    if optim_name == 'Lion 8-bit':
        config = {
            'max_split_size_mb': 256,
            'garbage_collection_threshold': 0.95
        }
        environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join([f"{key}:{val}" for key, val in config.items()])
