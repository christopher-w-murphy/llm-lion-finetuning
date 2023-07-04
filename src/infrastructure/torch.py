from typing import Dict, Union

from torch.cuda import memory_stats, is_available, get_device_name, current_device, empty_cache, mem_get_info


def verify_torch_installation() -> Dict[str, Union[bool, str]]:
    return {'is_cuda_available': is_available(), 'cuda_device': get_device_name(current_device())}


def get_memory_stats():
    return memory_stats()


def empty_cache_if_free_memory_is_low(min_memory_needed: float):
    """
    it's worth a shot
    """
    free_gpu_memory, _ = mem_get_info()
    if free_gpu_memory < min_memory_needed:
        empty_cache()
