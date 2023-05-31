from typing import Dict, Union

from torch.cuda import is_available, get_device_name, current_device


def verify_torch_installation() -> Dict[str, Union[bool, str]]:
    return {'is_cuda_available': is_available(), 'cuda_device': get_device_name(current_device())}
