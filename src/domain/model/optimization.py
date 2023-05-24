from typing import Tuple

from bitsandbytes.optim import Lion8bit
from peft import PeftModel
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_constant_schedule


adamw_hyperparameters = {
    'lr': 3e-5,
    'betas': (0.9, 0.99)
}

lion_hyperparameters = {
    'lr': 3e-6,
    'betas': (0.95, 0.98)
}

batch_size: int = 128


def get_optimizer(model: PeftModel, optim_name: str) -> Optimizer:
    if optim_name.lower() == 'adamw':
        return AdamW(model.parameters(), **adamw_hyperparameters)
    elif optim_name.lower() == 'lion':
        return Lion8bit(model.parameters(), **lion_hyperparameters)
    else:
        raise ValueError(f"Invalid optimizer name: {optim_name}.")


def get_optimizers(model: PeftModel, optim_name: str) -> Tuple[Optimizer, LambdaLR]:
    optimizer = get_optimizer(model, optim_name)
    lr_scheduler = get_constant_schedule(optimizer)
    return optimizer, lr_scheduler
