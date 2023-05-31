from typing import Tuple

from bitsandbytes.optim import AdamW8bit
from lion_pytorch.lion_pytorch import Lion
from peft import PeftModelForSeq2SeqLM
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_constant_schedule

from src.domain.configuration import process_optim_name


adamw_hyperparameters = {
    'lr': 1e-3,  # higher learning rate
    'betas': (0.9, 0.99)
}

lion_hyperparameters = {
    'lr': 1e-4,  # order of magnitude smaller than Adam
    'betas': (0.95, 0.98)
}


def get_optimizer(model: PeftModelForSeq2SeqLM, optim_name: str) -> Optimizer:
    optim_name = process_optim_name(optim_name)
    if optim_name == 'adamw32bit':
        return AdamW(model.parameters(), **adamw_hyperparameters)
    elif optim_name == 'adamw8bit':
        return AdamW8bit(model.parameters(), **adamw_hyperparameters)
    elif optim_name == 'lion32bit':
        return Lion(model.parameters(), **lion_hyperparameters)
    else:
        raise ValueError(f"Invalid optimizer name: {optim_name}.")


def get_optimizers(model: PeftModelForSeq2SeqLM, optim_name: str) -> Tuple[Optimizer, LambdaLR]:
    optimizer = get_optimizer(model, optim_name)
    lr_scheduler = get_constant_schedule(optimizer)
    return optimizer, lr_scheduler
