from bitsandbytes.optim import AdamW8bit, Lion8bit
from bitsandbytes.optim.optimizer import Optimizer8bit
from peft import PeftModel


adamw_hyperparameters = {
    'lr': 3e-5,
    'betas': (0.9, 0.99)
}

lion_hyperparameters = {
    'lr': 3e-6,
    'betas': (0.95, 0.98)
}


def get_optimizer(model: PeftModel, optim_name: str) -> Optimizer8bit:
    if optim_name.lower() == 'adamw':
        return AdamW8bit(model.parameters(), **adamw_hyperparameters)
    elif optim_name.lower() == 'lion':
        return Lion8bit(model.parameters(), **lion_hyperparameters)
    else:
        raise ValueError(f"Invalid optimizer name: {optim_name}.")
