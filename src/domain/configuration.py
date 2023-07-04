from typing import Set


label_pad_token_id: int = -100
limited_samples_count: int = 2
optim_names: Set[str] = {"AdamW 32-bit", "AdamW 8-bit", "Lion 32-bit", "Lion 8-bit"}
model_sizes: Set[str] = {"Small", "XXL"}
memory_needed_to_push_to_hub: float = 40.00 * 2**20  # convert MiB to bytes


def get_base_model_id(model_size: str) -> str:
    if model_size.lower() == 'small':
        return "google/flan-t5-small"
    elif model_size.lower() == 'xxl':
        return "philschmid/flan-t5-xxl-sharded-fp16"  # use sharded version so we can fine-tune using a single GPU
    else:
        raise ValueError('Invalid model size.')


def get_tokenizer_id(model_size: str) -> str:
    if model_size.lower() == 'small':
        return "google/flan-t5-small"
    elif model_size.lower() == 'xxl':
        return "google/flan-t5-xxl"
    else:
        raise ValueError('Invalid model size.')


def process_optim_name(optim_name: str) -> str:
    return optim_name.replace(' ', '').replace('-', '').lower()


def get_output_dir(model_size: str, optim_name: str) -> str:
    return f"lora-flan-t5-{model_size.lower()}-{process_optim_name(optim_name)}"
