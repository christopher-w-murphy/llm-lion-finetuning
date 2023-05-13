label_pad_token_id: int = -100


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


def get_output_dir(model_size: str) -> str:
    if model_size.lower() == 'small':
        return "lora-flan-t5-small"
    elif model_size.lower() == 'xxl':
        return "lora-flan-t5-xxl"
    else:
        raise ValueError('Invalid model size.')


def get_n_epochs(model_size: str) -> int:
    if model_size.lower() == 'small':
        return 1
    elif model_size.lower() == 'xxl':
        return 5
    else:
        raise ValueError('Invalid model size.')


def get_peft_model_id(model_size: str) -> str:
    if model_size.lower() == 'small':
        return "results-small"
    elif model_size.lower() == 'xxl':
        return "results"
    else:
        raise ValueError('Invalid model size.')
