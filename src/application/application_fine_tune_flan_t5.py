from time import time
from typing import Dict, Any

from src.infrastructure.transformers import load_base_model, load_tokenizer
from src.infrastructure.datasets import load_tokenized_train_dataset
from src.domain.configuration import get_tokenizer_id, get_base_model_id, get_peft_model_id
from src.domain.model import get_lora_model, get_data_collator, get_trainer, summarize_trainable_parameters


def app_fine_tune(config: Dict[str, Any]):
    config['step2_start'] = time()

    # Load the needed results from step 1.
    train_dataset = load_tokenized_train_dataset()
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    config['trainable_parameters'] = summarize_trainable_parameters(model)

    # Pad our inputs and labels.
    data_collator = get_data_collator(tokenizer, model)

    # Create Trainer instance.
    trainer = get_trainer(model, data_collator, train_dataset, config)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # Train model.
    trainer.train()

    # Save our model to use it for inference and evaluate it.
    peft_model_id = get_peft_model_id(config['model_size'])
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    config['step2_end'] = time()
    config['step2_time_diff'] = config['step2_end'] - config['step2_start']
