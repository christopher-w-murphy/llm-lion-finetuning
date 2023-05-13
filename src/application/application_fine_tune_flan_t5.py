from streamlit import write

from src.infrastructure.transformers import load_base_model, load_tokenizer
from src.infrastructure.datasets import load_tokenized_train_dataset
from src.domain.configuration import get_tokenizer_id, get_base_model_id, get_peft_model_id
from src.domain.model import get_lora_model, get_data_collator, get_trainer, summarize_trainable_parameters


def app_fine_tune(model_size: str):
    # Load the needed results from step 1.
    train_dataset = load_tokenized_train_dataset()
    write('Loaded Training Data')
    tokenizer_id = get_tokenizer_id(model_size)
    tokenizer = load_tokenizer(tokenizer_id)
    write('Loaded Tokenizer')

    # Load the base model.
    base_model_id = get_base_model_id(model_size)
    model = load_base_model(base_model_id)
    write('Loaded Base Model')

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    write('Loaded LoRA model')
    write(summarize_trainable_parameters(model))

    # Pad our inputs and labels.
    data_collator = get_data_collator(tokenizer, model)
    write('Loaded Data Collator')

    # Create Trainer instance.
    trainer = get_trainer(model, data_collator, train_dataset, model_size)
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    write('Created trainer instance')

    # Train model.
    trainer.train()
    write('Training complete')

    # Save our model to use it for inference and evaluate it.
    peft_model_id = get_peft_model_id(model_size)
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    write('Save complete')
