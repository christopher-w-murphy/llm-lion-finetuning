from time import time
from typing import Tuple, Dict

from huggingface_hub import HfApi

from src.infrastructure.streamlit import ConfigType
from src.infrastructure.transformers import load_base_model, load_tokenizer
from src.infrastructure.datasets import load_tokenized_train_dataset
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.configuration import get_tokenizer_id, get_base_model_id, get_peft_model_id
from src.domain.model import get_lora_model, get_data_collator, get_training_arguments, get_trainer, summarize_trainable_parameters
from src.domain.model.optimization import get_optimizers
from src.domain.model.evaluation import compute_metrics
from src.infrastructure.huggingface_hub import upload_results_file


def app_fine_tune(config: ConfigType, api: HfApi):
    step = 2
    config['steps'][step] = dict()
    config['steps'][step]['start_epoch'] = time()

    # Load the needed results from step 1.
    train_dataset = load_tokenized_train_dataset()
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    config['steps'][step]['trainable_parameters'] = summarize_trainable_parameters(model)

    # Pad our inputs and labels.
    data_collator = get_data_collator(tokenizer, model)

    # Create Trainer instance.
    training_arguments = get_training_arguments(config['model_size'], config['n_epochs'])
    optimizers = get_optimizers(model, config['optim_name'])
    rouge = load_rouge_metric()

    def compute_rouge_metric(eval_pred: Tuple[str, str]) -> Dict[str, float]:
        return compute_metrics(eval_pred, tokenizer, rouge)

    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        training_arguments=training_arguments,
        optimizers=optimizers,
        compute_metrics_function=compute_rouge_metric
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    # Train model.
    trainer.train()

    # Save our model to use it for inference and evaluate it.
    peft_model_id = get_peft_model_id(config['model_size'])
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    config['steps'][step]['elasped_time'] = time() - config['steps'][step]['start_epoch']
    # preserve results
    upload_results_file(config, api, step)
