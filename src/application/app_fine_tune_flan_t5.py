from io import StringIO
from logging import getLogger, INFO
from os import getenv
from time import time
from typing import Tuple, Dict

from datasets import Dataset
from huggingface_hub import login, logout
from streamlit.runtime.state import SessionStateProxy
from transformers import BatchEncoding

from src.infrastructure.logging import get_stream_handler
from src.infrastructure.datasets import load_samsum_dataset
from src.domain.configuration import limited_samples_count, get_tokenizer_id, get_base_model_id, get_output_dir
from src.infrastructure.transformers import load_tokenizer, load_base_model
from src.domain.transform import concatenate_train_test_data, tokenize_strings, max_sequence_length, preprocess_function
from src.domain.model import get_lora_model, summarize_trainable_parameters, get_data_collator, get_training_arguments, get_trainer
from src.domain.model.optimization import get_optimizers
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.model.evaluation import compute_metrics
from src.infrastructure.huggingface_hub import mock_saving, get_huggingface_hub_connection, upload_log


logger = getLogger(__name__)
# logger.setLevel(INFO)
from logging import DEBUG
logger.setLevel(DEBUG)
log_io = StringIO()
logger.addHandler(get_stream_handler(log_io))


def app(config: SessionStateProxy):
    # Log the configuration.
    for key, val in config.items():
        logger.info(f"{key}: {val}")

    """
    EL+T
    """
    elt_start_epoch = time()
    logger.info(f"EL+T start epoch: {elt_start_epoch}")

    # Load the `samsum` dataset.
    logger.debug("Before: dataset = load_samsum_dataset()")
    dataset = load_samsum_dataset()
    logger.debug("After: dataset = load_samsum_dataset()")

    if config['limit_samples']:
        for dataset_name in dataset:
            dataset[dataset_name] = dataset[dataset_name].select(range(limited_samples_count))

    logger.info(f"Train Dataset Size: {len(dataset['train'])}")
    logger.info(f"Test Dataset Size: {len(dataset['test'])}")

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    logger.debug("Before: tokenizer = load_tokenizer(tokenizer_id)")
    tokenizer = load_tokenizer(tokenizer_id)
    logger.debug("After: tokenizer = load_tokenizer(tokenizer_id)")

    # Batch our data efficiently.
    logger.debug("Before: concatenated_dataset = concatenate_train_test_data(dataset)")
    concatenated_dataset = concatenate_train_test_data(dataset)
    logger.debug("After: concatenated_dataset = concatenate_train_test_data(dataset)")

    logger.debug("Before: tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, 'dialogue')")
    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    logger.debug("After: tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, 'dialogue')")
    logger.debug("Before: max_source_length = max_sequence_length(tokenized_inputs)")
    max_source_length = max_sequence_length(tokenized_inputs)
    logger.debug("After: max_source_length = max_sequence_length(tokenized_inputs)")
    logger.info(f"Max Source Length: {max_source_length}")

    logger.debug("Before: tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, 'summary')")
    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    logger.debug("After: tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, 'summary')")
    logger.debug("Before: max_target_length = max_sequence_length(tokenized_targets, 90)")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    logger.debug("After: max_target_length = max_sequence_length(tokenized_targets, 90)")
    logger.info(f"Max Target Length: {max_target_length}")

    # Preprocess our dataset before training and save it to disk.
    def preprocess_func(target_dataset: Dataset) -> BatchEncoding:
        return preprocess_function(target_dataset, tokenizer, max_source_length, max_target_length)

    logger.debug("Before: tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=[...])")
    tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=["dialogue", "summary", "id"])
    logger.debug("After: tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=[...])")
    logger.info(f"EL+T Elasped Time [s]: {time() - elt_start_epoch}")

    """
    Train and Evaluate
    """
    train_start_epoch = time()
    logger.info(f"Training & Evaluation Start Epoch: {train_start_epoch}")

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    logger.debug("Before: model = load_base_model(base_model_id)")
    model = load_base_model(base_model_id)
    logger.debug("After: model = load_base_model(base_model_id)")

    # Prepare our model for the LoRA int-8 training using peft.
    logger.debug("Before: model = get_lora_model(model)")
    model = get_lora_model(model)
    logger.debug("After: model = get_lora_model(model)")
    logger.info(f"Trainable Parameters: {summarize_trainable_parameters(model)}")

    # Pad our inputs and labels.
    logger.debug("Before: data_collator = get_data_collator(tokenizer, model)")
    data_collator = get_data_collator(tokenizer, model)
    logger.debug("After: data_collator = get_data_collator(tokenizer, model)")

    # Create Trainer instance.
    output_dir = get_output_dir(config['model_size'], config['optim_name'])
    training_arguments = get_training_arguments(output_dir, config['n_epochs'])
    logger.debug("Before: optimizers = get_optimizers(model, config['optim_name'])")
    optimizers = get_optimizers(model, config['optim_name'])
    logger.debug("After: optimizers = get_optimizers(model, config['optim_name'])")
    logger.debug("Before: rouge = load_rouge_metric()")
    rouge = load_rouge_metric()
    logger.debug("After: rouge = load_rouge_metric()")

    def compute_rouge_metric(eval_pred: Tuple[str, str]) -> Dict[str, float]:
        return compute_metrics(eval_pred, tokenizer, rouge)

    logger.debug("Before: trainer = get_trainer(...)")
    trainer = get_trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        training_arguments=training_arguments,
        optimizers=optimizers,
        compute_metrics_function=compute_rouge_metric
    )
    logger.debug("After: trainer = get_trainer(...)")
    model.config.use_cache = False  # silence the warnings.

    # Train model.
    logger.debug("Before: trainer.train()")
    trainer.train()
    logger.debug("After: trainer.train()")
    logger.info(f"Training & Evaluation Elasped Time [s]: {time() - train_start_epoch}")

    # Save our model and upload the log.
    if not mock_saving():
        token = getenv('HUGGINGFACE_TOKEN')
        login(token=token)
        trainer.push_to_hub()
        api = get_huggingface_hub_connection(token=token)
        upload_log(log_io, api)
        logout()
