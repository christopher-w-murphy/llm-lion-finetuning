from io import StringIO
from logging import getLogger
from os import getenv
from time import time
from typing import Tuple, Dict

from datasets import Dataset
from huggingface_hub import login, logout
from streamlit.runtime.state import SessionStateProxy
from torch.cuda import memory_summary, memory_stats
from transformers import BatchEncoding

from src.infrastructure.logging import get_log_level, get_stream_handler
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
logger.setLevel(get_log_level())
# log_io = StringIO()
# logger.addHandler(get_stream_handler(log_io))


def app(config: SessionStateProxy):
    log = dict()
    log['streamlit_config'] = {key: val for key, val in config.items()}
    # Log the configuration.
    for key, val in log['streamlit_config'].items():
        logger.info(f"streamlit_config - {key}: {val}")

    """
    EL+T
    """
    log['elt'] = dict()
    # elt_start_epoch = time()
    # logger.info(f"EL+T start epoch: {elt_start_epoch}")
    log['elt']['start_epoch'] = time()

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    if config['limit_samples']:
        for dataset_name in dataset:
            dataset[dataset_name] = dataset[dataset_name].select(range(limited_samples_count))

    # logger.info(f"Train Dataset Size: {len(dataset['train'])}")
    # logger.info(f"Test Dataset Size: {len(dataset['test'])}")
    log['elt']['train_dataset_size'] = len(dataset['train'])
    log['elt']['test_dataset_size'] = len(dataset['test'])

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    # logger.info(f"Max Source Length: {max_source_length}")
    log['elt']['max_source_length'] = max_source_length

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    # logger.info(f"Max Target Length: {max_target_length}")
    log['elt']['max_target_length'] = max_target_length

    # Preprocess our dataset before training and save it to disk.
    def preprocess_func(target_dataset: Dataset) -> BatchEncoding:
        return preprocess_function(target_dataset, tokenizer, max_source_length, max_target_length)

    tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=["dialogue", "summary", "id"])
    # logger.info(f"EL+T Elasped Time [s]: {time() - elt_start_epoch}")
    log['elt']['elasped_time'] = time() - log['elt']['start_epoch']
    for key, val in log['elt'].items():
        logger.info(f"elt - {key}: {val}")

    """
    Train and Evaluate
    """
    log['train_and_eval'] = dict()
    # train_start_epoch = time()
    # logger.info(f"Training & Evaluation Start Epoch: {train_start_epoch}")
    log['train_and_eval']['start_time'] = time()

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    # logger.info(f"Trainable Parameters: {summarize_trainable_parameters(model)}")
    log['train_and_eval']['trainable_parameters_summary'] = summarize_trainable_parameters(model)

    # Pad our inputs and labels.
    data_collator = get_data_collator(tokenizer, model)

    # Create Trainer instance.
    output_dir = get_output_dir(config['model_size'], config['optim_name'])
    training_arguments = get_training_arguments(output_dir, config['n_epochs'])
    optimizers = get_optimizers(model, config['optim_name'])
    rouge = load_rouge_metric()

    def compute_rouge_metric(eval_pred: Tuple[str, str]) -> Dict[str, float]:
        return compute_metrics(eval_pred, tokenizer, rouge)

    trainer = get_trainer(
        model=model,
        # tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        training_arguments=training_arguments,
        optimizers=optimizers,
        compute_metrics_function=compute_rouge_metric
    )
    model.config.use_cache = False  # silence the warnings.

    # Train model.
    trainer.train()
    # logger.info(f"Model Memory Footprint: {trainer.model.get_memory_footprint(return_buffers=False)}")
    # logger.info(f"GPU Memory Summary:\n{memory_summary()}")
    # logger.info(f"Training & Evaluation Elasped Time [s]: {time() - train_start_epoch}")
    log['train_and_eval']['trainer_log'] = trainer.state.log_history
    log['train_and_eval']['model_memory_footprint'] = trainer.model.get_memory_footprint(return_buffers=False)
    log['train_and_eval']['cuda_memory_stats'] = memory_stats()
    log['train_and_eval']['elasped_time'] = time() - log['train_and_eval']['start_epoch']
    for key, val in log['train_and_eval'].items():
        logger.info(f"train_and_eval - {key}: {val}")

    # Save our model and upload the log.
    if not mock_saving():
        token = getenv('HUGGINGFACE_TOKEN')
        login(token=token)
        # trainer.push_to_hub()
        trainer.model.push_to_hub(output_dir)
        api = get_huggingface_hub_connection(token=token)
        # upload_log(log_io, api)
        upload_log(log, api)
        logout()
