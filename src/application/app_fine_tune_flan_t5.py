from logging import getLogger, INFO
from time import time
from typing import Tuple, Dict

from datasets import Dataset
from huggingface_hub import login
from transformers import BatchEncoding

from src.infrastructure.logging import get_logging_dir, get_file_handler
from src.infrastructure.streamlit import ConfigType, get_secret
from src.infrastructure.datasets import load_samsum_dataset
from src.domain.configuration import limited_samples_count, get_tokenizer_id, get_base_model_id, get_output_dir
from src.infrastructure.transformers import load_tokenizer, load_base_model
from src.domain.transform import concatenate_train_test_data, tokenize_strings, max_sequence_length, preprocess_function
from src.domain.model import get_lora_model, summarize_trainable_parameters, get_data_collator, get_training_arguments, get_trainer
from src.domain.model.optimization import get_optimizers
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.model.evaluation import compute_metrics
from src.infrastructure.huggingface_hub import get_huggingface_hub_connection, upload_results_file, upload_file


logger = getLogger(__name__)
logger.setLevel(INFO)
logging_dir = get_logging_dir()
logger.addHandler(get_file_handler(logging_dir))


def app(config: ConfigType):
    # Log Config
    for key, val in config.items():
        logger.info(f"{key}: {val}")

    """
    EL+T
    """
    # step = 1
    # config['steps'][step] = dict()
    # config['steps'][step]['start_epoch'] = time()
    elt_start_epoch = time()
    logger.info(f"EL+T start epoch: {elt_start_epoch}")

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    if config['limit_samples']:
        for dataset_name in dataset:
            dataset[dataset_name] = dataset[dataset_name].select(range(limited_samples_count))

    # config['steps'][step]['train_dataset_size'] = len(dataset['train'])
    # config['steps'][step]['test_dataset_size'] = len(dataset['test'])
    logger.info(f"Train Dataset Size: {len(dataset['train'])}")
    logger.info(f"Test Dataset Size: {len(dataset['test'])}")

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    # config['steps'][step]['max_source_length'] = max_source_length
    logger.info(f"Max Source Length: {max_source_length}")

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    # config['steps'][step]['max_target_length'] = max_target_length
    logger.info(f"Max Target Length: {max_target_length}")

    # Preprocess our dataset before training and save it to disk.
    def preprocess_func(target_dataset: Dataset) -> BatchEncoding:
        return preprocess_function(target_dataset, tokenizer, max_source_length, max_target_length)

    tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=["dialogue", "summary", "id"])
    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['test']

    # config['steps'][step]['elasped_time'] = time() - config['steps'][step]['start_epoch']
    logger.info(f"EL+T Elasped Time: {time() - elt_start_epoch}")

    """
    Train and Evaluate
    """
    # step = 2
    # config['steps'][step] = dict()
    # config['steps'][step]['start_epoch'] = time()
    train_start_epoch = time()
    logger.info(f"Training & Evaluation Start Epoch: {train_start_epoch}")

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    # config['steps'][step]['trainable_parameters'] = summarize_trainable_parameters(model)
    logger.info(f"Trainable Parameters: {summarize_trainable_parameters(model)}")

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
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_arguments=training_arguments,
        optimizers=optimizers,
        compute_metrics_function=compute_rouge_metric
    )
    model.config.use_cache = False  # silence the warnings.

    # Train model.
    trainer.train()

    # Save our model.
    login(token=get_secret('HUGGINGFACE_TOKEN'))
    trainer.push_to_hub()

    # config['steps'][step]['elasped_time'] = time() - config['steps'][step]['start_epoch']
    logger.info(f"Training & Evaluation Elasped Time: {time() - train_start_epoch}")
    # preserve results
    api = get_huggingface_hub_connection()
    # upload_results_file(config, api, step)

    # trainer.save_metrics("train", results.metrics)
    # upload_file(output_dir, "train_results.json", api)

    upload_file(str(get_logging_dir()), api)
