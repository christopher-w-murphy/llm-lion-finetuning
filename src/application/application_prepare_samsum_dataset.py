from time import time
from typing import Dict, Any

from src.infrastructure.datasets import load_samsum_dataset, train_datapath, eval_datapath
from src.infrastructure.transformers import load_tokenizer
from src.domain.configuration import get_tokenizer_id
from src.domain.transform import (
    concatenate_train_test_data,
    tokenize_strings,
    max_sequence_length,
    preprocess_function
)


def app_prepare_dataset(config: Dict[str, Any]):
    config['step1_start'] = time()

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    if config['limit_samples'] and 'n_samples' in config and config['n_samples']:
        dataset['train'] = dataset['train'].select(range(config['n_samples']))
        dataset['test'] = dataset['test'].select(range(config['n_samples']))

    config['train_dataset_size'] = len(dataset['train'])
    config['test_dataset_size'] = len(dataset['test'])

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    config['max_source_length'] = max_source_length

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    config['max_target_length'] = max_target_length

    # Preprocess our dataset before training and save it to disk.
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_source_length, max_target_length),
        batched=True,
        remove_columns=["dialogue", "summary", "id"]
    )
    tokenized_dataset["train"].save_to_disk(train_datapath)
    tokenized_dataset["test"].save_to_disk(eval_datapath)

    config['step1_end'] = time()
    config['step1_time_diff'] = config['step1_end'] - config['step1_start']
