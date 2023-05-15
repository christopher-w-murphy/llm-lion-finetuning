from time import time

from huggingface_hub import HfApi

from src.infrastructure.streamlit import ConfigType
from src.infrastructure.datasets import load_samsum_dataset, train_datapath, eval_datapath
from src.infrastructure.transformers import load_tokenizer
from src.domain.configuration import get_tokenizer_id, limited_samples_count
from src.domain.transform import (
    concatenate_train_test_data,
    tokenize_strings,
    max_sequence_length,
    preprocess_function
)
from src.infrastructure.huggingface_hub import upload_results_file


def app_prepare_dataset(config: ConfigType, api: HfApi):
    step = 1
    config['steps'][step] = dict()
    config['steps'][step]['start_epoch'] = time()

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    if config['limit_samples']:
        dataset['train'] = dataset['train'].select(range(limited_samples_count))
        dataset['test'] = dataset['test'].select(range(limited_samples_count))

    config['steps'][step]['train_dataset_size'] = len(dataset['train'])
    config['steps'][step]['test_dataset_size'] = len(dataset['test'])

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    config['steps'][step]['max_source_length'] = max_source_length

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    config['steps'][step]['max_target_length'] = max_target_length

    # Preprocess our dataset before training and save it to disk.
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_source_length, max_target_length),
        batched=True,
        remove_columns=["dialogue", "summary", "id"]
    )
    tokenized_dataset["train"].save_to_disk(train_datapath)
    tokenized_dataset["test"].save_to_disk(eval_datapath)

    config['steps'][step]['elasped_time'] = time() - config['steps'][step]['start_epoch']
    # preserve results
    upload_results_file(config, api, step)
