from streamlit import write

from src.infrastructure.datasets import load_samsum_dataset, train_datapath, eval_datapath
from src.infrastructure.transformers import load_tokenizer
from src.domain.configuration import get_tokenizer_id
from src.domain.transform import (
    concatenate_train_test_data,
    tokenize_strings,
    max_sequence_length,
    preprocess_function
)


def app_prepare_dataset(model_size: str):
    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    dataset['train'] = dataset['train'].select(range(10))
    dataset['test'] = dataset['test'].select(range(10))

    write(f"Train dataset size: {len(dataset['train'])}")
    write(f"Test dataset size: {len(dataset['test'])}")

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(model_size)
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    write(f"Max source length: {max_source_length}")

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    write(f"Max target length: {max_target_length}")

    # Preprocess our dataset before training and save it to disk.
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_source_length, max_target_length),
        batched=True,
        remove_columns=["dialogue", "summary", "id"]
    )
    tokenized_dataset["train"].save_to_disk(train_datapath)
    tokenized_dataset["test"].save_to_disk(eval_datapath)
