from datasets import Dataset, concatenate_datasets, DatasetDict
from numpy import percentile
from transformers import PreTrainedTokenizer, BatchEncoding

from src.domain.model import label_pad_token_id


def concatenate_train_test_data(dataset: DatasetDict) -> Dataset:
    return concatenate_datasets([dataset["train"], dataset["test"]])


def tokenize_strings(dataset: Dataset, tokenizer: PreTrainedTokenizer, feature: str) -> Dataset:
    features = ["dialogue", "summary"]
    assert feature in features
    return dataset.map(lambda x: tokenizer(x[feature], truncation=True), batched=True, remove_columns=features)


def max_sequence_length(tokenized_strings, ntile: int = 85) -> int:
    feature_lengths = [len(input_id) for input_id in tokenized_strings["input_ids"]]
    return int(percentile(feature_lengths, ntile))


def replace_token_ids(label, pad_token_id):
    """
    Replace all tokenizer.pad_token_id in the labels by label_pad_token_id when we want to ignore padding in the loss.
    """
    return [(token_id if token_id != pad_token_id else label_pad_token_id) for token_id in label]


def preprocess_function(
        sample_data: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int,
        max_target_length: int,
        padding: str = "max_length"
) -> BatchEncoding:
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample_data["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample_data["summary"],
        max_length=max_target_length,
        padding=padding, truncation=True
    )

    if padding == "max_length":
        labels["input_ids"] = [replace_token_ids(label, tokenizer.pad_token_id) for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
