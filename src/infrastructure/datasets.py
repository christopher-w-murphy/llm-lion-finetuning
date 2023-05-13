from datasets import load_dataset, DatasetDict, load_from_disk, Dataset


train_datapath = "data/train"
eval_datapath = "data/eval"


def load_samsum_dataset() -> DatasetDict:
    return load_dataset("samsum")


def load_tokenized_train_dataset() -> Dataset:
    return load_from_disk(train_datapath).with_format("torch")


def load_tokenized_eval_dataset() -> Dataset:
    return load_from_disk(eval_datapath).with_format("torch")
