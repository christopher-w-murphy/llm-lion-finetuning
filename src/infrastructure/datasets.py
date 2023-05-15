from pathlib import Path

from datasets import load_dataset, DatasetDict, load_from_disk, Dataset, DownloadMode


base_datapath = Path.cwd() / 'data'
train_datapath = base_datapath / 'train'
eval_datapath = base_datapath / 'eval'


def load_samsum_dataset() -> DatasetDict:
    return load_dataset("samsum", download_mode=DownloadMode.FORCE_REDOWNLOAD)


def load_tokenized_train_dataset() -> Dataset:
    return load_from_disk(str(train_datapath)).with_format("torch")


def load_tokenized_eval_dataset() -> Dataset:
    return load_from_disk(str(eval_datapath)).with_format("torch")
