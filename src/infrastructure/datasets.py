from datasets import load_dataset, DatasetDict, DownloadMode


def load_samsum_dataset() -> DatasetDict:
    return load_dataset("samsum", download_mode=DownloadMode.FORCE_REDOWNLOAD)
