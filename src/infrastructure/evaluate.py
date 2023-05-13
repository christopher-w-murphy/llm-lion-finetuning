from evaluate import load


def load_rouge_metric():
    return load("rouge")
