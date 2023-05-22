from evaluate import load, EvaluationModule


def load_rouge_metric() -> EvaluationModule:
    return load("rouge")
