from typing import Dict, Tuple

from evaluate import EvaluationModule
from numpy import where
from transformers import PreTrainedTokenizer

from src.domain.configuration import label_pad_token_id


def compute_metrics(
        eval_pred: Tuple[str, str],
        tokenizer: PreTrainedTokenizer,
        rouge: EvaluationModule
) -> Dict[str, float]:

    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = where(labels != label_pad_token_id, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
