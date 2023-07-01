from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel
from torch import float16


def load_tokenizer(tokenizer_id: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_id)


def load_base_model(base_model_id: str) -> PreTrainedModel:
    return AutoModelForSeq2SeqLM.from_pretrained(base_model_id, load_in_8bit=True, device_map="auto", torch_dtype=float16)
