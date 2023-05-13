from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel
from peft import PeftConfig


def load_tokenizer(tokenizer_id: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_id)


def load_base_model(base_model_id: str) -> PreTrainedModel:
    # return AutoModelForSeq2SeqLM.from_pretrained(base_model_id, load_in_8bit=True, device_map="auto")
    return AutoModelForSeq2SeqLM.from_pretrained(base_model_id, device_map="auto")


def load_model_from_config(config: PeftConfig) -> PreTrainedModel:
    return AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        # load_in_8bit=True,
        device_map={"": 0}
    )


def load_tokenizer_from_config(config: PeftConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(config.base_model_name_or_path)
