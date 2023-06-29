from peft import PeftModel, PeftModelForSeq2SeqLM
from transformers import PreTrainedModel
from pathlib import Path


def load_peft_model(base_model: PreTrainedModel, output_dir: str) -> PeftModelForSeq2SeqLM:
    """
    output_dir can either be a local filepath or a Huggingface repo ID
    """
    output_path = Path.cwd() / output_dir
    if output_path.exists():
        return PeftModel.from_pretrained(base_model, output_path)
    else:
        return PeftModel.from_pretrained(base_model, output_dir)
