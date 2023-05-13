from peft import PeftConfig, PeftModel
from transformers import PreTrainedModel


def load_configuration(peft_model_id: str) -> PeftConfig:
    return PeftConfig.from_pretrained(peft_model_id)


def load_peft_model(model: PreTrainedModel, peft_model_id: str) -> PeftModel:
    return PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
