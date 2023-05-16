from typing import Dict

from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_int8_training, get_peft_model, PeftModel
from transformers import (
    PreTrainedModel,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from src.domain.configuration import label_pad_token_id, get_output_dir
from src.domain.optimization import get_optimizer
from src.infrastructure.streamlit import ConfigType


def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )


def get_lora_model(base_model: PreTrainedModel) -> PeftModel:
    lora_config = get_lora_config()
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(base_model)
    # add LoRA adaptor
    return get_peft_model(model, lora_config)


def get_data_collator(tokenizer: PreTrainedTokenizer, model: PeftModel) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )


def get_training_arguments(config: ConfigType) -> Seq2SeqTrainingArguments:
    """
    Define hyperparameters
    """
    output_dir = get_output_dir(config['model_size'])
    logging_dir = f"{output_dir}/logs"
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        num_train_epochs=config['n_epochs'],
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no"
    )


def get_trainer(
        model: PeftModel,
        data_collator: DataCollatorForSeq2Seq,
        train_dataset: Dataset,
        config: ConfigType
) -> Seq2SeqTrainer:
    optimizer = get_optimizer(model, config['optim_name'])
    return Seq2SeqTrainer(
        model=model,
        args=get_training_arguments(config),
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(optimizer, None)
    )


def summarize_trainable_parameters(model: PeftModel) -> Dict[str, int]:
    """
    Counts the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return {'trainable_params': trainable_params, 'all_params': all_params}
