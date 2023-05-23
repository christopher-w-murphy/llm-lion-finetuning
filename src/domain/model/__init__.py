from typing import Dict, Tuple, Callable

from bitsandbytes.optim.optimizer import Optimizer8bit
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_int8_training, PeftModelForSeq2SeqLM
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    PreTrainedModel,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from src.domain.configuration import label_pad_token_id, get_output_dir


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )


def get_lora_model(base_model: PreTrainedModel) -> PeftModelForSeq2SeqLM:
    lora_config = get_lora_config()
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(base_model)
    # add LoRA adaptor
    return PeftModelForSeq2SeqLM(model, lora_config)


def get_data_collator(
        tokenizer: PreTrainedTokenizer,
        model: PeftModelForSeq2SeqLM
) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )


def get_training_arguments(model_size: str, n_epochs: int, optim_name: str) -> Seq2SeqTrainingArguments:
    """
    Define hyperparameters
    """
    output_dir = get_output_dir(model_size, optim_name)
    # logging_dir = f"{output_dir}/logs"
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # auto_find_batch_size=True,
        num_train_epochs=n_epochs,
        # logging_dir=logging_dir,
        # logging_strategy="steps",
        # logging_steps=500,
        # save_strategy="no",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        # push_to_hub=True,
        fp16=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
    )


def get_trainer(
        model: PeftModelForSeq2SeqLM,
        tokenizer: PreTrainedTokenizer,
        data_collator: DataCollatorForSeq2Seq,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_arguments: Seq2SeqTrainingArguments,
        optimizers: Tuple[Optimizer8bit, LambdaLR],
        compute_metrics_function: Callable
) -> Seq2SeqTrainer:
    return Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=optimizers,
        compute_metrics=compute_metrics_function
    )


def summarize_trainable_parameters(model: PeftModelForSeq2SeqLM) -> Dict[str, int]:
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
