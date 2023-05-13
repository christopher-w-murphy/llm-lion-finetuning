from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_int8_training, get_peft_model, PeftModel
from transformers import (
    PreTrainedModel,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from src.domain.configuration import label_pad_token_id, get_output_dir, get_n_epochs


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


def get_training_arguments(model_size: str) -> Seq2SeqTrainingArguments:
    """
    Define hyperparameters
    """
    output_dir = get_output_dir(model_size)
    logging_dir = f"{output_dir}/logs"
    n_epochs = get_n_epochs(model_size)
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,  # higher learning rate
        num_train_epochs=n_epochs,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
        report_to="tensorboard",
    )


def get_trainer(
        model: PeftModel,
        data_collator: DataCollatorForSeq2Seq,
        train_dataset: Dataset,
        model_size: str
) -> Seq2SeqTrainer:
    training_args = get_training_arguments(model_size)
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )


def summarize_trainable_parameters(model: PeftModel) -> str:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
