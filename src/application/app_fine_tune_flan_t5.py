from os import getenv
from time import time
from typing import Dict, Any
from warnings import warn

from datasets import Dataset
from huggingface_hub import login, logout
from tqdm import tqdm
from transformers import BatchEncoding

from src.infrastructure.datasets import load_samsum_dataset
from src.domain.configuration import limited_samples_count, get_tokenizer_id, get_base_model_id, get_output_dir
from src.infrastructure.transformers import load_tokenizer, load_base_model
from src.domain.transform import concatenate_train_test_data, tokenize_strings, max_sequence_length, preprocess_function
from src.infrastructure.torch import verify_torch_installation, get_memory_stats
from src.domain.model import get_lora_model, summarize_trainable_parameters, get_data_collator, get_training_arguments, get_trainer
from src.domain.model.optimization import get_optimizers
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.model.evaluation import evaluate_peft_model
from src.infrastructure.huggingface_hub import get_huggingface_hub_connection, upload_log, save_log
from src.infrastructure.peft import load_peft_model
from src.infrastructure.utilities import mock_saving


def app(config: Dict[str, Any]):
    # Log the configuration.
    log = dict()
    log['config'] = {key: val for key, val in config.items()}
    for key, val in log['config'].items():
        print(f"config - {key}: {val}")

    """
    Load and prepare the dataset
    """
    log['elt'] = dict()
    log['elt']['start_epoch'] = time()

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()
    log['elt']['train_dataset_size'] = len(dataset['train'])
    log['elt']['test_dataset_size'] = len(dataset['test'])

    # We need to convert our inputs (text) to token IDs.
    tokenizer_id = get_tokenizer_id(config['model_size'])
    tokenizer = load_tokenizer(tokenizer_id)

    # Batch our data efficiently.
    concatenated_dataset = concatenate_train_test_data(dataset)

    tokenized_inputs = tokenize_strings(concatenated_dataset, tokenizer, "dialogue")
    max_source_length = max_sequence_length(tokenized_inputs)
    log['elt']['max_source_length'] = max_source_length

    tokenized_targets = tokenize_strings(concatenated_dataset, tokenizer, "summary")
    max_target_length = max_sequence_length(tokenized_targets, 90)
    log['elt']['max_target_length'] = max_target_length

    # Preprocess our dataset before training and save it to disk.
    def preprocess_func(target_dataset: Dataset) -> BatchEncoding:
        return preprocess_function(target_dataset, tokenizer, max_source_length, max_target_length)

    tokenized_dataset = dataset.map(preprocess_func, batched=True, remove_columns=["dialogue", "summary", "id"])

    log['elt']['elasped_time'] = time() - log['elt']['start_epoch']
    for key, val in log['elt'].items():
        print(f"elt - {key}: {val}")

    """
    Fine-Tune T5 with LoRA and bnb int-8
    """
    log['train'] = dict()
    log['train']['start_epoch'] = time()
    log['train']['verify_torch_installation'] = verify_torch_installation()

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare for preserving results.
    output_dir = get_output_dir(config['model_size'], config['optim_name'])
    token = getenv('HUGGINGFACE_TOKEN')

    # Either load a pre-finetuned model or finetune a model.
    if config['eval_only'] is not None and config['eval_only']:
        model = load_peft_model(model, output_dir)
    else:
        # Prepare our model for the LoRA int-8 training using peft.
        model = get_lora_model(model)
        log['train']['trainable_parameters_summary'] = summarize_trainable_parameters(model)

        # Pad our inputs and labels.
        data_collator = get_data_collator(tokenizer, model)

        # Create Trainer instance.
        training_arguments = get_training_arguments(output_dir, config['n_epochs'])
        optimizers = get_optimizers(model, config['optim_name'])

        trainer = get_trainer(
            model=model,
            data_collator=data_collator,
            train_dataset=tokenized_dataset['train'],
            training_arguments=training_arguments,
            optimizers=optimizers
        )
        model.config.use_cache = False  # silence the warnings.

        # Train model.
        trainer.train()

        log['train']['trainer_log'] = trainer.state.log_history
        log['train']['model_memory_footprint'] = trainer.model.get_memory_footprint(return_buffers=False)
        log['train']['cuda_memory_stats'] = get_memory_stats()
        log['train']['train_batch_size'] = trainer.args.train_batch_size

        # Save our model to the hub
        if not mock_saving():
            try:
                login(token=token)
                trainer.model.push_to_hub(output_dir)
            except ValueError as e:
                warn(f"Unable to upload model due to, {e}. Trying to write model to disk instead.", UserWarning)
                trainer.model.save_pretrained(output_dir)

    log['train']['elasped_time'] = time() - log['train']['start_epoch']
    for key, val in log['train'].items():
        print(f"train - {key}: {val}")

    """
    Evaluate & run Inference with LoRA FLAN-T5
    """
    log['eval'] = dict()
    log['eval']['start_epoch'] = time()

    # Switch to inference mode.
    model.eval()
    model.config.use_cache = True  # reenable for inference

    # Run predictions.
    predictions, references = list(), list()
    test_dataset = tokenized_dataset['test'].with_format('torch')
    for idx, sample in enumerate(tqdm(test_dataset), start=1):
        prediction, reference = evaluate_peft_model(sample, model, tokenizer)
        predictions.append(prediction)
        references.append(reference)
        if 'limit_samples' in config and config['limit_samples'] and idx == limited_samples_count:
            break

    rouge = load_rouge_metric()
    log['eval']['results'] = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    log['eval']['elasped_time'] = time() - log['eval']['start_epoch']
    for key, val in log['eval'].items():
        print(f"eval - {key}: {val}")

    # Upload the log to the hub.
    if not mock_saving():
        try:
            api = get_huggingface_hub_connection(token=token)
            upload_log(log, api)
        except ValueError as e:
            warn(f"Unable to upload log due to, {e}. Trying to write log to disk instead.", UserWarning)
            save_log(log, output_dir)
        finally:
            logout()
