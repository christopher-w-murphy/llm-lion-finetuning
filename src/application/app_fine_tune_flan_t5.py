from logging import getLogger, INFO
from os import getenv
from time import time

from datasets import Dataset
from huggingface_hub import login, logout
from streamlit.runtime.state import SessionStateProxy
from torch.cuda import memory_stats
from transformers import BatchEncoding

from src.infrastructure.datasets import load_samsum_dataset
from src.domain.configuration import limited_samples_count, get_tokenizer_id, get_base_model_id, get_output_dir
from src.infrastructure.transformers import load_tokenizer, load_base_model
from src.domain.transform import concatenate_train_test_data, tokenize_strings, max_sequence_length, preprocess_function
from src.domain.model import get_lora_model, summarize_trainable_parameters, get_data_collator, get_training_arguments, get_trainer
from src.domain.model.optimization import get_optimizers
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.model.evaluation import evaluate_peft_model
from src.infrastructure.huggingface_hub import mock_saving, get_huggingface_hub_connection, upload_log


logger = getLogger(__name__)
logger.setLevel(INFO)


def app(config: SessionStateProxy):
    # Log the configuration.
    log = dict()
    log['streamlit_config'] = {key: val for key, val in config.items()}
    for key, val in log['streamlit_config'].items():
        logger.info(f"streamlit_config - {key}: {val}")

    """
    Load and prepare the dataset
    """
    log['elt'] = dict()
    log['elt']['start_epoch'] = time()

    # Load the `samsum` dataset.
    dataset = load_samsum_dataset()

    if config['limit_samples']:
        for dataset_name in dataset:
            dataset[dataset_name] = dataset[dataset_name].select(range(limited_samples_count))

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
        logger.info(f"elt - {key}: {val}")

    """
    Fine-Tune T5 with LoRA and bnb int-8
    """
    log['train'] = dict()
    log['train']['start_epoch'] = time()

    # Load the base model.
    base_model_id = get_base_model_id(config['model_size'])
    model = load_base_model(base_model_id)

    # Prepare our model for the LoRA int-8 training using peft.
    model = get_lora_model(model)
    log['train']['trainable_parameters_summary'] = summarize_trainable_parameters(model)

    # Pad our inputs and labels.
    data_collator = get_data_collator(tokenizer, model)

    # Create Trainer instance.
    output_dir = get_output_dir(config['model_size'], config['optim_name'])
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
    log['train']['cuda_memory_stats'] = memory_stats()
    log['train']['elasped_time'] = time() - log['train']['start_epoch']
    for key, val in log['train'].items():
        logger.info(f"train - {key}: {val}")

    """
    Evaluate & run Inference with LoRA FLAN-T5
    """
    log['eval'] = dict()
    log['eval']['start_epoch'] = time()

    # Switch to inference mode.
    model.eval()

    # Run predictions.
    predictions, references = list(), list()
    for sample in tokenized_dataset['test'].with_format('torch'):
        prediction, reference = evaluate_peft_model(sample, model, tokenizer)
        predictions.append(prediction)
        references.append(reference)

    rouge = load_rouge_metric()
    log['eval']['results'] = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    log['eval']['elasped_time'] = time() - log['train']['start_epoch']
    for key, val in log['eval'].items():
        logger.info(f"eval - {key}: {val}")

    # Save our model and upload the log.
    if not mock_saving():
        token = getenv('HUGGINGFACE_TOKEN')
        login(token=token)
        trainer.model.push_to_hub(output_dir)
        api = get_huggingface_hub_connection(token=token)
        upload_log(log, api)
        logout()
