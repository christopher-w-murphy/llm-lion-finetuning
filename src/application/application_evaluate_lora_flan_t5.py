from time import time

from huggingface_hub import HfApi
from streamlit import progress

from src.infrastructure.streamlit import ConfigType
from src.infrastructure.datasets import load_tokenized_eval_dataset
from src.infrastructure.peft import load_configuration, load_peft_model
from src.infrastructure.transformers import load_model_from_config, load_tokenizer_from_config
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.configuration import get_peft_model_id
from src.domain.model.evaluation import evaluate_peft_model
from src.infrastructure.huggingface_hub import upload_results_file


def app_evaluate_lora_model(config: ConfigType, api: HfApi):
    step = 3
    config['steps'][step] = dict()
    config['steps'][step]['start_epoch'] = time()

    # Load peft config for pre-trained checkpoint.
    peft_model_id = get_peft_model_id(config['model_size'])
    model_config = load_configuration(peft_model_id)

    # Load base LLM model and tokenizer.
    model = load_model_from_config(model_config)
    tokenizer = load_tokenizer_from_config(model_config)

    # Load the LoRA model.
    model = load_peft_model(model, peft_model_id)
    model.eval()

    # Evaluate the tokenized test dataset using rouge_score.
    metric = load_rouge_metric()
    test_dataset = load_tokenized_eval_dataset()

    predictions, references = [], []
    eval_progress = progress(0., text="Running Predictions")
    n_test_samples = len(test_dataset)
    for idx, sample in enumerate(test_dataset):
        prediction, reference = evaluate_peft_model(model, tokenizer, sample)
        predictions.append(prediction)
        references.append(reference)
        eval_progress.progress((idx + 1) / n_test_samples, text="Running Predictions")

    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    config['steps'][step]['rouge1'] = rogue['rouge1']
    config['steps'][step]['rouge2'] = rogue['rouge2']
    config['steps'][step]['rougeL'] = rogue['rougeL']
    config['steps'][step]['rougeLsum'] = rogue['rougeLsum']

    config['steps'][step]['time_diff'] = time() - config['steps'][step]['start_epoch']
    # preserve results
    upload_results_file(config, api, step)
