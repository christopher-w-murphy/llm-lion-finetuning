from streamlit import write, progress

from src.infrastructure.datasets import load_tokenized_eval_dataset
from src.infrastructure.peft import load_configuration, load_peft_model
from src.infrastructure.transformers import load_model_from_config, load_tokenizer_from_config
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.configuration import get_peft_model_id
from src.domain.evaluation import evaluate_peft_model


def app_evaluate_lora_model(model_size: str):
    # Load peft config for pre-trained checkpoint.
    peft_model_id = get_peft_model_id(model_size)
    config = load_configuration(peft_model_id)
    write('Configuration loaded')

    # Load base LLM model and tokenizer.
    model = load_model_from_config(config)
    write('Base model loaded')
    tokenizer = load_tokenizer_from_config(config)
    write('Tokenizer loaded')

    # Load the LoRA model.
    model = load_peft_model(model, peft_model_id)
    model.eval()
    write("Peft model loaded")

    # Evaluate the tokenized test dataset using rouge_score.
    metric = load_rouge_metric()
    write('Rouge metric loaded')
    test_dataset = load_tokenized_eval_dataset()
    write('Evaluation dataset loaded')

    predictions, references = [], []
    eval_progress = progress(0., text="Running Predictions")
    n_test_samples = len(test_dataset)
    for idx, sample in enumerate(test_dataset):
        prediction, reference = evaluate_peft_model(model, tokenizer, sample)
        predictions.append(prediction)
        references.append(reference)
        eval_progress.progress((idx + 1) / n_test_samples, text="Running Predictions")

    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    write(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
    write(f"rouge2: {rogue['rouge2'] * 100:2f}%")
    write(f"rougeL: {rogue['rougeL'] * 100:2f}%")
    write(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")
