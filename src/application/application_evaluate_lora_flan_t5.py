from streamlit import write, progress

from src.infrastructure.datasets import load_samsum_dataset, load_tokenized_eval_dataset
from src.infrastructure.peft import load_configuration, load_peft_model
from src.infrastructure.transformers import load_model_from_config, load_tokenizer_from_config
from src.infrastructure.evaluate import load_rouge_metric
from src.domain.configuration import get_peft_model_id
from src.domain.evaluation import sample_test_dataset, evaluate_peft_model


def app_evaluate_lora_model(model_size: str):
    # Load peft config for pre-trained checkpoint.
    peft_model_id = get_peft_model_id(model_size)
    config = load_configuration(peft_model_id)

    # Load base LLM model and tokenizer.
    model = load_model_from_config(config)
    tokenizer = load_tokenizer_from_config(config)

    # Load the LoRA model.
    model = load_peft_model(model, peft_model_id)
    model.eval()
    write("Peft model loaded")

    # Reload the dataset and select a random sample from it.
    dataset = load_samsum_dataset()
    sample = sample_test_dataset(dataset["test"])

    input_ids = tokenizer(sample["dialogue"], return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=10, do_sample=True, top_p=0.9)
    write(f"input sentence: {sample['dialogue']}\n{'---'* 20}")
    write(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

    # Evaluate the tokenized test dataset using rouge_score.
    metric = load_rouge_metric()
    test_dataset = load_tokenized_eval_dataset()

    predictions, references = [], []
    eval_progress = progress(0., "Running Predictions")
    n_test_samples = len(test_dataset)
    for idx, sample in enumerate(test_dataset):
        prediction, reference = evaluate_peft_model(model, tokenizer, sample)
        predictions.append(prediction)
        references.append(reference)
        eval_progress.progress((idx + 1) / n_test_samples)

    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)

    write(f"Rogue1: {rogue['rouge1'] * 100:2f}%")
    write(f"rouge2: {rogue['rouge2'] * 100:2f}%")
    write(f"rougeL: {rogue['rougeL'] * 100:2f}%")
    write(f"rougeLsum: {rogue['rougeLsum'] * 100:2f}%")
