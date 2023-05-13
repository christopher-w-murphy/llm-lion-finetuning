from time import time

from streamlit import title, selectbox, header, write, button

from src.application.application_prepare_samsum_dataset import app_prepare_dataset
from src.application.application_fine_tune_flan_t5 import app_fine_tune
from src.application.application_evaluate_lora_flan_t5 import app_evaluate_lora_model


def main():
    """
    A simple Streamlit app to train and evaluate the models.
    """
    title("Fine-Tune an LLM with Lion.")
    model_size = selectbox("Select FLAN-T5 model size.", ["Small", "XXL"], help="Fine-tune a 80M param model for 1 epoch, or fine-tune a 11B param model for 5 epochs.")

    header("Step 1. Load and prepare the Samsum dataset")
    load_and_prepare_dataset = button("Load and Prepare Dataset")
    if load_and_prepare_dataset:
        step1_start = time()
        app_prepare_dataset(model_size)
        step1_end = time()
        write(f"Step 1 took {step1_end - step1_start:.1f} seconds.")

    header("Step 2. Fine-Tune T5 with LoRA and bnb int-8")
    fine_tine_t5 = button("Fine-Tune T5")
    if fine_tine_t5:
        step2_start = time()
        app_fine_tune(model_size)
        step2_end = time()
        step2_time_diff = (step2_end - step2_start) / 3600
        write(f"Step 2 took {step2_time_diff:.2f} hours.")

    header("Step 3. Evaluate & run Inference with LoRA FLAN-T5")
    evaluate_model = button("Evaluate Model")
    if evaluate_model:
        step3_start = time()
        app_evaluate_lora_model(model_size)
        step3_end = time()
        step3_time_diff = (step3_end - step3_start) / 60
        write(f"Step 3 took {step3_time_diff:.2f} minutes.")


if __name__ == '__main__':
    main()
