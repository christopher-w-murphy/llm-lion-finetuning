from streamlit import title, selectbox, header, form, slider, form_submit_button, session_state, checkbox

from src.application.application_prepare_samsum_dataset import app_prepare_dataset
from src.application.application_fine_tune_flan_t5 import app_fine_tune
from src.application.application_evaluate_lora_flan_t5 import app_evaluate_lora_model


def main():
    """
    A simple Streamlit app to train and evaluate the models.
    """
    title("Fine-Tune an LLM with Lion.")

    with form("Configuration"):
        selectbox("Select FLAN-T5 model size.", ["Small", "XXL"], key="model_size", help="Fine-tune a 80M param model, or fine-tune a 11B param model.")
        slider("Number of Training Epochs", min_value=1, max_value=10, value=5, key="n_epochs", help="Adjust the number of fine-tuning training epochs.")
        checkbox("Limit Number of Samples?", key="limit_samples", help="If selected, use only the first 10 samples from the train and test datasets. Useful for testing the pipeline.")
        run = form_submit_button("Submit Run")

    if run:
        header("Step 1. Load and prepare the Samsum dataset")
        app_prepare_dataset(session_state)

        header("Step 2. Fine-Tune T5 with LoRA and bnb int-8")
        app_fine_tune(session_state)

        header("Step 3. Evaluate & run Inference with LoRA FLAN-T5")
        app_evaluate_lora_model(session_state)


if __name__ == '__main__':
    main()
