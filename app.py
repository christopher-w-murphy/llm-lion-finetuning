from streamlit import (
    title,
    form,
    selectbox,
    slider,
    checkbox,
    form_submit_button,
    header,
    session_state,
    success
)

from src.infrastructure.huggingface_hub import get_huggingface_hub_connection
from src.application.application_prepare_samsum_dataset import app_prepare_dataset
from src.application.application_fine_tune_flan_t5 import app_fine_tune
from src.application.application_evaluate_lora_flan_t5 import app_evaluate_lora_model


def main():
    """
    A simple Streamlit app to train and evaluate the models.
    """
    title("Fine-Tune an LLM with Lion")

    with form("Configuration"):
        selectbox(
            "Select Optimizer",
            ["AdamW", "Lion"],
            key="optim_name",
            help="AdamW is the default optimzer for training transformer models. "
                 "Lion is the optimizer we wish to test."
        )
        selectbox(
            "Select FLAN-T5 model size",
            ["Small", "XXL"],
            key="model_size",
            help="Fine-tune an 80M param model, or fine-tune an 11B param model."
        )
        slider(
            "Number of Training Epochs",
            min_value=1,
            max_value=10,
            value=5,
            key="n_epochs",
            help="Adjust the number of fine-tuning training epochs."
        )
        checkbox(
            "Limit Number of Samples?",
            key="limit_samples",
            help="If selected, use only the first 2 samples from the train and test datasets. "
                 "Useful for testing the pipeline."
        )
        run = form_submit_button("Submit Run")

    if run:
        session_state['steps'] = dict()
        api = get_huggingface_hub_connection()

        header("Step 1. Load and prepare the Samsum dataset")
        app_prepare_dataset(session_state, api)

        header("Step 2. Fine-Tune T5 with LoRA and bnb int-8")
        app_fine_tune(session_state, api)

        header("Step 3. Evaluate & run Inference with LoRA FLAN-T5")
        app_evaluate_lora_model(session_state, api)

        success('Training and Evaluation complete!')


if __name__ == '__main__':
    main()
