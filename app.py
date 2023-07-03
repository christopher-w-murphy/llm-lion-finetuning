from streamlit import (
    title,
    form,
    selectbox,
    slider,
    checkbox,
    form_submit_button,
    header,
    session_state,
    spinner,
    success
)

from src.application.app_fine_tune_flan_t5 import app
from src.domain.configuration import limited_samples_count, model_sizes, optim_names


def main():
    """
    A simple Streamlit app to train and evaluate the models.
    """
    title("Fine-Tune an LLM with Lion")

    with form("Configuration"):
        selectbox(
            "Select Optimizer",
            optim_names,
            key="optim_name",
            help="AdamW is the default optimzer for training transformer models. "
                 "Lion is the optimizer we wish to test."
        )
        selectbox(
            "Select FLAN-T5 model size",
            model_sizes,
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
            "Limit Number of Eval Samples?",
            key="limit_samples",
            help=f"If selected, use only the first {limited_samples_count} samples from and test dataset. "
                 "Useful for testing the pipeline."
        )
        run = form_submit_button("Submit Run")

    if run:
        header("Fine-Tune T5 with LoRA and bnb int-8 on Samsum dataset.")
        with spinner("This may take a while..."):
            app(session_state)
        success('Training and Evaluation complete!')


if __name__ == '__main__':
    main()
