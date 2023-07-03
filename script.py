from dotenv import load_dotenv
from tap import Tap

from src.application.app_fine_tune_flan_t5 import app
from src.domain.configuration import optim_names, model_sizes


class Parser(Tap):
    optim_name: str = "AdamW 32-bit"    # AdamW is the default optimzer for training transformer models. Lion is the optimizer we wish to test.
    model_size: str = "Small"           # Fine-tune an 80M param model, or fine-tune an 11B param model.
    n_epochs: int = 5                   # Adjust the number of fine-tuning training epochs.
    limit_samples: bool = False         # If selected, use only the first {limited_samples_count} samples from and test dataset. Useful for testing the pipeline.

    def process_args(self):
        if self.optim_name not in optim_names:
            raise ValueError(f"Invalid optimizer name, {self.optim_name}. The choices are: {optim_names}.")
        if self.model_size not in model_sizes:
            raise ValueError(f"Invalid model size, {self.model_size}. The choices are: {model_sizes}.")


def main():
    load_dotenv()
    args = Parser().parse_args()
    app({"optim_name": args.optim_name, "model_size": args.model_size, "n_epochs": args.n_epochs, "limit_samples": args.limit_samples})


if __name__ == '__main__':
    main()
