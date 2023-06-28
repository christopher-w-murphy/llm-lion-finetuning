from argparse import ArgumentParser
from os import getenv

from dotenv import load_dotenv

from src.application.app_fine_tune_flan_t5 import app


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("optim_name", type=str, choices=["AdamW 32-bit", "AdamW 8-bit", "Lion 32-bit"], default="AdamW 32-bit")
    parser.add_argument("model_size", type=str, choices=["Small", "XXL"], default="Small")
    parser.add_argument("n_epochs", type=int, default=5)
    parser.add_argument("limit_samples", type=bool, default=False)
    return parser


def main():
    if getenv("HUGGINGFACE_TOKEN") is None or getenv("SPACE_ID") is None:
        load_dotenv()

    parser = get_parser()
    args = parser.parse_args()

    app({"optim_name": args.optim_name, "model_size": args.model_size, "n_epochs": args.n_epochs, "limit_samples": args.limit_samples})


if __name__ == '__main__':
    main()
