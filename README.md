---
title: Llm Lion Finetuning
emoji: 🐨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.19.0
app_file: app.py
pinned: false
license: gpl-3.0
python_version: 3.9
---

# LLM Lion Fine-Tuning

This is the code associated with my blog post investigating the impact of the choice of optimizer when fine-tuning a Large Language Model.
See the post for details about the nature of the experiment and the results I found.

My [results](https://huggingface.co/datasets/chriswmurphy/llm-lion-finetuning) are available as a dataset on Hugging Face. 
The LoRA adapter models that resulted from fine-tuning are also available on Hugging Face as models: 
- [AdamW 32-bit](https://huggingface.co/chriswmurphy/lora-flan-t5-xxl-adamw32bit)
- [Lion 32-bit](https://huggingface.co/chriswmurphy/lora-flan-t5-xxl-lion32bit)
- [AdamW 8-bit](https://huggingface.co/chriswmurphy/lora-flan-t5-xxl-adamw8bit)
- [Lion 8-bit](https://huggingface.co/chriswmurphy/lora-flan-t5-xxl-lion8bit)

# Running
I tested this with Python 3.8, 3.9, and 3.10, and CUDA 11.3, and 11.8.
I used a single Nvidia A10G GPU.
If you want to save your results, you'll need a free Hugging Face account.
(If not, set the env var MOCK_SAVING=true)

## Hugging Face Spaces
This repo has a CI/CD pipeline setup to deploy to the Hugging Face Space: SPACE_ID = SPACE_AUTHOR_NAME/llm-lion-finetuning.
(The preamble of this README contains settings for setting up the environment on Spaces.)
My Space hardware was an _Nvidia A10G small_.
This provides you with an interactive environment for fine-tuning.

WARNING: Spaces will time out after at most 3 hours regardless of the tricks you use to maintain an active session.
As such, you'll only be able to fine-tune Flan-T5-XXL on the SAMSum dataset for ~1.5 epochs on this platform.

## AWS EC2
My EC2 instance type was a _g5.2xlarge_.
I needed 48 GB of disk space, and went with 100 GB to be on the safe side.

### Example AWS Usage
To fine-tune and evaluate the size XXL model using the 8-bit Lion optimizer
```
git clone https://github.com/christopher-w-murphy/llm-lion-finetuning.git
cd llm-lion-funetuning
source activate pytorch
pip install -r requirements.txt
python script.py --optim_name "Lion 8-bit" --model_size XXL
```

## Environment Variables

| Variable               | Type           | Secret | Description                                                                                                         |
|------------------------|----------------|--------|---------------------------------------------------------------------------------------------------------------------|
| HUGGINGFACE_TOKEN      | str            | yes    | Token for programmatically uploading results logs and model artifacts to Hugging Face hub.                          |
| SPACE_ID               | str            | no     | Dataset ID for where the results logs will be saved. Same value as Space ID. Set by default on Hugging Face Spaces. |
| TOKENIZERS_PARALLELISM | bool, optional | no     | If set to true, will suppress a warning messages that is frequently repeated.                                       |
| MOCK_SAVING            | bool, optional | no     | Whether to save results. Useful for pipeline testing purposes.                                                      |

To run the CI/CD pipeline to deploy the code to Hugging Face Spaces you'll need to have your HUGGINGFACE_TOKEN also stored as a secret on GitHub.
You'll also need your SPACE_AUTHOR_NAME set as a variable on GitHub.