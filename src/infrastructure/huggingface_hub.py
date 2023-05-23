from datetime import datetime
from io import BytesIO
from os import getenv

from huggingface_hub import HfApi

from src.infrastructure.utilities import convert_session_state_to_bytes, ConfigType, strtobool
from src.infrastructure.streamlit import get_secret


def get_huggingface_hub_connection() -> HfApi:
    token = get_secret('HUGGINGFACE_TOKEN')
    return HfApi(token=token)


def format_filename(config: ConfigType, step: int) -> str:
    start_epoch = config['steps'][step]['start_epoch']
    start_time = datetime.fromtimestamp(start_epoch).strftime('%Y-%m-%d_%H:%M:%S')
    return f"results_{start_time}_step{step}.json"


def mock_saving() -> bool:
    return getenv('MOCK_SAVING') is not None and strtobool(getenv('MOCK_SAVING'))


def upload_results_file(config: ConfigType, api: HfApi, step: int):
    if not mock_saving():
        config_bytes = convert_session_state_to_bytes(config)
        filename = format_filename(config, step)
        api.upload_file(
            path_or_fileobj=config_bytes,
            path_in_repo=filename,
            repo_id="chriswmurphy/llm-lion-finetuning",
            repo_type="dataset"
        )


def format_log_filename() -> str:
    log_upload_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return f"log_{log_upload_time}.txt"


def upload_log(log_io: BytesIO, api: HfApi):
    if not mock_saving():
        api.upload_file(
            path_or_fileobj=log_io,
            path_in_repo=format_log_filename(),
            repo_id="chriswmurphy/llm-lion-finetuning",
            repo_type="dataset"
        )
