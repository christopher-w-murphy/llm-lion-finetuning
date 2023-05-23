from datetime import datetime
from io import BytesIO, StringIO
from os import getenv
from typing import Optional

from huggingface_hub import HfApi

from src.infrastructure.streamlit import get_secret
from src.infrastructure.utilities import strtobool


def get_huggingface_hub_connection(token: Optional[str] = None) -> HfApi:
    if token is None:
        token = getenv('HUGGINGFACE_TOKEN')
    return HfApi(token=token)


def mock_saving() -> bool:
    return getenv('MOCK_SAVING') is not None and strtobool(getenv('MOCK_SAVING'))


def format_log_filename() -> str:
    log_upload_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    return f"log_{log_upload_time}.txt"


def upload_log(log_sio: StringIO, api: Optional[HfApi] = None):
    if api is None:
        api = get_huggingface_hub_connection()

    log_bio = BytesIO(log_sio.getvalue().encode('utf8'))

    api.upload_file(
        path_or_fileobj=log_bio,
        path_in_repo=format_log_filename(),
        repo_id="chriswmurphy/llm-lion-finetuning",
        repo_type="dataset"
    )
