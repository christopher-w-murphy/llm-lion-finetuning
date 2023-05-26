from datetime import datetime
from io import BytesIO, StringIO
from json import dump
from os import getenv
from typing import Optional, Dict, Any

from huggingface_hub import HfApi

from src.infrastructure.utilities import strtobool


def get_huggingface_hub_connection(token: Optional[str] = None) -> HfApi:
    if token is None:
        token = getenv('HUGGINGFACE_TOKEN')
    return HfApi(token=token)


def mock_saving() -> bool:
    return getenv('MOCK_SAVING') is not None and strtobool(getenv('MOCK_SAVING'))


def format_log_filename(upload_time: Optional[datetime] = None) -> str:
    if upload_time is None:
        upload_time = datetime.now()
    return f"log_{upload_time.strftime('%Y-%m-%d_%H:%M:%S')}.json"


def dicttobytes(dict_obj: Dict[str, Any]) -> BytesIO:
    # convert a Python dictionary to a JSON string
    sio = StringIO()
    dump(dict_obj, sio)
    # convert the string to bytes
    return BytesIO(sio.getvalue().encode('utf8'))


def upload_log(log_dict: Dict[str, Any], api: Optional[HfApi] = None):
    log_bio = dicttobytes(log_dict)
    if api is None:
        api = get_huggingface_hub_connection()
    api.upload_file(
        path_or_fileobj=log_bio,
        path_in_repo=format_log_filename(),
        repo_id=getenv("SPACE_ID"),  # make the dataset ID the same as the space ID
        repo_type="dataset"
    )
