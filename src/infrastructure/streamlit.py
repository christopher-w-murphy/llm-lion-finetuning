from os import getenv
from typing import Dict, Any, Union, Optional

from streamlit import secrets
from streamlit.runtime.state import SessionStateProxy


ConfigType = Union[SessionStateProxy, Dict[str, Any]]


def get_secret(secret_name: str) -> Optional[str]:
    if secret_name in secrets:
        return secrets[secret_name]
    return getenv(secret_name)
