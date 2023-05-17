from os import getenv
from typing import Dict, Any, Union, Optional

from streamlit import secrets, warning
from streamlit.runtime.state import SessionStateProxy


ConfigType = Union[SessionStateProxy, Dict[str, Any]]


def get_secret(secret_name: str) -> Optional[str]:
    try:
        return secrets[secret_name]
    except (FileNotFoundError, KeyError) as e:
        warning(e)
        return getenv(secret_name)
