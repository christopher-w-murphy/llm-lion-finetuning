from io import BytesIO, StringIO
from json import dump

from src.infrastructure.streamlit import ConfigType


def convert_session_state_to_bytes(config: ConfigType) -> BytesIO:
    """
    There's probably a cleaner way to do this.
    """
    # convert Streamlit session_state to Python dictionary
    results = {key: val for key, val in config.items()}
    # dump dictionary into string buffer in JSON format
    sio = StringIO()
    dump(results, sio)
    # convert string buffer to bytes buffer
    return BytesIO(sio.getvalue().encode('utf8'))


def strtobool(val: str) -> bool:
    """
    Convert a string representation of truth to true or false.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}.")
