import pytest

from src.infrastructure.utilities import strtobool


def test_strtobool():
    assert strtobool('True')
    assert not strtobool('FaLSe')
    with pytest.raises(ValueError):
        strtobool('apple')
