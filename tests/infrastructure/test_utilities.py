from pytest import raises

from src.infrastructure.utilities import strtobool


def test_strtobool():
    assert strtobool('True')
    assert not strtobool('FaLSe')
    with raises(ValueError):
        strtobool('apple')
