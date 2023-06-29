from os import environ
from pytest import raises

from src.infrastructure.utilities import strtobool, mock_saving


def test_strtobool():
    assert strtobool('True')
    assert not strtobool('FaLSe')
    with raises(ValueError):
        strtobool('apple')


def test_mock_saving():
    environ['MOCK_SAVING'] = 'true'
    assert mock_saving()
    environ['MOCK_SAVING'] = 'false'
    assert not mock_saving()
